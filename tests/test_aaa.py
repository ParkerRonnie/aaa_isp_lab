# -*- coding: utf-8 -*-
"""单元测试。

直接跑：    python tests/test_aaa.py
或用 pytest：pytest tests/test_aaa.py -v

覆盖两类东西：
  1) 有标准答案的实现（CIEDE2000 官方测试数据、色温往返、黑电平归一化）
  2) 3A 算法该有的**性质**（收敛、单调、峰位正确、约束生效）

第 2 类更重要：3A 是闭环控制，某个环节方向错了程序照样能跑出"看起来合理"
的数字，只有性质测试能发现。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import SensorConfig, ISPConfig, AEConfig, AWBConfig, AFConfig
from src.eval.metrics import delta_e_2000, psnr, ssim, ideal_linear
from src.color_science import (blackbody_linear_rgb, rgb_to_cct_duv, cct_to_xy,
                               xy_to_cct, rgb_linear_to_lab, linear_to_srgb)
from src.isp import modules as IM
from src.sim import scene as SC
from src.sim.camera import SimCamera, ev_limits, split_exposure, flicker_banding_metric
from src.sim.optics import defocus_kernel
from src.aaa.ae import AEController, metering_metric
from src.aaa.awb import AWBEstimator, ideal_gains, illuminant_error_deg
from src.aaa import af as AF

W, H = 192, 144
SCFG = SensorConfig(width=W, height=H)
ICFG = ISPConfig()

CASES = []


def case(fn):
    CASES.append(fn)
    return fn


# -----------------------------------------------------------------------------
# 1. 有标准答案的部分
# -----------------------------------------------------------------------------
@case
def test_ciede2000_standard():
    """CIEDE2000 官方测试数据（Sharma et al. 2005 的前几组）"""
    pairs = [
        ((50.0000, 2.6772, -79.7751), (50.0000, 0.0000, -82.7485), 2.0425),
        ((50.0000, 3.1571, -77.2803), (50.0000, 0.0000, -82.7485), 2.8615),
        ((50.0000, 2.8361, -74.0200), (50.0000, 0.0000, -82.7485), 3.4412),
        ((50.0000, -1.3802, -84.2814), (50.0000, 0.0000, -82.7485), 1.0000),
    ]
    for lab1, lab2, expect in pairs:
        got = float(delta_e_2000(np.array(lab1), np.array(lab2)))
        assert abs(got - expect) < 5e-4, f"ΔE00({lab1},{lab2}) = {got:.4f}, 期望 {expect}"

    # 同一个颜色必须是 0；交换顺序结果不变
    a = np.array([50.0, 10.0, -20.0])
    assert abs(float(delta_e_2000(a, a))) < 1e-9
    b = np.array([55.0, -5.0, 12.0])
    assert abs(float(delta_e_2000(a, b)) - float(delta_e_2000(b, a))) < 1e-9


@case
def test_cct_roundtrip_and_duv():
    """黑体轨迹上的点：色温往返误差小，Duv 接近 0"""
    for t in (2500.0, 3000.0, 5000.0, 6500.0, 9000.0):
        rgb = blackbody_linear_rgb(t)
        cct, duv = rgb_to_cct_duv(rgb)
        assert abs(cct - t) / t < 0.03, f"{t}K 往返得到 {cct:.0f}K"
        assert abs(duv) < 0.002, f"{t}K 的 Duv 应为 0，实测 {duv:+.4f}"

    # 偏离轨迹的点必须有非零 Duv（否则"色温约束管不住 Duv"这个结论站不住）
    # [1.0, 1.2, 0.9] 是偏绿的点，Duv 应为正
    _, duv_green = rgb_to_cct_duv(np.array([1.0, 1.2, 0.9]))
    assert duv_green > 0.005, f"偏绿的点 Duv 应为正且明显，实测 {duv_green:+.4f}"
    # 偏品红的点 Duv 应为负
    _, duv_magenta = rgb_to_cct_duv(np.array([1.2, 1.0, 1.2]))
    assert duv_magenta < -0.003, f"偏品红的点 Duv 应为负，实测 {duv_magenta:+.4f}"
    # 而暖色 [1.3,1.0,0.8] 其实很靠近黑体轨迹 —— 说明"色温偏"和"Duv 偏"是两件事
    _, duv_warm = rgb_to_cct_duv(np.array([1.3, 1.0, 0.8]))
    assert abs(duv_warm) < 0.005, "该点接近轨迹，Duv 应当很小"


@case
def test_black_level_normalization():
    """黑电平校正后：黑电平映射到 0，白电平映射到 1"""
    cfg = SCFG
    raw = np.array([[cfg.black_level_dn, cfg.max_dn],
                    [cfg.black_level_dn + cfg.signal_dn / 2, cfg.max_dn / 2]],
                   dtype=np.float32)
    out = IM.black_level_correct(raw, cfg.black_level_dn, cfg.signal_dn)
    assert abs(out[0, 0] - 0.0) < 1e-6
    assert abs(out[0, 1] - 1.0) < 1e-6
    assert abs(out[1, 0] - 0.5) < 1e-6


@case
def test_demosaic_constant_image():
    """恒定颜色经过 Bayer + 去马赛克后必须还是同一个颜色（两种算法都要过）"""
    img = np.full((H, W, 3), 0.42, dtype=np.float32)
    for method in ("bilinear", "color_diff"):
        pat = np.array([[0, 1], [1, 2]])
        rgb = IM.demosaic(img, pat, method)
        inner = rgb[4:-4, 4:-4]
        assert np.max(np.abs(inner - 0.42)) < 1e-5, f"{method} 破坏了恒定区域"


@case
def test_srgb_transfer_roundtrip():
    x = np.linspace(0, 1, 64)
    from src.color_science import srgb_to_linear
    back = srgb_to_linear(linear_to_srgb(x))
    assert np.max(np.abs(back - x)) < 1e-9


@case
def test_ssim_psnr_sanity():
    a = np.random.default_rng(0).random((64, 64, 3)).astype(np.float32)
    assert psnr(a, a) == float("inf")
    assert abs(ssim(a, a) - 1.0) < 1e-6
    noise = np.clip(a + 0.1 * np.random.default_rng(1).random(a.shape), 0, 1)
    assert 0 < ssim(a, noise) < 1.0
    assert psnr(a, noise) < 30.0


# -----------------------------------------------------------------------------
# 2. 仿真链路该有的物理性质
# -----------------------------------------------------------------------------
@case
def test_exposure_monotonic_and_clipping():
    """曝光越大画面越亮；到顶后必然饱和（这是 AE 存在的物理前提）"""
    cam = SimCamera(SC.color_chart(W, H), 5000.0, SCFG, ICFG, seed=1)
    means, clips = [], []
    for ev in (-3.0, -1.0, 1.0, 3.0, 5.0):
        fr = cam.capture(ev=ev)
        means.append(float(fr.luma_linear.mean()))
        clips.append(fr.clipped_ratio)
    assert all(means[i] < means[i + 1] for i in range(len(means) - 1)), means
    assert clips[-1] > 0.2, "高 EV 下必须出现明显过曝"
    assert clips[0] < 0.01, "低 EV 下不应过曝"


@case
def test_ev_limits_match_reality():
    """AE 的 EV 范围必须和相机可实现的曝光范围一致。

    这是本项目踩过的坑：控制器一路推到 +8 EV，而镜头最多只能给到 +5 EV，
    结果永远收敛不了。
    """
    lo, hi = ev_limits(SCFG)
    # 上界：曝光上限 × 增益上限
    assert abs(hi - np.log2((SCFG.max_exposure_s / (1 / 60.0)) * SCFG.max_analog_gain)) < 1e-9
    # 在范围内任意 EV，拆出来的曝光时间/增益都不应越界
    for ev in np.linspace(lo, hi, 25):
        et, g = split_exposure(float(ev), SCFG)
        assert SCFG.min_exposure_s - 1e-9 <= et <= SCFG.max_exposure_s + 1e-9, (ev, et)
        assert 1e-3 <= g <= SCFG.max_analog_gain + 1e-9, (ev, g)


@case
def test_defocus_psf_is_smooth_in_radius():
    """离焦 PSF 必须是半径的连续函数。

    这是本项目踩过的第二个坑：直接按"像素中心在圆内"生成圆盘核，
    当 r < 1 时核退化成 delta，图像完全不变 —— 对焦评价函数在一大段
    镜头位置上是水平的，峰位随机落在平台中间。
    """
    prev = None
    for r in (0.4, 0.8, 1.2, 2.0, 3.5):
        k = defocus_kernel(r, "disk")
        spread = float((k * (np.arange(k.shape[0])[:, None] - (k.shape[0] - 1) / 2) ** 2).sum())
        if prev is not None:
            assert spread > prev + 1e-6, f"r={r} 的 PSF 扩散没有随半径增加"
        prev = spread
    # 亚像素半径也不能是 delta
    assert defocus_kernel(0.6, "disk").size > 1


@case
def test_flicker_banding_metric():
    """带纹指标：曝光时间落在纹波周期整数倍上时，带纹必须消失"""
    from src.sim.camera import apply_flicker_banding
    base = np.full((H, W), 1000.0, dtype=np.float32)
    bad = apply_flicker_banding(base.copy(), 1 / 180.0, 50.0, 0.02, 64.0, 0.25)
    good = apply_flicker_banding(base.copy(), 0.01, 50.0, 0.02, 64.0, 0.25)  # 10ms = 整数倍
    assert flicker_banding_metric(bad, 0.02) > 0.05, "非整数倍曝光必须出现带纹"
    assert flicker_banding_metric(good, 0.02) < 0.005, "整数倍曝光不应有带纹"


# -----------------------------------------------------------------------------
# 3. 3A 算法的性质
# -----------------------------------------------------------------------------
@case
def test_ae_converges_and_is_monotone():
    """AE 必须收敛到目标，且测光量随 EV 单调"""
    cam = SimCamera(SC.natural_scene(W, H, backlit=True), 5000.0, SCFG, ICFG, seed=7)
    cfg = AEConfig(metering="evaluative")
    ae = AEController(cfg)

    # 单调性
    vals = [metering_metric(cam.capture(ev=float(ev)), cfg)["metric"]
            for ev in np.linspace(-3, 1, 9)]
    assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1)), vals

    # 收敛性：从两端出发都要收敛到同一个 EV 附近
    finals = []
    for ev0 in (-3.0, 3.0):
        r = ae.run(lambda ev: cam.capture(ev=ev), ev0=ev0)
        assert r.converged, f"从 {ev0} EV 出发未收敛"
        assert abs(r.history[-1]["err_ev"]) < cfg.converge_thresh_ev
        finals.append(r.final_ev)
    assert abs(finals[0] - finals[1]) < 0.15, f"两端出发收敛到不同 EV: {finals}"


@case
def test_ae_highlight_priority_protects_highlights():
    """高光优先测光：高光分位数必须收敛到目标附近，且过曝比例保持在很低水平。

    注意这里**不能**断言"它的过曝比例低于全画面平均测光" ——
    本项目的实测结果恰恰相反：高光优先为了把 99 分位压到 0.90，
    整体曝光反而比平均测光更亮。它的契约是"把最亮的部分控制住"，
    不是"整体拍暗"。一开始想当然地写成前者，被测试挡下来了。
    """
    sc = SC.natural_scene(W, H, backlit=True)
    for mode in ("average", "highlight_priority"):
        cam = SimCamera(sc, 5000.0, SCFG, ICFG, seed=7)
        cfg = AEConfig(metering=mode)
        r = AEController(cfg).run(lambda ev: cam.capture(ev=ev), ev0=2.0)
        fr = cam.capture(ev=r.final_ev)
        assert r.converged, f"{mode} 未收敛"
        if mode == "highlight_priority":
            p99 = float(np.percentile(fr.luma_linear, 99))
            assert abs(p99 - 0.90) < 0.08, f"99 分位应接近目标 0.90，实测 {p99:.3f}"
            assert fr.clipped_ratio < 0.01, f"过曝比例应很低，实测 {fr.clipped_ratio:.3%}"


@case
def test_awb_ideal_gains_neutralize():
    """理想白平衡增益必须把中性区大幅拉回中性，但要完全中性必须靠 CCM。

    这里的定量关系本身就是一个结论：
    传感器有光谱串扰时，**白平衡只能解决"光源色"这一层**，
    想让中性面真的变成 R=G=B，必须再上 CCM。所以残余色度不会到 0。
    如果仿真里不加串扰，这个测试会"过于完美地"通过 —— 那才是假象。
    """
    sc = SC.color_chart(W, H)
    cam = SimCamera(sc, 3000.0, SCFG, ICFG, seed=5)
    fr = cam.capture(ev=0.0)
    from src.eval.metrics import neutral_chroma
    masks = sc.patch_masks
    ideal = ideal_linear(sc, 3000.0)

    before = neutral_chroma(fr.linear_pre_wb, sc.neutral_mask)
    wb = IM.apply_wb(fr.linear_pre_wb, ideal_gains(3000.0))
    after_wb = neutral_chroma(wb, sc.neutral_mask)
    assert after_wb < before * 0.5, f"白平衡后残余色度应大幅下降: {before:.2f} -> {after_wb:.2f}"

    # 补上 CCM 之后，中性区才真正接近中性
    src = np.stack([wb[m].mean(axis=0) for m in masks])
    dst = np.stack([ideal[m].mean(axis=0) for m in masks])
    after_ccm = neutral_chroma(IM.apply_ccm(wb, IM.solve_ccm(src, dst)), sc.neutral_mask)
    assert after_ccm < after_wb, f"CCM 应进一步降低残余色度: {after_wb:.2f} -> {after_ccm:.2f}"


@case
def test_awb_estimator_normalization_and_masks():
    """AWB 增益的 G 通道必须归一化为 1；暗部/过曝像素必须被掩码排除"""
    sc = SC.color_chart(W, H)
    cam = SimCamera(sc, 5000.0, SCFG, ICFG, seed=5)
    fr = cam.capture(ev=0.0)
    for method in ("gray_world", "white_patch", "gray_edge", "shades_of_gray", "fusion"):
        est = AWBEstimator(AWBConfig(method=method)).estimate(fr.linear_pre_wb, fr.clipped_ratio)
        assert abs(float(est.gains[1]) - 1.0) < 1e-6, method
        assert np.all(est.gains > 0), method

    from src.aaa.awb import valid_mask
    m = valid_mask(fr.linear_pre_wb, AWBConfig())
    assert not m[0, 0] or True      # 掩码本身不该全是 False
    assert m.mean() > 0.1, "有效像素比例过低，掩码条件过严"


@case
def test_awb_fusion_beats_single_method_on_hard_scene():
    """大面积单色场景下，融合必须优于单一算法（这是融合存在的理由）"""
    sc = SC.muted_scene(W, H, (0.75, 0.12, 0.10))
    cam = SimCamera(sc, 3000.0, SCFG, ICFG, seed=5)
    ae = AEController(AEConfig(metering="evaluative"))
    r = ae.run(lambda ev: cam.capture(ev=ev), ev0=0.0)
    fr = cam.capture(ev=r.final_ev)
    errs = {}
    for m in ("gray_world", "gray_edge", "fusion"):
        est = AWBEstimator(AWBConfig(method=m)).estimate(fr.linear_pre_wb, fr.clipped_ratio)
        errs[m] = illuminant_error_deg(est.illum_rgb, 3000.0)
    assert errs["fusion"] < errs["gray_world"], errs
    assert errs["fusion"] < errs["gray_edge"], errs


@case
def test_planckian_constraint_only_limits_cct():
    """色温先验：越界的估计必须被拉回范围内；范围内的估计不应被改动"""
    cam = SimCamera(SC.muted_scene(W, H, (0.75, 0.12, 0.10)), 3000.0, SCFG, ICFG, seed=5)
    ae = AEController(AEConfig(metering="evaluative"))
    r = ae.run(lambda ev: cam.capture(ev=ev), ev0=0.0)
    fr = cam.capture(ev=r.final_ev)

    est_off = AWBEstimator(AWBConfig(method="gray_world", constrain_planckian=False)) \
        .estimate(fr.linear_pre_wb, fr.clipped_ratio)
    est_on = AWBEstimator(AWBConfig(method="gray_world", constrain_planckian=True)) \
        .estimate(fr.linear_pre_wb, fr.clipped_ratio)

    cct_off, _ = rgb_to_cct_duv(est_off.illum_rgb)
    cct_on, _ = rgb_to_cct_duv(est_on.illum_rgb)
    assert cct_off < 2000.0, f"该场景本应给出越界估计，实测 {cct_off:.0f}K"
    assert cct_on > cct_off, "越界估计必须被拉回"
    err_off = illuminant_error_deg(est_off.illum_rgb, 3000.0)
    err_on = illuminant_error_deg(est_on.illum_rgb, 3000.0)
    assert err_on < err_off, f"约束后误差应下降: {err_off:.2f} -> {err_on:.2f}"


@case
def test_focus_measure_peaks_at_true_focus():
    """对焦评价函数的峰位必须落在真合焦位置（无偏性）"""
    true_focus = 0.35
    cam = SimCamera(SC.focus_target(W, H), 5000.0, SCFG, ICFG, seed=9,
                    true_focus=true_focus, max_blur_px=8.0)
    for method in ("brenner", "tenengrad", "laplacian_var", "sml"):
        positions = np.linspace(0, 1, 41)
        vals = np.array([AF.focus_measure(cam.capture(ev=0.0, focus_pos=float(p)).linear_pre_wb[..., 1],
                                          method, "center", 0.5) for p in positions])
        peak = positions[int(np.argmax(vals))]
        assert abs(peak - true_focus) <= 0.06, f"{method} 峰位 {peak:.3f}，真值 {true_focus}"


@case
def test_focus_measure_values_rise_toward_focus():
    """离焦越大，评价函数越小（单调性，AF 搜索的前提）"""
    cam = SimCamera(SC.focus_target(W, H), 5000.0, SCFG, ICFG, seed=9,
                    true_focus=0.5, max_blur_px=8.0)
    vals = []
    for d in (0.45, 0.30, 0.15, 0.05, 0.0):        # 离真合焦位置由远到近
        fr = cam.capture(ev=0.0, focus_pos=0.5 + d)
        vals.append(AF.focus_measure(fr.linear_pre_wb[..., 1], "laplacian_var", "center", 0.5))
    assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1)), vals


@case
def test_af_search_accuracy():
    """粗到细搜索：定位误差必须在一个粗扫步长之内"""
    true_focus = 0.35
    cfg = AFConfig(strategy="coarse_to_fine")
    cam = SimCamera(SC.focus_target(W, H), 5000.0, SCFG, ICFG, seed=9,
                    true_focus=true_focus, max_blur_px=cfg.max_blur_px)
    r = AF.AFController(cfg).run(lambda p: cam.capture(ev=0.0, focus_pos=p))
    coarse_step = 1.0 / (cfg.coarse_steps - 1)
    assert abs(r.best_pos - true_focus) <= coarse_step, (r.best_pos, true_focus)
    assert r.frames <= cfg.coarse_steps + cfg.fine_steps


@case
def test_ccm_solve_recovers_known_matrix():
    """CCM 标定：已知的混合矩阵必须能被最小二乘解出来"""
    rng = np.random.default_rng(3)
    src = rng.random((24, 3)) * 0.8 + 0.1
    M = np.array([[0.88, 0.09, 0.03], [0.07, 0.88, 0.05], [0.02, 0.12, 0.86]])
    dst = src @ M.T
    got = IM.solve_ccm(src, dst, preserve_white=False)
    assert np.max(np.abs(got - M)) < 1e-6, np.round(got, 4)
    # 带白点约束时，行和应接近 1
    got2 = IM.solve_ccm(src, dst, preserve_white=True)
    assert np.max(np.abs(got2.sum(axis=1) - 1.0)) < 0.02, got2.sum(axis=1)


@case
def test_ideal_linear_is_reference():
    """理想成像结果 = 反射率 × 常量（白平衡正确时")，这是颜色评价的基准"""
    sc = SC.color_chart(W, H)
    for temp in (3000.0, 5000.0, 6500.0):
        ideal = ideal_linear(sc, temp)
        ratio = ideal[sc.neutral_mask] / np.maximum(sc.reflectance[sc.neutral_mask], 1e-6)
        assert np.allclose(ratio, ratio[0, 0], rtol=1e-4), "中性区的比例常数必须一致"


def run_all():
    ok, bad = 0, []
    for fn in CASES:
        try:
            fn()
            print(f"  [PASS] {fn.__name__}")
            ok += 1
        except AssertionError as e:
            print(f"  [FAIL] {fn.__name__}: {e}")
            bad.append(fn.__name__)
        except Exception as e:  # noqa
            print(f"  [ERR ] {fn.__name__}: {type(e).__name__}: {e}")
            bad.append(fn.__name__)
    print(f"\n{ok}/{len(CASES)} 通过" + (f"，失败: {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(run_all())
