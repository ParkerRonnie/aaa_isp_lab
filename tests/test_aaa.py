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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from aaa_isp_lab.config import (SensorConfig, ISPConfig, AEConfig, AWBConfig, AFConfig,
                                TemporalConfig)
from aaa_isp_lab.eval.metrics import delta_e_2000, psnr, ssim, ideal_linear
from aaa_isp_lab.color_science import (blackbody_linear_rgb, rgb_to_cct_duv, cct_to_xy,
                               xy_to_cct, rgb_linear_to_lab, linear_to_srgb)
from aaa_isp_lab.isp import modules as IM
from aaa_isp_lab.sim import scene as SC
from aaa_isp_lab.sim.camera import SimCamera, ev_limits, split_exposure, flicker_banding_metric
from aaa_isp_lab.sim.optics import defocus_kernel
from aaa_isp_lab.aaa.ae import AEController, metering_metric
from aaa_isp_lab.aaa.awb import (AWBEstimator, AWBStabilizer, ideal_gains,
                                 illuminant_error_deg)
from aaa_isp_lab.aaa import af as AF
from aaa_isp_lab.aaa.temporal import EMAFilter, SceneCutDetector, structure_signal
from aaa_isp_lab.eval import image_quality as IQ
from aaa_isp_lab.eval import metrics as MT

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
    from aaa_isp_lab.color_science import srgb_to_linear
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
    from aaa_isp_lab.sim.camera import apply_flicker_banding
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
    from aaa_isp_lab.eval.metrics import neutral_chroma
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

    from aaa_isp_lab.aaa.awb import valid_mask
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


# -----------------------------------------------------------------------------
# 4. 画质指标测量（每一项都要能和真值对上）
# -----------------------------------------------------------------------------
@case
def test_mtf_recovers_known_psf():
    """斜边法 MTF 必须能还原已知高斯 PSF 的解析解。

    这条测试是这一组测量可信度的基础：如果 MTF 测不准，
    报告里所有"清晰度"相关的数字都是废的。
    """
    for sigma in (0.7, 1.0, 1.5, 3.0):
        img = IQ.synthetic_slanted_edge(200, 160, sigma, angle_deg=5.0)
        r = IQ.slanted_edge_mtf(img)
        f_fine = np.linspace(0.0, 0.5, 4001)
        th_fine = IQ.gaussian_mtf_theory(f_fine, sigma, pixel_aperture=True)
        idx = np.where(th_fine < 0.5)[0]
        mtf50_th = float(f_fine[idx[0]])
        dev = abs(r.mtf50 / mtf50_th - 1.0)
        assert dev < 0.08, f"σ={sigma}: MTF50 {r.mtf50:.3f} vs 理论 {mtf50_th:.3f}（偏差 {dev:.1%}）"
        th = IQ.gaussian_mtf_theory(r.freqs, sigma, pixel_aperture=True)
        assert np.max(np.abs(r.mtf - th)) < 0.06, f"σ={sigma}: MTF 曲线偏差过大"
        # 边缘角度也要能测出来（斜边法的前提）
        assert abs(r.edge_angle_deg - 5.0) < 0.3, f"测出的边缘角度 {r.edge_angle_deg:.2f}°"


@case
def test_mtf_monotone_in_blur():
    """模糊越大 MTF50 越小（单调性）。反了说明测量链路有问题。"""
    vals = [IQ.slanted_edge_mtf(
        IQ.synthetic_slanted_edge(200, 160, sg, angle_deg=5.0)).mtf50
        for sg in (0.5, 1.0, 2.0, 4.0)]
    assert all(vals[i] > vals[i + 1] for i in range(len(vals) - 1)), vals


@case
def test_photon_transfer_recovers_sensor():
    """光子转换曲线必须把仿真的转换增益与读出噪声反推回来。"""
    cfg = SensorConfig(width=256, height=192, vignetting_strength=0.0)
    sc = SC.uniform_scene(256, 192, level=0.6)
    patch = (100, 140, 140, 180)
    pts = []
    for ev in np.linspace(-8.0, 0.5, 12):
        acc_m, acc_v = 0.0, 0.0
        reps = 3
        for seed in range(reps):
            cam = SimCamera(sc, 5000.0, cfg, ICFG, seed=700 + seed)
            fr = cam.capture(ev=float(ev))
            g = IQ.patch_channel_stats(fr.raw_dn, patch, cam.isp.pattern,
                                       cfg.black_level_dn)["G"]
            acc_m += g.mean_dn
            acc_v += g.std_dn ** 2
        pts.append(IQ.NoisePoint(acc_m / reps, float(np.sqrt(acc_v / reps)), reps))

    fit = IQ.fit_photon_transfer(pts, saturation_dn=cfg.signal_dn)
    k_true = cfg.full_well_e / cfg.signal_dn
    assert abs(fit.gain_e_per_dn / k_true - 1.0) < 0.05,         f"转换增益 K 偏差过大: {fit.gain_e_per_dn:.4f} vs {k_true:.4f}"
    # 12 个点、每点 3 帧平均，R² 到 0.999 量级即可（阈值定太死会变成
    # 在测随机数，测试本身反而变脆）
    assert fit.r2 > 0.995, f"拟合优度过低: {fit.r2}"
    # 读出噪声会叠加量化噪声，允许较宽的容差
    assert abs(fit.read_noise_e - cfg.read_noise_e) < 0.6,         f"读出噪声 {fit.read_noise_e:.3f} e- vs 真值 {cfg.read_noise_e}"


@case
def test_ptc_must_reject_saturated_points():
    """饱和点必须剔除：含着它拟合会把转换增益反推得明显偏大。

    这是本项目实测到的坑（不剔除时 K 被反推成真值的 1.5 倍），
    所以专门加一条测试把它钉住。
    """
    cfg = SensorConfig(width=256, height=192, vignetting_strength=0.0)
    sc = SC.uniform_scene(256, 192, level=0.6)
    patch = (100, 140, 140, 180)
    pts = []
    for ev in np.linspace(-4.0, 1.2, 10):     # 故意包含过曝点
        cam = SimCamera(sc, 5000.0, cfg, ICFG, seed=800)
        fr = cam.capture(ev=float(ev))
        g = IQ.patch_channel_stats(fr.raw_dn, patch, cam.isp.pattern,
                                   cfg.black_level_dn)["G"]
        pts.append(IQ.NoisePoint(g.mean_dn, g.std_dn, g.n_pixels))
    k_true = cfg.full_well_e / cfg.signal_dn
    with_clip = IQ.fit_photon_transfer(pts)
    without = IQ.fit_photon_transfer(pts, saturation_dn=cfg.signal_dn)
    err_clip = abs(with_clip.gain_e_per_dn / k_true - 1.0)
    err_clean = abs(without.gain_e_per_dn / k_true - 1.0)
    assert err_clean < 0.05, f"剔除饱和点后 K 仍偏 {err_clean:.1%}"
    assert err_clip > err_clean, "剔除饱和点应当让结果更好"


@case
def test_dynamic_range_formula():
    cfg = SensorConfig()
    k = cfg.full_well_e / cfg.signal_dn
    dr = IQ.dynamic_range_db(cfg.signal_dn, cfg.read_noise_e / k)
    assert abs(dr - IQ.sensor_dr_theory_db(cfg.full_well_e, cfg.read_noise_e)) < 1e-6
    # 满阱翻倍 -> +6 dB；读出噪声翻倍 -> -6 dB
    assert abs(IQ.dynamic_range_db(2 * cfg.signal_dn, cfg.read_noise_e / k) - dr - 6.02) < 0.1
    assert abs(IQ.dynamic_range_db(cfg.signal_dn, 2 * cfg.read_noise_e / k) - dr + 6.02) < 0.1


@case
def test_shading_metrics():
    """均匀图：均匀度 1、色阴影 0；带阴影的图：均匀度明显小于 1"""
    flat = np.full((120, 160, 3), 0.5, dtype=np.float32)
    m = IQ.shading_metrics(flat)
    assert abs(m["luma_uniformity"] - 1.0) < 1e-3, m
    assert m["d_uv_corner_max"] < 0.5, m

    from aaa_isp_lab.sim import optics
    vig = optics.apply_vignetting(flat, strength=0.5)
    m2 = IQ.shading_metrics(vig)
    assert m2["luma_uniformity"] < 0.75, m2
    # 镜头阴影本身是**分通道**衰减（三个通道衰减量不同），所以亮度阴影
    # 必然带出色阴影 —— 这正是 LSC 必须按 CFA 通道分别补偿的原因。
    # 一开始这条断言写成"纯亮度阴影不该有色阴影"，把模型的设计意图搞反了。
    assert m2["d_uv_corner_max"] > 1.0, m2

    # 反例：只有亮度衰减、三通道完全一致的阴影，不应产生色阴影
    gray_vig = vig.mean(axis=2, keepdims=True) * np.ones(3, dtype=np.float32)
    m3 = IQ.shading_metrics(gray_vig)
    assert m3["luma_uniformity"] < 0.75
    assert m3["d_uv_corner_max"] < 0.5, m3


@case
def test_gain_referred_model_lowers_dark_noise():
    """读出噪声后置的模型下，提高增益必须压低暗噪声折算值。

    这是"ISO 存在的意义"的量化证据，也是前面"增益不改善 SNR"说法的边界。
    """
    flat = SC.uniform_scene(192, 144, level=0.5)
    dark = SC.dim(flat, 0.0, "black")
    patch = (60, 84, 84, 108)
    noise = {}
    for model in ("iso_less", "gain_referred"):
        vals = []
        for g in (1, 16):
            cfg = SensorConfig(width=192, height=144, vignetting_strength=0.0,
                               read_noise_model=model, max_analog_gain=float(g),
                               bit_depth=14)
            cam = SimCamera(dark, 5000.0, cfg, ICFG, seed=900)
            fr = cam.capture(ev=float(np.log2(g)), ae_cfg=AEConfig(priority="gain_priority"))
            st = IQ.patch_channel_stats(fr.raw_dn, patch, cam.isp.pattern,
                                        cfg.black_level_dn)["G"]
            vals.append(st.std_dn * (cfg.full_well_e / cfg.signal_dn))
        noise[model] = vals
    # ISO 无关：高低增益的暗噪声折算值基本一样
    assert abs(noise["iso_less"][1] / noise["iso_less"][0] - 1.0) < 0.15, noise["iso_less"]
    # 读出噪声后置：16× 增益下暗噪声折算值必须明显下降
    assert noise["gain_referred"][1] < 0.7 * noise["gain_referred"][0], noise["gain_referred"]



# -----------------------------------------------------------------------------
# 时域
# -----------------------------------------------------------------------------
def _seq_camera(scene, seed=67, temp=5000.0):
    cam = SimCamera(scene, temp, SCFG, ICFG, seed=seed)
    return cam


def _stream(cam, ae_cfg=None, tcfg=None, n=40, ev0=0.0, on_frame=None):
    ctl = AEController(ae_cfg or AEConfig(), tcfg)
    return ctl.run_stream(lambda ev: cam.capture(ev=ev, wb_gains=ideal_gains(5000.0)),
                          ev0=ev0, n_frames=n, on_frame=on_frame)


@case
def test_run_stream_matches_run_without_filter():
    """钉住 _observe() 重构：无滤波时 run_stream 的前缀必须与 run 逐位一致。

    这是整块重构的安全网 —— 抽出 _observe() 时只要控制律有一行改动，
    既有的 8 处调用点和全部历史结论就都不可比了。"""
    sc = SC.natural_scene(W, H, backlit=True)
    for ev0 in (3.0, -3.0):
        cam1 = _seq_camera(sc, seed=7)
        r1 = AEController(AEConfig()).run(
            lambda ev: cam1.capture(ev=ev, wb_gains=ideal_gains(5000.0)), ev0=ev0)
        r2 = _stream(_seq_camera(sc, seed=7), n=40, ev0=ev0)
        n = len(r1.history)
        assert n > 0
        for a, b in zip(r1.history, r2.history[:n]):
            assert abs(a["ev"] - b["ev"]) < 1e-15, f"ev0={ev0} EV 序列不一致"
            assert abs(a["err_ev"] - b["err_ev"]) < 1e-15, f"ev0={ev0} 误差序列不一致"
        assert r1.iters == n


@case
def test_structure_signal_matches_texture_acutance():
    """结构信号与 eval 侧刻意重复实现，必须逐位相等。

    aaa 层不得 import eval（分层约束），所以 formula 抄了一份。
    这个测试就是防止两边悄悄漂移 —— 漂移了检测器就会用另一套公式，
    而报告里还写着"同一个量"。"""
    cam = _seq_camera(SC.natural_scene(W, H, backlit=True))
    fr = cam.capture(ev=0.0, wb_gains=ideal_gains(5000.0))
    a = structure_signal(fr.luma_linear)
    b = IQ.texture_acutance(fr.luma_linear)
    assert abs(a - b) < 1e-15, f"结构信号漂移了: {a} vs {b}"
    # 亮度整体缩放不变（这是它能当"只看内容"的判别器的原因）。
    # 注意要在 float64 里缩放：luma_linear 是 float32，直接 *100 会带进 ~1e-7
    # 的相对误差，把这条性质测试变成"测浮点精度"而不是"测标度不变性"。
    scaled = fr.luma_linear.astype(np.float64) * 100.0
    assert abs(structure_signal(scaled) - a) < 1e-12, \
        f"结构信号对亮度缩放不再不变: {a} vs {structure_signal(scaled)}"


@case
def test_jitter_floor_is_tiny():
    """**把负结论固化成测试**：不注入扰动时，整帧测光的抖动等于浮点噪声。

    480x360 下散粒噪声被空间平均掉，实测噪声地板约 1e-4 EV —— 比收敛阈值
    低两个数量级。所以任何"时域滤波把抖动降低 90%"的说法在这个分辨率下
    都是伪结论。这条测试就是防止以后有人（包括我自己）忘了这一点，
    拿一个凭空冒出来的"显著改善"去写报告。"""
    cam = _seq_camera(SC.natural_scene(W, H, backlit=True), seed=83)
    r = _stream(cam, n=48)
    h = r.history[24:]
    met = [abs(hx["err_ev"]) for hx in h]
    std = float(np.std(met))
    assert std < 1e-3, f"无注入时的抖动应等于浮点噪声，实测 std={std:.3e} EV"


@case
def test_no_cut_on_ae_own_convergence():
    """**最要命的一条假阳性**：AE 自己从 +3 EV 收敛，检测器必须零触发。

    根因是过曝时像素饱和会同时破坏曝光归一化与标度不变性 —— 实测三个信号
    全部超阈。护栏是"检测器只在 AE 连续稳定若干帧后才武装"，这条测试钉住它。"""
    tcfg = TemporalConfig(cut_enable=True)
    for ev0 in (3.0, 1.5):
        cam = _seq_camera(SC.natural_scene(W, H, backlit=True), seed=67)
        r = _stream(cam, tcfg=tcfg, n=40, ev0=ev0)
        assert len(r.cut_frames) == 0, \
            f"AE 自身收敛（ev0={ev0}）被误判成场景切换：帧 {r.cut_frames}"


@case
def test_scene_cut_detected_on_illumination_step():
    """光照阶跃必须在 3 帧内被检出，且类型判为光照（不是内容）。"""
    tcfg = TemporalConfig(cut_enable=True)
    sc = SC.natural_scene(W, H, backlit=True)
    cam = _seq_camera(sc, seed=67)
    cut = 20

    def on_frame(it):
        if it == cut:
            cam.set_scene(SC.dim(sc, 4.0))

    r = _stream(cam, tcfg=tcfg, n=40, on_frame=on_frame)
    assert r.cut_frames, "光照阶跃未被检出"
    lat = min(r.cut_frames) - cut
    assert 0 <= lat <= 3, f"检出延迟过大: {lat} 帧"
    assert r.history[min(r.cut_frames)]["cut_kind"] == "illumination", \
        "光照变化被判成了内容变化（分类用了 hist_dist 就会这样）"


@case
def test_executor_quantization_creates_limit_cycle():
    """S3：执行器量化**不需要任何噪声**就能造出抖动 —— 抖动来自控制结构。

    暗场景下曝光时间顶在上限、增益参与调节，所以增益档位量化起决定作用。
    步长越大，极限环越大。这条测试证明"时域抖动"不是噪声的产物。"""
    sc = SC.dim(SC.natural_scene(W, H, backlit=True), 1.0 / 40.0)
    base = _stream(_seq_camera(sc, seed=83), n=60)
    floor = float(np.std([h["err_ev"] for h in base.history[30:]]))
    for step, factor in ((1.0 / 6.0, 50.0), (1.0 / 3.0, 80.0)):
        scfg = SensorConfig(width=W, height=H, gain_step_ev=step)
        cam = SimCamera(sc, 5000.0, scfg, ICFG, seed=83)
        r = AEController(AEConfig()).run_stream(
            lambda ev: cam.capture(ev=ev, wb_gains=ideal_gains(5000.0)), n_frames=60)
        j = float(np.std([h["err_ev"] for h in r.history[30:]]))
        ratio = j / max(floor, 1e-12)
        assert ratio > factor, \
            f"gain_step={step:.4f} 的极限环只有地板的 {ratio:.1f} 倍（期望 >{factor}）"


@case
def test_awb_stabilizer_reduces_flicker_without_bias():
    """AWB 是开环估计器，时域滤波在这里是**干净的收益**：
    增益跳动显著下降，而光源角度误差的**均值不能变差**（平滑不引入偏差）。"""
    sc = SC.color_chart(W, H)
    cam = _seq_camera(sc, seed=97)
    ev = AEController(AEConfig()).run(
        lambda e: cam.capture(ev=e, wb_gains=ideal_gains(5000.0)), ev0=-1.0).final_ev

    def run(alpha):
        c = _seq_camera(sc, seed=97)
        st = AWBStabilizer(AWBConfig(),
                           TemporalConfig(enable=True, alpha_slow=alpha) if alpha else None)
        g, err = [], []
        for _ in range(40):
            fr = c.capture(ev=ev, wb_gains=ideal_gains(5000.0))
            a = st.estimate(fr.linear_pre_wb, clipped_ratio=fr.clipped_ratio)
            g.append(np.log2(np.clip(np.asarray(a.gains, float), 1e-9, None)))
            err.append(illuminant_error_deg(a.illum_rgb, 5000.0))
        g = np.asarray(g)[10:]
        return float(np.mean(np.abs(np.diff(g, axis=0)))), float(np.mean(err[10:]))

    f_raw, e_raw = run(None)
    f_fil, e_fil = run(0.2)
    assert f_fil < f_raw * 0.5, f"滤波没压住增益跳动: {f_raw:.5f} -> {f_fil:.5f}"
    assert abs(e_fil - e_raw) < 0.2, \
        f"滤波引入了偏差: 角度误差均值 {e_raw:.4f} -> {e_fil:.4f}"


@case
def test_temporal_filter_and_detector_disabled_are_noops():
    """默认关闭时，滤波与检测器都必须是恒等变换（既有实验数值不受影响）。"""
    f = EMAFilter(TemporalConfig(enable=False))
    assert f.smooth(1.0) == 1.0
    assert f.smooth(2.0) == 2.0

    cam = _seq_camera(SC.natural_scene(W, H, backlit=True))
    fr = cam.capture(ev=0.0, wb_gains=ideal_gains(5000.0))
    d = SceneCutDetector(TemporalConfig(cut_enable=False))
    for _ in range(5):
        assert not d.update(fr, 0.5).cut

    # AWB 稳定器在 temporal=None 时必须逐帧等价于原估计器
    est = AWBEstimator(AWBConfig())
    st = AWBStabilizer(AWBConfig(), None)
    a = est.estimate(fr.linear_pre_wb)
    b = st.estimate(fr.linear_pre_wb)
    assert np.allclose(a.gains, b.gains, atol=1e-12)


@case
def test_settle_frames_and_ringing_helpers():
    """评价口径本身也要有测试：未收敛必须如实返回 -1，不能拿"最后一帧达标"糊弄。"""
    assert MT.settle_frames([1.0, 1.0, 0.01, 0.01, 0.01], 0.05, hold=3) == 5
    assert MT.settle_frames([0.01, 0.01, 1.0], 0.05, hold=3) == -1
    # alpha 小 + 阻尼大 -> 共轭复极点，会振铃；alpha=1 是实极点，不振铃
    p = MT.pole_ringing_period(0.7, 0.2)
    assert np.isfinite(p) and p > 1.0, f"复极点应给出有限振铃周期，得到 {p}"
    assert not np.isfinite(MT.pole_ringing_period(0.7, 1.0)), "alpha=1 不应振铃"


@case
def test_executor_split_exposure_still_exact_when_quantization_off():
    """量化默认关闭时，split_exposure 必须与之前逐位一致（保护既有结论）。"""
    for ev in (-3.0, 0.0, 2.0):
        et, g = split_exposure(ev, SCFG)
        et2, g2 = split_exposure(ev, SensorConfig(width=W, height=H))
        assert et == et2 and g == g2
    et, g = split_exposure(0.0, SCFG)
    assert abs(et - 1.0 / 60.0) < 1e-12 and abs(g - 1.0) < 1e-12


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
