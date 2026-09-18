# -*- coding: utf-8 -*-
"""全部实验的定义。

每个 `exp_*` 函数负责一组实验，返回**纯数据**（数字 + ndarray），
不负责画图和排版 —— 呈现交给 `aaa_isp_lab.eval.report`，
编排交给 `aaa_isp_lab.cli`。这样实验本身可以单独调用、单独测试。
"""
import copy
import time

import numpy as np

from .config import SensorConfig, ISPConfig, AEConfig, AWBConfig, AFConfig
from .sim import scene as SC
from .sim.camera import SimCamera, ev_limits, flicker_banding_metric
from .sim.sensor import SensorSim
from .aaa.ae import AEController
from .aaa.awb import AWBEstimator, ideal_gains, illuminant_error_deg
from .color_science import rgb_to_cct_duv
from .aaa import af as AF
from .isp import modules as IM
from .eval import metrics as MT
from .eval import report as RP

N_CAPTURES = [0]      # 统计成像次数，报告里要写


def make_camera(scene, temp, scfg, icfg, seed=0, true_focus=0.5,
                max_blur_px=8.0, lsc_model="ideal", enable_lsc=True):
    cam = SimCamera(scene, temp, scfg, icfg, seed=seed, true_focus=true_focus,
                    lsc_model=lsc_model, max_blur_px=max_blur_px)
    cam.isp.cfg = copy.deepcopy(cam.isp.cfg)
    cam.isp.cfg.enable_lsc = enable_lsc
    base = cam.capture
    cam.capture = lambda *a, **k: (N_CAPTURES.__setitem__(0, N_CAPTURES[0] + 1),
                                   base(*a, **k))[1]
    return cam


# -----------------------------------------------------------------------------
# 1. ISP 链路
# -----------------------------------------------------------------------------
def exp_isp(scfg, icfg, size):
    """ISP 各阶段中间图 + LSC 开/关对照。"""
    sc = SC.color_chart(*size)
    cam = make_camera(sc, 5000.0, scfg, icfg, seed=3)
    fr = cam.capture(ev=0.0, wb_gains=ideal_gains(5000.0))

    # 单独跑一遍各阶段，取中间图像
    lin = IM.black_level_correct(fr.raw_dn, scfg.black_level_dn, scfg.signal_dn)
    lsc = IM.lsc_radial_model(size[1], size[0], scfg.vignetting_strength,
                              max_gain=icfg.lsc_max_gain)
    lin_lsc = IM.apply_lsc_bayer(lin, lsc, cam.isp.pattern)
    rgb = IM.demosaic(lin_lsc, cam.isp.pattern, icfg.demosaic)
    wb = IM.apply_wb(rgb, ideal_gains(5000.0))
    ccm = IM.apply_ccm(wb, cam.isp.ccm)
    disp = IM.tone_map(ccm, icfg.tone_mode, icfg.shoulder, icfg.contrast)

    # 均匀度必须在**匀光板**上测：色卡的中心与四角本来就是不同色块，
    # 在色卡上算"四角/中心"测的是画面内容，不是镜头阴影。
    sc_u = SC.uniform_scene(*size)
    cam_u_on = make_camera(sc_u, 5000.0, scfg, icfg, seed=3, enable_lsc=True)
    cam_u_off = make_camera(sc_u, 5000.0, scfg, icfg, seed=3, enable_lsc=False)
    img_off = cam_u_off.capture(ev=0.0, wb_gains=ideal_gains(5000.0)).linear_pre_wb
    img_on = cam_u_on.capture(ev=0.0, wb_gains=ideal_gains(5000.0)).linear_pre_wb

    return {
        "stages": {"RAW(Bayer)": fr.raw_dn / scfg.max_dn, "BLC+LSC": lin_lsc,
                   "去马赛克": rgb, "白平衡": wb, "CCM": ccm, "色调映射": disp},
        "no_lsc": img_off,
        "with_lsc": img_on,
        "uniformity_no_lsc": _uniformity(img_off),
        "uniformity_with_lsc": _uniformity(img_on),
    }


def _uniformity(img: np.ndarray) -> float:
    """亮度均匀度：四角平均 / 中心（越接近 1 越均匀）"""
    h, w = img.shape[:2]
    c = img[h // 2 - h // 12:h // 2 + h // 12, w // 2 - w // 12:w // 2 + w // 12].mean()
    k = max(1, int(min(h, w) * 0.06))
    corners = np.mean([img[:k, :k].mean(), img[:k, -k:].mean(),
                       img[-k:, :k].mean(), img[-k:, -k:].mean()])
    return float(corners / max(c, 1e-6))


# -----------------------------------------------------------------------------
# 2. AE 测光方式
# -----------------------------------------------------------------------------
def exp_ae_metering(scfg, icfg, size, ae_cfg):
    sc = SC.natural_scene(*size, backlit=True)
    modes = ["average", "center", "spot", "evaluative", "highlight_priority"]
    out = {"modes": {}, "img": {}, "target_code": MT.code_value(ae_cfg.target_linear)}

    subject = sc.subject_mask        # 场景自带的主体掩码

    for m in modes:
        cam = make_camera(sc, 5000.0, scfg, icfg, seed=7)
        cfg = copy.deepcopy(ae_cfg)
        cfg.metering = m
        ae = AEController(cfg)
        r = ae.run(lambda ev: cam.capture(ev=ev, wb_gains=ideal_gains(5000.0)), ev0=2.0)
        fr = cam.capture(ev=r.final_ev, wb_gains=ideal_gains(5000.0))
        out["modes"][m] = {
            "iters": r.iters, "ev": r.final_ev, "converged": bool(r.converged),
            "code": MT.code_value(r.final_metric),
            "mean_code": MT.code_value(float(fr.luma_linear.mean())),
            "subject_code": MT.code_value(float(fr.luma_linear[subject].mean())),
            "clip": float(fr.clipped_ratio),
        }
        out["img"][m] = fr.srgb_u8
    return out


# -----------------------------------------------------------------------------
# 3. AE 收敛过程（步长策略）
# -----------------------------------------------------------------------------
def exp_ae_convergence(scfg, icfg, size, ae_cfg):
    sc = SC.natural_scene(*size, backlit=True)
    runs = []
    settings = [("固定大步长 (阻尼 1.0)", 1.0, False),
                ("固定小步长 (阻尼 0.4)", 0.4, False),
                ("变步长（本项目采用）", 0.7, True)]
    for label, damp, adaptive in settings:
        cam = make_camera(sc, 5000.0, scfg, icfg, seed=11)
        cfg = copy.deepcopy(ae_cfg)
        cfg.damping = damp
        ae = AEController(cfg)
        r = ae.run(lambda ev: cam.capture(ev=ev, wb_gains=ideal_gains(5000.0)),
                   ev0=3.0, max_iters=14, adaptive=adaptive)
        runs.append({
            "label": label,
            "ev_hist": [h["ev"] for h in r.history] + [r.final_ev],
            "err_hist": [abs(h["err_ev"]) for h in r.history],
            "reversals": r.reversals,
            "final_err": abs(r.history[-1]["err_ev"]) if r.history else 0.0,
        })
    return {"runs": runs, "target_linear": ae_cfg.target_linear}


# -----------------------------------------------------------------------------
# 4. AE 标定曲线（EV -> 亮度）
# -----------------------------------------------------------------------------
def exp_ae_curve(scfg, icfg, size, ae_cfg):
    sc = SC.color_chart(*size)
    cam = make_camera(sc, 5000.0, scfg, icfg, seed=13)
    ae = AEController(ae_cfg)
    lut = ae.calibrate(lambda ev: cam.capture(ev=ev, add_noise=False), n=17,
                       ev_span=(-4.0, 4.0))
    codes = [MT.code_value(v) for v in lut["metric"]]
    target_code = MT.code_value(ae_cfg.target_linear)

    # 各测光方式的收敛点落在曲线上的位置
    pts = {}
    for m in ("average", "evaluative"):
        cfg = copy.deepcopy(ae_cfg)
        cfg.metering = m
        c2 = make_camera(sc, 5000.0, scfg, icfg, seed=13)
        r = AEController(cfg).run(lambda ev: c2.capture(ev=ev), ev0=2.0)
        pts[m] = (r.final_ev, MT.code_value(r.final_metric))

    return {"calib_evs": lut["evs"].tolist(), "calib_code": codes,
            "target_code": target_code, "converged_points": pts}


# -----------------------------------------------------------------------------
# 5. AE 执行器分配（快门 vs 增益）+ 抗闪烁
# -----------------------------------------------------------------------------
def exp_ae_policy(scfg, icfg, size, ae_cfg):
    # 必须在**低照度**下做这个对比：光照充足时 EV<0，增益被钳在 1.0，
    # 两种策略结果完全一样，看不出区别（第一版就踩了这个坑）。
    # 把场景整体调暗 12 倍（约 3.6 档），EV>0 才需要动用增益。
    sc = SC.dim(SC.focus_target(*size), 1.0 / 12.0, "focus_target_dim")
    rows = []
    for policy in ("shutter_priority", "gain_priority"):
        cfg = copy.deepcopy(ae_cfg)
        cfg.priority = policy
        cam = make_camera(sc, 5000.0, scfg, icfg, seed=17)
        ae = AEController(cfg)
        r = ae.run(lambda ev: cam.capture(ev=ev, ae_cfg=cfg, wb_gains=ideal_gains(5000.0)),
                   ev0=0.0)
        fr = cam.capture(ev=r.final_ev, ae_cfg=cfg, wb_gains=ideal_gains(5000.0))
        # 清晰度用对焦评价函数衡量：长曝光 + 运动 → 画面糊 → 评价值下降
        sharp = AF.focus_measure(fr.linear_pre_wb[..., 1], "tenengrad", "center", 0.5)
        rows.append({
            "policy": "优先快门" if policy == "shutter_priority" else "优先增益",
            "et_ms": fr.exposure_s * 1000.0, "gain": float(fr.gain),
            "snr_db": float(SensorSim.snr_db(fr.raw_dn, scfg)),
            "sharpness": sharp,
            "clip": float(fr.clipped_ratio),
        })

    # 带纹（抗闪烁）单独在 exp_ae_flicker 里测：
    # 那里用的是**匀光板**场景，行轮廓平坦，指标才有意义；
    # 这里如果顺手在标板上测，测到的是场景本身的行结构，是伪结论。
    return {"rows": rows}


def exp_ae_flicker(scfg, icfg, size, ae_cfg):
    # 带纹指标看的是"行均值轮廓上的周期性波动"，所以测试场景必须行轮廓平坦。
    # 用风景/色卡会把地平线、色卡行结构误判成带纹 —— 真实产线测带纹也是用
    # 匀光板/白墙，而不是随便找个场景拍。第一版用色卡测，结果"开抗闪烁"
    # 反而比关掉更差，就是场景选错导致的伪结论。
    sc = SC.uniform_scene(*size)
    runs = []
    for anti in (False, True):
        cfg = copy.deepcopy(ae_cfg)
        cfg.anti_flicker = anti
        cam = make_camera(sc, 5000.0, scfg, icfg, seed=23)
        cam.flicker_amplitude = 0.25
        r = AEController(cfg).run(
            lambda ev: cam.capture(ev=ev, ae_cfg=cfg, wb_gains=ideal_gains(5000.0)), ev0=-1.0)
        fr = cam.capture(ev=r.final_ev, ae_cfg=cfg, wb_gains=ideal_gains(5000.0))
        runs.append({
            "label": "抗闪烁 开" if anti else "抗闪烁 关",
            "ev_hist": [h["ev"] for h in r.history] + [r.final_ev],
            "et_ms": fr.exposure_s * 1000.0,
            "banding": flicker_banding_metric(fr.luma_linear),
            "gain": float(fr.gain),
            "snr_db": float(SensorSim.snr_db(fr.raw_dn, scfg)),
        })
    return {"runs": runs,
            "banding_off": runs[0]["banding"], "banding_on": runs[1]["banding"],
            "et_off_ms": runs[0]["et_ms"], "et_on_ms": runs[1]["et_ms"]}


# -----------------------------------------------------------------------------
# 6. AWB 算法对比
# -----------------------------------------------------------------------------
def exp_awb(scfg, icfg, size, awb_cfg):
    cases = [("color_chart", SC.color_chart(*size), 5000.0),
             ("natural_backlit", SC.natural_scene(*size, backlit=True), 6500.0),
             ("muted_red", SC.muted_scene(*size), 3000.0)]
    methods = ["gray_world", "white_patch", "gray_edge", "shades_of_gray", "fusion"]
    out = {"methods": methods, "scenes": {}}

    for name, sc, temp in cases:
        cam = make_camera(sc, temp, scfg, icfg, seed=29)
        # 先让 AE 把曝光调好再评 AWB：真实 AWB 就是在正常曝光的帧上工作的。
        # 若固定在 EV=0 拍，不同场景的实际曝光天差地别（亮场景直接过曝），
        # 测出来的 AWB 误差里混进了"曝光不对"这个无关变量。
        ae = AEController(AEConfig(metering="evaluative", damping=0.7))
        r = ae.run(lambda ev: cam.capture(ev=ev), ev0=0.0)
        fr = cam.capture(ev=r.final_ev)
        no_wb = fr.linear_pre_wb
        no_wb_err = illuminant_error_deg(
            np.array([1.0, 1.0, 1.0], dtype=np.float64), temp)      # 无白平衡 = 假设光源为 D65 中性

        d = {"temp": temp, "no_wb_img": no_wb, "no_wb_err": no_wb_err, "img": {},
             "methods": {}}
        for m in methods:
            cfg = copy.deepcopy(awb_cfg)
            cfg.method = m
            est = AWBEstimator(cfg).estimate(fr.linear_pre_wb, fr.clipped_ratio)
            img = IM.apply_wb(fr.linear_pre_wb, est.gains)
            d["img"][m] = img
            d["methods"][m] = {
                "err_deg": illuminant_error_deg(est.illum_rgb, temp),
                "cct": float(est.cct),
                "neutral_chroma": float(MT.neutral_chroma(img, sc.neutral_mask)),
                "weights": est.weights,
            }
        out["scenes"][name] = d
    return out


# -----------------------------------------------------------------------------
# 7. 色温先验约束
# -----------------------------------------------------------------------------
def exp_awb_constraint(scfg, icfg, size, awb_cfg):
    """色温先验约束到底管住了什么、没管住什么。

    这是本项目里最"反直觉"的一组结果，也是值得讲的一点：
    把估计光源约束到普朗克轨迹附近，只能限制**沿轨迹方向**（色温）的误差；
    对**垂直于轨迹方向**（Duv）的误差完全无能为力。
    而大面积单色场景造成的偏色，恰恰主要落在 Duv 方向上 ——
    估计色温看着完全正常，颜色却是错的。
    """
    cases = [("红色主导", SC.muted_scene(*size, (0.75, 0.12, 0.10)), 3000.0),
             ("蓝色主导", SC.muted_scene(*size, (0.10, 0.25, 0.80)), 3000.0),
             ("标准色卡", SC.color_chart(*size), 5000.0)]

    rows = []
    for label, sc, temp in cases:
        cam = make_camera(sc, temp, scfg, icfg, seed=31)
        ae = AEController(AEConfig(metering="evaluative", damping=0.7))
        r0 = ae.run(lambda ev: cam.capture(ev=ev), ev0=0.0)
        fr = cam.capture(ev=r0.final_ev)

        out = []
        for on in (False, True):
            cfg = copy.deepcopy(awb_cfg)
            cfg.method = "gray_world"          # 用最容易失效的那一路，才看得出约束的作用
            cfg.constrain_planckian = on
            est = AWBEstimator(cfg).estimate(fr.linear_pre_wb, fr.clipped_ratio)
            c, d = rgb_to_cct_duv(est.illum_rgb)
            out.append({"cct": float(c), "duv": float(d),
                        "err": illuminant_error_deg(est.illum_rgb, temp)})
        rows.append({"label": label, "true_cct": temp,
                     "raw_cct": out[0]["cct"], "raw_duv": out[0]["duv"],
                     "raw_err": out[0]["err"],
                     "cct": out[1]["cct"], "duv": out[1]["duv"], "err": out[1]["err"]})

    n_out = sum(1 for r in rows if r["raw_cct"] < 2000.0 or r["raw_cct"] > 12000.0)
    n_improved = sum(1 for r in rows if r["err"] < r["raw_err"] - 0.5)
    return {"rows": rows, "true_cct": 3000.0,
            "summary": {"越界场景数": n_out, "约束后误差下降的场景数": n_improved,
                        "结论": "约束只作用于色温方向；Duv 方向的误差不受影响"}}


# -----------------------------------------------------------------------------
# 8. CCM 标定与 AWB/CCM 耦合
# -----------------------------------------------------------------------------
def exp_ccm(scfg, icfg, size, awb_cfg):
    """CCM 标定 + AWB/CCM 耦合。

    耦合实验的设计：先把 AWB 故意判错光源（用 6500K 的增益去修 5000K 的场景），
    再比较"不加 CCM"和"加 CCM"的 ΔE。CCM 是在**白平衡正确**的前提下标定的，
    一旦白平衡错了，CCM 的交叉项会把通道间的错误继续混合放大 ——
    这正是"CCM 不是万能补丁"的量化证据。
    """
    sc = SC.color_chart(*size)
    masks = sc.patch_masks
    TEMP = 5000.0
    ideal = MT.ideal_linear(sc, TEMP)
    cam = make_camera(sc, TEMP, scfg, icfg, seed=37)
    fr = cam.capture(ev=0.0)

    no_wb = fr.linear_pre_wb
    ideal_wb = IM.apply_wb(no_wb, ideal_gains(TEMP))
    est = AWBEstimator(awb_cfg).estimate(no_wb, fr.clipped_ratio)
    awb_wb = IM.apply_wb(no_wb, est.gains)

    # 色卡最小二乘标定 CCM（在正确白平衡的图像上做，这是标定的标准做法）
    src = np.stack([ideal_wb[m].mean(axis=0) for m in masks])
    dst = np.stack([ideal[m].mean(axis=0) for m in masks])
    ccm5000 = IM.solve_ccm(src, dst)

    de = {
        "无白平衡": float(np.mean(MT.patch_delta_e(no_wb, ideal, masks))),
        "理想白平衡": float(np.mean(MT.patch_delta_e(ideal_wb, ideal, masks))),
        "AWB 估计白平衡": float(np.mean(MT.patch_delta_e(awb_wb, ideal, masks))),
        "正白平衡 + CCM": float(np.mean(MT.patch_delta_e(
            IM.apply_ccm(ideal_wb, ccm5000), ideal, masks))),
    }

    # --- 耦合：AWB 判错光源（把 5000K 的场景当成 3000K 处理，偏差约 1 档色温）---
    wrong_wb = IM.apply_wb(no_wb, ideal_gains(3000.0))
    e_wrong_raw = float(np.mean(MT.patch_delta_e(wrong_wb, ideal, masks)))
    e_wrong_ccm = float(np.mean(MT.patch_delta_e(
        IM.apply_ccm(wrong_wb, ccm5000), ideal, masks)))

    # CCM 随光源重标：量化"换光源必须重标 CCM"这件事
    cam2 = make_camera(SC.color_chart(*size), 3000.0, scfg, icfg, seed=41)
    fr2 = cam2.capture(ev=0.0)
    ideal2 = MT.ideal_linear(SC.color_chart(*size), 3000.0)
    wb2 = IM.apply_wb(fr2.linear_pre_wb, ideal_gains(3000.0))
    src2 = np.stack([wb2[m].mean(axis=0) for m in masks])
    dst2 = np.stack([ideal2[m].mean(axis=0) for m in masks])
    ccm3000 = IM.solve_ccm(src2, dst2)

    return {"delta_e": de, "ccm": ccm5000.tolist(),
            "coupling": {"matched": float(np.mean(MT.patch_delta_e(
                IM.apply_ccm(ideal_wb, ccm5000), ideal, masks))),
                "mismatched": e_wrong_ccm},
            "wrong_wb": {"raw": e_wrong_raw, "with_ccm": e_wrong_ccm},
            "matrix": {"5000K": ccm5000.tolist(), "3000K": ccm3000.tolist(),
                       "理想": np.eye(3).tolist()}}


# -----------------------------------------------------------------------------
# 9. AF 评价函数
# -----------------------------------------------------------------------------
def exp_af_curves(scfg, icfg, af_cfg, size, repeats=3):
    sc = SC.focus_target(*size)
    true = 0.35
    cam = make_camera(sc, 5000.0, scfg, icfg, seed=43, true_focus=true,
                      max_blur_px=af_cfg.max_blur_px)
    data = AF.measure_curve(cam, af_cfg, n=41, repeats=repeats, ev=0.0)
    positions = data["positions"]
    mets, noise = {}, {}
    for m, curves in data["curves"].items():
        c0 = curves[0]
        mets[m] = AF.curve_metrics(positions, c0)
        if repeats > 1:
            peaks = [AF.curve_metrics(positions, c)["peak_pos"] for c in curves]
            noise[m] = float(np.std(peaks))
        else:
            noise[m] = 0.0
    return {"positions": positions.tolist(), "true_focus": true,
            "curves": {m: c[0].tolist() for m, c in data["curves"].items()},
            "metrics": mets, "noise_std": noise,
            "position_count": len(positions), "repeats": repeats}


# -----------------------------------------------------------------------------
# 10. AF 搜索策略
# -----------------------------------------------------------------------------
def exp_af_search(scfg, icfg, af_cfg, size):
    sc = SC.focus_target(*size)
    true = 0.35
    positions = np.linspace(0, 1, 81)
    cam0 = make_camera(sc, 5000.0, scfg, icfg, seed=47, true_focus=true,
                       max_blur_px=af_cfg.max_blur_px)
    curve = np.asarray([AF.frame_measure(cam0.capture(ev=0.0, focus_pos=float(p)), af_cfg)
                        for p in positions])
    rows = []
    for strat in ("sweep", "hill_climb", "coarse_to_fine", "golden_section"):
        cam = make_camera(sc, 5000.0, scfg, icfg, seed=51, true_focus=true,
                          max_blur_px=af_cfg.max_blur_px)
        cfg = copy.deepcopy(af_cfg)
        cfg.strategy = strat
        r = AF.AFController(cfg).run(lambda p: cam.capture(ev=0.0, focus_pos=p))
        rows.append({"strategy": strat, "frames": r.frames, "best": r.best_pos,
                     "err": abs(r.best_pos - true), "visited": r.visited})
    return {"rows": rows, "curve_pos": positions.tolist(), "curve": curve.tolist(),
            "true_focus": true}


# -----------------------------------------------------------------------------
# 11. 3A 耦合
# -----------------------------------------------------------------------------
def exp_coupling(scfg, icfg, ae_cfg, awb_cfg, af_cfg, size):
    """3A 耦合：把曝光从欠曝扫到过曝，看 AWB 与 AF 各自怎么劣化。

    注意这里评的是 AF 的**评价函数质量**（动态范围、峰位误差），
    不是搜索帧数 —— 帧数由搜索策略决定，与曝光无关。
    一开始想用"帧数随曝光变化"来讲故事，但数据不支持，属于拿结论凑现象。
    """
    chart = SC.color_chart(*size)
    target = SC.focus_target(*size)
    true_focus = 0.35
    positions = np.linspace(0.0, 1.0, 41)
    rows = []

    for ev in (-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0):
        cam = make_camera(chart, 5000.0, scfg, icfg, seed=53)
        fr = cam.capture(ev=ev)
        est = AWBEstimator(awb_cfg).estimate(fr.linear_pre_wb, fr.clipped_ratio)
        awb_err = illuminant_error_deg(est.illum_rgb, 5000.0)

        cam2 = make_camera(target, 5000.0, scfg, icfg, seed=57, true_focus=true_focus,
                           max_blur_px=af_cfg.max_blur_px)
        curve = np.asarray([AF.frame_measure(
            cam2.capture(ev=ev, focus_pos=float(p)), af_cfg) for p in positions])
        mt = AF.curve_metrics(positions, curve)
        rows.append({
            "ev": ev, "awb_err": awb_err,
            "af_peak_err": abs(mt["peak_pos"] - true_focus),
            "af_dyn_db": mt["dynamic_range_db"],
            "snr_db": float(SensorSim.snr_db(fr.raw_dn, scfg)),
            "clip": float(fr.clipped_ratio),
        })
    return {"rows": rows, "true_focus": true_focus}
