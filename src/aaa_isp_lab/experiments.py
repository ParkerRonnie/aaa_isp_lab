# -*- coding: utf-8 -*-
"""全部实验的定义。

每个 `exp_*` 函数负责一组实验，返回**纯数据**（数字 + ndarray），
不负责画图和排版 —— 呈现交给 `aaa_isp_lab.eval.report`，
编排交给 `aaa_isp_lab.cli`。这样实验本身可以单独调用、单独测试。
"""
import copy
import time

import cv2
import numpy as np

from .config import SensorConfig, ISPConfig, AEConfig, AWBConfig, AFConfig
from .sim import scene as SC
from .sim.camera import SimCamera, ev_limits, flicker_banding_metric
from .sim.sensor import SensorSim
from .aaa.ae import AEController, metering_metric
from .aaa.awb import (AWBEstimator, AWBStabilizer, compute_statistics, ideal_gains,
                      illuminant_error_deg)
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


# -----------------------------------------------------------------------------
# 12. 画质指标：MTF / 信噪比 / 动态范围 / 阴影
# -----------------------------------------------------------------------------
def exp_image_quality(scfg, icfg, size, af_cfg):
    """画质调优用的标准测量：每一项都先自检再使用。

    这一节的存在意义：前面所有实验都是"仿真内部"的对比（哪个算法更好），
    这一节回答的是**产品规格书上的那些数字**——MTF50 多少、SNR 多少、
    动态范围几档、色阴影几个单位。这些才是"画质调优"的通用语言。
    """
    from .eval import image_quality as IQ

    out = {}

    # --- (a) 斜边法 MTF 的链路自检：与已知高斯 PSF 的解析解对比 ---
    rows = []
    curve_sigma = 1.0
    for sg in (0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0):
        img = IQ.synthetic_slanted_edge(240, 200, sg, angle_deg=5.0)
        r = IQ.slanted_edge_mtf(img)
        th = IQ.gaussian_mtf_theory(r.freqs, sg, pixel_aperture=True)
        f_fine = np.linspace(0.0, 0.5, 4001)
        th_fine = IQ.gaussian_mtf_theory(f_fine, sg, pixel_aperture=True)
        idx = np.where(th_fine < 0.5)[0]
        mtf50_th = float(f_fine[idx[0]]) if idx.size else float("nan")
        rows.append({
            "sigma": float(sg), "angle": float(r.edge_angle_deg),
            "mtf50": float(r.mtf50), "mtf50_theory": mtf50_th,
            "dev_pct": float((r.mtf50 / mtf50_th - 1.0) * 100.0),
            "curve_err": float(np.max(np.abs(r.mtf - th))),
        })
        if abs(sg - curve_sigma) < 1e-9:
            out["_curve"] = {"freqs": r.freqs.tolist(), "mtf": r.mtf.tolist(),
                             "theory": th.tolist(), "sigma": float(sg)}
    out["mtf_selfcheck"] = {"rows": rows,
                            "max_dev_pct": float(max(abs(r["dev_pct"]) for r in rows)),
                            "max_curve_err": float(max(r["curve_err"] for r in rows))}

    # --- (b) 仿真相机的 MTF50 vs 镜头位置（与 AF 评价函数对照）---
    # 在**线性域**测：显示链路（色调曲线 + USM 锐化）会改变边缘形状，
    # 拿显示图测出来的不是成像系统的 MTF，而是"成像 + 后期"的合成结果。
    # 这里同时把两条都测出来，正好量化锐化对 MTF 数字的影响。
    sc_edge = SC.slanted_edge_target(size[0], size[1], angle_deg=5.0, supersample=4)
    positions = np.linspace(0.0, 1.0, 17)
    mtf50_lin, mtf50_disp, tenengrads = [], [], []
    for p_ in positions:
        cam = make_camera(sc_edge, 5000.0, scfg, icfg, seed=61,
                          true_focus=0.35, max_blur_px=af_cfg.max_blur_px)
        fr = cam.capture(ev=0.0, focus_pos=float(p_))
        try:
            mtf50_lin.append(float(IQ.slanted_edge_mtf(fr.linear_pre_wb[..., 1]).mtf50))
        except Exception:
            mtf50_lin.append(float("nan"))
        try:
            mtf50_disp.append(float(IQ.slanted_edge_mtf(fr.srgb_u8[..., 1]).mtf50))
        except Exception:
            mtf50_disp.append(float("nan"))
        tenengrads.append(float(AF.focus_measure(fr.linear_pre_wb[..., 1],
                                                 "tenengrad", "center", 0.5)))
    out["mtf_vs_focus"] = {"positions": positions.tolist(), "mtf50": mtf50_lin,
                           "mtf50_display": mtf50_disp, "tenengrad": tenengrads,
                           "true_focus": 0.35}

    # --- (c) 噪声曲线与光子转换曲线（在 RAW 上分通道测量）---
    sc_flat = SC.uniform_scene(size[0], size[1], level=0.6)
    scfg_flat = SensorConfig(width=scfg.width, height=scfg.height,
                             vignetting_strength=0.0)   # 关阴影，隔离噪声
    patch = (size[1] // 2 - 20, size[1] // 2 + 20,
             size[0] // 2 - 20, size[0] // 2 + 20)
    pts = []
    for ev in np.linspace(-9.0, 0.5, 20):
        acc_m, acc_v = 0.0, 0.0
        reps = 4
        for seed in range(reps):
            cam = make_camera(sc_flat, 5000.0, scfg_flat, icfg, seed=200 + seed)
            fr = cam.capture(ev=float(ev))
            g = IQ.patch_channel_stats(fr.raw_dn, patch, cam.isp.pattern,
                                       scfg_flat.black_level_dn)["G"]
            acc_m += g.mean_dn
            acc_v += g.std_dn ** 2
        pts.append(IQ.NoisePoint(acc_m / reps, float(np.sqrt(acc_v / reps)), reps))
    ptc = IQ.fit_photon_transfer(pts, saturation_dn=scfg_flat.signal_dn)
    k_true = scfg_flat.full_well_e / scfg_flat.signal_dn
    out["noise"] = {
        "points": [{"mean": p.mean_dn, "std": p.std_dn, "snr_db": p.snr_db}
                   for p in pts],
        "gain": {"measured": ptc.gain_e_per_dn, "true": k_true,
                 "dev_pct": (ptc.gain_e_per_dn / k_true - 1.0) * 100.0},
        "read_noise": {"measured": ptc.read_noise_e, "true": scfg_flat.read_noise_e,
                       "dev_pct": (ptc.read_noise_e / scfg_flat.read_noise_e - 1.0) * 100.0},
        "r2": ptc.r2, "n_points": ptc.n_points,
    }

    # --- (d) 暗噪声 vs 增益：提高增益到底改善了什么 ---
    # 直接拍**全黑帧**测噪声底（"暗噪声"的标准测法）。
    # 前两版分别用"两种曝光策略的 SNR"和"逐增益拟合 PTC 截距"，
    # 都不行：前者被硬件上限钳到同一点，后者的截距在方差里占比太小、
    # 被拟合噪声淹没（甚至拟合出负截距）。
    #
    # 位深提到 14bit：12bit 时量化噪声 1/12 DN² 折算到输入端约 0.86 e-，
    # 会盖过读出噪声（1.8 e-）随增益下降到 0.11 e- 的过程，
    # 把要观察的效应整个埋掉 —— 这是真实 ISP 里也会遇到的测量条件问题。
    dark = SC.dim(sc_flat, 0.0, "black")
    rows = []
    for model in ("iso_less", "gain_referred"):
        for g in (1, 2, 4, 8, 16):
            cfg_g = SensorConfig(width=scfg.width, height=scfg.height,
                                 vignetting_strength=0.0, read_noise_model=model,
                                 max_analog_gain=float(g), bit_depth=14)
            ae_g = AEConfig(priority="gain_priority")
            k_g = cfg_g.full_well_e / cfg_g.signal_dn
            sig_dn = []
            for seed in range(6):
                cam = make_camera(dark, 5000.0, cfg_g, icfg, seed=500 + seed)
                fr = cam.capture(ev=float(np.log2(g)), ae_cfg=ae_g)
                st = IQ.patch_channel_stats(fr.raw_dn, patch, cam.isp.pattern,
                                            cfg_g.black_level_dn)["G"]
                sig_dn.append(st.std_dn)
            sigma_dn = float(np.mean(sig_dn))
            sigma_e = sigma_dn * k_g            # 折算到输入端电子数
            s_e = 2.0
            snr_low = 20.0 * np.log10(s_e / np.sqrt(s_e + sigma_e ** 2))
            rows.append({"model": model, "gain": float(g),
                         "read_noise_dn": sigma_dn,
                         "read_noise_e": float(sigma_e),
                         "snr_at_2e_db": float(snr_low)})
    out["gain_vs_noise"] = {"rows": rows,
                            "k_true": float(scfg.full_well_e / scfg.signal_dn)}

    # --- (e) 动态范围 ---
    dr_meas = IQ.dynamic_range_db(scfg_flat.signal_dn,
                                  scfg_flat.read_noise_e / k_true)
    out["dynamic_range"] = {
        "measured_db": float(dr_meas),
        "theory_db": float(IQ.sensor_dr_theory_db(scfg_flat.full_well_e,
                                                  scfg_flat.read_noise_e)),
        "stops": float(dr_meas / 6.02),
        "full_well_e": scfg_flat.full_well_e,
        "read_noise_e": scfg_flat.read_noise_e,
        "bit_depth": scfg_flat.bit_depth,
    }

    # --- (f) 阴影：亮度均匀度与色阴影（LSC 校正准 / 欠 / 过）---
    sc_u = SC.uniform_scene(size[0], size[1])
    rows = []
    for lsc_model, enable, label in (("ideal", False, "无 LSC"),
                                     ("radial2", True, "LSC 欠校正(0.7×)"),
                                     ("ideal", True, "LSC 正确"),
                                     ("radial2_over", True, "LSC 过校正(1.25×)")):
        cam = make_camera(sc_u, 5000.0, scfg, icfg, seed=71,
                          lsc_model=lsc_model, enable_lsc=enable)
        fr = cam.capture(ev=0.0, wb_gains=ideal_gains(5000.0))
        m = IQ.shading_metrics(fr.linear_pre_wb)
        rows.append({"label": label,
                     "luma_uniformity": m["luma_uniformity"],
                     "d_uv_corner_max": m["d_uv_corner_max"]})
    out["shading"] = {"rows": rows}

    return out


# -----------------------------------------------------------------------------
# 12. 时域抖动源：先把噪声地板测出来
# -----------------------------------------------------------------------------
def _stream_jitter(scene, temp, scfg, icfg, ae_cfg, temporal_cfg, seed, n_frames,
                   ev0=0.0, cam_setup=None):
    """跑一条**静止**序列，返回稳态窗内的三级抖动统计。

    窗口取后半段 [n/2, n)：前半段是收敛瞬态，把瞬态算进抖动是最常见的
    自欺欺人（会得到一个跟滤波强度无关的假大数）。
    """
    cam = make_camera(scene, temp, scfg, icfg, seed=seed)
    if cam_setup is not None:
        cam_setup(cam)
    ctl = AEController(copy.deepcopy(ae_cfg), temporal_cfg)
    r = ctl.run_stream(lambda ev: cam.capture(ev=ev, wb_gains=ideal_gains(temp)),
                       ev0=ev0, n_frames=n_frames)
    h = r.history[n_frames // 2:]
    ev_cmd = [x["ev"] for x in h]
    ev_ach = [x["ev_ach"] for x in h]
    met = [float(np.log2(max(x["metric"], 1e-9) / max(x["target"], 1e-9))) for x in h]
    st = MT.jitter_stats(ev_cmd, ev_ach, met)
    st["gain_mean"] = float(np.mean([x["gain"] for x in h]))
    st["et_ms_mean"] = float(np.mean([x["exposure_s"] for x in h])) * 1000.0
    st["ev_hist"] = [x["ev"] for x in r.history]
    st["ach_hist"] = [x["ev_ach"] for x in r.history]
    return st


def _setup_flicker(cam):
    cam.flicker_amplitude = 0.25
    cam.flicker_phase_jitter = 0.35


def exp_temporal_jitter_source(scfg, icfg, size, ae_cfg, temporal_cfg, n_frames=60):
    """12. 抖动从哪来 —— 必须先做，它决定后面所有结论是否有意义。

    整帧测光把光子散粒噪声平均掉了：480x360 下折算约 1e-4 EV，
    比收敛阈值 0.02 EV 低两个数量级。**所以静止场景的 AE 抖动恒等于浮点噪声**，
    "时域滤波把抖动降低 90%" 在这个分辨率下必然是伪结论。

    要让时域滤波有意义，扰动必须作为**显式的物理源**注入：

      S1 光源强度波动   LED 驱动纹波 / 市电电压波动，加在光学之前（乘性）
      S2 闪烁相位漂移   帧时序未锁相于市电，纹波相对读出起点的相位逐帧变
      S3 执行器量化     曝光时间寄存器步进 / 增益档位 —— **确定性，不需要噪声**

    S3 是关键：它说明抖动是**控制结构的产物**，不是噪声的产物。
    而且是分域的：亮场景（EV<0）增益钳在 1.0，抖动由曝光时间量化主导；
    暗场景（EV>0）曝光时间顶在上限，抖动由增益档位主导。

    floor_ratio 必须**逐条件**算：地板随曝光下降而涨，且强依赖测光方式
    （实测 evaluative 4e-5 vs spot 2.6e-3）。用一个全局地板去除所有源会误判。
    """
    bright = SC.natural_scene(*size, backlit=True)
    dim = SC.dim(bright, 1.0 / 40.0)

    specs = [
        ("S0 无注入（噪声地板）", "S0", {}, None),
        ("S1 光源波动 0.5%", "S1", {}, lambda c: setattr(c, "illum_ripple_frac", 0.005)),
        ("S1 光源波动 2%", "S1", {}, lambda c: setattr(c, "illum_ripple_frac", 0.02)),
        ("S2 闪烁相位抖动", "S2", {}, _setup_flicker),
        ("S3 增益量化 1/6 EV", "S3", {"gain_step_ev": 1.0 / 6.0}, None),
        ("S3 增益量化 1/3 EV", "S3", {"gain_step_ev": 1.0 / 3.0}, None),
        ("S3 曝光时间量化 10us", "S3", {"et_step_s": 1e-5}, None),
        ("S4 量化 + 光源波动", "S4", {"gain_step_ev": 1.0 / 6.0},
         lambda c: setattr(c, "illum_ripple_frac", 0.005)),
    ]

    rows = []
    for regime, scene in (("亮场景", bright), ("暗场景 (1/40)", dim)):
        floor = None
        for label, source, overrides, setup in specs:
            s = copy.deepcopy(scfg)
            for k, v in overrides.items():
                setattr(s, k, v)
            st = _stream_jitter(scene, 5000.0, s, icfg, ae_cfg, temporal_cfg,
                                seed=83, n_frames=n_frames, cam_setup=setup)
            j = st["metric"]["std"]
            if floor is None:
                floor = j
            rows.append({
                "regime": regime, "label": label, "source": source,
                "jitter_metric_std": j,
                "jitter_metric_p2p": st["metric"]["p2p"],
                "jitter_ev_cmd_std": st["ev_cmd"]["std"],
                "jitter_ev_ach_std": st["ev_ach"]["std"],
                "gain_mean": st["gain_mean"], "et_ms_mean": st["et_ms_mean"],
                "floor": floor,
                "floor_ratio": (j / floor) if floor > 1e-12 else 0.0,
                "ev_hist": st["ev_hist"], "ach_hist": st["ach_hist"],
            })

    # 显著性的门槛：相对地板 20 倍才认为"这个源值得滤波"（低于它就该如实写"不显著"）
    sig = [r for r in rows if r["source"] != "S0" and r["floor_ratio"] >= 20.0]
    return {"rows": rows, "window": [n_frames // 2, n_frames],
            "n_frames": int(n_frames), "seeds": [83],
            "significant_ratio": 20.0, "n_significant": len(sig)}


# -----------------------------------------------------------------------------
# 13. AE 时域滤波：抖动 vs 响应速度的权衡
# -----------------------------------------------------------------------------
def exp_ae_temporal(scfg, icfg, size, ae_cfg, temporal_cfg, n_frames=72):
    """13. 时域滤波 vs 场景切换检测：抖动与延迟分别由谁决定。

    一条序列同时产出两个指标，省一半成像：
      前半段（切换前）静止 -> 稳态**曝光抖动**（AE 有没有追着扰动跑）
      在第 cut 帧做内容切换 -> 切换后重收敛帧数
    `cut` 帧（真值）由参数直接给，检测器只负责把它**认出来**，两条线可以分开评估。

    抖动源用 S1 光源波动（实验 12 实测 30x 地板、且严格线性），
    它是真实场景里"一直存在"的那种扰动。

    **实测结论与最初的设想相反，这里如实按数据写：**

    1) 时域滤波并没有降低曝光抖动。最好的自适应档只比无滤波好 23%，
       而固定强滤波（alpha<=0.2）**反而失稳**（曝光抖动 0.18 / 1.75 EV）——
       因为控制律是变步长的（d 最大 0.9，过曝补偿还能推到 1.0），慢滤波叠加
       这个大增益把闭环推向振荡。所以"把滤波开大一点"不是免费午餐。
    2) **场景切换检测器才是真正有价值的那一半**：固定 alpha=0.4 时把重收敛
       从 17 帧砍到 6 帧，而抖动分毫不变。它买到的是"切换时立刻松手"，
       不牺牲稳态平滑。
    3) 自适应 alpha 规则下检测器是**冗余的**（大误差本来就触发 fast），
       如实记录，不硬凑它有用。

    同时把二阶极点的**解析振铃周期**写进数据供对照；对不上就写
    "线性化在高频段失效"，不改实测数据去迁就公式。
    """
    bright = SC.natural_scene(*size, backlit=True)
    chart = SC.color_chart(*size)
    # 先让 AE 稳定，再做内容切换；这个帧号同时作为重收敛的真值起点
    cut = n_frames // 2
    ripple = 0.005          # 0.5% 光源波动（实验 12 实测 30x 地板，显著）

    def _run(label, alpha_fast, alpha_slow, adaptive, cut_enable):
        tc = copy.deepcopy(temporal_cfg)
        tc.enable = alpha_slow < 1.0
        tc.adaptive = adaptive
        tc.alpha_fast = alpha_fast
        tc.alpha_slow = alpha_slow
        tc.cut_enable = cut_enable
        cam = make_camera(bright, 5000.0, scfg, icfg, seed=89)
        cam.illum_ripple_frac = ripple

        def on_frame(it):
            if it == cut:
                cam.set_scene(chart)

        ctl = AEController(copy.deepcopy(ae_cfg), tc)
        r = ctl.run_stream(
            lambda ev: cam.capture(ev=ev, wb_gains=ideal_gains(5000.0)),
            ev0=0.0, n_frames=n_frames, on_frame=on_frame, cut_frame=cut)

        h = r.history
        # 稳态抖动：切换**前**的后半段（前 1/4 是收敛瞬态，不能算）
        w0, w1 = n_frames // 4, cut
        steady = h[w0:w1]
        st = MT.jitter_stats(
            [x["ev"] for x in steady],
            [x["ev_ach"] for x in steady],
            [float(np.log2(max(x["metric"], 1e-9) / max(x["target"], 1e-9)))
             for x in steady])
        # 切换后的过冲与方向反转（振铃的直接证据）
        post = h[cut:]
        ev_post = [x["ev"] for x in post]
        tgt = float(np.mean([x["ev"] for x in h[w1 - 5:w1]])) if w1 > w0 else 0.0
        overshoot = max((abs(e - tgt) for e in ev_post), default=0.0)
        rev = 0
        for i in range(2, len(ev_post)):
            a = ev_post[i - 1] - ev_post[i - 2]
            b = ev_post[i] - ev_post[i - 1]
            if a * b < 0:
                rev += 1

        return {
            "label": label,
            "alpha_slow": float(alpha_slow),
            "alpha_fast": float(alpha_fast),
            "adaptive": bool(adaptive),
            "detector": bool(cut_enable),
            "jitter_metric_std": st["metric"]["std"],
            "jitter_metric_p2p": st["metric"]["p2p"],
            "jitter_ev_cmd_std": st["ev_cmd"]["std"],
            "jitter_ev_ach_std": st["ev_ach"]["std"],
            "settle_frames": int(r.settle_frames),
            "overshoot_ev": float(overshoot),
            "reversals_post": int(rev),
            "n_cut_detected": len(r.cut_frames),
            "cut_frames": list(r.cut_frames),
            "ev_hist": [x["ev"] for x in h],
            "err_hist": [abs(x["err_ev"]) for x in h],
        }

    # 关键对照是**成对**的：同一个稳态滤波强度下检测器开/关。
    # 只有成对比较才能把"检测器的贡献"从"滤波强度的贡献"里分离出来。
    #
    # 坑（曾经静默失效过一次）：检测器触发时滤波器切到的是 **alpha_fast**，
    # 所以 alpha_fast 必须真的比稳态用的 alpha_slow 更快，否则"检测器开了"
    # 和"检测器关了"数值完全一样，看起来一切正常但结论是假的。
    # 前两列就是 (alpha_fast, alpha_slow)。
    configs = [
        ("无滤波 (alpha=1)", 1.0, 1.0, False, False),
        ("固定 alpha=0.5（检测器关）", 0.9, 0.5, False, False),
        ("固定 alpha=0.5 + 检测器", 0.9, 0.5, False, True),
        ("固定 alpha=0.4（检测器关）", 0.9, 0.4, False, False),
        ("固定 alpha=0.4 + 检测器", 0.9, 0.4, False, True),
        ("固定 alpha=0.2（检测器关）", 0.9, 0.2, False, False),
        ("固定 alpha=0.1（检测器关）", 0.9, 0.1, False, False),
        ("自适应 alpha（检测器关）", 0.9, 0.2, True, False),
        ("自适应 alpha + 检测器", 0.9, 0.2, True, True),
    ]
    rows = [_run(*c) for c in configs]

    # 帕累托最优：在 (曝光抖动, 重收敛帧数) 平面上没有被别人同时压过的点。
    #
    # 轴选**曝光抖动**而不是画面抖动：画面抖动对 i.i.d. 的光源波动本来就不可约
    # （AE 慢一帧，压不掉当前帧的亮度误差），滤波真正能改的是"AE 移动了多少" ——
    # 也就是会不会出现肉眼可见的亮度抽动。
    #
    # settle_frames = -1 表示**未收敛**，是最差，不是最好 —— 曾经把它当小值
    # 处理，结果"失稳的那一行"被判成了 Pareto 最优。
    def _settle(r):
        return float("inf") if r["settle_frames"] < 0 else float(r["settle_frames"])

    def dominated(row, others):
        for o in others:
            if o is row:
                continue
            better_or_eq = (o["jitter_ev_cmd_std"] <= row["jitter_ev_cmd_std"] * 1.05
                            and _settle(o) <= _settle(row) * 1.05)
            strictly = (o["jitter_ev_cmd_std"] < row["jitter_ev_cmd_std"] * 0.95
                        or _settle(o) < _settle(row) * 0.95)
            if better_or_eq and strictly:
                return True
        return False

    for r in rows:
        r["pareto_optimal"] = not dominated(r, rows)

    return {"rows": rows, "cut_frame": int(cut), "window": [n_frames // 4, cut],
            "n_frames": int(n_frames), "ripple": ripple,
            "ringing_period_analytic": MT.pole_ringing_period(ae_cfg.damping, 0.2)}


# -----------------------------------------------------------------------------
# 14. 场景切换检测：门限标定、误触发率与分离度
# -----------------------------------------------------------------------------
def _cut_probe(scene, temp, scfg, icfg, ae_cfg, temporal_cfg, seed, n_frames,
               ev0=0.0, switch=None, cut=None, ripple=0.0):
    """跑一条序列，返回检测器在**已武装**帧上的信号与检出情况。"""
    tc = copy.deepcopy(temporal_cfg)
    tc.enable = False           # 只测检测器，不叠加滤波，避免两个变量纠缠
    tc.cut_enable = True
    cam = make_camera(scene, temp, scfg, icfg, seed=seed)
    cam.illum_ripple_frac = ripple

    def on_frame(it):
        if switch is not None and it == cut:
            switch(cam)

    ctl = AEController(copy.deepcopy(ae_cfg), tc)
    r = ctl.run_stream(lambda ev: cam.capture(ev=ev, wb_gains=ideal_gains(temp)),
                       ev0=ev0, n_frames=n_frames, on_frame=on_frame)
    armed = [h for h in r.history if h.get("armed")]
    return r, armed


def exp_scene_cut(scfg, icfg, size, ae_cfg, temporal_cfg, seeds=(67, 73), n_frames=36):
    """14. 场景切换检测 —— 门限由数据定，并给出误触发率与分离度。

    这一节要回答三个问题，每个都必须有数字：

    1) **误触发率**。最要命的假阳性是"AE 自己还在收敛"，因为过曝时像素饱和
       会同时破坏曝光归一化和结构信号的标度不变性。所以静止条件里必须包含
       "从 +3 EV 起步让 AE 收敛"这一条，且要求它零触发。
    2) **分离度**。静止时信号的最坏值 vs 真实切换时的最小信号，中间隔了多少倍。
       计划判据：门限 < 0.5 x 切换最小值。
    3) **等亮度色温切换检测不到**（实测 5000K->3000K 等亮度时三个信号全部低于
       门限）。这是这套检测器的**边界**：它测的是亮度与内容变化，不是色度变化。
       如实写出来，不藏。

    检测器只在 AE 连续稳定 `settle_hold` 帧之后才武装（见 temporal.SceneCutDetector），
    所以静止条件里包含 AE 收敛过程，正好检验这条护栏。
    """
    bright = SC.natural_scene(*size, backlit=True)
    chart = SC.color_chart(*size)
    uni = SC.uniform_scene(*size)
    dim = SC.dim(bright, 1.0 / 40.0)
    cut = n_frames // 2

    # --- 静止条件：任何一条触发都算误报 ---
    static_conds = [
        ("natural 亮光", bright, 0.0, 0.0),
        ("natural 暗光(1/40)", dim, 0.0, 0.0),
        ("natural 光源波动 0.5%", bright, 0.0, 0.005),
        ("uniform 亮光", uni, 0.0, 0.0),
        # 最重要的一条：AE 自身收敛（+3EV 起步，重度过曝）
        ("AE 自身收敛 (+3EV 起步)", bright, 3.0, 0.0),
    ]
    static = []
    for label, scene, ev0, rip in static_conds:
        for seed in seeds:
            r, armed = _cut_probe(scene, 5000.0, scfg, icfg, ae_cfg, temporal_cfg,
                                  seed, n_frames, ev0=ev0, ripple=rip)
            mx = {"d_metric_ev": 0.0, "texture_ratio": 0.0, "hist_dist": 0.0}
            for h in armed:
                for k in mx:
                    mx[k] = max(mx[k], float(h["cut_parts"].get(k, 0.0)))
            static.append({"cond": label, "seed": seed,
                           "n_armed": len(armed), "fired": len(r.cut_frames),
                           **{f"max_{k}": v for k, v in mx.items()}})

    # --- 真实切换事件（truth_kind 是分类的真值，用来实测分类准确率）---
    events = [
        ("内容切换 natural→colorchart", "content", bright, lambda c: c.set_scene(chart)),
        ("光照 ×4", "illumination", bright, lambda c: c.set_scene(SC.dim(bright, 4.0))),
        ("光照 ×1/4", "illumination", bright, lambda c: c.set_scene(SC.dim(bright, 0.25))),
        ("色温 5000K→3000K（等亮度）", "illumination",
         bright, lambda c: c.set_scene(bright, 3000.0)),
    ]
    cuts = []
    for label, truth, scene, fn in events:
        for seed in seeds:
            r, _ = _cut_probe(scene, 5000.0, scfg, icfg, ae_cfg, temporal_cfg,
                              seed, n_frames, switch=fn, cut=cut)
            h = r.history[cut]
            lat = (min(r.cut_frames) - cut) if r.cut_frames else -1
            # 分类结果取**首个检出帧**的 kind（没检出就是 none）
            est = r.history[min(r.cut_frames)]["cut_kind"] if r.cut_frames else "none"
            cuts.append({"event": label, "seed": seed, "truth_kind": truth,
                         "est_kind": est,
                         "latency": lat, "detected": bool(r.cut_frames),
                         **{k: float(h["cut_parts"].get(k, 0.0))
                            for k in ("d_metric_ev", "texture_ratio", "hist_dist")}})

    # --- 汇总：分离度与两个比率 ---
    sig_names = ("d_metric_ev", "texture_ratio", "hist_dist")
    thresh = {"d_metric_ev": temporal_cfg.cut_metric_ev,
              "texture_ratio": temporal_cfg.cut_texture_ratio,
              "hist_dist": temporal_cfg.cut_hist_dist}
    static_max = {k: max(r[f"max_{k}"] for r in static) for k in sig_names}
    # 等亮度色温切换检测不到是已知边界，算分离度时排除，单独列出
    det_cuts = [c for c in cuts if "色温" not in c["event"]]
    cut_min = {k: min(c[k] for c in det_cuts) for k in sig_names}

    n_static_runs = len(static)
    n_fired = sum(1 for r in static if r["fired"])
    n_det = sum(1 for c in cuts if c["detected"])
    cct_events = [c for c in cuts if "色温" in c["event"]]

    return {
        "static": static, "cuts": cuts,
        "thresholds": thresh, "static_max": static_max, "cut_min": cut_min,
        "separation": {k: (cut_min[k] / static_max[k]) if static_max[k] > 1e-12 else float("inf")
                       for k in sig_names},
        "margin_vs_thresh": {k: (cut_min[k] / thresh[k]) if thresh[k] > 1e-12 else float("inf")
                             for k in sig_names},
        "false_trigger_rate": (n_fired / n_static_runs) if n_static_runs else 0.0,
        "n_static_runs": n_static_runs, "n_static_fired": n_fired,
        "detect_rate": (n_det / len(cuts)) if cuts else 0.0,
        "n_cut_runs": len(cuts), "n_detected": n_det,
        "cct_equal_luma_detected": sum(1 for c in cct_events if c["detected"]),
        "cct_equal_luma_total": len(cct_events),
        # 分类准确率：分母只算**检出来了的**事件（没检出来的谈不上分对分错）
        "kind_correct": sum(1 for c in cuts if c["detected"] and c["est_kind"] == c["truth_kind"]),
        "kind_total": n_det,
        "kind_accuracy": (sum(1 for c in cuts if c["detected"]
                              and c["est_kind"] == c["truth_kind"]) / n_det) if n_det else 0.0,
        "cut_frame": int(cut), "n_frames": int(n_frames), "seeds": list(seeds),
    }


# -----------------------------------------------------------------------------
# 15. AWB 时域稳定：不引入偏差，且能压住颜色呼吸
# -----------------------------------------------------------------------------
def exp_awb_temporal(scfg, icfg, size, awb_cfg, temporal_cfg,
                     n_frames=60, temp_a=5000.0, temp_b=3000.0):
    """15. AWB 的对数域时域稳定。

    序列：前半段 5000K，第 cut 帧切到 3000K（同场景，只换光源）。
    曝光固定在**第一个光源下 AE 收敛到的 EV**，全程不变 —— 这样比的是
    AWB 本身的时域行为，不掺 AE 的动作。

    三个指标，缺一不可：
      gain_flicker   稳态窗内 mean|d log2(gain)| —— "颜色呼吸"的量化
      angle_err_*    光源角度误差的均值/标准差。**滤波不能让它变差** ——
                     平滑如果不引入偏差，均值应该基本不变；这是"平滑无害"的正面证据
      settle_frames  切光源后重新回到误差阈内所需帧数

    与 AE 侧不同，AWB 在这里**确实**有可压的抖动：估计器逐帧独立，
    没有任何记忆，画面噪声直接体现为增益的逐帧跳动。
    """
    sc = SC.color_chart(*size)
    cut = n_frames // 2
    thr = 3.0           # 稳定判据：光源角度误差回到 3 度以内

    # 固定曝光：取第一个光源下 AE 收敛到的 EV，全程不变
    ae_tmp = AEConfig()
    ae_tmp.ev_min, ae_tmp.ev_max = ev_limits(scfg)
    cam0 = make_camera(sc, temp_a, scfg, icfg, seed=97)
    ev_fix = AEController(ae_tmp).run(
        lambda ev: cam0.capture(ev=ev, wb_gains=ideal_gains(temp_a)), ev0=-1.0).final_ev

    configs = [
        ("无时域滤波", None, False),
        ("alpha=0.9", 0.9, True),
        ("alpha=0.5", 0.5, True),
        ("alpha=0.2", 0.2, True),
    ]
    rows = []
    for label, alpha, enable in configs:
        tcfg = copy.deepcopy(temporal_cfg)
        tcfg.enable = enable
        if alpha is not None:
            tcfg.alpha_slow = alpha
        cam = make_camera(sc, temp_a, scfg, icfg, seed=97)
        stab = AWBStabilizer(copy.deepcopy(awb_cfg), tcfg if enable else None)
        g_hist, err_hist = [], []
        for i in range(n_frames):
            if i == cut:
                cam.set_scene(sc, temp_b)
            fr = cam.capture(ev=ev_fix, wb_gains=ideal_gains(temp_a))
            a = stab.estimate(fr.linear_pre_wb, clipped_ratio=fr.clipped_ratio)
            g_hist.append(np.log2(np.clip(np.asarray(a.gains, dtype=np.float64), 1e-9, None)))
            err_hist.append(illuminant_error_deg(a.illum_rgb, temp_a if i < cut else temp_b))
        g = np.asarray(g_hist)
        e = np.asarray(err_hist)
        steady = slice(n_frames // 4, cut)
        flicker = float(np.mean(np.abs(np.diff(g[steady], axis=0)))) if cut > n_frames // 4 else 0.0
        # 切光源后的重收敛（连续 3 帧回到阈值内）
        settle, run = -1, 0
        for i in range(cut, n_frames):
            run = run + 1 if e[i] < thr else 0
            if run >= 3:
                settle = i - cut + 1
                break
        post = slice(cut + settle if settle > 0 else cut, n_frames)
        rows.append({
            "label": label, "alpha": alpha, "enable": bool(enable),
            "gain_flicker": flicker,
            "angle_err_mean": float(e[steady].mean()),
            "angle_err_std": float(e[steady].std()),
            "angle_err_post_mean": float(e[post].mean()) if e[post].size else float("nan"),
            "settle_frames": settle,
            "err_hist": [float(x) for x in e],
            "n_reset": int(stab.n_reset), "n_hold": int(stab.n_hold),
        })

    base = rows[0]
    return {"rows": rows, "cut_frame": int(cut), "n_frames": int(n_frames),
            "ev_fix": float(ev_fix), "temp_a": temp_a, "temp_b": temp_b,
            "settle_thresh_deg": thr, "steady_window": [n_frames // 4, cut],
            # 无滤波基线的稳态误差，报告里用来对照"滤波有没有引入偏差"
            "baseline_angle_err_mean": base["angle_err_mean"]}


# -----------------------------------------------------------------------------
# 16. C++ 统计通路：性能对照与等价性
# -----------------------------------------------------------------------------
def _native_frames(scfg, icfg, size, ae_cfg, temp=5000.0):
    """取一组**在各自 AE 收敛工作点上**的帧（不是随机曝光）。"""
    scenes = [("color_chart", SC.color_chart(*size)),
              ("natural", SC.natural_scene(*size, backlit=True)),
              ("uniform", SC.uniform_scene(*size))]
    out = []
    for nm, sc in scenes:
        cam = make_camera(sc, temp, scfg, icfg, seed=97)
        ev = AEController(copy.deepcopy(ae_cfg)).run(
            lambda e: cam.capture(ev=e, wb_gains=ideal_gains(temp)), ev0=-1.0).final_ev
        out.append((nm, cam.capture(ev=ev, wb_gains=ideal_gains(temp))))
    return out


def exp_native_port(scfg, icfg, size, ae_cfg, awb_cfg, repeats=100, warmup=20):
    """16. C++ 统计重写：2×2 因子对照（算法 × 实现）+ 定点第 5 格 + 等价性。

    对照设计回答的是**两个不同的问题**，混在一起就会得出"用 C 更快"这种
    似是而非的结论：
        py_naive -> py_opt   算法改进值多少（同一门语言）
        py_opt   -> c_opt    实现/语言值多少（同一套算法）
        c_f64 作为"朴素 C"的反向检验：**朴素 C 未必赢 numpy**

    **诚实条款**：如果 Δ实现 < 1（C 没赢），如实写出来。数值/访存类的小核在
    numpy 的 SIMD 加缓存友好实现面前输掉很正常，那本身就是有信息量的结果。
    """
    from .native import backend, bench, loader, pyopt

    st = loader.status()
    if not st.get("available"):
        return {"available": False, "reason": st.get("reason", "共享库不可用")}

    frames = _native_frames(scfg, icfg, size, ae_cfg)
    fr0 = frames[0][1]
    linear0 = np.ascontiguousarray(fr0.linear_pre_wb, dtype=np.float32)
    rcfg = AWBConfig(method="fusion")

    py_opt_fn = pyopt.compute_statistics_opt
    c_f64 = backend.awb_stats_fn("f64")
    c_f32 = backend.awb_stats_fn("f32")
    c_q16 = backend.awb_stats_fn("q16")

    timing = {
        "awb_py_naive": bench.measure(lambda: compute_statistics(linear0, rcfg), warmup, repeats),
        "awb_py_opt": bench.measure(lambda: py_opt_fn(linear0, rcfg), warmup, repeats),
        "awb_c_f64": bench.measure(lambda: c_f64(linear0, rcfg), warmup, repeats),
        "awb_c_f32": bench.measure(lambda: c_f32(linear0, rcfg), warmup, repeats),
        "awb_c_q16": bench.measure(lambda: c_q16(linear0, rcfg), warmup, repeats),
    }
    # AE 侧测两个模式：
    #   average   —— **没有算法差异**（都是一次求和），所以它度量的是"纯语言差异"
    #   evaluative—— C 侧有真改进（分区循环 + 缓存），但两边都没什么可省
    # 这个对照比"故意写一份慢 C"更有说服力：它是真实的、不是构造出来的。
    a_cfg = AEConfig(metering="evaluative")
    c_meter = backend.metering_fn("evaluative", precision="f32")
    timing["ae_py_naive"] = bench.measure(lambda: metering_metric(fr0, a_cfg), warmup, repeats)
    timing["ae_c_f32"] = bench.measure(lambda: c_meter(fr0, a_cfg), warmup, repeats)
    avg_cfg = AEConfig(metering="average")
    c_avg = backend.metering_fn("average", precision="f32")
    timing["ae_avg_py"] = bench.measure(lambda: metering_metric(fr0, avg_cfg), warmup, repeats)
    timing["ae_avg_c"] = bench.measure(lambda: c_avg(fr0, avg_cfg), warmup, repeats)

    floor = bench.null_call_floor()

    equiv = []
    for nm, fr in frames:
        lin = np.ascontiguousarray(fr.linear_pre_wb, dtype=np.float32)
        a = compute_statistics(lin, rcfg)
        b = c_f64(lin, rcfg)
        rel = {}
        for k in ("gray_world", "white_patch", "gray_edge", "shades_of_gray"):
            x = np.asarray(a.estimators[k], float)
            y = np.asarray(b.estimators[k], float)
            rel[k] = float(np.max(np.abs(x - y) / np.maximum(np.abs(x), 1e-12)))
        equiv.append({
            "scene": nm,
            "n_valid_equal": bool(a.n_valid == b.n_valid),
            "sat_mean_rel": float(abs(a.sat_mean - b.sat_mean) / max(abs(a.sat_mean), 1e-12)),
            "frac_ge_abs": float(abs(a.frac_ge - b.frac_ge)),
            **{f"{k}_rel": v for k, v in rel.items()},
        })

    t = {k: v["median_us"] for k, v in timing.items()}
    safe = lambda x: max(x, 1e-9)                                       # noqa: E731
    decomp = {
        "algo_gain_awb": t["awb_py_naive"] / safe(t["awb_py_opt"]),
        "impl_gain_awb": t["awb_py_opt"] / safe(t["awb_c_f32"]),
        "total_gain_awb": t["awb_py_naive"] / safe(t["awb_c_f32"]),
        "impl_gain_ae_evaluative": t["ae_py_naive"] / safe(t["ae_c_f32"]),
        # ★ 最能说明问题的一行：在**没有算法差异**的模式上换语言只值多少。
        # 把它和上面的 9.2 倍并排看，就能判断"加速来自语言还是来自算法"。
        "language_only_gain": t["ae_avg_py"] / safe(t["ae_avg_c"]),
    }
    npx = int(linear0.shape[0] * linear0.shape[1])
    budget = {
        "frame_budget_pct": bench.frame_budget_pct(t["awb_c_f32"]),
        "extrap_1080p_us": bench.extrapolate_us(t["awb_c_f32"], npx, 1920 * 1080),
        "extrap_4k_us": bench.extrapolate_us(t["awb_c_f32"], npx, 3840 * 2160),
    }
    return {"available": True, "env": bench.environment(), "timing": timing,
            "null_call": floor, "null_call_us": floor["median_us"],
            "equivalence": equiv, "decomposition": decomp, "budget": budget,
            "size": list(size), "repeats": int(repeats), "warmup": int(warmup)}


# -----------------------------------------------------------------------------
# 17. 定点化误差预算
# -----------------------------------------------------------------------------
def exp_fixed_point(scfg, icfg, size, ae_cfg, awb_cfg, repeats=50):
    """17. 定点化误差：误差分解 + 位宽/bin 扫描 + AE/AWB 端到端。

    顺序统计量的误差用**三条路**分解，这样每一项都能归因到单一变量：
        float64 + partition   numpy 现状（金标准）
        float64 + 直方图       隔离"直方图估计器"本身的误差（与 C 无关，纯 Python 可算）
        Q16 + 直方图          隔离"纯定点化"的误差（C）
    于是 |p99_q16 − p99_numpy| ≤ |p99_hist64 − p99_numpy| + |p99_q16 − p99_hist64|。
    """
    from .native import backend, loader

    st = loader.status()
    if not st.get("available"):
        return {"available": False, "reason": st.get("reason", "共享库不可用")}

    frames = _native_frames(scfg, icfg, size, ae_cfg)
    rcfg = AWBConfig(method="fusion")
    MODES = ("average", "center", "spot", "evaluative", "highlight_priority")

    # --- (a) AE 定点误差：逐模式绝对差 + 折算 EV ---
    ae_rows = []
    for nm, fr in frames:
        for mode in MODES:
            cfg = AEConfig(metering=mode)
            ref = metering_metric(fr, cfg)["metric"]
            fp = float(fr.clipped_ratio)
            q = backend.metering_fn(mode, precision="q16")(fr, cfg)["metric"]
            d = abs(ref - q)
            ae_rows.append({
                "scene": nm, "mode": mode, "ref": ref, "q16": q, "abs_diff": d,
                "ev": d / max(ref, 1e-9) / np.log(2.0),
                # 分位数走直方图，上界是一个 bin 宽；均值类上界 1.8e-5
                "bound": 9.77e-4 if mode == "highlight_priority" else 1.8e-5,
                "clip_ratio": fp,
            })

    # --- (b) AWB 定点误差：逐通道 + 端到端角度 ---
    awb_rows = []
    q_fn = backend.awb_stats_fn("q16")
    for nm, fr in frames:
        lin = np.ascontiguousarray(fr.linear_pre_wb, dtype=np.float32)
        a = compute_statistics(lin, rcfg)
        b = q_fn(lin, rcfg)
        r1 = AWBEstimator(AWBConfig(method="fusion")).estimate(
            fr.linear_pre_wb, fr.clipped_ratio)
        r2 = AWBEstimator(AWBConfig(method="fusion"), stats_fn=q_fn).estimate(
            fr.linear_pre_wb, fr.clipped_ratio)
        awb_rows.append({
            "scene": nm,
            "n_valid_ref": a.n_valid, "n_valid_q16": b.n_valid,
            "gray_world_abs": float(np.max(np.abs(np.asarray(a.estimators["gray_world"], float)
                                                  - np.asarray(b.estimators["gray_world"], float)))),
            "white_patch_abs": float(np.max(np.abs(np.asarray(a.estimators["white_patch"], float)
                                                   - np.asarray(b.estimators["white_patch"], float)))),
            "gains_rel": float(np.max(np.abs(np.asarray(r1.gains, float)
                                             - np.asarray(r2.gains, float))
                                      / np.maximum(np.abs(r1.gains), 1e-12))),
            "angle_diff_deg": float(abs(illuminant_error_deg(r2.illum_rgb, 5000.0)
                                        - illuminant_error_deg(r1.illum_rgb, 5000.0))),
        })

    # --- (c) 位宽扫描（AE average 与 highlight 各一条）---
    by_bits = []
    fr0 = frames[0][1]
    for bits in (8, 10, 12, 14, 16):
        row = {"bits": bits}
        for mode in ("average", "highlight_priority"):
            cfg = AEConfig(metering=mode)
            ref = metering_metric(fr0, cfg)["metric"]
            got = float(backend.metering_fn(mode, precision="q16",
                                            bit_depth=bits)(fr0, cfg)["metric"])
            row[f"{mode}_abs"] = abs(ref - got)
        by_bits.append(row)

    # --- (d) bin 数扫描（分位数）---
    by_bins = []
    for bins in (64, 256, 1024, 4096):
        cfg = AEConfig(metering="highlight_priority")
        ref = metering_metric(fr0, cfg)["metric"]
        got = float(backend.metering_fn("highlight_priority", precision="q16",
                                        hist_bins=bins)(fr0, cfg)["metric"])
        by_bins.append({"bins": bins, "abs": abs(ref - got),
                        "bin_width": 1.0 / bins})

    return {"available": True, "ae_rows": ae_rows, "awb_rows": awb_rows,
            "by_bits": by_bits, "by_bins": by_bins, "size": list(size),
            "ref_numpy": np.__version__,
            # 定点化的边界，报告里明写
            "scope": "只定点化逐像素统计通路；融合权重、对数域几何平均、CCT 约束、"
                     "AE 控制律仍在 double"}
