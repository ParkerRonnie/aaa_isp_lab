# -*- coding: utf-8 -*-
"""冒烟测试：把整条链路跑一遍，任何一步报错都能立刻定位。"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np

from aaa_isp_lab.config import SensorConfig, ISPConfig, AEConfig, AWBConfig, AFConfig
from aaa_isp_lab.sim import scene as S
from aaa_isp_lab.sim.camera import SimCamera
from aaa_isp_lab.aaa.ae import AEController, metering_metric
from aaa_isp_lab.aaa.awb import AWBEstimator, ideal_gains, illuminant_error_deg
from aaa_isp_lab.aaa import af as AF
from aaa_isp_lab.eval import metrics as MT

W, H = 320, 240
scfg = SensorConfig(width=W, height=H)
icfg = ISPConfig()

t0 = time.time()
scenes = S.build_all(W, H)
print("[ok] 场景构建", {k: v.reflectance.shape for k, v in scenes.items()})

# --- 相机 + ISP ---
cam = SimCamera(scenes["color_chart"], 5000.0, scfg, icfg, seed=1, true_focus=0.45)
fr = cam.capture(ev=0.0)
print("[ok] 成像", fr.srgb_u8.shape, fr.srgb_u8.dtype,
      "clip=%.3f" % fr.clipped_ratio, "et=%.5f gain=%.2f" % (fr.exposure_s, fr.gain))

# --- AE ---
for mode in ("average", "center", "spot", "evaluative", "highlight_priority"):
    ae = AEController(AEConfig(metering=mode))
    r = ae.run(lambda ev: cam.capture(ev=ev), ev0=+2.5)
    print(f"[ok] AE {mode:18s} iters={r.iters} ev={r.final_ev:+.3f} "
          f"metric={r.final_metric:.4f} conv={r.converged} rev={r.reversals}")

ae = AEController(AEConfig())
lut = ae.calibrate(lambda ev: cam.capture(ev=ev, add_noise=False), n=9)
print("[ok] AE 标定表", "evs", np.round(lut["evs"], 2)[:3], "...",
      "初始EV=", round(ae.lut_initial_ev(lut, 0.18), 3))

# --- AWB ---
est = AWBEstimator(AWBConfig())
r = est.estimate(fr.linear_pre_wb, fr.clipped_ratio)
print("[ok] AWB", r.method, "gains", np.round(r.gains, 3),
      "cct=%.0f" % r.cct, "err=%.2f deg" % illuminant_error_deg(r.illum_rgb, 5000.0))

for m in ("gray_world", "white_patch", "gray_edge", "shades_of_gray", "fusion"):
    e = AWBEstimator(AWBConfig(method=m)).estimate(fr.linear_pre_wb, fr.clipped_ratio)
    print(f"     {m:16s} err={illuminant_error_deg(e.illum_rgb, 5000.0):6.2f} deg "
          f"cct={e.cct:7.0f}")

# --- 色彩指标 ---
ideal = MT.ideal_linear(scenes["color_chart"], 5000.0)
from aaa_isp_lab.isp.modules import apply_wb
d_before = MT.neutral_chroma(fr.linear_ccm, scenes["color_chart"].neutral_mask)
wb = apply_wb(fr.linear_pre_wb, ideal_gains(5000.0))
d_after = MT.neutral_chroma(wb, scenes["color_chart"].neutral_mask)
print(f"[ok] 中性区残余色度: 无WB={d_before:.2f} 理想WB后={d_after:.2f}")
print("     ΔE00(理想WB) =", round(float(np.mean(MT.patch_delta_e(
    wb, ideal, scenes["color_chart"].patch_masks))), 3))

# --- CCM 标定 ---
from aaa_isp_lab.isp.modules import solve_ccm
masks = scenes["color_chart"].patch_masks
src = np.stack([wb[m].mean(axis=0) for m in masks])
dst = np.stack([ideal[m].mean(axis=0) for m in masks])
ccm = solve_ccm(src, dst)
print("[ok] CCM 标定\n", np.round(ccm, 4))
from aaa_isp_lab.isp.modules import apply_ccm
print("     ΔE00(标定后) =", round(float(np.mean(MT.patch_delta_e(
    apply_ccm(wb, ccm), ideal, masks))), 3))

# --- AF ---
fcam = SimCamera(scenes["focus_target"], 5000.0, scfg, icfg, seed=2, true_focus=0.35)
for measure in AF.MEASURES:
    vs = [AF.frame_measure(fcam.capture(ev=0.0, focus_pos=p), AFConfig(measure=measure))
          for p in np.linspace(0, 1, 21)]
    print(f"[ok] AF {measure:14s} peak@{np.linspace(0,1,21)[int(np.argmax(vs))]:.2f} "
          f"(true=0.35) val={max(vs):.4g}")

for strat in ("sweep", "hill_climb", "coarse_to_fine", "golden_section"):
    c = AF.AFController(AFConfig(strategy=strat))
    r = c.run(lambda p: fcam.capture(ev=0.0, focus_pos=p))
    print(f"[ok] AF-seq {strat:16s} frames={r.frames:3d} best={r.best_pos:.3f} "
          f"err={abs(r.best_pos - 0.35):.3f}")

print("[done] 用时 %.1fs" % (time.time() - t0))
