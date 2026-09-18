# -*- coding: utf-8 -*-
"""命令行入口。

    python -m aaa_isp_lab            # 跑完全部实验并生成报告
    aaa-isp-lab --fast               # 安装后可直接用命令
    aaa-isp-lab --out docs           # 指定输出目录

注意 `run()` 与 `main()` 的分工：
`run()` 返回产物路径字典，给程序化调用用；
`main()` 只返回退出码 —— 因为 console script 的包装器会执行
`sys.exit(main())`，如果 main 返回的是字典，Python 会把它打印出来并以
退出码 1 结束（命令明明跑成功了，CI 却红）。这个坑只有装成命令行工具才会暴露。
"""
import argparse
import json
import os
import time

import numpy as np

from .config import SensorConfig, ISPConfig, AEConfig, AWBConfig, AFConfig
from .sim.camera import ev_limits
from .eval import report as RP
from . import experiments as EX


def run(argv=None) -> dict:
    """跑完全部实验并生成报告，返回产物路径。"""
    ap = argparse.ArgumentParser(
        prog="aaa-isp-lab",
        description="3A（AE/AWB/AF）算法与 ISP 画质调优实验平台")
    ap.add_argument("--fast", action="store_true", help="快速模式（小图、少重复）")
    ap.add_argument("--out", default="out", help="输出目录（默认 out/）")
    args = ap.parse_args(argv)

    size = (256, 192) if args.fast else (480, 360)
    repeats = 2 if args.fast else 5
    outdir = args.out
    os.makedirs(outdir, exist_ok=True)

    scfg = SensorConfig(width=size[0], height=size[1])
    icfg = ISPConfig()
    ae_cfg = AEConfig()
    awb_cfg = AWBConfig()
    af_cfg = AFConfig()
    lo, hi = ev_limits(scfg)
    ae_cfg.ev_min, ae_cfg.ev_max = lo, hi

    t0 = time.time()
    print(f"仿真分辨率 {size[0]}×{size[1]}，EV 可用范围 [{lo:+.2f}, {hi:+.2f}]")

    steps = [
        ("AE 测光", lambda: EX.exp_ae_metering(scfg, icfg, size, ae_cfg)),
        ("AE 收敛", lambda: EX.exp_ae_convergence(scfg, icfg, size, ae_cfg)),
        ("AE 标定曲线", lambda: EX.exp_ae_curve(scfg, icfg, size, ae_cfg)),
        ("AE 曝光策略/抗闪烁", lambda: EX.exp_ae_policy(scfg, icfg, size, ae_cfg)),
        ("AWB 对比", lambda: EX.exp_awb(scfg, icfg, size, awb_cfg)),
        ("AWB 色温先验", lambda: EX.exp_awb_constraint(scfg, icfg, size, awb_cfg)),
        ("CCM 与耦合", lambda: EX.exp_ccm(scfg, icfg, size, awb_cfg)),
        ("AF 评价函数", lambda: EX.exp_af_curves(scfg, icfg, af_cfg, size, repeats)),
        ("AF 搜索策略", lambda: EX.exp_af_search(scfg, icfg, af_cfg, size)),
        ("3A 耦合", lambda: EX.exp_coupling(scfg, icfg, ae_cfg, awb_cfg, af_cfg, size)),
    ]
    R = {}
    for name, fn in steps:
        t = time.time()
        R[name] = fn()
        print(f"  [ok] {name:20s} {time.time() - t:5.1f}s  "
              f"(累计成像 {EX.N_CAPTURES[0]})")

    res = {
        "meta": {"time": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "size": list(size), "seed": 2026,
                 "n_captures": EX.N_CAPTURES[0]},
        "isp": EX.exp_isp(scfg, icfg, size),
        "ae_metering": R["AE 测光"],
        "ae_convergence": R["AE 收敛"],
        "ae_curve": R["AE 标定曲线"],
        "ae_policy": R["AE 曝光策略/抗闪烁"],
        "ae_flicker": EX.exp_ae_flicker(scfg, icfg, size, ae_cfg),
        "awb": R["AWB 对比"],
        "awb_constraint": R["AWB 色温先验"],
        "ccm": R["CCM 与耦合"],
        "af_curves": R["AF 评价函数"],
        "af_search": R["AF 搜索策略"],
        "coupling": R["3A 耦合"],
    }

    print("生成报告 ...")
    paths = RP.build_report(res, outdir)
    with open(os.path.join(outdir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(_sanitize(res), f, ensure_ascii=False, indent=2, default=str)

    print("\n完成，用时 %.1fs，成像 %d 次" % (time.time() - t0, EX.N_CAPTURES[0]))
    print(f"  {paths['md']}")
    print(f"  {paths['html']}")
    return paths


def main(argv=None) -> int:
    """命令行入口：只返回退出码。"""
    run(argv)
    return 0


def _sanitize(obj, depth=0):
    if depth > 6:
        return "..."
    if isinstance(obj, dict):
        return {str(k): _sanitize(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v, depth + 1) for v in obj]
    if isinstance(obj, np.ndarray):
        return f"<ndarray {obj.shape}>"
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)


if __name__ == "__main__":
    main()
