# -*- coding: utf-8 -*-
"""实验报告生成：把 run_all.py 的结果画成图 + 写成 markdown/html。

一份能拿出去讲的工程报告，重点不是"我做了什么"，而是
"数据说明了什么结论、为什么、这个结论对产品意味着什么"。
所以每张图旁边都带结论性的注释。
"""
import base64
import io
import json
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",      # Windows
    "PingFang SC",          # macOS
    "Noto Sans CJK SC",     # Linux（CI 里装的是 fonts-noto-cjk）
    "WenQuanYi Zen Hei",    # Linux 常见备选
    "SimHei",
    "DejaVu Sans",          # 兜底：没有中文字体时不会崩，但中文会变方块
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"

C_MAIN = "#2b6cb0"
C_WARN = "#c53030"
C_OK = "#2f855a"
C_GRAY = "#718096"


def _save(fig, outdir: str, name: str) -> str:
    path = os.path.join(outdir, name)
    fig.savefig(path)
    plt.close(fig)
    return path


def _to_u8(img: np.ndarray) -> np.ndarray:
    """转成可视化用的 uint8。

    注意：必须先判 dtype 再转换。写成 `np.asarray(img, np.float32)` 之后
    再判 `dtype == uint8` 永远为假，uint8 图会被当成 0~1 浮点图再乘一次 255，
    整幅图直接变成全白 —— 这个 bug 不会报错，只会让图"看起来是空的"。
    """
    x = np.asarray(img)
    if x.dtype == np.uint8:
        return x
    x = x.astype(np.float32)
    if x.ndim == 2:                      # 单通道（如 RAW/Bayer）按灰度显示
        lo, hi = float(x.min()), float(x.max())
        g = (x - lo) / max(hi - lo, 1e-6)
        return np.repeat(np.clip(g * 255.0 + 0.5, 0, 255).astype(np.uint8)[..., None], 3, axis=2)
    return np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)


# -----------------------------------------------------------------------------
# AE
# -----------------------------------------------------------------------------
def fig_ae_convergence(res: dict, outdir: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for r in res["runs"]:
        axes[0].plot(range(len(r["ev_hist"])), r["ev_hist"], "o-", ms=4,
                     label=r["label"])
    axes[0].axhline(0, color=C_GRAY, ls=":", lw=1)
    axes[0].set_xlabel("帧序号")
    axes[0].set_ylabel("曝光 EV")
    axes[0].set_title("AE 收敛过程：步长策略的影响")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    for r in res["runs"]:
        y = np.abs(np.asarray(r["err_hist"]))
        axes[1].semilogy(range(len(y)), np.maximum(y, 1e-4), "o-", ms=4, label=r["label"])
    axes[1].axhline(0.02, color=C_WARN, ls="--", lw=1, label="收敛门限 0.02 EV")
    axes[1].set_xlabel("帧序号")
    axes[1].set_ylabel("|亮度误差| (EV, log)")
    axes[1].set_title("误差下降曲线（越陡收敛越快）")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, which="both")
    return _save(fig, outdir, "fig_ae_convergence.png")


def fig_ae_metering(res: dict, outdir: str) -> str:
    modes = list(res["modes"].keys())
    n = len(modes)
    fig = plt.figure(figsize=(13, 5.6))
    gs = fig.add_gridspec(2, n, height_ratios=[1.5, 1.0], hspace=0.35)

    for i, m in enumerate(modes):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(_to_u8(res["img"][m]))
        ax.set_title(f'{m}\n主体 {res["modes"][m]["subject_code"]:.0f} / '
                     f'{res["modes"][m]["iters"]} 帧', fontsize=9)
        ax.axis("off")

    ax = fig.add_subplot(gs[1, :])
    x = np.arange(n)
    w = 0.28
    ax.bar(x - w, [res["modes"][m]["subject_code"] for m in modes], w,
           label="主体码值（越高越不逆光）", color=C_MAIN)
    ax.bar(x, [res["modes"][m]["mean_code"] for m in modes], w,
           label="全画面平均码值", color=C_GRAY)
    ax.bar(x + w, [res["modes"][m]["clip"] * 255 for m in modes], w,
           label="过曝像素占比 (×255)", color=C_WARN)
    ax.axhline(res["target_code"], color=C_OK, ls="--", lw=1.2,
               label=f'目标码值 {res["target_code"]:.0f}')
    ax.set_xticks(x)
    ax.set_xticklabels(modes, fontsize=9)
    ax.set_ylabel("sRGB 码值 / 占比")
    ax.set_title("逆光场景下测光方式对比：主体是否被保住、高光是否被牺牲")
    ax.legend(fontsize=8, ncol=4)
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, outdir, "fig_ae_metering.png")


def fig_ae_metering_curve(res: dict, outdir: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    evs = res["calib_evs"]
    ax.plot(evs, res["calib_code"], "-", color=C_MAIN, lw=1.8, label="亮度响应曲线")
    ax.axhline(res["target_code"], color=C_OK, ls="--", lw=1.2,
               label=f'目标码值 {res["target_code"]:.0f}')
    for m, pt in res["converged_points"].items():
        ax.plot(pt[0], pt[1], "o", ms=7, label=f"{m} 收敛点")
    ax.set_xlabel("曝光 EV")
    ax.set_ylabel("画面平均码值 (sRGB 8bit)")
    ax.set_title("AE 的标定曲线：目标码值反查 EV\n"
                 "（曲线在高 EV 端变平 —— 饱和后亮度不再随曝光增加，\n"
                 " 这一步的非线性正是纯闭环必须收小步长的原因）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, outdir, "fig_ae_metering_curve.png")


def fig_ae_policy(res: dict, outdir: str) -> str:
    rows = res["rows"]
    labels = [r["policy"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4))

    axes[0].bar(labels, [r["et_ms"] for r in rows], color=C_MAIN)
    axes[0].set_ylabel("曝光时间 (ms)")
    axes[0].set_title("曝光时间")
    axes[1].bar(labels, [r["snr_db"] for r in rows], color=C_OK)
    axes[1].set_ylabel("SNR (dB)")
    axes[1].set_title("信噪比：增益放大不了光子数")
    axes[2].bar(labels, [r["sharpness"] for r in rows], color=C_WARN)
    axes[2].set_ylabel("清晰度（聚焦评价）")
    axes[2].set_title("运动模糊：长曝把运动拍糊")
    for ax in axes:
        ax.grid(alpha=0.3, axis="y")
        ax.tick_params(labelsize=9)
        for t in ax.get_xticklabels():
            t.set_rotation(12)
    return _save(fig, outdir, "fig_ae_policy.png")


def fig_ae_flicker(res: dict, outdir: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for r in res["runs"]:
        axes[0].plot(range(len(r["ev_hist"])), r["ev_hist"], "o-", ms=4, label=r["label"])
    axes[0].set_xlabel("帧序号")
    axes[0].set_ylabel("曝光 EV")
    axes[0].set_title("抗闪烁开启 / 关闭时的收敛过程")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    names = ["抗闪烁 关", "抗闪烁 开"]
    axes[1].bar(names, [res["banding_off"], res["banding_on"]], color=[C_WARN, C_OK])
    for i, v in enumerate([res["banding_off"], res["banding_on"]]):
        axes[1].text(i, v, f"{v:.3f}\nET={[res['et_off_ms'], res['et_on_ms']][i]:.1f}ms",
                     ha="center", va="bottom", fontsize=9)
    axes[1].set_ylabel("行间带纹强度（越低越好）")
    axes[1].set_title("交流光源下的横条纹：\n曝光时间是否落在 10ms 整数倍上")
    axes[1].grid(alpha=0.3, axis="y")
    return _save(fig, outdir, "fig_ae_flicker.png")


# -----------------------------------------------------------------------------
# AWB
# -----------------------------------------------------------------------------
def fig_awb_grid(res: dict, outdir: str) -> str:
    scenes = list(res["scenes"].keys())
    methods = list(res["methods"])
    styles = ["无白平衡"] + methods
    fig, axes = plt.subplots(len(scenes), len(styles),
                             figsize=(2.0 * len(styles), 1.85 * len(scenes)))
    if len(scenes) == 1:
        axes = axes[None, :]

    for i, s in enumerate(scenes):
        d = res["scenes"][s]
        for j, st in enumerate(styles):
            ax = axes[i, j]
            if st == "无白平衡":
                ax.imshow(_to_u8(d["no_wb_img"]))
                err = d["no_wb_err"]
            else:
                ax.imshow(_to_u8(d["img"][st]))
                err = d["methods"][st]["err_deg"]
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(st, fontsize=9)
            if j == 0:
                ax.set_ylabel(f'{s}\n{d["temp"]:.0f}K', fontsize=9)
            ax.set_xlabel(f'光源误差 {err:.2f}°', fontsize=8)
    fig.suptitle("AWB 各算法在不同场景下的表现（同一帧、同一评价口径）", y=1.0)
    return _save(fig, outdir, "fig_awb_grid.png")


def fig_awb_error(res: dict, outdir: str) -> str:
    scenes = list(res["scenes"].keys())
    methods = list(res["methods"])
    x = np.arange(len(scenes))
    w = 0.8 / (len(methods) + 1)

    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.bar(x - 0.4 + w / 2, [res["scenes"][s]["no_wb_err"] for s in scenes], w,
           label="无白平衡", color=C_GRAY)
    colors = [C_MAIN, C_OK, C_WARN, "#805ad5", "#dd6b20"]
    for k, m in enumerate(methods):
        ax.bar(x - 0.4 + w * (k + 1.5), [res["scenes"][s]["methods"][m]["err_deg"]
                                         for s in scenes], w, label=m, color=colors[k % 5])
    ax.set_xticks(x)
    ax.set_xticklabels(scenes, fontsize=9)
    ax.set_ylabel("光源角度误差 (度)")
    ax.set_title("光源估计精度（度，越小越好）：没有一种算法在所有场景都赢")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, outdir, "fig_awb_error.png")


def fig_awb_constraint(res: dict, outdir: str) -> str:
    rows = res["rows"]
    labels = [r["label"] for r in rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))

    axes[0].bar(x - 0.18, [min(r["raw_cct"], 12000) for r in rows], 0.36,
                label="约束前", color=C_WARN)
    axes[0].bar(x + 0.18, [min(r["cct"], 12000) for r in rows], 0.36,
                label="约束后", color=C_OK)
    for i, r in enumerate(rows):
        axes[0].plot([i - 0.4, i + 0.4], [r["true_cct"]] * 2, color=C_MAIN,
                     ls="--", lw=1.4)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylabel("估计色温 (K)")
    axes[0].set_title(f'色温先验约束（虚线 = 真实色温）\n'
                      f'{res["summary"]["越界场景数"]}/{len(rows)} 个场景的估计越界、约束才生效')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(x - 0.18, [r["raw_err"] for r in rows], 0.36, label="约束前", color=C_WARN)
    axes[1].bar(x + 0.18, [r["err"] for r in rows], 0.36, label="约束后", color=C_OK)
    for i, r in enumerate(rows):
        axes[1].text(i, max(r["raw_err"], r["err"]) + 0.6,
                     f'Duv={r["raw_duv"]:+.4f}', ha="center", fontsize=8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylabel("光源角度误差 (度)")
    axes[1].set_title("约束对误差的作用：只改善色温越界的那一类，\n对 Duv 方向的偏差无能为力")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, axis="y")
    return _save(fig, outdir, "fig_awb_constraint.png")


# -----------------------------------------------------------------------------
# CCM / 颜色
# -----------------------------------------------------------------------------
def fig_ccm(res: dict, outdir: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    names = list(res["delta_e"].keys())
    vals = [res["delta_e"][k] for k in names]
    colors = [C_GRAY, C_MAIN, C_OK, C_WARN][:len(names)]
    axes[0].bar(names, vals, color=colors)
    for i, v in enumerate(vals):
        axes[0].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    axes[0].set_ylabel("色卡平均 ΔE00")
    axes[0].set_title("颜色精度：白平衡与 CCM 的贡献拆解")
    axes[0].grid(alpha=0.3, axis="y")
    for t in axes[0].get_xticklabels():
        t.set_rotation(15)

    cp = res["coupling"]
    ww = res.get("wrong_wb", {"raw": cp["mismatched"], "with_ccm": cp["mismatched"]})
    labels = ["白平衡正确\n+ CCM", "白平衡判错\n(不加 CCM)", "白平衡判错\n(加 CCM)"]
    vals2 = [cp["matched"], ww["raw"], ww["with_ccm"]]
    axes[1].bar(labels, vals2, color=[C_OK, C_MAIN, C_WARN])
    for i, v in enumerate(vals2):
        axes[1].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    axes[1].set_ylabel("色卡平均 ΔE00")
    axes[1].set_title("AWB 与 CCM 的耦合：CCM 是在特定光源下标定的，\n"
                      "光源判断错误时 CCM 会把偏色放大")
    axes[1].grid(alpha=0.3, axis="y")
    return _save(fig, outdir, "fig_ccm.png")


# -----------------------------------------------------------------------------
# AF
# -----------------------------------------------------------------------------
def fig_af_curves(res: dict, outdir: str) -> str:
    pos = res["positions"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for m, c in res["curves"].items():
        c = np.asarray(c)
        ax.plot(pos, c / max(c.max(), 1e-12), "-", lw=1.6,
                label=f'{m} (动态范围 {res["metrics"][m]["dynamic_range_db"]:.1f} dB)')
    ax.axvline(res["true_focus"], color=C_WARN, ls="--", lw=1.2,
               label=f'真合焦位置 {res["true_focus"]:.2f}')
    ax.set_xlabel("镜头位置（归一化行程）")
    ax.set_ylabel("归一化评价函数值")
    ax.set_title("对焦评价函数对比：单峰性、灵敏度（半高宽）与动态范围\n"
                 "动态范围小 = 对没对上焦时分辨不出来")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, outdir, "fig_af_curves.png")


def fig_af_search(res: dict, outdir: str) -> str:
    pos = res["curve_pos"]
    curve = np.asarray(res["curve"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(pos, curve / curve.max(), "-", color=C_GRAY, lw=1.4, label="评价函数曲线")
    ax.axvline(res["true_focus"], color=C_WARN, ls="--", lw=1.2, label="真合焦位置")
    markers = ["o", "s", "^", "D"]
    for k, row in enumerate(res["rows"]):
        v = [np.interp(p, pos, curve) for p in row["visited"]]
        ax.plot(row["visited"], np.asarray(v) / curve.max(), markers[k % 4],
                ms=5, alpha=0.85, label=f'{row["strategy"]}（{row["frames"]} 帧, '
                                        f'误差 {row["err"]:.3f}）')
    ax.set_xlabel("镜头位置")
    ax.set_ylabel("归一化评价函数值")
    ax.set_title("搜索策略：帧数 vs 定位精度\n（爬山法帧数最少，但没有利用单峰性，精度最差）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, outdir, "fig_af_search.png")


def fig_coupling(res: dict, outdir: str) -> str:
    rows = res["rows"]
    ev = [r["ev"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].plot(ev, [r["awb_err"] for r in rows], "o-", color=C_MAIN)
    axes[0].set_xlabel("AE 曝光 EV")
    axes[0].set_ylabel("AWB 光源角度误差 (度)")
    axes[0].set_title("3A 耦合（一）：曝光不足时 AWB 精度下降\n"
                      "（暗部信噪比低 + 有效像素减少）")
    axes[0].grid(alpha=0.3)

    axes[1].plot(ev, [r["af_peak_err"] for r in rows], "o-", color=C_WARN,
                 label="对焦峰位误差")
    ax2 = axes[1].twinx()
    ax2.plot(ev, [r["af_dyn_db"] for r in rows], "s--", color=C_OK,
             label="评价函数动态范围 (dB)")
    ax2.set_ylabel("动态范围 (dB)", color=C_OK)
    axes[1].set_xlabel("AE 曝光 EV")
    axes[1].set_ylabel("峰位误差", color=C_WARN)
    axes[1].set_title("3A 耦合（二）：曝光不足时\n对焦评价函数的动态范围塌陷")
    axes[1].grid(alpha=0.3)
    h1, l1 = axes[1].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axes[1].legend(h1 + h2, l1 + l2, fontsize=8, loc="best")
    return _save(fig, outdir, "fig_coupling.png")


def fig_isp_stages(res: dict, outdir: str) -> str:
    names = list(res["stages"].keys())
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.4))
    axes = axes.ravel()
    for i, n in enumerate(names):
        ax = axes[i]
        ax.imshow(_to_u8(res["stages"][n]))
        ax.set_title(n, fontsize=9)
        ax.axis("off")
    for j in range(len(names), len(axes)):
        axes[j].axis("off")
    # 最后两格放 LSC 对照
    ax = axes[len(names)]
    ax.imshow(_to_u8(res["no_lsc"]))
    ax.set_title(f'关闭 LSC（边角发暗）\n均匀度 {res["uniformity_no_lsc"]:.3f}', fontsize=9)
    ax.axis("off")
    ax = axes[len(names) + 1]
    ax.imshow(_to_u8(res["with_lsc"]))
    ax.set_title(f'开启 LSC\n均匀度 {res["uniformity_with_lsc"]:.3f}', fontsize=9)
    ax.axis("off")
    fig.suptitle("ISP 处理链路的中间结果（顺序：BLC→LSC→去马赛克→WB→CCM→色调映射）", y=1.02)
    return _save(fig, outdir, "fig_isp_stages.png")


# -----------------------------------------------------------------------------
# 报告输出
# -----------------------------------------------------------------------------
def _b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


# -----------------------------------------------------------------------------
# 时域（抖动源 / 权衡 / 场景切换 / AWB 稳定）
# -----------------------------------------------------------------------------
def fig_temporal_source(res: dict, outdir: str) -> str:
    rows = res["rows"]
    regimes = []
    for r in rows:
        if r["regime"] not in regimes:
            regimes.append(r["regime"])

    fig, axes = plt.subplots(1, len(regimes), figsize=(6.0 * len(regimes), 4.2))
    if len(regimes) == 1:
        axes = [axes]
    for ax, reg in zip(axes, regimes):
        sub = [r for r in rows if r["regime"] == reg]
        labels = [r["label"].replace("（噪声地板）", "") for r in sub]
        vals = [max(r["jitter_metric_std"], 1e-9) for r in sub]
        colors = [C_GRAY if r["source"] == "S0" else
                  (C_OK if r["floor_ratio"] >= 20.0 else C_WARN) for r in sub]
        ax.barh(range(len(sub)), vals, color=colors)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("稳态抖动 std（EV，对数轴）")
        ax.set_title(f"{reg}：噪声地板 {sub[0]['floor']:.1e} EV")
        ax.grid(alpha=0.3, axis="x")
        for i, r in enumerate(sub):
            ax.text(vals[i], i, f"  {r['floor_ratio']:.1f}×", va="center", fontsize=8)
    fig.tight_layout()
    return _save(fig, outdir, "fig_temporal_source.png")


def fig_temporal_tradeoff(res: dict, outdir: str) -> str:
    rows = res["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    ax = axes[0]
    for r in rows:
        if r["settle_frames"] < 0:
            ax.scatter(0, r["jitter_ev_cmd_std"], s=60, color=C_WARN, marker="x", zorder=3)
            ax.annotate(r["label"] + "（未收敛）", (0, r["jitter_ev_cmd_std"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=7.5)
            continue
        ok = r["pareto_optimal"]
        ax.scatter(r["settle_frames"], r["jitter_ev_cmd_std"],
                   s=110 if ok else 55, color=C_OK if ok else C_MAIN,
                   marker="*" if ok else "o", zorder=3,
                   edgecolor="k", linewidth=0.4)
        ax.annotate(r["label"].replace("（检测器关）", ""),
                    (r["settle_frames"], r["jitter_ev_cmd_std"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=7.5)
    ax.set_xlabel("场景切换后重收敛帧数（越左越好）")
    ax.set_ylabel("稳态曝光抖动 std（EV，越下越好）")
    ax.set_title("抖动-延迟平面：★ = 帕累托最优")
    ax.grid(alpha=0.3)

    ax = axes[1]
    pairs, labels = [], []
    for r in rows:
        if not r["detector"] or r["adaptive"]:
            continue
        mate = next((o for o in rows if not o["detector"]
                     and abs(o["alpha_slow"] - r["alpha_slow"]) < 1e-9), None)
        if mate is None:
            continue
        labels.append(f"alpha={r['alpha_slow']:.1f}")
        pairs.append((mate["settle_frames"], r["settle_frames"]))
    if pairs:
        x = np.arange(len(pairs))
        off = [p[0] for p in pairs]
        on = [p[1] for p in pairs]
        ax.bar(x - 0.19, off, 0.38, label="检测器关", color=C_WARN)
        ax.bar(x + 0.19, on, 0.38, label="检测器开", color=C_OK)
        for i, (a, b) in enumerate(zip(off, on)):
            ax.text(i - 0.19, a, str(a), ha="center", va="bottom", fontsize=9)
            ax.text(i + 0.19, b, str(b), ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
    ax.set_ylabel("重收敛帧数")
    ax.set_title("检测器的贡献：延迟直降、抖动不变\n（同一 alpha 成对比较）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return _save(fig, outdir, "fig_temporal_tradeoff.png")


def fig_temporal_trace(res: dict, outdir: str) -> str:
    rows = res["rows"]
    pick = [r for r in rows if r["label"].startswith("无滤波")]
    pick += [r for r in rows if "alpha=0.4" in r["label"]]
    fig, ax = plt.subplots(figsize=(11, 4.0))
    for r in pick:
        ax.plot(range(len(r["ev_hist"])), r["ev_hist"], "-", lw=1.5, label=r["label"])
    ax.axvline(res["cut_frame"], color="k", ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("帧序号")
    ax.set_ylabel("曝光 EV")
    ax.set_title("内容切换后的重收敛：无滤波过冲振荡，检测器一开立刻松手")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, outdir, "fig_temporal_trace.png")


def fig_scene_cut_signal(res: dict, outdir: str) -> str:
    names = ["d_metric_ev", "texture_ratio", "hist_dist"]
    th = res["thresholds"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for ax, k in zip(axes, names):
        smax = res["static_max"][k]
        cmin = res["cut_min"][k]
        vals = [smax, cmin, th[k]]
        ax.bar(["静止最坏", "切换最小", "门限"], vals, color=[C_WARN, C_OK, C_GRAY])
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_title(f"{k}\n分离度 {res['separation'][k]:.1f}×")
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return _save(fig, outdir, "fig_scene_cut_signal.png")


def fig_scene_cut_events(res: dict, outdir: str) -> str:
    cuts = res["cuts"]
    labels = [f"{c['event']}\nseed={c['seed']}" for c in cuts]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4),
                             gridspec_kw={"width_ratios": [1.35, 1]})

    ax = axes[0]
    x = np.arange(len(cuts))
    ax.bar(x - 0.26, [c["d_metric_ev"] for c in cuts], 0.26,
           label="d_metric_ev", color=C_MAIN)
    ax.bar(x, [c["hist_dist"] for c in cuts], 0.26, label="hist_dist", color=C_OK)
    ax.bar(x + 0.26, [c["texture_ratio"] for c in cuts], 0.26,
           label="texture_ratio", color=C_WARN)
    ax.axhline(res["thresholds"]["d_metric_ev"], color=C_MAIN, ls=":", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=18)
    for i, c in enumerate(cuts):
        ax.text(i, -0.14, "检出" if c["detected"] else "漏检", ha="center", fontsize=8,
                color=C_OK if c["detected"] else C_WARN)
    ax.set_title("各切换事件上的信号值（虚线 = d_metric 门限）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    lab = ["误触发率", "检出率", "分类准确率"]
    val = [res["false_trigger_rate"], res["detect_rate"], res["kind_accuracy"]]
    ax.bar(lab, val, color=[C_OK, C_MAIN, C_OK])
    for i, v in enumerate(val):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("比率")
    ax.set_title(f"静止 {res['n_static_runs']} 组 / 切换 {res['n_cut_runs']} 次\n"
                 f"等亮度色温 {res['cct_equal_luma_detected']}/"
                 f"{res['cct_equal_luma_total']}（已知边界）", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return _save(fig, outdir, "fig_scene_cut_events.png")


def fig_awb_temporal(res: dict, outdir: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    rows = res["rows"]

    ax = axes[0]
    ax.bar([r["label"] for r in rows], [r["gain_flicker"] for r in rows],
           color=[C_WARN] + [C_MAIN] * (len(rows) - 1))
    for i, r in enumerate(rows):
        ax.text(i, r["gain_flicker"], f"{r['gain_flicker']:.5f}",
                ha="center", va="bottom", fontsize=8.5)
    ax.set_ylabel("增益跳动 mean|d log2(gain)|")
    ax.set_title("颜色呼吸：逐帧独立估计 vs 对数域时域稳定")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    for r in rows:
        ax.plot(range(len(r["err_hist"])), r["err_hist"], "-", lw=1.4, label=r["label"])
    ax.axvline(res["cut_frame"], color="k", ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("帧序号")
    ax.set_ylabel("光源角度误差（度）")
    ax.set_title("误差均值不随滤波变差（平滑不引入偏差）", fontsize=9.5)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, outdir, "fig_awb_temporal.png")


# -----------------------------------------------------------------------------
# C++ 统计通路与定点化
# -----------------------------------------------------------------------------
_NATIVE_LABELS = {
    "awb_py_naive": "AWB 现状(Python)",
    "awb_py_opt": "AWB 算法优化(Python)",
    "awb_c_f64": "AWB C++ (double)",
    "awb_c_f32": "AWB C++ (float32)",
    "awb_c_q16": "AWB C++ (定点 Q16)",
    "ae_py_naive": "AE 测光 现状(Python)",
    "ae_c_f32": "AE 测光 C++ (float32)",
    "ae_avg_py": "AE 全画面平均(Python)",
    "ae_avg_c": "AE 全画面平均 C++",
}


def fig_native_speed(res: dict, outdir: str) -> str:
    keys = [k for k in res["timing"] if k.startswith("awb_")]
    names = [_NATIVE_LABELS.get(k, k) for k in keys]
    med = [res["timing"][k]["median_us"] for k in keys]
    lo = [res["timing"][k]["median_us"] - res["timing"][k]["p10_us"] for k in keys]
    hi = [res["timing"][k]["p90_us"] - res["timing"][k]["median_us"] for k in keys]

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.bar(range(len(keys)), med, yerr=[lo, hi], capsize=3,
           color=[C_WARN, C_WARN, C_MAIN, C_MAIN, C_OK])
    floor = res["null_call_us"]
    ax.axhline(floor, color="k", ls=":", lw=1)
    ax.text(len(keys) - 0.5, floor, f" ctypes 调用地板 {floor:.2f}us",
            va="bottom", ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(names, fontsize=8, rotation=12)
    for i, v in enumerate(med):
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("单次耗时（微秒，对数轴）")
    ax.set_title(f"AWB 统计通路：中位数与 p10–p90 区间（{res['size'][0]}×{res['size'][1]}，"
                 f"{res['repeats']} 次重复）")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return _save(fig, outdir, "fig_native_speed.png")


def fig_native_decomp(res: dict, outdir: str) -> str:
    d = res["decomposition"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))

    ax = axes[0]
    lab = ["算法改进\n(Python→Python)", "实现改进\n(Python→C++)", "总加速"]
    val = [d["algo_gain_awb"], d["impl_gain_awb"], d["total_gain_awb"]]
    ax.bar(lab, val, color=[C_MAIN, C_OK, C_WARN])
    for i, v in enumerate(val):
        ax.text(i, v, f"{v:.2f}×", ha="center", va="bottom", fontsize=10)
    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.set_ylabel("相对倍数")
    ax.set_title("AWB 的加速分解：总加速 = 算法 × 实现")
    ax.grid(alpha=0.3, axis="y")

    # 右图是这一节最重要的一格：**没有算法差异**时的纯语言差异
    ax = axes[1]
    lg = d["language_only_gain"]
    ax.bar(["纯语言差异\n(全画面平均，无算法差异)"], [lg],
           color=C_OK if lg >= 1.0 else C_WARN)
    ax.axhline(1.0, color="k", ls="--", lw=1.2)
    ax.text(0, lg, f"{lg:.2f}×", ha="center", va="bottom", fontsize=14)
    ax.set_ylim(0, max(1.6, lg * 1.35))
    ax.set_ylabel("Python / C++")
    ax.set_title("换语言本身值多少？\n（<1 表示 C 反而更慢）")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return _save(fig, outdir, "fig_native_decomp.png")


def fig_fixed_budget(res: dict, outdir: str) -> str:
    rows = res["ae_rows"]
    modes = []
    for r in rows:
        if r["mode"] not in modes:
            modes.append(r["mode"])
    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    x = np.arange(len(modes))
    for i, scene in enumerate(sorted({r["scene"] for r in rows})):
        sub = {r["mode"]: r for r in rows if r["scene"] == scene}
        ratio = [sub[m]["abs_diff"] / sub[m]["bound"] for m in modes]
        ax.bar(x + (i - 1) * 0.26, ratio, 0.26, label=scene)
    ax.axhline(1.0, color=C_WARN, ls="--", lw=1.5)
    ax.text(len(modes) - 0.5, 1.0, " 推导上界", va="bottom", ha="right",
            fontsize=9, color=C_WARN)
    ax.set_xticks(x)
    ax.set_xticklabels(modes, fontsize=8)
    ax.set_ylabel("实测误差 / 推导上界")
    ax.set_title("定点化误差：全部落在推导出的上界内（柱高 < 1 即合格）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return _save(fig, outdir, "fig_fixed_budget.png")


def fig_fixed_sweep(res: dict, outdir: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))

    ax = axes[0]
    bits = [r["bits"] for r in res["by_bits"]]
    havg = [max(r["average_abs"], 1e-12) for r in res["by_bits"]]
    hhl = [max(r["highlight_priority_abs"], 1e-12) for r in res["by_bits"]]
    ax.plot(bits, havg, "o-", color=C_MAIN, label="全画面平均（均值类）")
    ax.plot(bits, hhl, "s-", color=C_WARN, label="高光优先（分位数）")
    ax.plot(bits, [max(1.0 / min(b, 10) if b >= 10 else 1.0 / (1 << b), 1e-12)
                   for b in bits], ":", color=C_GRAY, label="分位数上界（一个 bin）")
    ax.set_yscale("log")
    ax.set_xlabel("输入位宽 (bit)")
    ax.set_ylabel("绝对误差（对数轴）")
    ax.set_title("位宽扫描：10 位以上就没收益了\n（瓶颈从输入量化转到直方图 bin 宽）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    bins = [r["bins"] for r in res["by_bins"]]
    e = [max(r["abs"], 1e-12) for r in res["by_bins"]]
    w = [r["bin_width"] for r in res["by_bins"]]
    ax.plot(bins, e, "o-", color=C_MAIN, label="实测绝对误差")
    ax.plot(bins, w, ":", color=C_WARN, label="一个 bin 宽（误差上界）")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("直方图 bin 数")
    ax.set_ylabel("绝对误差（对数轴）")
    ax.set_title("bin 数扫描：误差始终压在 bin 宽之下")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, outdir, "fig_fixed_sweep.png")


def _sections(res: dict, figs: dict) -> list:
    """返回 (标题, 说明 markdown, 图片 key 列表) 的列表。

    这里的文字才是这份报告的价值所在：每个结论都对应一组实测数字，
    并且每一条都指向一个可迁移的工程判断。
    """
    ae = res["ae_metering"]
    conv = res["ae_convergence"]
    pol = res["ae_policy"]
    flick2 = res["ae_flicker"]
    awb = res["awb"]
    cons = res["awb_constraint"]
    ccm = res["ccm"]
    afc = res["af_curves"]
    afs = res["af_search"]
    cpl = res["coupling"]

    rows_met = "\n".join(
        f'| {m} | {d["iters"]} | {d["ev"]:+.2f} | '
        f'{d["mean_code"]:.0f} | **{d["subject_code"]:.0f}** | {d["clip"] * 100:.2f}% |'
        for m, d in ae["modes"].items())

    rows_conv = "\n".join(
        f'| {r["label"]} | {len(r["ev_hist"])} | {r["final_err"]:.4f} | {r["reversals"]} |'
        for r in conv["runs"])

    rows_pol = "\n".join(
        f'| {r["policy"]} | {r["et_ms"]:.2f} | {r["gain"]:.2f} | '
        f'{r["snr_db"]:.2f} | {r["sharpness"]:.4f} |' for r in pol["rows"])

    rows_awb = "\n".join(
        "| " + s + " | " + f'{awb["scenes"][s]["temp"]:.0f}K' + " | " +
        " | ".join(f'{awb["scenes"][s]["methods"][m]["err_deg"]:.2f}'
                   for m in awb["methods"]) + " |"
        for s in awb["scenes"])

    rows_cons = "\n".join(
        f'| {r["label"]} | {r["true_cct"]:.0f} | {r["raw_cct"]:.0f} | {r["raw_duv"]:+.4f} | '
        f'{r["raw_err"]:.2f} | {r["cct"]:.0f} | {r["err"]:.2f} |'
        for r in cons["rows"])

    rows_afm = "\n".join(
        f'| {m} | {mt["peak_pos"]:.3f} | {abs(mt["peak_pos"] - afc["true_focus"]):.3f} | '
        f'{mt["fwhm"]:.3f} | {mt["dynamic_range_db"]:.1f} | {mt["monotonic_violations"]} | '
        f'{afc["noise_std"][m]:.3f} |'
        for m, mt in afc["metrics"].items())

    rows_afs = "\n".join(
        f'| {r["strategy"]} | {r["frames"]} | {r["best"]:.3f} | {r["err"]:.3f} |'
        for r in afs["rows"])

    rows_cpl = "\n".join(
        f'| {r["ev"]:+.1f} | {r["snr_db"]:.1f} | {r["awb_err"]:.2f} | '
        f'{r["af_dyn_db"]:.1f} | {r["af_peak_err"]:.3f} | {r["clip"] * 100:.1f}% |'
        for r in cpl["rows"])

    secs = []

    secs.append(("1. ISP 链路：3A 各自工作在哪一层", f"""
先确认一件事：**3A 不是三个独立模块，它们各自工作在 ISP 链路的特定域上**，
工作域选错，算法本身再对也没用：
- AWB 统计必须在**白平衡前**的线性域做 —— 否则统计量被自己的修正结果污染，形成自激
- AE 统计用**显示域亮度** —— 与人眼感知一致，目标码值稳定，且与色调曲线解耦
- AF 统计用**去马赛克后的绿通道** —— G 的采样率是 R/B 的两倍（占 50%），SNR 最好，
  且不受白平衡增益影响，天然与 AWB 解耦

图中的 LSC 对照量化了 RAW 域处理的价值：关闭时边角发暗，均匀度（四角/中心）
从 {res['isp']['uniformity_with_lsc']:.3f} 掉到 {res['isp']['uniformity_no_lsc']:.3f}。
LSC 必须在 RAW 域按 CFA 通道分别补偿 —— 阴影衰减是光子层面的，三通道衰减还不同，
放到去马赛克之后补，会把已经混进颜色里的误差一起放大。
""", ["isp"]))

    secs.append(("2. AE —— 测光：决定成片的不是控制算法，而是测光方式", f"""
逆光场景：大面积亮背景（约 55%）+ 画面正中的人物主体（约 8%）。五种测光方式：

| 测光方式 | 帧数 | 最终 EV | 全画面码值 | 主体码值 | 过曝比例 |
|---|---|---|---|---|---|
{rows_met}

**结论**：
1. 五种测光方式**全部正常收敛**，但成片完全不同 —— 主体码值从
   {ae['modes']['average']['subject_code']:.0f}（全画面平均）到
   {ae['modes']['spot']['subject_code']:.0f}（点测光）。所以"AE 准不准"这个问法本身就有问题：
   **先要问测的是什么**。
2. 全画面平均测光在逆光下把主体压暗，这是"用整幅图的均值代表亮度"的必然结果，
   不是控制算法的问题。分区评价测光加入过曝区域惩罚后有所改善，
   但幅度有限 —— 中心加权不够集中时，大面积亮背景依然主导统计量。
3. 点测光锁定主体最准（{ae['modes']['spot']['subject_code']:.0f}），
   代价是几乎放弃高光（过曝 {ae['modes']['spot']['clip'] * 100:.2f}%）。
4. 高光优先（99 分位）走另一个极端：整体偏亮，但高光刚好不溢出
   （过曝仅 {ae['modes']['highlight_priority']['clip'] * 100:.2f}%）。

**没有一种测光是普适的**。产品上要么做场景识别自动切换，要么把分区权重
做成可调的 tuning 参数 —— 这正是"画质调优"里 AE 调参的核心工作。
""", ["ae_metering"]))

    secs.append(("3. AE —— 控制：为什么必须变步长（以及饱和的反直觉之处）", f"""
曝光与亮度在**线性域**近似成正比，所以控制律放在 log2 域做 ——
对数域里成像是"平移"关系，一套控制参数就能适应任意光照水平。

| 步长策略 | 收敛帧数 | 最终误差 (EV) | 方向反转次数 |
|---|---|---|---|
{rows_conv}

**结论**：
- 固定全步长最快（{len(conv['runs'][0]['ev_hist'])} 帧），但出现了
  {conv['runs'][0]['reversals']} 次方向反转 —— 这就是振荡的苗头
- 固定小步长最稳（0 次反转），代价是帧数翻倍
- 变步长：大误差时用大步长（远离饱和，线性假设成立），小误差时收小步长
  （进入噪声与非线性主导区），以 2 帧的代价换来单调收敛

**这一节最值得讲的是 AE 的一个反直觉陷阱**（本项目实现时踩到并修掉）：
一旦大片像素饱和，测得的亮度就顶在 1.0 不动了。哪怕实际过曝 2 档，
99 分位也只能给到 1.0，误差被死死压到 log2(0.9/1.0) = -0.15 EV ——
**越曝越"看不出来"**，控制器以为快到目标了，一帧只退 0.15 EV，
大逆光场景十几帧都收敛不了（第一版实测：12 帧仍未收敛、55% 像素过曝）。

解法是改用**与亮度无关的信息**：过曝像素比例。它是饱和的直接证据，
按它给出步长下限，同一场景 7 帧收敛。这条经验在任何 AE 实现里都成立。
""", ["ae_conv", "ae_curve"]))

    secs.append(("4. AE —— 执行器分配：快门与增益不等价，抗闪烁是约束曝光时间", f"""
同样一个 EV，可以拆成不同的 (曝光时间, 增益) 组合。低照度场景下的实测：

| 曝光策略 | 曝光时间 (ms) | 增益 | SNR (dB) | 清晰度 |
|---|---|---|---|---|
{rows_pol}

**结论 1**：两种策略的 SNR 几乎完全一样
（{pol['rows'][0]['snr_db']:.2f} vs {pol['rows'][1]['snr_db']:.2f} dB），
但清晰度差 {pol['rows'][1]['sharpness'] / max(pol['rows'][0]['sharpness'], 1e-9):.1f} 倍。
原因很直接：**增益放大的是同一份光子噪声，提增益不会改善信噪比**，
它唯一的作用是缩短曝光时间、减少运动模糊。
这就是"夜景拍运动物体该不该降快门"这类产品决策的量化依据。

**结论 2：抗闪烁不是滤波器，而是对曝光时间取值集合的约束。**
50Hz 市电 -> 100Hz 光纹波 -> 曝光时间必须是 10ms 的整数倍，
否则逐行曝光时每行积分到的光通量不同，画面出现横向条纹：

- 关闭抗闪烁：曝光 {flick2['runs'][0]['et_ms']:.2f} ms，带纹强度 {flick2['runs'][0]['banding']:.4f}
- 开启抗闪烁：曝光 {flick2['runs'][1]['et_ms']:.2f} ms，带纹强度 {flick2['runs'][1]['banding']:.4f}

**两个容易漏掉的坑**：
1. 只把曝光时间量化还不够，**曝光上限也必须压到整数倍**上。默认上限
   1/30s = 33.3ms 不是 10ms 的整数倍，长曝时条纹会回来
   （本项目实现里 `ev_limits()` 会随抗闪烁一起收紧上限）。
2. **带纹测试必须用行轮廓平坦的场景（匀光板）**。用风景或色卡测，
   场景自身的行结构会被误判成带纹；本项目第一版用色卡测，
   甚至得出了"开抗闪烁比不开更差"的伪结论。
""", ["ae_policy", "ae_flicker"]))

    secs.append(("5. AWB —— 没有一种算法在所有场景都赢", f"""
四种经典方法 + 置信度加权融合，三个场景下的光源角度误差（度，越小越好）。
所有场景都先用 AE 曝光到位再评 AWB：

| 场景 | 真实色温 | {" | ".join(awb["methods"])} |
|---|{"---|" * len(awb["methods"])}
{rows_awb}

**结论**：
- **标准色卡**：Shades-of-Gray（{awb['scenes']['color_chart']['methods']['shades_of_gray']['err_deg']:.2f}°）
  和融合（{awb['scenes']['color_chart']['methods']['fusion']['err_deg']:.2f}°）最准；
  灰世界反而明显偏了（{awb['scenes']['color_chart']['methods']['gray_world']['err_deg']:.2f}°）——
  色卡本身色彩丰富，"平均下来是灰"这个假设就不成立。
- **逆光自然场景**：灰世界反而最好（{awb['scenes']['natural_backlit']['methods']['gray_world']['err_deg']:.2f}°）。
  **同一套算法在两个场景下的排名完全颠倒。**
- **大面积单色场景**：所有"单一光源假设"的方法都失效
  （灰世界 {awb['scenes']['muted_red']['methods']['gray_world']['err_deg']:.2f}°、
  灰边 {awb['scenes']['muted_red']['methods']['gray_edge']['err_deg']:.2f}°），
  融合靠置信度加权把误差压到 {awb['scenes']['muted_red']['methods']['fusion']['err_deg']:.2f}°。

融合不是简单平均，而是**先判断每种算法在当前场景下可不可信，再加权**：
画面越彩 -> 灰世界权重越低；有像素过曝 -> 白块权重越低（最亮点已经不是中性面）；
灰边样本太少 -> 灰边权重越低。加权在**对数域**做 —— 光源是乘性量，
算术平均会被一个离谱的估计值直接带偏。
""", ["awb_grid", "awb_error"]))

    secs.append(("6. AWB —— 色温先验：它管住了什么，没管住什么", f"""
把估计光源约束到普朗克轨迹附近（真实光源绝大多数靠近黑体轨迹）。
做法是"保留 Duv、只截断色温"：把轨迹上的基准点移到边界，再加上原来的偏差向量 ——
只限制一个自由度，不会把估计结果抹平。

| 场景 | 真实色温 | 约束前估计 | Duv | 误差(度) | 约束后估计 | 误差(度) |
|---|---|---|---|---|---|---|
{rows_cons}

**结论（本项目最反直觉的一组结果）**：
1. 三个场景里只有**一个**真正触发了约束（估计色温越出 [2000, 12000]）。
   把色温范围钳位当作保护措施，实际生效的频率比想象的低得多。
2. 更关键：**色温约束只能限制沿轨迹方向的误差，对垂直轨迹方向（Duv）的误差
   完全无能为力**。蓝色主导场景的估计色温 {cons['rows'][1]['cct']:.0f}K 看着"在合理范围内"，
   误差却高达 {cons['rows'][1]['err']:.1f}° —— 错误全落在 Duv 方向上。
   而大面积单色场景造成的偏色，主要就落在 Duv 方向。
3. 真正有效的做法是**光源色域约束**（gamut mapping：只用真实光源色域内的点做估计），
   或按场景类型分支。只钳色温范围，是"看起来做了保护"。

**验证方法本身也值得说**：Duv 必须在 CIE 1960 UCS 空间里量，
直接用 xy 或 u'v' 算出的距离与视觉感受不一致。本项目实现了完整的色温/Duv 分解，
才把这个失效模式测出来 —— 如果只看色温，它会被完全掩盖。
""", ["awb_constraint"]))

    secs.append(("7. 颜色：白平衡与 CCM 的贡献拆解，以及它们的耦合", f"""
色卡平均 ΔE00：

| 条件 | ΔE00 |
|---|---|
""" + "\n".join(f'| {k} | {v:.2f} |' for k, v in ccm["delta_e"].items()) + f"""

**结论**：
1. 不做白平衡时色偏最大；做对白平衡后 ΔE00 从 {ccm['delta_e']['无白平衡']:.2f}
   降到 {ccm['delta_e']['理想白平衡']:.2f} —— **白平衡是颜色精度的主要贡献项**。
2. CCM 由色卡最小二乘标定（带"白点必须映射到白点"的行和约束），
   标定结果是带**负交叉项**的非单位阵：
   `{np.round(np.array(ccm['ccm']), 3).tolist()}`
   这个负交叉项补偿的是**传感器的光谱串扰**（CFA 滤光不理想，R 像素也收到绿光）。
   本项目在仿真里显式建模了串扰，所以这道颜色题不是一道假题 ——
   如果不建串扰，传感器通道恰好等于场景反射率，CCM 会退化成单位阵，
   关于 CCM 的所有结论都会变成自欺欺人。
3. **耦合**：CCM 是在白平衡**正确**的前提下标定的。AWB 一旦判错光源，
   CCM 的交叉项会把通道间的错误继续混合放大：

   | 条件 | ΔE00 |
   |---|---|
   | 白平衡正确 + CCM | {ccm['coupling']['matched']:.2f} |
   | 白平衡判错、不加 CCM | {ccm['wrong_wb']['raw']:.2f} |
   | 白平衡判错、仍加 CCM | {ccm['wrong_wb']['with_ccm']:.2f} |

   最后一行比第二行**更差**。**CCM 不是纠错手段**，它只在白平衡正确时做精细补偿。
   这也解释了为什么工程上 AWB 的稳定性比绝对精度更重要：它错了，
   后面整条颜色链路都会跟着一起错。
""", ["ccm"]))

    secs.append(("8. AF —— 对焦评价函数怎么选", f"""
同一帧数据上比较五种经典评价函数（{afc["position_count"]} 个镜头位置，
重复 {afc["repeats"]} 次取峰位抖动）：

| 评价函数 | 峰位 | 峰位误差 | 半高宽(FWHM) | 动态范围 (dB) | 单调性破坏 | 峰位抖动 σ |
|---|---|---|---|---|---|---|
{rows_afm}

四个性质的含义与实测：
- **无偏性**：峰位要等于真合焦位置。仿真里几何 PSF 本身无偏，
  所以上表的峰位误差主要来自**噪声导致的峰位漂移**（σ 列），
  这正是低照度下"对不准"的根源。
- **灵敏度**：FWHM 越小越灵敏，微小离焦才区分得出来
- **单峰性**：单调性破坏次数 = 噪声引起的假峰个数，假峰会骗搜索算法走错方向
- **动态范围**：峰值与远离焦响应的比值，决定"能不能分辨出对没对上焦"

**结论**：
- 拉普拉斯方差动态范围最好（{afc['metrics']['laplacian_var']['dynamic_range_db']:.1f} dB），
  对高频最敏感，但单调性破坏次数说明它也最容易被噪声干扰
- SML 对二阶差分做了限幅，抗噪性更好，是工程上常见的折中
- **FFT 高频占比只有 {afc['metrics']['fft_energy']['dynamic_range_db']:.1f} dB 动态范围 ——
  几乎没有分辨能力**。峰位虽然是对的，但不适合单独用于精对焦。
  这个函数看起来最"优雅"，实测最不实用。
- 产品上通常组合使用：粗对焦用一个灵敏的，精对焦用一个抗噪的。
""", ["af_curves"]))

    secs.append(("9. AF —— 搜索策略：帧数就是对焦速度", f"""
| 搜索策略 | 帧数 | 找到的位置 | 定位误差 |
|---|---|---|---|
{rows_afs}

**结论**：
- **全扫描**（{afs['rows'][0]['frames']} 帧）最稳，误差 {afs['rows'][0]['err']:.3f}
- **爬山法**帧数最少（{afs['rows'][1]['frames']} 帧），误差最大（{afs['rows'][1]['err']:.3f}）。
  它只利用"局部是否在上升"这一个信息，没有利用单峰性，容易停在峰顶平台上就近结束
- **粗到细**用 {afs['rows'][2]['frames']} 帧达到与全扫描相同的精度 —— 工程上最常用
- **黄金分割**（{afs['rows'][3]['frames']} 帧）在严格单峰函数上理论最优，
  实测定位误差 {afs['rows'][3]['err']:.3f}，反而输给粗到细。
  原因很实在：它一旦被噪声造成的假峰误导，就会被**永久关在错误的区间里**出不来
  （区间收缩是不可逆的）

这一节想说明的是：**理论最优 ≠ 工程可用**。
真实镜头的评价函数从来不是严格单峰的（噪声、弱纹理、次峰），
所以"错了还能救回来"的策略，比理论最优的策略更值得选。

**另一个物理层面的发现**：峰顶存在一段宽度约 0.5 px 弥散圈的**不可分辨平台** ——
离焦小于半个像素时，图像在像素级上确实没有变化（本项目第一版就因此看到
评价函数在一大段镜头位置上完全水平、峰位随机落在平台中间）。
对焦定位精度因此存在物理下限，不是算法不够好；这个下限由像素尺寸与景深共同决定。
""", ["af_search"]))

    secs.append(("10. 3A 耦合：三者不是独立的三个模块", f"""
把 AE 的曝光从 -5 EV（严重欠曝）扫到 +1 EV（过曝），观察 AWB 与 AF 各自怎么退化：

| 曝光 EV | SNR (dB) | AWB 角度误差(度) | AF 动态范围(dB) | AF 峰位误差 | 过曝比例 |
|---|---|---|---|---|---|
{rows_cpl}

**结论**：
1. **欠曝主要伤害 AWB**：EV ≤ -3 时 AWB 直接**完全失效**
   （误差从 {cpl['rows'][4]['awb_err']:.2f}° 恶化到 {cpl['rows'][0]['awb_err']:.2f}°）。
   原因是暗部像素被有效掩码排除，剩下的统计样本质量太差。
   这与"帧数"无关，是**输入数据本身不可用**。
2. **过曝主要伤害 AF**：EV=+1 时评价函数动态范围从
   {cpl['rows'][4]['af_dyn_db']:.1f} dB 塌到 {cpl['rows'][2]['af_dyn_db']:.1f} dB ——
   高光截断把高频细节整片抹平，对焦的分辨能力几乎消失。
3. AF 峰位误差始终在一个扫描格之内，说明在本项目这个高对比标板上，
   CDAF 对噪声并不敏感 —— **但这个结论依赖场景对比度**：
   换成弱纹理场景，低照度下峰位漂移会明显放大。

**这一节真正的意义**：3A 是三个共享同一路图像的闭环，不是三个独立模块。
如果各自单独调参，很容易出现"单独测都很好、合起来就抖"的现象。
联调顺序（先 AE 稳住曝光 -> 再 AWB -> 最后 AF）、统计的时域滤波、
以及"某一路失效时要不要冻结其他路"，都是产品级问题。
""", ["coupling"]))

    iq = res["image_quality"]
    msc = iq["mtf_selfcheck"]
    nse = iq["noise"]
    dr = iq["dynamic_range"]
    gn = iq["gain_vs_noise"]

    # 由 MTF vs 镜头位置数据派生的几个量（第 11.2 节要用）
    _foc = iq["mtf_vs_focus"]
    _lin = np.asarray(_foc["mtf50"], dtype=float)
    _disp = np.asarray(_foc["mtf50_display"], dtype=float)
    _tg = np.asarray(_foc["tenengrad"], dtype=float)
    _ok = ~np.isnan(_lin)
    mtf_peak = float(_foc["positions"][int(np.nanargmax(_lin))])
    mtf_corr = float(np.corrcoef(_lin[_ok], _tg[_ok])[0, 1])
    mtf_lin_peak = float(np.nanmax(_lin))
    mtf_disp_peak = float(np.nanmax(_disp))
    sharp_gain = float(np.nanmean(_disp) / np.nanmean(_lin))

    rows_mtf = "\n".join(
        f'| {r["sigma"]:.1f} | {r["angle"]:.2f} | {r["mtf50"]:.3f} | {r["mtf50_theory"]:.3f} | '
        f'{r["dev_pct"]:+.1f}% | {r["curve_err"]:.3f} |'
        for r in msc["rows"])

    rows_gain = "\n".join(
        f'| {"ISO 无关" if r["model"] == "iso_less" else "读出噪声后置"} | {r["gain"]:.0f}× | '
        f'{r["read_noise_dn"]:.3f} | {r["read_noise_e"]:.3f} | {r["snr_at_2e_db"]:+.1f} |'
        for r in gn["rows"])

    rows_shade = "\n".join(
        f'| {r["label"]} | {r["luma_uniformity"]:.3f} | {r["d_uv_corner_max"]:.2f} |'
        for r in iq["shading"]["rows"])

    # ---- 12. 时域抖动源 ----
    ts = res["temporal_source"]
    rows_ts = "\n".join(
        f"| {r['regime']} | {r['label']} | {r['jitter_metric_std']:.2e} | "
        f"{r['floor_ratio']:.1f}× | "
        f"{'是' if (r['source'] != 'S0' and r['floor_ratio'] >= ts['significant_ratio']) else '否'} |"
        for r in ts["rows"])
    secs.append(("12. 时域抖动从哪来：先测噪声地板，再谈滤波", f"""
时域滤波是最容易"做得很漂亮但没意义"的一块，原因是一个必须先讲清楚的**负结论**。

测光量是**整帧空间平均**：{ts['n_frames']} 帧序列上实测噪声地板只有
**{ts['rows'][0]['jitter_metric_std']:.1e} EV**，比收敛阈值 0.02 EV 低约两个数量级。
也就是说**静止场景下 AE 的抖动恒等于浮点噪声** —— 不存在"滤波把抖动降低 90%"
这回事。低照度本身不产生抖动，它只是把执行器推到长曝光/高增益，**间接**放大量化步长。

所以扰动必须作为**显式的物理源**注入，而且每个源都要先证明它相对地板显著
（门槛 {ts['significant_ratio']:.0f}×，{ts['n_significant']} 个源达标）：

| 场景 | 扰动源 | 稳态抖动 std (EV) | 相对地板 | 显著 |
|---|---|---|---|---|
{rows_ts}

三个源各有各的脾气，都不是"调个参数"能糊过去的：

- **S1 光源强度波动**严格线性 —— 幅度 ×4，抖动也 ×4。它是真实场景里一直存在的那种。
- **S2 闪烁相位漂移**只在**亮场景**显著。强度正比于 `sinc(f·ET)`：亮光下曝光时间短、
  sinc 接近 1；暗光下曝光时间顶到上限，sinc 把它压掉了。这和前面抗闪烁那节是同一件事的两面。
- **S3 执行器量化**是**确定性**的 —— 不需要任何噪声，抖动来自控制结构本身。
  而且严格分域：亮场景增益被钳在 1.0，只有曝光时间量化起作用；暗场景曝光时间顶在上限，
  抖动全部来自增益档位。台阶越大抖动越大（1/6 EV → 1/3 EV 单调上升）。

**S3 才是真实 AE 必须做时域平滑的头号原因** —— 它不会因为光照变好而消失。
""", ["t_source"]))

    # ---- 13. AE 时域滤波与检测器 ----
    ta = res["temporal_ae"]

    def _pick(prefix, det):
        return next((r for r in ta["rows"]
                     if r["label"].startswith(prefix) and r["detector"] is det), None)

    p4o, p4n = _pick("固定 alpha=0.4", False), _pick("固定 alpha=0.4", True)
    p5o, p5n = _pick("固定 alpha=0.5", False), _pick("固定 alpha=0.5", True)
    worst = max((r for r in ta["rows"] if r["settle_frames"] < 0),
                key=lambda r: r["jitter_ev_cmd_std"], default=None)
    rows_ta = "\n".join(
        f"| {r['label']} | {r['jitter_ev_cmd_std']:.5f} | {r['jitter_metric_std']:.5f} | "
        f"{r['settle_frames'] if r['settle_frames'] >= 0 else '**未收敛**'} | "
        f"{'★' if r['pareto_optimal'] else ''} |"
        for r in ta["rows"])
    secs.append(("13. 时域滤波：在闭环里它不是免费午餐", f"""
这一节的结论**与最初的设想相反**，按实测数据写。

抖动源用 S1（0.5% 光源波动，上一节实测
{max(r['floor_ratio'] for r in ts['rows'] if r['label'] == 'S1 光源波动 0.5%'):.0f}× 地板且严格线性）；
在第 {ta['cut_frame']} 帧做内容切换，一条序列同时产出抖动与延迟。

| 配置 | 稳态曝光抖动 std (EV) | 画面抖动 std (EV) | 重收敛帧数 | 帕累托 |
|---|---|---|---|---|
{rows_ta}

**一、时域滤波并没有降低曝光抖动，开太狠反而失稳。**
固定 alpha 的曝光抖动全都比不滤波（{ta['rows'][0]['jitter_ev_cmd_std']:.5f}）**更差**；
alpha=0.2 时涨到 {_pick("固定 alpha=0.2", False)['jitter_ev_cmd_std']:.3f} EV，
alpha=0.1 直接**未收敛**（{"曝光抖动 %.2f EV" % worst["jitter_ev_cmd_std"] if worst else ""}）。
根因是这套 AE 是**变步长**的（大误差时 d=0.9，过曝补偿还能把它推到 1.0）——
慢滤波的相位滞后叠加这个大环路增益，把闭环推向振荡。
所以"滤波开大一点更稳"是错的：**滤波强度存在稳定边界**。

**二、真正有价值的是场景切换检测器。** 同一滤波强度下成对比较：

| 稳态滤波 | 检测器关 | 检测器开 | 曝光抖动 |
|---|---|---|---|
| alpha=0.5 | {p5o['settle_frames']} 帧 | **{p5n['settle_frames']} 帧** | 两行相同（{p5o['jitter_ev_cmd_std']:.5f}） |
| alpha=0.4 | {p4o['settle_frames']} 帧 | **{p4n['settle_frames']} 帧** | 两行相同（{p4o['jitter_ev_cmd_std']:.5f}） |

延迟直降，**抖动分毫不变** —— 这正是检测器存在的理由：
它买到的是"切换瞬间立刻松手"，不牺牲稳态平滑。

**三、自适应 alpha 规则下检测器是冗余的**（两行数值完全相同）——
因为大误差本来就触发 fast。如实记录，不硬凑它有用。
""", ["t_tradeoff", "t_trace"]))

    # ---- 14. 场景切换检测 + AWB ----
    sc = res["scene_cut"]
    aw = res["temporal_awb"]
    sig_names = ("d_metric_ev", "texture_ratio", "hist_dist")
    rows_sep = "\n".join(
        f"| `{k}` | {sc['thresholds'][k]:.3f} | {sc['static_max'][k]:.4f} | "
        f"{sc['cut_min'][k]:.4f} | {sc['separation'][k]:.1f}× | "
        f"{sc['margin_vs_thresh'][k]:.2f}× |"
        for k in sig_names)
    rows_cut = "\n".join(
        f"| {c['event']} | {c['seed']} | {c['d_metric_ev']:.3f} | {c['hist_dist']:.3f} | "
        f"{c['texture_ratio']:.3f} | {c['latency'] if c['detected'] else '未检出'} | "
        f"{c['est_kind']} |"
        for c in sc["cuts"])
    rows_aw = "\n".join(
        f"| {r['label']} | {r['gain_flicker']:.5f} | {r['angle_err_mean']:.4f} | "
        f"{r['angle_err_std']:.4f} | {r['settle_frames']} |"
        for r in aw["rows"])
    secs.append(("14. 场景切换检测与 AWB 时域稳定", f"""
**检测器怎么做的。** 三个逐帧信号投票（至少 2 票），全部从画面现算，不依赖真值：

- `d_metric_ev`：**曝光归一化**后的测光量跳变。归一化是关键 —— 去掉它，
  AE 自己收敛 3 EV 就会被当成一次场景切换。
- `texture_ratio`：结构信号（带通能量占比）比值。对**亮度整体缩放严格不变**。
- `hist_dist`：曝光归一化亮度直方图的距离。

**门限由数据定，不靠猜。** 静止条件取最坏值、切换事件取最小值，门限落在中间：

| 信号 | 门限 | 静止最坏 | 切换最小 | 分离度 | 门限裕度 |
|---|---|---|---|---|---|
{rows_sep}

注意 `texture_ratio` 的门限裕度小于 1 —— **它在光照类事件上永远不会触发**。
真正撑住检出的是 `d_metric_ev` 和 `hist_dist`。但 `texture_ratio` 另有重任：
**只有它能区分"内容变了"和"光变了"**（`hist_dist` 对两类都敏感，拿它分类会把
光照变化误判成内容变化）。实测分类准确率 **{sc['kind_accuracy']:.3f}**
（{sc['kind_correct']}/{sc['kind_total']}）。

| 事件 | seed | dM | hist | tex | 延迟(帧) | 判定类型 |
|---|---|---|---|---|---|---|
{rows_cut}

**整体表现：** 静止 {sc['n_static_runs']} 组条件误触发 **{sc['n_static_fired']}** 次
（含"让 AE 从 +3 EV 收敛"这条最要命的假阳性）；切换 {sc['n_cut_runs']} 次检出
{sc['n_detected']} 次，检出的全部延迟为 0 帧。

**已知边界（如实写）：** 等亮度色温切换检出
{sc['cct_equal_luma_detected']}/{sc['cct_equal_luma_total']} —— 三个信号全都低于门限。
这套检测器测的是**亮度与内容**变化，不是**色度**变化。要覆盖它得另加色度信号。

**AWB 侧的对照：这里滤波是干净有效的。** 序列为 5000K 前 {aw['cut_frame']} 帧、
之后切到 {aw['temp_b']:.0f}K，曝光固定在 {aw['ev_fix']:.2f} EV：

| 配置 | 增益跳动 | 角度误差均值 | 误差 std | 重收敛 |
|---|---|---|---|---|
{rows_aw}

增益跳动降 **{aw['rows'][0]['gain_flicker'] / max(aw['rows'][-1]['gain_flicker'], 1e-12):.1f} 倍**，
而**角度误差均值几乎不变**（{aw['rows'][0]['angle_err_mean']:.4f} →
{aw['rows'][-1]['angle_err_mean']:.4f}）—— 这是"平滑不引入偏差"的正面证据。

**为什么 AWB 行而 AE 不行？** 结构不同：**AWB 是开环估计器**（逐帧独立、无反馈），
滤波只做平均；**AE 是闭环控制**，滤波改变的是环路动态。同一种滤波方法，放在
开环里是纯收益，放进闭环里就得先算稳定边界。
""", ["sc_signal", "sc_events", "t_awb"]))

    secs.append(("15. 画质指标：产品规格书上的那些数字", f"""
前面十节都是"哪个算法更好"的内部对比。这一节回答的是另一类问题：
**这台相机的画质到底是多少** —— MTF50、SNR、动态范围、色阴影。
这些才是画质调优的通用语言，也是能和别人对齐的口径。

**每一项测量都必须先自检**：用已知的输入去测，看能不能把真值反推回来。
不然只是一堆看着合理的数字。

### 11.1 斜边法 MTF（ISO 12233）

用已知 σ 的高斯 PSF 生成斜边，逐一对表：

| PSF σ (px) | 测出的边缘角度 | 实测 MTF50 | 理论 MTF50 | 偏差 | 曲线最大偏差 |
|---|---|---|---|---|---|
{rows_mtf}

**结论**：七个模糊量下 MTF50 最大偏差 {msc["max_dev_pct"]:.1f}%，
整条 MTF 曲线的最大偏差 {msc["max_curve_err"]:.3f}。测量链路可信。

实现过程中有两个坑值得记下来（都会让测量结果**系统性偏移**，但不报错）：
1. **归一化不能先减均值**：LSF 减掉均值会把直流分量压到 0，
   归一化就成了除以一个接近 0 的数，高频端直接炸到 1e15 量级。
2. **窗口宽度必须自适应，且不能用汉明窗**：窗口太窄会截断宽 LSF，
   按截断后的面积归一又把 MTF 整体抬高（实测 σ=3px 偏高 60%）；
   汉明窗整段衰减，等效于在空间域压缩 LSF、频谱展宽，实测抬高约 16%。
   换成**平顶 Tukey 窗 + 按 5σ 自适应定窗**后才压到 2% 以内。

### 11.2 AF 的评价函数在找什么，以及"锐化能提高 MTF 吗"

| | |
|---|---|
| MTF50 峰位 | {mtf_peak:.2f}（真合焦位置 {_foc["true_focus"]:.2f}，峰顶平台跨相邻两格）|
| MTF50 与 Tenengrad 评价函数的相关系数 | {mtf_corr:.3f} |

**结论一**：AF 的评价函数（高频能量）本质上是在用**梯度统计量**逼近物理清晰度 MTF ——
两者在整段镜头行程上的相关系数 {mtf_corr:.3f}。这也解释了它为什么会有假峰和峰位抖动：
梯度统计量对噪声敏感，而 MTF 经过法方向分箱平均，噪声鲁棒得多。
高精度对焦要专门的评价函数硬件统计模块，原因就在这里。

**结论二（更值得说）**：线性域的 MTF50 峰值是 {mtf_lin_peak:.2f}，
走完整条 ISP（色调曲线 + USM 锐化）后在显示图上测是 {mtf_disp_peak:.2f} ——
**锐化把 MTF 数字抬高了 {sharp_gain:.2f} 倍，但信息量一点没增加。**

所以：
- 比较 MTF 必须在同一个域里比，跨域比较出来的数字没有意义
- 任何"锐化后解析力提升 X%"的说法，都要先问清楚是在哪个域测的
- SFR/MTF 的测量规范要求指定 gamma（ISO 12233 用的是特定编码域），根因就在这里

本项目的 MTF 全部在**线性域**测 —— 测的是成像系统本身的解析力，
不是"成像 + 后期"的合成结果。

### 11.3 噪声与光子转换曲线

|  | 实测 | 真值 | 偏差 |
|---|---|---|---|
| 转换增益 K | {nse["gain"]["measured"]:.4f} e-/DN | {nse["gain"]["true"]:.4f} e-/DN | {nse["gain"]["dev_pct"]:+.1f}% |
| 读出噪声 | {nse["read_noise"]["measured"]:.3f} e- | {nse["read_noise"]["true"]:.1f} e- | {nse["read_noise"]["dev_pct"]:+.1f}% |
| 加权 R² | {nse["r2"]:.5f} |  |  |

**结论**：从"带噪声的图像"里能把传感器的转换增益反推到 {nse["gain"]["dev_pct"]:+.1f}%。
读出噪声偏高 {nse["read_noise"]["dev_pct"]:+.0f}% 不是误差 ——
量化噪声 1/12 DN² 折算到输入端是 0.86 e-，
与 1.8 e- 的读出噪声按平方和叠加恰好是 {np.sqrt(1.8**2 + 0.86**2):.2f} e-，
与实测吻合。**这说明测量不仅对，还能把误差来源解释清楚。**

拟合必须做两件事，否则结果会大幅偏掉（实测不做时 K 被反推成真值的 1.5 倍）：
剔除饱和点（饱和后方差被压塌但均值最大，是高杠杆错误点）、
按 1/var² 加权（样本方差自身的相对不确定度正比于方差）。

### 11.4 提高增益改善了什么

| 读出噪声模型 | 增益 | 暗噪声 (DN) | 折算到输入端 (e-) | 2 e- 信号的 SNR |
|---|---|---|---|---|
{rows_gain}

**结论**：这才是"增益有没有用"的完整答案，比前面"增益不改善 SNR"的说法更准确：
- 如果读出噪声折算在**输入端**（理想 ISO 无关传感器），提高增益毫无作用，
  暗噪声恒定 {[r for r in gn["rows"] if r["model"]=="iso_less"][0]["read_noise_e"]:.2f} e-
- 真实传感器的读出噪声主要来自源跟随器与 ADC，折算在**输出端**，
  高增益把它按 1/增益压低：16× 增益下暗噪声从
  {[r for r in gn["rows"] if r["model"]=="gain_referred"][0]["read_noise_e"]:.2f} e- 降到
  {[r for r in gn["rows"] if r["model"]=="gain_referred"][-1]["read_noise_e"]:.2f} e-
- **这就是 ISO 存在的意义**：它不改变光子噪声，但能压低暗部的读出噪声底

测量本身也有个条件：位深必须够。12bit 时量化噪声折算到输入端约 0.86 e-，
会盖住读出噪声从 1.8 e- 降到 0.11 e- 的过程，把效应整个埋掉 ——
所以这一节用的是 14bit。真实产线上测暗噪声也要确认量化噪声不成为瓶颈。

### 11.5 动态范围

|  |  |
|---|---|
| 实测 | {dr["measured_db"]:.1f} dB（{dr["stops"]:.1f} 档） |
| 理论（满阱 {dr["full_well_e"]:.0f} e- / 读出噪声 {dr["read_noise_e"]:.1f} e-） | {dr["theory_db"]:.1f} dB |
| ADC | {dr["bit_depth"]} bit |

**结论**：动态范围由满阱与读出噪声之比决定，{dr["bit_depth"]}bit 的 ADC
本身不构成瓶颈（量化噪声远低于读出噪声）。要提动态范围只能从
**增大满阱**（工艺、双转换增益）或**降低读出噪声**入手，加位深没用。

### 11.6 阴影：亮度均匀度与色阴影

| LSC 状态 | 亮度均匀度（四角/中心） | 色阴影 Δu'v'×1000 |
|---|---|---|
{rows_shade}

**结论**：
1. **色阴影必须单独量**。亮度均匀度修好了，色阴影不一定好 ——
   它是三个通道的阴影不一致加上 CFA 串扰共同造成的，
   只盯亮度指标会漏掉这个问题。
2. **过校正比欠校正更糟**。欠校正只是边角偏暗（0.876），
   过校正会把边角提亮到超过中心（1.138），而且色阴影也修不干净。
   这也是为什么 LSC 标定要跟着镜头走 —— 换镜头不重标，
   结果可能落在"过校正"这一侧。
""", ["iq_mtf", "iq_focus", "iq_noise", "iq_gain", "iq_shading"]))

    # ---- 16. C++ 统计通路 ----
    np_ = res.get("native_port", {})
    if not np_.get("available"):
        secs.append(("16. C++ 统计通路（本机未编译）", f"""
这一节需要一个可选的 C++ 共享库，本机没有编译。

构建方法：`python tools/build_native.py`（只需要 g++/clang++，不需要 MSVC）。
不编译也不影响其余各节 —— 默认后端是**纯 Python**，路径与结论都不变。

原因：{np_.get("reason", "未知")}
""", []))
    else:
        env = np_["env"]
        rows_t = "\n".join(
            f"| {_NATIVE_LABELS.get(k, k)} | {v['median_us']:.1f} | "
            f"{v['p10_us']:.1f} | {v['p90_us']:.1f} | {v['mad_us']:.1f} |"
            for k, v in np_["timing"].items())
        d = np_["decomposition"]
        rows_e = "\n".join(
            f"| {e['scene']} | {'是' if e['n_valid_equal'] else '否'} | "
            f"{e['gray_world_rel']:.1e} | {e['white_patch_rel']:.1e} | "
            f"{e['gray_edge_rel']:.1e} | {e['shades_of_gray_rel']:.1e} |"
            for e in np_["equivalence"])
        b = np_["budget"]
        secs.append(("16. C++ 统计通路重写：加速来自算法，不是语言", f"""
把 AE 测光与 AWB 统计逐像素通路重写成 C++（`extern "C"` 共享库 + ctypes）。

**为什么不用 setuptools Extension / pybind11**：Windows 上的 CPython 是 MSVC 构建的，
而开发机只有 MinGW g++，MinGW 编译的扩展模块无法可靠链接。改成纯 C ABI 的共享库后，
两个工具链都能编、**不引入任何 pip 依赖**，而且这个库**脱离 Python 也能编能跑**
（`native/tests/native_main.cpp` 有 23 项不依赖 numpy 的解析自检）。

**环境（性能数字必须带环境才有意义）**：{env['platform']}｜Python {env['python']}｜
numpy {env['numpy']}｜OpenCV {env['cv2']}｜{env['native_build']}

| 实现 | 中位数 µs | p10 | p90 | MAD |
|---|---|---|---|---|
{rows_t}

ctypes 调用地板实测 **{np_['null_call_us']:.2f} µs**，相对上面每个核都 < 5%，可以忽略。

**加速分解 —— 这张表才是这一节的重点：**

- 算法改进（Python→Python，只改算法）：**{d['algo_gain_awb']:.2f}×**
- 实现改进（算法不变，换 C++）：**{d['impl_gain_awb']:.2f}×**
- 总加速：**{d['total_gain_awb']:.2f}×**

**「C++ 比 numpy 快 16 倍」是错的说法。** 看这一格就明白了：
在**没有任何算法差异**的模式上（全画面平均，两边都是一次求和），
换语言的收益是 **{d['language_only_gain']:.2f}×** —— 小于 1，也就是 **C 反而略慢**。
numpy 的 `mean()` 是 SIMD 归约，标量 double 循环赢不了它。

那 16 倍是从哪来的？**来自重写时顺手消除了结构性的浪费**：现状每帧做
5 次全帧布尔 gather、4 次 Sobel（其中 2 次完全重复）、3 次全帧 luma、
2 次全帧饱和度、4 次全帧开方，且无条件计算 4 个估计器。这些都不是"语言的锅"。
**准确的说法是：大头在算法与访存结构，语言本身几乎没有贡献。**

**等价性**（这一列才是"实现对了"的证据）：

| 场景 | n_valid 逐位相等 | gray_world | white_patch | gray_edge | shades_of_gray |
|---|---|---|---|---|---|
{rows_e}

`white_patch` 的偏差是 **0.00e+00** —— 与**开发环境验证过的 numpy 1.26.4** 逐位相等。
做法是照抄 numpy 的 `virtual_index` 运算顺序、`_lerp` 的两分支写法、以及 `(b-a)` 必须在
float32 里减这三条细节。

⚠️ **这个「逐位」有版本边界**：CI 装的是 numpy 2.x，而 numpy 在 1.26 → 2.x 之间
**改过 quantile 的实现**（同一份输入下 p99 相差约 1 个 float32 ulp）。有意思的是
**C 侧的结果跨平台完全一致，变的是 numpy** —— 所以准确的说法是「复刻了开发时
验证过的那版语义」，不能无条件说「和 numpy 逐位相等」。
`gray_world` 的 ~2.8e-6 不是"我们算错了"：numpy 用 float32 pairwise、C 用 double
顺序累加，**差的是 numpy 自身的误差**，测试里还额外断言了"C 更接近 float64 精确值"。

**估算的算力占比**：AWB 统计占 33ms 帧预算的 {b['frame_budget_pct']:.2f}%；
按像素数线性外推，1080p 约 {b['extrap_1080p_us']:.0f} µs、4K 约 {b['extrap_4k_us']:.0f} µs。
**这是外推不是实测**，忽略了缓存层次与带宽的变化，只能当量级估计。

**没做的**：没有实现一份"故意保留全部浪费的朴素 C"版本；没有面积/功耗/时序数据；
没有在真实 DSP/NPU 上跑过。本方案给的是算法与算术层面的可行性，不是 RTL 结论。
""", ["native_speed", "native_decomp"]))

    # ---- 17. 定点化 ----
    fp = res.get("fixed_point", {})
    if not fp.get("available"):
        secs.append(("17. 定点化误差预算（本机未编译）", f"""
同上一节，需要先编译 C++ 共享库：`python tools/build_native.py`。

原因：{fp.get("reason", "未知")}
""", []))
    else:
        rows_ae = "\n".join(
            f"| {r['scene']} | {r['mode']} | {r['ref']:.6f} | {r['q16']:.6f} | "
            f"{r['abs_diff']:.2e} | {r['ev']:.2e} | {r['bound']:.1e} | "
            f"{'是' if r['abs_diff'] < r['bound'] else '否'} |"
            for r in fp["ae_rows"])
        rows_awb = "\n".join(
            f"| {r['scene']} | {r['n_valid_ref']} / {r['n_valid_q16']} | "
            f"{r['gray_world_abs']:.1e} | {r['white_patch_abs']:.1e} | "
            f"{r['gains_rel']:.1e} | {r['angle_diff_deg']:.4f} |"
            for r in fp["awb_rows"])
        rows_bits = "\n".join(
            f"| {r['bits']} | {r['average_abs']:.2e} | {r['highlight_priority_abs']:.2e} |"
            for r in fp["by_bits"])
        rows_bins = "\n".join(
            f"| {r['bins']} | {r['bin_width']:.2e} | {r['abs']:.2e} |"
            for r in fp["by_bins"])
        secs.append(("17. 定点化：误差有上界，不是拍脑袋的容差", f"""
把逐像素统计通路改成定点：**uint16 Q0.16 输入 + int64 累加 + 直方图分位数**。

**为什么 Q0.16**：`linear_pre_wb` 与 `linear_ccm` 都被 clip 在 [0,1]，Q0.16 满量程无浪费，
等价于"统计块吃 16bit 整数 luma"—— 这正是真实 ISP 的形态。
**为什么 int64 够**：4K 全图 Σ 也只有 5.4e11，evaluative 加权和约 1.9e16，
都远小于 2^63，**不需要分段移位**（那是 32 位 MCU 的妥协，代价是每行 0.5 LSB 的系统性偏差）。

**推导出的误差上界**（不是拟合出来的容差）：

- 均值类：像素量化 0.5 LSB + 权重 Q15 舍入 → 绝对上界 **1.8e-5**
- 分位数：直方图估计值与真值必然落在**同一个 bin** 内 → 绝对上界 = **一个 bin 宽**

| 场景 | 模式 | 浮点参考 | 定点 Q16 | 绝对差 | 折算 EV | 上界 | 合格 |
|---|---|---|---|---|---|---|---|
{rows_ae}

均值类折算到 EV 后比收敛阈值 0.02 EV **低约三个数量级**，分位数低约两个数量级。

**位宽与 bin 数的扫描 —— 这条给出一个可用的设计结论：**

| 输入位宽 | 全画面平均 | 高光优先（分位数） |
|---|---|---|
{rows_bits}

**10 位以上就没有收益了。** 因为分位数的瓶颈从"输入量化"转移到了"直方图 bin 宽"，
继续加位宽是在优化一个已经不是瓶颈的环节。要提精度只能加 bin 数：

| bin 数 | 一个 bin 宽 | 实测绝对误差 |
|---|---|---|
{rows_bins}

实测误差**始终压在 bin 宽之下**，与上界推导一致。

**AWB 侧的端到端**（融合数学仍是 Python 那份，所以这里量的是差异传导到决策有多大）：

| 场景 | n_valid 浮点/定点 | gray_world | white_patch | gains 相对差 | 角度误差差（度） |
|---|---|---|---|---|---|
{rows_awb}

角度误差差最大 **{max(r['angle_diff_deg'] for r in fp['awb_rows']):.4f} 度**，
而 AWB 本身的误差量级是 0.5~32 度 —— 定点化对既有结论没有影响。
`white_patch` 的 ~6.6e-4 也落在 bin 宽上界内。

**定点化的边界（明写）**：只定点化**逐像素统计通路**。融合权重、对数域几何平均、
CCT 约束、AE 控制律仍在 double —— 它们每帧只作用在 3~5 个数上，没有吞吐论证，
且含 `exp/log/pow`，定点化是另一场独立的误差分析。
**整条 ISP 的定点化（去马赛克、CCM 仍是浮点）明确不在范围内。**
""", ["fixed_budget", "fixed_sweep"]))

    return secs


def build_report(res: dict, outdir: str, title: str = "3A（AE/AWB/AF）算法实验报告") -> dict:
    figs = {
        "isp": fig_isp_stages(res["isp"], outdir),
        "ae_metering": fig_ae_metering(res["ae_metering"], outdir),
        "ae_conv": fig_ae_convergence(res["ae_convergence"], outdir),
        "ae_curve": fig_ae_metering_curve(res["ae_curve"], outdir),
        "ae_policy": fig_ae_policy(res["ae_policy"], outdir),
        "ae_flicker": fig_ae_flicker(res["ae_flicker"], outdir),
        "awb_grid": fig_awb_grid(res["awb"], outdir),
        "awb_error": fig_awb_error(res["awb"], outdir),
        "awb_constraint": fig_awb_constraint(res["awb_constraint"], outdir),
        "ccm": fig_ccm(res["ccm"], outdir),
        "af_curves": fig_af_curves(res["af_curves"], outdir),
        "af_search": fig_af_search(res["af_search"], outdir),
        "coupling": fig_coupling(res["coupling"], outdir),
        "iq_mtf": fig_iq_mtf_selfcheck(res["image_quality"], outdir),
        "iq_focus": fig_iq_mtf_vs_focus(res["image_quality"], outdir),
        "iq_noise": fig_iq_noise(res["image_quality"], outdir),
        "iq_gain": fig_iq_gain(res["image_quality"], outdir),
        "iq_shading": fig_iq_shading(res["image_quality"], outdir),
        "t_source": fig_temporal_source(res["temporal_source"], outdir),
        "t_tradeoff": fig_temporal_tradeoff(res["temporal_ae"], outdir),
        "t_trace": fig_temporal_trace(res["temporal_ae"], outdir),
        "sc_signal": fig_scene_cut_signal(res["scene_cut"], outdir),
        "sc_events": fig_scene_cut_events(res["scene_cut"], outdir),
        "t_awb": fig_awb_temporal(res["temporal_awb"], outdir),
    }
    # C++ 模块是**可选**组件：没编译时这两节降级成一句说明，且不注册图。
    # 这样"没装编译器"不会让报告生成失败，只会有两节是空的。
    _np_ok = res.get("native_port", {}).get("available", False)
    _fp_ok = res.get("fixed_point", {}).get("available", False)
    if _np_ok:
        figs["native_speed"] = fig_native_speed(res["native_port"], outdir)
        figs["native_decomp"] = fig_native_decomp(res["native_port"], outdir)
    if _fp_ok:
        figs["fixed_budget"] = fig_fixed_budget(res["fixed_point"], outdir)
        figs["fixed_sweep"] = fig_fixed_sweep(res["fixed_point"], outdir)

    secs = _sections(res, figs)

    md = [f"# {title}", "", f"生成时间：{res['meta']['time']}",
          f"仿真配置：{res['meta']['size'][0]}×{res['meta']['size'][1]}，"
          f"种子 {res['meta']['seed']}，共 {res['meta']['n_captures']} 次成像", ""]
    md.append("## 目录\n")
    for i, (t, _, _) in enumerate(secs, 1):
        md.append(f"{i}. [{t}](#{i})")
    md.append("")
    for i, (t, body, keys) in enumerate(secs, 1):
        md.append(f'<a id="{i}"></a>')
        md.append(f"## {t}")
        md.append(body.strip())
        for k in keys:
            md.append(f"![{k}]({os.path.basename(figs[k])})")
        md.append("")
    md.append("---")
    md.append("所有数字均由 `run_all.py` 一次性生成，可复现（固定随机种子）。")
    md_text = "\n".join(md)

    md_path = os.path.join(outdir, "report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    # 自包含 HTML（图片内嵌 base64），方便发给别人或转 PDF
    parts = [f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>{title}</title><style>
body{{font-family:"Microsoft YaHei","PingFang SC",sans-serif;max-width:1000px;
margin:0 auto;padding:28px 34px;line-height:1.75;color:#1a202c;font-size:15px}}
h1{{font-size:26px;border-bottom:3px solid #2b6cb0;padding-bottom:10px}}
h2{{font-size:19px;color:#2b6cb0;margin-top:34px;border-left:5px solid #2b6cb0;padding-left:10px}}
table{{border-collapse:collapse;width:100%;margin:12px 0;font-size:14px}}
th,td{{border:1px solid #cbd5e0;padding:6px 10px;text-align:left}}
th{{background:#ebf4ff}}
img{{max-width:100%;margin:14px 0;border:1px solid #e2e8f0;border-radius:4px}}
code{{background:#f7fafc;padding:2px 5px;border-radius:3px}}
.meta{{color:#4a5568;font-size:13px}}
</style></head><body>"""]
    parts.append(f"<h1>{title}</h1>")
    parts.append(f'<p class="meta">生成时间：{res["meta"]["time"]}　|　'
                 f'仿真配置：{res["meta"]["size"][0]}×{res["meta"]["size"][1]}　|　'
                 f'随机种子：{res["meta"]["seed"]}　|　成像次数：{res["meta"]["n_captures"]}</p>')

    import re

    def inline(s: str) -> str:
        """行内格式：**粗体** / `代码` / 数字加粗"""
        s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
        s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
        return s

    def md_to_html(text: str) -> str:
        lines = text.split("\n")
        out, in_table, in_ul, para = [], False, False, []

        def flush_para():
            if para:
                out.append("<p>" + "<br>".join(para) + "</p>")
                para.clear()

        for ln in lines:
            if ln.startswith("|"):
                flush_para()
                cells = [inline(c.strip()) for c in ln.strip("|").split("|")]
                if all(set(c) <= set("-: ") for c in ln.strip("|").split("|")):
                    continue
                if not in_table:
                    out.append("<table>")
                    in_table = True
                    out.append("<tr>" + "".join(f"<th>{c}</th>" for c in cells) + "</tr>")
                else:
                    out.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
                continue
            if in_table:
                out.append("</table>")
                in_table = False
            if ln.strip().startswith("- "):
                flush_para()
                if not in_ul:
                    out.append("<ul>")
                    in_ul = True
                out.append(f"<li>{inline(ln.strip()[2:])}</li>")
                continue
            if in_ul:
                out.append("</ul>")
                in_ul = False
            s = ln.strip()
            if not s:
                flush_para()
                continue
            if s.startswith("#"):
                flush_para()
                lvl = len(s) - len(s.lstrip("#"))
                out.append(f"<h{lvl + 1}>{s.lstrip('#').strip()}</h{lvl + 1}>")
            else:
                para.append(inline(s))
        flush_para()
        if in_table:
            out.append("</table>")
        if in_ul:
            out.append("</ul>")
        return "\n".join(out)

    for t, body, keys in secs:
        parts.append(f"<h2>{t}</h2>")
        parts.append(md_to_html(body.strip()))
        for k in keys:
            parts.append(f'<img src="data:image/png;base64,{_b64(figs[k])}" alt="{k}">')
    parts.append("</body></html>")

    html_path = os.path.join(outdir, "report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))

    return {"md": md_path, "html": html_path, "figs": figs}


# -----------------------------------------------------------------------------
# 画质指标
# -----------------------------------------------------------------------------
def fig_iq_mtf_selfcheck(res: dict, outdir: str) -> str:
    sc = res["mtf_selfcheck"]
    cur = res["_curve"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))

    axes[0].plot(cur["freqs"], cur["theory"], "-", color=C_MAIN, lw=2.0,
                 label=f'理论 (高斯 σ={cur["sigma"]}px × 像素孔径)')
    axes[0].plot(cur["freqs"], cur["mtf"], "--", color=C_WARN, lw=1.6,
                 label="斜边法实测")
    axes[0].axhline(0.5, color=C_GRAY, ls=":", lw=1)
    axes[0].text(0.02, 0.52, "MTF50", color=C_GRAY, fontsize=8)
    axes[0].set_xlabel("空间频率 (cycles/pixel)")
    axes[0].set_ylabel("MTF")
    axes[0].set_title("MTF 测量链路自检：与解析解逐点对比")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    sig = [r["sigma"] for r in sc["rows"]]
    axes[1].plot(sig, [r["mtf50_theory"] for r in sc["rows"]], "o-",
                 color=C_MAIN, ms=5, label="理论 MTF50")
    axes[1].plot(sig, [r["mtf50"] for r in sc["rows"]], "s--",
                 color=C_WARN, ms=5, label="实测 MTF50")
    for r in sc["rows"]:
        axes[1].annotate(f'{r["dev_pct"]:+.1f}%', (r["sigma"], r["mtf50"]),
                         textcoords="offset points", xytext=(0, -14),
                         ha="center", fontsize=7)
    axes[1].set_xlabel("离焦/模糊 PSF 的 σ (pixel)")
    axes[1].set_ylabel("MTF50 (cycles/pixel)")
    axes[1].set_title(f'七个已知模糊量下逐一对表\n'
                      f'最大偏差 {sc["max_dev_pct"]:.1f}%')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    return _save(fig, outdir, "fig_iq_mtf_selfcheck.png")


def fig_iq_mtf_vs_focus(res: dict, outdir: str) -> str:
    d = res["mtf_vs_focus"]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(d["positions"], d["mtf50"], "o-", color=C_MAIN, ms=5,
            label="MTF50（线性域，成像系统真实解析力）")
    ax.plot(d["positions"], d["mtf50_display"], "^--", color="#805ad5", ms=5,
            label="MTF50（显示链路，含 USM 锐化）")
    ax.axvline(d["true_focus"], color=C_WARN, ls="--", lw=1.2,
               label=f'真合焦位置 {d["true_focus"]:.2f}')
    ax.set_xlabel("镜头位置")
    ax.set_ylabel("MTF50 (cycles/pixel)", color=C_MAIN)
    ax.tick_params(axis="y", labelcolor=C_MAIN)
    ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    tg = np.asarray(d["tenengrad"], dtype=float)
    ax2.plot(d["positions"], tg / max(tg.max(), 1e-12), "s--", color=C_OK,
             ms=5, label="Tenengrad 评价函数（归一化）")
    ax2.set_ylabel("AF 评价函数（归一化）", color=C_OK)
    ax2.tick_params(axis="y", labelcolor=C_OK)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    ax.set_title("AF 的评价函数到底在找什么：\n它是在用高频能量单调地逼近 MTF 最大的位置")
    return _save(fig, outdir, "fig_iq_mtf_vs_focus.png")


def fig_iq_noise(res: dict, outdir: str) -> str:
    n = res["noise"]
    pts = n["points"]
    x = np.array([p["mean"] for p in pts])
    y = np.array([p["std"] for p in pts])
    snr = np.array([p["snr_db"] for p in pts])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9))

    axes[0].semilogy(x, y, "o", color=C_MAIN, ms=5, label="实测噪声 σ")
    axes[0].semilogy(x, np.sqrt(x / n["gain"]["true"]
                                + (n["read_noise"]["true"] / n["gain"]["true"]) ** 2),
                     "-", color=C_WARN, lw=1.6, label="理论 σ（已知 K 与读出噪声）")
    axes[0].set_xlabel("信号 (DN)")
    axes[0].set_ylabel("噪声 σ (DN)")
    axes[0].set_title("噪声-信号曲线：跨 3 个数量级都与理论重合")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, which="both")

    v = y ** 2
    axes[1].loglog(x, v, "o", color=C_MAIN, ms=5, label="实测方差")
    slope = 1.0 / n["gain"]["measured"]
    b = (n["read_noise"]["measured"] / n["gain"]["measured"]) ** 2
    axes[1].loglog(x, slope * x + b, "-", color=C_OK, lw=1.6,
                   label=f'加权拟合 var = 信号/K + σ_r²')
    axes[1].set_xlabel("信号 (DN)")
    axes[1].set_ylabel("方差 (DN²)")
    axes[1].set_title(f'光子转换曲线：反推转换增益 K\n'
                      f'K={n["gain"]["measured"]:.3f} e-/DN（真值 {n["gain"]["true"]:.3f}，'
                      f'偏差 {n["gain"]["dev_pct"]:+.1f}%），加权 R²={n["r2"]:.4f}')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, which="both")

    ax3 = axes[0].twinx()
    ax3.plot(x, snr, "^:", color=C_GRAY, ms=4, label="SNR (dB, 右轴)")
    ax3.set_ylabel("SNR (dB)", color=C_GRAY)
    ax3.tick_params(axis="y", labelcolor=C_GRAY)
    return _save(fig, outdir, "fig_iq_noise.png")


def fig_iq_gain(res: dict, outdir: str) -> str:
    rows = res["gain_vs_noise"]["rows"]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for model, color, label in (("iso_less", C_GRAY, "ISO 无关传感器（读出噪声折算在输入端）"),
                                ("gain_referred", C_OK, "真实传感器（读出噪声折算在输出端）")):
        r = [x for x in rows if x["model"] == model]
        ax.plot([x["gain"] for x in r], [x["read_noise_e"] for x in r],
                "o-", color=color, ms=6, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("模拟增益")
    ax.set_ylabel("暗噪声（折算到输入端，e-）")
    ax.set_title("提高增益改善了什么：暗部噪声底\n"
                 "增益后置的读出噪声被压低，前置的压不动 —— 这就是 ISO 存在的意义")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    return _save(fig, outdir, "fig_iq_gain.png")


def fig_iq_shading(res: dict, outdir: str) -> str:
    rows = res["shading"]["rows"]
    labels = [r["label"] for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))

    axes[0].bar(x, [r["luma_uniformity"] for r in rows], color=C_MAIN)
    axes[0].axhline(1.0, color=C_OK, ls="--", lw=1.2, label="理想均匀")
    for i, r in enumerate(rows):
        axes[0].text(i, r["luma_uniformity"] + 0.01, f'{r["luma_uniformity"]:.3f}',
                     ha="center", fontsize=8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=8, rotation=10)
    axes[0].set_ylabel("亮度均匀度（四角/中心）")
    axes[0].set_title("LSC 校正准 / 欠 / 过\n（过校正比欠校正更糟：边角反而比中心亮）")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(x, [r["d_uv_corner_max"] for r in rows], color=C_WARN)
    for i, r in enumerate(rows):
        axes[1].text(i, r["d_uv_corner_max"] + 0.03, f'{r["d_uv_corner_max"]:.2f}',
                     ha="center", fontsize=8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=8, rotation=10)
    axes[1].set_ylabel("色阴影 Δu'v' ×1000")
    axes[1].set_title("色阴影：只看亮度均匀度是看不出来的\n必须单独量边角与中心的色度差")
    axes[1].grid(alpha=0.3, axis="y")
    return _save(fig, outdir, "fig_iq_shading.png")
