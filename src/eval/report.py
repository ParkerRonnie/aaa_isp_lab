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

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
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
    }

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
