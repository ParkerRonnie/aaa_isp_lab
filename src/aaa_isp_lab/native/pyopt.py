# -*- coding: utf-8 -*-
"""纯 Python 的**算法优化版**统计通路 —— 2×2 因子对照里的 `py_opt` 那一格。

它存在的唯一目的是回答一个问题：**改进来自算法，还是来自换语言？**

做法是：和 `aaa/ae.py`、`aaa/awb.py` 里的现状**同一种语言**，只改算法：
  - AWB：把 5 次全帧布尔 gather 合成 1 次掩码复用；4 次 Sobel 合成 1 次；
        `mag > t` 改成 `gx²+gy² > t²`（省掉 4 次全帧 sqrt）；按需算估计器。
  - AE：center 模式的权重图**缓存**（原实现每帧重建整幅 h×w 权重图、
        每像素一次 exp）。
数值结果必须与现状**在容差内一致**（由 tests/test_native.py 钉住）。

不要用这个模块替换生产路径 —— 它只服务于性能对照实验。
"""
import cv2
import numpy as np

from ..aaa.awb import (AWBStatistics, _saturation, gray_edge, gray_world,
                       shades_of_gray, valid_mask, white_patch)

# --- AE：缓存中心权重图（原实现每帧重建整幅 h×w 权重图 + h×w 次 exp）---
_CENTER_CACHE = {}


def center_weight_cached(h, w, ratio):
    key = (h, w, round(float(ratio), 6))
    wmap = _CENTER_CACHE.get(key)
    if wmap is None:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        r2 = ((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2
        wmap = np.exp(-r2 / (2 * 0.35 ** 2 * max(ratio, 1e-3)))
        if len(_CENTER_CACHE) > 8:
            _CENTER_CACHE.clear()
        _CENTER_CACHE[key] = wmap
    return wmap


def metering_metric_opt(frame, cfg, _unused=None):
    """与 `ae.metering_metric` 同签名；唯一实质改动是 center 的权重表缓存。"""
    from ..aaa.ae import _HIGHLIGHT_TARGET, _METER_LABELS
    from ..color_science import linear_to_srgb

    luma = frame.luma_linear.astype(np.float32)
    h, w = luma.shape
    if cfg.metering == "center":
        wmap = center_weight_cached(h, w, cfg.center_weight)
        metric = float((luma * wmap).sum() / wmap.sum())
        target = cfg.target_linear
        detail = {"mode": _METER_LABELS["center"]}
    else:
        # 其余模式没有可改的算法，直接走原实现（对照才有意义）
        from ..aaa.ae import metering_metric
        return metering_metric(frame, cfg)

    detail["code_value"] = float(linear_to_srgb(np.array(metric)) * 255.0)
    detail["clip_ratio"] = frame.clipped_ratio
    return {"metric": metric, "target": target, "detail": detail}


def compute_statistics_opt(linear, cfg):
    """AWB 统计的算法优化版（与 `awb.compute_statistics` 同返回结构）。

    优化点，逐条对应原实现的浪费：
      - 5 次 `linear[mask]` 布尔 gather  -> 0（掩码只算一次，各估计器复用）
      - 4 次 cv2.Sobel                   -> 1
      - 4 次全帧 np.sqrt                 -> 0（比较平方）
      - 2 次全帧 _saturation             -> 1
      - 无条件算 4 个估计器              -> 按需
    """
    mask = valid_mask(linear, cfg)
    n_valid = int(mask.sum())
    if n_valid < 16:
        return AWBStatistics(estimators={}, n_valid=n_valid, sat_mean=0.0)

    px = linear[mask]                      # 唯一的一次 gather，各估计器复用
    sat = _saturation(px)
    sat_mean = float(np.mean(sat))

    need = (cfg.method == "fusion") or (cfg.method in ("gray_world", "white_patch",
                                                       "gray_edge", "shades_of_gray"))
    want = {"gray_world": False, "white_patch": False, "gray_edge": False,
            "shades_of_gray": False}
    if cfg.method == "fusion":
        want.update({"gray_world": True, "white_patch": True, "gray_edge": True})
    else:
        want[cfg.method] = True

    est = {}
    if want["gray_world"]:
        est["gray_world"] = px.mean(axis=0) if px.shape[0] >= 16 else np.ones(3)
    if want["white_patch"]:
        est["white_patch"] = (np.percentile(px, 99.5, axis=0)
                              if px.shape[0] >= 16 else np.ones(3))
    if want["shades_of_gray"]:
        p = cfg.sog_p
        est["shades_of_gray"] = (np.mean(np.clip(px, 1e-6, None) ** p, axis=0)) ** (1.0 / p)

    frac_ge = 0.0
    if want["gray_edge"] or cfg.method == "fusion":
        # **一次** Sobel：原来 gray_edge 里 2 次 + 融合块里 2 次 = 4 次
        gray = 0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2]
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag2 = gx * gx + gy * gy            # 不做 sqrt
        t2 = cfg.gray_edge_thresh ** 2
        sat_ok = (_saturation(linear) < cfg.near_gray_sat_max)
        strong = mag2 > t2
        if want["gray_edge"]:
            keep = mask & strong & sat_ok
            kp = linear[keep]
            est["gray_edge"] = kp.mean(axis=0) if kp.shape[0] >= 16 else np.ones(3)
        if cfg.method == "fusion":
            frac_ge = float(np.mean(mask & strong & sat_ok))

    # 保证与现状相同的键集合（report 的 detail["estimators"] 要四个键）
    for k, ones in (("gray_world", True), ("white_patch", True),
                    ("gray_edge", True), ("shades_of_gray", True)):
        if k not in est:
            est[k] = np.ones(3)

    return AWBStatistics(estimators=est, n_valid=n_valid,
                         sat_mean=sat_mean, frac_ge=frac_ge)
