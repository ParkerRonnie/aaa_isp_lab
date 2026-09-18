# -*- coding: utf-8 -*-
"""AWB（自动白平衡）。

所有白平衡算法的本质都是同一个假设的不同形式：
    "画面里存在某种统计意义上的中性（灰色）内容"
差别只在于怎么定义"中性"、以及在假设不成立时怎么退化。

    Gray World      全画面均值是灰的         —— 大面积单色场景必崩
    White Patch     最亮的点是白的           —— 有高光/过曝时必崩
    Gray Edge       灰像素的梯度是灰的       —— 弱纹理场景样本不足
    Shades-of-Gray  均值的高阶矩是灰的       —— p 越大越接近 White Patch
    Fusion（本项目）用场景统计量给上面几家加权 —— 单家失效时自动降权

最后把估计光源约束到普朗克轨迹附近（色温先验）：绝大多数真实光源
（日光/白炽灯/荧光灯/LED）都靠近黑体轨迹，这个先验能救回单家算法
在大面积单色场景下的严重失效。
"""
from dataclasses import dataclass, field
import numpy as np
import cv2

from ..config import AWBConfig
from ..color_science import (blackbody_linear_rgb, rgb_linear_to_xyz, xy_to_cct,
                             cct_to_xy, illuminant_angle_deg)


@dataclass
class AWBResult:
    illum_rgb: np.ndarray          # 估计光源（G 归一化为 1）
    gains: np.ndarray              # 白平衡增益（G 归一化为 1）
    method: str
    cct: float = 0.0               # 估计色温
    weights: dict = field(default_factory=dict)
    detail: dict = field(default_factory=dict)


# -----------------------------------------------------------------------------
# 有效像素筛选
# -----------------------------------------------------------------------------
def valid_mask(linear: np.ndarray, cfg: AWBConfig) -> np.ndarray:
    """排除过曝像素与过暗像素。

    过曝像素已经失去颜色信息（三个通道都被截断），参与统计只会
    把估计光源往白色拉；过暗像素则以噪声为主，SNR 太低。
    """
    luma = 0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2]
    not_clip = np.max(linear, axis=2) < 0.98
    not_dark = luma > cfg.near_gray_val_min
    return not_clip & not_dark


def _saturation(linear: np.ndarray) -> np.ndarray:
    """饱和度 (max-min)/max，同时支持 (h,w,3) 与 (N,3) 两种输入"""
    mx = np.max(linear, axis=-1)
    mn = np.min(linear, axis=-1)
    return (mx - mn) / np.maximum(mx, 1e-6)


# -----------------------------------------------------------------------------
# 各类估计器
# -----------------------------------------------------------------------------
def gray_world(linear: np.ndarray, mask: np.ndarray, cfg: AWBConfig) -> np.ndarray:
    px = linear[mask]
    if px.shape[0] < 16:
        return np.ones(3)
    return px.mean(axis=0)


def white_patch(linear: np.ndarray, mask: np.ndarray, cfg: AWBConfig) -> np.ndarray:
    px = linear[mask]
    if px.shape[0] < 16:
        return np.ones(3)
    # 用 99.5 分位而非单点最大值：对孤立坏点/噪声更稳
    return np.percentile(px, 99.5, axis=0)


def gray_edge(linear: np.ndarray, mask: np.ndarray, cfg: AWBConfig) -> np.ndarray:
    """灰边法：只在"有梯度但颜色接近中性"的像素上统计。

    这部分像素最接近"白纸上的阴影/灰面"，是白点最可靠的来源。
    """
    gray = (0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2])
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    keep = mask & (mag > cfg.gray_edge_thresh) & (_saturation(linear) < cfg.near_gray_sat_max)
    px = linear[keep]
    if px.shape[0] < 16:
        return np.ones(3)
    return px.mean(axis=0)


def shades_of_gray(linear: np.ndarray, mask: np.ndarray, cfg: AWBConfig) -> np.ndarray:
    px = np.clip(linear[mask], 1e-6, None)
    if px.shape[0] < 16:
        return np.ones(3)
    p = cfg.sog_p
    return (np.mean(px ** p, axis=0)) ** (1.0 / p)


# -----------------------------------------------------------------------------
# 色温先验约束
# -----------------------------------------------------------------------------
def constrain_to_planckian(illum: np.ndarray, cfg: AWBConfig, win: float = 3.0):
    """把估计光源拉回 [cct_min, cct_max] 的色温范围内。

    做法是"保留 Duv、只截断色温"：先求该点到黑体轨迹的偏差向量，
    色温越界时把轨迹上的基准点移到边界，再加上原来的偏差向量。
    这样只限制色温这一个自由度，不会把估计结果整体抹平。
    """
    xyz = rgb_linear_to_xyz(np.clip(illum, 0, None))
    s = float(xyz.sum())
    if s <= 1e-9:
        return illum, 0.0
    x, y = float(xyz[0] / s), float(xyz[1] / s)
    cct = xy_to_cct(x, y)
    # 估计点跑到轨迹外很远时，McCamy 近似会给出 0 或负值。
    # 这不是"算不出来"，而是"离真实光源非常远"的信号 —— 必须按极端越界处理，
    # 直接返回原值等于在最需要约束的时候放弃了约束。
    if not np.isfinite(cct) or cct <= 0:
        cct = cfg.cct_min * 0.5

    cct_c = float(np.clip(cct, cfg.cct_min, cfg.cct_max))
    if abs(cct_c - cct) < 1e-6:
        return illum / illum[1], cct

    # 越界程度 -> 拉回强度（越界越多拉得越狠，边界处平滑过渡，避免闪烁）
    if cct < cfg.cct_min:
        over = (cfg.cct_min - cct) / cfg.cct_min
    else:
        over = (cct - cfg.cct_max) / cfg.cct_max
    t = float(np.clip(over / 0.15, 0.0, 1.0))
    t = float(np.clip(t * win / 3.0, 0.0, 1.0))

    # 轨迹上的对应点，以及与它的偏差（Duv 方向）
    x0, y0 = cct_to_xy(cct)
    x1, y1 = cct_to_xy(cct_c)
    dx, dy = x - x0, y - y0
    x_new, y_new = x1 + dx, y1 + dy
    y_new = float(np.clip(y_new, 0.05, 0.90))

    # 由 xy 反推 RGB 方向（固定 Y=1）
    from ..color_science import M_XYZ2SRGB
    xyz_new = np.array([x_new / y_new, 1.0, (1 - x_new - y_new) / y_new])
    rgb = M_XYZ2SRGB @ xyz_new
    rgb = np.clip(rgb, 1e-6, None)
    rgb = rgb / rgb[1]

    cur = np.clip(illum / illum[1], 1e-6, None)
    out = np.exp((1 - t) * np.log(cur) + t * np.log(rgb))
    return out / out[1], cct


# -----------------------------------------------------------------------------
# 主估计器
# -----------------------------------------------------------------------------
class AWBEstimator:
    def __init__(self, cfg: AWBConfig = None):
        self.cfg = cfg or AWBConfig()

    def estimate(self, linear_pre_wb: np.ndarray, clipped_ratio: float = 0.0) -> AWBResult:
        cfg = self.cfg
        mask = valid_mask(linear_pre_wb, cfg)
        n_valid = int(mask.sum())
        if n_valid < 16:
            return AWBResult(illum_rgb=np.ones(3), gains=np.ones(3),
                             method=cfg.method, detail={"fallback": "有效像素不足"})

        sat = _saturation(linear_pre_wb[mask])
        sat_mean = float(np.mean(sat))

        est = {
            "gray_world": gray_world(linear_pre_wb, mask, cfg),
            "white_patch": white_patch(linear_pre_wb, mask, cfg),
            "gray_edge": gray_edge(linear_pre_wb, mask, cfg),
            "shades_of_gray": shades_of_gray(linear_pre_wb, mask, cfg),
        }

        if cfg.method != "fusion":
            illum = est[cfg.method]
        else:
            # --- 置信度：每种算法在什么场景下不可信 ---
            # 灰世界：画面越彩，灰世界假设越不成立
            w_gw = float(np.clip(np.exp(-2.5 * sat_mean), 0.05, 1.0))
            # 白块：一旦有像素过曝，最亮点已经不是中性面（可能是高光/光源）
            w_wp = float(np.clip(1.0 - clipped_ratio / max(cfg.clip_guard, 1e-6), 0.02, 1.0))
            # 灰边：样本数不足时不可信
            gx = cv2.Sobel(0.2126 * linear_pre_wb[..., 0] + 0.7152 * linear_pre_wb[..., 1]
                           + 0.0722 * linear_pre_wb[..., 2], cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(0.2126 * linear_pre_wb[..., 0] + 0.7152 * linear_pre_wb[..., 1]
                           + 0.0722 * linear_pre_wb[..., 2], cv2.CV_32F, 0, 1, ksize=3)
            mag = np.sqrt(gx * gx + gy * gy)
            frac_ge = float(np.mean(mask & (mag > cfg.gray_edge_thresh)
                                    & (_saturation(linear_pre_wb) < cfg.near_gray_sat_max)))
            w_ge = float(np.clip(frac_ge / 0.03, 0.0, 1.0))

            base = np.array(cfg.fusion_weights, dtype=np.float64)
            w = np.array([w_gw, w_ge, w_wp]) * base
            if w.sum() <= 1e-9:
                w = base.copy()
            w = w / w.sum()

            # 在对数域做加权几何平均：光源是乘性量，算术平均会被异常值带偏
            stack = np.stack([est["gray_world"] / est["gray_world"][1],
                              est["gray_edge"] / est["gray_edge"][1],
                              est["white_patch"] / est["white_patch"][1]], axis=0)
            illum = np.exp(np.sum(w[:, None] * np.log(np.clip(stack, 1e-6, None)), axis=0))
            illum = illum / illum[1]

            est["_weights"] = {"gray_world": float(w[0]), "gray_edge": float(w[1]),
                               "white_patch": float(w[2]), "sat_mean": sat_mean,
                               "gray_edge_frac": frac_ge}

        illum = np.clip(illum, 1e-6, None)
        illum = illum / illum[1]

        raw_cct = 0.0
        if cfg.constrain_planckian:
            illum, raw_cct = constrain_to_planckian(illum, cfg)
            illum = illum / illum[1]

        gains = 1.0 / illum
        gains = gains / gains[1]

        xyz = rgb_linear_to_xyz(illum)
        cct = xy_to_cct(float(xyz[0] / xyz.sum()), float(xyz[1] / xyz.sum()))

        return AWBResult(illum_rgb=illum.astype(np.float32),
                         gains=gains.astype(np.float32),
                         method=cfg.method, cct=cct,
                         weights=est.get("_weights", {}),
                         detail={"estimators": {k: v.tolist() for k, v in est.items()
                                                if not k.startswith("_")},
                                 "n_valid": n_valid, "raw_cct": raw_cct})


def ideal_gains(true_temp_k: float) -> np.ndarray:
    """真值白平衡增益（用于计算"距离完美还有多远"）"""
    illum = blackbody_linear_rgb(true_temp_k)
    g = 1.0 / illum
    return (g / g[1]).astype(np.float32)


def illuminant_error_deg(est_illum: np.ndarray, true_temp_k: float) -> float:
    return illuminant_angle_deg(est_illum, blackbody_linear_rgb(true_temp_k))
