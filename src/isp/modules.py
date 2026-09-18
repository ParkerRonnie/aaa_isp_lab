# -*- coding: utf-8 -*-
"""ISP 各处理模块。顺序即真实 ISP 的典型顺序：

    BLC -> LSC -> 去马赛克 -> AWB -> CCM -> 色调映射 -> 降噪/锐化

这个顺序不是随意的，每一步都有物理/工程理由（README 里有说明）：
  - BLC/LSC 必须在去马赛克前做（否则颜色插值会放大黑电平和阴影误差）
  - AWB 必须在 CCM 前做（CCM 是在白平衡后的色彩空间上标定的）
  - 色调映射必须在最后（它是非线性压缩，放前面会破坏线性域的运算）
"""
import numpy as np
import cv2

from ..color_science import linear_to_srgb, linear_to_gamma22


# -----------------------------------------------------------------------------
# 黑电平
# -----------------------------------------------------------------------------
def black_level_correct(raw: np.ndarray, black_level: float, signal_dn: float) -> np.ndarray:
    """减去光学黑电平并归一化到 [0,1]。

    黑电平来自暗电流与读出电路偏置，是"0 光"时传感器的输出。
    不扣除会导致暗部发灰、AWB 灰世界统计被整体抬高的偏置污染。
    """
    lin = (raw.astype(np.float32) - float(black_level)) / float(signal_dn)
    return np.clip(lin, 0.0, 1.0)


# -----------------------------------------------------------------------------
# 镜头阴影校正
# -----------------------------------------------------------------------------
def lsc_radial_model(h: int, w: int, assumed_strength: float,
                     color_dependency: float = 0.15,
                     max_gain: float = 3.0) -> np.ndarray:
    """LSC 增益表（这里用解析式代替真实的标定表）。

    assumed_strength 是**假设**的衰减强度，可以和镜头的真实衰减不一致 ——
    这就复现了真实工程中最常见的问题：LSC 标定不准 / 换镜头未重标，
    导致边缘亮度与色彩出现残余偏差。
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2) / np.sqrt(2.0)
    base = 1.0 - assumed_strength * (r ** 2)
    gain = np.empty((h, w, 3), dtype=np.float32)
    for ch in range(3):
        delta = color_dependency * (ch - 1) * 0.15 * (r ** 2)
        gain[..., ch] = 1.0 / np.clip(base + delta, 1e-3, 1.0)
    return np.clip(gain, 1.0, max_gain)


def apply_lsc(linear: np.ndarray, gain_map: np.ndarray) -> np.ndarray:
    """RGB 域补偿（若 LSC 放在去马赛克之后）"""
    return linear * gain_map


def apply_lsc_bayer(bayer: np.ndarray, gain_map: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    """RAW 域补偿：每个像素按其所属 CFA 通道取对应的增益。

    真实 ISP 的 LSC 一定在 RAW 域做（在建 Raw 之前），因为 shading 是
    光子层面的衰减，且各通道衰减不同；放到去马赛克之后补偿会把
    已经混入的颜色误差一起放大。
    """
    g = np.empty_like(bayer)
    for i in range(2):
        for j in range(2):
            g[i::2, j::2] = gain_map[i::2, j::2, pattern[i, j]]
    return bayer * g


# -----------------------------------------------------------------------------
# 去马赛克
# -----------------------------------------------------------------------------
def _channel_masks(pattern: np.ndarray, shape) -> np.ndarray:
    masks = np.zeros((3,) + shape, dtype=np.float32)
    for i in range(2):
        for j in range(2):
            masks[pattern[i, j], i::2, j::2] = 1.0
    return masks


def _norm_conv(values: np.ndarray, mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """归一化卷积：只在 mask=1 的位置取样本做加权插值。

    除以卷积后的 mask 是为了处理边缘和非均匀采样密度
    （R/B 的采样点数只有 G 的 1/2，直接卷积会导致权重错误）。
    """
    num = cv2.filter2D(values * mask, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    den = cv2.filter2D(mask, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    return num / np.maximum(den, 1e-6)


_K_RB = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 4.0
_K_G = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], dtype=np.float32) / 4.0


def demosaic_bilinear(raw: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    """逐通道双线性插值。速度快，但高频处会出现明显的伪彩（拉链纹）。"""
    masks = _channel_masks(pattern, raw.shape)
    out = np.empty(raw.shape + (3,), dtype=np.float32)
    for ch in range(3):
        out[..., ch] = _norm_conv(raw, masks[ch], _K_G if ch == 1 else _K_RB)
    return np.clip(out, 0.0, 1.0)


def demosaic_color_diff(raw: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    """色差插值（绿通道引导）。

    先插 G（G 采样最密、最可靠），再在 R-G / B-G 的色差域做插值。
    色差在局部近似常数，所以插值误差远小于直接插 R/B，
    伪彩和拉链纹显著减少 —— 这是比双线性更好的经典做法。
    """
    masks = _channel_masks(pattern, raw.shape)
    g = _norm_conv(raw, masks[1], _K_G)

    out = np.empty(raw.shape + (3,), dtype=np.float32)
    out[..., 1] = g
    for ch in (0, 2):
        d = (raw - g) * masks[ch]                       # 只在 R(或B) 位置有值
        d_full = _norm_conv(d, masks[ch], _K_RB)        # 色差域插值
        out[..., ch] = g + d_full
    return np.clip(out, 0.0, 1.0)


def demosaic(raw: np.ndarray, pattern: np.ndarray, method: str = "color_diff") -> np.ndarray:
    if method == "bilinear":
        return demosaic_bilinear(raw, pattern)
    if method == "color_diff":
        return demosaic_color_diff(raw, pattern)
    raise ValueError(f"unknown demosaic method: {method}")


# -----------------------------------------------------------------------------
# 白平衡 / CCM
# -----------------------------------------------------------------------------
def apply_wb(linear: np.ndarray, gains: np.ndarray) -> np.ndarray:
    return np.clip(linear * np.asarray(gains, dtype=np.float32)[None, None, :], 0.0, 1.0)


def apply_ccm(linear: np.ndarray, ccm: np.ndarray) -> np.ndarray:
    ccm = np.asarray(ccm, dtype=np.float32)
    return np.clip(linear @ ccm.T, 0.0, 1.0)


def solve_ccm(src: np.ndarray, dst: np.ndarray, preserve_white: bool = True) -> np.ndarray:
    """由色卡标定 CCM：最小二乘求解 3x3 矩阵。

    src: (N,3) 实测线性 RGB（白平衡后）
    dst: (N,3) 目标线性 RGB（色卡参考值）
    preserve_white: 约束行和为 1，保证白色不偏色（工程上通常强制）
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if preserve_white:
        # 增广：把"白点必须映射到白点"作为强约束加进去（大权重）
        W = 50.0
        src = np.vstack([src, W * np.ones((1, 3))])
        dst = np.vstack([dst, W * np.ones((1, 3))])
    ccm, *_ = np.linalg.lstsq(src, dst, rcond=None)
    return ccm.T.astype(np.float32)


# -----------------------------------------------------------------------------
# 色调映射
# -----------------------------------------------------------------------------
def tone_map(linear: np.ndarray, mode: str = "srgb", shoulder: float = 1.0,
             contrast: float = 1.0) -> np.ndarray:
    """线性 -> 显示域。

    shoulder < 1 时启用高光肩部压缩（指数型软拐点，C1 连续），
    否则超过 1.0 的部分被直接硬截断（表现为死白、无层次）。
    contrast 是 S 曲线强度，用于单独观察对比度对 AE 测光的影响。
    """
    x = np.clip(linear, 0.0, None)
    if shoulder < 1.0:
        k = float(shoulder)
        x = np.where(x > k, k + (1.0 - k) * (1.0 - np.exp(-(x - k) / (1.0 - k))), x)
    y = linear_to_gamma22(x) if mode == "gamma22" else linear_to_srgb(x)
    if abs(contrast - 1.0) > 1e-6:
        s = y * y * (3.0 - 2.0 * y)          # smoothstep，单调
        y = np.clip(y + (s - y) * (contrast - 1.0), 0.0, 1.0)
    return np.clip(y, 0.0, 1.0)


# -----------------------------------------------------------------------------
# 降噪 / 锐化（显示域，等效于 YUV 通路上的处理）
# -----------------------------------------------------------------------------
def denoise_sharpen(srgb_u8: np.ndarray, enable_denoise: bool = True,
                    denoise_sigma: float = 0.35, enable_sharpen: bool = True,
                    sharpen_amount: float = 0.45) -> np.ndarray:
    out = srgb_u8
    if enable_denoise:
        # 双边滤波：保边降噪，避免把噪声当细节锐化放大
        out = cv2.bilateralFilter(out, 5, max(1.0, 255.0 * denoise_sigma), 3.0)
    if enable_sharpen:
        blur = cv2.GaussianBlur(out, (0, 0), 1.0)
        out = cv2.addWeighted(out, 1.0 + sharpen_amount, blur, -sharpen_amount, 0)
    return np.clip(out, 0, 255).astype(np.uint8)
