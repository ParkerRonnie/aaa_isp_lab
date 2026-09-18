# -*- coding: utf-8 -*-
"""光学仿真：离焦 PSF、镜头阴影（vignetting）、横向色差。

这些模块存在的意义：AF/AE/AWB 的输入必须带有真实的"病"，否则
3A 算法只是空转。离焦提供 AF 的搜索目标，vignetting 提供 AE/AWB
的边缘亮度与色偏耦合，色差提供 AWB 的颜色误差来源。
"""
import numpy as np
import cv2


# -----------------------------------------------------------------------------
# 离焦 PSF
# -----------------------------------------------------------------------------
def defocus_kernel(radius_px: float, kind: str = "disk", ksize: int = None,
                   supersample: int = 8) -> np.ndarray:
    """生成离焦 PSF（抗锯齿）。

    disk    : 几何光学下的圆盘弥散圈（真实离焦的主要成分）
    gaussian: 衍射/像差近似

    为什么必须做**抗锯齿的亚像素覆盖**（这个坑很值得讲）：
    如果直接按"像素中心到圆心的距离 <= r"生成圆盘核，当 r < 1 时整个核
    只剩中心一个像素 —— PSF 退化成一个 delta，离焦完全不起作用。
    结果是半径 0~1 px 的所有离焦位置产生**完全相同**的图像，
    对焦评价函数在那一段是一条水平线，AF 搜索会随机停在平台上的任意位置。
    本项目第一版就出现了这个现象：tenengrad 的峰位稳定地落在 0.50
    （行程中点），而不是真值 0.35。

    正确做法是按像素被圆盘覆盖的面积（supersample 子采样）来定权重，
    这样 PSF 是半径的连续函数，0.3 px 的离焦也能被正确表达。
    """
    if radius_px <= 1e-3:
        return np.array([[1.0]], dtype=np.float32)

    if kind == "gaussian":
        sigma = max(radius_px / 2.0, 0.25)
        if ksize is None:
            ksize = int(2 * np.ceil(3 * sigma) + 1) | 1
        yy, xx = np.mgrid[0:ksize, 0:ksize].astype(np.float32)
        cy = cx = (ksize - 1) / 2.0
        r2 = (yy - cy) ** 2 + (xx - cx) ** 2
        k = np.exp(-r2 / (2 * sigma ** 2)).astype(np.float32)
        return k / k.sum()

    if kind != "disk":
        raise ValueError(f"unknown psf kind: {kind}")

    if ksize is None:
        # 覆盖圆盘本体 + 1 像素余量，保证抗锯齿边缘完整落在核内
        ksize = int(2 * np.ceil(radius_px) + 3) | 1
    ksize = max(3, ksize | 1)

    cy = cx = (ksize - 1) / 2.0
    # 每个像素内取 supersample x supersample 个子采样点
    offs = (np.arange(supersample) + 0.5) / supersample - 0.5
    gy_base = np.arange(ksize, dtype=np.float32)[:, None] - cy
    gx_base = np.arange(ksize, dtype=np.float32)[None, :] - cx

    k = np.zeros((ksize, ksize), dtype=np.float64)
    for dy in offs:
        for dx in offs:
            d2 = (gy_base + dy) ** 2 + (gx_base + dx) ** 2
            k += (d2 <= radius_px ** 2)
    k /= (supersample ** 2)

    s = k.sum()
    return (k / s).astype(np.float32) if s > 0 else np.array([[1.0]], dtype=np.float32)


def apply_psf(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    if kernel.shape == (1, 1):
        return img
    return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE)


def motion_blur_kernel(length_px: float, angle_deg: float = 0.0,
                       supersample: int = 4) -> np.ndarray:
    """运动模糊 PSF（线性匀速运动）。

    同样做超采样抗锯齿：先在高分辨率网格上画线段，再降采样，
    否则短曝光/慢速运动时的模糊长度会被量化到整数像素。
    """
    if length_px <= 1e-3:
        return np.array([[1.0]], dtype=np.float32)

    n = int(np.ceil(length_px)) + 2
    n = max(3, n | 1)
    ss = max(1, int(supersample))
    big = np.zeros((n * ss, n * ss), dtype=np.float32)
    c = (n * ss - 1) / 2.0
    rad = np.radians(angle_deg)
    half = length_px * ss / 2.0
    p0 = (int(round(c - half * np.cos(rad))), int(round(c - half * np.sin(rad))))
    p1 = (int(round(c + half * np.cos(rad))), int(round(c + half * np.sin(rad))))
    cv2.line(big, p0, p1, 1.0, 1, lineType=cv2.LINE_8)
    k = big.reshape(n, ss, n, ss).mean(axis=(1, 3))

    s = k.sum()
    return (k / s).astype(np.float32) if s > 0 else np.array([[1.0]], dtype=np.float32)


def apply_chromatic_aberration(img: np.ndarray, shift_px: float) -> np.ndarray:
    """横向色差：R/B 相对 G 有径向缩放差。

    这是 AWB 在边缘区域出现色偏的物理来源之一，也是"边缘区域
    灰色像素不灰"的原因。
    """
    if abs(shift_px) < 1e-3:
        return img
    h, w = img.shape[:2]
    out = img.copy()
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    for ch, scale in ((0, 1.0 + shift_px / max(w, h) * 2), (2, 1.0 - shift_px / max(w, h) * 2)):
        M = np.array([[scale, 0, cx * (1 - scale)],
                      [0, scale, cy * (1 - scale)]], dtype=np.float32)
        out[..., ch] = cv2.warpAffine(img[..., ch], M, (w, h),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REPLICATE)
    return out


# -----------------------------------------------------------------------------
# 镜头阴影
# -----------------------------------------------------------------------------
def vignetting_map(h: int, w: int, strength: float = 0.45,
                   color_dependency: float = 0.15) -> np.ndarray:
    """RAW 域的 cos^4 风格亮度衰减图，返回乘性衰减 map (h,w,3)。

    color_dependency > 0 时，三个通道衰减不同 —— 这是真实镜头
    （尤其广角）边缘出现色偏的第二个来源，也是 LSC 必须分通道做的原因。
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    # 归一化半径，四角约 1.0
    r = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2) / np.sqrt(2.0)
    base = 1.0 - strength * (r ** 2)

    out = np.empty((h, w, 3), dtype=np.float32)
    for ch in range(3):
        # 通道间衰减差异（简单线性插值，仅用于制造色偏）
        delta = color_dependency * (ch - 1) * 0.15 * (r ** 2)
        out[..., ch] = np.clip(base + delta, 0.05, 1.0)
    return out


def apply_vignetting(img: np.ndarray, strength: float = 0.45) -> np.ndarray:
    h, w = img.shape[:2]
    return img * vignetting_map(h, w, strength)
