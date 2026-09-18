# -*- coding: utf-8 -*-
"""评价指标。

3A 的好坏必须用数字说话，本项目用到三类指标：
    亮度类  —— AE   ：目标码值误差、过曝比例、收敛帧数
    颜色类  —— AWB  ：光源角度误差（度）、估计色温、灰阶 ΔE00
    清晰度类 —— AF  ：峰位误差、半高宽、动态范围、帧数
外加通用的 PSNR/SSIM 用于 ISP 链路本身的正确性验证。
"""
import numpy as np
import cv2

from ..color_science import rgb_linear_to_lab, linear_to_srgb


# -----------------------------------------------------------------------------
# 通用图像质量
# -----------------------------------------------------------------------------
def psnr(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10(data_range ** 2 / mse))


def ssim(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    """标准 SSIM（11x11 高斯窗，σ=1.5），灰度域计算。"""
    def gray(x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 3:
            x = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
        return x.astype(np.float32)

    x, y = gray(a), gray(b)
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
    xx = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x * mu_x
    yy = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y * mu_y
    xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_x * mu_y
    s = ((2 * mu_x * mu_y + C1) * (2 * xy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (xx + yy + C2))
    return float(s.mean())


# -----------------------------------------------------------------------------
# 色差
# -----------------------------------------------------------------------------
def delta_e_76(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    d = np.asarray(lab1, np.float64) - np.asarray(lab2, np.float64)
    return np.sqrt(np.sum(d * d, axis=-1))


def delta_e_2000(lab1, lab2, kL=1.0, kC=1.0, kH=1.0) -> np.ndarray:
    """CIEDE2000（Sharma 标准实现）。

    实现后用标准测试数据校验过：Lab(50,2.6772,-79.7751) vs
    Lab(50,0,-82.7485) 应为 2.0425（见 tests/test_aaa.py）。
    """
    lab1 = np.asarray(lab1, dtype=np.float64)
    lab2 = np.asarray(lab2, dtype=np.float64)
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    Cbar = 0.5 * (C1 + C2)
    G = 0.5 * (1.0 - np.sqrt(Cbar ** 7 / (Cbar ** 7 + 25.0 ** 7)))

    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = np.hypot(a1p, b1)
    C2p = np.hypot(a2p, b2)

    def hp(ap, bp):
        h = np.degrees(np.arctan2(bp, ap))
        return np.where(h < 0, h + 360.0, h)

    h1p, h2p = hp(a1p, b1), hp(a2p, b2)

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = h2p - h1p
    dhp = np.where(dhp > 180.0, dhp - 360.0, dhp)
    dhp = np.where(dhp < -180.0, dhp + 360.0, dhp)
    dhp = np.where(C1p * C2p == 0.0, 0.0, dhp)
    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2.0)

    Lbarp = 0.5 * (L1 + L2)
    Cbarp = 0.5 * (C1p + C2p)

    hsum = h1p + h2p
    hdiff = np.abs(h1p - h2p)
    hbarp = np.where(hdiff <= 180.0, 0.5 * hsum,
                     np.where(hsum < 360.0, 0.5 * (hsum + 360.0), 0.5 * (hsum - 360.0)))
    hbarp = np.where(C1p * C2p == 0.0, hsum, hbarp)

    T = (1.0 - 0.17 * np.cos(np.radians(hbarp - 30.0))
         + 0.24 * np.cos(np.radians(2.0 * hbarp))
         + 0.32 * np.cos(np.radians(3.0 * hbarp + 6.0))
         - 0.20 * np.cos(np.radians(4.0 * hbarp - 63.0)))

    dtheta = 30.0 * np.exp(-(((hbarp - 275.0) / 25.0) ** 2))
    RC = 2.0 * np.sqrt(Cbarp ** 7 / (Cbarp ** 7 + 25.0 ** 7))
    SL = 1.0 + (0.015 * (Lbarp - 50.0) ** 2) / np.sqrt(20.0 + (Lbarp - 50.0) ** 2)
    SC = 1.0 + 0.045 * Cbarp
    SH = 1.0 + 0.015 * Cbarp * T
    RT = -np.sin(np.radians(2.0 * dtheta)) * RC

    t1 = dLp / (kL * SL)
    t2 = dCp / (kC * SC)
    t3 = dHp / (kH * SH)
    return np.sqrt(t1 ** 2 + t2 ** 2 + t3 ** 2 + RT * t2 * t3)


# -----------------------------------------------------------------------------
# 与色彩/3A 直接相关的评价
# -----------------------------------------------------------------------------
def ideal_linear(scene, true_temp_k: float) -> np.ndarray:
    """理想成像结果（线性域）。

    推导：反射率 r × 光源 L × 理想白平衡增益 (1/L，G 归一化)
          = r × L_G。也就是说，白平衡正确时，理想图像就是反射率本身
          乘一个常数。这个结论让"颜色准不准"有了明确的无争议基准。
    """
    from ..color_science import blackbody_linear_rgb
    return (scene.reflectance * blackbody_linear_rgb(true_temp_k)[1]).astype(np.float32)


def patch_delta_e(measured_linear: np.ndarray, ideal_linear_rgb: np.ndarray,
                  masks) -> np.ndarray:
    """逐个色卡 patch 的 ΔE00（线性域 -> Lab 后比较）"""
    out = []
    for m in masks:
        if not np.any(m):
            continue
        a = measured_linear[m].mean(axis=0)
        b = ideal_linear_rgb[m].mean(axis=0)
        out.append(float(delta_e_2000(rgb_linear_to_lab(a), rgb_linear_to_lab(b))))
    return np.asarray(out)


def neutral_chroma(linear_rgb: np.ndarray, neutral_mask: np.ndarray) -> float:
    """中性区残余色度：白平衡后本应为 0。

    取中性像素的 Lab，算 (a*, b*) 的模长均值。
    这是"灰是不是真的灰"最直观的量化——比看图片靠谱。
    """
    if not np.any(neutral_mask):
        return float("nan")
    lab = rgb_linear_to_lab(linear_rgb[neutral_mask])
    return float(np.mean(np.hypot(lab[..., 1], lab[..., 2])))


def code_value(linear_luma: float) -> float:
    """线性亮度 -> sRGB 8bit 码值（AE 报数用，工程上习惯用码值交流）"""
    return float(linear_to_srgb(np.asarray(linear_luma)) * 255.0)


def subject_luma(frame, mask: np.ndarray) -> float:
    """指定区域（如逆光场景的前景主体）的平均线性亮度"""
    if not np.any(mask):
        return float("nan")
    return float(frame.luma_linear[mask].mean())
