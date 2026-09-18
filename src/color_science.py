# -*- coding: utf-8 -*-
"""颜色科学基础：色温 <-> 色度、sRGB/XYZ/Lab 变换、普朗克轨迹。

说明（重要，也是面试可讲的假设）：本项目把"传感器 RGB"近似当作 sRGB
原色处理。真实 ISP 会在 sensor RGB -> XYZ 的标定矩阵上做 AWB，本近似
不影响 3A 算法自身的收敛性/鲁棒性对比，但会引入固定的模型误差，
因此所有 ΔE 指标只用于**相对比较**，不作为绝对精度结论。
"""
import numpy as np

# --- sRGB (D65) 标准矩阵 -----------------------------------------------------
M_SRGB2XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float64)

M_XYZ2SRGB = np.linalg.inv(M_SRGB2XYZ)

WHITE_XYZ_D65 = M_SRGB2XYZ @ np.ones(3)

LAMBDA = np.arange(380.0, 781.0, 5.0)   # nm


# --- 传递函数 -----------------------------------------------------------------
def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * (x ** (1 / 2.4)) - 0.055)


def linear_to_gamma22(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0) ** (1.0 / 2.2)


# --- CIE 1931 CMF（Wyman/Sloan/Shirley 2013 多高斯拟合，免查表） ----------------
def _pgauss(x, mu, s1, s2):
    s = np.where(x < mu, s1, s2)
    return np.exp(-0.5 * ((x - mu) / s) ** 2)


def cmf_xyz(lam: np.ndarray) -> np.ndarray:
    """返回 (N,3) 的 CIE 1931 2° 配色函数拟合值"""
    x = (1.056 * _pgauss(lam, 599.8, 37.9, 31.0)
         + 0.362 * _pgauss(lam, 442.0, 16.0, 26.7)
         - 0.065 * _pgauss(lam, 501.1, 20.4, 26.2))
    y = (0.821 * _pgauss(lam, 568.8, 46.9, 40.5)
         + 0.286 * _pgauss(lam, 530.9, 16.3, 31.1))
    z = (1.217 * _pgauss(lam, 437.0, 11.8, 36.0)
         + 0.681 * _pgauss(lam, 459.0, 26.0, 13.8))
    return np.stack([x, y, z], axis=-1)


CMF = cmf_xyz(LAMBDA)


# --- 普朗克黑体 ---------------------------------------------------------------
def planck_spectral_radiance(lam_nm: np.ndarray, temp_k: float) -> np.ndarray:
    """普朗克黑体光谱辐亮度（相对值，h/k/c 常数已合并）"""
    lam_m = lam_nm * 1e-9
    h, c, k = 6.62607015e-34, 2.99792458e8, 1.380649e-23
    return (2 * h * c ** 2 / lam_m ** 5) / (np.exp(h * c / (lam_m * k * temp_k)) - 1.0)


def cct_to_xy(temp_k: float) -> tuple:
    """色温 -> CIE 1931 色度坐标（Kang et al. 2002 拟合，1667-25000K）"""
    t = float(np.clip(temp_k, 1667.0, 25000.0))
    if t <= 4000.0:
        xc = (-0.2661239e9 / t ** 3 - 0.2343589e6 / t ** 2
              + 0.8776956e3 / t + 0.179910)
    else:
        xc = (-3.0258469e9 / t ** 3 + 2.1070379e6 / t ** 2
              + 0.2226347e3 / t + 0.240390)
    if t <= 2222.0:
        yc = -1.1063814 * xc ** 3 - 1.34811020 * xc ** 2 + 2.18555832 * xc - 0.20219683
    elif t <= 4000.0:
        yc = -0.9549476 * xc ** 3 - 1.37418593 * xc ** 2 + 2.09137015 * xc - 0.16748867
    else:
        yc = 3.0817580 * xc ** 3 - 5.87338670 * xc ** 2 + 3.75112997 * xc - 0.37001483
    return float(xc), float(yc)


def xy_to_cct(x: float, y: float) -> float:
    """CIE 色度 -> 相关色温（McCamy 近似）"""
    if y <= 1e-6:
        return 0.0
    n = (x - 0.3320) / (y - 0.1858)
    return float(-449.0 * n ** 3 + 3525.0 * n ** 2 - 6823.3 * n + 5520.33)


# --- 光源 <-> 线性 RGB --------------------------------------------------------
def blackbody_linear_rgb(temp_k: float) -> np.ndarray:
    """给定色温的光源，在（近似）线性 sRGB 空间的颜色。

    归一化到 G=1，正好可以直接当作 AWB 的"理想增益倒数"使用。
    """
    spd = planck_spectral_radiance(LAMBDA, temp_k)
    xyz = CMF.T @ spd * 5.0        # Δλ = 5nm
    rgb = M_XYZ2SRGB @ xyz
    rgb = np.clip(rgb, 1e-6, None)
    return rgb / rgb[1]


def apply_illuminant(reflectance: np.ndarray, temp_k: float) -> np.ndarray:
    """反射率 + 光源色温 -> 场景线性 RGB（相机看到的"辐射亮度"形状）"""
    return reflectance * blackbody_linear_rgb(temp_k)[None, None, :]


def rgb_linear_to_xyz(rgb: np.ndarray) -> np.ndarray:
    return rgb @ M_SRGB2XYZ.T


def xyz_to_lab(xyz: np.ndarray, white_xyz: np.ndarray = None) -> np.ndarray:
    white = WHITE_XYZ_D65 if white_xyz is None else white_xyz
    r = np.asarray(xyz, dtype=np.float64) / white
    eps, kappa = 216.0 / 24389.0, 24389.0 / 27.0
    f = np.where(r > eps, np.cbrt(r), (kappa * r + 16.0) / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def rgb_linear_to_lab(rgb: np.ndarray) -> np.ndarray:
    return xyz_to_lab(rgb_linear_to_xyz(rgb))


def lab_to_xyz(lab: np.ndarray, white_xyz: np.ndarray = None) -> np.ndarray:
    white = WHITE_XYZ_D65 if white_xyz is None else white_xyz
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    eps, kappa = 216.0 / 24389.0, 24389.0 / 27.0

    def finv(f):
        return np.where(f ** 3 > eps, f ** 3, (116.0 * f - 16.0) / kappa)

    return np.stack([finv(fx), finv(fy), finv(fz)], axis=-1) * white


def illuminant_angle_deg(rgb_a: np.ndarray, rgb_b: np.ndarray) -> float:
    """两个光源 RGB 向量的夹角误差（度）——AWB 估计精度的标准指标"""
    a = np.asarray(rgb_a, dtype=np.float64).ravel()
    b = np.asarray(rgb_b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 90.0
    cos = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


# --- CIE 1960 UCS 与 Duv -----------------------------------------------------
def xy_to_uv60(x: float, y: float) -> tuple:
    """CIE 1931 (x,y) -> CIE 1960 UCS (u,v)。Duv 必须在这个空间里量。"""
    d = -2.0 * x + 12.0 * y + 3.0
    if abs(d) < 1e-12:
        return 0.0, 0.0
    return 4.0 * x / d, 6.0 * y / d


def duv(x: float, y: float) -> float:
    """相对普朗克轨迹的距离（带符号）。

    正 = 轨迹上方（偏绿），负 = 轨迹下方（偏品红）。

    为什么必须单独看这个量：色温（沿轨迹方向）和 Duv（垂直轨迹方向）
    是两个独立的误差分量。大面积单色场景把 AWB 带偏时，误差经常主要
    落在 Duv 方向上 —— 这时"把色温约束到合理范围"完全不起作用，
    色温看着很正确，颜色却是错的。
    """
    cct = xy_to_cct(x, y)
    if cct <= 0:
        cct = 6500.0
    x0, y0 = cct_to_xy(cct)
    u, v = xy_to_uv60(x, y)
    u0, v0 = xy_to_uv60(x0, y0)
    dist = float(np.hypot(u - u0, v - v0))
    # 用 v 的相对位置定符号（轨迹近似单调）
    return dist if v > v0 else -dist


def rgb_to_cct_duv(rgb: np.ndarray) -> tuple:
    """线性 RGB -> (色温, Duv)"""
    xyz = rgb_linear_to_xyz(np.clip(np.asarray(rgb, dtype=np.float64), 0, None))
    s = float(xyz.sum())
    if s <= 1e-12:
        return 0.0, 0.0
    x, y = float(xyz[0] / s), float(xyz[1] / s)
    if y <= 1e-9:
        return 0.0, 0.0
    return xy_to_cct(x, y), duv(x, y)
