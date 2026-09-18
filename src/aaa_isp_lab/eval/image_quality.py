# -*- coding: utf-8 -*-
"""画质指标测量：MTF / 信噪比 / 动态范围 / 阴影。

这里的每一项都是相机产线上真实在用的测量方法，写成"输入图像 -> 输出数字"的
纯函数，和仿真解耦 —— 换成真实相机拍的照片也能直接用。

    斜边法 MTF      ISO 12233   清晰度/解析力
    光子转换曲线     EMVA 1288   转换增益 K(e-/DN) 与读出噪声(e-)
    SNR / 动态范围   ISO 15739   噪声与宽容度
    色阴影/亮度均匀   ——          镜头与 LSC 的效果

**每个测量都必须能被验证**，否则只是些看着合理的数字。验证方式写在
`tests/test_aaa.py` 与 `experiments.exp_image_quality()` 里：
    用已知 σ 的高斯 PSF 生成斜边 -> 测出的 MTF 必须对上解析解
    仿真里已知转换增益与读出噪声 -> 光子转换曲线必须把它们反推回来
"""
from dataclasses import dataclass, field
import numpy as np
import cv2


# =============================================================================
# 1. 斜边法 MTF（ISO 12233）
# =============================================================================
@dataclass
class MTFResult:
    freqs: np.ndarray          # 空间频率, cycles/pixel
    mtf: np.ndarray            # 归一化 MTF, MTF(0)=1
    mtf50: float               # MTF 降到 0.5 处的频率
    mtf10: float
    mtf_nyquist: float         # 奈奎斯特频率(0.5 cy/px)处的值
    edge_angle_deg: float
    esf: np.ndarray = None
    lsf: np.ndarray = None
    oversample: int = 4


def _window(n: int, kind) -> np.ndarray:
    """窗函数。kind: 'tukey' | 'hamming' | None"""
    if not kind:
        return np.ones(n)
    if kind == "hamming":
        return np.hamming(n)
    if kind == "tukey":
        from scipy.signal import windows
        return windows.tukey(n, alpha=0.5)
    raise ValueError(f"未知窗函数: {kind}")


def _edge_positions(img: np.ndarray, grad_thresh_ratio: float = 0.35):
    """逐行找亚像素边缘位置。

    做法：对每行取水平梯度，在梯度最大的位置附近做抛物线插值得到亚像素峰值。
    只保留梯度足够强的行（弱行会被噪声主导）。
    """
    gx = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    h, w = gx.shape
    ys, xs = [], []
    for y in range(h):
        row = np.abs(gx[y])
        if row.max() < grad_thresh_ratio * np.abs(gx).max():
            continue
        x0 = int(np.argmax(row))
        if x0 < 1 or x0 >= w - 1:
            continue
        # 抛物线插值：峰值位置 = x0 + 0.5*(a-c)/(a-2b+c)
        a, b, c = row[x0 - 1], row[x0], row[x0 + 1]
        denom = a - 2 * b + c
        dx = 0.0 if abs(denom) < 1e-9 else 0.5 * (a - c) / denom
        ys.append(y)
        xs.append(x0 + np.clip(dx, -0.5, 0.5))
    return np.asarray(ys, dtype=np.float64), np.asarray(xs, dtype=np.float64)


def _build_esf(dist, vals, half_width: float, oversample: int):
    """把散点按法向距离分箱平均，得到 ESF。"""
    nbins = int(2 * half_width * oversample)
    edges = np.linspace(-half_width, half_width, nbins + 1)
    idx = np.clip(np.digitize(dist, edges) - 1, 0, nbins - 1)

    esf = np.full(nbins, np.nan)
    centers = np.full(nbins, np.nan)
    for i in range(nbins):
        sel = idx == i
        if np.count_nonzero(sel) >= 3:      # 样本太少的箱子丢掉，避免噪声尖刺
            esf[i] = vals[sel].mean()
            centers[i] = dist[sel].mean()
    ok = ~np.isnan(esf)
    if np.count_nonzero(ok) < max(8, nbins * 0.5):
        raise ValueError("ESF 有效箱数不足，边缘可能太陡或太糊")
    esf = np.interp(centers[ok], centers[ok], esf[ok])
    step = (centers[ok][-1] - centers[ok][0]) / (np.count_nonzero(ok) - 1)
    return esf, step


def slanted_edge_mtf(img: np.ndarray, oversample: int = 4,
                     max_freq: float = 0.5, window: str = "tukey",
                     half_width: float = None) -> MTFResult:
    """ISO 12233 斜边法测 MTF。

    步骤（每一步都有存在的理由）：
      1) 逐行找亚像素边缘 -> 线性拟合，得到边缘的斜率与角度
         （不能用"假设边缘是垂直的"，斜边法的全部意义就在于把采样点
          在法方向上错开，从而突破像素间距对频率上限的限制）
      2) 把每个像素投影到边缘法方向，按 1/oversample 像素分箱求均值 -> ESF
         （分箱相当于在法方向上重采样，把有效采样率提高 oversample 倍）
      3) ESF 求导 -> LSF
      4) 加窗（汉明）后 FFT -> 幅度谱 -> 归一化

    **窗口宽度必须自适应**：窗口太窄会把宽 LSF 截断，归一化时又按截断后的
    面积重新归一，结果 MTF 被整体抬高 —— 实测过 σ=3px 的高斯 PSF 会被
    测成 MTF50 偏大 60%。所以先粗测一次 ESF，由 10%~90% 过渡宽度估计
    边缘宽度，再按 4σ 定窗（σ ≈ w90 / 2.563，高斯 CDF 的标准关系）。

    img 建议用**线性域**或已知 gamma 的亮度图；不同 gamma 会得到不同的 MTF，
    比较时必须用同一个域。
    """
    img = np.asarray(img, dtype=np.float32)
    if img.ndim == 3:
        img = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]

    ys, xs = _edge_positions(img)
    if ys.size < 8:
        raise ValueError("找不到足够的边缘行，检查图像是否真的包含斜边")

    # 边缘直线: x = a*y + b
    a, b = np.polyfit(ys, xs, 1)
    angle = float(np.degrees(np.arctan(a)))

    # 每个像素到直线的**垂直**距离
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    dist_all = (xx - a * yy - b) / np.sqrt(1.0 + a * a)

    if half_width is None:
        # 第一遍：用宽窗粗测过渡宽度，据此定最终窗口
        probe = 16.0
        keep = np.abs(dist_all) < probe
        esf0, _ = _build_esf(dist_all[keep], img[keep], probe, oversample)
        lo, hi = np.percentile(esf0, [10, 90])
        span = hi - lo
        trans = float(np.sum((esf0 > lo + 0.1 * span) & (esf0 < lo + 0.9 * span))) / oversample
        sigma_est = max(trans / 2.563, 0.35)     # 高斯 CDF: w90 = 2.563σ
        half_width = float(np.clip(5.0 * sigma_est, 4.0, probe))

    keep = np.abs(dist_all) < half_width
    d = dist_all[keep]
    v = img[keep]
    esf, step = _build_esf(d, v, half_width, oversample)

    # ESF -> LSF
    lsf = np.gradient(esf, step)
    # 加窗：抑制两端截断带来的振铃。
    # 这里用**平顶 Tukey 窗**而不是汉明窗 —— 汉明窗整段都在衰减，等效于把
    # LSF 在空间域变窄，频谱随之展宽，实测会把 MTF50 抬高约 16%（σ≥1px 时）。
    # Tukey 窗中间是一段平坦区（不展宽），只在两端做余弦收尾（防截断），
    # 实测偏差可以压到 2% 以内。
    if window:
        lsf = lsf * _window(lsf.size, window)

    # 归一化用**加窗后 LSF 的直流分量**（即 LSF 的面积），不能先减均值 ——
    # 减掉均值会把直流压到 0，归一化就成了除以一个接近 0 的数，
    # 高频端直接炸到 1e15 量级（这个错误不会报错，只会给出一堆天文数字）。
    spec = np.abs(np.fft.rfft(lsf))
    if spec[0] <= 1e-12:
        raise ValueError("LSF 直流分量异常")
    mtf = spec / spec[0]

    freqs = np.fft.rfftfreq(lsf.size, d=step)
    sel = freqs <= max_freq
    freqs, mtf = freqs[sel], mtf[sel]

    def _cross(level: float) -> float:
        """找 MTF 首次降到 level 的频率（线性插值）"""
        below = np.where(mtf < level)[0]
        if below.size == 0:
            return float(freqs[-1])
        i = below[0]
        if i == 0:
            return float(freqs[0])
        f0, f1 = freqs[i - 1], freqs[i]
        m0, m1 = mtf[i - 1], mtf[i]
        if m0 == m1:
            return float(f1)
        return float(f0 + (level - m0) * (f1 - f0) / (m1 - m0))

    nyq = float(np.interp(0.5, freqs, mtf)) if freqs[-1] >= 0.5 else float(mtf[-1])

    return MTFResult(freqs=freqs, mtf=mtf, mtf50=_cross(0.5), mtf10=_cross(0.1),
                     mtf_nyquist=nyq, edge_angle_deg=angle,
                     esf=esf, lsf=lsf, oversample=oversample)


def synthetic_slanted_edge(w: int, h: int, sigma_px: float, angle_deg: float = 5.0,
                           contrast: float = 0.8, level: float = 0.5,
                           supersample: int = 8) -> np.ndarray:
    """物理正确地合成一条斜边，用于验证 MTF 测量链路。

    顺序很重要：**先超采样画理想阶跃 -> 在高分辨率域做高斯模糊 -> 再降采样平均**。
    这样模糊作用在连续域上，降采样顺带带进像素孔径，
    可以直接和 `gaussian_mtf_theory(..., pixel_aperture=True)` 对比。

    如果反过来（先生成图像再对采样后的图做模糊），虽然理论上等效，
    但小 σ 时数值差异会被放大，自检会看到 +10% 的假偏差。
    """
    ss = max(1, int(supersample))
    yy, xx = np.mgrid[0:h * ss, 0:w * ss].astype(np.float64) / ss
    x0 = w * 0.5 + (yy - h / 2.0) * np.tan(np.radians(angle_deg))
    img = np.where(xx > x0, level + contrast / 2, level - contrast / 2).astype(np.float32)
    if sigma_px > 0:
        k = int(2 * np.ceil(3 * sigma_px * ss) + 1)
        img = cv2.GaussianBlur(img, (k, k), sigma_px * ss, sigma_px * ss,
                               borderType=cv2.BORDER_REPLICATE)
    if ss > 1:
        img = img.reshape(h, ss, w, ss).mean(axis=(1, 3))
    return img


def gaussian_mtf_theory(freqs: np.ndarray, sigma_px: float,
                        pixel_aperture: bool = True) -> np.ndarray:
    """已知高斯 PSF 的理论 MTF，用于验证测量链路。

    高斯 PSF exp(-x²/2σ²) 的傅里叶变换是 exp(-2π²σ²f²)（f 单位 cycles/pixel）。
    斜边法在法方向分箱平均，等效于再叠加一个宽度约 1 像素的矩形孔径，
    所以理论曲线要乘上 |sinc(f)| —— 不扣掉它，测量值会比"PSF 本身"偏低，
    看起来像测量误差，其实是定义差异。
    """
    mtf = np.exp(-2.0 * np.pi ** 2 * sigma_px ** 2 * freqs ** 2)
    if pixel_aperture:
        mtf = mtf * np.abs(np.sinc(freqs))
    return mtf


# =============================================================================
# 2. 噪声 / 光子转换曲线
# =============================================================================
@dataclass
class NoisePoint:
    mean_dn: float       # 去黑电平后的平均信号 (DN)
    std_dn: float        # 标准差 (DN)
    n_pixels: int

    @property
    def snr_db(self) -> float:
        return 20.0 * np.log10(self.mean_dn / self.std_dn) if self.std_dn > 0 else float("inf")


@dataclass
class PhotonTransferResult:
    gain_e_per_dn: float     # 转换增益 K
    read_noise_e: float      # 读出噪声（折算到输入端，单位 e-）
    r2: float
    n_points: int
    points: list = field(default_factory=list)


def patch_channel_stats(raw: np.ndarray, patch, pattern: np.ndarray,
                        black_level: float = 0.0) -> dict:
    """从一个平坦区域按 CFA 通道分别统计噪声。

    必须在 RAW 上、分通道统计：去马赛克会做颜色插值，把相邻像素的噪声
    平均掉，测出来的噪声会明显偏低；LSC 增益又会让边缘的噪声被放大。
    产线上做 SNR 测量也是这么做的（RAW 分通道 + 均匀光）。

    patch: (y0, y1, x0, x1)
    """
    y0, y1, x0, x1 = patch
    out = {}
    names = {0: "R", 1: "G", 2: "B"}
    for i in range(2):
        for j in range(2):
            ch = int(pattern[i, j])
            sub = raw[y0:y1, x0:x1]
            sub = sub[i::2, j::2]
            if sub.size < 16:
                continue
            vals = sub.astype(np.float64) - black_level
            key = names[ch]
            # 同色通道可能出现两次（G），合并起来
            if key in out:
                prev = out[key]
                n = prev.n_pixels + vals.size
                mean = (prev.mean_dn * prev.n_pixels + vals.mean() * vals.size) / n
                # 合并两组样本的方差
                var = ((prev.std_dn ** 2 + prev.mean_dn ** 2) * prev.n_pixels
                       + (vals.var() + vals.mean() ** 2) * vals.size) / n - mean ** 2
                out[key] = NoisePoint(mean, float(np.sqrt(max(var, 0.0))), n)
            else:
                out[key] = NoisePoint(float(vals.mean()), float(vals.std()), vals.size)
    return out


def fit_photon_transfer(points, saturation_dn: float = None,
                        clip_ratio: float = 0.90) -> PhotonTransferResult:
    """由 (均值, 方差) 序列拟合光子转换曲线（EMVA 1288）。

    模型（单位 DN）：
        var = mean / K + (Nr / K)²
    斜率给转换增益 K (e-/DN)，截距给读出噪声 Nr (e-)。

    两个必须做的处理，否则拟合结果会大幅偏掉（本项目实测：不做这两步，
    K 会被反推成真值的 1.5 倍、读出噪声偏大 17 倍）：

    1) **剔除饱和点**。信号一旦顶到满阱，方差会被截断压塌（越亮反而越"干净"），
       而它的均值又最大 —— 在最小二乘里就是一个高杠杆的错误点，
       会把斜率整体拽偏。真实 PTC 测量也必须先做饱和检测。
    2) **按 1/var² 加权**。样本方差的自身不确定度正比于方差本身
       （相对误差约 sqrt(2/N)），高信号点的方差大、更不可靠，
       不加权等于让最不可靠的点主导拟合。

    saturation_dn 给定时按 clip_ratio 剔除接近饱和的点。
    """
    pts = [p for p in points if p.mean_dn > 0 and p.std_dn > 0]
    if saturation_dn:
        pts = [p for p in pts if p.mean_dn < clip_ratio * saturation_dn]
    if len(pts) < 3:
        raise ValueError("光子转换曲线至少需要 3 个有效点（剔除饱和点后）")

    x = np.array([p.mean_dn for p in pts])
    y = np.array([p.std_dn ** 2 for p in pts])
    w = 1.0 / np.maximum(y, 1e-12) ** 2          # 1/var²

    sw = w.sum()
    swx = (w * x).sum()
    swy = (w * y).sum()
    swxx = (w * x * x).sum()
    swxy = (w * x * y).sum()
    denom = sw * swxx - swx ** 2
    if abs(denom) < 1e-18:
        raise ValueError("加权拟合退化，信号范围可能过窄")
    slope = (sw * swxy - swx * swy) / denom
    intercept = (swy * swxx - swx * swxy) / denom
    if slope <= 0:
        raise ValueError("拟合斜率为负，数据不符合光子转换模型")

    K = 1.0 / slope
    Nr = K * np.sqrt(max(intercept, 0.0))

    pred = slope * x + intercept
    ss_res = float(np.sum(w * (y - pred) ** 2))
    ybar = float(np.sum(w * y) / sw)
    ss_tot = float(np.sum(w * (y - ybar) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return PhotonTransferResult(gain_e_per_dn=float(K), read_noise_e=float(Nr),
                                r2=r2, n_points=len(pts), points=pts)


# =============================================================================
# 3. 动态范围
# =============================================================================
def dynamic_range_db(saturation_dn: float, dark_noise_dn: float) -> float:
    """动态范围 = 20log10(满阱信号 / 暗噪声)，单位 dB。

    也有用"档"表示的：DR_stops = DR_dB / 6.02。
    注意这里的分母是**暗噪声**（读出噪声），不是某个中间亮度下的噪声 ——
    动态范围的定义就是"最亮与最暗可分辨"之比。
    """
    if dark_noise_dn <= 0:
        return float("inf")
    return float(20.0 * np.log10(saturation_dn / dark_noise_dn))


def sensor_dr_theory_db(full_well_e: float, read_noise_e: float) -> float:
    """传感器理论动态范围（仅受满阱与读出噪声限制）"""
    return float(20.0 * np.log10(full_well_e / read_noise_e))


# =============================================================================
# 4. 阴影：亮度均匀度与色阴影
# =============================================================================
def _xy_to_uprime_vprime(x: np.ndarray, y: np.ndarray):
    d = -2.0 * x + 12.0 * y + 3.0
    d = np.where(np.abs(d) < 1e-12, 1e-12, d)
    return 4.0 * x / d, 9.0 * y / d


def shading_metrics(linear_rgb: np.ndarray, center_ratio: float = 1 / 6.0,
                    corner_ratio: float = 0.08) -> dict:
    """亮度均匀度与**色阴影**。

    色阴影是比亮度阴影更容易被忽略、也更难修的问题：镜头阴影在三个通道上
    衰减不同，加上 CFA 串扰，边角的**颜色**会和中心不一样。
    只看亮度均匀度是看不出来的，必须单独量色度差。

    指标：
      luma_uniformity  四角亮度 / 中心亮度（1.0 = 完全均匀）
      d_uv_corner_max  四角与中心的 (u', v') 色度差最大值 ×1000
                       （相机规格书里常见的写法，一般要求 < 10~20）
    """
    from ..color_science import rgb_linear_to_xyz

    img = np.asarray(linear_rgb, dtype=np.float64)
    h, w = img.shape[:2]

    def _slice(cy, cx, ry, rx):
        y0, y1 = int(cy - ry), int(cy + ry)
        x0, x1 = int(cx - rx), int(cx + rx)
        return img[max(y0, 0):y1, max(x0, 0):x1]

    ch, cw = max(2, int(h * center_ratio)), max(2, int(w * center_ratio))
    center = _slice(h // 2, w // 2, ch, cw)
    kh, kw = max(2, int(h * corner_ratio)), max(2, int(w * corner_ratio))
    corners = {
        "左上": _slice(kh, kw, kh, kw),
        "右上": _slice(kh, w - kw, kh, kw),
        "左下": _slice(h - kh, kw, kh, kw),
        "右下": _slice(h - kh, w - kw, kh, kw),
    }

    def _luma(block):
        return float((0.2126 * block[..., 0] + 0.7152 * block[..., 1]
                      + 0.0722 * block[..., 2]).mean())

    def _uv(block):
        rgb = block.reshape(-1, 3).mean(axis=0)
        xyz = rgb_linear_to_xyz(rgb)
        s = float(xyz.sum())
        if s <= 1e-12:
            return np.array([0.0, 0.0])
        x, y = xyz[0] / s, xyz[1] / s
        u, v = _xy_to_uprime_vprime(np.array([x]), np.array([y]))
        return np.array([float(u[0]), float(v[0])])

    luma_c = _luma(center)
    uv_c = _uv(center)

    lumas, duvs = {}, {}
    for name, blk in corners.items():
        lumas[name] = _luma(blk) / max(luma_c, 1e-9)
        duvs[name] = float(np.linalg.norm(_uv(blk) - uv_c) * 1000.0)

    return {
        "luma_uniformity": float(min(lumas.values())),
        "luma_uniformity_mean": float(np.mean(list(lumas.values()))),
        "luma_corners": lumas,
        "d_uv_corner_max": float(max(duvs.values())),
        "d_uv_corners": duvs,
    }


# =============================================================================
# 5. 综合：一张图上的 SNR 与纹理锐度
# =============================================================================
def texture_acutance(img: np.ndarray) -> float:
    """纹理锐度（简化版 acutance）：带通能量 / 总能量。

    用 Laplacian-of-Gaussian 能量相对图像总能量归一化，
    反映"细节被保留了多少"，对降噪/锐化调参的强弱很敏感。
    """
    x = np.asarray(img, dtype=np.float64)
    if x.ndim == 3:
        x = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
    x = x - x.mean()
    log = cv2.GaussianBlur(x, (0, 0), 1.0) - cv2.GaussianBlur(x, (0, 0), 3.0)
    denom = float(np.mean(x * x))
    return float(np.mean(log * log) / denom) if denom > 0 else 0.0
