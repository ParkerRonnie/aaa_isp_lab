# -*- coding: utf-8 -*-
"""AF（自动对焦）。

对比度检测 AF（CDAF）的全部逻辑建立在两件事上：
  1) 对焦评价函数：图像的高频能量。合焦 -> 高频最强 -> 评价函数取极大值
  2) 搜索策略：评价函数理论上单峰，于是可以用爬山/粗到细/黄金分割来减少帧数

工程上评价一个对焦评价函数，看四个性质（本项目会逐项量化）：
    无偏性  峰值位置 = 真合焦位置
    单峰性  只有一个主峰，没有会误导搜索的次峰
    灵敏度  峰值附近足够陡（半高宽小），否则微小的离焦区分不出来
    抗噪性  低照度/高增益下峰位漂移小

搜索策略的目标是**用最少的帧数**找到峰：每一帧都是一次真实曝光，
帧数直接等价于对焦速度和功耗。
"""
from dataclasses import dataclass, field
import numpy as np
import cv2

from ..config import AFConfig

MEASURES = ("brenner", "tenengrad", "laplacian_var", "sml", "fft_energy")


@dataclass
class AFResult:
    strategy: str
    measure: str
    best_pos: float
    best_value: float
    frames: int                       # 消耗的帧数（= 评价次数）
    visited: list = field(default_factory=list)
    values: list = field(default_factory=list)
    converged: bool = True


# -----------------------------------------------------------------------------
# 预处理与 ROI
# -----------------------------------------------------------------------------
def _prep(plane: np.ndarray) -> np.ndarray:
    """去掉低频照明分量，只保留结构。

    低照度下画面整体亮度变化会被某些评价函数误认为"高频"，
    先减去局部均值可以显著改善抗噪性。
    """
    x = plane.astype(np.float32)
    x = x - cv2.GaussianBlur(x, (0, 0), 8.0)
    return x


def _roi_regions(h: int, w: int, roi: str, ratio: float):
    if roi == "full":
        return [(0, h, 0, w)]
    if roi == "center":
        ch, cw = int(h * ratio), int(w * ratio)
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        return [(y0, y0 + ch, x0, x0 + cw)]
    if roi == "multi":
        # 3x3 分区：模拟多点对焦，最终取各区响应的均值
        # （真实相机常取"最近主体优先"，这里取均值是为了让曲线更平滑）
        regions = []
        bh, bw = h // 3, w // 3
        for i in range(3):
            for j in range(3):
                regions.append((i * bh, (i + 1) * bh, j * bw, (j + 1) * bw))
        return regions
    raise ValueError(roi)


# -----------------------------------------------------------------------------
# 对焦评价函数
# -----------------------------------------------------------------------------
def _brenner(x: np.ndarray) -> float:
    d = x[:, 2:] - x[:, :-2]
    return float(np.mean(d * d)) if d.size else 0.0


def _tenengrad(x: np.ndarray, thresh_ratio: float = 0.0) -> float:
    gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
    mag2 = gx * gx + gy * gy
    if thresh_ratio > 0 and mag2.size:
        # 阈值门限：滤掉纯噪声梯度，低对比场景下这一点很关键
        t = thresh_ratio * float(np.max(mag2))
        mag2 = mag2[mag2 > t]
    return float(np.mean(mag2)) if mag2.size else 0.0


def _laplacian_var(x: np.ndarray) -> float:
    lap = cv2.Laplacian(x, cv2.CV_32F, ksize=3)
    return float(lap.var())


def _sml(x: np.ndarray, thresh: float = 0.0) -> float:
    """Sum-Modified-Laplacian：对二阶差分做限幅，抗噪性比 Laplacian 方差好"""
    dx = np.abs(2 * x[:, 1:-1] - x[:, :-2] - x[:, 2:])
    dy = np.abs(2 * x[1:-1, :] - x[:-2, :] - x[2:, :])
    if thresh > 0:
        dx = np.minimum(dx, thresh)
        dy = np.minimum(dy, thresh)
    return float(np.mean(dx) + np.mean(dy))


def _fft_energy(x: np.ndarray, cutoff_ratio: float = 0.15) -> float:
    f = np.fft.fftshift(np.abs(np.fft.fft2(x)))
    h, w = f.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((yy - cy) / max(cy, 1)) ** 2 + ((xx - cx) / max(cx, 1)) ** 2)
    hi = f[r > cutoff_ratio].sum()
    all_ = f.sum()
    return float(hi / max(all_, 1e-9))


def focus_measure(plane: np.ndarray, method: str = "tenengrad",
                  roi: str = "center", roi_ratio: float = 0.5) -> float:
    if method not in MEASURES:
        raise ValueError(f"unknown focus measure: {method}")
    h, w = plane.shape[:2]
    x = _prep(plane)
    vals = []
    for (y0, y1, x0, x1) in _roi_regions(h, w, roi, roi_ratio):
        patch = x[y0:y1, x0:x1]
        if patch.size < 16:
            continue
        if method == "brenner":
            vals.append(_brenner(patch))
        elif method == "tenengrad":
            vals.append(_tenengrad(patch, thresh_ratio=0.02))
        elif method == "laplacian_var":
            vals.append(_laplacian_var(patch))
        elif method == "sml":
            vals.append(_sml(patch))
        else:
            vals.append(_fft_energy(patch))
    return float(np.mean(vals)) if vals else 0.0


def frame_measure(frame, cfg: AFConfig) -> float:
    """在绿通道上评价。

    理由：G 的采样率是 R/B 的两倍（Bayer 里占 50%），SNR 最好；
    且 G 不受白平衡增益影响，因此 AF 与 AWB 解耦。
    """
    return focus_measure(frame.linear_pre_wb[..., 1], cfg.measure, cfg.roi, cfg.roi_ratio)


# -----------------------------------------------------------------------------
# 搜索策略
# -----------------------------------------------------------------------------
class AFController:
    def __init__(self, cfg: AFConfig = None):
        self.cfg = cfg or AFConfig()

    def run(self, capture_fn) -> AFResult:
        c = self.cfg
        strat = c.strategy
        if strat == "sweep":
            return self._sweep(capture_fn)
        if strat == "hill_climb":
            return self._hill_climb(capture_fn)
        if strat == "coarse_to_fine":
            return self._coarse_to_fine(capture_fn)
        if strat == "golden_section":
            return self._golden_section(capture_fn)
        raise ValueError(strat)

    def _eval(self, capture_fn, pos, res: AFResult) -> float:
        v = frame_measure(capture_fn(float(pos)), self.cfg)
        res.visited.append(float(pos))
        res.values.append(float(v))
        res.frames += 1
        return v

    def _sweep(self, capture_fn) -> AFResult:
        c = self.cfg
        res = AFResult(strategy="sweep", measure=c.measure, best_pos=0.0,
                       best_value=-1.0, frames=0)
        for pos in np.linspace(c.lens_range[0], c.lens_range[1], c.sweep_steps):
            v = self._eval(capture_fn, pos, res)
            if v > res.best_value:
                res.best_value, res.best_pos = v, float(pos)
        return res

    def _coarse_to_fine(self, capture_fn) -> AFResult:
        """粗扫定位大致范围，再在邻域细扫。

        分两阶段的原因：评价函数虽然单峰，但远离峰值时几乎平坦，
        直接用细步长扫会浪费大量帧数；粗扫能快速进入峰附近，
        代价是可能有 ±(粗步长/2) 的定位误差，所以要用细扫补回精度。
        """
        c = self.cfg
        res = AFResult(strategy="coarse_to_fine", measure=c.measure, best_pos=0.0,
                       best_value=-1.0, frames=0)
        lo, hi = c.lens_range

        coarse = np.linspace(lo, hi, c.coarse_steps)
        vals = [self._eval(capture_fn, p, res) for p in coarse]
        k = int(np.argmax(vals))
        res.best_value, res.best_pos = float(vals[k]), float(coarse[k])

        step = (hi - lo) / (c.coarse_steps - 1)
        fine_lo = max(lo, coarse[k] - step)
        fine_hi = min(hi, coarse[k] + step)
        for pos in np.linspace(fine_lo, fine_hi, c.fine_steps):
            v = self._eval(capture_fn, pos, res)
            if v > res.best_value:
                res.best_value, res.best_pos = v, float(pos)
        return res

    def _hill_climb(self, capture_fn) -> AFResult:
        """爬山法：朝评价函数增大的方向走，方向反转则缩小步长。

        这是最省帧数的策略，但风险是：噪声造成的假峰会让它走错方向，
        所以需要 patience（连续几次不改善才认为到顶）而不是一见下降就掉头。
        """
        c = self.cfg
        res = AFResult(strategy="hill_climb", measure=c.measure, best_pos=0.0,
                       best_value=-1.0, frames=0)
        lo, hi = c.lens_range
        pos = 0.5 * (lo + hi)
        step = c.hill_step
        v = self._eval(capture_fn, pos, res)
        res.best_value, res.best_pos = v, pos
        direction = 1.0
        no_improve = 0

        for _ in range(c.max_iters):
            if step < c.hill_min_step:
                break
            nxt = float(np.clip(pos + direction * step, lo, hi))
            if nxt == pos:
                direction = -direction
                step *= 0.5
                no_improve = 0
                continue
            nv = self._eval(capture_fn, nxt, res)
            if nv > res.best_value:
                res.best_value, res.best_pos = nv, nxt
                pos, v = nxt, nv
                no_improve = 0
            else:
                no_improve += 1
                direction = -direction       # 反向，同时压步长
                step *= 0.5
                if no_improve >= c.hill_patience:
                    break
        return res

    def _golden_section(self, capture_fn) -> AFResult:
        """黄金分割搜索：对单峰函数在给定帧数下近似最优。

        前提是函数严格单峰；一旦有次峰，它会被"关"在错误的区间里出不来，
        这是一个很好的反例：理论最优 ≠ 工程可用。
        """
        c = self.cfg
        res = AFResult(strategy="golden_section", measure=c.measure, best_pos=0.0,
                       best_value=-1.0, frames=0)
        lo, hi = c.lens_range
        phi = (np.sqrt(5.0) - 1.0) / 2.0
        a, b = lo, hi
        c1 = b - phi * (b - a)
        c2 = a + phi * (b - a)
        f1 = self._eval(capture_fn, c1, res)
        f2 = self._eval(capture_fn, c2, res)
        for _ in range(c.max_iters):
            if abs(b - a) < c.hill_min_step:
                break
            if f1 < f2:
                a, c1, f1 = c1, c2, f2
                c2 = a + phi * (b - a)
                f2 = self._eval(capture_fn, c2, res)
            else:
                b, c2, f2 = c2, c1, f1
                c1 = b - phi * (b - a)
                f1 = self._eval(capture_fn, c1, res)
        res.best_pos = 0.5 * (a + b)
        res.best_value = max(f1, f2)
        return res


# -----------------------------------------------------------------------------
# 评价函数质量分析（无偏性 / 单峰性 / 灵敏度 / 抗噪性）
# -----------------------------------------------------------------------------
def measure_curve(camera, af_cfg: AFConfig, n: int = 41, repeats: int = 1,
                  ev: float = 0.0):
    """扫描镜头全程，得到每种评价函数的曲线。

    repeats 次重复用于量化抗噪性：同一镜头位置、不同噪声实现下的
    峰位抖动越小，说明该评价函数在低照度下越可靠。

    注意：同一位置只曝光一次，5 种评价函数共用这一帧 —— 保证对比是
    在同一份噪声/同一份场景内容上进行的（否则差异里混入了噪声方差）。
    """
    positions = np.linspace(af_cfg.lens_range[0], af_cfg.lens_range[1], n)
    out = {m: [] for m in MEASURES}
    for _ in range(repeats):
        acc = {m: [] for m in MEASURES}
        for p in positions:
            fr = camera.capture(ev=ev, focus_pos=float(p))
            green = fr.linear_pre_wb[..., 1]
            for m in MEASURES:
                acc[m].append(focus_measure(green, m, af_cfg.roi, af_cfg.roi_ratio))
        for m in MEASURES:
            out[m].append(np.asarray(acc[m], dtype=np.float64))
    return {"positions": positions,
            "curves": {m: np.stack(v, axis=0) for m, v in out.items()},
            "true_focus": camera.true_focus}


def curve_metrics(positions: np.ndarray, curve: np.ndarray) -> dict:
    """四个性质的可量化版本。curve: (n,)"""
    v = np.asarray(curve, dtype=np.float64)
    n = v.size
    apex = int(np.argmax(v))
    peak_pos = float(positions[apex])
    peak_val = float(v[apex])
    base = float(np.median(np.concatenate([v[:max(2, n // 6)], v[-max(2, n // 6):]])))
    rng = peak_val - base

    # 灵敏度：半高全宽（越小越灵敏）
    if rng > 1e-12:
        above = np.where(v >= base + 0.5 * rng)[0]
        fwhm = float(positions[above[-1]] - positions[above[0]]) if above.size else float("inf")
    else:
        fwhm = float("inf")

    # 单峰性：峰两侧的导数符号翻转次数（理想为 0）
    d = np.diff(v)
    viol = 0
    for side in (d[:apex], d[apex:]):
        if side.size:
            s = np.sign(side)
            s = s[s != 0]
            if s.size > 1:
                viol += int(np.sum(s[1:] != s[:-1]))

    # 动态范围：峰值 vs 远离焦点的响应（决定"能不能分辨出对没对上"）
    edge = np.concatenate([v[:max(2, n // 8)], v[-max(2, n // 8):]])
    dyn_db = float(20 * np.log10(peak_val / max(float(np.mean(edge)), 1e-12)) + 1e-12)

    return {"peak_pos": peak_pos, "peak_val": peak_val, "base": base,
            "fwhm": fwhm, "monotonic_violations": viol, "dynamic_range_db": dyn_db}
