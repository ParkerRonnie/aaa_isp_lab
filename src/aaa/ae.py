# -*- coding: utf-8 -*-
"""AE（自动曝光）。

核心是"测光 + 闭环控制"两件事：
  测光  —— 决定用画面里的哪些像素、以什么统计量代表"亮度"
            逆光场景下测光方式的选择比控制算法本身更影响成片
  控制  —— 决定用多少 EV 修正、步长多大、多久算收敛
            曝光与亮度近似线性，所以控制律在 log2 域上做（对数域里
            成像是"平移"关系，控制律与世界光照无关，这是 AE 能在
            任何光照下用同一套参数的原因）

同时实现"标定查表 + 微调"（真实相机的做法）和纯闭环，用于对比收敛速度。
"""
from dataclasses import dataclass, field
import numpy as np

from ..config import AEConfig
from ..color_science import linear_to_srgb


@dataclass
class AEResult:
    history: list = field(default_factory=list)   # 每次迭代的记录
    final_ev: float = 0.0
    iters: int = 0
    converged: bool = False
    final_metric: float = 0.0
    target: float = 0.18
    reversals: int = 0          # 方向反转次数（振荡程度）
    max_overshoot_ev: float = 0.0


# -----------------------------------------------------------------------------
# 测光
# -----------------------------------------------------------------------------
def _zone_weights(h: int, w: int, zones, sigma: float) -> np.ndarray:
    zy, zx = zones
    yy, xx = np.mgrid[0:zy, 0:zx].astype(np.float32)
    cy, cx = (zy - 1) / 2.0, (zx - 1) / 2.0
    r2 = ((yy - cy) / max(zy, 1)) ** 2 + ((xx - cx) / max(zx, 1)) ** 2
    return np.exp(-r2 / (2 * sigma ** 2))


def _center_weight(h: int, w: int, ratio: float = 1.0) -> np.ndarray:
    """中心加权（高斯），ratio 控制中心区域的集中度"""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r2 = ((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2
    return np.exp(-r2 / (2 * 0.35 ** 2 * max(ratio, 1e-3)))


def metering_metric(frame, cfg: AEConfig, raw_linear_fallback: bool = False) -> dict:
    """返回 (度量值, 目标值, 细节)。

    度量值一律定义在**线性域**（目标 0.18 = 18% 中灰，对应 sRGB 码值 118）。
    在线性域做比较的好处：控制律与色调曲线解耦，换 tone curve 不用重标。
    """
    luma = frame.luma_linear.astype(np.float32)
    h, w = luma.shape
    clip_ratio = frame.clipped_ratio

    if cfg.metering == "average":
        metric = float(luma.mean())
        target = cfg.target_linear
        detail = {"mode": "全画面平均"}

    elif cfg.metering == "center":
        wmap = _center_weight(h, w, cfg.center_weight)
        metric = float((luma * wmap).sum() / wmap.sum())
        target = cfg.target_linear
        detail = {"mode": "中心加权"}

    elif cfg.metering == "spot":
        sh = max(1, int(h * cfg.spot_ratio))
        sw = max(1, int(w * cfg.spot_ratio))
        y0, x0 = (h - sh) // 2, (w - sw) // 2
        metric = float(luma[y0:y0 + sh, x0:x0 + sw].mean())
        target = cfg.target_linear
        detail = {"mode": "点测光（中央窗口）"}

    elif cfg.metering == "evaluative":
        zy, zx = cfg.zones
        wmap = _zone_weights(h, w, cfg.zones, cfg.zone_sigma)
        # 按分区统计：分区越靠近中心权重越大，且过曝分区额外加权（高光保护）
        zsum = 0.0
        wsum = 0.0
        clip_zone = 0.0
        ys = np.linspace(0, h, zy + 1).astype(int)
        xs = np.linspace(0, w, zx + 1).astype(int)
        for i in range(zy):
            for j in range(zx):
                blk = luma[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
                if blk.size == 0:
                    continue
                zmean = float(blk.mean())
                # 该分区里有多少像素已经顶到接近饱和
                cfrac = float(np.mean(blk > 0.95))
                zsum += wmap[i, j] * (zmean + cfg.highlight_weight * cfrac)
                wsum += wmap[i, j]
                clip_zone += wmap[i, j] * cfrac
        metric = float(zsum / max(wsum, 1e-6))
        target = cfg.target_linear
        detail = {"mode": "分区评价测光（含高光保护）", "clip_zone": float(clip_zone / max(wsum, 1e-6))}

    elif cfg.metering == "highlight_priority":
        # 用高分位数代表"高光"，目标是让最亮的那部分刚好不过曝。
        # 取 99 分位而不是 95：95 分位在亮背景占比大的画面里仍然偏低，
        # 会给出"整体过曝但高光没截断"的反直觉结果。
        metric = float(np.percentile(luma, 99))
        target = 0.90
        detail = {"mode": "高光优先（99 分位）"}

    else:
        raise ValueError(cfg.metering)

    detail["code_value"] = float(linear_to_srgb(np.array(metric)) * 255.0)
    detail["clip_ratio"] = clip_ratio
    return {"metric": metric, "target": target, "detail": detail}


# -----------------------------------------------------------------------------
# 控制器
# -----------------------------------------------------------------------------
class AEController:
    def __init__(self, cfg: AEConfig = None):
        self.cfg = cfg or AEConfig()

    # -- 标定表（真实相机做法：一次性标定 EV->亮度 曲线）----------------------
    def calibrate(self, capture_fn, n: int = 13, ev_span=(-6.0, 6.0)) -> dict:
        """标定 EV -> log2(度量值) 曲线，并求逆得到"一步到位"的初始 EV。

        注意：标定必须在无噪声或平均多帧的条件下做，否则查表会带来偏差。
        """
        c = self.cfg
        evs = np.linspace(ev_span[0], ev_span[1], n)
        vals = []
        for ev in evs:
            fr = capture_fn(float(ev))
            vals.append(metering_metric(fr, c)["metric"])
        vals = np.asarray(vals, dtype=np.float64)
        vals = np.maximum(vals, 1e-6)
        logv = np.log2(vals)
        # 单调化处理（饱和后曲线会变平，导致不可逆；用累计最大保证单调）
        logv = np.maximum.accumulate(logv)
        return {"evs": evs, "metric": vals, "log_metric": logv}

    def lut_initial_ev(self, lut: dict, target: float) -> float:
        logt = np.log2(max(target, 1e-9))
        evs, logv = lut["evs"], lut["log_metric"]
        # 找曲线穿过目标值的区间并线性插值
        idx = np.searchsorted(logv, logt)
        if idx <= 0:
            return float(evs[0])
        if idx >= len(logv):
            return float(evs[-1])
        x0, x1 = logv[idx - 1], logv[idx]
        t = 0.0 if x1 == x0 else (logt - x0) / (x1 - x0)
        return float(evs[idx - 1] + t * (evs[idx] - evs[idx - 1]))

    # -- 闭环 ----------------------------------------------------------------
    def run(self, capture_fn, ev0: float = 0.0, max_iters: int = None,
            adaptive: bool = True, on_frame=None) -> AEResult:
        """闭环收敛。

        capture_fn(ev) -> Frame
        on_frame(it) 在每帧成像前调用，用于模拟场景/光照突变
        （重收敛速度是 AE 评价的核心指标之一：真实场景一直在变）
        """
        c = self.cfg
        max_iters = max_iters or c.max_iters
        res = AEResult(target=0.0)
        ev = float(np.clip(ev0, c.ev_min, c.ev_max))
        prev_err = None
        prev_ev = None
        cur_ev = ev

        for it in range(max_iters):
            if on_frame is not None:
                on_frame(it)
            fr = capture_fn(cur_ev)
            m = metering_metric(fr, c)
            metric, target = m["metric"], m["target"]
            res.target = target

            err_ev = float(np.log2(max(target, 1e-9) / max(metric, 1e-9)))
            # 变步长：大误差时线性假设成立（远离饱和），可用大步长；
            # 小误差时受噪声与色调曲线非线性影响，必须收小步长否则来回振荡
            if adaptive:
                d = 0.9 if abs(err_ev) > 1.0 else c.damping
            else:
                d = c.damping
            # --- 过曝时的误差修正（AE 最关键的一个坑）---
            # 一旦大片像素饱和，测得的亮度就顶在 1.0 不动了：
            # 哪怕实际过曝 2 档，p99 也只能给到 1.0，算出来的误差被死死压到
            # log2(0.9/1.0) = -0.15 EV。也就是说**越曝越"看不出来"**，
            # 控制器会以为快到目标了，一帧只退 0.15 EV，大逆光场景十几帧都收敛不了。
            # 解法：改用"过曝像素比例"这个与亮度无关的信息，给出一个步长下限。
            # 过曝比例是饱和的直接证据，比亮度统计可靠得多。
            clip = float(fr.clipped_ratio)
            if err_ev < 0 and clip > 0.02:
                err_ev = min(err_ev, -(0.3 + 2.0 * clip))
            if clip > 0.005 and err_ev < 0:
                d = min(1.0, d * (1.0 + c.clip_boost * float(np.sqrt(clip))))

            if prev_err is not None and err_ev * prev_err < 0:
                res.reversals += 1

            res.history.append({
                "iter": it, "ev": cur_ev, "metric": metric, "target": target,
                "err_ev": err_ev, "damping": d,
                "code_value": m["detail"]["code_value"],
                "clip_ratio": fr.clipped_ratio,
                "exposure_s": fr.exposure_s, "gain": fr.gain,
            })

            if abs(err_ev) < c.converge_thresh_ev:
                res.converged = True
                res.final_ev = cur_ev
                res.final_metric = metric
                res.iters = it + 1
                break

            prev_err = err_ev
            nxt = float(np.clip(cur_ev + d * err_ev, c.ev_min, c.ev_max))
            res.max_overshoot_ev = max(res.max_overshoot_ev, abs(nxt - cur_ev))
            cur_ev = nxt
        else:
            res.final_ev = cur_ev
            res.iters = max_iters
            res.final_metric = res.history[-1]["metric"]

        return res
