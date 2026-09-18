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
import math
import numpy as np

from ..config import AEConfig, TemporalConfig
from ..color_science import linear_to_srgb
from .temporal import EMAFilter, SceneCutDetector, achieved_ev


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


@dataclass
class AEStreamResult(AEResult):
    """逐帧流式结果（run_stream 的返回）。

    继承 AEResult 以保持向后兼容：只新增字段，不改动既有字段的语义。
    注意 iters 在流式下等于 n_frames，**与"收敛帧数"不同义** ——
    收敛帧数看 settle_frames。
    """
    n_frames: int = 0
    settle_frames: int = -1            # 切换后重收敛所需帧数，-1 = 未收敛
    cut_frames: list = field(default_factory=list)    # 检出场景切换的帧号
    jitter: dict = field(default_factory=dict)        # 三级抖动统计
    achieved_ev: list = field(default_factory=list)   # 每帧实际达成的 EV


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
    def __init__(self, cfg: AEConfig = None, temporal: TemporalConfig = None):
        self.cfg = cfg or AEConfig()
        # 时域策略是可选的第二个配置。语义边界：cfg 管控制律（怎么算这一步），
        # temporal 管时域策略（怎么用历史）。
        self.temporal_cfg = temporal
        # **状态挂实例，不挂 cfg** —— 实验普遍 copy.deepcopy(cfg) 后逐实验改参，
        # 状态若进了配置对象就会被一起克隆，时序语义直接错乱
        # （同 ISPPipeline.last_lsc_map 的范式）。
        self._filter = EMAFilter(temporal) if (temporal and temporal.enable) else None
        self._detector = (SceneCutDetector(temporal)
                          if (temporal and temporal.cut_enable) else None)

    def reset_temporal(self) -> None:
        """清空时域状态（换场景序列时用，避免上一段的历史污染下一段）。"""
        if self._filter is not None:
            self._filter.reset()
        if self._detector is not None:
            self._detector.reset()

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

    # -- 单帧观测（run 与 run_stream 共用的控制律）----------------------------
    def _observe(self, fr, cur_ev: float, adaptive: bool,
                 err_prev: float = None, it: int = 0) -> tuple:
        """测光 -> (可选) log2 域时域滤波 -> 误差 -> 步长与过曝补偿。

        run() 与 run_stream() 共用这一份控制律。**不共用就必然分叉** ——
        以后改 damping / clip_boost 要改两处，早晚不一致。

        无时域滤波时 y_hat == y，误差式退化为原来的
        log2(target / metric)，数值与之前逐位相同。
        """
        c = self.cfg
        m = metering_metric(fr, c)
        metric, target = m["metric"], m["target"]

        y = math.log2(max(metric, 1e-9))
        clip = float(fr.clipped_ratio)
        y_hat = y if self._filter is None else self._filter.smooth(
            y, clipped_ratio=clip, err_prev=err_prev)
        err_ev = float(math.log2(max(target, 1e-9)) - y_hat)

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
        if err_ev < 0 and clip > 0.02:
            err_ev = min(err_ev, -(0.3 + 2.0 * clip))
        if clip > 0.005 and err_ev < 0:
            d = min(1.0, d * (1.0 + c.clip_boost * float(np.sqrt(clip))))

        rec = {
            "iter": it, "ev": cur_ev, "metric": metric, "target": target,
            "err_ev": err_ev, "damping": d,
            "code_value": m["detail"]["code_value"],
            "clip_ratio": fr.clipped_ratio,
            "exposure_s": fr.exposure_s, "gain": fr.gain,
            # 新增键（既有键的名字与语义都不变，report 仍在读 metric/err_ev）
            "y_hat": float(y_hat), "ev_ach": float(achieved_ev(fr)),
        }
        return rec, err_ev, d

    # -- 闭环（收敛即停）------------------------------------------------------
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
        cur_ev = ev

        for it in range(max_iters):
            if on_frame is not None:
                on_frame(it)
            fr = capture_fn(cur_ev)
            rec, err_ev, d = self._observe(fr, cur_ev, adaptive, prev_err, it)
            res.target = rec["target"]

            if prev_err is not None and err_ev * prev_err < 0:
                res.reversals += 1

            res.history.append(rec)

            if abs(err_ev) < c.converge_thresh_ev:
                res.converged = True
                res.final_ev = cur_ev
                res.final_metric = rec["metric"]
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

    # -- 流式闭环（逐帧持续运行，不因收敛而停）--------------------------------
    def run_stream(self, capture_fn, ev0: float = 0.0, n_frames: int = 60,
                   adaptive: bool = True, on_frame=None, cut_frame: int = None,
                   settle_thresh_ev: float = None,
                   settle_hold: int = None) -> AEStreamResult:
        """逐帧持续运行，**不因收敛而 break**。

        为什么必须单独有一条路径：run() 一旦收敛就停（ae.py 的 break），
        静止序列后面就没有帧了 —— 而"稳态抖动"恰恰只能在收敛之后的帧上测。
        这也更贴近真实相机：AE 是每帧都在跑的，不是收敛一次就关机。

        n_frames **必须显式给**：默认沿用 cfg.max_iters(=12) 会静默截断，
        那样测到的"抖动"其实是还没进稳态的瞬态。

        on_frame(it) 与 run() 同约定，在成像**之前**调用 —— 所以第 k 帧施加的
        场景切换，首个受影响的帧就是 k，真值标号没有歧义。

        收敛只做记录（settle_frames），不停机。
        抖动统计不在这里算：aaa 层不得依赖 eval 层，序列都留在 history 里，
        由 eval.metrics 去统计（见 aaa/temporal.py 的分层说明）。
        """
        c = self.cfg
        tc = self.temporal_cfg
        n_frames = int(n_frames)
        if settle_thresh_ev is None:
            settle_thresh_ev = tc.settle_thresh_ev if tc else 0.02
        if settle_hold is None:
            settle_hold = int(tc.settle_hold) if tc else 3
        settle_hold = max(1, int(settle_hold))

        res = AEStreamResult(target=0.0, n_frames=n_frames)
        cur_ev = float(np.clip(ev0, c.ev_min, c.ev_max))
        prev_err = None
        settled_flags = []
        armed_prev = False          # 首次稳定之前不武装检测器（见 temporal.SceneCutDetector）
        settle_run = 0

        for it in range(n_frames):
            if on_frame is not None:
                on_frame(it)
            fr = capture_fn(cur_ev)
            rec, err_ev, d = self._observe(fr, cur_ev, adaptive, prev_err, it)
            res.target = rec["target"]

            if prev_err is not None and err_ev * prev_err < 0:
                res.reversals += 1

            # --- 场景切换检测 ---
            # armed 用**上一帧**的稳定状态（滞后一帧）：切换发生的那一帧
            # 误差同时跳变，若用本帧状态判断就会先把检测器关掉，正好漏检。
            if self._detector is not None:
                dec = self._detector.update(fr, rec["metric"], armed=armed_prev)
                rec["cut"] = bool(dec.cut)
                rec["cut_kind"] = dec.kind
                rec["cut_parts"] = dec.parts
                rec["armed"] = bool(armed_prev)
                if dec.cut:
                    res.cut_frames.append(it)
                    # 检出切换就重置滤波器，而不是继续把它抹平 ——
                    # 这样才能同时拿到低抖动（稳态）与低延迟（切换时）
                    if self._filter is not None:
                        self._filter.force_fast(tc.fast_frames_after_cut if tc else 3)

            settled_flags.append(abs(err_ev) < settle_thresh_ev)
            rec["settled"] = bool(settled_flags[-1])
            res.history.append(rec)
            # 连续稳定 settle_hold 帧才算"稳"，检测器才武装
            settle_run = settle_run + 1 if settled_flags[-1] else 0
            armed_prev = settle_run >= settle_hold

            prev_err = err_ev
            nxt = float(np.clip(cur_ev + d * err_ev, c.ev_min, c.ev_max))
            if tc is not None and tc.ev_slew_ev_per_frame > 0:
                lim = float(tc.ev_slew_ev_per_frame)
                nxt = cur_ev + float(np.clip(nxt - cur_ev, -lim, lim))
            res.max_overshoot_ev = max(res.max_overshoot_ev, abs(nxt - cur_ev))
            cur_ev = nxt

        res.final_ev = cur_ev
        res.iters = n_frames
        res.final_metric = res.history[-1]["metric"]
        res.converged = bool(settled_flags and settled_flags[-1])
        res.achieved_ev = [h["ev_ach"] for h in res.history]

        # 切换后的重收敛帧数：从 cut_frame（真值）起，首次连续 settle_hold 帧达标
        if cut_frame is not None:
            start = int(cut_frame)
            hold = 0
            for i in range(start, len(settled_flags)):
                hold = hold + 1 if settled_flags[i] else 0
                if hold >= settle_hold:
                    res.settle_frames = i - start + 1
                    break

        return res
