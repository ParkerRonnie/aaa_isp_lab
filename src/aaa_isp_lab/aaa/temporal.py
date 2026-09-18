# -*- coding: utf-8 -*-
"""3A 的时域原语：对数域 EMA 滤波 + 场景切换检测。

单独成模块的理由：AE 与 AWB 共用同一套时域策略，而"平滑历史"和"判断世界
是否突然变了"这两件事都不属于控制律本身（控制律在 ae.py / awb.py）。

分层约束：本模块**不 import eval**（层次是 sim -> isp -> aaa -> eval，
aaa 不得反向依赖 eval）。结构信号因此与 eval.image_quality.texture_acutance
刻意重复实现，并由测试 test_structure_signal_matches_texture_acutance
钉住两边一致 —— 一旦漂移就报错，而不是让检测器悄悄用另一套公式。

一个必须先说清楚的负结论（见 README「时域滤波」一节）：
测光量是整帧空间平均，480x360 下光子散粒噪声被平均掉，静止场景的抖动
实测只有 ~4e-5 EV（evaluative）/ ~3e-4 EV（spot），**比收敛阈值 0.02 EV
低两个数量级**。也就是说在这个仿真里，"抖动"不是噪声的产物。
真正需要时域平滑的是**执行器量化引起的极限环**和**光源本身的逐帧波动** ——
这两样在仿真里必须显式注入才存在（见 sim/camera.py 的扰动源）。
"""
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import TemporalConfig
from ..sim.camera import ET_REF

_EPS = 1e-9


# ---------------------------------------------------------------------------
# 辅助量
# ---------------------------------------------------------------------------
def achieved_ev(frame) -> float:
    """这一帧**实际**达成的曝光 EV。

    注意不能用 Frame.exposure_ev —— 那是控制器的**请求值**。执行器量化
    （ET 寄存器步进 / 增益 1/6 EV 步进）之后，实际值与请求值不相等，
    只看请求值会把量化引起的抖动完全藏起来。
    """
    f = max(frame.exposure_s / ET_REF * frame.gain, _EPS)
    return math.log2(f)


def structure_signal(img) -> float:
    """带通能量 / 总能量（简化 acutance），只看"内容"变没变。

    对**整体亮度缩放严格不变**：x -> k*x 时分子分母同为 k^2 倍。
    因此这一路不需要曝光归一化，专门用来把"场景内容变了"从"曝光变了"里分出来。

    与 eval.image_quality.texture_acutance 同式，刻意重复（见模块 docstring）。
    """
    x = np.asarray(img, dtype=np.float64)
    if x.ndim == 3:
        x = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
    x = x - x.mean()
    log = cv2.GaussianBlur(x, (0, 0), 1.0) - cv2.GaussianBlur(x, (0, 0), 3.0)
    denom = float(np.mean(x * x))
    return float(np.mean(log * log) / denom) if denom > 0 else 0.0


# 对数域固定分箱区间（单位：log2）。实测曝光归一化后的亮度跨场景落在
# [-5.3, +2.6]（uniform 最窄 0.50~0.62；natural 逆光最宽 0.088~6.25），
# 取 [-6, +3] 留出余量，32 箱即 0.28 档/箱。
_HIST_LOG2_RANGE = (-6.0, 3.0)


def norm_luma_hist(frame, bins: int = 32) -> np.ndarray:
    """曝光归一化亮度直方图（对数域，概率，和为 1）。

    两个设计要点，都是实测定下来的：

    1) 除以本帧**实际**曝光因子（不是请求值）。这样"整体变亮/变暗"不改变
       直方图形状 —— 否则 AE 自己收敛 3 EV 就会被当成一次场景切换。
    2) 在 log2 域固定分箱，且用**固定**区间而不是逐帧 min/max 重标定。
       固定区间实测：±3 EV 内距离 ≤ 0.055；
       逐帧 min/max 重标定实测：±1 EV 就到 0.39 —— 等于把曝光变化又引回来了。

    局限：像素饱和后缩放关系不再成立，直方图会变形（检测器对此有饱和护栏）。
    """
    y = np.asarray(frame.luma_linear, dtype=np.float64) / (2.0 ** achieved_ev(frame))
    z = np.log2(np.clip(y, 1e-6, None))
    h, _ = np.histogram(np.clip(z, *_HIST_LOG2_RANGE), bins=bins,
                        range=_HIST_LOG2_RANGE)
    s = float(h.sum())
    return h / s if s > 0 else h


# ---------------------------------------------------------------------------
# 对数域 EMA 滤波器
# ---------------------------------------------------------------------------
class EMAFilter:
    """作用在**测光量**上的对数域一阶 EMA。

    稳态用 alpha_slow 压低抖动；误差大（收敛期）或刚检出场景切换时切到
    alpha_fast 避免滞后。

    为什么作用在测光量而不是 EV 指令：扰动从**测量**侧进入，滤波指令并不改变
    扰动的传递路径，只压低控制动作的高频；而且 ae.py 的过曝补偿是**故意**要求
    大步长快速退曝光，平滑指令会延迟退曝光，放大"越曝越看不出来"那个坑。

    过曝旁路：饱和是无噪声的硬信号，滤波会在 1~2 帧内把它稀释掉，
    所以过曝帧直接采纳原始测量值，不经过历史。

    状态在这里（实例属性），不在 TemporalConfig 上 —— 否则 deepcopy(cfg)
    会把陈旧状态一起复制，时序语义错乱。
    """

    def __init__(self, cfg: TemporalConfig):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self._state = None
        self._forced_fast = 0

    def force_fast(self, n: int = 1) -> None:
        """接下来 n 帧强制用 alpha_fast（场景切换后调用）。"""
        self._forced_fast = max(self._forced_fast, int(n))

    @property
    def state(self):
        return self._state

    def smooth(self, y: float, clipped_ratio: float = 0.0,
               err_prev: float = None) -> float:
        c = self.cfg
        y = float(y)
        if not c.enable:
            return y
        if self._state is None:
            self._state = y
            return y
        if clipped_ratio > c.clip_bypass_ratio:
            # 旁路：饱和帧直接采纳，别让历史把"过曝了"这个硬信号稀释掉
            self._state = y
            return y
        if self._forced_fast > 0:
            a = c.alpha_fast
            self._forced_fast -= 1
        elif c.adaptive and err_prev is not None and abs(err_prev) > c.conv_band_ev:
            a = c.alpha_fast
        else:
            a = c.alpha_slow
        self._state = a * y + (1.0 - a) * self._state
        return self._state


# ---------------------------------------------------------------------------
# 场景切换检测
# ---------------------------------------------------------------------------
@dataclass
class CutDecision:
    cut: bool = False
    kind: str = "none"                      # none|illumination|content
    signal: float = 0.0                     # 超阈最严重的那个信号的相对幅度
    parts: dict = field(default_factory=dict)


class SceneCutDetector:
    """三信号投票制场景切换检测。

    三个信号各管一段：
      d_metric_ev   曝光归一化测光量的跳变 —— 对"光变了"和"内容变了"都敏感，
                    归一化之后对**曝光自身的收敛**不敏感（这是消除最大误触发源的关键）
      texture_ratio 结构信号比值 —— 对亮度整体缩放严格不变，只对**内容**敏感
      hist_dist     曝光归一化亮度直方图距离 —— 对内容敏感，饱和处失效

    防误触发的六条（对应 README 里的实测数字）：
      1) 曝光归一化                      —— 否则 AE 自己收敛就是一次"场景切换"
      2) 参考量用慢 EMA 而不是前一帧      —— 单帧噪声无法触发
      3) 投票制，>= cut_votes 个信号超阈
      4) 门限由数据标定（实验 15），不靠猜
      5) 触发后的不应期 + 参考量重置      —— 否则陈旧参考会连续触发
      6) 饱和护栏：过曝帧抬高 metric 门限 —— 归一化在饱和处失效
    """

    def __init__(self, cfg: TemporalConfig):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self._ref_metric = None
        self._ref_tex = None
        self._ref_hist = None
        self._cool = 0
        self._armed = False

    def update(self, frame, metric: float, armed: bool = True) -> CutDecision:
        """armed=False 时只跟踪参考量、不判切换。

        为什么需要 armed：AE 还没收敛的时候，"场景变了"和"AE 还没调好"
        **在观测上是不可分的**。实测（192x144，从 +3 EV 起步）AE 自身收敛
        就会让三个信号全部超阈 —— 根因是过曝时像素饱和，归一化与标度不变性
        同时失效（连本该与亮度无关的 texture_ratio 都冲到 1.4）。
        真实相机也是等 AE 稳定后才监测场景切换的。

        未武装时参考量**直接跟当前帧、不做 EMA**：收敛过程中画面变化幅度远大于
        噪声，用慢 EMA 只会让参考量拖着一个陈旧的中间态，等到刚武装的瞬间
        立刻误触发（实测就是这样在 AE 刚稳定那一刻报了假警）。
        """
        c = self.cfg
        # --- 三个信号 ---
        y_n = math.log2(max(float(metric), _EPS)) - achieved_ev(frame)
        tex = structure_signal(frame.luma_linear)
        hist = norm_luma_hist(frame)
        zero_parts = {"d_metric_ev": 0.0, "texture_ratio": 0.0, "hist_dist": 0.0}

        if self._ref_metric is None or not armed:
            # 未武装：参考量直接跟当前帧
            self._ref_metric, self._ref_tex, self._ref_hist = y_n, tex, hist
            self._armed = False
            return CutDecision(parts=zero_parts)

        if not self._armed:
            # 刚武装：以当前帧为基准重新开始，不继承收敛过程的历史
            self._ref_metric, self._ref_tex, self._ref_hist = y_n, tex, hist
            self._armed = True
            return CutDecision(parts=zero_parts)

        a = c.ref_alpha
        d_metric = abs(y_n - self._ref_metric)
        d_tex = abs(math.log2(max(tex, _EPS) / max(self._ref_tex, _EPS)))
        d_hist = 1.0 - float(np.minimum(hist, self._ref_hist).sum())

        parts = {"d_metric_ev": d_metric, "texture_ratio": d_tex, "hist_dist": d_hist}

        if not c.cut_enable:
            return CutDecision(parts=parts)

        # 更新参考量（慢 EMA）
        self._ref_metric = a * y_n + (1.0 - a) * self._ref_metric
        self._ref_tex = a * tex + (1.0 - a) * self._ref_tex
        self._ref_hist = a * hist + (1.0 - a) * self._ref_hist

        if self._cool > 0:
            self._cool -= 1
            return CutDecision(parts=parts)

        # 6) 饱和护栏：过曝时归一化关系失效，抬高 metric 门限
        sat_guard = 1.0 + 4.0 * min(1.0, float(frame.clipped_ratio) / 0.10)
        v_metric = d_metric > c.cut_metric_ev * sat_guard
        v_tex = d_tex > c.cut_texture_ratio
        v_hist = d_hist > c.cut_hist_dist
        votes = int(v_metric) + int(v_tex) + int(v_hist)

        if votes < c.cut_votes:
            return CutDecision(parts=parts)

        # 分类：**只能用 texture_ratio 判**，不能用 hist_dist。
        # 实测（实验 14，192x144，2 seed）：hist_dist 对两类变化都敏感
        # （内容切换 0.57、光照 x4 0.71），拿它分类会把光照变化误判成内容变化；
        # 而 texture_ratio 对亮度整体缩放严格不变，只在内容变化时响
        # （内容 2.45 vs 光照 0.09~0.43），是真正的判别器。
        kind = "content" if v_tex else "illumination"
        rel = max(
            d_metric / max(c.cut_metric_ev, _EPS),
            d_tex / max(c.cut_texture_ratio, _EPS),
            d_hist / max(c.cut_hist_dist, _EPS),
        )
        # 5) 不应期 + 参考重置为当前帧，避免陈旧参考连续触发
        self._cool = int(c.cut_refractory)
        self._ref_metric, self._ref_tex, self._ref_hist = y_n, tex, hist
        return CutDecision(cut=True, kind=kind, signal=float(rel), parts=parts)
