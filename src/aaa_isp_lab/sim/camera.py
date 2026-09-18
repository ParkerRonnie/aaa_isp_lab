# -*- coding: utf-8 -*-
"""仿真相机：把场景 + 光学 + 传感器 + ISP 串成一个可被 3A 反复调用的对象。

3A 的本质是**闭环控制**：控制器给出 (曝光, 对焦, 白平衡) -> 相机成像 ->
统计 -> 控制器再修正。所以这里提供一个 `capture(...)` 接口，
3A 算法只依赖这个接口，与真实相机 SDK 的结构一致。
"""
from dataclasses import dataclass, field
import numpy as np

from ..config import SensorConfig, ISPConfig
from ..color_science import apply_illuminant
from . import optics
from .sensor import SensorSim
from ..isp.pipeline import ISPPipeline

ET_REF = 1.0 / 60.0          # 基准曝光时间：EV=0 对应 1/60s @ gain 1.0


@dataclass
class Frame:
    """一帧成像结果。linear_pre_wb 是 AWB 的输入，srgb 是"出图"。"""
    raw_dn: np.ndarray
    linear_pre_wb: np.ndarray   # 去马赛克 + LSC 后、白平衡前（AWB 统计域）
    linear_ccm: np.ndarray      # 白平衡 + CCM 后，线性域（AE/评价用）
    srgb_u8: np.ndarray         # 色调映射 + 降噪锐化后的显示图
    exposure_ev: float
    exposure_s: float
    gain: float
    focus_pos: float
    clipped_ratio: float
    meta: dict = field(default_factory=dict)

    @property
    def luma_linear(self) -> np.ndarray:
        return (0.2126 * self.linear_ccm[..., 0]
                + 0.7152 * self.linear_ccm[..., 1]
                + 0.0722 * self.linear_ccm[..., 2])


def ev_limits(cfg: SensorConfig, anti_flicker: bool = False,
              mains_hz: float = 50.0) -> tuple:
    """相机**能实现**的 EV 范围。

    这一步很容易被忽略：AE 控制器的输出范围必须限制在物理可实现的
    区间内，否则控制器会一直积分类似"下一帧就好了"的误差，
    表现为到不了目标值却永远不收敛（本项目第一版就踩了这个坑：
    控制器一路推到 +8 EV，而镜头实际最多给到 +5 EV）。
    """
    et_min = cfg.min_exposure_s / ET_REF
    et_max = cfg.max_exposure_s / ET_REF
    if anti_flicker:
        step = (1.0 / (2.0 * mains_hz)) / ET_REF
        et_max = max(step, float(np.floor(et_max / step)) * step)
    return (float(np.log2(et_min)), float(np.log2(et_max * cfg.max_analog_gain)))


def split_exposure(ev: float, cfg: SensorConfig, priority: str = "shutter_priority",
                   anti_flicker: bool = False, mains_hz: float = 50.0) -> tuple:
    """把总曝光因子 EV 拆成 (曝光时间, 增益)。

    这是 AE 的"执行器分配"环节，直接决定画质：
      - 优先快门 -> 噪声低，但长曝易运动模糊
      - 优先增益 -> 不糊，但噪声高（增益放大的是同一份光子噪声）
      - 抗闪烁  -> 曝光时间被强制量化到光源纹波周期的整数倍
                    （50Hz 市电 -> 100Hz 纹波 -> 曝光必须是 10ms 的整数倍）
    """
    E = 2.0 ** ev                                   # 相对基准的总曝光因子
    et_min = cfg.min_exposure_s / ET_REF
    et_max = cfg.max_exposure_s / ET_REF
    g_max = cfg.max_analog_gain

    if anti_flicker:
        # 纹波周期 = 1/(2*mains)；同时把曝光上限也压到整数倍，
        # 否则长曝时仍然会有带纹（真实工程中常被忽略的一个坑）
        period = 1.0 / (2.0 * mains_hz)
        step = period / ET_REF
        n_max = max(1, int(np.floor(et_max / step)))
        et_max = n_max * step
        if priority == "shutter_priority":
            n = max(1, int(round(E / step)))
            et = min(max(n * step, step), et_max)
            gain = E / et
            gain = min(max(gain, 1e-3), g_max)
        else:
            gain = min(max(E, 1e-3), g_max)
            n = max(1, int(round((E / gain) / step)))
            et = min(max(n * step, step), et_max)
        return _quantize(et, gain, cfg)

    if priority == "shutter_priority":
        et = min(max(E, et_min), et_max)
        gain = E / et
        gain = min(max(gain, 1e-3), g_max)
    else:  # gain_priority
        gain = min(max(E, 1.0), g_max)
        et = E / gain
        et = min(max(et, et_min), et_max)
        gain = E / et
        gain = min(max(gain, 1e-3), g_max)
    return _quantize(et, gain, cfg)


def _quantize(et_norm: float, gain: float, cfg: SensorConfig) -> tuple:
    """把（归一化曝光时间, 增益）落到执行器真正能实现的格点上。

    这是"抖动"的**确定性**来源，也是真实 AE 必须做时域平滑的头号原因：
    控制器算出的 EV 是连续的，落到格点后必然带半个步长的残差；残差符号随手
    场景微扰翻转，曝光就在相邻两个格点之间来回跳，形成**极限环**。
    关键是它**不需要任何噪声** —— 抖动来自控制结构，不是来自传感器。

    步长默认为 0（无限细分 = 关闭），关闭时数值与之前逐位相同。
    """
    if cfg.et_step_s > 0:
        step = cfg.et_step_s / ET_REF
        et_norm = max(1, int(round(et_norm / step))) * step
    if cfg.gain_step_ev > 0:
        s = float(cfg.gain_step_ev)
        gain = float(2.0 ** (s * round(float(np.log2(max(gain, 1e-9))) / s)))
    return et_norm * ET_REF, gain


class SimCamera:
    def __init__(self, scene, illum_temp_k: float, sensor_cfg: SensorConfig = None,
                 isp_cfg: ISPConfig = None, seed: int = 0,
                 true_focus: float = 0.5, lsc_model: str = "ideal",
                 max_blur_px: float = 4.0):
        self.scene = scene
        self.illum_temp_k = illum_temp_k
        self.sensor_cfg = sensor_cfg or SensorConfig()
        self.isp = ISPPipeline(self.sensor_cfg, isp_cfg or ISPConfig(), lsc_model=lsc_model)
        self.sensor = SensorSim(self.sensor_cfg, seed=seed)
        self.rng = np.random.default_rng(seed + 1)
        self.true_focus = true_focus
        self.max_blur_px = max_blur_px
        # 光源纹波幅度：0 表示直流光源（无闪烁）；抗闪烁实验中设为 0.25
        self.flicker_amplitude = 0.0

        # --- 时域扰动源 ---------------------------------------------------
        # 默认**全部关闭**，关闭时数值与之前逐位相同（既有实验的结论不受影响）。
        # 存在的理由：整帧测光把光子散粒噪声平均掉了（480x360 下约 4e-5 EV，
        # 比收敛阈值低两个数量级），所以静止场景的 AE 抖动本来等于浮点噪声。
        # 要让"时域滤波"这件事有意义，扰动必须作为**显式的物理源**注入。
        self.illum_ripple_frac = 0.0        # 光源强度逐帧波动（相对 RMS）
        self.flicker_phase = 0.0            # 闪烁相位（单位：纹波周期）
        self.flicker_phase_jitter = 0.0     # 每帧相位的随机抖动（周期）
        self.flicker_phase_drift = 0.0      # 每帧相位的固定漂移（周期）
        # 独立随机流：**绝不能用 self.rng** —— 它的序列已被 motion_angle 消费，
        # 共用会改变既有实验的随机数，让所有历史结论静默失效。
        self._illum_rng = np.random.default_rng(seed + 101)

        # 场景在线性 RGB 下的"辐射亮度形状"（反射率 × 光源）
        self.radiance = apply_illuminant(scene.reflectance, illum_temp_k)

        self.exposure_ev = 0.0
        self.focus_pos = 0.0
        self.wb_gains = np.ones(3, dtype=np.float32)
        # 运动模糊方向：随场景固定，代表相机相对被摄体的运动
        self.motion_angle = float(self.rng.uniform(0, 180))

    # -- 控制接口 -------------------------------------------------------------
    def set_exposure_ev(self, ev: float) -> None:
        self.exposure_ev = float(ev)

    def set_focus(self, pos: float) -> None:
        self.focus_pos = float(np.clip(pos, 0.0, 1.0))

    def set_wb_gains(self, gains: np.ndarray) -> None:
        self.wb_gains = np.asarray(gains, dtype=np.float32)

    def set_scene(self, scene, temp_k: float = None) -> None:
        """切换场景（用于模拟场景/光照突变的时间序列）。

        走的是与 __init__ **同一个** apply_illuminant，因此
        Frame.meta 里的 scene / illum_temp_k 始终是真值，检测器的
        真值标号是免费的。

        注意这里**不重启 SensorSim 的噪声流** —— 切换只换世界，不换相机。
        物理上正确，同时也避免了"切换处有一个可被检测器识别的伪影"这种作弊。
        """
        if temp_k is not None:
            self.illum_temp_k = float(temp_k)
        self.scene = scene
        self.radiance = apply_illuminant(scene.reflectance, self.illum_temp_k)

    # -- 成像 ----------------------------------------------------------------
    def capture(self, ev: float = None, focus_pos: float = None,
                wb_gains: np.ndarray = None, add_noise: bool = True,
                ae_cfg=None) -> Frame:
        ev = self.exposure_ev if ev is None else float(ev)
        focus_pos = self.focus_pos if focus_pos is None else float(focus_pos)
        wb = self.wb_gains if wb_gains is None else np.asarray(wb_gains, dtype=np.float32)

        priority = getattr(ae_cfg, "priority", "shutter_priority")
        anti_flicker = bool(getattr(ae_cfg, "anti_flicker", False))
        mains = getattr(ae_cfg, "mains_hz", 50.0)
        motion_speed = getattr(ae_cfg, "motion_speed_px_per_s", 0.0)

        exposure_s, gain = split_exposure(ev, self.sensor_cfg, priority,
                                          anti_flicker, mains)

        # --- 光学：离焦 + 色差 + 运动模糊 + 阴影 ---
        img = self.radiance
        if self.illum_ripple_frac > 0:
            # 光源强度逐帧波动（乘性）：发生在光学与光电转换之前，
            # 等效于入射光通量本身在变，而不是传感器在变。
            img = img * (1.0 + self.illum_ripple_frac
                         * float(self._illum_rng.normal()))
        blur_px = self.max_blur_px * abs(focus_pos - self.true_focus)
        img = optics.apply_psf(img, optics.defocus_kernel(blur_px, "disk"))
        img = optics.apply_chromatic_aberration(img, 0.8)
        # 光谱串扰：发生在光电转换之前（CFA 滤光不理想），
        # 是"传感器 RGB 不等于场景颜色"的根本原因，也是 CCM 存在的理由
        if self.sensor_cfg.crosstalk is not None:
            C = np.asarray(self.sensor_cfg.crosstalk, dtype=np.float32)
            img = img @ C.T
        if motion_speed > 0:
            img = optics.apply_psf(img, optics.motion_blur_kernel(
                motion_speed * exposure_s, self.motion_angle))
        img = optics.apply_vignetting(img, self.sensor_cfg.vignetting_strength)

        # --- 传感器 ---
        # 曝光因子 = (曝光时间/基准时间) × 增益，使 exposure_factor=1 时
        # 线性域数值 == 物理辐射亮度（EV 与光圈的换算关系随之成立）
        exposure_factor = (exposure_s / ET_REF) * gain
        raw = self.sensor.expose(self.sensor.mosaic(img), exposure_factor,
                                 exposure_s, gain, add_noise)

        # --- 光源闪烁带纹（乘性，等效于入射光通量的行间波动）---
        # 是否出现带纹只取决于光源与曝光时间，与 AE 是否开启抗闪烁无关；
        # 抗闪烁只是让 AE 去**选**一个不会产生带纹的曝光时间。
        if self.flicker_amplitude > 0:
            # 相位逐帧推进：帧时序未锁相于市电时，纹波相对读出起点的相位
            # 每一帧都不同 —— 这会让"整帧平均后的等效曝光"逐帧变化。
            if self.flicker_phase_drift or self.flicker_phase_jitter:
                self.flicker_phase = (
                    self.flicker_phase + self.flicker_phase_drift
                    + self.flicker_phase_jitter * float(self._illum_rng.normal())
                ) % 1.0
            raw = apply_flicker_banding(raw, exposure_s, mains,
                                        self.sensor_cfg.readout_s,
                                        self.sensor_cfg.black_level_dn,
                                        self.flicker_amplitude, self.flicker_phase)

        # --- ISP ---
        out = self.isp.process(raw, wb_gains=wb)

        return Frame(
            raw_dn=raw,
            linear_pre_wb=out["linear_pre_wb"],
            linear_ccm=out["linear_ccm"],
            srgb_u8=out["srgb_u8"],
            exposure_ev=ev, exposure_s=exposure_s, gain=gain, focus_pos=focus_pos,
            clipped_ratio=SensorSim.clipped_ratio(raw, self.sensor_cfg),
            meta={"illum_temp_k": self.illum_temp_k, "scene": self.scene.name,
                  "blur_px": blur_px},
        )


def apply_flicker_banding(raw: np.ndarray, exposure_s: float, mains_hz: float = 50.0,
                          readout_s: float = 0.02, black_level: float = 0.0,
                          amplitude: float = 0.25, phase: float = 0.0) -> np.ndarray:
    """交流光源下的行间亮度带纹。

    物理模型：传感器逐行曝光，每行的积分窗口 [t0, t0+T] 内光通量按
    纹波频率 f = 2*mains 变化。行平均增益 =
        1 + A * sinc(f*T) * sin(2πf(t0 + T/2))
    其中 sinc(x) = sin(πx)/(πx)。

    结论（程序里可验证）：当 T = k/f（50Hz 市电 -> 10ms 的整数倍）时
    sinc(k) = 0，带纹完全消失 —— 这正是抗闪烁要约束曝光时间的原因。
    """
    h = raw.shape[0]
    f = 2.0 * mains_hz
    sinc = np.sinc(f * exposure_s)                    # sin(πfT)/(πfT)
    t0 = np.linspace(0.0, readout_s, h, dtype=np.float64)
    # phase（单位：纹波周期）：帧时序未锁相于市电时它逐帧变化。
    # 注意 sinc 项在多帧平均下**不会**被平均掉多少 —— 相位改变的是正弦项的
    # 取值，而 sinc 只取决于曝光时间，所以相位漂移正好制造出逐帧的曝光等效波动，
    # 且在 ET 为半周期奇数倍附近最大。
    row_gain = 1.0 + amplitude * sinc * np.sin(
        2 * np.pi * f * (t0 + exposure_s / 2.0) + 2 * np.pi * phase)
    row_gain = row_gain[:, None].astype(np.float32)
    return raw * row_gain


def flicker_banding_metric(plane: np.ndarray, readout_s: float = 0.02,
                          mains_hz: float = 50.0, black_level: float = 0.0) -> float:
    """带纹强度：行均值轮廓上，**光源纹波频率处**的幅度（相对值）。

    三个设计要点，每一个都踩过坑：

    1) plane 必须是**去马赛克后的亮度图**，不能传 RAW。
       Bayer 下奇偶行的颜色组合不同（R+G 与 G+B），行均值天然交替，
       会让指标凭空多出一个巨大的"带纹"，真实信号被完全淹没。

    2) 去趋势只能用低阶多项式（这里用二次）。
       用中值/均值滤波去趋势是错的：纹波在画面上只有 2~3 个周期，
       窗口稍小就会把真实的带纹一起当成趋势滤掉。第一版用 rows/8 的
       中值窗，测出来的"带纹强度"只剩真实值的百分之几。

    3) 只在**已知的纹波频率**上取幅度，而不是全频段标准差。
       纹波频率 = 读出时间 × 纹波频率（50Hz 市电 -> 100Hz 纹波，
       20ms 读出 -> 画面上正好 2 个周期）。这样场景自身的结构
       （地平线、色卡行）落在别的频点上，不会污染指标。
    """
    rows = np.asarray(plane, dtype=np.float64)
    if rows.ndim == 3:
        rows = (0.2126 * rows[..., 0] + 0.7152 * rows[..., 1] + 0.0722 * rows[..., 2])
    prof = rows.mean(axis=1) - black_level
    n = prof.size
    if n < 16:
        return 0.0

    x = np.linspace(-1.0, 1.0, n)
    trend = np.polyval(np.polyfit(x, prof, 2), x)      # 只去二次趋势
    resid = prof - trend

    spec = np.fft.rfft(resid)
    k = np.arange(spec.size)                           # 频点 k = k 个周期/帧
    k_expected = float(readout_s * 2.0 * mains_hz)
    band = np.abs(k - k_expected) <= 1.5
    if not np.any(band):
        band = (k >= 1) & (k <= 4)
    # 取频带内的峰值 bin：频带留了 ±1.5 bin 的余量（读出时间有标称误差），
    # 但信号只落在其中一个 bin 上，取平均会把幅度摊薄 3 倍
    amp = 2.0 * float(np.max(np.abs(spec[band]))) / n    # 单边幅度
    return float(amp / max(float(np.mean(rows)), 1e-6))
