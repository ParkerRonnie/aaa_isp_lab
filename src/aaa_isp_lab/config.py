# -*- coding: utf-8 -*-
"""集中式 tuning 参数表。

模拟真实 ISP/3A 工程的参数管理方式：所有可调量集中在一处，
便于做 sweep 实验与版本对比（否则调参结果不可追溯）。
"""
from dataclasses import dataclass, field, asdict
from typing import Tuple, Optional
import json


# -----------------------------------------------------------------------------
# 传感器
# -----------------------------------------------------------------------------
@dataclass
class SensorConfig:
    width: int = 480
    height: int = 360
    bit_depth: int = 12
    black_level_dn: int = 64                  # 光学黑电平 (DN)
    full_well_e: float = 12000.0              # 满阱电子数
    read_noise_e: float = 1.8                 # 读出噪声 (e- rms)
    dark_current_e_per_s: float = 30.0        # 暗电流 (e-/s)
    # 读出噪声参考位置，决定"提高增益到底有没有用"：
    #   'iso_less'     读出噪声折算到**输入端**且与增益无关（理想 ISO 无关传感器）
    #                  -> 增益只放大同一份光子噪声，暗部 SNR 不变
    #   'gain_referred'读出噪声在**增益之后**加入（真实传感器的典型情况：
    #                  噪声主要来自源跟随器与 ADC），折算到输入端要除以增益
    #                  -> 高增益压低输入折算噪声，暗部 SNR 变好
    # 这就是 ISO 存在的意义，也是"增益无用论"不成立的区域。
    read_noise_model: str = "iso_less"
    min_exposure_s: float = 1.0 / 8000.0
    max_exposure_s: float = 1.0 / 30.0
    max_analog_gain: float = 16.0             # 模拟增益上限
    cfa: str = "RGGB"
    readout_s: float = 0.020                  # 逐行读出总时长（用于闪烁仿真）
    vignetting_strength: float = 0.45         # 镜头阴影强度（RAW 域）
    # 光谱串扰（CCM 存在的原因）。真实传感器的 CFA 不是理想窄带滤光片：
    # R 像素也会收到一部分绿光，反之亦然。行和为 1 表示不改变整体亮度。
    # 没有它，传感器通道就"恰好等于"场景反射率，CCM 会退化成单位阵，
    # 颜色题变成一道假题 —— 这是仿真里最容易骗到自己的地方。
    crosstalk: Optional[Tuple[Tuple[float, ...], ...]] = (
        (0.88, 0.09, 0.03),
        (0.07, 0.88, 0.05),
        (0.02, 0.12, 0.86),
    )

    @property
    def max_dn(self) -> float:
        return float((1 << self.bit_depth) - 1)

    @property
    def signal_dn(self) -> float:
        """白电平对应的信号 DN（扣除黑电平）"""
        return self.max_dn - self.black_level_dn


# -----------------------------------------------------------------------------
# ISP
# -----------------------------------------------------------------------------
@dataclass
class ISPConfig:
    enable_lsc: bool = True
    lsc_max_gain: float = 3.0                 # 阴影校正增益上限，防边缘噪声放大
    demosaic: str = "color_diff"              # 'bilinear' | 'color_diff'
    enable_ccm: bool = True
    # 默认 CCM = 单位阵；标定后由 calibrate_ccm.py 覆盖
    ccm: Optional[Tuple[Tuple[float, ...], ...]] = None
    tone_mode: str = "srgb"                   # 'gamma22' | 'srgb'
    shoulder: float = 0.85                    # 高光拐点（线性域），1.0 表示不压缩
    contrast: float = 1.0                     # S 曲线强度
    enable_denoise: bool = True
    denoise_sigma: float = 0.35               # 双边滤波 sigmaColor（8bit 域）
    enable_sharpen: bool = True
    sharpen_amount: float = 0.45


# -----------------------------------------------------------------------------
# AE
# -----------------------------------------------------------------------------
@dataclass
class AEConfig:
    # 目标亮度：线性域 18% 中灰（对应 sRGB 8bit 约 118）
    target_linear: float = 0.18
    metering: str = "evaluative"              # average|center|spot|evaluative|highlight_priority
    center_weight: float = 2.0                # center 测光的中心/边缘权重比
    spot_ratio: float = 0.15                  # spot 测光的窗口占画面比
    zones: Tuple[int, int] = (5, 5)           # evaluative 分区数
    zone_sigma: float = 0.45                  # 分区权重的高斯半径（归一化坐标）
    highlight_weight: float = 0.60            # evaluative 中高光保护权重
    damping: float = 0.70                     # 步长阻尼，1.0 = 全步长（易振荡）
    # 过曝补偿：像素一旦饱和，亮度就不再随曝光增加，测得的误差被严重低估，
    # 控制器会"以为快到了"而走得极慢。按过曝像素比例放大步长可以救回来。
    clip_boost: float = 1.6
    max_iters: int = 12
    converge_thresh_ev: float = 0.02          #  |ΔEV| 低于该值即认为收敛
    # 相对基准曝光的 EV 范围，必须与相机的物理可实现范围一致
    # （由 sim.camera.ev_limits(SensorConfig) 计算：默认约 [-7.06, +5.00]）
    ev_min: float = -7.0
    ev_max: float = 5.0
    # 曝光策略：'shutter_priority' 优先用快门，'gain_priority' 优先用增益
    priority: str = "shutter_priority"
    # 抗闪烁：光源纹波频率 = 2 × 市电频率
    anti_flicker: bool = False
    mains_hz: float = 50.0
    # 运动模糊：快门越长越糊（元/秒），用于量化 ET/增益的取舍
    motion_speed_px_per_s: float = 300.0


# -----------------------------------------------------------------------------
# AWB
# -----------------------------------------------------------------------------
@dataclass
class AWBConfig:
    method: str = "fusion"                    # gray_world|white_patch|gray_edge|shades_of_gray|fusion
    sog_p: float = 6.0                        # Shades-of-Gray 的 Minkowski p
    gray_edge_thresh: float = 0.10            # 灰边检测的梯度阈值
    near_gray_sat_max: float = 0.25           # 近中性像素的白点比例上限
    near_gray_val_min: float = 0.15           # 近中性像素的最低亮度（避开暗部噪声）
    fusion_weights: Tuple[float, float, float] = (0.35, 0.30, 0.35)  # (gray_world, gray_edge, sog)
    constrain_planckian: bool = True          # 把估计光源约束到普朗克轨迹附近
    cct_min: float = 2000.0
    cct_max: float = 12000.0
    max_duv: float = 0.03
    clip_guard: float = 0.02                  # 过曝像素占比超过该值则弃用 white_patch


# -----------------------------------------------------------------------------
# AF
# -----------------------------------------------------------------------------
@dataclass
class AFConfig:
    measure: str = "tenengrad"                 # brenner|tenengrad|laplacian_var|sml|fft_energy
    roi: str = "center"                        # full|center|multi
    roi_ratio: float = 0.5                     # center ROI 边长占比
    strategy: str = "coarse_to_fine"           # sweep|hill_climb|coarse_to_fine|golden_section
    # 粗扫只负责"进到峰附近"，精度交给细扫。粗扫步数少、细扫步数多才是
    # 省帧数的关键：6 粗 + 9 细 = 15 帧能达到 0.05 的定位分辨率，
    # 而全扫描要达到同样分辨率需要 21 帧。
    coarse_steps: int = 6
    fine_steps: int = 9
    sweep_steps: int = 21          # 全扫描基线（21 帧 -> 定位分辨率 0.05）
    max_iters: int = 30
    # 镜头行程归一化为 [0,1]，blur_radius = max_blur_px * |z - z_true|
    lens_range: Tuple[float, float] = (0.0, 1.0)
    # 行程两端对应的最大弥散圈半径。这个值决定了"对焦定位分辨率的物理下限"：
    # 离焦小于 0.5 px 时 PSF 退化为 delta（图像在像素级上确实没变化），
    # 因此峰顶必然存在一段宽度约 0.5/max_blur_px 的平台。
    # 8 px 是权衡：平台足够窄（~0.06 行程），远端又不会糊到没有梯度。
    max_blur_px: float = 8.0
    # 爬山法步长
    hill_step: float = 0.08
    hill_min_step: float = 0.004
    hill_patience: int = 2


# -----------------------------------------------------------------------------
# 实验
# -----------------------------------------------------------------------------
@dataclass
class LabConfig:
    seed: int = 2026
    size: Tuple[int, int] = (480, 360)         # (w, h)

    def dump(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2, default=str)
