# -*- coding: utf-8 -*-
"""传感器仿真：光电转换、Bayer 采样、噪声、量化。

链路：入射辐射 -> 光电转换（满阱限制）-> 光子散粒噪声 -> 读出噪声
      -> 暗电流 -> ADC 量化（含黑电平）
这条链路决定了 3A 的两个物理边界：
  1) 满阱 -> 高光过曝后信息不可恢复（AE 高光保护的意义）
  2) 噪声 -> 低照度下必须提增益，但增益不改善 SNR（AE 的 ET/增益取舍）
"""
from dataclasses import dataclass
import numpy as np

from ..config import SensorConfig

CFA_ORDER = {"R": 0, "G": 1, "B": 2}


class SensorSim:
    def __init__(self, cfg: SensorConfig, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.pattern = np.array([[CFA_ORDER[cfg.cfa[0]], CFA_ORDER[cfg.cfa[1]]],
                                 [CFA_ORDER[cfg.cfa[2]], CFA_ORDER[cfg.cfa[3]]]], dtype=int)

    # -- Bayer 采样 -----------------------------------------------------------
    def mosaic(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        out = np.empty((h, w), dtype=np.float32)
        for i in range(2):
            for j in range(2):
                out[i::2, j::2] = rgb[i::2, j::2, self.pattern[i, j]]
        return out

    # -- 主链路 ---------------------------------------------------------------
    def expose(self, bayer_radiance: np.ndarray, exposure_factor: float,
               exposure_s: float, gain: float, add_noise: bool = True) -> np.ndarray:
        """曝光 -> DN。

        归一化约定（关键，决定了所有 3A 的输入尺度）：
          bayer_radiance 的 1.0 = 满阱对应的辐射亮度；exposure_factor 是
          **相对基准的曝光因子**（基准 = ET_REF × 增益 1.0），因此
              electrons = radiance × exposure_factor × full_well
          即 exposure_factor = 1 时，DN 归一化值正好等于入射辐射亮度，
          这样"线性域数值"与"物理亮度"一一对应，AE 的目标 0.18 才有意义。

        绝对曝光时间 exposure_s 只用于暗电流（它与真实时间成正比，
        而曝光因子是"相对量"，两者不能混用）。
        """
        cfg = self.cfg
        electrons = (bayer_radiance.astype(np.float64) * exposure_factor
                     * cfg.full_well_e + cfg.dark_current_e_per_s * exposure_s)

        # 满阱截断：饱和后信息永久丢失（AE 高光保护针对的正是这一步）
        electrons = np.clip(electrons, 0.0, cfg.full_well_e)

        if add_noise:
            # 光子散粒噪声（泊松）+ 读出噪声
            lam = np.maximum(electrons, 0.0)
            electrons = self.rng.poisson(lam).astype(np.float64)
            electrons = np.clip(electrons, 0.0, cfg.full_well_e)
            # 读出噪声的折算位置决定增益的作用（详见 SensorConfig.read_noise_model）
            nr = cfg.read_noise_e
            if getattr(cfg, "read_noise_model", "iso_less") == "gain_referred":
                nr = nr / max(gain, 1e-6)
            electrons += self.rng.normal(0.0, nr, size=electrons.shape)

        # 量化：电子 -> DN（含黑电平），12bit 定点
        dn = electrons / cfg.full_well_e * cfg.signal_dn + cfg.black_level_dn
        dn = np.clip(np.round(dn), 0, cfg.max_dn)
        return dn.astype(np.float32)

    # -- 辅助指标 -------------------------------------------------------------
    def electrons_to_dn(self, electrons: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        dn = electrons / cfg.full_well_e * cfg.signal_dn + cfg.black_level_dn
        return np.clip(np.round(dn), 0, cfg.max_dn).astype(np.float32)

    @staticmethod
    def clipped_ratio(dn: np.ndarray, cfg: SensorConfig, thresh_ratio: float = 0.98) -> float:
        """过曝像素占比（顶到接近满阱的像素比例）"""
        thresh = cfg.black_level_dn + cfg.signal_dn * thresh_ratio
        return float(np.mean(dn >= thresh))

    @staticmethod
    def snr_db(dn: np.ndarray, cfg: SensorConfig) -> float:
        """估计信噪比：以局部 8x8 平坦块的均值/标准差近似（信号 e- 与噪声 e-）。

        只用于相对比较（同一场景不同曝光策略），不作为绝对噪声指标。
        """
        import cv2
        sig = (dn - cfg.black_level_dn) / cfg.signal_dn * cfg.full_well_e
        mu = cv2.blur(sig, (8, 8))
        sd = np.sqrt(np.maximum(cv2.blur((sig - mu) ** 2, (8, 8)), 1e-6))
        valid = mu > 50.0
        if not np.any(valid):
            return 0.0
        ratio = mu[valid] / sd[valid]
        return float(20.0 * np.log10(np.median(ratio)))
