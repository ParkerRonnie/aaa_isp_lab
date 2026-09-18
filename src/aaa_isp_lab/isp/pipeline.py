# -*- coding: utf-8 -*-
"""ISP 主管线：把各模块按固定顺序串起来，并把中间结果暴露给 3A。

关键设计：管线**同时输出多个域的图像**，因为 3A 各自工作在
不同的域上，这一点经常被忽略：
    AWB 统计  -> linear_pre_wb（白平衡前，否则统计量已被自己修正过）
    AE  统计  -> 显示域亮度（与人眼感知一致，目标码值稳定）
    AF  统计  -> 去马赛克后的绿通道（信噪比最高，且与色彩无关）
"""
import numpy as np

from ..config import SensorConfig, ISPConfig
from . import modules as M

_IDENTITY_CCM = np.eye(3, dtype=np.float32)


class ISPPipeline:
    def __init__(self, sensor_cfg: SensorConfig, isp_cfg: ISPConfig, lsc_model: str = "ideal"):
        self.scfg = sensor_cfg
        self.cfg = isp_cfg
        self.lsc_model = lsc_model
        # 由 CFA 字符串解析 Bayer 排列
        cfa = sensor_cfg.cfa
        idx = {"R": 0, "G": 1, "B": 2}
        self.pattern = np.array([[idx[cfa[0]], idx[cfa[1]]],
                                 [idx[cfa[2]], idx[cfa[3]]]], dtype=int)
        self.ccm = (_IDENTITY_CCM if isp_cfg.ccm is None
                    else np.asarray(isp_cfg.ccm, dtype=np.float32))
        self.last_lsc_map = None

    # -- LSC ----------------------------------------------------------------
    def _lsc_map(self, h: int, w: int, true_strength: float) -> np.ndarray:
        if self.lsc_model == "ideal":
            assumed = true_strength
        elif self.lsc_model == "radial2":       # 标定不准：只补了一部分
            assumed = true_strength * 0.7
        elif self.lsc_model == "radial2_over":  # 标定过冲：补过头，边缘发亮
            assumed = true_strength * 1.25
        else:
            raise ValueError(self.lsc_model)
        return M.lsc_radial_model(h, w, assumed, max_gain=self.cfg.lsc_max_gain)

    # -- 主流程 --------------------------------------------------------------
    def process(self, raw_dn: np.ndarray, wb_gains: np.ndarray = None,
                lsc_gain_map: np.ndarray = None, true_vignetting: float = None) -> dict:
        cfg = self.cfg

        lin = M.black_level_correct(raw_dn, self.scfg.black_level_dn, self.scfg.signal_dn)

        if cfg.enable_lsc:
            if lsc_gain_map is None:
                strength = 0.45 if true_vignetting is None else true_vignetting
                lsc_gain_map = self._lsc_map(raw_dn.shape[0], raw_dn.shape[1], strength)
            # LSC 在 RAW 域做：每个像素按其 CFA 通道取增益
            lin = M.apply_lsc_bayer(lin, lsc_gain_map, self.pattern)
        self.last_lsc_map = lsc_gain_map

        rgb = M.demosaic(lin, self.pattern, cfg.demosaic)
        linear_pre_wb = rgb.copy()

        gains = np.ones(3, dtype=np.float32) if wb_gains is None else np.asarray(wb_gains, np.float32)
        rgb_wb = M.apply_wb(rgb, gains)

        rgb_ccm = M.apply_ccm(rgb_wb, self.ccm) if cfg.enable_ccm else rgb_wb

        display = M.tone_map(rgb_ccm, cfg.tone_mode, cfg.shoulder, cfg.contrast)
        srgb_u8 = np.clip(display * 255.0 + 0.5, 0, 255).astype(np.uint8)
        srgb_u8 = M.denoise_sharpen(srgb_u8, cfg.enable_denoise, cfg.denoise_sigma,
                                    cfg.enable_sharpen, cfg.sharpen_amount)

        return {
            "linear_pre_wb": linear_pre_wb,      # AWB 域
            "linear_wb": rgb_wb,
            "linear_ccm": rgb_ccm,               # AE/AF 评价域
            "display_linear": display,
            "srgb_u8": srgb_u8,
            "lsc_map": lsc_gain_map,
        }
