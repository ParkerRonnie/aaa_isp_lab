# -*- coding: utf-8 -*-
"""ctypes 结构体与调用封装。

ctypes 的结构体布局必须与 native/include/aaa_stats.h 严格一致 —— C 侧有
static_assert 钉住尺寸，这里在导入时也校验一遍，两边任何一边改了都会被立刻发现。
"""
import ctypes as C

import numpy as np

from . import loader

# --- 与 aaa_stats.h 的 enum 一一对应 ---------------------------------------
METER_MODES = {
    "average": 0, "center": 1, "spot": 2, "evaluative": 3, "highlight_priority": 4,
}
PRECISIONS = {"f64": 0, "f32": 1, "q16": 2}

STATUS_MSG = {
    0: "ok", -1: "null pointer", -2: "invalid shape/stride",
    -3: "invalid mode or precision", -4: "parameter out of range",
    -5: "internal error",
}
# 与 aaa_stats.h 的 enum 一一对应
ERR_OK, ERR_NULL, ERR_SHAPE, ERR_MODE, ERR_RANGE, ERR_INTERNAL = 0, -1, -2, -3, -4, -5


class AEParams(C.Structure):
    _fields_ = [
        ("mode", C.c_int32), ("precision", C.c_int32),
        ("zones_y", C.c_int32), ("zones_x", C.c_int32),
        ("hist_bins", C.c_int32), ("bit_depth", C.c_int32),
        ("flags", C.c_int32), ("reserved0", C.c_int32),
        ("center_ratio", C.c_double), ("spot_ratio", C.c_double),
        ("zone_sigma", C.c_double), ("highlight_weight", C.c_double),
        ("clip_level", C.c_double), ("reserved1", C.c_double),
        ("zone_w", C.c_void_p), ("zone_w_len", C.c_int32), ("reserved2", C.c_int32),
    ]


class AEResult(C.Structure):
    _fields_ = [
        ("metric", C.c_double), ("aux", C.c_double),
        ("n_used", C.c_int64), ("hist_peak_bin", C.c_int64),
        ("status", C.c_int32), ("reserved0", C.c_int32),
    ]


class AWBParams(C.Structure):
    _fields_ = [
        ("precision", C.c_int32), ("hist_bins", C.c_int32), ("bit_depth", C.c_int32),
        ("need_gray_world", C.c_int32), ("need_white_patch", C.c_int32),
        ("need_gray_edge", C.c_int32), ("need_sog", C.c_int32), ("reserved0", C.c_int32),
        ("clip_level", C.c_double), ("near_gray_val_min", C.c_double),
        ("near_gray_sat_max", C.c_double), ("gray_edge_thresh", C.c_double),
        ("white_patch_q", C.c_double), ("sog_p", C.c_double),
    ]


class AWBStats(C.Structure):
    _fields_ = [
        ("gray_world", C.c_double * 3), ("white_patch", C.c_double * 3),
        ("gray_edge", C.c_double * 3), ("shades_of_gray", C.c_double * 3),
        ("sat_mean", C.c_double), ("frac_ge", C.c_double), ("edge_mag_mean", C.c_double),
        ("n_valid", C.c_int64), ("n_gray_edge", C.c_int64), ("n_pixels", C.c_int64),
        ("status", C.c_int32), ("reserved0", C.c_int32),
    ]


# 布局守卫：与 C 侧 static_assert 的期望值一致。改了任何一边这里都会炸。
assert C.sizeof(AEParams) == 96, f"AEParams 布局不符: {C.sizeof(AEParams)} != 96"
assert C.sizeof(AEResult) == 40, f"AEResult 布局不符: {C.sizeof(AEResult)} != 40"
assert C.sizeof(AWBParams) == 80, f"AWBParams 布局不符: {C.sizeof(AWBParams)} != 80"
assert C.sizeof(AWBStats) == 152, f"AWBStats 布局不符: {C.sizeof(AWBStats)} != 152"


def _configure(lib):
    """配置函数签名。只做一次（CDLL 的属性赋值是幂等的，但省点开销）。"""
    if getattr(lib, "_aaa_configured", False):
        return
    lib.aaa_ae_ctx_create.argtypes = [C.POINTER(AEParams), C.c_int32, C.c_int32,
                                      C.POINTER(C.c_void_p)]
    lib.aaa_ae_ctx_create.restype = C.c_int32
    lib.aaa_ae_ctx_destroy.argtypes = [C.c_void_p]
    lib.aaa_ae_ctx_destroy.restype = None
    lib.aaa_ae_meter_f32.argtypes = [C.c_void_p, C.c_void_p, C.c_int32, C.c_int32,
                                     C.c_int32, C.c_int32, C.POINTER(AEResult)]
    lib.aaa_ae_meter_f32.restype = C.c_int32
    lib.aaa_ae_meter_u16.argtypes = [C.c_void_p, C.c_void_p, C.c_int32, C.c_int32,
                                     C.c_int32, C.c_int32, C.POINTER(AEResult)]
    lib.aaa_ae_meter_u16.restype = C.c_int32
    lib.aaa_awb_ctx_create.argtypes = [C.POINTER(AWBParams), C.c_int32, C.c_int32,
                                       C.POINTER(C.c_void_p)]
    lib.aaa_awb_ctx_create.restype = C.c_int32
    lib.aaa_awb_ctx_destroy.argtypes = [C.c_void_p]
    lib.aaa_awb_ctx_destroy.restype = None
    lib.aaa_awb_stats_f32.argtypes = [C.c_void_p, C.c_void_p, C.c_int32, C.c_int32,
                                      C.c_int32, C.c_int32, C.c_int32,
                                      C.POINTER(AWBStats)]
    lib.aaa_awb_stats_f32.restype = C.c_int32
    lib.aaa_awb_stats_u16.argtypes = [C.c_void_p, C.c_void_p, C.c_int32, C.c_int32,
                                      C.c_int32, C.c_int32, C.c_int32,
                                      C.POINTER(AWBStats)]
    lib.aaa_awb_stats_u16.restype = C.c_int32
    try:
        lib._aaa_configured = True
    except Exception:            # noqa: BLE001  —— CDLL 可能不允许设属性，无妨
        pass


def make_params(cfg, mode, precision="f32", hist_bins=1024, bit_depth=16):
    """把项目的 AEConfig 翻译成 C 侧参数结构体。"""
    from ..aaa.ae import _HIGHLIGHT_TARGET, _METER_LABELS   # noqa: F401
    p = AEParams()
    p.mode = METER_MODES[mode]
    p.precision = PRECISIONS[precision]
    p.zones_y, p.zones_x = int(cfg.zones[0]), int(cfg.zones[1])
    p.hist_bins = int(hist_bins)
    p.bit_depth = int(bit_depth)
    p.flags = 0
    p.center_ratio = float(cfg.center_weight)
    p.spot_ratio = float(cfg.spot_ratio)
    p.zone_sigma = float(cfg.zone_sigma)
    p.highlight_weight = float(cfg.highlight_weight)
    p.clip_level = 0.95          # Python 里是硬编码常量，这里参数化
    p.zone_w = None
    p.zone_w_len = 0
    return p


def make_awb_params(cfg, precision="f32", hist_bins=1024, bit_depth=16, need=None):
    """把项目的 AWBConfig 翻译成 C 侧参数结构体。

    `need` 控制按需计算哪个估计器 —— Python 现状是**无条件算全部 4 个**，
    哪怕 method="gray_world" 只用 1 个（4 倍无用功）。默认全开，
    以便与 numpy 后端做逐项对照；基准里会单独测"只算需要的"那一档。
    """
    need = need or {}
    p = AWBParams()
    p.precision = PRECISIONS[precision]
    p.hist_bins = int(hist_bins)
    p.bit_depth = int(bit_depth)
    p.need_gray_world = 1 if need.get("gray_world", True) else 0
    p.need_white_patch = 1 if need.get("white_patch", True) else 0
    p.need_gray_edge = 1 if need.get("gray_edge", True) else 0
    p.need_sog = 1 if need.get("sog", True) else 0
    p.clip_level = 0.98                # Python 里是硬编码常量，这里参数化
    p.near_gray_val_min = float(cfg.near_gray_val_min)
    p.near_gray_sat_max = float(cfg.near_gray_sat_max)
    p.gray_edge_thresh = float(cfg.gray_edge_thresh)
    p.white_patch_q = 99.5             # Python 里是硬编码的百分位
    p.sog_p = float(cfg.sog_p)
    return p


def _require_lib():
    lib = loader.try_load()
    if lib is None:
        raise RuntimeError(
            "C++ 统计库不可用。编译：python tools/build_native.py\n"
            "（这是可选组件，纯 Python 路径不受影响）")
    _configure(lib)
    return lib
