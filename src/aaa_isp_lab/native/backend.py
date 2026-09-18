# -*- coding: utf-8 -*-
"""把 C++ 统计库包装成与 Python 版**同签名、同返回结构**的可调用对象。

关键约定：返回的 dict 形状必须与 `aaa.ae.metering_metric` 完全一致
（`metric` / `target` / `detail{mode, code_value, clip_ratio[, clip_zone]}`），
这样 `AEController` 只要换一个注入的函数就能切后端，控制律一行都不用改。

`code_value` 在 **Python 侧**用同一个 `linear_to_srgb` 转换 —— 两条路径的差别
就只剩 metric 本身，不会把 pow 的实现差异混进等价性比较里。
"""
import ctypes as C

import numpy as np

from ..aaa.ae import _HIGHLIGHT_TARGET, _METER_LABELS
from ..color_science import linear_to_srgb
from . import api, loader

# 上下文按 (模式, 精度, 形状, 分区, 关键参数) 缓存复用。
# 这也顺便模拟了真实驱动里"分辨率变了要重建表"的行为。
_CTX_CACHE = {}
_MAX_CTX = 8


def available():
    return loader.try_load() is not None


def _check_array(arr, name="luma"):
    """调用前的预校验。

    这一层**不是**为了性能，而是为了让错误以 ValueError 的形式出现，
    而不是让 C 读到飞、把宿主进程打崩（ctypes 段错误会直接杀掉 pytest）。
    真正的校验在 C 侧也有一份。
    """
    if arr.dtype != np.float32:
        raise ValueError(f"{name} 必须是 float32，得到 {arr.dtype}")
    if arr.ndim != 2:
        raise ValueError(f"{name} 必须是 2 维，得到 {arr.ndim} 维")
    if not arr.flags["C_CONTIGUOUS"] and min(arr.strides) <= 0:
        raise ValueError(f"{name} 含负步长（反向视图），不支持")
    return arr


def metering_fn(mode, precision="f32", hist_bins=1024, bit_depth=16, impl="c"):
    """返回一个 `fn(frame, cfg) -> dict` 的测光函数。

    impl="python" 时直接返回项目原有的 `metering_metric`（**同一个函数对象**，
    所以默认路径按构造即逐位不变）。
    """
    if impl == "python":
        from ..aaa.ae import metering_metric
        return metering_metric

    lib = api._require_lib()

    def _fn(frame, cfg):
        luma = np.ascontiguousarray(frame.luma_linear, dtype=np.float32)
        _check_array(luma)
        rows, cols = luma.shape
        key = (mode, precision, rows, cols, cfg.zones, cfg.spot_ratio,
               cfg.center_weight, cfg.zone_sigma, cfg.highlight_weight, hist_bins, bit_depth)
        ctx = _CTX_CACHE.get(key)
        if ctx is None:
            if len(_CTX_CACHE) >= _MAX_CTX:
                _CTX_CACHE.clear()
            params = api.make_params(cfg, mode, precision, hist_bins, bit_depth)
            handle = C.c_void_p()
            rc = lib.aaa_ae_ctx_create(C.byref(params), rows, cols, C.byref(handle))
            if rc != api.STATUS_MSG[0] and rc != 0:
                raise RuntimeError(f"aaa_ae_ctx_create 失败: {api.STATUS_MSG.get(rc, rc)}")
            ctx = (handle, params)
            _CTX_CACHE[key] = ctx

        handle, params = ctx
        res = api.AEResult()
        if precision == "q16":
            # 量化尺度必须与 C 侧一致：C 用 q_scale_ = 2^bit_depth - 1。
            # 传成固定 65535 的话位宽扫描会完全测不出差别（踩过一次）。
            scale = float((1 << int(bit_depth)) - 1)
            q = np.rint(np.clip(luma, 0.0, 1.0) * scale).astype(np.uint16)
            rc = lib.aaa_ae_meter_u16(handle, q.ctypes.data, rows, cols,
                                      cols, 1, C.byref(res))
            del q
        else:
            rc = lib.aaa_ae_meter_f32(handle, luma.ctypes.data, rows, cols,
                                      cols, 1, C.byref(res))
        if rc != 0:
            raise RuntimeError(f"aaa_ae_meter 失败: {api.STATUS_MSG.get(rc, rc)}")

        metric = float(res.metric)
        target = _HIGHLIGHT_TARGET if mode == "highlight_priority" else float(cfg.target_linear)
        detail = {
            "mode": _METER_LABELS[mode],
            "code_value": float(linear_to_srgb(np.asarray(metric)) * 255.0),
            "clip_ratio": float(frame.clipped_ratio),
        }
        if mode == "evaluative":
            detail["clip_zone"] = float(res.aux)
        return {"metric": metric, "target": target, "detail": detail}

    return _fn


def awb_stats_fn(precision="f32", hist_bins=1024, bit_depth=16, impl="c", need=None):
    """返回一个 `fn(linear_pre_wb, cfg) -> AWBStatistics` 的统计函数。

    直接喂给 `AWBEstimator(cfg, stats_fn=...)` —— **融合数学仍是 Python 侧那份**
    （awb.fuse_illuminants），所以两条通路只在统计环节分叉，决策环节完全一致。
    """
    if impl == "python":
        from ..aaa.awb import compute_statistics
        return compute_statistics

    from ..aaa.awb import AWBStatistics
    lib = api._require_lib()

    def _fn(linear, cfg):
        arr = np.ascontiguousarray(linear, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"linear_pre_wb 必须是 (H,W,3)，得到 {arr.shape}")
        rows, cols = arr.shape[:2]
        key = ("awb", precision, rows, cols, cfg.near_gray_val_min,
               cfg.near_gray_sat_max, cfg.gray_edge_thresh, cfg.sog_p, hist_bins)
        ctx = _CTX_CACHE.get(key)
        if ctx is None:
            if len(_CTX_CACHE) >= _MAX_CTX:
                _CTX_CACHE.clear()
            params = api.make_awb_params(cfg, precision, hist_bins, bit_depth, need)
            handle = C.c_void_p()
            rc = lib.aaa_awb_ctx_create(C.byref(params), rows, cols, C.byref(handle))
            if rc != 0:
                raise RuntimeError(f"aaa_awb_ctx_create 失败: {api.STATUS_MSG.get(rc, rc)}")
            ctx = (handle, params)
            _CTX_CACHE[key] = ctx

        handle, params = ctx
        res = api.AWBStats()
        # 步长必须从数组真实 strides 取：(H,W,3) 的连续数组里通道是**交织**的，
        # stride_col 是 3 而不是 1（这里踩过一次：传成 1 会让 C 读到错位的
        # 通道数据，表现为 n_valid 只剩 1 个像素）。
        sr, sc, schan = (s // arr.itemsize for s in arr.strides)
        if precision == "q16":
            q = np.rint(np.clip(arr, 0.0, 1.0) * 65535.0).astype(np.uint16)
            rc = lib.aaa_awb_stats_u16(handle, q.ctypes.data, rows, cols,
                                       sr, sc, schan, C.byref(res))
            del q
        else:
            rc = lib.aaa_awb_stats_f32(handle, arr.ctypes.data, rows, cols,
                                       sr, sc, schan, C.byref(res))
        if rc != 0:
            raise RuntimeError(f"aaa_awb_stats 失败: {api.STATUS_MSG.get(rc, rc)}")

        est = {
            "gray_world": np.array(res.gray_world, dtype=np.float64),
            "white_patch": np.array(res.white_patch, dtype=np.float64),
            "gray_edge": np.array(res.gray_edge, dtype=np.float64),
            "shades_of_gray": np.array(res.shades_of_gray, dtype=np.float64),
        }
        return AWBStatistics(estimators=est, n_valid=int(res.n_valid),
                             sat_mean=float(res.sat_mean), frac_ge=float(res.frac_ge))

    return _fn


def native_status():
    """给报告用的状态报告（含构建信息，性能数字必须带上它才有意义）。"""
    return loader.status()
