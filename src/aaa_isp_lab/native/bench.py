# -*- coding: utf-8 -*-
"""性能测量方法学。

这个项目原本**没有任何 benchmark**，所以"C 更快"是一个全新的证据标准，
必须自带方法学，否则就是项目自己最反对的那种无依据数字。

三条规矩：

1. **报中位数，不报最小值**。最小值只反映"运气最好的一次"，对偶发抖动
   毫无抵抗力；中位数配 p10/p90/MAD 才能看出分散度。
2. **ctypes 调用开销要测不要估**。用 `aaa_null_call()` 实测每次调用的地板，
   在报告里作为水平虚线，并明确标出地板占比超过 5% 的核 —— 那些核的数字
   要打折扣看。
3. **测试里绝不断言任何性能数字**。性能只能作为"数据 + 分散度 + 环境"
   呈现，不能作为 gate —— 否则就是在测随机数。
"""
import platform
import statistics
import sys
import time

import numpy as np


def environment():
    """环境元数据。**必须随数字一起报**，否则数字没有意义。"""
    import cv2
    from . import loader
    st = loader.status()
    out = {
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "cv2": cv2.__version__,
        "native_available": bool(st.get("available")),
        "native_build": st.get("build_info"),
        "native_path": st.get("lib_path"),
    }
    return out


def measure(fn, warmup=20, repeats=100):
    """跑 fn 若干次，返回耗时样本的统计量（单位：微秒）。

    固定同一份输入、不重新生成随机帧 —— 所有实现看到同一份内存布局，
    也不会把 RNG 的开销算进去。
    """
    for _ in range(int(warmup)):
        fn()
    samples = []
    for _ in range(int(repeats)):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1e6)
    a = np.asarray(samples, dtype=np.float64)
    med = float(np.median(a))
    return {
        "median_us": med,
        "p10_us": float(np.percentile(a, 10)),
        "p90_us": float(np.percentile(a, 90)),
        # MAD 用中位数做中心，比标准差更抗离群
        "mad_us": float(np.median(np.abs(a - med))),
        "min_us": float(a.min()),
        "n": int(a.size),
    }


def null_call_floor(warmup=200, repeats=2000):
    """ctypes 调用开销地板：调一个什么都不做的导出函数。

    这个数决定了哪些核的测量是有意义的 —— 如果一个核只要 1us 而地板是
    0.2us，那测出来的主要是 ctypes 而不是 C。
    """
    from . import api
    lib = api._require_lib()
    lib.aaa_null_call.restype = api.C.c_int32
    return measure(lib.aaa_null_call, warmup=warmup, repeats=repeats)


def summarize_ratio(a_us, b_us):
    """b 相对 a 的加速比（a/b）。"""
    if b_us <= 0:
        return float("nan")
    return a_us / b_us


def frame_budget_pct(us, fps=30.0):
    """耗时占帧预算的百分比（默认 33.3ms @ 30fps）。"""
    return us / (1e6 / fps) * 100.0


def extrapolate_us(us, from_px, to_px):
    """按像素数线性外推到目标分辨率。

    **这是外推，不是实测** —— 忽略了缓存层次与带宽的变化，
    只能当量级估计。报告里必须原样这么写。
    """
    return us * (to_px / from_px)
