# -*- coding: utf-8 -*-
"""C++ 统计通路（可选加速组件）。

没编译共享库时，这里所有入口都会优雅不可用，项目照常跑纯 Python 路径。
编译：`python tools/build_native.py`
"""
from . import backend, loader  # noqa: F401

__all__ = ["backend", "loader", "status", "is_available", "metering_fn"]

# 只导出不会抛异常的入口；需要库的函数（metering_fn 等）在 backend 里。
status = loader.status
is_available = backend.available
metering_fn = backend.metering_fn
