# -*- coding: utf-8 -*-
"""共享库的查找与加载。

**这个模块的任何函数都不许抛异常。** 理由：C++ 库是可选的加速组件，
没编译时整个项目必须照常跑纯 Python 路径；而测试是按"能否加载"条件注册的
（见 tests/test_native.py），一旦这里抛异常，`run_all()` 会把它算作失败，
CI 在没有编译器的机器上就会红。
"""
import ctypes
import os

ABI_VERSION = 1

_LIB_NAMES = ("aaa_stats.dll", "libaaa_stats.so", "libaaa_stats.dylib")


def _candidate_paths():
    """按优先级给出候选路径。

    顺序：环境变量（调试/CI 用）-> 包目录（构建脚本的默认输出）-> 仓库的
    native/build（独立构建产物）。
    """
    env = os.environ.get("AAA_STATS_LIB")
    if env:
        yield os.path.abspath(env)
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    for name in _LIB_NAMES:
        yield os.path.join(pkg_dir, name)
    root = os.path.dirname(os.path.dirname(os.path.dirname(pkg_dir)))
    for name in _LIB_NAMES:
        yield os.path.join(root, "native", "build", name)


class Library:
    """已加载的共享库。属性都是 ctypes 函数对象，签名在 api.py 里配置。"""

    def __init__(self, cdll, path):
        self.cdll = cdll
        self.path = path

    def __getattr__(self, name):
        return getattr(self.cdll, name)


_CACHE = {}


def try_load(force=False):
    """尝试加载共享库。成功返回 Library，失败返回 None。**绝不抛异常。**"""
    if not force and "lib" in _CACHE:
        return _CACHE["lib"]

    lib = None
    for path in _candidate_paths():
        if not path or not os.path.isfile(path):
            continue
        try:
            cdll = ctypes.CDLL(path)
            cdll.aaa_abi_version.restype = ctypes.c_int32
            if int(cdll.aaa_abi_version()) != ABI_VERSION:
                continue          # 版本不匹配：当作不可用，而不是带着旧 ABI 跑
            cdll.aaa_null_call.restype = ctypes.c_int32
            cdll.aaa_build_info.restype = ctypes.c_char_p
            cdll.aaa_status_string.restype = ctypes.c_char_p
            cdll.aaa_status_string.argtypes = [ctypes.c_int32]
            lib = Library(cdll, path)
            break
        except Exception:         # noqa: BLE001  —— 任何失败都只是"这个候选不可用"
            continue

    _CACHE["lib"] = lib
    return lib


def status():
    """给报告与 CI 用的状态报告。

    注意 `available` 必须与 `lib_path` 严格一致 —— 这是条件注册机制的自我守卫：
    如果"编译悄悄失败"表现为"测试悄悄变少"，那是最难发现的一种假绿。
    测试 test_native_backend_status_is_honest 就钉住这一点。
    """
    lib = try_load()
    out = {
        "available": lib is not None,
        "lib_path": None,
        "reason": None,
        "abi_version": None,
        "build_info": None,
        "candidates": [p for p in _candidate_paths()],
    }
    if lib is None:
        out["reason"] = "未找到可加载的共享库（运行 python tools/build_native.py 编译）"
        return out
    out["lib_path"] = lib.path
    out["abi_version"] = int(lib.aaa_abi_version())
    try:
        out["build_info"] = lib.aaa_build_info().decode("utf-8", "replace")
    except Exception:            # noqa: BLE001
        out["build_info"] = None
    return out
