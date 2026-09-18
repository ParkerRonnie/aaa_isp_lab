# -*- coding: utf-8 -*-
"""编译 C++ 统计库（可选组件）。

为什么不走 setuptools Extension：
    Windows 上的 CPython 是 MSVC 构建的，而开发机只有 MinGW g++，
    MinGW 编译的扩展模块无法可靠链接 MSVC 的 Python。改成
    「C++ 核心 + extern "C" 的 C ABI + 共享库」之后，两个工具链都能编，
    而且不引入任何 pip 依赖（保住「依赖以 pyproject.toml 为唯一来源」）。

    C++ 源码在 native/，编出来的库放进包目录，Python 侧用 ctypes 加载。
    没编译时整个项目照常跑纯 Python 路径（见 src/aaa_isp_lab/native/loader.py）。

用法：
    python tools/build_native.py                 # 编共享库到包目录
    python tools/build_native.py --with-bench    # 同时编独立可执行（脱离 Python 的自检/基准）
    python tools/build_native.py --check         # 只校验已存在的库能加载且 ABI 版本对（CI 硬 gate）
    python tools/build_native.py --print-cmd     # 只打印将执行的命令（CI 日志留痕）
    python tools/build_native.py --clean / --cxx clang++ / --opt -O2
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NATIVE = os.path.join(ROOT, "native")
PKG_DIR = os.path.join(ROOT, "src", "aaa_isp_lab", "native")
BUILD_DIR = os.path.join(NATIVE, "build")

# 退出码约定（CI 据此判断）：0 成功 / 2 找不到编译器 / 3 库加载或 ABI 校验失败
EXIT_OK, EXIT_NO_COMPILER, EXIT_BAD_LIB = 0, 2, 3

# 编译选项。三条**硬性禁令**，写在这里也写进 native/README.md：
#   -ffast-math  会破坏 NaN 语义与浮点等价性断言（本项目的等价性测试依赖 IEEE 语义）
#   -march=native 本地 g++ 8.1 与 CI 的 gcc 会用不同指令集，跨机性能数字就不可比了
#   OpenMP/多线程 与 numpy 的线程策略不可比，且引入不可复现的方差
# 另外不用 std::filesystem（g++ 8 需要额外 -lstdc++fs，交叉编译容易踩）。
BASE_FLAGS = ["-std=c++17", "-Wall", "-Wextra", "-Wno-unused-parameter",
              # -DNDEBUG 不只是关 assert：aaa_build_info() 会把它报成 release/debug，
              # 而这个字符串要原样写进报告。不传的话报告里会写着 debug 却在跑 -O3。
              "-DNDEBUG"]


def find_compiler(explicit=None):
    for cand in ([explicit] if explicit else []) + [os.environ.get("CXX"),
                                                     "g++", "clang++", "c++"]:
        if not cand:
            continue
        path = shutil.which(cand)
        if path:
            return path
    return None


def sources():
    return sorted(glob.glob(os.path.join(NATIVE, "src", "*.cpp")))


def shared_flags():
    if sys.platform == "win32":
        # 必须用 -static（全静态），**不是**只要 -static-libgcc -static-libstdc++。
        # 实测踩到：只加后两个开关时，DLL 仍然依赖 libwinpthread-1.dll ——
        # 这个 MinGW 构建下 std::string / std::vector / 异常处理会拉进 pthread，
        # 而 CPython 进程的 PATH 上没有它，ctypes.CDLL 会报
        # "Could not find module ... (or one of its dependencies)"。
        # 用 objdump -p <dll> | grep "DLL Name" 可以核验：应当只剩系统 DLL。
        return ["-shared", "-static", "-static-libgcc", "-static-libstdc++"]
    return ["-shared", "-fPIC", "-fvisibility=hidden",
            "-static-libstdc++", "-static-libgcc"]


def lib_name():
    if sys.platform == "win32":
        return "aaa_stats.dll"
    if sys.platform == "darwin":
        return "libaaa_stats.dylib"
    return "libaaa_stats.so"


def lib_path():
    return os.path.join(PKG_DIR, lib_name())


def build_shared(cxx, opt, dry=False):
    cmd = ([cxx] + BASE_FLAGS + [opt, "-DAAA_STATS_BUILD",
                                 "-I", os.path.join(NATIVE, "include"),
                                 "-I", os.path.join(NATIVE, "src")]
           + sources() + shared_flags() + ["-o", lib_path()])
    return run(cmd, dry)


def build_standalone(cxx, opt, dry=False):
    os.makedirs(BUILD_DIR, exist_ok=True)
    exe = os.path.join(BUILD_DIR, "aaa_native" + (".exe" if sys.platform == "win32" else ""))
    cmd = ([cxx] + BASE_FLAGS + [opt,
                                 "-I", os.path.join(NATIVE, "include"),
                                 "-I", os.path.join(NATIVE, "src"),
                                 os.path.join(NATIVE, "tests", "native_main.cpp")]
           + sources() + ["-o", exe, "-static", "-static-libgcc", "-static-libstdc++"])
    return run(cmd, dry), exe


def run(cmd, dry=False):
    printable = " ".join(cmd)
    if dry:
        print(printable)
        return 0
    print(f"  $ {printable}")
    return subprocess.call(cmd)


def do_check():
    """只校验「已存在的库能被加载且 ABI 版本对得上」。

    CI 里这一步是**硬 gate**：如果编译悄悄失败、而测试又是条件注册的，
    整套测试会静默变少却仍然全绿 —— 这是最难发现的一种假绿。
    """
    sys.path.insert(0, os.path.join(ROOT, "src"))
    try:
        from aaa_isp_lab.native import loader
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] 无法导入 loader：{type(exc).__name__}: {exc}")
        return EXIT_BAD_LIB
    st = loader.status()
    if not st.get("available"):
        print(f"[FAIL] 共享库不可用：{st.get('reason')}")
        print("       先跑：python tools/build_native.py")
        return EXIT_BAD_LIB
    print(f"[ok] 库路径   {st['lib_path']}")
    print(f"[ok] ABI 版本 {st['abi_version']}（期望 {loader.ABI_VERSION}）")
    print(f"[ok] 构建信息 {st.get('build_info')}")
    if st["abi_version"] != loader.ABI_VERSION:
        print("[FAIL] ABI 版本不匹配 —— 库是旧版，请重新编译")
        return EXIT_BAD_LIB
    return EXIT_OK


def main():
    ap = argparse.ArgumentParser(description="编译 aaa_stats 共享库（可选组件）")
    ap.add_argument("--check", action="store_true", help="只校验库可加载且 ABI 版本一致")
    ap.add_argument("--print-cmd", action="store_true", help="只打印命令，不执行")
    ap.add_argument("--with-bench", action="store_true", help="同时编独立可执行（自检 + 基准）")
    ap.add_argument("--clean", action="store_true", help="删除编译产物")
    ap.add_argument("--cxx", default=None, help="指定编译器")
    ap.add_argument("--opt", default="-O3", help="优化档（默认 -O3）")
    args = ap.parse_args()

    if args.check:
        return do_check()

    if args.clean:
        for p in (lib_path(), BUILD_DIR):
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            elif os.path.isfile(p):
                os.remove(p)
        print("[ok] 已清理产物")
        return EXIT_OK

    cxx = find_compiler(args.cxx)
    if not cxx:
        print("[FAIL] 找不到 C++ 编译器。装 g++ / clang++，或用 --cxx 指定路径。",
              file=sys.stderr)
        print("       注意：不编也能跑纯 Python 路径，这个组件是可选的。", file=sys.stderr)
        return EXIT_NO_COMPILER

    os.makedirs(PKG_DIR, exist_ok=True)
    print(f"编译器：{cxx}")
    rc = build_shared(cxx, args.opt, dry=args.print_cmd)
    if rc != 0:
        print(f"[FAIL] 编译共享库失败（退出码 {rc}）", file=sys.stderr)
        return rc
    if not args.print_cmd:
        print(f"[ok] {lib_path()}  ({os.path.getsize(lib_path())} 字节)")

    if args.with_bench:
        rc2, exe = build_standalone(cxx, args.opt, dry=args.print_cmd)
        if rc2 != 0:
            print(f"[FAIL] 编译独立可执行失败（退出码 {rc2}）", file=sys.stderr)
            return rc2
        if not args.print_cmd:
            print(f"[ok] {exe}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
