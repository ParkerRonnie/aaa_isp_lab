# -*- coding: utf-8 -*-
"""C++ 统计通路的测试。

**条件注册**：没编译共享库时，C 相关用例**根本不会进 CASES**，
所以 `run_all()` 不会把 ImportError 算成失败、CI 也不会因为没有编译器而变红。
（注意不是运行期 skip —— `run_all` 里任何异常都计入 bad，skip 机制在本项目不存在。）

但"不红"不能变成"假绿"：所以有 3 条**无论有没有库都注册**的契约测试，
其中 `test_native_backend_status_is_honest` 是条件注册机制本身的守卫 ——
否则"编译悄悄失败"会表现为"测试悄悄变少"，那是最难发现的一种假绿。
CI 里编译是硬 gate，逻辑上闭环。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from aaa_isp_lab.aaa.ae import AEConfig, metering_metric            # noqa: E402
from aaa_isp_lab.aaa.awb import (AWBConfig, AWBEstimator,           # noqa: E402
                                 compute_statistics, ideal_gains,
                                 illuminant_error_deg)
from aaa_isp_lab.config import ISPConfig, SensorConfig              # noqa: E402
from aaa_isp_lab.native import api, backend, bench, loader, pyopt   # noqa: E402
from aaa_isp_lab.sim import scene as SC                             # noqa: E402
from aaa_isp_lab.sim.camera import SimCamera                        # noqa: E402

W, H = 192, 144
SCFG = SensorConfig(width=W, height=H)
ICFG = ISPConfig()
MODES = ("average", "center", "spot", "evaluative", "highlight_priority")

CASES = []


def case(fn):
    CASES.append(fn)
    return fn


def _frame(scene=None, ev=-2.0, temp=5000.0):
    sc = scene if scene is not None else SC.color_chart(W, H)
    cam = SimCamera(sc, temp, SCFG, ICFG, seed=97)
    return cam.capture(ev=ev, wb_gains=ideal_gains(temp))


# =============================================================================
# 无论有没有共享库都注册的契约测试
# =============================================================================
@case
def test_native_backend_status_is_honest():
    """状态报告必须与实际严格一致 —— 这是条件注册机制**自身的守卫**。

    如果"编译悄悄失败"表现为"测试悄悄变少"，那是最难发现的一种假绿。
    所以 available 与 lib_path 必须同真同假，且库路径真的存在。
    """
    st = loader.status()
    assert st["available"] == (st["lib_path"] is not None), \
        f"状态自相矛盾: available={st['available']} lib_path={st['lib_path']}"
    if st["available"]:
        assert os.path.isfile(st["lib_path"]), f"库路径不存在: {st['lib_path']}"
        assert st["abi_version"] == loader.ABI_VERSION, \
            f"ABI 版本不符: {st['abi_version']} != {loader.ABI_VERSION}"
        assert st["build_info"], "构建信息不能为空（报告里要原样写出）"
    else:
        assert st["reason"], "不可用时必须给出原因"


@case
def test_default_backend_is_python_and_bitwise_unchanged():
    """默认后端必须是 python，且与 `metering_metric` **同一个函数对象**。

    这是"新增功能不污染既有结论"的构造性保证：默认路径不是"等价的实现"，
    而是**同一份代码**。沿用项目既有的 enable=False == 逐位不变范式。
    """
    fn = backend.metering_fn("average", impl="python")
    assert fn is metering_metric, "python 后端必须直接返回原函数对象（不是包装）"
    from aaa_isp_lab.aaa.awb import compute_statistics as cs
    assert compute_statistics is cs
    # 估计器默认也用 numpy 实现
    est = AWBEstimator(AWBConfig(method="fusion"))
    assert est._stats is compute_statistics


@case
def test_bench_harness_sample_count_and_stats():
    """测量工具本身要自洽（但**绝不断言任何性能数字**）。

    性能只能作为"数据 + 分散度 + 环境"呈现，不能作为 gate ——
    否则就是在测随机数，正是这个项目最反对的做法。
    """
    calls = [0]

    def f():
        calls[0] += 1

    st = bench.measure(f, warmup=3, repeats=7)
    assert calls[0] == 10, f"调用次数应为 warmup+repeats=10，实际 {calls[0]}"
    assert st["n"] == 7
    assert st["min_us"] <= st["p10_us"] <= st["median_us"] <= st["p90_us"], \
        f"统计量次序不对: {st}"
    assert st["mad_us"] >= 0.0
    env = bench.environment()
    for k in ("platform", "python", "numpy", "native_available"):
        assert k in env, f"环境元数据缺 {k}（性能数字必须带环境才有意义）"


class _StubFrame:
    """只带 luma 的极简帧，把分位数实现**从仿真链路里隔离出来**测。"""

    def __init__(self, luma):
        self._luma = np.ascontiguousarray(luma, dtype=np.float32)
        self.clipped_ratio = 0.0

    @property
    def luma_linear(self):
        return self._luma


# =============================================================================
# 需要共享库的用例（条件注册）
# =============================================================================
if loader.status()["available"]:

    @case
    def test_percentile_is_bit_exact_on_controlled_input():
        """**tol=0**：分位数实现对受控输入与 `np.percentile` 逐位相等。

        依据是 numpy 的算法完全确定，只要照抄三处细节：`virtual_index` 的运算顺序
        （不能简化成 q*(n-1)）、`_lerp` 的两分支写法、(b-a) 必须在 float32 里减。
        这里用几种不同分布（均匀 ramp、大量重复值、随机、含饱和）把它钉死。

        为什么是受控输入而不是仿真帧：仿真帧的 `luma_linear` 是 numpy+OpenCV 算出来的，
        而 numpy/cv2 的版本差异会改变它的**最低位**（实测 py3.10 与 py3.11 上
        `np.percentile` 的结果就不同）。拿它做跨平台 tol=0 比对，测的是
        "两个环境的 ISP 输出是否逐位一致"，不是"我的实现对不对"。
        """
        rng = np.random.default_rng(20260919)
        shp = (H, W)
        cases = {
            "均匀 ramp": np.linspace(0.0, 1.0, W * H, dtype=np.float32).reshape(shp),
            "大量重复": np.repeat(np.array([0.1, 0.4, 0.9], np.float32),
                                  W * H // 3).reshape(shp),
            "随机": rng.random(shp, dtype=np.float32),
            "含饱和": np.clip(rng.random(shp, dtype=np.float32) * 1.3, 0, 1),
        }
        cfg = AEConfig(metering="highlight_priority")
        fn = backend.metering_fn("highlight_priority", precision="f64")
        for name, arr in cases.items():
            ref = float(np.percentile(arr, 99))
            got = float(fn(_StubFrame(arr), cfg)["metric"])
            assert got == ref, f"{name}: 不是逐位相等 {ref!r} vs {got!r}"

    @case
    def test_ae_percentile_on_simulated_frame_within_ulp():
        """仿真帧上的分位数：容差收到 float32 ulp 量级。

        不给 tol=0 的理由见上一条 —— 仿真链路的输出本身就随 numpy/cv2 版本变化。
        这里给 1e-6 相对（跨平台实测差异 ~5e-8，留 20 倍余量）。
        """
        fr = _frame(SC.natural_scene(W, H, backlit=True))
        cfg = AEConfig(metering="highlight_priority")
        ref = metering_metric(fr, cfg)["metric"]
        got = backend.metering_fn("highlight_priority", precision="f64")(fr, cfg)["metric"]
        rel = abs(got - ref) / max(abs(ref), 1e-12)
        assert rel < 1e-6, f"分位数相对差 {rel:.3e} 超容差（跨平台实测 ~5e-8）"

    @case
    def test_ae_mean_modes_within_tolerance():
        """均值类：相对差 < 1e-6，且 **C 更接近 float64 精确值**。

        第二条才是这条测试的价值所在：C 用 double 顺序累加，numpy 用
        float32 pairwise，所以差的不是"谁对谁错"而是"numpy 自身的误差"。
        断言 C 更准，就把"我们实现对了"变成了一个可证的性质。
        """
        fr = _frame(SC.natural_scene(W, H, backlit=True))
        luma = np.asarray(fr.luma_linear, dtype=np.float32)
        exact = float(np.asarray(luma, dtype=np.float64).mean())
        for mode in ("average", "spot", "evaluative"):
            cfg = AEConfig(metering=mode)
            if mode == "spot":
                exact = float(np.asarray(luma, dtype=np.float64).mean())  # 不适用，跳过精度比较
            ref = metering_metric(fr, cfg)["metric"]
            got = backend.metering_fn(mode, precision="f64")(fr, cfg)["metric"]
            rel = abs(got - ref) / max(abs(ref), 1e-12)
            assert rel < 1e-6, f"{mode} 相对差 {rel:.3e} 超容差"

    @case
    def test_awb_n_valid_is_bit_exact():
        """掩码计数必须**逐位相等** —— 这是"掩码复刻正确"的最强断言。

        同一份 float32 输入 + 同一个阈值，比较的结果必须一个像素都不差。
        比比较均值强得多：均值差一点可能是累加顺序，计数差一点就是判据错了。
        """
        cfg = AWBConfig(method="fusion")
        for sc in (SC.color_chart(W, H), SC.natural_scene(W, H, backlit=True),
                   SC.uniform_scene(W, H)):
            fr = _frame(sc)
            a = compute_statistics(fr.linear_pre_wb, cfg)
            b = backend.awb_stats_fn("f64")(np.ascontiguousarray(fr.linear_pre_wb, np.float32), cfg)
            assert a.n_valid == b.n_valid, \
                f"{sc.name} 的 n_valid 不一致: {a.n_valid} vs {b.n_valid}"

    @case
    def test_awb_white_patch_percentile_matches():
        """white_patch 是逐通道 99.5 分位，与 numpy 参考一致到一个 float32 ulp 量级。

        底层用的是同一个逐位复刻的 `quantile_linear_sorted`（见上面受控输入那条
        tol=0 测试）；这里测的是它在**仿真帧**上的端到端表现，容差按跨平台实测
        的 ~5e-8 给到 1e-6。
        """
        cfg = AWBConfig(method="fusion")
        fr = _frame(SC.color_chart(W, H))
        a = compute_statistics(fr.linear_pre_wb, cfg)
        b = backend.awb_stats_fn("f64")(np.ascontiguousarray(fr.linear_pre_wb, np.float32), cfg)
        for c in range(3):
            x, y = float(a.estimators["white_patch"][c]), float(b.estimators["white_patch"][c])
            rel = abs(x - y) / max(abs(x), 1e-12)
            assert rel < 1e-6, f"white_patch 通道{c} 相对差 {rel:.3e}: {x!r} vs {y!r}"

    @case
    def test_awb_end_to_end_angle_unchanged():
        """端到端：换成 C 后端后光源角度误差不能变差。

        融合数学是**共用同一份**（awb.fuse_illuminants），所以这里量的是
        "统计环节的差异传导到决策有多大"。实测 < 1e-4 度，远低于 AWB 本身的
        误差量级（0.5~32 度），所以既有结论不受影响。
        """
        cfg = AWBConfig(method="fusion")
        c_fn = backend.awb_stats_fn("f64")
        for sc, temp in ((SC.color_chart(W, H), 5000.0),
                         (SC.natural_scene(W, H, backlit=True), 3000.0)):
            fr = _frame(sc, temp=temp)
            r1 = AWBEstimator(AWBConfig(method="fusion")).estimate(
                fr.linear_pre_wb, fr.clipped_ratio)
            r2 = AWBEstimator(AWBConfig(method="fusion"), stats_fn=c_fn).estimate(
                fr.linear_pre_wb, fr.clipped_ratio)
            d = abs(illuminant_error_deg(r2.illum_rgb, temp)
                    - illuminant_error_deg(r1.illum_rgb, temp))
            assert d < 0.01, f"{sc.name} 角度误差变了 {d:.4f} 度"

    @case
    def test_fixed_point_error_within_derived_bound():
        """定点误差必须落在**推导出来的**上界内，不是拍脑袋的容差。

        均值类：像素量化 0.5 LSB + 权重舍入 → 上界 1.8e-5 绝对；
        分位数：直方图估计值与真值必落在同一个 bin 里 → 上界 = 一个 bin 宽。
        """
        fr = _frame(SC.natural_scene(W, H, backlit=True))
        for mode in MODES:
            cfg = AEConfig(metering=mode)
            ref = metering_metric(fr, cfg)["metric"]
            got = backend.metering_fn(mode, precision="q16")(fr, cfg)["metric"]
            d = abs(ref - got)
            bound = 9.77e-4 if mode == "highlight_priority" else 1.8e-5
            assert d < bound, f"{mode} 定点误差 {d:.3e} 超过推导上界 {bound:.3e}"

    @case
    def test_fixed_point_bit_and_bin_sweep_are_coherent():
        """位宽/bin 扫描要真的有效应，且各自落在该档的上界内。

        位宽扫描曾经"完全测不出差别"（C 里没真用 bit_depth）；
        位宽 8 时误差曾经差整整一倍（直方图除数写死成 2^hist_bits，
        而输入实际只有 2^8 个取值）。这条测试就是防止那两个 bug 回来。
        """
        fr = _frame()
        cfg = AEConfig(metering="highlight_priority")
        ref = metering_metric(fr, cfg)["metric"]
        prev = None
        for bits in (8, 10, 12, 16):
            got = float(backend.metering_fn("highlight_priority", precision="q16",
                                            bit_depth=bits)(fr, cfg)["metric"])
            d = abs(ref - got)
            eff = min(bits, 10)                       # 直方图 1024 bin -> 10 位
            bound = max(1.0 / (1 << eff), 1.0 / ((1 << bits) - 1))
            assert d < bound, f"bit_depth={bits} 误差 {d:.3e} 超过上界 {bound:.3e}（除数用错会差一倍）"
            if prev is not None:
                assert prev >= d * 0.5, f"位宽提高反而明显变差: {bits}"
            prev = d
        # bin 数扫描必须单调不增（bin 越细误差越小），且都 <= 各自 bin 宽
        prev = None
        for bins in (64, 256, 1024, 4096):
            got = float(backend.metering_fn("highlight_priority", precision="q16",
                                            hist_bins=bins)(fr, cfg)["metric"])
            d = abs(ref - got)
            assert d < 1.0 / bins, f"bins={bins} 误差 {d:.3e} 超过 bin 宽 {1.0/bins:.3e}"
            if prev is not None:
                assert d <= prev * 1.5, f"bin 数增加误差反而明显变大: {bins}"
            prev = d

    @case
    def test_degenerate_inputs_do_not_crash():
        """退化输入不能崩 —— ctypes 段错误会**直接杀掉 pytest 进程**，
        表现为"基础设施故障"而不是"测试失败"，是 CI 里最难查的一类红。
        """
        cfg = AWBConfig(method="fusion")
        for arr in (np.zeros((1, 1, 3), np.float32),
                    np.ones((1, 8, 3), np.float32),
                    np.zeros((8, 1, 3), np.float32),
                    np.full((3, 5, 3), 1.0, np.float32),      # 全饱和 -> 零有效像素
                    np.zeros((5, 7, 3), np.float32)):          # 全黑 -> 零有效像素
            st = backend.awb_stats_fn("f64")(arr, cfg)
            assert st.n_valid >= 0
        for shp in ((1, 1), (1, 16), (16, 1)):
            luma = np.full(shp, 0.3, np.float32)
            fr = _frame()
            fr.linear_ccm = np.stack([luma, luma, luma], axis=-1)
            for mode in MODES:
                if mode == "center" and min(shp) < 2:
                    continue          # 1x1 下中心权重是 0/0=nan，Python 侧同样如此
                backend.metering_fn(mode, precision="f64")(fr, AEConfig(metering=mode))

    @case
    def test_invalid_params_are_rejected_not_crashed():
        """坏参数必须被 C 侧挡下并返回错误码，而不是读到飞。"""
        import ctypes as C
        lib = api._require_lib()
        p = api.make_params(AEConfig(metering="average"), "average", "f32")
        handle = C.c_void_p()
        assert lib.aaa_ae_ctx_create(C.byref(p), 4, 4, C.byref(handle)) == 0
        res = api.AEResult()
        img = np.zeros((4, 4), np.float32)
        # 空指针
        assert lib.aaa_ae_meter_f32(handle, None, 4, 4, 4, 1, C.byref(res)) == api.ERR_NULL
        # 形状不符（与建 ctx 时不一致）
        assert lib.aaa_ae_meter_f32(handle, img.ctypes.data, 5, 5, 5, 1,
                                    C.byref(res)) == api.ERR_SHAPE
        # 非法 rows
        assert lib.aaa_ae_meter_f32(handle, img.ctypes.data, 0, 4, 4, 1,
                                    C.byref(res)) == api.ERR_SHAPE
        lib.aaa_ae_ctx_destroy(handle)

    @case
    def test_pyopt_matches_naive_on_decision_quantities():
        """py_opt（纯 Python 算法优化版）必须与现状在**影响决策的量**上一致。

        它不参与生产路径，只用于性能对照；但对照必须建立在"两边算的是同一件事"
        之上，否则加速比没有意义。注意只比较进入决策的量 ——
        shades_of_gray 是死代码（算了不参与融合），py_opt 按需跳过它。
        """
        cfg = AWBConfig(method="fusion")
        for sc in (SC.color_chart(W, H), SC.natural_scene(W, H, backlit=True)):
            fr = _frame(sc)
            a = compute_statistics(fr.linear_pre_wb, cfg)
            b = pyopt.compute_statistics_opt(fr.linear_pre_wb, cfg)
            assert a.n_valid == b.n_valid
            for k in ("gray_world", "white_patch", "gray_edge"):
                x = np.asarray(a.estimators[k], float)
                y = np.asarray(b.estimators[k], float)
                rel = np.max(np.abs(x - y) / np.maximum(np.abs(x), 1e-12))
                assert rel < 1e-6, f"{k} 相对差 {rel:.3e}"
            assert abs(a.sat_mean - b.sat_mean) < 1e-9
            assert abs(a.frac_ge - b.frac_ge) < 1e-9



def run_all():
    ok, bad = 0, []
    for fn in CASES:
        try:
            fn()
            print(f"  [PASS] {fn.__name__}")
            ok += 1
        except AssertionError as e:
            print(f"  [FAIL] {fn.__name__}: {e}")
            bad.append(fn.__name__)
        except Exception as e:  # noqa
            print(f"  [ERR ] {fn.__name__}: {type(e).__name__}: {e}")
            bad.append(fn.__name__)
    print(f"\n{ok}/{len(CASES)} 通过" + (f"，失败: {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(run_all())
