# aaa_stats —— 3A 统计通路的 C++ 实现

把 AE 测光与 AWB 统计的逐像素通路用 C++ 重写了一份，编成 `extern "C"` 共享库，
Python 侧用 ctypes 加载。**这是可选组件**：不编译时项目跑纯 Python 路径，
结论完全不变（默认后端是纯 Python，且与 `ae.metering_metric` 是**同一个函数对象**）。

---

## 为什么是共享库 + ctypes，而不是 setuptools Extension / pybind11

| 方案 | 本地（Win10 + 只有 MinGW g++） | CI（ubuntu + gcc） | 新增 pip 依赖 |
|---|---|---|---|
| setuptools Extension | ✗ Windows 的 CPython 是 MSVC 构建，MinGW 扩展链不上 | ✓ | 需 pybind11 |
| **`extern "C"` + ctypes** | **✓** | **✓** | **0** |

附带三个好处：不引入任何 pip 依赖（保住「依赖以 pyproject.toml 为唯一来源」这条原则）、
不必改 `[build-system]`、而且**这个库脱离 Python 也能编能跑**——见
`native/tests/native_main.cpp`，它有 23 项不依赖 numpy 的解析自检。
这一点也是"我真会写 C++"而不是"我写了个 Python 插件"的证据。

---

## 构建与验证

```bash
python tools/build_native.py --with-bench   # 编共享库 + 独立可执行
python tools/build_native.py --check        # 校验可加载 + ABI 版本（CI 硬 gate）
./native/build/aaa_native --self-test       # 23 项解析自检，不经过 Python
./native/build/aaa_native --bench --impl q16 --repeats 100
```

`build_native.py` 的开关：`--print-cmd`（只打印命令，CI 日志留痕）、`--clean`、
`--cxx clang++`、`--opt -O2`。找不到编译器时退出码 **2**；库不可加载时 **3**。

### 编译选项的四条禁令

- **不许 `-ffast-math`**：会破坏 NaN 语义与浮点等价性断言。
  本项目的等价性测试依赖 IEEE 语义（尤其 `_center_weight` 在 `h==1` 时是 `0/0=nan`，
  这是**保持与 Python 一致**的行为，不是 bug）。
- **必须 `-ffp-contract=off`**。GCC 对 C++ 默认 `-ffp-contract=fast`，会把
  `a + b*c` 融合成一条 FMA —— 精度更高但**结果不同**。`percentile` 的逐位复刻
  依赖严格的 IEEE 语义，而不同发行版的 GCC 默认目标不同（有的基线已含 FMA）。
  实测：**不加这一条，同一份代码在 MinGW 8.1 上与 numpy 逐位相等，
  在 ubuntu 的 gcc 13 上就差 4e-8** —— CI 直接红，而本地全绿。
- **不许 `-march=native`**：本地与 CI 的指令集不同，跨机性能数字就不可比了。
- **不许 OpenMP / 多线程**：与 numpy 的线程策略不可比，且引入不可复现的方差。

另外不用 `std::filesystem`（g++ 8 需要额外 `-lstdc++fs`）。`-std=c++17` 在 g++ 8.1 上够用。

### Windows/MinGW 特有的坑（都踩过）

1. **必须 `-static`，不是只要 `-static-libgcc -static-libstdc++`。**
   这个 MinGW 构建下 `std::string` / `std::vector` / 异常处理会拉进
   `libwinpthread-1.dll`，而 CPython 进程的 PATH 上没有它。ctypes 报的是
   `Could not find module ... (or one of its dependencies)` —— **看起来像文件不存在，
   实际是依赖缺失**，很有迷惑性。
   核验方法：`objdump -p aaa_stats.dll | grep "DLL Name"`，应当只剩
   `KERNEL32.dll` 与 `msvcrt.dll`。
2. **必须传 `-DNDEBUG`。** 否则 `aaa_build_info()` 会把 `-O3` 的构建谎报成 `debug`，
   而这个字符串要原样写进报告的元数据里 —— 报错了等于整节性能数字的前提是假的。

---

## ABI 契约

对外只有 `include/aaa_stats.h` 一个头文件。**三条纪律**，C 侧用 `static_assert`
固化尺寸与偏移，Python 侧（`src/aaa_isp_lab/native/api.py`）在导入时再校验一遍：

1. **结构体只含定宽整型 / `double` / 指针**，且显式留 padding。
   跨 MSVC 与 MinGW 的布局按此保证一致，不靠"通常一样"。
2. **绝不跨边界传所有权。** 句柄是不透明指针，析构在库内做（RAII 留在库里）。
3. **绝不抛异常出边界。** 每个导出函数都在 `try/catch` 里，错误一律用返回码表示。

还有一条不是纪律但同样重要：**所有参数在 C 里校验**。ctypes 传错参数导致的段错误
会**直接杀掉宿主进程**（pytest 直接挂），表现为"基础设施故障"而不是"测试失败" ——
是 CI 里最难定位的一类红。所有导出函数在入口做完整校验（空指针、行列数、
步长、模式、精度、`hist_bins` 是否为 2 的幂、`bit_depth` 范围）。

### 输入约定

**输入永远是「指针 + 行/列/通道步长」，不要求连续** —— Python 侧因此永不拷贝
（真实 ISP 的统计块本来就带 stride）。注意 `(H,W,3)` 的连续数组里通道是**交织**的，
`stride_col` 是 3 而不是 1（这里踩过一次：传成 1 会让 C 读到错位的通道数据，
表现为 `n_valid` 只剩 1 个像素）。

---

## 定点格式

| 环节 | 格式 | 理由 |
|---|---|---|
| 输入 luma / RGB | **uint16，Q0.16（`bit_depth` 可变）** | `linear_pre_wb` 与 `linear_ccm` 都被 clip 在 [0,1]，满量程无浪费；等价于"统计块吃 16bit 整数 luma"，是真实 ISP 的形态 |
| 累加器 | **int64** | 4K 全图 Σ ≈ 5.4e11，evaluative 加权和 ≈ 1.9e16，都远小于 2^63 —— **不需要分段移位**（那是 32 位 MCU 的妥协，代价是每行 0.5 LSB 的系统性偏差） |
| 权重 | **Q0.15** | 相对分辨率 3e-5，比像素量化细一档，不成为误差主项 |
| 分位数 | **2^k bin 直方图**，bin 索引是**纯移位** | 无除法、无浮点比较器、单遍顺序访存 |
| SoG 的 `x^6` | **`__int128`** | Q16 下 `v^6` 最大 7.9e28，远超 int64。这是 GCC/Clang 扩展 —— 本项目的构建前提就是「不用 MSVC」，所以这个依赖成立 |
| 最终标量 | **double** | 每帧一个数，控制律在 double 里跑。「整数数据通路 + 浮点边界标量」是个清晰可辩护的边界 |

### 误差上界（**推导出来的**，不是拟合的容差）

- **均值类**：像素量化 0.5 LSB + 权重 Q15 舍入 → 绝对上界 **1.8e-5**
  （折算 EV ≈ 1.4e-4，比收敛阈值 0.02 EV 低约 140 倍；实测还要好，约 6.8e-6）
- **分位数**：直方图估计值与真值必然落在**同一个 bin** 内 → 绝对上界 = **一个 bin 宽**
  （1024 bin 时 9.77e-4，折算 EV ≈ 1.6e-3）

### 两个位宽/bin 的陷阱（都踩过）

1. **直方图的除数必须是 `2^min(输入位宽, bin 位数)`**，不能写死 `2^hist_bits`。
   输入 8 位配 1024 bin 时真正起作用的是 8 位，除数用错**结果会整整差一倍**
   （实测 8 位下误差 0.507）。
2. **量化尺度必须跟着 `bit_depth` 走**（`q_scale = 2^bit_depth - 1`）。
   一开始写死 65535，导致位宽扫描**完全测不出差别** —— 看起来一切正常，
   实际上那个扫描是假的。

### 定点化的边界（明写）

只定点化**逐像素统计通路**。融合权重、对数域几何平均、CCT 约束、AE 控制律
仍在 double —— 它们每帧只作用在 3~5 个数上，没有吞吐论证，
且含 `exp/log/pow`，定点化是另一场独立的误差分析。
**整条 ISP 的定点化（去马赛克、CCM 仍是浮点）明确不在范围内。**

---

## 逐位复刻 numpy 的 `percentile`

`highlight_priority` 的 99 分位与 `white_patch` 的 99.5 分位都可以与
`np.percentile(..., method='linear')` **逐位相等**（实测偏差 0.00e+00）。
依据是 numpy 的算法完全确定，只要照抄三处细节：

1. `virtual_index = n*q + (1 + q*(1-1-1)) - 1` —— **运算顺序不能简化成 `q*(n-1)`**，
   否则末几位会不一致。
2. `_lerp(a, b, t) = t < 0.5 ? a + (b-a)*t : b - (b-a)*(1-t)` —— 两个分支数值等价，
   但浮点上不同，必须照抄分支。
3. 输入保持 **float32**，`(b-a)` 必须在 float32 里减，只有乘 `t` 时才提升到 double。

现成的证据：`np.linspace(0,1,172800)` 的 p99 实测是 `0.9900000077486039`，
与解析值 0.99 差 7.7e-9 —— 正是上面这些浮点细节造成的。

⚠️ **一个不许写进报告的结论**：不要说「C 用直方图把分位数从 O(n log n) 变成 O(n)」。
`np.percentile` 走的是 `arr.partition()`（introselect），**本来就是 O(n)**。
C 在分位数上的真实优势只有三条：不拷贝整帧、顺序访存（partition 是数据相关的随机交换）、
定点下不需要浮点比较器。这三条都要靠实测支撑。

---

## cv2.Sobel 的等效核（**实测反解**，不是猜的）

`awb.gray_edge` 与融合权重里的 `frac_ge` 都用到 `cv2.Sobel(..., ksize=3)`。
等效核用脉冲图反解、再用线性 ramp 验证一致性，结果是**相关**（correlation）而非卷积：

```
dx: [-1  0  1]      dy: [-1 -2 -1]
    [-2  0  2]          [ 0  0  0]
    [-1  0  1]          [ 1  2  1]

out[y,x] = Σ K[a,b] · img[y+a-1, x+b-1]
```

边界是 **`BORDER_REFLECT_101`**（OpenCV 的 `BORDER_DEFAULT`），
即 `fedcba|abcdefgh|hgfedcb`，**不重复**边界像素（区别于 `BORDER_REFLECT`）。
核验方法：线性 ramp 的 dx 内部恒为 ±8，而默认边界的边界值是 0.0，
`BORDER_REPLICATE` / `BORDER_REFLECT` 都给 4.0 —— 三者可区分。

### `mag > t` 不需要开方

Python 现状是 `mag = sqrt(gx²+gy²)`，但 `mag` **只用于 `mag > thresh` 这个比较**，
数值本身从不用到。所以 `gx² + gy² > t²` 完全等价，**一次开方都不需要**。
定点下更彻底：`gx²+gy²` 在 int64 里是**精确无舍入**的，连浮点比较器都不用。
这是只有真的动手做定点化才会发现的优化。

---

## 性能：数字怎么读

测量方法学在 `src/aaa_isp_lab/native/bench.py`：
warmup + 重复 + **中位数** + p10/p90/MAD，并实测 **ctypes 调用地板**
（`aaa_null_call`，本机约 0.2 µs，相对每个核都 < 5%）。
环境元数据（平台 / Python / numpy / OpenCV / **编译器与 -O 档**）随数字一起报。

**最重要的一条结论：加速来自算法，不是语言。**

- 算法改进（Python→Python）：1.45×
- 实现改进（算法不变，换 C++）：11.1×
- 总加速：16.1×
- **纯语言差异**（全画面平均，两边都没有算法可改）：**0.89×** —— C 反而略慢

最后一格是关键：numpy 的 `mean()` 是 SIMD 归约，标量 double 循环赢不了。
那 16 倍来自重写时顺手消除了结构性的浪费（5 次全帧 gather、4 次 Sobel 其中 2 次重复、
4 次全帧开方、无条件算 4 个估计器）。**「C++ 比 numpy 快 16 倍」是错的说法。**

⚠️ **性能数字不得作为测试断言。** 只能作为"数据 + 分散度 + 环境"呈现。
可复现的是等价性与误差，不是耗时。

---

## 已知疑点（只记录，未改语义）

- **`shades_of_gray` 算了但没进融合。** `AWBConfig.fusion_weights` 的注释写
  `(gray_world, gray_edge, sog)`，而 `awb.py` 实际按 `(gray_world, gray_edge,
  white_patch)` 取。SoG 被计算（含逐元素 6 次幂）、被写进 `detail['estimators']`，
  但不参与决策。改它会推翻现有全部 AWB 结论（报告、README、讲稿里都在引用那些
  角度误差数字），所以本次只记录、不动。
- **`AWBEstimator.estimate` 无条件计算全部 4 个估计器**，哪怕 `method='gray_world'`
  只用 1 个。C 侧有 `need_*` 开关按需计算。
