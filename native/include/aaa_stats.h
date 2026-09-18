/* =============================================================================
 * aaa_stats.h —— 3A 统计通路的 C ABI（唯一对外接口）
 *
 * 设计目标：让 Python（ctypes）与 C/C++ 双方都能安全地调用这一份统计内核，
 * 而**不把所有权、异常、C++ 类型跨过边界**。
 *
 * 三条 ABI 纪律（在 aaa_stats.cpp 里用 static_assert 固化）：
 *   1) 结构体只含定宽整型 / double / 指针，且显式留 padding。
 *      跨 MSVC 与 MinGW 的结构体布局按此保证一致，不靠"通常一样"。
 *   2) **绝不跨边界传所有权**：句柄是不透明指针，析构在库内做（RAII 留在库内）。
 *   3) **绝不抛异常出边界**：每个导出函数都在 try/catch 里，错误一律用返回码表示。
 *
 * 还有一条不是纪律但同样重要：**所有参数在 C 里校验**。
 * 因为 ctypes 传错参数导致的段错误会直接杀掉宿主进程（pytest 直接挂），
 * 表现为"基础设施故障"而不是"测试失败"——是 CI 里最难定位的一类红。
 *
 * 为什么是共享库 + ctypes 而不是 setuptools Extension：
 *   Windows 上的 CPython 是 MSVC 构建的，而本机只有 MinGW g++，
 *   MinGW 编译的扩展模块无法可靠链接 MSVC 的 Python。
 *   共享库 + extern "C" 只依赖纯 C ABI，两个工具链都能编，
 *   而且这个库**脱离 Python 也能编能跑**（见 native/tests/native_main.cpp）。
 * ========================================================================== */
#ifndef AAA_STATS_H
#define AAA_STATS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AAA_STATS_ABI_VERSION 1

#if defined(_WIN32)
#  define AAA_API __declspec(dllexport)
#else
#  define AAA_API __attribute__((visibility("default")))
#endif

/* ---- 状态码：所有导出函数返回它。0 = 成功，负数 = 出错。绝不抛异常。 ---- */
enum {
    AAA_OK           =  0,
    AAA_ERR_NULL     = -1,   /* 空指针 */
    AAA_ERR_SHAPE    = -2,   /* rows/cols/stride 非法 */
    AAA_ERR_MODE     = -3,   /* 模式或精度档位越界 */
    AAA_ERR_RANGE    = -4,   /* 参数取值越界（如 hist_bins 非 2 的幂） */
    AAA_ERR_INTERNAL = -5    /* 内部错误（含 bad_alloc） */
};

/* ---- AE 测光模式（与 ae.py 的 cfg.metering 一一对应）---- */
enum {
    AAA_METER_AVERAGE   = 0,   /* 全画面平均 */
    AAA_METER_CENTER    = 1,   /* 中心高斯加权 */
    AAA_METER_SPOT      = 2,   /* 中央窗口平均 */
    AAA_METER_EVALUATIVE = 3,  /* 分区评价测光 + 高光保护（Python 默认） */
    AAA_METER_HIGHLIGHT = 4    /* 高光优先（99 分位） */
};

/* ---- 算术精度档位 ----
 * F64：参考实现，double 顺序累加，用于证明实现正确；
 * F32：生产通路，float32 输入 + **double 累加**（比 numpy 的 float32 pairwise 更准）；
 * Q16：定点通路，uint16 Q0.16 输入 + int64 累加。 */
enum { AAA_PREC_F64 = 0, AAA_PREC_F32 = 1, AAA_PREC_Q16 = 2 };

/* =============================================================================
 * AE 测光
 * ========================================================================== */

typedef struct {
    int32_t mode;             /* AAA_METER_* */
    int32_t precision;        /* AAA_PREC_* */
    int32_t zones_y;          /* evaluative 分区数 */
    int32_t zones_x;
    int32_t hist_bins;        /* 定点分位数直方图 bin 数；0 -> 1024；须为 2 的幂 */
    int32_t bit_depth;        /* 定点输入位宽；0 -> 16 */
    int32_t flags;            /* 位 0：使用外部 zone_w 表 */
    int32_t reserved0;
    double  center_ratio;     /* 中心权重集中度（Python 默认 2.0）*/
    double  spot_ratio;       /* 点测光窗口占比（0.15）*/
    double  zone_sigma;       /* 分区权重高斯半径（0.45）*/
    double  highlight_weight; /* 分区测光里的高光保护权重（0.60）*/
    double  clip_level;       /* 判"顶到饱和"的阈值（Python 里硬编码 0.95）*/
    double  reserved1;
    const double* zone_w;     /* zones_y*zones_x 行优先高斯权重；可空 = 用内置表 */
    int32_t zone_w_len;
    int32_t reserved2;
} aaa_ae_params;

typedef struct {
    double  metric;           /* 测光量（线性域）*/
    double  aux;              /* evaluative: clip_zone；其余模式 0 */
    int64_t n_used;           /* 参与统计的像素数（诊断用）*/
    int64_t hist_peak_bin;    /* 分位数落在哪个 bin（诊断，用于误差归属分析）*/
    int32_t status;
    int32_t reserved0;
} aaa_ae_result;

/* 不透明上下文：缓存与形状/参数相关的量（中心权重图、直方图缓冲）。
 * 对外只有指针，析构在库内 —— ABI 纪律 2。 */
typedef struct aaa_ae_ctx aaa_ae_ctx;

AAA_API int32_t aaa_ae_ctx_create(const aaa_ae_params* params,
                                  int32_t rows, int32_t cols,
                                  aaa_ae_ctx** out);
AAA_API int32_t aaa_ae_ctx_reconfigure(aaa_ae_ctx* ctx,
                                       const aaa_ae_params* params,
                                       int32_t rows, int32_t cols);
AAA_API void    aaa_ae_ctx_destroy(aaa_ae_ctx* ctx);

/* 主入口：float32 输入（F64/F32 档位） */
AAA_API int32_t aaa_ae_meter_f32(aaa_ae_ctx* ctx, const float* luma,
                                 int32_t rows, int32_t cols,
                                 int32_t stride_row, int32_t stride_col,
                                 aaa_ae_result* out);
/* 主入口：uint16 Q0.16 输入（Q16 档位）*/
AAA_API int32_t aaa_ae_meter_u16(aaa_ae_ctx* ctx, const uint16_t* luma,
                                 int32_t rows, int32_t cols,
                                 int32_t stride_row, int32_t stride_col,
                                 aaa_ae_result* out);
/* 无状态便捷入口（内部建/销 ctx），给"脱离 Python 也能用"的场景 */
AAA_API int32_t aaa_ae_meter_once_f32(const float* luma,
                                      int32_t rows, int32_t cols,
                                      const aaa_ae_params* params,
                                      aaa_ae_result* out);

/* =============================================================================
 * AWB 统计
 *
 * 只产出**统计量**，融合数学（置信度加权、对数域几何平均、普朗克约束）
 * 仍由 Python 侧执行 —— 见 Python 封装里的 fuse_illuminants。
 * 这样两条通路共用同一份决策代码，杜绝两边漂移。
 * ========================================================================== */

typedef struct {
    int32_t precision;
    int32_t hist_bins;
    int32_t bit_depth;
    /* 按需计算哪个估计器（Python 现状是无条件全算，哪怕只用 1 个）*/
    int32_t need_gray_world;
    int32_t need_white_patch;
    int32_t need_gray_edge;
    int32_t need_sog;
    int32_t reserved0;
    double  clip_level;        /* 0.98 */
    double  near_gray_val_min; /* 0.15 */
    double  near_gray_sat_max; /* 0.25 */
    double  gray_edge_thresh;  /* 0.10 */
    double  white_patch_q;     /* 99.5（百分位）*/
    double  sog_p;             /* 6.0 */
} aaa_awb_params;

typedef struct {
    double  gray_world[3];
    double  white_patch[3];
    double  gray_edge[3];
    double  shades_of_gray[3];
    double  sat_mean;          /* 掩码像素的平均饱和度 */
    double  frac_ge;           /* 全帧里"灰边"像素占比 */
    double  edge_mag_mean;     /* 诊断量 */
    int64_t n_valid;
    int64_t n_gray_edge;
    int64_t n_pixels;
    int32_t status;
    int32_t reserved0;
} aaa_awb_stats;

typedef struct aaa_awb_ctx aaa_awb_ctx;

AAA_API int32_t aaa_awb_ctx_create(const aaa_awb_params* params,
                                   int32_t rows, int32_t cols,
                                   aaa_awb_ctx** out);
AAA_API int32_t aaa_awb_ctx_reconfigure(aaa_awb_ctx* ctx,
                                        const aaa_awb_params* params,
                                        int32_t rows, int32_t cols);
AAA_API void    aaa_awb_ctx_destroy(aaa_awb_ctx* ctx);

/* RGB 交织输入（H, W, 3），通道步长通常为 1 */
AAA_API int32_t aaa_awb_stats_f32(aaa_awb_ctx* ctx, const float* rgb,
                                  int32_t rows, int32_t cols,
                                  int32_t stride_row, int32_t stride_col,
                                  int32_t stride_chan, aaa_awb_stats* out);
AAA_API int32_t aaa_awb_stats_u16(aaa_awb_ctx* ctx, const uint16_t* rgb,
                                  int32_t rows, int32_t cols,
                                  int32_t stride_row, int32_t stride_col,
                                  int32_t stride_chan, aaa_awb_stats* out);

/* =============================================================================
 * 元信息与自检
 * ========================================================================== */

AAA_API int32_t     aaa_abi_version(void);
/* 编译器/标准/-O 档等构建信息，静态字面量。报告里要原样写出，否则性能数字没有意义。 */
AAA_API const char* aaa_build_info(void);
/* 空调用：ABI 冒烟测试，同时用作 ctypes 调用开销的**实测地板**。 */
AAA_API int32_t     aaa_null_call(void);
AAA_API const char* aaa_status_string(int32_t code);

#ifdef __cplusplus
}
#endif
#endif /* AAA_STATS_H */
