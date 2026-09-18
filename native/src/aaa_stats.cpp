// =============================================================================
// aaa_stats.cpp —— 导出层：参数校验 + 异常兜底 + 转发到内部实现
//
// 这一层只干三件事，其余全是实现细节：
//   1) 校验**所有**入参（因为 ctypes 传错参数导致的段错误会杀掉宿主进程）
//   2) 把异常兜在边界内（ABI 纪律 3）
//   3) 把 C 结构体参数翻译成内部的 C++ 类型
// =============================================================================
#include "aaa_stats.h"

#include <cstddef>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <vector>

#include "ae_metering.hpp"
#include "awb_stats.hpp"

// -----------------------------------------------------------------------------
// ABI 纪律 1：结构体布局用编译期断言固化，不靠"MSVC 与 MinGW 通常一样"
// -----------------------------------------------------------------------------
static_assert(sizeof(aaa_ae_result) == 40, "aaa_ae_result 布局变了");
static_assert(offsetof(aaa_ae_result, metric) == 0, "metric 偏移变了");
static_assert(offsetof(aaa_ae_result, n_used) == 16, "n_used 偏移变了");
static_assert(sizeof(aaa_awb_stats) == 152, "aaa_awb_stats 布局变了");
static_assert(offsetof(aaa_awb_stats, sat_mean) == 96, "sat_mean 偏移变了");
static_assert(sizeof(aaa_ae_params) == 96, "aaa_ae_params 布局变了");
static_assert(sizeof(aaa_awb_params) == 80, "aaa_awb_params 布局变了");
// Python 侧（api.py 的 ctypes Structure）必须与这些尺寸一致，否则 ctypes 会
// 按错误的步长解释内存。所有尺寸都是 8 的倍数，跨工具链对齐规则一致。
static_assert(sizeof(aaa_ae_result) % 8 == 0, "");
static_assert(sizeof(aaa_awb_stats) % 8 == 0, "");
static_assert(sizeof(aaa_ae_params) % 8 == 0, "");
static_assert(sizeof(aaa_awb_params) % 8 == 0, "");

namespace {

constexpr int32_t kAbiVersion = AAA_STATS_ABI_VERSION;

// 构建信息：报告里要原样写出，否则性能数字没有意义（不知道用什么编的）
const char* build_info_literal() {
#ifdef __clang__
    const char* cc = "clang " __clang_version__;
#elif defined(__GNUC__)
    const char* cc = "gcc " __VERSION__;
#else
    const char* cc = "unknown compiler";
#endif
#ifdef NDEBUG
    const char* mode = "release";
#else
    const char* mode = "debug";
#endif
    static const std::string info = std::string(cc) + " | C++" + std::to_string(__cplusplus / 100 % 100) +
                                    " | " + mode + " | " + __DATE__ + " " + __TIME__;
    return info.c_str();
}

// 统一的数据指针校验：任何一项不合法都返回错误码，绝不让内部读到飞
int32_t check_layout(int32_t rows, int32_t cols,
                     int32_t stride_row, int32_t stride_col) {
    if (rows <= 0 || cols <= 0) return AAA_ERR_SHAPE;
    if (stride_row < 1 || stride_col < 1) return AAA_ERR_SHAPE;
    return AAA_OK;
}

bool valid_precision(int32_t p) {
    return p == AAA_PREC_F64 || p == AAA_PREC_F32 || p == AAA_PREC_Q16;
}

int32_t validate_ae_params(const aaa_ae_params* p) {
    if (!p) return AAA_ERR_NULL;
    if (p->mode < AAA_METER_AVERAGE || p->mode > AAA_METER_HIGHLIGHT) return AAA_ERR_MODE;
    if (!valid_precision(p->precision)) return AAA_ERR_MODE;
    if (p->zones_y <= 0 || p->zones_x <= 0) return AAA_ERR_RANGE;
    if (p->zones_y > 64 || p->zones_x > 64) return AAA_ERR_RANGE;
    if (p->hist_bins != 0 && (p->hist_bins & (p->hist_bins - 1)) != 0) return AAA_ERR_RANGE;
    if (p->bit_depth != 0 && (p->bit_depth < 4 || p->bit_depth > 16)) return AAA_ERR_RANGE;
    if (!(p->spot_ratio > 0.0 && p->spot_ratio <= 1.0)) return AAA_ERR_RANGE;
    if (p->flags & 1) {
        if (!p->zone_w) return AAA_ERR_NULL;
        if (p->zone_w_len < p->zones_y * p->zones_x) return AAA_ERR_RANGE;
    }
    return AAA_OK;
}

}  // namespace

// -----------------------------------------------------------------------------
// 上下文：缓存与形状/参数相关的量（中心权重图、直方图缓冲、分区边界）
// 对外只有不透明指针，析构在库内 —— ABI 纪律 2
// -----------------------------------------------------------------------------
struct aaa_ae_ctx {
    aaa_ae_params params{};
    int32_t rows = 0, cols = 0;
    bool params_valid = false;
    aaa::AEMetering impl;      // 内部实现（自带缓存）
};

struct aaa_awb_ctx {
    aaa_awb_params params{};
    int32_t rows = 0, cols = 0;
    bool params_valid = false;
    aaa::AWBStats impl;
};

extern "C" {

int32_t aaa_abi_version(void) { return kAbiVersion; }
const char* aaa_build_info(void) { return build_info_literal(); }
int32_t aaa_null_call(void) { return 0; }

const char* aaa_status_string(int32_t code) {
    switch (code) {
        case AAA_OK:           return "ok";
        case AAA_ERR_NULL:     return "null pointer";
        case AAA_ERR_SHAPE:    return "invalid shape/stride";
        case AAA_ERR_MODE:     return "invalid mode or precision";
        case AAA_ERR_RANGE:    return "parameter out of range";
        case AAA_ERR_INTERNAL: return "internal error";
        default:               return "unknown";
    }
}

int32_t aaa_ae_ctx_create(const aaa_ae_params* params, int32_t rows, int32_t cols,
                          aaa_ae_ctx** out) {
    if (!out) return AAA_ERR_NULL;
    *out = nullptr;
    int32_t rc = validate_ae_params(params);
    if (rc != AAA_OK) return rc;
    rc = check_layout(rows, cols, 1, 1);
    if (rc != AAA_OK) return rc;
    try {
        auto* ctx = new aaa_ae_ctx();
        ctx->params = *params;
        ctx->rows = rows;
        ctx->cols = cols;
        ctx->params_valid = true;
        rc = ctx->impl.configure(*params, rows, cols);
        if (rc != AAA_OK) { delete ctx; return rc; }
        *out = ctx;
        return AAA_OK;
    } catch (const std::bad_alloc&) {
        return AAA_ERR_INTERNAL;
    } catch (...) {
        return AAA_ERR_INTERNAL;
    }
}

int32_t aaa_ae_ctx_reconfigure(aaa_ae_ctx* ctx, const aaa_ae_params* params,
                               int32_t rows, int32_t cols) {
    if (!ctx || !params) return AAA_ERR_NULL;
    int32_t rc = validate_ae_params(params);
    if (rc != AAA_OK) return rc;
    rc = check_layout(rows, cols, 1, 1);
    if (rc != AAA_OK) return rc;
    try {
        ctx->params = *params;
        ctx->rows = rows;
        ctx->cols = cols;
        return ctx->impl.configure(*params, rows, cols);
    } catch (...) {
        return AAA_ERR_INTERNAL;
    }
}

void aaa_ae_ctx_destroy(aaa_ae_ctx* ctx) {
    delete ctx;      // 析构在库内：C++ 的 delete 不跨边界
}

int32_t aaa_ae_meter_f32(aaa_ae_ctx* ctx, const float* luma,
                         int32_t rows, int32_t cols,
                         int32_t stride_row, int32_t stride_col,
                         aaa_ae_result* out) {
    if (!ctx || !luma || !out) return AAA_ERR_NULL;
    if (!ctx->params_valid) return AAA_ERR_INTERNAL;
    int32_t rc = check_layout(rows, cols, stride_row, stride_col);
    if (rc != AAA_OK) return rc;
    // 帧尺寸必须与建 ctx 时一致（分区边界与权重图都按那个形状缓存的）
    if (rows != ctx->rows || cols != ctx->cols) return AAA_ERR_SHAPE;
    try {
        rc = ctx->impl.meter_f32(luma, rows, cols, stride_row, stride_col, *out);
        out->status = rc;
        return rc;
    } catch (const std::bad_alloc&) {
        return AAA_ERR_INTERNAL;
    } catch (...) {
        return AAA_ERR_INTERNAL;
    }
}

int32_t aaa_ae_meter_u16(aaa_ae_ctx* ctx, const uint16_t* luma,
                         int32_t rows, int32_t cols,
                         int32_t stride_row, int32_t stride_col,
                         aaa_ae_result* out) {
    if (!ctx || !luma || !out) return AAA_ERR_NULL;
    if (!ctx->params_valid) return AAA_ERR_INTERNAL;
    int32_t rc = check_layout(rows, cols, stride_row, stride_col);
    if (rc != AAA_OK) return rc;
    if (rows != ctx->rows || cols != ctx->cols) return AAA_ERR_SHAPE;
    try {
        rc = ctx->impl.meter_u16(luma, rows, cols, stride_row, stride_col, *out);
        out->status = rc;
        return rc;
    } catch (...) {
        return AAA_ERR_INTERNAL;
    }
}

int32_t aaa_ae_meter_once_f32(const float* luma, int32_t rows, int32_t cols,
                              const aaa_ae_params* params, aaa_ae_result* out) {
    aaa_ae_ctx* ctx = nullptr;
    int32_t rc = aaa_ae_ctx_create(params, rows, cols, &ctx);
    if (rc != AAA_OK) return rc;
    rc = aaa_ae_meter_f32(ctx, luma, rows, cols, cols, 1, out);
    aaa_ae_ctx_destroy(ctx);
    return rc;
}

// --- AWB ---
int32_t aaa_awb_ctx_create(const aaa_awb_params* params, int32_t rows, int32_t cols,
                           aaa_awb_ctx** out) {
    if (!out) return AAA_ERR_NULL;
    *out = nullptr;
    if (!params) return AAA_ERR_NULL;
    if (!valid_precision(params->precision)) return AAA_ERR_MODE;
    if (params->hist_bins != 0 && (params->hist_bins & (params->hist_bins - 1)) != 0)
        return AAA_ERR_RANGE;
    int32_t rc = check_layout(rows, cols, 1, 1);
    if (rc != AAA_OK) return rc;
    if (!(params->sog_p > 0.0)) return AAA_ERR_RANGE;
    try {
        auto* ctx = new aaa_awb_ctx();
        ctx->params = *params;
        ctx->rows = rows;
        ctx->cols = cols;
        ctx->params_valid = true;
        rc = ctx->impl.configure(*params, rows, cols);
        if (rc != AAA_OK) { delete ctx; return rc; }
        *out = ctx;
        return AAA_OK;
    } catch (...) {
        return AAA_ERR_INTERNAL;
    }
}

int32_t aaa_awb_ctx_reconfigure(aaa_awb_ctx* ctx, const aaa_awb_params* params,
                                int32_t rows, int32_t cols) {
    if (!ctx || !params) return AAA_ERR_NULL;
    int32_t rc = check_layout(rows, cols, 1, 1);
    if (rc != AAA_OK) return rc;
    try {
        ctx->params = *params;
        ctx->rows = rows;
        ctx->cols = cols;
        return ctx->impl.configure(*params, rows, cols);
    } catch (...) {
        return AAA_ERR_INTERNAL;
    }
}

void aaa_awb_ctx_destroy(aaa_awb_ctx* ctx) { delete ctx; }

int32_t aaa_awb_stats_f32(aaa_awb_ctx* ctx, const float* rgb,
                          int32_t rows, int32_t cols,
                          int32_t sr, int32_t sc, int32_t schan,
                          aaa_awb_stats* out) {
    if (!ctx || !rgb || !out) return AAA_ERR_NULL;
    if (!ctx->params_valid) return AAA_ERR_INTERNAL;
    int32_t rc = check_layout(rows, cols, sr, sc);
    if (rc != AAA_OK) return rc;
    if (schan < 1) return AAA_ERR_SHAPE;
    if (rows != ctx->rows || cols != ctx->cols) return AAA_ERR_SHAPE;
    try {
        rc = ctx->impl.compute_f32(rgb, rows, cols, sr, sc, schan, *out);
        out->status = rc;
        return rc;
    } catch (const std::bad_alloc&) {
        return AAA_ERR_INTERNAL;
    } catch (...) {
        return AAA_ERR_INTERNAL;
    }
}

int32_t aaa_awb_stats_u16(aaa_awb_ctx* ctx, const uint16_t* rgb,
                          int32_t rows, int32_t cols,
                          int32_t sr, int32_t sc, int32_t schan,
                          aaa_awb_stats* out) {
    if (!ctx || !rgb || !out) return AAA_ERR_NULL;
    int32_t rc = check_layout(rows, cols, sr, sc);
    if (rc != AAA_OK) return rc;
    try {
        rc = ctx->impl.compute_u16(rgb, rows, cols, sr, sc, schan, *out);
        out->status = rc;
        return rc;
    } catch (...) {
        return AAA_ERR_INTERNAL;
    }
}

}  // extern "C"
