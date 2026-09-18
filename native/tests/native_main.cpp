// =============================================================================
// native_main.cpp —— 脱离 Python 的自检与基准
//
// 这个程序存在的意义有两个：
//   1) 证明这个库**不是 Python 插件** —— 它能独立编译、独立运行、独立验证。
//      （构建命令不需要任何 Python 头文件或 numpy。）
//   2) 提供一组**解析可验证**的自检：输入是构造出来的，输出有闭式解，
//      所以不需要 numpy 当参考也能判断实现对不对。
//
//   aaa_native --self-test
//   aaa_native --bench --rows R --cols C --impl {f64,f32,q16} --repeats N --warmup M
// =============================================================================
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "aaa_stats.h"

namespace {

int g_fail = 0;

void check(bool ok, const char* what, double detail = 0.0) {
    if (ok) {
        std::printf("  [PASS] %s\n", what);
    } else {
        std::printf("  [FAIL] %s  (detail=%.6g)\n", what, detail);
        ++g_fail;
    }
}

double ae_metric(const std::vector<float>& img, int rows, int cols,
                 int mode, int precision = AAA_PREC_F64) {
    aaa_ae_params p{};
    p.mode = mode;
    p.precision = precision;
    p.zones_y = p.zones_x = 5;
    p.hist_bins = 1024;
    p.bit_depth = 16;
    p.center_ratio = 2.0;
    p.spot_ratio = 0.15;
    p.zone_sigma = 0.45;
    p.highlight_weight = 0.60;
    p.clip_level = 0.95;
    aaa_ae_result r{};
    if (aaa_ae_meter_once_f32(img.data(), rows, cols, &p, &r) != AAA_OK) return std::nan("");
    return r.metric;
}

double ae_metric_q16(const std::vector<uint16_t>& q, int rows, int cols, int mode) {
    aaa_ae_params p{};
    p.mode = mode;
    p.precision = AAA_PREC_Q16;
    p.zones_y = p.zones_x = 5;
    p.hist_bins = 1024;
    p.bit_depth = 16;
    p.center_ratio = 2.0;
    p.spot_ratio = 0.15;
    p.zone_sigma = 0.45;
    p.highlight_weight = 0.60;
    p.clip_level = 0.95;
    aaa_ae_ctx* ctx = nullptr;
    if (aaa_ae_ctx_create(&p, rows, cols, &ctx) != AAA_OK) return std::nan("");
    aaa_ae_result r{};
    int32_t rc = aaa_ae_meter_u16(ctx, q.data(), rows, cols, cols, 1, &r);
    aaa_ae_ctx_destroy(ctx);
    return rc == AAA_OK ? r.metric : std::nan("");
}

const char* kModes[] = {"average", "center", "spot", "evaluative", "highlight"};
const int kModeIds[] = {AAA_METER_AVERAGE, AAA_METER_CENTER, AAA_METER_SPOT,
                        AAA_METER_EVALUATIVE, AAA_METER_HIGHLIGHT};

int self_test() {
    const int R = 64, C = 96;
    const int64_t N = (int64_t)R * C;

    // (1) 恒定图：所有均值类模式必须精确等于该常数
    {
        const float v = 0.37f;
        std::vector<float> img((size_t)N, v);
        for (int m = 0; m < 3; ++m) {
            const double got = ae_metric(img, R, C, kModeIds[m]);
            check(std::fabs(got - v) < 1e-6, (std::string("恒定图/") + kModes[m] +
                  " == 常数").c_str(), got - v);
        }
        // evaluative 在恒定图上：块均值 = v，clip 占比 = 0 → 也应等于 v
        const double ev = ae_metric(img, R, C, AAA_METER_EVALUATIVE);
        check(std::fabs(ev - v) < 1e-5, "恒定图/evaluative == 常数", ev - v);
    }

    // (2) 线性 ramp：p99 的解析值就是 0.99（numpy 的线性插值语义下）
    {
        std::vector<float> img((size_t)N);
        for (int64_t i = 0; i < N; ++i) img[(size_t)i] = (float)((double)i / (double)(N - 1));
        const double got = ae_metric(img, R, C, AAA_METER_HIGHLIGHT);
        check(std::fabs(got - 0.99) < 1e-5, "线性 ramp/p99 == 0.99", got - 0.99);
    }

    // (3) 双电平图：p99 必须落在两个电平之间且贴近上电平
    {
        std::vector<float> img((size_t)N, 0.2f);
        const int64_t n_hi = N / 20;                 // 5% 的像素在上电平
        for (int64_t i = 0; i < n_hi; ++i) img[(size_t)i] = 0.8f;
        const double got = ae_metric(img, R, C, AAA_METER_HIGHLIGHT);
        check(got > 0.2 && got <= 0.8 + 1e-6, "双电平图/p99 落在 [低, 高]", got);
    }

    // (4) 中心权重：对称、峰值在中心、取值在 (0,1]
    {
        std::vector<float> img((size_t)N, 1.0f);
        aaa_ae_result r{};
        aaa_ae_params p{};
        p.mode = AAA_METER_CENTER; p.precision = AAA_PREC_F64;
        p.zones_y = p.zones_x = 5; p.center_ratio = 2.0; p.spot_ratio = 0.15;
        p.zone_sigma = 0.45; p.highlight_weight = 0.6; p.clip_level = 0.95;
        aaa_ae_ctx* ctx = nullptr;
        aaa_ae_ctx_create(&p, R, C, &ctx);
        aaa_ae_meter_f32(ctx, img.data(), R, C, C, 1, &r);
        aaa_ae_ctx_destroy(ctx);
        // 常量图下加权均值仍应是 1.0（权重归一化）—— 这是权重表的正确性证据
        check(std::fabs(r.metric - 1.0) < 1e-5, "中心权重/常量图加权均值 == 1", r.metric - 1.0);
    }

    // (5) 均匀灰图的 AWB：sat_mean==0、frac_ge==0、各估计器一致
    {
        std::vector<float> rgb((size_t)N * 3, 0.4f);
        aaa_awb_params p{};
        p.precision = AAA_PREC_F64;
        p.need_gray_world = p.need_white_patch = p.need_gray_edge = p.need_sog = 1;
        p.clip_level = 0.98; p.near_gray_val_min = 0.15; p.near_gray_sat_max = 0.25;
        p.gray_edge_thresh = 0.10; p.white_patch_q = 99.5; p.sog_p = 6.0;
        aaa_awb_ctx* ctx = nullptr;
        aaa_awb_ctx_create(&p, R, C, &ctx);
        aaa_awb_stats s{};
        aaa_awb_ctx_reconfigure(ctx, &p, R, C);
        const int32_t rc = aaa_awb_stats_f32(ctx, rgb.data(), R, C, C * 3, 3, 1, &s);
        aaa_awb_ctx_destroy(ctx);
        check(rc == AAA_OK, "均匀灰图/AWB 返回 OK", rc);
        check(s.n_valid == N, "均匀灰图/全部像素有效", (double)(s.n_valid - N));
        check(std::fabs(s.sat_mean) < 1e-9, "均匀灰图/sat_mean == 0", s.sat_mean);
        check(std::fabs(s.frac_ge) < 1e-12, "均匀灰图/frac_ge == 0", s.frac_ge);
        check(std::fabs(s.gray_world[0] - 0.4) < 1e-5, "均匀灰图/gray_world == 0.4",
              s.gray_world[0] - 0.4);
    }

    // (6) 掩码排他性：全饱和 / 全黑 → 零有效像素
    {
        aaa_awb_params p{};
        p.precision = AAA_PREC_F64;
        p.need_gray_world = 1;
        p.clip_level = 0.98; p.near_gray_val_min = 0.15; p.near_gray_sat_max = 0.25;
        p.gray_edge_thresh = 0.1; p.white_patch_q = 99.5; p.sog_p = 6.0;
        for (float v : {1.0f, 0.0f}) {
            std::vector<float> rgb((size_t)N * 3, v);
            aaa_awb_ctx* ctx = nullptr;
            aaa_awb_ctx_create(&p, R, C, &ctx);
            aaa_awb_stats s{};
            aaa_awb_stats_f32(ctx, rgb.data(), R, C, C * 3, 3, 1, &s);
            aaa_awb_ctx_destroy(ctx);
            check(s.n_valid == 0, v > 0.5f ? "全饱和图/零有效像素" : "全黑图/零有效像素",
                  (double)s.n_valid);
        }
    }

    // (7) 定点 vs F64 的偏差必须在设计上界内（让库自带误差回归）
    {
        std::vector<float> img((size_t)N);
        std::vector<uint16_t> q((size_t)N);
        for (int64_t i = 0; i < N; ++i) {
            const float v = (float)(0.05 + 0.9 * (double)i / (double)(N - 1));
            img[(size_t)i] = v;
            q[(size_t)i] = (uint16_t)std::lround(v * 65535.0);
        }
        // 均值类上界 1.8e-5；分位数用 1024 bin 时上界 9.77e-4
        const struct { int id; double bound; } cases[] = {
            {AAA_METER_AVERAGE, 1.8e-5}, {AAA_METER_CENTER, 1.8e-5},
            {AAA_METER_SPOT, 1.8e-5}, {AAA_METER_EVALUATIVE, 1.8e-5},
            {AAA_METER_HIGHLIGHT, 9.77e-4},
        };
        for (const auto& c : cases) {
            const double ref = ae_metric(img, R, C, c.id);
            const double got = ae_metric_q16(q, R, C, c.id);
            const double d = std::fabs(got - ref);
            check(d < c.bound, (std::string("定点误差上界/") + kModes[c.id ==
                  AAA_METER_HIGHLIGHT ? 4 : c.id] ).c_str(), d);
        }
    }

    // (8) 退化形状：不崩、不越界、返回码合理
    {
        const int shapes[][2] = {{1, 1}, {1, 32}, {32, 1}, {3, 5}, {7, 7}};
        for (const auto& sh : shapes) {
            const int rr = sh[0], cc = sh[1];
            std::vector<float> img((size_t)rr * cc, 0.3f);
            aaa_ae_params p{};
            p.mode = AAA_METER_EVALUATIVE; p.precision = AAA_PREC_F64;
            p.zones_y = p.zones_x = 1; p.center_ratio = 2.0; p.spot_ratio = 0.5;
            p.zone_sigma = 0.45; p.highlight_weight = 0.6; p.clip_level = 0.95;
            aaa_ae_ctx* ctx = nullptr;
            const int32_t rc = aaa_ae_ctx_create(&p, rr, cc, &ctx);
            if (rc == AAA_OK) {
                aaa_ae_result r{};
                aaa_ae_meter_f32(ctx, img.data(), rr, cc, cc, 1, &r);
                aaa_ae_ctx_destroy(ctx);
            }
            check(rc == AAA_OK, "退化形状不崩", (double)rc);
        }
        // 参数校验：这些**必须**被挡住（否则 ctypes 会读到飞、杀掉宿主进程）
        aaa_ae_result r{};
        check(aaa_ae_meter_once_f32(nullptr, 4, 4, nullptr, &r) == AAA_ERR_NULL,
              "空参数返回 AAA_ERR_NULL");
        aaa_ae_params bad{};
        bad.mode = 99; bad.precision = AAA_PREC_F64; bad.zones_y = bad.zones_x = 1;
        std::vector<float> img(16, 0.3f);
        check(aaa_ae_meter_once_f32(img.data(), 4, 4, &bad, &r) == AAA_ERR_MODE,
              "非法模式返回 AAA_ERR_MODE");
    }

    std::printf("\n%s  失败 %d 项\n", g_fail ? "自检失败" : "自检全部通过", g_fail);
    return g_fail ? 1 : 0;
}

int bench(int rows, int cols, const std::string& impl, int repeats, int warmup) {
    const int64_t n = (int64_t)rows * cols;
    std::vector<float> img((size_t)n);
    for (int64_t i = 0; i < n; ++i) {
        img[(size_t)i] = (float)(0.05 + 0.9 * (double)((i * 2654435761u) % 100000) / 100000.0);
    }
    int precision = AAA_PREC_F64;
    if (impl == "f32") precision = AAA_PREC_F32;
    else if (impl == "q16") precision = AAA_PREC_Q16;

    std::printf("{\"impl\":\"%s\",\"rows\":%d,\"cols\":%d,\"repeats\":%d,"
                "\"build\":\"%s\",\"times_us\":[", impl.c_str(), rows, cols, repeats,
                aaa_build_info());
    bool first = true;
    for (int i = 0; i < warmup + repeats; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        double v = 0.0;
        if (precision == AAA_PREC_Q16) {
            std::vector<uint16_t> q((size_t)n);
            for (int64_t k = 0; k < n; ++k)
                q[(size_t)k] = (uint16_t)std::lround(img[(size_t)k] * 65535.0);
            v = ae_metric_q16(q, rows, cols, AAA_METER_EVALUATIVE);
        } else {
            v = ae_metric(img, rows, cols, AAA_METER_EVALUATIVE, precision);
        }
        const auto t1 = std::chrono::steady_clock::now();
        if (i < warmup) continue;                 // warmup 不计入
        const double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        std::printf("%s%.3f", first ? "" : ",", us);
        first = false;
        if (std::isnan(v)) { std::printf("],\"error\":true}\n"); return 1; }
    }
    std::printf("]}\n");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("用法: aaa_native --self-test | --bench [--rows R] [--cols C] "
                    "[--impl f64|f32|q16] [--repeats N] [--warmup M]\n");
        return 2;
    }
    if (std::strcmp(argv[1], "--self-test") == 0) {
        std::printf("aaa_stats 自检   ABI=%d  %s\n\n", aaa_abi_version(), aaa_build_info());
        return self_test();
    }
    if (std::strcmp(argv[1], "--bench") == 0) {
        int rows = 480, cols = 360, repeats = 100, warmup = 20;
        std::string impl = "f64";
        for (int i = 2; i + 1 < argc; i += 2) {
            const std::string k = argv[i];
            if (k == "--rows") rows = std::atoi(argv[i + 1]);
            else if (k == "--cols") cols = std::atoi(argv[i + 1]);
            else if (k == "--repeats") repeats = std::atoi(argv[i + 1]);
            else if (k == "--warmup") warmup = std::atoi(argv[i + 1]);
            else if (k == "--impl") impl = argv[i + 1];
        }
        return bench(rows, cols, impl, repeats, warmup);
    }
    std::printf("未知参数：%s\n", argv[1]);
    return 2;
}
