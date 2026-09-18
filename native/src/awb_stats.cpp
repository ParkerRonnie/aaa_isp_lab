// =============================================================================
// awb_stats.cpp —— AWB 统计通路的实现
// =============================================================================
#include "awb_stats.hpp"

#include <algorithm>
#include <cmath>

#include "ae_metering.hpp"   // 复用 quantile_linear_sorted

namespace aaa {

int32_t reflect101(int32_t i, int32_t n) {
    if (n <= 1) return 0;
    while (i < 0 || i >= n) {
        if (i < 0) i = -i;
        else i = 2 * (n - 1) - i;
    }
    return i;
}

int32_t AWBStats::configure(const aaa_awb_params& p, int32_t rows, int32_t cols) {
    p_ = p;
    rows_ = rows;
    cols_ = cols;
    gray_.assign((size_t)rows * (size_t)cols, 0.0f);
    gray_q_.assign((size_t)rows * (size_t)cols, (uint16_t)0);
    keep_.assign((size_t)rows * (size_t)cols, 0);
    if (p.precision == AAA_PREC_Q16) {
        hist_bins_ = p.hist_bins ? p.hist_bins : 1024;
        hist_bits_ = 0;
        for (int b = 1; b < hist_bins_; b <<= 1) ++hist_bits_;
        for (auto& hh : hist_) hh.assign((size_t)hist_bins_, 0);
    }
    for (auto& c : ch_) {
        c.clear();
        c.reserve((size_t)rows * (size_t)cols / 4 + 16);
    }
    scratch_.clear();
    ready_ = true;
    return AAA_OK;
}

int32_t AWBStats::compute_f32(const float* rgb, int32_t rows, int32_t cols,
                              int32_t sr, int32_t sc, int32_t schan,
                              aaa_awb_stats& out) {
    if (!ready_ || rows != rows_ || cols != cols_) return AAA_ERR_SHAPE;
    if (!rgb) return AAA_ERR_NULL;
    out = aaa_awb_stats{};
    out.n_pixels = (int64_t)rows * cols;

    const double clip = p_.clip_level;
    const double val_min = p_.near_gray_val_min;
    const double sat_max = p_.near_gray_sat_max;
    const double p_sog = p_.sog_p;
    const int64_t n_pix = (int64_t)rows * cols;

    // ---- Pass 1：逐像素顺序访存，一次读进 (R,G,B) 把所有能算的都算掉 ----
    double sum[3] = {0.0, 0.0, 0.0};
    double sog_sum[3] = {0.0, 0.0, 0.0};
    double sat_sum = 0.0;
    int64_t n_valid = 0;
    for (auto& c : ch_) c.clear();

    for (int32_t y = 0; y < rows; ++y) {
        const float* row = rgb + (int64_t)y * sr;
        float* grow = gray_.data() + (size_t)y * cols;
        for (int32_t x = 0; x < cols; ++x) {
            const float* px = row + (int64_t)x * sc;
            const float r = px[0];
            const float g = px[1 * (int64_t)schan];
            const float b = px[2 * (int64_t)schan];
            const double luma = 0.2126 * (double)r + 0.7152 * (double)g + 0.0722 * (double)b;
            grow[x] = (float)luma;

            const double mx = std::max((double)r, std::max((double)g, (double)b));
            const double mn = std::min((double)r, std::min((double)g, (double)b));
            const bool not_clip = mx < clip;
            const bool not_dark = luma > val_min;
            if (!(not_clip && not_dark)) continue;    // 掩码不命中：跳过一切累加

            ++n_valid;
            sum[0] += r; sum[1] += g; sum[2] += b;
            sat_sum += (mx - mn) / std::max(mx, 1e-6);
            if (p_.need_white_patch) {
                ch_[0].push_back(r); ch_[1].push_back(g); ch_[2].push_back(b);
            }
            if (p_.need_sog) {
                // Python 是 px ** 6.0（逐元素 pow）。这里用三次乘法 ——
                // 数值上不完全相同（pow 的实现路径不同），但等价性测试给的
                // 相对容差 1e-6 对这一处有充分余量（见报告里的容差表）。
                const double r2 = (double)r * (double)r, g2 = (double)g * (double)g,
                             b2 = (double)b * (double)b;
                sog_sum[0] += r2 * r2 * r2;
                sog_sum[1] += g2 * g2 * g2;
                sog_sum[2] += b2 * b2 * b2;
            }
        }
    }

    if (n_valid < 16) {
        // 与 Python 一致：有效像素不足时四个估计器都退化成 ones
        for (int c = 0; c < 3; ++c) {
            out.gray_world[c] = out.white_patch[c] = out.gray_edge[c] = out.shades_of_gray[c] = 1.0;
        }
        out.n_valid = n_valid;
        return AAA_OK;
    }

    if (p_.need_gray_world) {
        for (int c = 0; c < 3; ++c) out.gray_world[c] = sum[c] / (double)n_valid;
    }
    if (p_.need_white_patch) {
        // 每通道的 99.5 分位，逐位复刻 np.percentile（同 AE 的 highlight 路径）。
        // 注意要用 inplace 版本：收集出来的样本是**未排序**的，直接按"已排序"
        // 去取第 k 个元素会得到一个语义完全不同的数（这里踩过一次）。
        for (int c = 0; c < 3; ++c) {
            auto& v = ch_[c];
            scratch_.assign(v.begin(), v.end());
            out.white_patch[c] = quantile_linear_inplace(scratch_.data(),
                                                         (int64_t)scratch_.size(),
                                                         p_.white_patch_q / 100.0);
        }
    }
    if (p_.need_sog) {
        for (int c = 0; c < 3; ++c) {
            out.shades_of_gray[c] = std::pow(sog_sum[c] / (double)n_valid, 1.0 / p_sog);
        }
    }
    out.sat_mean = sat_sum / (double)n_valid;
    out.n_valid = n_valid;

    // ---- Pass 2：只对 gray 缓冲做**一次** Sobel，同时得出灰边的两样东西 ----
    // gray_edge 的 keep 掩码（用于掩码均值）与 frac_ge 的计数（全帧占比）。
    // 两处都只需要布尔，所以不做 sqrt。
    const double t2 = p_.gray_edge_thresh * p_.gray_edge_thresh;
    double ge_sum[3] = {0.0, 0.0, 0.0};
    int64_t n_ge = 0, frac_cnt = 0, mag_cnt = 0;
    double mag_sum = 0.0;

    for (int32_t y = 0; y < rows; ++y) {
        const float* row = rgb + (int64_t)y * sr;
        for (int32_t x = 0; x < cols; ++x) {
            // 与 Python 的 valid_mask 相同的判据（这里要独立判一次，因为
            // frac_ge 的分母是**全帧**而不是掩码像素）
            const float* px = row + (int64_t)x * sc;
            const float r = px[0];
            const float g = px[1 * (int64_t)schan];
            const float b = px[2 * (int64_t)schan];
            const double mx = std::max((double)r, std::max((double)g, (double)b));
            const double mn = std::min((double)r, std::min((double)g, (double)b));
            const double mx_safe = std::max(mx, 1e-6);
            const bool in_mask = (mx < clip) &&
                (0.2126 * (double)r + 0.7152 * (double)g + 0.0722 * (double)b > val_min);
            const bool sat_ok = ((mx - mn) / mx_safe) < sat_max;

            if (!in_mask && !(sat_ok)) continue;   // 两个用途都不需要这个像素

            // Sobel（相关，REFLECT_101 边界）
            double gx = 0.0, gy = 0.0;
            const int y0 = reflect101(y - 1, rows), y1 = y, y2 = reflect101(y + 1, rows);
            const int x0 = reflect101(x - 1, cols), x1 = x, x2 = reflect101(x + 1, cols);
            const float* gy0 = gray_.data() + (size_t)y0 * cols;
            const float* gy1 = gray_.data() + (size_t)y1 * cols;
            const float* gy2 = gray_.data() + (size_t)y2 * cols;
            const double a00 = gy0[x0], a01 = gy0[x1], a02 = gy0[x2];
            const double a10 = gy1[x0], a12 = gy1[x2];
            const double a20 = gy2[x0], a21 = gy2[x1], a22 = gy2[x2];
            gx = -a00 + a02 - 2.0 * a10 + 2.0 * a12 - a20 + a22;
            gy = -a00 - 2.0 * a01 - a02 + a20 + 2.0 * a21 + a22;
            const double mag2 = gx * gx + gy * gy;   // 不做 sqrt

            if (in_mask && sat_ok && mag2 > t2) {
                ge_sum[0] += r; ge_sum[1] += g; ge_sum[2] += b;
                ++n_ge;
            }
            if (in_mask && sat_ok) {
                ++frac_cnt;
                if (mag2 > t2) { ++mag_cnt; mag_sum += std::sqrt(mag2); }
            }
        }
    }

    if (p_.need_gray_edge) {
        if (n_ge < 16) {
            out.gray_edge[0] = out.gray_edge[1] = out.gray_edge[2] = 1.0;
        } else {
            for (int c = 0; c < 3; ++c) out.gray_edge[c] = ge_sum[c] / (double)n_ge;
        }
    }
    out.n_gray_edge = n_ge;
    out.frac_ge = (double)mag_cnt / (double)n_pix;
    out.edge_mag_mean = mag_cnt > 0 ? mag_sum / (double)mag_cnt : 0.0;
    return AAA_OK;
}

// -----------------------------------------------------------------------------
// 定点通路：uint16 Q0.16 输入 + int64 累加 + 直方图分位数
//
// SoG 的 x^6 用 __int128 累加：Q16 下 v^6 最大 65535^6 ≈ 7.9e28，远超 int64
// （9.2e18）。这是 GCC/Clang 的扩展 —— 本项目的构建前提就是"不用 MSVC"
// （见 native/README.md），所以这个依赖是成立的；代价是换精度而不是换设计。
// -----------------------------------------------------------------------------
int32_t AWBStats::compute_u16(const uint16_t* rgb, int32_t rows, int32_t cols,
                              int32_t sr, int32_t sc, int32_t schan,
                              aaa_awb_stats& out) {
    if (!ready_ || rows != rows_ || cols != cols_) return AAA_ERR_SHAPE;
    if (!rgb) return AAA_ERR_NULL;
    out = aaa_awb_stats{};
    out.n_pixels = (int64_t)rows * cols;

    constexpr double kInv16 = 1.0 / 65535.0;
    const int32_t clip_q = (int32_t)std::lround(p_.clip_level * 65535.0);
    const int32_t vmin_q = (int32_t)std::lround(p_.near_gray_val_min * 65535.0);
    const double sat_max = p_.near_gray_sat_max;
    const double t2 = p_.gray_edge_thresh * p_.gray_edge_thresh
                      * 65535.0 * 65535.0;      // mag² 的 Q30 尺度阈值
    const int64_t n_pix = (int64_t)rows * cols;
    const int shift = 16 - hist_bits_;

    // ---- Pass 1 ----
    int64_t sum[3] = {0, 0, 0};
    __int128 sog_sum[3] = {0, 0, 0};
    int64_t sat_sum_q = 0;              // 饱和度的 Q16 累加
    int64_t n_valid = 0;
    if (p_.need_white_patch) {
        for (auto& hh : hist_) std::fill(hh.begin(), hh.end(), 0);
    }

    for (int32_t y = 0; y < rows; ++y) {
        const uint16_t* row = rgb + (int64_t)y * sr;
        uint16_t* grow = gray_q_.data() + (size_t)y * cols;
        for (int32_t x = 0; x < cols; ++x) {
            const uint16_t* px = row + (int64_t)x * sc;
            const int32_t r = px[0];
            const int32_t g = px[1 * (int64_t)schan];
            const int32_t b = px[2 * (int64_t)schan];
            // luma 用定点权重：0.2126/0.7152/0.0722 放大到 1e4（权重舍入误差
            // 约 1e-4 相对，会被单独计入误差预算，不假装它是精确的）
            const int32_t luma_q = (2126 * r + 7152 * g + 722 * b) / 10000;
            grow[x] = (uint16_t)std::min(std::max(luma_q, 0), 65535);

            const int32_t mx = std::max(r, std::max(g, b));
            const int32_t mn = std::min(r, std::min(g, b));
            if (!(mx < clip_q && luma_q > vmin_q)) continue;

            ++n_valid;
            sum[0] += r; sum[1] += g; sum[2] += b;
            sat_sum_q += (mx > 0) ? (((int64_t)(mx - mn) * 65535) / mx) : 0;
            if (p_.need_white_patch) {
                hist_[0][(size_t)(r >> shift)] += 1;
                hist_[1][(size_t)(g >> shift)] += 1;
                hist_[2][(size_t)(b >> shift)] += 1;
            }
            if (p_.need_sog) {
                const __int128 r6 = (__int128)r * r * r * r * r * r;
                const __int128 g6 = (__int128)g * g * g * g * g * g;
                const __int128 b6 = (__int128)b * b * b * b * b * b;
                sog_sum[0] += r6; sog_sum[1] += g6; sog_sum[2] += b6;
            }
        }
    }

    if (n_valid < 16) {
        for (int c = 0; c < 3; ++c) {
            out.gray_world[c] = out.white_patch[c] = out.gray_edge[c] = out.shades_of_gray[c] = 1.0;
        }
        out.n_valid = n_valid;
        return AAA_OK;
    }

    if (p_.need_gray_world) {
        for (int c = 0; c < 3; ++c) {
            out.gray_world[c] = ((double)sum[c] / (double)n_valid) * kInv16;
        }
    }
    if (p_.need_white_patch) {
        const double q = p_.white_patch_q / 100.0;
        const double vi = (double)n_valid * q + (1.0 + q * (1.0 - 1.0 - 1.0)) - 1.0;
        for (int c = 0; c < 3; ++c) {
            const int32_t* h = hist_[c].data();
            double scale;
            if (vi <= 0.0) scale = hist_order_stat(h, hist_bins_, 0);
            else if (vi >= (double)(n_valid - 1)) scale = hist_order_stat(h, hist_bins_, n_valid - 1);
            else {
                const int64_t lo = (int64_t)std::floor(vi);
                const double gg = vi - (double)lo;
                const double a = hist_order_stat(h, hist_bins_, lo);
                const double b = hist_order_stat(h, hist_bins_, lo + 1);
                scale = a + (b - a) * gg;
            }
            out.white_patch[c] = scale / (double)(1 << hist_bits_);
        }
    }
    if (p_.need_sog) {
        for (int c = 0; c < 3; ++c) {
            const double mean_q6 = (double)(sog_sum[c] / (__int128)n_valid);
            out.shades_of_gray[c] = std::pow(mean_q6, 1.0 / p_.sog_p) * kInv16;
        }
    }
    out.sat_mean = ((double)sat_sum_q / (double)n_valid) * kInv16;
    out.n_valid = n_valid;

    // ---- Pass 2：一次 Sobel（gx²+gy² 在 int64 里精确无舍入）----
    int64_t ge_sum[3] = {0, 0, 0};
    int64_t n_ge = 0, frac_cnt = 0, mag_cnt = 0;
    double mag_sum = 0.0;
    for (int32_t y = 0; y < rows; ++y) {
        const uint16_t* row = rgb + (int64_t)y * sr;
        for (int32_t x = 0; x < cols; ++x) {
            const uint16_t* px = row + (int64_t)x * sc;
            const int32_t r = px[0];
            const int32_t g = px[1 * (int64_t)schan];
            const int32_t b = px[2 * (int64_t)schan];
            const int32_t mx = std::max(r, std::max(g, b));
            const int32_t mn = std::min(r, std::min(g, b));
            const int32_t luma_q = (2126 * r + 7152 * g + 722 * b) / 10000;
            const bool in_mask = (mx < clip_q) && (luma_q > vmin_q);
            const bool sat_ok = (mx > 0) && (((double)(mx - mn) / (double)mx) < sat_max);
            if (!in_mask && !sat_ok) continue;

            const int y0 = reflect101(y - 1, rows), y2 = reflect101(y + 1, rows);
            const int x0 = reflect101(x - 1, cols), x2 = reflect101(x + 1, cols);
            const uint16_t* gy0 = gray_q_.data() + (size_t)y0 * cols;
            const uint16_t* gy1 = gray_q_.data() + (size_t)y * cols;
            const uint16_t* gy2 = gray_q_.data() + (size_t)y2 * cols;
            const int64_t a00 = gy0[x0], a01 = gy0[x], a02 = gy0[x2];
            const int64_t a10 = gy1[x0], a12 = gy1[x2];
            const int64_t a20 = gy2[x0], a21 = gy2[x], a22 = gy2[x2];
            const int64_t gx = -a00 + a02 - 2 * a10 + 2 * a12 - a20 + a22;
            const int64_t gy = -a00 - 2 * a01 - a02 + a20 + 2 * a21 + a22;
            const int64_t mag2 = gx * gx + gy * gy;      // 精确，无舍入

            if (in_mask && sat_ok && (double)mag2 > t2) {
                ge_sum[0] += r; ge_sum[1] += g; ge_sum[2] += b;
                ++n_ge;
            }
            if (in_mask && sat_ok) {
                ++frac_cnt;
                if ((double)mag2 > t2) { ++mag_cnt; mag_sum += std::sqrt((double)mag2); }
            }
        }
    }

    if (p_.need_gray_edge) {
        if (n_ge < 16) {
            out.gray_edge[0] = out.gray_edge[1] = out.gray_edge[2] = 1.0;
        } else {
            for (int c = 0; c < 3; ++c) {
                out.gray_edge[c] = ((double)ge_sum[c] / (double)n_ge) * kInv16;
            }
        }
    }
    out.n_gray_edge = n_ge;
    out.frac_ge = (double)mag_cnt / (double)n_pix;
    out.edge_mag_mean = mag_cnt > 0 ? (mag_sum / (double)mag_cnt) * kInv16 : 0.0;
    return AAA_OK;
}

}  // namespace aaa
