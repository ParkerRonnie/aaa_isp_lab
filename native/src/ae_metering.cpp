// =============================================================================
// ae_metering.cpp —— AE 测光统计的实现
//
// 与 Python 现状（ae.py::metering_metric）相比，这一份的两个真改进写在
// configure() 与 center 分支里：权重表只建一次（Python 那边每帧重建整幅
// h×w 权重图、每像素一次 exp）。其余模式是等价的直译。
// =============================================================================
#include "ae_metering.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace aaa {

// -----------------------------------------------------------------------------
// numpy 逐位复刻
// -----------------------------------------------------------------------------
double quantile_linear_sorted(const float* s, int64_t n, double q) {
    if (n <= 0) return std::nan("");
    if (n == 1) return (double)s[0];
    // 照抄 numpy 的 _compute_virtual_index(n, q, alpha=1, beta=1)：
    //   n*q + (alpha + q*(1-alpha-beta)) - 1
    // 直接写成 q*(n-1) 在末几位会不一致，这里的顺序不能动。
    const double vi = (double)n * q + (1.0 + q * (1.0 - 1.0 - 1.0)) - 1.0;
    if (vi <= 0.0) return (double)s[0];
    if (vi >= (double)(n - 1)) return (double)s[n - 1];
    const double pf = std::floor(vi);
    const int64_t pi = (int64_t)pf;
    const double g = vi - pf;
    const float a = s[pi];
    const float b = s[pi + 1];
    const float diff = b - a;                 // float32 减法：与 numpy 一致
    // numpy 的 _lerp：t < 0.5 走前一支，否则走后一支（两个分支数值上等价，
    // 但浮点上不同，必须照抄分支）
    return (g < 0.5) ? ((double)a + (double)diff * g)
                     : ((double)b - (double)diff * (1.0 - g));
}

double quantile_linear_inplace(float* buf, int64_t n, double q) {
    if (n <= 0) return std::nan("");
    if (n == 1) return (double)buf[0];
    const double vi = (double)n * q + (1.0 + q * (1.0 - 1.0 - 1.0)) - 1.0;
    if (vi <= 0.0) return (double)*std::min_element(buf, buf + n);
    if (vi >= (double)(n - 1)) return (double)*std::max_element(buf, buf + n);
    const double pf = std::floor(vi);
    const int64_t pi = (int64_t)pf;
    const double g = vi - pf;
    // 只要 virtual_index 附近的两个元素，用 nth_element 而不是全排序
    std::nth_element(buf, buf + pi, buf + n);
    const float a = buf[pi];
    const float b = *std::min_element(buf + pi + 1, buf + n);
    const float diff = b - a;
    return (g < 0.5) ? ((double)a + (double)diff * g)
                     : ((double)b - (double)diff * (1.0 - g));
}

int64_t linspace_int(int64_t i, double start, double stop, int64_t num) {
    if (num <= 1) return (int64_t)start;
    if (i >= num - 1) return (int64_t)stop;      // numpy 把末元素精确设为 stop
    const double step = (stop - start) / (double)(num - 1);
    return (int64_t)(start + (double)i * step);  // astype(int) 是**截断**
}

// -----------------------------------------------------------------------------
// configure：把所有与帧无关的量一次算好
// -----------------------------------------------------------------------------
int32_t AEMetering::configure(const aaa_ae_params& p, int32_t rows, int32_t cols) {
    p_ = p;
    rows_ = rows;
    cols_ = cols;
    center_w_.clear();
    ys_.clear();
    xs_.clear();
    zone_w_.clear();
    sort_buf_.clear();

    const double ratio = std::max(p.center_ratio, 1e-3);
    const double denom = 2.0 * 0.35 * 0.35 * ratio;

    // 定点尺度由 bit_depth 决定（位宽扫描要真的改变量化误差）
    const int bits = (p.bit_depth >= 4 && p.bit_depth <= 16) ? p.bit_depth : 16;
    q_scale_ = (1 << bits) - 1;
    q_bits_ = bits;
    inv_q_scale_ = 1.0 / (double)q_scale_;
    clip_q_ = (int32_t)std::lround(p.clip_level * (double)q_scale_);

    if (p.mode == AAA_METER_CENTER) {
        // 这一整幅权重图在 Python 那边是**每帧**重建的（h×w 次 exp）。
        // 它只依赖 (h, w, ratio)，所以在这里建一次就够。
        const double cy = (rows - 1) / 2.0;
        const double cx = (cols - 1) / 2.0;
        if (p.precision == AAA_PREC_Q16) {
            // 定点路径用**可分离**形式：exp(-(a+b)) = exp(-a)·exp(-b)。
            // 只需要 h+w 个 Q0.15 系数，h×w 的权重图根本不落内存。
            // 代价是数值上与"先加后 exp"不完全相同 —— 这个差别会被单独计量。
            center_wy_q15_.resize((size_t)rows);
            center_wx_q15_.resize((size_t)cols);
            for (int32_t y = 0; y < rows; ++y) {
                const double dy = (cy != 0.0) ? ((double)y - cy) / cy : std::nan("");
                center_wy_q15_[(size_t)y] = (int32_t)std::lround(std::exp(-dy * dy / denom) * 32768.0);
            }
            for (int32_t x = 0; x < cols; ++x) {
                const double dx = (cx != 0.0) ? ((double)x - cx) / cx : std::nan("");
                center_wx_q15_[(size_t)x] = (int32_t)std::lround(std::exp(-dx * dx / denom) * 32768.0);
            }
        } else {
            center_w_.resize((size_t)rows * (size_t)cols);
            for (int32_t y = 0; y < rows; ++y) {
                const double dy = (cy != 0.0) ? ((double)y - cy) / cy : std::nan("");
                for (int32_t x = 0; x < cols; ++x) {
                    // h==1 或 w==1 时是 0/0 = nan —— **Python 现在就是 nan**，
                    // 这里保持同样的行为（IEEE 除法给出一样的 nan），不"修"它。
                    const double dx = (cx != 0.0) ? ((double)x - cx) / cx : std::nan("");
                    const double r2 = dy * dy + dx * dx;
                    center_w_[(size_t)y * cols + x] = (float)std::exp(-r2 / denom);
                }
            }
        }
    } else if (p.mode == AAA_METER_EVALUATIVE) {
        const int64_t zy = p.zones_y, zx = p.zones_x;
        ys_.resize((size_t)zy + 1);
        xs_.resize((size_t)zx + 1);
        for (int64_t i = 0; i <= zy; ++i) ys_[(size_t)i] = linspace_int(i, 0.0, (double)rows, zy + 1);
        for (int64_t j = 0; j <= zx; ++j) xs_[(size_t)j] = linspace_int(j, 0.0, (double)cols, zx + 1);

        // 分区权重：Python 的 _zone_weights 里 h、w 两个参数**根本没用到**，
        // 所以它天然就是一张 zy*zx 的常量表，可以在这里一次算好。
        zone_w_.resize((size_t)zy * (size_t)zx);
        const double zcy = (zy - 1) / 2.0, zcx = (zx - 1) / 2.0;
        const double zden = 2.0 * p.zone_sigma * p.zone_sigma;
        for (int64_t i = 0; i < zy; ++i) {
            for (int64_t j = 0; j < zx; ++j) {
                const double dy = ((double)i - zcy) / (double)std::max<int64_t>(zy, 1);
                const double dx = ((double)j - zcx) / (double)std::max<int64_t>(zx, 1);
                zone_w_[(size_t)(i * zx + j)] = std::exp(-(dy * dy + dx * dx) / zden);
            }
        }
    }

    if (p.mode == AAA_METER_HIGHLIGHT) {
        if (p.precision == AAA_PREC_Q16) {
            // 定点分位数走直方图：bin 索引是纯移位，无除法、无比较器、
            // 单遍顺序访存。误差上界 = 一个 bin 宽（见下面的 hist_order_stat）。
            hist_bins_ = p.hist_bins ? p.hist_bins : 1024;
            hist_bits_ = 0;
            for (int b = 1; b < hist_bins_; b <<= 1) ++hist_bits_;
            hist_.assign((size_t)hist_bins_, 0);
        } else {
            sort_buf_.resize((size_t)rows * (size_t)cols);
        }
    }
    ready_ = true;
    return AAA_OK;
}

double hist_order_stat(const int32_t* h, int bins, int64_t k) {
    int64_t acc = 0;
    for (int b = 0; b < bins; ++b) {
        if (acc + (int64_t)h[b] > k) {
            const int64_t within = k - acc;
            const double frac = h[b] > 0 ? (double)within / (double)h[b] : 0.0;
            return (double)b + frac;
        }
        acc += (int64_t)h[b];
    }
    return (double)(bins - 1);
}

// -----------------------------------------------------------------------------
// 均值类模式（average / center / spot）—— 顺序访存，double 累加
// -----------------------------------------------------------------------------
int32_t AEMetering::meter_mean_like(const float* luma, int32_t sr, int32_t sc,
                                    aaa_ae_result& out) const {
    if (p_.mode == AAA_METER_CENTER) {
        // Σ(luma·w) / Σw，权重图已缓存
        double acc = 0.0, wsum = 0.0;
        for (int32_t y = 0; y < rows_; ++y) {
            const float* row = luma + (int64_t)y * sr;
            const float* wrow = center_w_.data() + (size_t)y * cols_;
            for (int32_t x = 0; x < cols_; ++x) {
                const double w = (double)wrow[x];
                acc += (double)row[(int64_t)x * sc] * w;
                wsum += w;
            }
        }
        out.metric = acc / wsum;
        out.aux = 0.0;
        out.n_used = (int64_t)rows_ * cols_;
        return AAA_OK;
    }
    if (p_.mode == AAA_METER_SPOT) {
        // 窗口边界照抄 Python：sh = max(1, int(h*ratio))，起点是 (h-sh)//2
        const int32_t sh = std::max(1, (int32_t)((double)rows_ * p_.spot_ratio));
        const int32_t sw = std::max(1, (int32_t)((double)cols_ * p_.spot_ratio));
        const int32_t y0 = (rows_ - sh) / 2, x0 = (cols_ - sw) / 2;
        double acc = 0.0;
        for (int32_t y = y0; y < y0 + sh; ++y) {
            const float* row = luma + (int64_t)y * sr;
            for (int32_t x = x0; x < x0 + sw; ++x) acc += (double)row[(int64_t)x * sc];
        }
        out.metric = acc / ((double)sh * (double)sw);
        out.aux = 0.0;
        out.n_used = (int64_t)sh * sw;
        return AAA_OK;
    }
    // average：全画面
    double acc = 0.0;
    for (int32_t y = 0; y < rows_; ++y) {
        const float* row = luma + (int64_t)y * sr;
        for (int32_t x = 0; x < cols_; ++x) acc += (double)row[(int64_t)x * sc];
    }
    out.metric = acc / ((double)rows_ * (double)cols_);
    out.aux = 0.0;
    out.n_used = (int64_t)rows_ * cols_;
    return AAA_OK;
}

// -----------------------------------------------------------------------------
// 主分发
// -----------------------------------------------------------------------------
int32_t AEMetering::meter_f32(const float* luma, int32_t rows, int32_t cols,
                              int32_t sr, int32_t sc, aaa_ae_result& out) const {
    if (!ready_ || rows != rows_ || cols != cols_) return AAA_ERR_SHAPE;
    if (!luma) return AAA_ERR_NULL;
    out = aaa_ae_result{};

    if (p_.mode != AAA_METER_EVALUATIVE && p_.mode != AAA_METER_HIGHLIGHT) {
        return meter_mean_like(luma, sr, sc, out);
    }

    if (p_.mode == AAA_METER_HIGHLIGHT) {
        // 拷一份再选第 99 分位。Python 的 np.percentile 内部也会 flatten 拷贝
        // （introselect partition，本来就是 O(n)），所以这里没有复杂度优势，
        // 真正的差别在访存：numpy 是数据相关的随机交换，这里是顺序读写。
        const int64_t n = (int64_t)rows * cols;
        for (int32_t y = 0; y < rows; ++y) {
            const float* row = luma + (int64_t)y * sr;
            float* dst = sort_buf_.data() + (int64_t)y * cols;
            for (int32_t x = 0; x < cols; ++x) dst[x] = row[(int64_t)x * sc];
        }
        // 只需要 virtual_index 附近的两个元素，用 nth_element 而非全排序
        const double q = 0.99;
        const double vi = (double)n * q + (1.0 + q * (1.0 - 1.0 - 1.0)) - 1.0;
        double metric;
        if (vi <= 0.0) metric = (double)sort_buf_[0];
        else if (vi >= (double)(n - 1)) metric = (double)sort_buf_[n - 1];
        else {
            const int64_t pi = (int64_t)std::floor(vi);
            std::nth_element(sort_buf_.begin(), sort_buf_.begin() + pi, sort_buf_.end());
            const float a = sort_buf_[(size_t)pi];
            const float b = *std::min_element(sort_buf_.begin() + pi + 1, sort_buf_.end());
            const double g = vi - std::floor(vi);
            const float diff = b - a;
            metric = (g < 0.5) ? ((double)a + (double)diff * g)
                               : ((double)b - (double)diff * (1.0 - g));
        }
        out.metric = metric;
        out.aux = 0.0;
        out.n_used = n;
        out.hist_peak_bin = -1;
        return AAA_OK;
    }

    // evaluative：分区统计 + 高光保护
    const int64_t zy = p_.zones_y, zx = p_.zones_x;
    double zsum = 0.0, wsum = 0.0, clip_zone = 0.0;
    for (int64_t i = 0; i < zy; ++i) {
        const int64_t y0 = ys_[(size_t)i], y1 = ys_[(size_t)i + 1];
        if (y1 <= y0) continue;
        for (int64_t j = 0; j < zx; ++j) {
            const int64_t x0 = xs_[(size_t)j], x1 = xs_[(size_t)j + 1];
            if (x1 <= x0) continue;
            double s = 0.0;
            int64_t cnt = 0, clip_cnt = 0;
            for (int64_t y = y0; y < y1; ++y) {
                const float* row = luma + (int64_t)y * sr;
                for (int64_t x = x0; x < x1; ++x) {
                    const float v = row[(int64_t)x * sc];
                    s += (double)v;
                    if ((double)v > p_.clip_level) ++clip_cnt;
                    ++cnt;
                }
            }
            if (cnt == 0) continue;
            const double zmean = s / (double)cnt;
            const double cfrac = (double)clip_cnt / (double)cnt;
            const double w = zone_w_[(size_t)(i * zx + j)];
            zsum += w * (zmean + p_.highlight_weight * cfrac);
            wsum += w;
            clip_zone += w * cfrac;
        }
    }
    out.metric = zsum / std::max(wsum, 1e-6);
    out.aux = clip_zone / std::max(wsum, 1e-6);
    out.n_used = (int64_t)rows * cols;
    return AAA_OK;
}

// -----------------------------------------------------------------------------
// 定点通路：uint16 Q0.16 输入 + int64 累加 + 直方图分位数
//
// 为什么 int64 就够：4K 全图 Σ 也只有 3840*2160*65535 ≈ 5.4e11，evaluative 的
// 加权和约 1.9e16，都远小于 2^63 —— **不需要分段移位**（那是 32 位 MCU 的妥协，
// 代价是每行 0.5 LSB 的系统性偏差）。
// -----------------------------------------------------------------------------
int32_t AEMetering::meter_u16(const uint16_t* luma, int32_t rows, int32_t cols,
                              int32_t sr, int32_t sc, aaa_ae_result& out) const {
    if (!ready_ || rows != rows_ || cols != cols_) return AAA_ERR_SHAPE;
    if (!luma) return AAA_ERR_NULL;
    out = aaa_ae_result{};
    const int64_t n = (int64_t)rows * cols;

    if (p_.mode == AAA_METER_HIGHLIGHT) {
        // 直方图的**有效位数** = min(输入位宽, bin 位数)。
        // 输入 8 位配 1024 bin 时，真正起作用的是 8 位，除数必须用 2^8 而不是
        // 2^10 —— 用错的话结果会整整差一倍（实测 8 位下误差 0.507，就是这么来的）。
        const int shift = (q_bits_ > hist_bits_) ? (q_bits_ - hist_bits_) : 0;
        const int eff_bits = (q_bits_ < hist_bits_) ? q_bits_ : hist_bits_;
        std::fill(hist_.begin(), hist_.end(), 0);
        for (int32_t y = 0; y < rows; ++y) {
            const uint16_t* row = luma + (int64_t)y * sr;
            for (int32_t x = 0; x < cols; ++x) {
                const uint16_t v = row[(int64_t)x * sc];
                const int b = (shift > 0) ? (int)(v >> shift) : (int)v;
                hist_[(size_t)(b < hist_bins_ ? b : hist_bins_ - 1)] += 1;
            }
        }
        // 与 numpy 一致的虚索引定义（照抄，别简化成 q*(n-1)）
        const double q = 0.99;
        const double vi = (double)n * q + (1.0 + q * (1.0 - 1.0 - 1.0)) - 1.0;
        double scale_val;
        if (vi <= 0.0) scale_val = hist_order_stat(hist_.data(), hist_bins_, 0);
        else if (vi >= (double)(n - 1)) scale_val = hist_order_stat(hist_.data(), hist_bins_, n - 1);
        else {
            const int64_t lo = (int64_t)std::floor(vi);
            const double g = vi - (double)lo;
            const double a = hist_order_stat(hist_.data(), hist_bins_, lo);
            const double b = hist_order_stat(hist_.data(), hist_bins_, lo + 1);
            scale_val = a + (b - a) * g;
        }
        out.hist_peak_bin = (int64_t)scale_val;
        out.metric = scale_val / (double)(1 << eff_bits);
        out.n_used = n;
        return AAA_OK;
    }

    if (p_.mode == AAA_METER_CENTER) {
        // 可分离 Q15 权重：w = wy[y]·wx[x]，乘积是 Q30，用 int64 累加
        int64_t acc = 0, wsum = 0;
        for (int32_t y = 0; y < rows; ++y) {
            const uint16_t* row = luma + (int64_t)y * sr;
            const int64_t wy = center_wy_q15_[(size_t)y];
            for (int32_t x = 0; x < cols; ++x) {
                const int64_t w = wy * center_wx_q15_[(size_t)x];       // Q30
                acc += (int64_t)row[(int64_t)x * sc] * w;
                wsum += w;
            }
        }
        out.metric = (wsum > 0) ? ((double)acc / (double)wsum) * inv_q_scale_ : 0.0;
        out.n_used = n;
        return AAA_OK;
    }

    if (p_.mode == AAA_METER_SPOT) {
        const int32_t sh = std::max(1, (int32_t)((double)rows * p_.spot_ratio));
        const int32_t sw = std::max(1, (int32_t)((double)cols * p_.spot_ratio));
        const int32_t y0 = (rows - sh) / 2, x0 = (cols - sw) / 2;
        int64_t acc = 0;
        for (int32_t y = y0; y < y0 + sh; ++y) {
            const uint16_t* row = luma + (int64_t)y * sr;
            for (int32_t x = x0; x < x0 + sw; ++x) acc += (int64_t)row[(int64_t)x * sc];
        }
        out.metric = ((double)acc / ((double)sh * (double)sw)) * inv_q_scale_;
        out.n_used = (int64_t)sh * sw;
        return AAA_OK;
    }

    if (p_.mode == AAA_METER_EVALUATIVE) {
        const int64_t zy = p_.zones_y, zx = p_.zones_x;
        const uint16_t clip_q = (uint16_t)clip_q_;
        double zsum = 0.0, wsum = 0.0, clip_zone = 0.0;
        for (int64_t i = 0; i < zy; ++i) {
            const int64_t y0 = ys_[(size_t)i], y1 = ys_[(size_t)i + 1];
            if (y1 <= y0) continue;
            for (int64_t j = 0; j < zx; ++j) {
                const int64_t x0 = xs_[(size_t)j], x1 = xs_[(size_t)j + 1];
                if (x1 <= x0) continue;
                int64_t s = 0, cnt = 0, clip_cnt = 0;
                for (int64_t y = y0; y < y1; ++y) {
                    const uint16_t* row = luma + (int64_t)y * sr;
                    for (int64_t x = x0; x < x1; ++x) {
                        const uint16_t v = row[(int64_t)x * sc];
                        s += (int64_t)v;
                        if (v > clip_q) ++clip_cnt;
                        ++cnt;
                    }
                }
                if (cnt == 0) continue;
                const double zmean = ((double)s * inv_q_scale_) / (double)cnt;
                const double cfrac = (double)clip_cnt / (double)cnt;
                const double w = zone_w_[(size_t)(i * zx + j)];
                zsum += w * (zmean + p_.highlight_weight * cfrac);
                wsum += w;
                clip_zone += w * cfrac;
            }
        }
        out.metric = zsum / std::max(wsum, 1e-6);
        out.aux = clip_zone / std::max(wsum, 1e-6);
        out.n_used = n;
        return AAA_OK;
    }

    // average
    int64_t acc = 0;
    for (int32_t y = 0; y < rows; ++y) {
        const uint16_t* row = luma + (int64_t)y * sr;
        for (int32_t x = 0; x < cols; ++x) acc += (int64_t)row[(int64_t)x * sc];
    }
    out.metric = ((double)acc / (double)n) * inv_q_scale_;
    out.n_used = n;
    return AAA_OK;
}

}  // namespace aaa
