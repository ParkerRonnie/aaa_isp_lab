// =============================================================================
// ae_metering.hpp —— AE 测光统计的内部实现（不对外，不属于 ABI）
// =============================================================================
#ifndef AAA_AE_METERING_HPP
#define AAA_AE_METERING_HPP

#include <cstdint>
#include <vector>

#include "aaa_stats.h"

namespace aaa {

// numpy 的 np.percentile(a, q, method='linear') 的**逐位复刻**。
//
// 为什么能逐位复刻：numpy 的算法是确定的，只要照抄它的运算顺序即可。
//   virtual_index = n*q + (1 + q*(1-1-1)) - 1      <- 顺序不能简化成 q*(n-1)
//   previous = floor(virtual_index); gamma = virtual_index - previous
//   _lerp(a,b,t) = t < 0.5 ? a+(b-a)*t : b-(b-a)*(1-t)
// 其中 a、b 是**输入数组的元素**（float32），(b-a) 必须在 float32 里做，
// 只有乘 gamma 时才提升到 double —— 这正是 numpy 的行为。
//
// 输入必须是**已排序**的 float32 数组（调用方负责排序）。
double quantile_linear_sorted(const float* sorted, int64_t n, double q);

// 同上，但接受**未排序**的缓冲（内部只选择需要的两个元素，不全排序）。
// 注意：这个函数会就地打乱 buf 的内容 —— 调用方不要指望调用后数据还在。
double quantile_linear_inplace(float* buf, int64_t n, double q);

// 计算 np.linspace(start, stop, num).astype(int) 的第 i 个元素。
// 必须复刻 numpy 的构造方式（start + i*step，末元素精确为 stop），
// 否则某些形状下会因为 1 ulp 的差异而截断到不同整数，分区边界就错一格。
int64_t linspace_int(int64_t i, double start, double stop, int64_t num);

// 直方图的第 k 个顺序统计量（bin 内按计数位置线性插值）。
// 误差有**可证上界：一个 bin 宽** —— 估计值与真值必然落在同一个 bin 里。
// 定点分位数走这条路：bin 索引是纯移位，无除法、无浮点比较器、单遍顺序访存。
double hist_order_stat(const int32_t* h, int bins, int64_t k);

class AEMetering {
public:
    AEMetering() = default;

    // 预计算与形状/参数相关的量：中心权重图、分区边界、分区权重表。
    // 这是相对 Python 现状的第一个真改进 —— 那边每帧都重建整幅 h×w 权重图。
    int32_t configure(const aaa_ae_params& p, int32_t rows, int32_t cols);

    int32_t meter_f32(const float* luma, int32_t rows, int32_t cols,
                      int32_t stride_row, int32_t stride_col,
                      aaa_ae_result& out) const;

    // 定点通路（Q0.16 输入 + int64 累加 + 直方图分位数）见下一步实现。
    int32_t meter_u16(const uint16_t* luma, int32_t rows, int32_t cols,
                      int32_t stride_row, int32_t stride_col,
                      aaa_ae_result& out) const;

private:
    aaa_ae_params p_{};
    int32_t rows_ = 0, cols_ = 0;
    bool ready_ = false;

    // center：整幅权重图，只在这里构建一次（Python 那边是每帧重建）
    std::vector<float> center_w_;
    // center 的定点版本：exp(-(a+b)) = exp(-a)·exp(-b) 可分离，所以只需要
    // h+w 个 Q0.15 系数，**h×w 的权重图根本不落内存**（480x360 下省约 1.4MB 带宽）。
    std::vector<int32_t> center_wy_q15_, center_wx_q15_;
    // 定点分位数的直方图缓冲
    mutable std::vector<int32_t> hist_;
    int hist_bits_ = 10;
    int32_t hist_bins_ = 1024;
    // 定点尺度：由 bit_depth 决定。输入被量化到 [0, 2^bits-1]，
    // 所以 scale = 2^bits - 1（**不是**固定 65535 —— 位宽扫描要真的有变化）
    int32_t q_scale_ = 65535;
    int q_bits_ = 16;
    double inv_q_scale_ = 1.0 / 65535.0;
    int32_t clip_q_ = 62259;      // round(clip_level * q_scale_)
    // evaluative：分区边界与权重
    std::vector<int64_t> ys_, xs_;
    std::vector<double> zone_w_;
    // 分位数用的排序缓冲（复用，避免每次分配）
    mutable std::vector<float> sort_buf_;

    int32_t meter_mean_like(const float* luma, int32_t sr, int32_t sc,
                            aaa_ae_result& out) const;
};

}  // namespace aaa

#endif  // AAA_AE_METERING_HPP
