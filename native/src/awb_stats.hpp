// =============================================================================
// awb_stats.hpp —— AWB 统计通路的内部实现（不对外）
//
// 相对 Python 现状（awb.py）省掉的重复工作，全部实测可量化：
//   5 次全帧布尔掩码 gather  -> 0（不落临时数组）
//   4 次 Sobel               -> 1（gray_edge 与融合块各算 2 次完全相同的）
//   3 次全帧 luma            -> 1
//   2 次全帧 _saturation     -> 1
//   4 次全帧 np.sqrt         -> 0（见下）
//
// 关于 sqrt：Python 里 mag = sqrt(gx²+gy²) 只用于 `mag > thresh` 这个比较，
// 数值本身从不用到。所以用 gx²+gy² > thresh² 完全等价，开方一次都不需要。
// 定点下更彻底：gx²+gy² 在 int64 里是**精确无舍入**的，连浮点比较器都不用。
// 这是只有真的动手做定点化才会发现的优化。
// =============================================================================
#ifndef AAA_AWB_STATS_HPP
#define AAA_AWB_STATS_HPP

#include <cstdint>
#include <vector>

#include "aaa_stats.h"

namespace aaa {

class AWBStats {
public:
    AWBStats() = default;

    int32_t configure(const aaa_awb_params& p, int32_t rows, int32_t cols);

    int32_t compute_f32(const float* rgb, int32_t rows, int32_t cols,
                        int32_t stride_row, int32_t stride_col, int32_t stride_chan,
                        aaa_awb_stats& out);

    int32_t compute_u16(const uint16_t* rgb, int32_t rows, int32_t cols,
                        int32_t stride_row, int32_t stride_col, int32_t stride_chan,
                        aaa_awb_stats& out);

private:
    aaa_awb_params p_{};
    int32_t rows_ = 0, cols_ = 0;
    bool ready_ = false;

    // 复用的缓冲：luma 平面（第二遍 Sobel 要用）与各通道的分位数样本
    std::vector<float> gray_;
    std::vector<uint16_t> gray_q_;   // 定点通路的 Q0.16 luma 平面（第二遍 Sobel 用）
    std::vector<float> ch_[3];
    std::vector<float> scratch_;
    std::vector<uint8_t> keep_;
    // 定点通路：每通道一条直方图（white_patch 的分位数）与 SoG 用的大整数累加
    mutable std::vector<int32_t> hist_[3];
    int hist_bits_ = 10;
    int32_t hist_bins_ = 1024;
};

// 3×3 Sobel，border = REFLECT_101（OpenCV 的 BORDER_DEFAULT）。
//
// 等效核是**实测反解**出来的，不是猜的（见 native/README.md 的步骤 1）：
//   dx: [-1 0 1; -2 0 2; -1 0 1]    dy: [-1 -2 -1; 0 0 0; 1 2 1]
//   out[y,x] = Σ K[a,b] · img[y+a-1, x+b-1]     （相关，不是卷积）
// 用线性 ramp 验证过：内部恒为 ±8，且默认 border 的边界值 0.0 与
// REPLICATE/REFLECT 的 4.0 不同 —— 证实默认是 REFLECT_101。
int32_t reflect101(int32_t i, int32_t n);

}  // namespace aaa

#endif  // AAA_AWB_STATS_HPP
