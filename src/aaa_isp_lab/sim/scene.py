# -*- coding: utf-8 -*-
"""合成场景：提供**已知真值**（光源色温、参考反射率）的测试场景。

为什么用合成场景而不是直接拍照片：
  拍照片无法知道"真实光源色温是多少"，也就无法量化 AWB 的照度误差；
  无法知道"理想合焦位置"，也就无法量化 AF 的定位精度。
  仿真里两者都是真值可控的，所以每个 3A 指标都能给出数字。

场景输出统一为 float32 线性反射率 (h, w, 3)，值域约 [0, 1.2]，
1.0 ≈ 理想白（含镜面/高光 >1 的部分会被传感器截断，用于暴露高光问题）。
"""
from dataclasses import dataclass
import numpy as np

from ..color_science import srgb_to_linear


@dataclass
class Scene:
    name: str
    reflectance: np.ndarray          # (h,w,3) 线性反射率
    # 各通道近中性灰区域的掩码（用于 AWB 的"应还原为中性"的评价）
    neutral_mask: np.ndarray = None
    note: str = ""

    def __post_init__(self):
        if self.neutral_mask is None:
            self.neutral_mask = np.zeros(self.reflectance.shape[:2], dtype=bool)


# -----------------------------------------------------------------------------
# 1) 色卡（AWB / CCM 标定的核心场景）
# -----------------------------------------------------------------------------
# 经典的 24 色卡 sRGB(8bit) 参考值，最后一行 6 个是中性灰阶
# （灰阶序列是 AWB 评价的关键：它们必须被还原成 R=G=B）
CC24_SRGB = np.array([
    [115, 82, 68], [194, 150, 130], [98, 122, 157], [87, 108, 67],
    [133, 128, 177], [103, 189, 170], [214, 126, 44], [80, 91, 166],
    [193, 90, 99], [94, 60, 108], [157, 188, 64], [224, 163, 46],
    [56, 61, 150], [70, 148, 73], [175, 54, 60], [231, 199, 31],
    [187, 86, 149], [8, 133, 161], [243, 243, 242], [200, 200, 200],
    [160, 160, 160], [122, 122, 121], [85, 85, 85], [52, 52, 52],
], dtype=np.float64)


def color_chart(w: int, h: int, patch_scale: float = 0.85,
                bg: float = 0.15) -> Scene:
    """4x6 标准色卡。返回场景 + 每个 patch 的 mask 列表（用于 CCM 标定）。"""
    refl = np.full((h, w, 3), bg, dtype=np.float32)
    chart = CC24_SRGB.reshape(4, 6, 3) / 255.0
    chart_lin = srgb_to_linear(chart) * patch_scale

    margin_x, margin_y = 0.04, 0.05
    pw = (w * (1 - 2 * margin_x)) / 6.0
    ph = (h * (1 - 2 * margin_y)) / 4.0
    inset = 0.12   # patch 间隙，避免边界混色

    masks = []
    for r in range(4):
        for c in range(6):
            x0 = int(round(w * margin_x + c * pw + pw * inset / 2))
            x1 = int(round(w * margin_x + (c + 1) * pw - pw * inset / 2))
            y0 = int(round(h * margin_y + r * ph + ph * inset / 2))
            y1 = int(round(h * margin_y + (r + 1) * ph - ph * inset / 2))
            refl[y0:y1, x0:x1] = chart_lin[r, c]

    for r in range(4):
        for c in range(6):
            m = np.zeros((h, w), dtype=bool)
            x0 = int(round(w * margin_x + c * pw + pw * inset / 2))
            x1 = int(round(w * margin_x + (c + 1) * pw - pw * inset / 2))
            y0 = int(round(h * margin_y + r * ph + ph * inset / 2))
            y1 = int(round(h * margin_y + (r + 1) * ph - ph * inset / 2))
            # 再内缩 15% 避开插值边缘
            dy, dx = max(1, int((y1 - y0) * 0.15)), max(1, int((x1 - x0) * 0.15))
            m[y0 + dy:y1 - dy, x0 + dx:x1 - dx] = True
            masks.append(m)

    # 第 4 行（索引 3）是中性灰阶
    neutral = np.zeros((h, w), dtype=bool)
    for idx in range(18, 24):
        neutral |= masks[idx]

    s = Scene(name="color_chart", reflectance=refl.astype(np.float32),
              neutral_mask=neutral,
              note="4x6 标准色卡，含 6 级中性灰阶；用于 AWB 照度误差与 CCM 标定")
    s.patch_masks = masks          # 动态挂载，供 CCM 标定使用
    return s


# -----------------------------------------------------------------------------
# 2) 对焦标板（AF 的核心场景）
# -----------------------------------------------------------------------------
def focus_target(w: int, h: int) -> Scene:
    """西门子星 + 倾斜边缘 + 高频纹理。

    离焦时高频能量掉得最快，因此这个场景能让对焦评价函数
    呈现清晰的单峰形状（AF 搜索的前提）。
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    lum = np.full((h, w), 0.55, dtype=np.float32)

    # --- 西门子星（左上）：角向正弦，含全频段 ---
    r = np.sqrt((yy - cy * 0.62) ** 2 + (xx - cx * 0.62) ** 2)
    theta = np.arctan2(yy - cy * 0.62, xx - cx * 0.62)
    star = 0.5 + 0.5 * np.sign(np.sin(theta * 24.0))
    star[r > min(w, h) * 0.22] = 0.55
    lum = np.where(r <= min(w, h) * 0.22, star, lum)

    # --- 高频棋盘（右上）：固定空间频率，评价函数灵敏度的直接来源 ---
    board = (((xx // 3).astype(int) + (yy // 3).astype(int)) % 2).astype(np.float32)
    m = (xx > w * 0.55) & (yy < h * 0.42)
    lum = np.where(m, 0.25 + 0.55 * board, lum)

    # --- 倾斜边缘（左下）：10° 斜边，用于锐度/MTF 类分析 ---
    edge_x = w * 0.10 + (yy - h * 0.72) * np.tan(np.radians(10.0))
    m = (yy > h * 0.52) & (xx < w * 0.45)
    lum = np.where(m, np.where(xx > edge_x, 0.85, 0.12), lum)

    # --- 低对比纹理（右下）：模拟低照度下的弱纹理 ---
    rng = np.random.default_rng(7)
    noise_tex = cv2_box_blur(rng.random((h, w)).astype(np.float32), 2)
    m = (yy > h * 0.52) & (xx > w * 0.55)
    lum = np.where(m, 0.45 + 0.10 * noise_tex, lum)

    refl = np.repeat(lum[..., None], 3, axis=2)
    return Scene(name="focus_target", reflectance=refl, note="对焦标板：西门子星/棋盘/斜边/弱纹理")


def cv2_box_blur(img: np.ndarray, ksize: int) -> np.ndarray:
    import cv2
    return cv2.blur(img, (2 * ksize + 1, 2 * ksize + 1))


# -----------------------------------------------------------------------------
# 3) 自然场景（AE 的核心场景：逆光）
# -----------------------------------------------------------------------------
def natural_scene(w: int, h: int, backlit: bool = True) -> Scene:
    """合成"自然"场景：大面积天空/逆光 + 画面正中的人物主体 + 暗部地面。

    构图是刻意选的（经典的"逆光人像"）：
      - 天空/背景约占画面 55%，逆光时非常亮（大面积过曝）
      - 人物主体位于**画面正中央**，只占约 8% 面积
    这个比例关系才是测光方式差异的来源：
      平均/高光优先测光被大面积亮背景主导 -> 主体被压暗
      中心/点测光锁定主体 -> 主体保住
    如果主体占了画面 30%，无论哪种测光都会"顺便"保住它，
    测光方式的差异就测不出来了 —— 这是设计测试场景时最容易犯的错。
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xn = xx / w
    yn = yy / h

    # --- 天空：从上到下渐变，逆光时整体过曝 ---
    top, bottom = (2.30, 1.35) if backlit else (0.80, 0.58)
    sky_lum = top + (bottom - top) * yn
    sky = np.stack([sky_lum * 0.80, sky_lum * 0.92, sky_lum * 1.06], axis=-1)

    # --- 远山剪影 ---
    ridge = 0.50 + 0.05 * np.sin(xn * 9.0) + 0.02 * np.sin(xn * 21.0)
    ground = np.zeros((h, w, 3), dtype=np.float32)
    ground[..., 0] = 0.14 + 0.04 * np.sin(xn * 13.0)
    ground[..., 1] = 0.17 + 0.06 * np.sin(xn * 11.0 + 1.0)
    ground[..., 2] = 0.12
    img = np.where((yn > ridge)[..., None], ground, sky)

    # --- 主体：画面正中的逆光人物（浅色衣服，本身是中高反射率）---
    person = (np.abs(xn - 0.5) < 0.085) & (yn > 0.30) & (yn < 0.80)
    img[person] = np.array([0.52, 0.51, 0.50], dtype=np.float32)
    head = ((xn - 0.5) ** 2 + (yn - 0.26) ** 2) < (0.055 ** 2)
    img[head] = np.array([0.42, 0.34, 0.29], dtype=np.float32)
    subject = person | head

    # --- 彩色物体：检验 AWB 是否把它们误当白点 ---
    m = (yn > 0.86) & (xn > 0.74) & (xn < 0.86)
    img[m] = np.array([0.70, 0.14, 0.11], dtype=np.float32)
    m = (yn > 0.86) & (xn > 0.87) & (xn < 0.97)
    img[m] = np.array([0.82, 0.66, 0.10], dtype=np.float32)

    # --- 中性灰参考块：AWB 可以依赖的白点 ---
    ref = (yn > 0.88) & (yn < 0.97) & (xn > 0.05) & (xn < 0.18)
    img[ref] = np.array([0.78, 0.78, 0.78], dtype=np.float32)

    # --- 太阳：小面积高亮，逆光时必须过曝 ---
    if backlit:
        sun_c = (0.78 * w, 0.16 * h)
        r = np.sqrt((xx - sun_c[0]) ** 2 + (yy - sun_c[1]) ** 2)
        img += (1.2 * np.exp(-(r ** 2) / (2 * (0.14 * w) ** 2))
                + 3.0 * (r < 0.03 * w))[..., None]

    refl = np.clip(img, 0.0, None).astype(np.float32)
    s = Scene(name="natural_backlit" if backlit else "natural",
              reflectance=refl, neutral_mask=(subject | ref),
              note="逆光自然场景：大面积亮背景 + 正中主体 + 暗部地面 + 中性参考块")
    s.subject_mask = subject      # 动态挂载：AE 评价"主体有没有被保住"用
    return s


def dim(scene: Scene, factor: float, name: str = None) -> Scene:
    """把场景整体调暗（模拟低照度），用于观察 ET/增益取舍与噪声劣化。"""
    return Scene(name=name or f"{scene.name}_dim{factor:g}",
                 reflectance=(scene.reflectance * factor).astype(np.float32),
                 neutral_mask=scene.neutral_mask,
                 note=scene.note + f"（整体亮度 ×{factor:g}）")


# -----------------------------------------------------------------------------
# 4) 单一色物体场景（AWB 的失效场景）
# -----------------------------------------------------------------------------
def muted_scene(w: int, h: int, dominant_rgb=(0.75, 0.12, 0.10)) -> Scene:
    """大面积单色场景：灰世界假设被打破，用于暴露 AWB 算法失效。"""
    rng = np.random.default_rng(11)
    refl = np.zeros((h, w, 3), dtype=np.float32)
    base = np.array(dominant_rgb, dtype=np.float32)
    tex = 0.9 + 0.2 * rng.random((h, w, 1)).astype(np.float32)
    refl[:] = base[None, None, :] * tex
    # 角落留一小块中性参考，勉强给算法一根救命稻草
    refl[: int(h * 0.12), int(w * 0.88):, :] = 0.8
    neutral = np.zeros((h, w), dtype=bool)
    neutral[: int(h * 0.12), int(w * 0.88):] = True
    return Scene(name="muted_red", reflectance=refl, neutral_mask=neutral,
                 note="大面积单色场景：灰世界假设失效，AWB 会明显偏色")


def uniform_scene(w: int, h: int, level: float = 0.55) -> Scene:
    """匀光板：行轮廓完全平坦。

    测试行间带纹（交流光源闪烁）必须用这种场景。用风景/色卡测的话，
    场景自身的行间结构（地平线、色卡行）会被误判成带纹，
    测量结果直接失去意义 —— 这也是产线上测带纹要用白墙/匀光板的原因。
    """
    refl = np.full((h, w, 3), level, dtype=np.float32)
    return Scene(name="uniform", reflectance=refl,
                 note="匀光板：行/列轮廓平坦，用于带纹与均匀性测试")


def build_all(w: int = 480, h: int = 360) -> dict:
    return {
        "color_chart": color_chart(w, h),
        "focus_target": focus_target(w, h),
        "natural_backlit": natural_scene(w, h, backlit=True),
        "natural_normal": natural_scene(w, h, backlit=False),
        "muted_red": muted_scene(w, h),
        "uniform": uniform_scene(w, h),
    }
