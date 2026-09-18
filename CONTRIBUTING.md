# 开发说明

## 环境

```bash
pip install -e ".[dev]"     # 可编辑安装 + 测试/检查工具
pytest                       # 跑单元测试（38 个）
aaa-isp-lab --fast           # 端到端跑一遍，约 45s
aaa-isp-lab                  # 全量，约 4 分钟（时域那几个实验是逐帧序列，占大头）
```

不安装也能跑（源码目录直接执行）：

```bash
python run_all.py --fast
# 或
PYTHONPATH=src python -m aaa_isp_lab --fast
```

## 目录约定

```
src/aaa_isp_lab/
├── config.py          所有可调参数集中在这里（模拟真实 ISP 的 tuning 表）
├── color_science.py   色温/色度、sRGB/XYZ/Lab、普朗克轨迹、Duv
├── sim/               物理建模：场景、光学、传感器、相机
├── isp/               ISP 处理链路：BLC→LSC→去马赛克→WB→CCM→色调映射
├── aaa/               3A 算法：AE / AWB / AF
├── eval/              评价指标（metrics）与报告呈现（report）
├── experiments.py     各组实验的定义（只产出数据，不画图）
└── cli.py             命令行入口与结果落盘
```

**分层原则**：`sim` 只负责"世界怎么成像"，`isp` 只负责"怎么把 RAW 变成图"，
`aaa` 只负责"怎么决策"，`eval` 只负责"怎么评价"。任何一层都不应该反向依赖上层。

新增内容时请遵守这条线，否则实验会变成一堆无法单独验证的脚本。

## 加一个实验

1. 在 `experiments.py` 里加 `exp_xxx(...)`，**返回纯数据**（数字 + ndarray），不要画图
2. 在 `eval/report.py` 里加对应的画图函数，并在 `build_report()` 的 `figs` 字典里登记
3. 在 `_sections()` 里加一节：标题 + 结论文字 + 图。**结论必须由实测数据支持**，
   数据对不上就改结论，不要改数据
4. 在 `cli.py` 的 `steps` 列表里注册
5. 如果这条结论值得长期守住，在 `tests/` 里加一个性质测试

## 加一个场景 / 一种算法

- 场景：在 `sim/scene.py` 里加构造函数，返回 `Scene`（反射率 + 中性区掩码），
  并在 `build_all()` 里登记。**场景的构图决定能不能测出差异**，
  例如测光方式对比要求亮背景占大头、主体只占小部分
- 算法：AE 在 `aaa/ae.py` 的 `metering_metric` 里加分支；
  AWB 在 `aaa/awb.py` 里加估计器并在 `fusion` 的置信度里给权重；
  AF 在 `aaa/af.py` 的 `focus_measure` 里加分支并登记到 `MEASURES`

## 测试约定

`tests/test_aaa.py` 里的测试分两类，都要保持：

1. **有标准答案的**（CIEDE2000 官方测试数据、色温往返、黑电平归一化）——
   这类测试保证实现对
2. **算法性质的**（收敛、单调、峰位正确、约束生效）—— 这类更重要：
   3A 是闭环控制，某个环节方向错了程序照样能跑出"看起来合理"的数字

新增算法时优先补第 2 类测试。

## 图表里的中文

matplotlib 需要 CJK 字体。Windows 用自带的 Microsoft YaHei；
Linux 需要 `fonts-noto-cjk` 或 SimHei，否则中文会渲染成方块（不报错，只是图不可读）。

## 发布产物

`docs/` 里是**已发布的报告快照**（报告 + 图表），随代码一起提交，README 直接引用它。
`out/` 是生成目录，已在 `.gitignore` 里忽略。

刷新快照：

```bash
python -m aaa_isp_lab --out docs      # 重新生成到 docs/
git add docs && git commit -m "docs: 更新实验报告快照"
```
