# Planar MCA: 双圆弧型面生成与坐标逆向拟合

这是一个可运行的 Python 项目：用**吸力面两段圆弧、压力面两段圆弧**生成二维叶型，并从有序二维叶型坐标拟合回模型参数。前、尾缘由与主体切线相接的**椭圆型二次有理曲线**封闭。输出重构坐标、参数、误差、几何检查和论文风格的 PDF/PNG 图。

## 模型边界（请先读）

本项目给出的是 **planar-equivalent MCA（平面等效 MCA）**，不是 NASA TN D-5437 的锥面恒角度变化率生成器，也没有包装 OpenMCAD 源码。NASA 的原始 11 类输入量包含前/尾缘流面半径与锥面轴向距离；仅从二维归一化截面不能唯一反求这些量。此处的输出参数是本项目明确规定的四圆弧模型的最佳拟合参数，不能标作原设计参数。OpenMCAD 的[软件记录](https://hal.science/hal-03923093)和 [LAVA 开放模型](https://gitlab.lava.polymtl.ca/depots_publics/modeles/catalogue_aubes/-/tree/master/rotor67)是正向 MCA 参考；当前环境中 HAL 下载受到站点访问限制，因此没有声称已对其源码做逐行移植或数值复现。

Derksen 和 Rogalsky 的 [Bezier-PARSEC 论文](https://doi.org/10.1016/j.advengsoft.2010.05.002)说明用优化来提高叶型重构能力，也讨论了几何参数化在气动设计中的作用。本项目借鉴“参数化 + 优化 + 基线比较”的实验思想；使用的是几何坐标误差目标，**不**进行论文中的压力分布逆设计，也不借用其 BP 控制点公式。这里四圆弧的曲率在接头处允许阶跃，这是 MCA 的固有自由度；位置和切线连续，不能同时要求各圆弧接头都达到 G2 而仍保持任意不同曲率。

## 数学定义

每侧五个参数：`y_le, theta_le, k_front, k_rear, x_join`。`x_le=0.002`、`x_te=0.998` 为默认归一化主体截断位置，`theta` 以弧度计，`k` 为有符号曲率，单位为 `1/c`。主体 x 单调且切线角保持在 `(-π/2, π/2)`。从起点 `(x0,y0,theta0)` 沿半径 `1/|k|` 的圆弧到 x，有

```text
sin(theta(x)) = sin(theta0) + k (x-x0)
y(x) = y0 + [cos(theta0)-cos(theta(x))]/k
```

`k→0` 时使用直线极限 `y=y0+(x-x0)tan(theta0)`。后段从前段接头的 `(y,theta)` 精确续接，因此**位置和切线连续**，不靠惩罚项近似。前/尾缘端点切线的交点是有理二次曲线控制点 `P`；权重 `0<w<1` 时得到椭圆型非退化圆锥曲线弧：

```text
C(t) = [(1-t)^2 A + 2w t(1-t) P + t^2 B] /
       [(1-t)^2 + 2w t(1-t) + t^2],  0 ≤ t ≤ 1
```

控制点的切线交点构造保证边缘与主体 **G1** 相接。`w` 控制前后缘丰满度，优化器在 `(0.005,0.95)` 内搜索。极小权重可能表示模型借助很薄的边缘帽逼近参考坐标；查看 JSON 中的 `le_weight/te_weight`，不要将其解释为真实制造椭圆轴比。

逆向拟合先按 TE→LE→TE 排序和弦线元数据分侧，采用 PCHIP 在公共余弦 x 网格取样。两侧分别用有界、多起点非线性最小二乘拟合圆弧，随后对前尾缘权重作一维有界优化。主体曲线的目标为同一 x 上的 y 残差；最后报告**观测点到重构折线的正交最短距离**、逆向距离及对称 RMS，所有距离以弦长 c 无量纲化。拟合后检查正厚度、自交、封闭面积、边缘投影和圆弧接头切角。没有加样条补偿，所以较低误差真实反映四圆弧模型的能力。

## Yang 中弧线与厚度

`mca/extraction.py` 参考随项目提供的 `yang/.../src/blex2d.f90` 和 `normal_thick2.f90`：对吸力面点 `S` 在压力面寻找 `P`，满足 `(S-P)·(T_S+T_P)=0`；中弧点取 `(S+P)/2`，半厚度取 `|S-P|/2`。这里对 PCHIP 插值后的两条 x 单调表面用 Brent 求根，找不到可靠配对时输出 NaN，不假造垂直厚度。该提取用于**诊断与初值理解**；几何正向模型是四个显式圆弧。对于强回弯、前后缘非 x 单值区段，此诊断不代替精确 CAD 法向定义。

## 安装与使用

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
planar-mca fit path/to/section.dat --output results/my_section
planar-mca generate results/my_section/section.json --output regenerated.dat
```

输入是有顺序的二维 `x y` 行，惯例为 TE→LE→TE；可以有任意首行标题及 `#` 注释。对开放尾缘、旋转/缩放坐标，推荐在文件头写明设计弦的两端：

```text
# leading_edge 0.0 0.0
# trailing_edge 1.0 0.0
```

元数据缺失时会由首末点中点与中间最上游点推断；对非常规点序、严重偏置采样或三维截面应先人工核对。输出 `.dat` 是归一化封闭轮廓，`.csv` 同时给出原单位坐标，`.json` 记录参数、配准、误差、几何检查及求解状态；PDF/PNG 为轮廓与 Yang 法向半厚度诊断。`geometry.valid=false` 表示结果不可直接用于下游 CFD/CAD。

## BP3434 基线实验

下列复现实验使用用户提供的 BP3434 项目里五个 `blade_xyn.test_section_*`，并读取其**已经生成的** BP3333E `.dat`。比较在同一基线坐标、同一“点到折线最短距离”定义下重新计算，不把 BP 工程自带的 `optimized_rms` 直接混用：

```bash
python scripts/benchmark_bp3434.py \
  --bp-root /Users/jiaxinkai/Desktop/p/BP3434 \
  --output results/bp3434_baseline
```

| 截面 | 本项目 RMS/c | BP3333E RMS/c | 本项目几何检查 |
|---|---:|---:|---|
| 01 | 0.001267 | 0.001339 | 通过 |
| 02 | 0.000766 | 0.000750 | 通过 |
| 03 | 0.000521 | 0.000626 | 通过 |
| 04 | 0.000553 | 0.000609 | 通过 |
| 05 | 0.000102 | 0.000846 | 通过 |

完整误差（MAE、最大误差、双向 RMS）和每截面重构图见 [`results/bp3434_baseline/comparison.csv`](results/bp3434_baseline/comparison.csv)。[`parameter_comparison.csv`](results/bp3434_baseline/parameter_comparison.csv) 还把原始坐标、MCA 和 BP3333E 的**共同可观测参数**（最大垂直厚度及位置、投影范围、面积）放在同一口径下。各模型的原生参数分别保存在自己的 JSON 中；MCA 的曲率/接头与 BP 的 Bezier/PARSEC 控制参数定义不同，不宜逐列数值比较。五个截面的本项目 RMS 平均约 `0.000642c`，BP3333E 约 `0.000835c`；这只说明**这五个给定截面**在本次同口径评估下的几何拟合情况，不说明某方法在一般叶型或气动性能上占优。第 02 截面 BP3333E 略低。边缘权重和接头位置有时处于边界，提示参数可能不可辨识；不能仅凭轮廓误差宣称找回原始设计参数。

## 模块

| 文件 | 职责 |
|---|---|
| `mca/model.py` | 解析四圆弧与前尾缘椭圆型帽 |
| `mca/io.py` | 读取、定向、归一化及物理坐标恢复 |
| `mca/extraction.py` | Yang 法向中弧与半厚度提取 |
| `mca/fitting.py` | 有界多起点几何反求 |
| `mca/validation.py` | 线段正交距离、自交、厚度、接头检查 |
| `mca/plotting.py` | 工程论文风格 PDF/600 dpi PNG |
| `mca/cli.py` | 正向/逆向命令行 |
| `scripts/benchmark_bp3434.py` | 五截面共同误差口径比较 |

## 参考资料

1. Crouse, Janetzke & Schwirian, *A Computer Program for Composing Compressor Blading from Simulated Circular-Arc Elements on Conical Surfaces*, NASA TN D-5437 (1969), [NASA PDF](https://ntrs.nasa.gov/api/citations/19690027504/downloads/19690027504.pdf)。明确 MCA 两侧各两圆弧，以及锥面恒转角率与平面截面的区别。
2. Kojtych & Batailly, *OpenMCAD, an open blade generator: from Multiple-Circular-Arc profiles to Computer-Aided Design model* (2022), [HAL 软件记录](https://hal.science/hal-03923093)；[LAVA 示例模型](https://gitlab.lava.polymtl.ca/depots_publics/modeles/catalogue_aubes/-/tree/master/rotor67)。本项目未依赖其源码。
3. Derksen & Rogalsky, *Bezier-PARSEC: An optimized aerofoil parameterization for design*, *Advances in Engineering Software* 41 (2010) 923–930, [DOI](https://doi.org/10.1016/j.advengsoft.2010.05.002)。用户提供的 PDF 已用于核对优化思想与 BP 模型边界。
4. 用户提供的 Yang `parablade2d-test` 中 `blex2d.f90`、`normal_thick2.f90`、`Geom2Para.f90`。这里只复现其法向配对原理，未移植整个 Fortran 工程。

## 已知限制与下一步

- 当前模型输入是**单站二维截面**；锥面半径、真实堆叠、三维投影仍需独立实现并用 NASA/OpenMCAD 的原始参数表闭环验证。
- 自交检查对采样后的闭合折线进行；生成函数用较密的 801+161 点检查，但它不是连续曲线的形式化拓扑证明。用于制造之前还需 CAD/网格独立检验。
- 主体截断位置固定为 `0.002c`、`0.998c`，可在 `mca/model.py` 调整，跨叶型比较时须保持一致。优化边缘帽权重及圆弧接头可能得到多组相近解；此版本没有参数置信区间。
- 中弧与厚度使用 Yang 式法向配对；MCA 正向参数是**表面圆弧参数**，不是简单地把一条双圆弧中弧法向偏置，因为那样通常不再得到圆弧型面。
