# TransBl 项目说明

TransBl 是一个用于叶轮机械叶片几何转换的 Fortran 程序。它把三维叶片截面几何从笛卡尔坐标形式转换到 MISES 所需的准二维叶栅坐标形式，并生成 `blade.<case>`、`stream.<case>` 和 `ises.<case>` 等输入文件。

项目的核心用途可以概括为：

1. 读取 `geom.<case>` 中的两条叶片表面曲线。
2. 将输入点重采样到 MISES 可接受的点数。
3. 把 `(x,y,z)` 转换为柱坐标中的 `(r,theta,z)`。
4. 沿子午面计算 `m' = integral(dm/r)`。
5. 在 `(m',theta)` 平面中写出叶型几何和流道/边界条件文件。

## 目录结构

```text
.
├── src/                 # Fortran 源码
├── TransBl/             # Visual Studio/Intel Fortran 工程与算例文件
├── TransBl.sln          # Visual Studio 解决方案
└── TransBl.suo          # Visual Studio 用户状态文件
```

`src/` 是主要维护对象；`TransBl/Debug/` 中的 `.obj`、`.mod`、`.exe` 等文件是编译产物。

## 主要源码

| 文件 | 作用 |
| --- | --- |
| `src/main.f90` | 主程序，串联读取几何、重采样、坐标转换、尾缘切削和文件写出。 |
| `src/redistr.f90` | 按三维弧长和余弦分布重采样叶片表面点。 |
| `src/trans.f90` | 将平面 Cartesian 坐标 `(x,y)` 转为柱坐标 `(r,theta)`。 |
| `src/mpcal.f90` | 沿 `(z,r)` 子午曲线积分计算 `m'` 和子午弦长。 |
| `src/write_ises.f90` | 原始边界条件转换实现，读取 `opr.dat`。 |
| `src/write_ises2.f90` | 边界条件转换的开发/对比版本。 |
| `src/write_ises3.f90` | 较完整的边界条件转换实现，读取 `opr.<case>`。 |
| `src/circle.f90` | 根据三点拟合圆，用于可选的圆弧尾缘处理。 |
| `src/tangent.f90` | 计算叶片曲线与拟合圆之间的切点。 |
| `src/sutherland.f90` | 用 Sutherland 公式计算空气动力黏度。 |
| `src/udprog.for` | 旧式 Fortran 插值库，提供 `UD0321/UD0322/UD0327`。 |

## 输入文件

### `geom.<case>`

主程序启动后要求输入 `case` 名称，然后读取：

```text
geom.<case>
```

该文件包含：

1. 叶片数 `nblade`。
2. 输入点数上限 `nx`。
3. 两个叶片表面点块。

每个点块一般包含一段标题、坐标类型标识、点数，以及若干行：

```text
x y z
```

代码假设有两条曲线，分别对应 `j=1` 和 `j=2`。每条曲线上的点从前缘走向尾缘。

### `opr.<case>` 或 `opr.dat`

边界条件写出例程会读取运行工况：

```text
lscale rpm pt1 tt1 alpha1 p2
```

含义如下：

| 字段 | 含义 |
| --- | --- |
| `lscale` | 几何长度缩放系数。 |
| `rpm` | 转速，单位 rpm。 |
| `pt1` | 入口总压。 |
| `tt1` | 入口总温。 |
| `alpha1` | 入口绝对流角，单位 degree。 |
| `p2` | 出口静压。 |

当前仓库中 `write_ises3.f90` 是最完整版本，读取 `opr.<case>`；旧版本 `write_ises.f90` 和 `write_ises2.f90` 读取 `opr.dat`。最终使用哪个版本取决于工程文件实际编译进来的源码。

## 输出文件

输入 case 为 `r37` 时，程序会生成类似：

| 文件 | 内容 |
| --- | --- |
| `blade.r37` | MISES 叶型几何，坐标为 `(m',theta)`。 |
| `stream.r37` | MISES 流道/流线参考文件。 |
| `ises.r37` | MISES 运行参数和边界条件。 |
| `r37_check.plt` | 用于检查尾缘切削前后几何的 Tecplot 风格文件。 |

## 核心算法流程

### 1. 读取两条叶片表面

`main.f90` 从 `geom.<case>` 读取两条三维曲线。数组约定为：

```text
x(j,i), y(j,i), z(j,i)
```

其中 `j=1,2` 表示两侧表面，`i` 表示该表面上的点序号。

### 2. 重采样到 201 点

MISES 对叶片点数有限制，代码中设置：

```fortran
npmax = 201
```

`redistr` 会把每条表面重采样到 201 点。分布方式不是简单等距，而是基于三维弧长的余弦分布，因此前缘和尾缘附近点更密，中部点更疏。

### 3. 转换到柱坐标

`trans` 将 `(x,y)` 转为：

```text
r     = sqrt(x^2 + y^2)
theta = atan2(y,x)
```

`z` 保持为轴向坐标。

### 4. 计算 `m'`

`mpcal` 沿子午面曲线 `(z,r)` 积分：

```text
dm  = sqrt(dz^2 + dr^2)
m' += dm / r_avg
```

`m'` 是 MISES 叶栅坐标中的流向坐标。程序还会把两条表面末端的 `m'` 调整到同一个平均尾缘值，以减小数值积分和插值带来的闭合误差。

### 5. 尾缘切削

MISES 通常需要钝尾缘以施加 Kutta 条件。当前代码默认：

```fortran
ndel = 7
```

也就是从两条表面的尾缘各删除 7 个点。运行时会询问是否修改 `ndel(1), ndel(2)`。

源码中还保留了基于三点拟合圆和切线点的尾缘处理方案，但目前被注释掉，实际执行的是手动删除点数方案。

### 6. 写出 MISES 文件

程序写出：

1. `blade.<case>`：叶片表面坐标。
2. `stream.<case>`：流道入口、叶片前缘、叶片尾缘、出口位置。
3. `ises.<case>`：边界条件、雷诺数、转捩参数和 MISES 控制参数。

## 编译检查

可以用 gfortran 做语法检查：

```bash
gfortran -fsyntax-only src/*.f90 src/*.for
```

该命令可以通过，但会保留一些旧 Fortran 风格警告，例如：

- `pause` 是已删除特性。
- `udprog.for` 使用 fixed-form Fortran、assigned `goto` 和旧式 DO 终止方式。
- 部分 `write` 格式使用老式 `$` 控制不换行输出。

这些警告来自原始代码风格，不一定代表当前程序无法在 Intel Fortran/Visual Studio 工程中编译。

## 已知注意点

1. `write_ises.f90`、`write_ises2.f90`、`write_ises3.f90` 都定义了同名 `write_ises` 子程序，实际工程中只能编译其中一个同名实现，否则会出现重复符号。
2. `write_ises3.f90` 中入口密度迭代处原代码有一行：

   ```fortran
   err = abs(rho1-rho1)/rho1
   ```

   这会恒等于 0，疑似应比较更新前后的 `rho10` 和 `rho1`。因为这会影响数值逻辑，本文档只指出风险，没有替开发者直接更改。
3. 当前主程序会把每条输入表面都重采样到 201 点，即使原始点数少于 201。
4. 项目中的 `TransBl/Debug/` 是编译输出目录，不建议作为源码维护对象。

