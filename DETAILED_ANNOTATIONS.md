# TransBl 详细注释说明

本文档是对 `src/` 源码的补充注释，按执行顺序解释每个重要变量、算法阶段和子程序职责。它不是用户手册，而是面向后续维护者的代码阅读笔记。

## 1. 主程序 `main.f90`

`main.f90` 是整个程序的调度入口。它完成四类任务：

1. 读入几何。
2. 建立 MISES 坐标。
3. 处理尾缘和叶型输出。
4. 根据工况生成流道与边界条件文件。

### 1.1 关键数组

```fortran
real,allocatable:: x(:,:),y(:,:),z(:,:),mp(:,:),theta(:,:),r(:,:)
```

这些数组都是二维数组：

```text
第 1 维：叶片表面编号，j=1 或 j=2
第 2 维：表面点编号，i=1..n(j)
```

各数组含义：

| 数组 | 含义 |
| --- | --- |
| `x,y,z` | 输入几何点的 Cartesian 坐标。 |
| `r` | 由 `(x,y)` 计算得到的半径。 |
| `theta` | 由 `(x,y)` 计算得到的周向角。 |
| `mp` | MISES 使用的 `m'` 坐标。 |

### 1.2 输入 case 名称

程序只读取一个屏幕输入：

```fortran
read(*,*) fname
```

如果输入 `r37`，则几何文件为：

```text
geom.r37
```

后续输出文件也使用同一个后缀。

### 1.3 点数上限和内存分配

```fortran
read(1,*) nblade, nx
nx = max(nx,201)
```

`nx` 来自几何文件，但程序强制至少为 201，因为后续会统一重采样到：

```fortran
npmax = 201
```

这保证 `x(2,nx)` 等数组有足够空间容纳重采样结果。

### 1.4 读取两条曲线

```fortran
do j=1,2
  read(1,*) str
  read(1,*) str
  read(1,*) n(j)
  do i=1,n(j)
    read(1,*) x(j,i), y(j,i), z(j,i)
  enddo
enddo
```

这里假设几何文件里恰好有两个表面块。每个块前两行被当作文本标签读入 `str`，第三行是点数，之后是点坐标。

### 1.5 重采样

主程序分别对两侧调用：

```fortran
call redistr(n(1),x(1,1:n(1)),y(1,1:n(1)),z(1,1:n(1)),npmax,...)
call redistr(n(2),x(2,1:n(2)),y(2,1:n(2)),z(2,1:n(2)),npmax,...)
```

注意 `redistr` 的第一个参数 `n` 是输入输出参数。调用后：

```fortran
n(1) = 201
n(2) = 201
```

### 1.6 坐标变换

对每一个点执行：

```fortran
call trans(x(j,i),y(j,i),r(j,i),theta(j,i))
```

得到：

```text
r     = sqrt(x*x + y*y)
theta = atan2(y,x)
```

这里没有修改 `z`，因为 `z` 已经是轴向坐标。

### 1.7 计算 `m'`

```fortran
call mpcal(n(j),1,z(j,1:n(j)),r(j,1:n(j)),mp(j,1:n(j)),chordm)
```

`n0=1` 表示以第一个点为 `m'=0` 的参考点。对每一段：

```text
dm = sqrt(dz^2 + dr^2)
mp(i) = mp(i-1) + sign * dm / ravg
```

其中：

```text
ravg = 0.5 * (r(i) + r(i-1))
```

`chordm` 是未除以半径的子午弦长：

```fortran
chordm = m(n)-m(1)
```

### 1.8 修正两侧尾缘 `m'` 不一致

两条表面理论上应在尾缘拥有相同的 `m'` 值，但离散积分会产生小差异。代码先求平均：

```fortran
mavg = 0.5*(mp(1,n(1))+mp(2,n(2)))
```

然后对每条曲线做线性拉伸：

```fortran
mp(j,i) = mp(j,i)+dm/(mp(j,n(j))-mp(j,1))*(mp(j,i)-mp(j,1))
```

该修正保持前缘 `mp(j,1)` 不动，把尾缘移动到共同平均值。

### 1.9 尾缘切削

默认：

```fortran
ndel = 7
```

写 `blade.<case>` 时使用：

```fortran
do i=n(j)-ndel(j),1,-1
...
do i=2,n(j)-ndel(j)
```

因此被删除的是每条表面靠近尾缘的最后 `ndel(j)` 个点。

### 1.10 入口/出口斜率

入口斜率 `sinl`：

1. 取从前缘向内部 10% 弦长的位置。
2. 分别在两条表面上插值得到 `theta`。
3. 取两侧平均值作为中弧线点。
4. 用前缘点到该中弧线点的斜率作为入口斜率。

出口斜率 `sout` 采用类似方式，从尾缘向内部取 10% 弦长位置。

最终写入：

```fortran
write(2,*) sinl, sout, chinl, chout, 2.*pi/real(nblade)
```

最后一项是单个叶道的周向周期角。

## 2. 重采样 `redistr.f90`

`redistr` 的目标是把任意点数的三维曲线转换为固定点数 `m`。

### 2.1 弧长归一化

```fortran
s(1) = 0.
do i=2,n
  s(i) = s(i-1) + sqrt(...)
end do
do i=1,n
  s(i) = s(i)/s(n)
end do
```

`s` 代表归一化三维弧长，范围为 `[0,1]`。

### 2.2 余弦分布

```fortran
dtheta = pi/real(m-1)
theta = pi
do i=1,m
  s1(i) = 0.5*(1.0+cos(theta))
  theta = theta - dtheta
end do
```

当 `theta` 从 `pi` 走到 `0` 时，`s1` 从 `0` 走到 `1`。余弦分布会在两端聚点，因此适合叶型前缘和尾缘。

### 2.3 插值

```fortran
call UD0322(s,x,n,s1,x1,m,1)
call UD0322(s,y,n,s1,y1,m,1)
call UD0322(s,z,n,s1,z1,m,1)
```

`NTYPE=1` 表示线性插值。

## 3. 子午坐标 `mpcal.f90`

`mpcal` 同时计算两个坐标：

| 名称 | 含义 |
| --- | --- |
| `m` | 子午面真实弧长。 |
| `mp` | 除以局部半径后的无量纲/展开坐标。 |

核心公式：

```text
dm = sqrt(dz^2 + dr^2)
dmp = sign * dm / ravg
```

`sign` 由整体流向和局部段方向决定。这样做的目的是在点列局部方向不完全单调时，仍尽量保持整体 `m'` 的方向一致。

## 4. 坐标转换 `trans.f90`

`trans` 很短，但它决定了周向角定义：

```fortran
r = sqrt(x*x+y*y)
theta = atan2(y,x)
```

源码里保留了旧注释：

```fortran
!theta = atan2(x,y)
```

这说明历史数据可能存在轴交换约定。如果发现输出叶型整体旋转或方向异常，应优先检查这里的角度定义是否和输入几何坐标系一致。

## 5. 边界条件 `write_ises*.f90`

三个文件都定义同名：

```fortran
subroutine write_ises(...)
```

所以同一目标程序中只能选择一个版本编译。

### 5.1 输入参数

```fortran
write_ises(fname,chordm,r1,r2,m1,m2,b1,b2,beta2)
```

| 参数 | 含义 |
| --- | --- |
| `fname` | case 后缀。 |
| `chordm` | 子午弦长，用于 Reynolds 数。 |
| `r1,r2` | 入口/出口半径。 |
| `m1,m2` | 入口/出口 `m'` 坐标。 |
| `b1,b2` | 流道高度因子，主程序传入 1.0。 |
| `beta2` | 出口相对流角，单位 degree。 |

### 5.2 热力学假设

所有版本都使用完美气体空气：

```fortran
gamma = 1.4
rgas = 287.
cp = gamma/(gamma-1.0)*rgas
```

叶轮周速：

```fortran
omega = 2*pi*rpm/60.
u1 = r1*omega
u2 = r2*omega
```

环形面积：

```fortran
a1 = 2*pi*r1*b1
a2 = 2*pi*r2*b2
```

### 5.3 `write_ises3.f90` 的迭代逻辑

`write_ises3` 外层猜测出口子午速度 `v2m`：

```fortran
v2m = 0.3*c2
```

每次迭代：

1. 根据出口密度、面积、速度计算质量流量。
2. 用质量流量反推入口子午速度。
3. 根据入口绝对速度更新入口静温、静压、密度。
4. 通过相对总焓关系得到出口相对速度。
5. 由出口相对流角得到新的 `v2m`。
6. 用松弛因子 `rf=0.4` 更新。

### 5.4 输出到 `ises.<case>`

写出的主要量包括：

| 量 | 含义 |
| --- | --- |
| `m1r` | 入口相对 Mach 数。 |
| `m2r` | 出口相对 Mach 数。 |
| `s1r` | 入口相对流向斜率。 |
| `s2r` | 出口相对流向斜率。 |
| `Re` | Reynolds 数。 |
| `NCRIT` | 转捩判据参数，代码中写为 9.0。 |
| `TRANS1/TRANS2` | 两侧转捩初值。 |

`write_ises3` 会根据 `m1r>1.0` 判断入口是否超声，并写出不同的 ISES 控制整数。

## 6. 几何辅助函数

### 6.1 `circle.f90`

`circle` 用三点拟合圆：

```fortran
subroutine circle(x1,y1,x2,y2,x3,y3,xo,yo,r)
```

方法是：

1. 求弦 `(1,2)` 和 `(2,3)` 的中点。
2. 求两条弦方向的法线。
3. 两条法线交点即圆心。
4. 圆心到任一点的距离即半径。

该函数目前在主程序中没有启用，只保留在注释掉的尾缘圆弧处理代码里。

### 6.2 `tangent.f90`

`tangent` 用于找到叶片表面和拟合圆之间的切点：

```fortran
subroutine tangent(ilte,n,x,y,xo,yo,r,xt,yt,nt)
```

`ilte=1` 表示前缘搜索，`ilte=2` 表示尾缘搜索。它先从指定边缘向内扫描，找到第一个在圆外的点，然后利用局部表面斜率和半径向量确定圆上的切点。

### 6.3 `vector_product` 和 `dot_product`

这两个函数把二维向量存为 complex：

```text
real(vec) = x 分量
imag(vec) = y 分量
```

`vector_product` 实际执行的是复数乘法，可用于二维旋转；`dot_product` 是普通点积。

## 7. 插值库 `udprog.for`

`udprog.for` 是 fixed-form Fortran 文件，包含：

| 子程序 | 功能 |
| --- | --- |
| `UD0321` | 插值、导数和积分入口。 |
| `UD0322` | 只做插值，是当前主流程主要使用的入口。 |
| `UD0327` | 实际实现线性插值或三次样条插值。 |

当前代码中：

```fortran
call UD0322(..., NTYPE=1)
```

表示使用线性插值。

`UD0327` 也支持 `NTYPE=0` 的样条插值，但当前主流程没有启用。

## 8. 物性函数 `sutherland.f90`

`sutherland(t)` 根据温度 `t` 计算空气动力黏度：

```fortran
sutherland = visco0*(t/t0)**1.5*(t0+ts)/(t+ts)
```

常数：

| 常数 | 值 | 含义 |
| --- | --- | --- |
| `t0` | 273.16 | 参考温度。 |
| `ts` | 124.0 | Sutherland 温度。 |
| `visco0` | 1.7161e-5 | 参考黏度。 |

该函数在 `write_ises*` 中用于 Reynolds 数：

```fortran
re = rho1*w1*chordm/sutherland(t1)
```

## 9. 维护建议

1. 如果要继续现代化代码，优先把三个同名 `write_ises` 版本整理为不同子程序名或模块过程。
2. 如果要提高可移植性，可以把 `pause`、`assigned goto`、老式 fixed-form 插值库逐步替换掉。
3. 如果要改变 `theta=atan2(y,x)` 的定义，应同步验证所有算例的 `blade.<case>` 输出方向。
4. 如果要修正 `write_ises3.f90` 的入口密度迭代误差，应先用已有算例对比 `ises.<case>` 输出，避免改变历史结果。
5. 如果要调整尾缘处理，建议先启用 `circle/tangent` 路线做对比图，再决定是否替代当前 `ndel` 删除点方案。

