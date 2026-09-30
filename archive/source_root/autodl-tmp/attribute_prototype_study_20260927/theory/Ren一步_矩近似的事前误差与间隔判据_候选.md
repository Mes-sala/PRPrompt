# Ren欧式一步：矩近似的事前误差与间隔判据

2026-09-27。理论可行性草案；只审查已冻结R25的moment0/moment2，不改其算子、η≤0.1阈值或复制范围，不启动实验。下面是实数运算中的确定性界，不使用真实节点分配、真实软位置或真实类计数来计算证书。结论只涉及是否保持完整算子的预测，不涉及真实标签或Accuracy。

**判断：存在一套仅依赖现有事前量的有限尺度界，值得至多一次冻结规则后的离线覆盖核验。** 它在已分析的联合集中族中具有正确的阶数分离；但η≤0.1本身不能保证任何节点被认证。若覆盖只靠放宽阈值或反复收紧常数才能出现，应停止这条证书路线。

## 1. 信息和算子严格对应R25

未标注池为 $ z_i $，其大小 $ n $、均值 $ g $、中心化节点 $ x_i=z_i-g $、协方差 $ \Sigma=n^{-1}\sum_i x_ix_i^T $，最大半径 $ R=\max_i\|x_i\| $。支持原型为 $ a_c $，每类支持数 $ K_c=5 $。预先计算

$$
\pi_c=\operatorname{softmax}_c(2g^Ta_c-\|a_c\|^2),\quad
d_c=2a_c-\sum_j2\pi_ja_j,\quad h_c=\Sigma d_c,
$$

$$
\nu_c=d_c^T\Sigma d_c,\qquad
\bar\nu=\sum_c\pi_c\nu_c,\qquad
\eta=R\max_c\|d_c\|\le0.1.
$$

因此 $ \sum_c\pi_cd_c=0 $。用 $ m_c=N_c/(n\pi_c) $ 表示未知真实类计数与零阶计数之比；这是证明符号，**不能读取它来计算事前规则**。R25两种冻结近似为

$$
\widehat m_c^{(0)}=1,\qquad
\widehat m_c^{(2)}=1+\tfrac12(\nu_c-\bar\nu),\qquad
\widetilde u_c=g+h_c.
$$

给定冻结复制倍数 $ r\in\{1,8\} $，记 $ T_c=rn\pi_c $。两个预测原型均为

$$
b_c^{(k)}=
\frac{K_ca_c+T_c\widehat m_c^{(k)}(g+h_c)}
{K_c+T_c\widehat m_c^{(k)}},\qquad k\in\{0,2\}.
$$

除R25已有的支持/池矩外，下文只使用这些预测原型、查询范数和已封存查询内积。没有新增温度、参数拟合或真实softmax。

## 2. 相对责任的简单二阶余项

对一个节点记 $ t_c=x^Td_c $、$ v=\sum_j\pi_jt_j^2 $，则 $ |t_c|\le\eta $、$ \sum_c\pi_ct_c=0 $。去掉类别共同logit后，精确责任满足

$$
f_c:=\frac{q_c}{\pi_c}=\frac{e^{t_c}}{Z},\qquad
Z=\sum_j\pi_je^{t_j}.
$$

令 $ E_{3,c}=e^{t_c}-1-t_c-t_c^2/2 $、$ w=Z-1 $。Jensen不等式给出 $ w\ge0 $；Taylor余项给出

$$
|E_{3,c}|\le\frac{e^\eta}{6}|t_c|^3,\qquad
0\le w\le\frac{e^\eta}{2}v.
$$

直接通分得到**精确恒等式**

$$
f_c-\left[1+t_c+\tfrac12(t_c^2-v)\right]
=\frac{E_{3,c}-\sum_j\pi_jE_{3,j}
-w[t_c+\tfrac12(t_c^2-v)]}{1+w}.
$$

因为 $ t_c^2,v\le\eta^2 $，故逐节点余项 $ \rho_c $ 满足

$$
|\rho_c|\le e^\eta\eta
\left[\frac{t_c^2}{6}+\left(\frac23+\frac\eta2\right)v\right].
$$

取未标注池平均，定义完全事前的常数

$$
B_c=e^\eta\eta
\left[\frac{\nu_c}{6}+\left(\frac23+\frac\eta2\right)\bar\nu\right].
$$

于是两种计数近似分别有

$$
|m_c-\widehat m_c^{(0)}|\le
\delta_c^{(0)}:=\tfrac12|\nu_c-\bar\nu|+B_c,
\qquad
|m_c-\widehat m_c^{(2)}|\le\delta_c^{(2)}:=B_c.
$$

这里界的是**相对软计数**，没有多乘一次 $ n $ 或 $ 1/\pi_c $。二阶近似在该实数域内满足 $ \widehat m_c^{(2)}\ge1-\eta^2/2>0 $；实际浮点输入的非正或非有限情况仍应按R25弃权，不裁剪后宣称同一个规则。

## 3. 分子误差和分母下界

令

$$
e_c=\frac1n\sum_i x_i[f_{ic}-1-t_{ic}].
$$

由 $ \frac1n\sum_i x_i=0 $ 与 $ \frac1n\sum_i x_it_{ic}=h_c $，真实软位置满足精确关系

$$
u_c-g=\frac{h_c+e_c}{m_c}.
$$

只用最大半径即可给出

$$
\|e_c\|\le J_c:=R\left[\tfrac12(\nu_c+\bar\nu)+B_c\right].
$$

这是本草案唯一使用的位置余项界，不继续增加第三矩、方向包络或可调常数。它可能保守，但计算简单且不需要节点softmax。

分母也能事前控制。由于 $ \pi $ 下 $ t_c $ 的均值为零、区间长度不超过 $ 2\eta $，Hoeffding引理给出 $ Z_i\le e^{\eta^2/2} $。同时节点平均 $ t_{ic} $ 为零，故 $ n^{-1}\sum_i e^{t_{ic}}\ge1 $。于是

$$
m_c\ge e^{-\eta^2/2},\qquad
\underline m_c:=\max\{e^{-\eta^2/2},1-\delta_c^{(0)}\}\le m_c.
$$

这不依赖真实计数。若不希望使用Hoeffding引理，可只取正的 $ 1-\delta_c^{(0)} $；本草案固定采用上面的最大值，不按结果择优更换。

## 4. 直接给最终原型误差，避免逐层放大

真实Ren原型可写成

$$
M_c=g+\frac{K_c(a_c-g)+T_c(h_c+e_c)}{K_c+T_cm_c}.
$$

对任一冻结近似 $ b_c=b_c^{(k)} $，有精确恒等式

$$
M_c-b_c=
\frac{T_c}{K_c+T_cm_c}
\left[e_c+(1-\widehat m_c^{(k)})h_c
+(\widehat m_c^{(k)}-m_c)(b_c-g)\right].
$$

所以一个简单、完整的事前半径为

$$
\boxed{
E_c^{(k)}=
\frac{T_c}{K_c+T_c\underline m_c}
\left[J_c+|1-\widehat m_c^{(k)}|\|h_c\|
+\delta_c^{(k)}\|b_c^{(k)}-g\|\right],
\quad\|M_c-b_c^{(k)}\|\le E_c^{(k)}.
}
$$

复制倍数不会在这个界里被不合理地单独放大：$ T_c/(K_c+T_c\underline m_c)\le1/\underline m_c $。这并不说明复制对分类无影响，因为预测原型、支持权重及分类间隔仍随 $ r $ 改变。

若要判断**类别专属位置是否可删除**，还必须覆盖“保留真实计数、把位置替换为共同均值”的对照，不能只比较采用近似计数的common_mass0。令

$$
C_c=g+\frac{K_c(a_c-g)}{K_c+T_cm_c},\qquad
c_c^{(k)}=g+\frac{K_c(a_c-g)}{K_c+T_c\widehat m_c^{(k)}}.
$$

其中 $ C_c $ 仍然不实际计算。直接通分得

$$
\boxed{
E_{c,\mathrm{common}}^{(k)}=
\frac{T_c\delta_c^{(k)}}{K_c+T_c\underline m_c}
\|c_c^{(k)}-g\|,
\quad\|C_c-c_c^{(k)}\|\le E_{c,\mathrm{common}}^{(k)}.
}
$$

$ c_c^{(0)} $ 就是冻结common_mass0。$ c_c^{(2)} $ 只是以冻结二阶计数构造的解析参考中心，不是增加一个需要选参的新方法；若严格只核验已有分支，可固定仅使用 $ k=0 $ 的共同池参考，同时分别审查moment0和moment2的完整预测区间。

## 5. 将原型球变成含间隔的三分判断

对于任一事前原型中心 $ b $、半径 $ E $ 和查询 $ z $，令 $ d=\|z-b\| $。完整欧式分数有精确区间

$$
- (d+E)^2\le-\|z-M\|^2
\le-[\max(0,d-E)]^2.
$$

距离由 $ \|z\|^2-2z^Tb+\|b\|^2 $ 计算；R25的封存内积足以重建。无需查询标签。若某候选类 $ y $ 满足

$$
d_y+E_y<\min_{c\ne y}\max(0,d_c-E_c),
$$

它就是该未知真实原型组的唯一赢家。对完整原型和真实计数共同池分别应用这个条件：

- **可忽略位置：** 两者均认证为同一赢家。删除类别专属位置，在该查询上保持原Ren预测。
- **必须保留位置以保持预测：** 两者均认证，赢家不同。此时删除位置必然改变原Ren输出；这不是说保留位置一定提升Accuracy。
- **弃权：** 任一赢家无法认证、存在并列、输入不合法或不满足原η域。不能用预测器之间的实际一致代替缺失的认证。

设查询数为 $ Q $，认证异判数为 $ D $，弃权数为 $ U $，则完整/真实计数共同池的未知异判比例处于

$$
[D/Q,(D+U)/Q].
$$

保持原有1%容差时，上界不超过1%才可称“在该预算下可忽略”，下界超过1%才可称“若要满足该预算则必须保留”，其余整组弃权。这不需要计算真实分配或读取查询标签。moment0和moment2应分别报告，不能事后选择认证率更高者后当成单一冻结规则。

## 6. 为什么值得一次核验，又为什么可能失败

在联合极限使用的方向族中，$ R=O(\varepsilon) $、$ d_c=O(\varepsilon) $、$ \eta=O(\varepsilon^2) $、$ \nu_c=O(\varepsilon^4) $。因此

$$
J_c=O(\varepsilon^5),\quad B_c=O(\varepsilon^6),\quad
\delta_c^{(0)}=O(\varepsilon^4).
$$

对固定 $ r $ 和联合 $ r=\varepsilon^{-2} $，上面的完整原型半径均为 $ O(\varepsilon^5) $；由于查询与原型距离为 $ O(\varepsilon) $，分数误差为 $ O(\varepsilon^6) $。联合极限的严格非零主分类间隔是四阶，因此在该有限族的实数极限中，有望认证完整和共同池各自的不同赢家。这是比“绝对反馈小”更有意义的可行性依据；并非只用趋同球强行证明共同池等价。

但实际几何未必接近这个族，且以下限制不能绕开：

1. $ R $ 与绝对值三角界丢弃了三阶位置矩的抵消和方向信息；最坏节点可能使 $ J_c $ 很松。均值、协方差相同的池仍可有不同高阶矩，不能从这几个量恢复实际余项符号。
2. 严格非零间隔是关键。协方差仅共同缩放类对方向时，反馈可以大却不异判；接近并列时，反馈很小也可能需要弃权。不能用一个标量χ或η代替上述分数区间。
3. moment2改善计数余项，但两个分支共用同一个一阶位置近似，主导 $ J_c $ 不变；不能预设moment2会带来更多有效认证。
4. η≤0.1只控制展开，不保证足够的分类间隔。域外不扩阈值；全弃权是允许且需保留的结果。
5. 以上是实数界。数值协方差的抵消、负的近零投影方差及距离舍入需要保守误差包络；普通float64容差检查不等于严格区间算术。未解决数值包络前，只能称解析界的数值核验。

**建议的单次go/stop：** 先封存本草案公式，只读取R25事前包产生完整与共同池区间，再与事后包作独立合法性核验；不运行新的图实验、不读标签。只有出现可复核、非空的“位置必须保留”认证，同时保留可忽略和弃权边界，才值得把它作为联合机制的指导性结果继续发展。若全弃权、只认证共同池原本近乎精确的设置，或数值包络吞没关键间隔，则归档为“有限界正确但不够实用”，不追加一轮轮常数优化。是否执行此单次核验由主任务决定，本草案没有执行它。


## 7. 预数据清单与本地自检记录

所需量均已存在于R25事前包：`anchor`、`mean`、`support_counts`、`mean_assignment`、`moment_soft`、`projected_variance`、`mass0/mass2`、`covariance_diagonal`、各矩预测原型，以及`query_norm_squared/query_anchor_dot/query_mean_dot/query_moment_soft_dot`；R、η、n在对应pre JSON中。无需新的GPU统计，也无需读取任何真实N/u/q或标签。k2共同池中心可由相同A/g/计数直接计算。

主任务和moment_predictor分别复核了余项常数、两个均值零条件、Hoeffding下界和pooled分式恒等式。纯NumPy自检见[代码](check_ren_moment_certificate_candidate.py)及[记录](Ren一步_矩近似事前界_纯NumPy核验.json)：103个合成族中82个满足原η域，328个矩/复制分支，认证错判0；分数区间最大浮点超界约6.35e-16，位置/计数余项无违反。三个固定复制构造均能区分r1同判与r8的25%位置异判。这仅验证公式及证书可以非空，不是真实图覆盖结果。

R26规则另保留1e-10规范近并列保护。当前R25数据已生成，但其域内外分类差异汇总尚未对制定规则者解盲；因此下一步只能称盲态离线验证，不能称数据生成前预注册。本文及自检均未读取R25真实数据包。
