# 主线02消融结果

两跳传播宏均 +4.509 pp；Cora +22.240、CiteSeer +17.240 pp，但 Actor −3.816、Cornell −10.270 pp。中心化宏均 −6.180 pp，Computers −23.498、Photo −20.700 pp；只在 Cora、CiteSeer、Actor 均值为正。软原型修正在 CiteSeer +4.760、Cora +2.760 pp，整体 −0.591 pp。额外迭代、自适应分配与伪标签对应均无普遍收益。本线保留 CiteSeer 条件下的修正教师，不能推广为所有数据集默认教师。它仍是直接读出方案，不是可学习Prompt。


口径：以下增量均为预定义完整分支减消融分支，单位为准确率百分点；总体采用8数据集等权宏均值。每个checkpoint内配对5个相同支持采样seed，再对checkpoint汇总。5个seed不是5张独立图，未宣称统计显著性或独立外部确认。主线03的tr1与gsh完整分支不同，不能混用消融分母。


详细配对表如下。


每项按同seed、同checkpoint配对。差值=完整分支−消融分支。正值是保留模块收益；并非统计显著性结论。

## 中心化

完整：`A2_center`；消融：`rect_uncentered`。

| 数据集 | checkpoint | SHA前缀 | 配对增量均值±标准差pp |
|---|---|---|---:|
| Cora | Direct | direct | +0.060 ± 0.541 |
| CiteSeer | Direct | direct | +0.620 ± 1.092 |
| PubMed | Direct | direct | -1.120 ± 0.850 |
| Computers | Direct | direct | -23.498 ± 4.213 |
| Photo | Direct | direct | -20.700 ± 3.639 |
| Actor | Direct | direct | +0.724 ± 2.238 |
| Chameleon | Direct | direct | -3.904 ± 6.415 |
| Cornell | Direct | direct | -1.622 ± 7.302 |

## 软原型修正

完整：`A2_center`；消融：`A2_norect`。

| 数据集 | checkpoint | SHA前缀 | 配对增量均值±标准差pp |
|---|---|---|---:|
| Cora | Direct | direct | +2.760 ± 1.644 |
| CiteSeer | Direct | direct | +4.760 ± 1.585 |
| PubMed | Direct | direct | -0.200 ± 2.662 |
| Computers | Direct | direct | -3.140 ± 6.833 |
| Photo | Direct | direct | -6.385 ± 4.533 |
| Actor | Direct | direct | -0.816 ± 1.374 |
| Chameleon | Direct | direct | -0.088 ± 3.456 |
| Cornell | Direct | direct | -1.622 ± 5.604 |

## 两跳传播

完整：`A2_center`；消融：`A2_noprop`。

| 数据集 | checkpoint | SHA前缀 | 配对增量均值±标准差pp |
|---|---|---|---:|
| Cora | Direct | direct | +22.240 ± 2.329 |
| CiteSeer | Direct | direct | +17.240 ± 3.080 |
| PubMed | Direct | direct | +3.380 ± 1.681 |
| Computers | Direct | direct | +6.117 ± 3.664 |
| Photo | Direct | direct | +0.521 ± 4.137 |
| Actor | Direct | direct | -3.816 ± 2.689 |
| Chameleon | Direct | direct | +0.658 ± 5.546 |
| Cornell | Direct | direct | -10.270 ± 9.819 |

## 3次相对1次迭代

完整：`A2_center`；消融：`A2_oneiter`。

| 数据集 | checkpoint | SHA前缀 | 配对增量均值±标准差pp |
|---|---|---|---:|
| Cora | Direct | direct | +0.260 ± 0.336 |
| CiteSeer | Direct | direct | +0.960 ± 0.404 |
| PubMed | Direct | direct | -0.660 ± 1.031 |
| Computers | Direct | direct | -0.215 ± 1.801 |
| Photo | Direct | direct | -2.077 ± 2.234 |
| Actor | Direct | direct | -0.395 ± 1.152 |
| Chameleon | Direct | direct | +0.219 ± 1.316 |
| Cornell | Direct | direct | +0.541 ± 2.961 |

## 自适应类别分配

完整：`A2_center`；消融：`A2_uniform`。

| 数据集 | checkpoint | SHA前缀 | 配对增量均值±标准差pp |
|---|---|---|---:|
| Cora | Direct | direct | +2.660 ± 1.635 |
| CiteSeer | Direct | direct | +4.760 ± 1.627 |
| PubMed | Direct | direct | +0.060 ± 2.479 |
| Computers | Direct | direct | -3.501 ± 7.179 |
| Photo | Direct | direct | -6.582 ± 4.170 |
| Actor | Direct | direct | -0.579 ± 1.380 |
| Chameleon | Direct | direct | +0.789 ± 3.236 |
| Cornell | Direct | direct | -3.243 ± 5.199 |

## 伪标签节点对应

完整：`A2_center`；消融：`A2_permassign`。

| 数据集 | checkpoint | SHA前缀 | 配对增量均值±标准差pp |
|---|---|---|---:|
| Cora | Direct | direct | +2.960 ± 1.776 |
| CiteSeer | Direct | direct | +4.940 ± 1.218 |
| PubMed | Direct | direct | +0.080 ± 2.994 |
| Computers | Direct | direct | -3.790 ± 7.125 |
| Photo | Direct | direct | -6.481 ± 4.121 |
| Actor | Direct | direct | -0.645 ± 1.331 |
| Chameleon | Direct | direct | +1.184 ± 3.736 |
| Cornell | Direct | direct | -1.081 ± 3.082 |

## 中心化×修正的2×2交互

定义：`A2_center - A2_norect - rect_uncentered + rawprop`。描述性交互，正值不代表已证明普遍协同。

| 数据集 | checkpoint | SHA前缀 | 交互pp |
|---|---|---|---:|
| Cora | Direct | direct | -1.660 |
| CiteSeer | Direct | direct | -1.820 |
| PubMed | Direct | direct | -0.380 |
| Computers | Direct | direct | -2.172 |
| Photo | Direct | direct | -6.812 |
| Actor | Direct | direct | -0.934 |
| Chameleon | Direct | direct | -0.351 |
| Cornell | Direct | direct | +1.081 |