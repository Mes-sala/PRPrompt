# 第53轮固定设计：目标检测Focal Loss用于支持留一训练

来源：[ICCV2017 Focal Loss](https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html)，[arXiv原文](https://arxiv.org/pdf/1708.02002)第3.2节已读，原文定义 $ FL=-(1-p_t)^\gamma\log p_t $。借鉴难样本重加权，仅替换支持留一CE，保留原强token、KD、100步和k50读出；没有前景背景不均衡，因此不照搬alpha类权重。不同于第49轮Poly-1的加性一阶项，本轮为乘性聚焦，仍可能失败，按准确率判断。

固定候选gamma1/2，不扫query。需匹配CE梯度尺度：Focal对CE梯度的系数为 $ (1-p_t)^\gamma-\gamma p_t(1-p_t)^{\gamma-1}\log p_t $。为区分难度重加权和整体缩放，使用每步系数均值stop-gradient乘CE作为动态常数对照；再用同批系数置换stop-gradient乘逐样本CE作为难度对应破坏对照。注意直接detach原focal权重会遗漏其导数，不能冒充原Focal。预检用解析梯度对autograd核对，gamma0精确恢复，极端概率有限。

正式矩阵固定为Cora两权重×seed80100..80104×16方法=160dev。七组训练（原强提示、Focal1/2、动态均值1/2、置换1/2）各评估原型与k50读出，另加Frozen/GP。先Cora两权重5支持开发，原读出与k50读出共享每组训练；继续以同批强基线而非Frozen增益决定扩大20支持。支持/query隔离和反复查询限制延续。若某loss方案改变CE/KD相对强度，须报告对应尺度，不能直接归因难样本机制。

尺度/置换系数均在各自训练轨迹计算，不是强制与Focal整条轨迹同尺度；检验相同规则的缩放替代。每步系数min/mean/max归档。稳定实现以CE及expm1计算1-p，不额外归一或改KD。

## 开发后固定确认

Focal2原型75.92未超过scale2的76.00/perm2的75.96，但scale2对原提示75.48有+0.52pp，按用户准确率优先目标确认更简单对照。固定两权重seed81100..81119共400：Frozen/GP及原提示、scale2、focal2、perm2各原型/k50两读出。主候选为GraphMAE scale2原型，其他是匹配/迁移对照；不得将确认后最佳任意改为预选主候选。query仍反复使用confirm500。

## 二次固定确认

主候选scale2原型74.04对73.46，区间跨零且低于强传播75.00；次要组合scale2+k50为75.68较原传播高0.68。为防确认后挑选，另固定GraphMAE seed82100..82119×原十方法=200，主候选改为scale2+k50并在新支持前明确。仍相同confirm500，不称新独立测试。匹配focal2/perm2加传播以核对难样本特异性。
