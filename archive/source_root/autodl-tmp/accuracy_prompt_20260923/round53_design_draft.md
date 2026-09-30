# 第53轮草案：目标检测Focal Loss用于支持留一训练

来源：[ICCV2017 Focal Loss](https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html)，[arXiv原文](https://arxiv.org/pdf/1708.02002)第3.2节已读，原文定义 $ FL=-(1-p_t)^\gamma\log p_t $。借鉴难样本重加权，拟仅替换支持留一CE，保留原强token、KD、100步和k50读出；没有前景背景不均衡，因此不照搬alpha类权重。不同于第49轮Poly-1的加性一阶项，本轮为乘性聚焦，仍可能失败，按准确率判断。

固定候选gamma1/2，不扫query。需匹配CE梯度尺度：Focal对CE梯度的系数为 $ (1-p_t)^\gamma-\gamma p_t(1-p_t)^{\gamma-1}\log p_t $。为区分难度重加权和整体缩放，拟使用每步系数均值stop-gradient乘CE作为动态常数对照；再用同批系数置换stop-gradient乘逐样本CE作为难度对应破坏对照。注意直接detach原focal权重会遗漏其导数，不能冒充原Focal。预检用解析梯度对autograd核对，gamma0精确恢复，极端概率有限。

尚未实现或固定正式矩阵。先Cora两权重5支持开发，原读出与k50读出共享每组训练；继续以同批强基线而非Frozen增益决定扩大20支持。支持/query隔离和反复查询限制延续。若某loss方案改变CE/KD相对强度，须报告对应尺度，不能直接归因难样本机制。
