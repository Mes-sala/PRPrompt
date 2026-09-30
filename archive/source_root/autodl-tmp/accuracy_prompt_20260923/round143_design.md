# R143 后备：logit标准化蒸馏

来源：[Logit Standardization in Knowledge Distillation, CVPR2024](https://openaccess.thecvf.com/content/CVPR2024/html/Sun_Logit_Standardization_in_Knowledge_Distillation_CVPR_2024_paper.html)，原文从https://arxiv.org/pdf/2403.01427读取（CVF PDF403）；官方references_LogitStandardization commit eb569ac3783f2eed16fbc2b021101175af4fb2f2，mdistiller/distillers/KD.py。原文Algorithm1分母总体标准差K，代码torch.std默认样本标准差K-1并加1e-7；本方案明确采用官方代码版本，不能称两者逐数相同。

只改KD输入logits的标准化，监督LOO-CE、教师门控、冻结checkpoint、896 late tokens、Adam100和k50原读出保持。固定教师是当前平滑岭Q，其等效logits取log Q。标准化teacher不反馈回门控，避免更换多个模块。基础温度暂固定1（本地迁移选择，非复刻论文配置）。

拟比较both原幅度、both初始化token梯度范数匹配、stop-std梯度对照、student-only、teacher-only，并保留原KL及旧CE/Direct。stop与both前向数值一致但梯度不同，检验可微标准差的作用；单侧分离教师锐化和学生尺度效应。由于只为准确率，不要求上述机制成立，但结论要对应控制。R100是支持CE LogitNorm，不等同于本轮全节点KD两端标准化。

先官方函数快照对齐、独立解析梯度、finite difference、平移及近似尺度不变、uniform边界；再真实两ck无query初始化诊断。正式训练矩阵尚未封存，未运行图训练。

## 正式矩阵与决策封存

Cora两个checkpoint，开发seed212100..104、每类5-shot、官方test固定前500；预检seed212199、12步。zbr=both原幅度；zbm=both初始化token梯度范数匹配；zss=stop-std，与zbm共用both的初始匹配权重（不是对自身重新匹配，因此同参数时前向目标保持相同，隔离标准差梯度）；zpm=student-only按自身初始梯度匹配；ztm=teacher-only按自身初始梯度匹配；dk2=原KL双权重。再加scale2、旧CE、Direct、frozen/GraphPrompt及mean/k50。205正式，70全更新轨迹、140快照、100近邻；45条冒烟额外identity，16轨迹32快照22近邻。

教师Q固定，gate依然Q，原scale2监督目标和Adam100不变。每个variant都保存初始五种KD梯度和固定权重、每步loss各分量、原始梯度与首末快照。CPU独立标准差公式使用sum(centered²)/(K-1)重建，KL以显式概率log比重建，全部Adam状态由NumPy逐步回放。模型/图/支持query/预测SHA及k50几何审计继承。

主阈值：最佳候选对两ck旧CE/scale2最强值与Direct都至少+0.20pp才固定20支持确认。控制胜出只记真实准确率配方，不硬归因标准化。标准差符号控制如优于完整梯度，保留证据。目标是准确率，不要求创新。反复同图query限制保持。脚本/配置/分析/设计启动前SHA封存。
