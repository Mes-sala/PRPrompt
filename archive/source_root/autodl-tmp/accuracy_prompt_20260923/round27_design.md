# 第二十七轮：带类别质量约束的原型修正

来源：[SwAV原文](https://proceedings.neurips.cc/paper/2020/hash/70feb62b69f16e0238f741fab228fec2-Abstract.html)，[补充材料的Sinkhorn伪代码](https://proceedings.neurips.cc/paper_files/paper/2020/file/70feb62b69f16e0238f741fab228fec2-Supplemental.pdf)。借鉴视觉聚类中的分配归一化，不复制多视图训练或重训主干。

现有修正教师迭代3次，用温度0.05的softmax给未标注输入分配类别，再以beta0.5混合支持中心和加权中心。本轮只替换该分配：log域Sinkhorn固定50步，令每节点分配和为1、类别平均质量逼近指定比例。uniform指定 $ 1/C $；mixed指定初始原型softmax平均比例与均匀分布各占一半，之后3次修正固定该先验；uniform_shuffle在分配后打乱节点行，保留质量但破坏特征对应。图真实类别比例不读取，不假设真实均衡。

共享参数和融合位置：单一冻结两层GCN，修正教师条件token在第一层激活之前加入，gamma3；仅token训练，最终仍支持余弦原型分类。修正后教师固定，同时用于门控与KL；直接修正教师分类器单独比较。其他训练、支持、query保持一致。

四种教师（原softmax/uniform/mixed/uniform_shuffle）各直接分类和token两方法，加Frozen、GraphPrompt、平滑岭直接教师和后层岭token，共12方法×3图×2权重×5支持seed38100..38104=360条。记录每次分配行和/类别质量误差及先验；单元检查实际函数的边际与有限性，再3步预检。若质量误差大于1e-3，判工程预检失败，不能冒称满足约束。强候选才追加新支持确认；反复使用同图query的限制延续。
