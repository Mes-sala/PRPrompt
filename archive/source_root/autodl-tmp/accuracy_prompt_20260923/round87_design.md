# 第87轮：提示梯度中心化

来源：[ECCV2020 Gradient Centralization](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460613.pdf)，[作者代码库](https://github.com/Yonghongwei/Gradient-Centralization)。原方法在优化器矩估计前去除梯度向量均值，可应用于视觉网络训练。此处迁移到6×128提示token；不要求原创。

原门控、输入、冻结checkpoint、损失、100步和最终中心化+k50保持；GraphMAE的Adam与GraphMAE2的graftedShampoo分别作为匹配强根。

计划：每token沿128隐藏坐标去均值 $ G'=G-\operatorname{mean}_{h}(G) $，再交给对应优化器。对照为沿6类别去均值、原梯度方向乘 $ \|G'\|/\|G\| $ 的范数控制、identity恢复。记录原始/实际梯度，NumPy独立投影及更新重放。第一轮不额外调学习率。

注意轴语义：本项目T是类别×隐藏坐标、gate@tanh(T)，并非PyTorch Linear的out×in存储；每token去隐藏均值是明确的适配选择，不宣称与论文FC轴完全相同。门控各行类别和接近0，类别方向梯度在零初始化时可能已接近零均值；类轴对照可能近似退化，需先原语与真实首步无查询诊断。不能把去均值的梯度推断成Adam后token均值必然固定。梯度包含已有L2项，不额外weight decay。

开发预定CiteSeer seeds133100..104，预检133199。已读取作者algorithm-GC/algorithm/Adam.py：gc_loc=True在矩估计前处理；本轮固定此位置。实现gradient87.py零梯度返回零；先primitive与恢复对照再封存源启动。只有超过当前强根才展开20支持确认，同图重复查询的边界继续保留。

正式125条，预检29条（3步）；方法gch隐藏轴、gcc类别轴、gcn范数控制、gci预检恒等恢复。使用完整含L2梯度变换，GraphMAE Adam及GraphMAE2 GraftedMatrix都只接收变换后梯度，后者预条件矩阵和shadow Adam也都据此更新。保存变换前后100步梯度及SHA，独立NumPy核验投影与优化器重放。训练源封存后不改。

作者源码：https://github.com/Yonghongwei/Gradient-Centralization/blob/master/algorithm-GC/algorithm/Adam.py
