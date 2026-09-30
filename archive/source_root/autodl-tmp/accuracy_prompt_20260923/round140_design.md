# R140封存：Sophia-H晚层Prompt优化

来源与固定官方G版对照、H版算法原语与真图HVP有限差分见round140_design_draft.md、round140_probe_note.md。本轮使用论文Hutchinson H版，不使用真实标签梯度平方冒充GNB。

保持Cora、冻结2层128维GCN、896晚层token、原岭teacher/gate/KD/动态scale2与.001T²、未中心化k50。sh0 rho.01、sh1 rho10、sh2 rho100、shp rho10固定坐标置换；均beta1 .965、beta2 .99、lr .01、eps1e-12，每0/10/.../90步一次高斯HVP（100梯度+10HVP）。ssg同动量sign无HVP。a110标准Adam110步，原scale2 Adam100与普通CE保留。原生参数float32，动量/曲率更新CPUdouble再舍入回float32。梯度系数当步detach，HVP以本步冻结系数局部目标计算；不对系数求导。随机探针CPUdouble生成后转float32，独立seed+1400000；固定坐标置换seed+1400001，不改变均值/分布但破坏参数位置对应。

选择rho10/100依据无query裁剪诊断，保留rho.01近符号退化参照：不是根据准确率挑参数。负随机估计不代表真实对角为负。100+10HVP与Adam110只是反向次数代理，不是相等FLOPs或时间；结果保留seconds与显式hvp_calls，训练支持only并用全图无标签KD。

两checkpoint GraphMAE/GraphMAE2；开发support208100..104、每类5、dev500；每seed Direct1、每ck四基础+八来源双读出16，共205。smoke208199、12调用41条，a110对应14步，Hessian两次刷新；必须identity与smoke全审计通过才启动正式。

审计70训练来源完整更新（Sophia50）/140首末梯度快照/400实际HVP/100近邻；smoke14/28/16/20。所有实际高斯/置换重生、u*HVP估计精确核对，全部HVP在CPU冻结系数目标独立重建，并检查该点梯度及loss；HVP allclose atol2e-5/rtol2e-4且相对误差<1e-3（预检前固定）。每步参数/m/h独立NumPy回放；原teacher、晚层forward、原型、k50、split/checkpoint/source/predictionSHA与准确率同审计。已通过独立double有限差分不替代正式HVP检查。

扩大门槛：sh0/sh1/sh2/shp/ssg k50最佳同时超出两ck token/scale2/a110最高者及Direct至少.20pp。平手按上述来源顺序再GraphMAE/GraphMAE2；控制胜出按准确率可继续，但不能归因为正确曲率。通过后固定20新支持确认，不改选；重复同图query范围限制保留。
