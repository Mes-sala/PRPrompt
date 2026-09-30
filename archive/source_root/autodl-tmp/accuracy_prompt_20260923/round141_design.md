# R141封存：加权GNB的Prompt更新

来源与混合目标公式、独立穷举验证见round141_design_draft.md；原H版停止结论不变。采用非负GGN估计而非完整Hessian。固定官方来源commit，但本轮是加权多目标适配，不声称官方Sophia-G直接复现。

保留Cora两层冻结GCN、896晚层token、teacher/gate/原动态scale2支持CE+KD+.001T²、原未中心化k50。每10步在student支持LOO logits与全图logits独立采伪标签，用sqrt(c/S)和sqrt(1/N)加权伪CE求梯度；平方后加2*.001/896精确L2对角。原support标签、原型和teacher不变。实际训练梯度仍来自原目标。

g01/g1/g10的rho=.01/1/10；gp10=与g10同尺度、固定曲率坐标置换；gsg=同动量sign，无曲率。CPUdouble m/h、beta .965/.99、lr .01、eps1e-12、无额外weight_decay；GPU真实token float32，每步舍入。保留原Adam100/普通CE及Adam110反向次数参照。100普通梯度+10伪梯度不是严格等于110次Adam的FLOPs/时间，单列调用数和seconds。

两ck×5模式的无query100步诊断均通过；另v2替换置换为rho10并两ck补测通过，旧gp1诊断保留。平均裁剪g01约99.97–99.98%、g1约97.3–98.5%、g10约81.7–88.0%；未依据query选参数。采样概率来自GPUfloat32 logits复制到CPUdouble的softmax，CPU独立均匀随机流seed+1410000；CDF抽样原uniform/probability/label全部保存。坐标置换独立seed+1410001。

正式support209100..209104、官方train每类5shot、dev500；205条。smoke209199、12步41条（Adam预算14步、伪梯度两次刷新）。identity和smoke全审计后才正式。正式70更新轨迹/140首末快照/50GNB状态/400伪梯度/100近邻，smoke14/28/10/16/20。

审计对所有采样随机数独立再生，概率与CPU重建logits对齐；每次混合伪梯度CPU重算，atol2e-5/rtol2e-4且相对误差<1e-3，伪loss差<1e-3，均在预检前固定。对应原目标loss/gradient、每步m/h/参数的独立NumPy回放、曲率非负与L2项、teacher/forward/原型/k50、SHA及split/accuracy均完整核验。公式原语通过不替代真实伪梯度审计。

扩大门槛：g01/g1/g10/gp10/gsg k50最佳同时超过两ck token/scale2/a110最高者及Direct至少.20pp。平手顺序同上再GraphMAE/GraphMAE2。若置换或sign胜出可以按准确率继续，但不能归为正确GGN位置收益。通过后固定20新支持确认，不事后换选；同图重复query局限保留。
