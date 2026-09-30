# R137固定设计：SOAP特征基中的Adam与预算匹配

来源与实现审查见 `round137_design_draft.md`、`soap137_replay_note.md`。官方源码commit固定；本轮实际为双精度优化器状态适配，不声称原生float32逐位复现，图模型和896token仍float32。新文件 `optimizer137_v2.py` 和 `run_round137_soap_v2.py`，旧文件/失败probe保留。

真实图预检204199发现原生float32投影与独立double投影最大差1.38774e-8，896坐标中774个在1e-8以内，小分母放大为首个有效参数更新2.23163e-4差异。原生相同运算复算误差0，说明是数值敏感而非参数保存错误；不放宽5e-5状态回放门槛。v2把Q/GG/m/v与CPU影子token计算改为double，并在每次更新后回写float32真实token、同步舍入后的CPU影子。两ck×三个模式12调用真实梯度回放最大1.16415e-10，来源 `soap137_graph_probe_v2.json`。没有使用query准确率选择精度。

模式：sop=每10次有效更新QR刷新；sfp=冻结首次真实特征基；sip=恒等基；三者均首次调用只初始化不更新，100梯度调用99次参数更新。betas .9/.999、eps1e-8、lr.01、无额外weight_decay；按官方顺序先更新参数再统计，m先回原坐标后投新基，v只按排序置换。a99=普通torch Adam99调用/99更新，原scale2=100/100；原token普通CE也保留。官方eps/偏差修正顺序和torch Adam不完全相同，因此sip不要求与a99位精确；用独立实际公式与预算控制定位差异。

保持冻结2层GCN、Cora后层token、原平滑岭门控、原KD、动态scale2监督+.001T²、未中心化k50。主变化只有优化器更新；没有新增分类器或查询标签训练。

Cora GraphMAE/GraphMAE2，开发support种子204100–204104、每类5个，query旧dev500。每seed独立Direct一条，每ck四基础frozen/GP/base/GP-k50及token/scale2/sop/sfp/sip/a99六模式双读出，共165。冒烟204199、12调用，共33，覆盖首次初始化与第10次更新换基，Adam99对应11调用/11更新。正式50更新轨迹、100梯度快照、30 SOAP完整状态轨迹；冒烟10/20/6。

独立审计记录所有原梯度、每步参数/m/v、实际Q/GG/排序；NumPy核验统计矩阵、合法特征残差、QR正交与三角关系，再在实际合法基上重构全轨迹，不能将两个简并eigh基不同当失败，也不能跳过基合法性。每步float32参数舍入明确。teacher/late forward/原型/k50、SHA/支持/query/精度、首末实际loss梯度继续独立检查。全部smoke覆盖正确且通过才正式。

扩大门槛：候选sop/sfp/sip/a99的k50均值须同时超过两ck token/scale2中的最高者与Direct至少0.20pp；平手顺序sop/sfp/sip/a99再GraphMAE/GraphMAE2。若控制胜出可继续准确率确认，但不称SOAP换基收益。正式候选需固定20新支持确认；同图查询重复使用的边界不变。R136已失败，不将它升级主根。
