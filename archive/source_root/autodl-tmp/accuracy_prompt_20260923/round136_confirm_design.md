# R136固定确认：Cora GraphMAE NS矩阵方向与Adam步幅

开发205条全审计通过。固定GraphMAE nmm+k50：77.36%，全局最强scale2 76.88（+0.48pp，4胜1负）；旧CE76.72、gsh76.12、Direct73.32；无momentum NS77.12、无NS动量76.92。开发通过≥0.20pp门槛，但不能用五seed宣称已可靠提高。

不改变run_round136_muon.py/optimizer136.py/muon136_draft.py及audit136.py。新20支持203100–203119，Cora官方训练池每类5个；固定confirm后500节点，已重复使用此查询半区，属于同图支持采样证据。

两ck保留frozen/GP/base/GP-k50及token/scale2/gsh/nmm/nmg/nmc各mean/k50，Direct每seed一次；删除开发明显弱的nmr和nma原幅度分支。每seed33，共660；200完整更新轨迹、400首末快照、320邻域。复用完整45条smoke和恒等恢复，不重复修改来源与容差。

主候选只固定GraphMAE nmm。主要判断必须对两ck各token/scale2/gsh以及Direct七参照全部配对区间下界>0。使用20支持的10000bootstrap、固定seed136000，整数正确个数计算胜负平；报告所有均值/差/95%区间。无NS及无momentum对照用于限定归因，不把单模型胜弱教师说成NS机制成功。不因确认结果改选GraphMAE2。若主判断失败停止当前配置；若通过再用第二组新支持复核，依然明确同图限制。
