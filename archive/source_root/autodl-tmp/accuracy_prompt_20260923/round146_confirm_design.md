# R146 固定PiSSA确认

开发已审计245条。固定GraphMAE2/head_pp_qeps-50（principal-r1+Prompt）75.88%，相对本批最强GraphMAE旧CE75.64%为+.24pp；3胜2负，配对差[1.4,1.4,-.8,2.,-2.8]。同ck随机75.44%、LoRA-r1 75.72%、r2 75.40%，不能把相对旧根的全部增益归于主奇异初始化。

新support216200..216219，confirm后500query，100步配置完全不变。完整980条/360轨迹/720快照/480近邻，先独立审计再固定分析。主判据为相对Direct及两ck旧CE/scale2共5项的10000次配对bootstrap区间下界全部>0（seed146000），才第二组新20support。机制参考同ck随机/r1/r2/adapter-only单列。不得用确认中的对照改选候选。区间仅同图support抽样，query跨轮重复使用，不是新图泛化或独立测试集证明。
