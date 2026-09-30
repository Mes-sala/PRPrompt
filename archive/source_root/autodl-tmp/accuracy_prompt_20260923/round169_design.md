# R169 KRAdapter图适配方案

来源：[ICCV2025论文](https://arxiv.org/abs/2508.00230)，全文kradapter169_paper.md；作者代码ea88a711c7ea1a9cb9a0350879c69efac852c0a6，原README和toy源码保留。已核对论文3.2/3.3与实际README前向；源码merge错误转置且unmerge拼写weigh，均不调用。

共享两层GCN checkpoint、第二层有效权重适配和类条件Prompt共同训练；基W冻结但有效W变化，不是纯冻结前向或预测融合。论文列式Khatri-Rao、README行式乘积、toy行式次序分别记录：不把形状相同当实现相同。论文scale .1和Kaiming负斜率sqrt(1/11)，README scale2和斜率5，toy scale1和斜率sqrt5。

候选p1/p2为论文方向scale.1/2，r1/r2为README方向scale.1/2，t1为toy次序与初始化scale1；k1为p1低学习率.00025，其他.0025。每种KR含128×11零因子与128×12随机因子，2944参数，加896Prompt。b11/b12为参数量相邻普通LoRA rank11/12（2816/3072因子参数），加旧l1/l4；预算比较不只用rank1弱对照。损失与teacher/k50不变，Prompt.01。

20条两checkpoint真实GCN三步轨迹已通过独立逐行Kronecker前向、首末梯度和全部Adam步回放。尚未正式运行，开发支持239100..104，smoke239199，完整285与110轨迹/220快照/140邻域审计后分析。候选顺序p1/p2/r1/r2/t1/k1/b11/b12/l1/l4，整数正确数解平局，各先GraphMAE。对最强旧根及Direct均至少+.20才固定20新支持确认；旧l4复用已存确认，不因新机制低于旧对照而事后换胜者。


正式封存：先61条12步smoke，再285条100步。真实GCN20轨迹最大参数和梯度差均2.9802322e−8。合并/解合并接口不调用，直接构造有效权重。
