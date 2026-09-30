# 第122轮：事后平方信号的独立支持集确认

## 触发与证据边界

R121主要候选（GraphMAE/tr1与GraphMAE2/gsh的alpha0.5/0.75）均未过门槛，保持stop，不改写该轮结论。R121反向形状对照GraphMAE/gsh的带符号平方+k50得到69.64%，相对未变换gsh68.72%为+0.92个百分点，但低于Direct69.80%。这是查看开发结果后提出的假设，明确属于事后选择。

借鉴来源仍为ECCV2010 Fisher归一化式15，alpha2超出原论文0..1范围，是本地反向对照产生的经验迁移；不宣称论文支持平方优势，也不新增可训练Prompt模块。

## 固定方案与对照

唯一主要候选：GraphMAE，gsh根，训练100步后中心化，再sign(z)|z|²，原型与k50读出均重建。20新support seeds188100..188119，confirm查询子集；不在本轮重新选择alpha、checkpoint、优化器或读出。预先固定420条：Direct/原始Frozen/原始GraphPrompt；tr1、gsh、postbase、postgp四根，各自恒等/平方，mean+k50；另gsh平方根mean+k50作为反向形状对照。

主要比较平方gsh k50相对未变换gsh k50，必须平均delta>0且配对support bootstrap95%下界>0才保留“匹配基线改善”。同时报告tr1和Direct；如果无法超过Direct，不宣称最佳任务准确率。相对同幂Frozen/GraphPrompt只能证明当前checkpoint和协议的条件价值，不证明独创机制。平方tr1用于识别优化器交互，mean用于诊断；不能事后将它们改为主要候选。

如主要比较通过，再另立新seed组重复；如未通过，停止该候选。bootstrap抽样单位支持集种子，同图同query相关，且跨轮重复利用confirm query，不能当独立数据集显著性。

## 复用与审计

复用R121冻结driver、power121与其manifest_v2，保持公式、源文件、无query训练、shared tokens、独立几何重建、邻域/权重/传播、argmax、优化器审计契约。R121原失败smoke与v2修订保留。新配置/分析器/runner另封manifest122；全审计通过后再计算配对结论。
