# 第120轮草案：Yogi二阶矩方向

已核对[Adaptive Methods for Nonconvex Optimization, NeurIPS2018](https://proceedings.neurips.cc/paper/2018/file/90365351ccc7437a1309dc64e4db32a3-Paper.pdf)算法2、第3.2节及实验epsilon。借鉴其加性符号二阶矩更新，应用到单768token，冻结同GCN；不是新增Prompt分支或终端融合，不推断原文收敛界在此成立。

使用 $ v_t=v_{t-1}-(1-beta_2)sign(v_{t-1}-g_t^2)g_t^2 $，beta1=.9、beta2=.999，零矩初始化，保留偏差校正。eps位于sqrt外。原文算法为表述简化省略debias，正文说明可使用有偏差校正版本。拟同时试eps=1e-8与论文实验的1e-3；前者固定当前Adam数值约定，以分开分母更新与epsilon影响，后者需额外匹配Adam eps1e-3。

optimizer120.py支持raw、范数匹配、仅幅度、identity；参考方向GraphMAE为Adam、GraphMAE2为graftedShampoo，均使用相同本候选梯度流的影子状态，不能解读为另一个独立训练模型。无梯度时Yogi矩保持不变，但momentum可能保留；若基线方向为零，幅度控制保持零更新，不能凭空构造方向。

初版check120.py错误地对零基线方向要求幅度控制仍等于非零Yogi范数，因此断言失败；优化器本身正确执行零方向边界。check120_v2.py显式验证这些零分支，原检查保留。16组eps/base/control全检查通过，独立NumPy方向误差4.44e-16；验证identity恢复当前根和零梯度不衰减Yogi二阶矩。只完成原语，真实图未运行，正式种子/配置未封存。待接入driver、全方向/更新审计、几何和步长控制。

建议正式比较两种eps各raw/matched/amplitude、当前两根及额外Adam eps1e-3参照，均100步；有收益再与匹配lr/预算确认，不把短程内epsilon和学习率依赖忽略。最终超参和规模须在启动前封存。
