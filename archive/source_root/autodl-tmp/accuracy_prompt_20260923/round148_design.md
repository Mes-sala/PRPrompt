# R148 LoRA+学习率分组

来源[ICML2024 LoRA+](https://proceedings.mlr.press/v235/hayou24a.html)，已下载原PDF并读第4节末至第5节。作者仓库references_LoRAPlus固定commit a0c44bfec0eff97b9574469366bc6b0ccc28838d，官方create_loraplus_optimizer把B的学习率设为A的倍率。理论为大宽度与额外假设下的缩放结论，不能直接断言128维GCN必定获益。这里借鉴优化规则，不主张Prompt机制原创。

原rank1零B LoRA，A初始化与R146同种子support+146000。共享第二GCN层forward与late Prompt，原W/bias冻结但有效W改变。Prompt始终Adam.01，只调整因子；物理DeltaW及token均方正则保持.001。与作者将普通训练参数归A组不同，本地显式保持Prompt学习率不变，防止Prompt步长成为混淆项。

各方法(A学习率,B学习率)：eq(.01,.01)；u4(.01,.04)；u16(.01,.16)；d4(.0025,.01)；d16(.000625,.01)；h4(.04,.04)；l4(.0025,.0025)；r4(.04,.01)。前八联合Prompt，ua/da为u4/d4仅adapter。u4/u16与d4/d16分别升B和降A；h4/l4检验整体幅度；r4反向比例对照。不能只比较不同倍率却把全部效果解释为因子不对称。所有联合1152参数，仅adapter256。

10组四步toy更新与抽取的官方优化器逐位一致。真实两ck×10三步轨迹20项过独立CPU梯度和NumPy按坐标学习率Adam重放，最大梯度5.9605e−8、更新8.9407e−8。所有初始状态独立clone，CPU参数分组没有共享初值污染。

两ck、support218100..104、dev前500query，原teacher/gate/CEscale2+KL、100步不变。285条/110轨迹220快照140近邻；先seed218199的61条12步冒烟24轨迹48快照30近邻、旧根identity及eq与旧LoRA三步对应检查，全过才正式。原Direct/frozen/GraphPrompt、旧CE/scale2保留。候选为十种适配×两ck的k50最高，相对最强原根及Direct均至少+.20pp才锁定新20support确认；普通eq或整体步长对照胜出同样按准确率保留，但不归于LoRA+比例机制。查询标签在预测封存后才读取。

R147 best74.96低于强根75.20，无新强根升级。query跨轮反复使用，开发不是独立泛化证据；不按中途准确率修改倍率或增加训练步数。
