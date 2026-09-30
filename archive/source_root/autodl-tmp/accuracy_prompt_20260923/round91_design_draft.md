# 第91轮草案：中心对数比教师门控（未实现）

跨领域来源：[Aitchison 1982组成数据分析](https://doi.org/10.1111/j.2517-6161.1982.tb01195.x)。本次检索确认文章元数据，出版社全文读取受限；具体CLR公式已由[scikit-bio官方文档](https://scikit.bio/docs/dev/generated/skbio.stats.composition.clr.html)核对，不声称已通读原文。借用对概率组成取对数并减几何均值的表示，不要求原创。

设当前rectified教师最终概率 $ Q=\operatorname{softmax}(S/0.1) $，CLR为 $ C=\log Q-\operatorname{mean}_{class}(\log Q) $。本项目可直接计算 $ C=S/0.1-\operatorname{mean}_{class}(S/0.1) $，这与CLR严格等价，避免softmax下溢后再log或添加epsilon。实现前明确现有raw_scores已含/.2，故应用raw_scores*2再减类均值，不重复缩放。

新门控 $ B_c=\hat A(C-\operatorname{mean}_{node}C) $；原门控 $ B_q=\hat A(Q-\operatorname{mean}_{node}Q) $。主候选将Bc的Frobenius范数匹配Bq；保留原Bc、原Bq乘Bc/Bq范数比的纯幅度控制。无额外token，无终端模型融合；仍是一个冻结GCN、768token、原CE+L2和当前Adam/Grafted、100步、中心化+k50。Direct完全沿用原教师，不把CLR门控称新Direct分类器。

精确代数边界：这就是类别中心化logits，不是新概率校准。正温度改变只会线性缩放CLR，范数匹配后温度效应抵消，所以不扫描CLR温度；现有概率门控基准仍为.1。潜在好处仅是保留低概率类别相对信息，是否有效需要实测。

原语检查log_softmax CLR等价、常数logit平移不变、类和0、等范数、零方向分支与恢复原概率门控。独立CPU重建教师/logits/门控，NumPy更新重放。拟140100..104、预检140199，CiteSeer两checkpoint；超过对应当前根才新20支持确认，同图重复查询限制继续保留。当前仅草案，无实现或结果。
