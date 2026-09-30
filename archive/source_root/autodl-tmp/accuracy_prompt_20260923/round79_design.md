# 第79轮：教师门控概率的锐化与平滑

借鉴[DINO](https://openaccess.thecvf.com/content/ICCV2021/papers/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.pdf)的教师softmax锐化思想，并核对[官方DINOLoss实现](https://raw.githubusercontent.com/facebookresearch/dino/main/main_dino.py)。仅迁移概率温度这一组件，不复现EMA教师、多视图或logit中心化，不宣称防坍缩机制成立。正式论文入口亦见[arXiv](https://arxiv.org/abs/2104.14294)。

现有修正原型教师score已除0.2，故门控概率应softmax(score*0.2/t)，不能再次直接除t。固定t=0.1/0.4，t0.2恢复；只改教师门控概率，训练CE/查询温度不改，无KD主根。第57轮只改支持训练温度，此处改变注入提示的节点类别混合，位置不同。

门控为Ahat(Q-mean(Q))。两温度各三模式：原幅度、Frobenius范数匹配原门控、原门控仅乘候选/原范数比。匹配的是门控矩阵整体范数，不是每个节点范数，也不保证任意训练token下实际隐藏扰动一致；需记录实际扰动范围。这样区分门控方向与幅度，而非把锐化简单解释为更强提示。

不读取query标签。所有节点教师argmax在正温度下不变，直接教师类别预测不因本变换提高；改善只可能来自提示优化。保持早层768token、零初始化、Adam0.01/100步、最终只读中心化+k50，两权重新支持118100–118104。带原强根、新KD、Direct及对应幅度控制。若退化门控norm为0则停审，不静默除零。

当前只有gate79原语及合成算子审计，真实教师、训练驱动、逐节点门控重构和正式矩阵尚待实现。


## 固定执行

两温度×三模式×两读出12项，五强提示来源×两读出10项、Frozen/GraphPrompt2项，两权重各24方法，另Direct每支持一次，共245条。支持118100–118104。预检118199共53条：49条标准矩阵加两权重原温度0.2恢复的两读出。全部100步正式、3步预检。

保存实际门控、门控概率与各自SHA、范数比和最终隐藏扰动RMS。独立CPU重算教师score与温度、图传播、范数匹配及实际扰动，再核对最终GCN/原型/近邻传播。amplitude模式的gate_teacher_sha对应原概率，raw/matched对应温度后概率；不沿用旧未变门控的SHA声称新门控。KD不启用，KD元数据保持原参考。驱动+init73+gate79联合源码封存。预检全部通过后才进入正式。
