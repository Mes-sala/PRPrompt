# 第80轮：教师门控锐化固定迁移到PubMed

第79轮CiteSeer两组20支持均确认raw温度0.1相对原提示的增益，但未稳定超过幅度控制或Direct。依据准确率目标保留该配方；下一步只作固定跨数据集迁移，不再次搜温度。来源仍为第79轮DINO教师概率锐化借鉴，不增加新原创性要求。

复用run_round71_pubmed_v2.py已验证的PubMed强根：原未中心化修正教师、早层token_kd3_ref50；GraphMAE继续VAT0.3，GraphMAE2保持无VAT；均保留原型读出，不叠加失败的只读中心化或k50。两权重冻结GCN，3类×128=384提示参数。

只将用于门控的教师概率改为softmax(score*2)，KD监督目标仍为原softmax(score)，即隔离门控变化。VAT扰动强度、随机流及训练损失保持原实现。raw锐化、范数匹配锐化、原门控仅幅度匹配三个固定分支；强根和独立Direct同时保留，GraphMAE带无VAT参照。不能把CiteSeer的中心化教师或768参数直接复制过来。

预期新支持121100–121104、同既有5-shot/开发500协议。尚需核对原VAT前向/teacher元数据并实现独立门控与KD分离审计，正式矩阵待封存。当前仅迁移设计，未实现、未启动。若增量不存在，保留CiteSeer局部收益，不继续强行推广。


## 封存实现

run_round80_gate.py新增gate80_prompt，复用VAT函数，其KD目标仍原teacher，门控独立保存gate_probability/gate/kd_teacher和SHA。正式5支持×两checkpoint，共70条；两checkpoint独立输出目录避免旧config冲突，Direct只在GraphMAE批生成一次。每组包括原根、raw、matched、amplitude、Frozen、GraphPrompt；GraphMAE另有无VAT和Direct。均原型读出，无k50。

预检121199、3步共16条，额外tr2温度0.2要求恢复原根分数(atol8e-5)及预测完全一致。先完成两权重预检及审计，才开始正式100步。audit80在CPU独立重建教师、门控、KD目标和最终GCN/原型，核对VAT配置与扰动RMS；audit71检验Direct与预测引用；summarize重算精度和封存SHA。源码及配置SHA见round80_source_manifest.json。正式训练开始后不可编辑训练代码。
