# 当前可保留的Cora强方案

第53轮保留：GraphMAE冻结主干＋后层条件token＋动态整体CE缩放＋50近邻概率传播。原型教师、token位置及传播均沿用已有有效配置；本轮只改支持监督损失的整体权重。方案实现为 `run_round53_focal.py` 的 `head_scale2_qeps-50`。

## 训练与推断

- 预训练权重：服务器 `/root/ProG/Experiment/pre_trained_model/Cora/GraphMAE.GCN.128hidden_dim.pth`，结果逐条保存SHA；主干为冻结两层GCN128。
- 每类5个支持节点；第一层激活后、第二层卷积前注入C×128条件token，幅度10，平滑岭教师KD，Adam学习率0.01训练100步。
- 每步先计算支持留一原型分类概率。令真实类别概率为 $ p_t $，计算 $ a_t=(1-p_t)^2-2p_t(1-p_t)\log p_t $。实际监督损失为 $ \operatorname{stopgrad}(\operatorname{mean}(a_t))CE $；这是所有支持样本共同的单一系数，不进行逐样本Focal重加权。KD与token L2不变。
- 最终提示嵌入归一化后取50近邻，权重为余弦相似度除以0.2后softmax，无自边。以原型概率为固定锚，alpha0.9传播20步，每步支持概率钳制真实one-hot。

## 已核对的效果和边界

初次确认20支持：75.68%对原传播75.00%，+0.68pp，配对区间[0.02,1.30]。随后预先固定组合，在另外20支持复核：76.48%对75.52%，+0.96pp [0.54,1.51]，16胜2负2平。Focal及置换对照均76.53%，与本方案差异不明确；不宣称难样本对应关系必要。

所有数字为相同图、反复使用的confirm500查询。两批支持不构成独立图证据，也不能按不同批次绝对准确率排名。没有将Cora收益推广到PubMed、异配图或新权重。GraphMAE2对应改动仅局部小增量，未作同等保留结论。

## 复核命令

在服务器复用环境运行；使用新的输出目录，已有目录严格校验配置与来源SHA，不能更改后盲目续跑。

```bash
/root/autodl-tmp/prog_env/bin/python /root/autodl-tmp/accuracy_prompt_20260923/run_round53_focal.py --out /root/autodl-tmp/accuracy_prompt_20260923/recheck_scale2_82100 --datasets Cora --checkpoints GraphMAE --methods head_token_mean head_token_qeps-50 head_scale2_mean head_scale2_qeps-50 head_focal2_qeps-50 head_perm2_qeps-50 --seeds 82100 --steps 100 --phase confirm
```

上面是复核已有支持的命令示例，本文没有再次执行，不计新增结果。证据入口：[第53轮设计](round53_design.md)、[再确认结果](round53b_confirm/summary.md)、[配对审计](round53b_increment_audit.json)。

## 第70轮迁移结论：不添加只读中心化

CiteSeer有效的只读中心化在Cora开发阶段曾提高：GraphMAE scale2+k50 74.36%→75.12%。但预选后的20新支持107100–107119确认失败：中心化74.65%对原75.91%，−1.26pp [−1.98,−0.42]，2胜17负1平。GraphMAE2中心化73.81%对原74.92%，−1.11pp [−1.81,−0.34]。原方案继续作为Cora默认，不根据开发局部正向替换。

中心化后原型也较弱（GraphMAE73.32%对75.02%）；传播虽恢复部分准确率，仍未超过原强读出。两读出严格共用同一提示参数，520条封存预测与独立几何重构通过。[确认结果](round70_confirm/summary.md)、[配对审计](round70_confirm_pairs.json)。同图重复查询限制不变。
