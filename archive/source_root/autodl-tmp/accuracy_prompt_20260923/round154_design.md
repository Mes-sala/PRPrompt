# R154：LoRA 行列屏蔽与共享层 Prompt

## 借鉴与实现边界

读 [LoRA Dropout 预印本 §3.2 式(2)、式(8)](https://huggingface.co/papers/2404.09610)，本地保存`loradrop154_paper.md`。检索页关联仓库`AdvMarTech/LoRA_LLM_Accident`（commit46d9f865f532c45dcd25ea7a83e5d00b7f07ab80）实为船舶事故论文，README及train_final.py与目标机制不符，因此明确拒绝作为作者实现依据。本轮仅是按预印本公式实现的本地迁移，不宣称官方代码复现或正式版逐项复现。

式(2)屏蔽输入/输出维：$\Delta W_m=\operatorname{diag}(m_B)BA\operatorname{diag}(m_A)$；$m$为独立Bernoulli(q)，$q=1-p$。没有rank维屏蔽。论文公式不含inverted缩放，期望增量$q^2BA$。本地每训练步整个图共享一对行列掩码，不是节点独立mask；由CPU独立generator(seed+154000)预生成保存，不能由query选择。

## 共享位置及矩阵

冻结原第二层W/bias，LoRA-r1通过权重增量改变该层；Prompt通过teacher gate乘可训练token改变同一层输入。原W路径、固定teacher、KD和scale2目标均保留。

- m1/m3：论文式未缩放行列屏蔽，p=.1/.3，联合Prompt。
- i1/i3：同一mask流，但按$q^{-2}$反向缩放，联合Prompt，本地控制变体。
- g1/g3：无随机屏蔽，确定性$q^2BA$，匹配m1/m3的期望权重增量。
- a1/a3：未缩放屏蔽p=.1/.3、无Prompt。
- l1/l4：旧LoRA-r1，无屏蔽，A/B LR分别.01/.0025，Prompt均.01。

前8模式Adam .01/100步，A为原LoRA-r1 kaiming(seed+146000)、B=0，Prompt=0。256适配参数、联合Prompt1152。正则统一基于确定性推理增量的均方，.001 token均方+.001 evalDeltaW均方；不将随机正则范数额外混入对照。

推理使用**期望权重增量**：m/a/g为$q^2BA$，i为BA。不是论文测试时随机预测集成，也不将“期望权重预测”称为“期望预测”。论文MC集成本轮未运行，明确保留为后续选项；不在query上挑mask或集成次数。R12隐藏特征dropout与R97 token DropConnect是不同位置的既有工作。

## 验证、协议和决策

先穷举3输入×3输出所有64种掩码，逐概率加权核对增量期望；检查显式diag等价、有限差分梯度、零B恢复及mask复现。真实GCN三步独立CPU回放必须检查保存mask逐位相同、每次首末梯度按实际mask重建、全步Adam状态和最终期望权重读出；不能用无mask前向审计有mask训练。

Cora每类5-shot；GraphMAE/GraphMAE2；support224100..224104，dev query500，正式285预测/110轨迹/220快照/140近邻。smoke224199/12步61条。保存全部旧CE/scale2/Frozen/GraphPrompt/Direct参照。预测先封存后评分。固定10候选×2ck k50选择空间；最佳者同时超过同批旧CE/scale2 k50最强者和Direct .20百分点才确认。20新support确认不换赢家；控制胜出只作准确率结论，不冒充dropout机制。重复图/query的支持bootstrap不代表独立图泛化。

预封存结果：六组穷举期望/梯度/零B原语通过，最大期望误差4.44e−16。真实两个GCN×十方法的20条三步独立回放通过，最大梯度差2.2352e−8、Adam回放7.45e−9，实际mask逐位核验，无query标签。
