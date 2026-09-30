# R150 OFT分块权重旋转

来源[NeurIPS2023 OFT](https://arxiv.org/html/2306.07280v3)第3.2–3.3节，作者实现commit59ab7eae36a963ef8152ee5f1da460f10310ff53，oft-db/oft_utils/attention_processor.py458–522。代码采用Q=(I−S)(I+S)^−1，与论文符号相反但表示类相同；本地遵循代码符号。torch Linear[out,in]对应W_eff=W Q^T，bias保持原值。保持权重行Gram，不等于保证节点表示几何或准确率不变。

本地用每块严格上三角独立坐标生成S=−S^T，Cayley用solve实现。作者存完整块参数，存在冗余；这里是紧凑参数化，不声称Adam轨迹逐位等同完整参数版。block4/8分别192/448可训练旋转参数，加896Prompt为1088/1344。原W冻结但有效W改变，属于Prompt+PEFT共享第二GCN forward。

c4/c8为Cayley联合，n4/n8为同参数且初始梯度相同的Q=I−2S线性对照；a4/a8/b4/b8为对应adapter-only。另保留普通LoRA-r1/l4（整体小步长）。四项矩阵检验通过正交性、行Gram、单位初始化、数值梯度与官方展开forward；真实两ck×block4/8×两种变换8项初值bit-exact且首步梯度相同。多步独立CPU inverse-Cayley/邻接传播/bias/目标与Adam重放已通过。

support220100..104，dev前500query，两ck，285条/110轨迹220快照140近邻；先seed220199的61条12步冒烟24轨迹48快照30近邻和旧根identity。原teacher/gate/scale2 CE+KD、100步Adam.01、token和物理DeltaW正则.001不变；l4单独沿用其A/B=.0025。保持原Direct/旧CE/scale2/frozen/GraphPrompt。

十适配×两ck中k50最高相对最强原根和Direct都至少+.20pp才固定20新support确认；线性控制或单adapter胜出也按准确率保留，但不归为正交约束/Prompt贡献。R149最高73.88低于根74.20，梯度初始化未升级，不增加新强根。query跨轮反复使用，增益仍需确认，不能借用图像生成的结果声称图任务有效。不采用COFT投影或额外幅度参数，避免混淆。
