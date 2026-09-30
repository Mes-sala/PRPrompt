# 第55轮固定设计：将有效动态CE缩放迁移PubMed

第53轮Cora的整体缩放加k50读出已在另外20支持复核，不需要先追求更多结构改动。下一步固定gamma2缩放规则，迁移到PubMed强提示训练；不照搬Cora的传播，因为第48轮在PubMed失败。

GraphMAE保留VAT0.3、ref50教师、早层gamma3；GraphMAE2保留原ref50强根。唯一目标改动是支持留一CE乘当前Focal2解析系数均值（detach）。带原CE、真正Focal2、置换系数，普通Frozen/GraphPrompt及直接教师。拟直接固定20支持确认，不在PubMed扫gamma；如显存限制按权重串行，不改变数据协议。

实现必须给VAT训练函数接入同一损失接口，不能偷换为无VAT的focal_prompt。预检scale gamma0精确恢复原VAT和非VAT根，随机方向生成器/探测步/一致性权重不变；追踪CE系数、KD和VAT损失变化以解释相对权重。若标签平滑第54轮有强增量，再作为独立方法加进固定矩阵，否则不扩展。

固定PubMed两权重×20支持seed85100..85119×7方法=280confirm，方法为Frozen/GP/ref50直接教师/原强根/scale2/focal2/perm2。仍反复使用同图confirm500；只能证明支持抽样稳定性，不能当跨图泛化。复用全部数据/权重，完成预检再清理，正式输出保留。

实现只修改VAT函数支持监督loss接口，默认original沿用CE；新增独立lossgen不影响原noisegen。GraphMAE2调用非VAT focal_prompt，强根仍为原token_prompt。预检gamma0恢复（VAT与非VAT），核对source模型冻结与解析系数；不迁移任何Cora传播。第54轮未超过新强根，标签平滑不加进本矩阵。
