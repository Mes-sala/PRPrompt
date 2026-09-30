# R157既有强基线同支持比较

原R73源码原样运行，CiteSeer、两GraphMAE checkpoint、227100..227104、dev前500，100步。Directcenter、posttdnone、posttdboth、ins1，原型和k50都保留，共65行。原方法保持原教师/损失/初始化，不能据这种跨方法比较归因LoRA机制；机制归因只看R157匹配scale2。

运行原summarize、teacher/geometry审计，另核对新旧目录split/graph/checkpoint哈希完全匹配。该旧流程保存终态及训练history，但不保存每步梯度，明确没有R157式全步Adam回放。审计通过后比较预先固定GraphMAE l4与全部旧强基线；不从本轮重选checkpoint或读出。
