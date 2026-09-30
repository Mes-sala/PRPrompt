# R129预检恢复审计

原smoke129的零margin与旧根训练后token逐位相同断言失败；原runner终态250229退出，正式未启动。保留原日志/config/source/manifest/预检，不重写已有记录。诊断smoke129_identity_diagnosis和smoke129_root_repeat_diagnosis显示独立原根重复也有2.78e-7量级token、1.43e-6 score差异，四组预测均一致；符合GPU聚合浮点非逐位复现，未证明其他硬件全局确定性。

v2只修改跨运行恒等标准：token atol1e-6、score atol5e-6，rtol0，所有预测精确相等；比现有独立CPU几何/优化器回放容差更紧。新增所有保存基础logits上m0值与梯度逐位恒等。完整CPU真实梯度、全部NumPy损失、全轨迹优化器、教师/读出几何阈值保持。原失败不是准确率失败。

原smoke129完整审计先复核，不再次生成重复49条；全部通过后才运行原配置194100..104、225条正式。源码/原manifest不修改，恢复入口与v2审计另外封存。
