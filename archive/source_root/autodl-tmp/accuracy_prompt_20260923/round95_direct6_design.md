# 第95轮补充六步直接教师对照

确认已启动后发现缺少同教师Direct。单独源码run_round95_direct6.py只运行directplain_mean，固定原六步无阻尼教师，沿用同一特征中心化、原型修正和支持/query。没有提示训练，加载权重不用于Direct计算，checkpoint仍为Direct/None。原训练源码不修改；此为预先说明的补充诊断，不据结果重新选择候选。

沿用开发148100..104和确认149100..119，共25条，不增加或改变已用标签。预测先封存再评分；独立CPU逐类重建六步center/proposal，检查source/graph/split/预测SHA。与原三步Direct和dap提示同支持配对。补测开始于确认结果尚未读取时，不声称是最初矩阵预注册。后续确认若开展需同步保留该对照。
