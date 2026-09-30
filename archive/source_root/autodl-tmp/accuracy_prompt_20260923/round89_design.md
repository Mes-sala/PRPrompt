# 第89轮：AdaBelief提示更新

来源：[NeurIPS2020 AdaBelief](https://proceedings.neurips.cc/paper/2020/file/d9d4f495e875a2e075a1a4a6e1b9770f-Paper.pdf)，算法2；[作者页面](https://juntang-zhuang.github.io/adabelief/)。借鉴视觉分类、语言模型和生成模型中的残差二阶矩更新：先更新梯度EMA，再积累当前梯度与新EMA差值的平方。此处仅以准确率为目标，不要求原创。

拟固定当前教师门控、一个冻结GCN、6×128零初始化token、CE+L2、100步和中心化+k50，只改提示参数的更新方向。beta1=.9、beta2=.999、lr=.01；实现前核对作者源码与论文的epsilon位置（论文算法2在残差矩累积中也加epsilon），明确选择并封存，不默认为普通Adam分母epsilon。不开启rectify、AMSGrad或额外weight decay。

比较：原AdaBelief方向；将其范数匹配当前根方向；当前根方向乘AdaBelief/根范数比作为幅度控制。GraphMAE根为Adam，GraphMAE2根为保留graftedShampoo。影子根在候选自身梯度上更新，不能声称数值步长等于另一条独立根轨迹。保留真正独立tr1/gsh及Direct/Frozen/GraphPrompt。保存完整梯度与两类状态诊断，NumPy独立重放；恒等恢复、零梯度带动量等原语先过。

避免与第88轮确认135100..119重叠；正式使用136100..104，预检136199。只在超过对应强根后确认新20支持；同图重复查询限制继续记录。实现完成，待预检及正式运行。

源码核对：作者update_0.2.0分支pypi_packages/adabelief_pytorch0.2.0/adabelief_pytorch/AdaBelief.py，先更新m，再残差平方EMA，epsilon用in-place累加到残差矩，然后偏差修正与分母epsilon。本轮显式1e-8（视觉配置）、rectify=False、amsgrad=False、weight_decay=0（损失已有L2），非库默认参数。CPUdouble更新、float32token；保存原根/Belief/实际方向及SHA。原语独立NumPy20步含零梯度，恒等与Adam/grafted恢复；正式125、预检29。
