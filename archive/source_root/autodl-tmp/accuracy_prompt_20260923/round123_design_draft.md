# R123：条件径向投影的无查询前置诊断

来源：[AdamP，ICLR2021原文](https://sanghyukchun.github.io/home/media/papers/heo2021iclr_adamp.pdf)，[作者实现](https://raw.githubusercontent.com/clovaai/AdamP/master/adamp/adamp.py)。已核对先逐channel、再整层的夹角判据，默认delta0.1、eps1e-8，投影分母为norm+eps而余弦分母为各norm.clamp_min(eps)。当前迁移目标是grafted Shampoo更新，不冒充官方AdamP优化器。

现有Prompt的tanh与加法不具尺度不变性，原理论不直接适用。先在R122封存gsh梯度上重放既有tokens，测条件是否触发及投影实际变化，不读query标签、不改模型、不把诊断当作新准确率结果。零tokens触发投影但方向不变，必须区别触发次数和非零变化次数。

公式：对参数组$ T $、梯度$ G $、方向$ D $，若绝对余弦最大值小于$ 0.1/\sqrt{d} $，令$ U=T/(\|T\|+10^{-8}) $，输出$ D-U\langle U,D\rangle $。依作者顺序选择第一个满足条件的组视图。

若完整轨迹从无非零投影，条件分支沿该轨迹退化为基线；不需要再重复相同准确率实验。若存在非零变化，下一步封存真实训练方案：原式/范数匹配/仅幅度/恒等控制，各使用其自己的梯度流，再比较原Adam和gsh。该重放只是原轨迹上的探针，不能代表介入后的梯度或性能。草案尚非正式开发实验。
