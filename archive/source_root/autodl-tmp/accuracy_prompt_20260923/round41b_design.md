# 第41轮补充：Cora强权重完整复核

固定GraphMAE checkpoint、旧100步强提示、k=10、alpha=.9、20步传播，不再调参。支持seed65100..65119，查询沿用confirm500，不能称独立测试。10方法共200条，见round41b_selection.json。主要比较为knn.9钳制减原提示；同时检验同alpha无钳制/随机重连、.5近邻、原图、Frozen同传播、普通GraphPrompt及教师。按同支持种子配对区间报告，不把随机图劣化当有效性的充分证据。训练和传播源码直接复用已审计run_round41_propagation.py。

依据上一轮Cora/GraphMAE次要对照+.70pp，先补齐强权重归因，再决定是否作为训练内一致性的基线。
