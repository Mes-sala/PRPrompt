# 历史 v1 F100 固定配方

复用保存的历史 AST 原函数；两层 GCN 冻结，5-shot，100 步，零 token 初始化。
M1 温度 0.1、幅度 10、sample std；普通 Adam lr=0.01、eps=1e-8、weight_decay=0。
使用留一支持原型训练和最终中心化原型读出，无 M2、M3、KD 或分类头。
拟合只接收支持标签；查询预测封存后评分。历史函数体在 `historical.py` 与 `reference/` 中保持原样。
