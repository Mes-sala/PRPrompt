# 第十八轮：固定方案跨九种预训练权重

此前借鉴[VPT](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4175_ECCV_2022_paper.php)的内部提示位置、R2D2岭教师和普通邻域平滑，已在GraphMAE上有配对收益。本轮不再增添损失，固定方案并寻找更合适的已有预训练权重，不是预训练集成。

Cora/CiteSeer/PubMed × DGI、GraphCL、GraphMAE、GraphMAE2、Bandana、S2GAE、SimGRACE、Edgepred_GPPT、Edgepred_Gprompt × 5支持seed25100..25104 × 8方法 =1080正式指标。方法：Frozen、GraphPrompt、岭教师、平滑岭教师、早层岭token10、后层岭token10、原型教师token3、平滑岭后层token10。100步Adam0.01，沿用同一split/原型/无标签边与特征，所有checkpoint同批支持。

模型结构与权重sidecar哈希已对27个文件核验。仍标ProG-adapted及source_parity实际内容，不宣称严格复现上游。MultiGprompt不是同一两层GCN架构，不混入本矩阵；不代表其无效。每个模型单独冻结，Prompt只改变该主干内部，单分类头。

先27模型各1支持×8方法×3步预检，检测shape、no-op、冻结与数据入口；预检不纳入正式指标。固定正式5支持后仅用于候选筛选，最多每图选一个新权重/方案，再与GraphMAE/GraphMAE2强参照及直接教师做20新支持确认。严禁按开发最高准确率直接声称新最佳。无需复制checkpoint、数据或环境。
