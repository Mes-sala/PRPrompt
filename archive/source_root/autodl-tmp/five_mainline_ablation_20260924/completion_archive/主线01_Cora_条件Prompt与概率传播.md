# 主线01消融结果

概率传播的数据集宏均增益为 +0.745 pp（38/63 checkpoint格为正），但在 CiteSeer、Actor、Chameleon 的均值为负。打乱门控对应后下降很大，完整对应的宏均增益 +18.070 pp；这支持节点与门控对应的重要性，不证明机制原创。动态CE缩放相对普通CE为 −0.633 pp，8个数据集均值全部为负，不能继续作为通用增强项。KD为 −0.155 pp，收益依数据集变化。保留条件Prompt和传播作为有条件方向，动态缩放应降级为历史对照。


口径：以下增量均为预定义完整分支减消融分支，单位为准确率百分点；总体采用8数据集等权宏均值。每个checkpoint内配对5个相同支持采样seed，再对checkpoint汇总。5个seed不是5张独立图，未宣称统计显著性或独立外部确认。主线03的tr1与gsh完整分支不同，不能混用消融分母。


详细配对表如下。


每项按同seed、同checkpoint配对。差值=完整分支−消融分支。正值是保留模块收益；并非统计显著性结论。

## 动态CE缩放

完整：`A1_scale2_qeps`；消融：`A1_token_qeps`。

| 数据集 | checkpoint | SHA前缀 | 配对增量均值±标准差pp |
|---|---|---|---:|
| Cornell | DGI | 7b79975af25a | +0.000 ± 0.000 |
| Cornell | Edgepred_GPPT | 72a6b63fbfbf | -0.541 ± 1.209 |
| Cornell | Edgepred_Gprompt | f656050f9497 | -0.541 ± 1.209 |
| Cornell | GraphCL | 2f6b499f430b | -0.541 ± 1.209 |
| Cornell | GraphMAE | efa244a8c256 | +0.541 ± 1.209 |
| Cornell | SimGRACE | 0a9e536877f7 | +0.000 ± 0.000 |
| Chameleon | DGI | 649de463352a | +0.965 ± 0.988 |
| Chameleon | Edgepred_GPPT | 4f7fee992773 | -0.570 ± 1.321 |
| Chameleon | Edgepred_Gprompt | 0f63e6a81dbc | +0.044 ± 1.056 |
| Chameleon | GraphCL | 07058ab2a2fa | -1.447 ± 1.839 |
| Chameleon | GraphMAE | fae19b2ebf82 | -2.237 ± 4.796 |
| Chameleon | SimGRACE | 525eec28fed1 | -0.263 ± 1.831 |
| Cora | Bandana | df27fae92ee3 | +0.100 ± 0.534 |
| Cora | DGI | fb2b361637cb | -0.560 ± 0.808 |
| Cora | Edgepred_GPPT | a279caf82ae2 | -3.060 ± 1.050 |
| Cora | Edgepred_Gprompt | 87bd546f7897 | -0.220 ± 0.335 |
| Cora | GraphCL | 3e74255f6fd2 | -1.200 ± 2.713 |
| Cora | GraphMAE | fb9bb0e2e2bd | +0.460 ± 0.329 |
| Cora | GraphMAE2 | b4ccffb19908 | +0.460 ± 0.532 |
| Cora | S2GAE | 5ed4965565a5 | -0.100 ± 0.992 |
| Cora | SimGRACE | 1e1fb8143d25 | -0.860 ± 3.108 |
| CiteSeer | Bandana | 8e2a1c825227 | +0.060 ± 0.329 |
| CiteSeer | DGI | 460f97d1cd97 | +0.820 ± 2.425 |
| CiteSeer | DGI | 8f4dd5622b98 | +0.700 ± 2.444 |
| CiteSeer | Edgepred_GPPT | 71a6af7c13c8 | -4.920 ± 1.883 |
| CiteSeer | Edgepred_GPPT | e5488f6ef537 | -4.600 ± 1.799 |
| CiteSeer | Edgepred_Gprompt | 5b3b70a23a78 | -0.480 ± 0.646 |
| CiteSeer | Edgepred_Gprompt | 7bf67dc55e3a | -0.380 ± 0.614 |
| CiteSeer | GraphCL | b4708f27c7e5 | -2.180 ± 4.144 |
| CiteSeer | GraphCL | ba86bc2a10e8 | -3.700 ± 6.640 |
| CiteSeer | GraphMAE | c95a7169f993 | +0.940 ± 0.740 |
| CiteSeer | GraphMAE | f4d07b34b0a6 | +0.940 ± 0.740 |
| CiteSeer | GraphMAE2 | cf4ad5bbee38 | +0.140 ± 0.541 |
| CiteSeer | S2GAE | bd209ec5181e | -1.660 ± 3.988 |
| CiteSeer | SimGRACE | 86e4c3c121d2 | -3.820 ± 2.962 |
| CiteSeer | SimGRACE | da8250d13cc2 | -3.900 ± 1.904 |
| Actor | DGI | 8ab886f3fc7e | +0.000 ± 0.123 |
| Actor | Edgepred_GPPT | ade64f6f3b9b | -0.421 ± 0.448 |
| Actor | Edgepred_Gprompt | 30b9c2c89a1d | +0.000 ± 0.596 |
| Actor | GraphCL | 73eda304d4ee | +0.145 ± 0.346 |
| Actor | GraphMAE | 39d2fb2110d1 | +0.039 ± 0.036 |
| Actor | SimGRACE | 049c20350cd6 | -0.145 ± 0.337 |
| Photo | DGI | a110375cbf8a | -0.171 ± 1.014 |
| Photo | Edgepred_GPPT | 021cfd4af878 | -0.385 ± 2.088 |
| Photo | Edgepred_Gprompt | 1c80d00b449e | -0.053 ± 0.138 |
| Photo | GraphCL | 28ba37961798 | -0.927 ± 0.663 |
| Photo | GraphMAE | cf3667a6a9c1 | -0.051 ± 0.172 |
| Photo | SimGRACE | 92300a0c4162 | -0.088 ± 0.724 |
| Computers | DGI | a5f0b042a971 | -0.460 ± 1.696 |
| Computers | Edgepred_GPPT | 37699b780525 | -0.674 ± 0.703 |
| Computers | Edgepred_Gprompt | fd9e216e5d41 | -0.162 ± 0.370 |
| Computers | GraphCL | f821af6f2440 | -1.952 ± 4.102 |
| Computers | GraphMAE | 61e493ff08ad | +0.527 ± 1.259 |
| Computers | SimGRACE | a4ae8c26f33c | -0.173 ± 0.263 |
| PubMed | Bandana | a4e198b2ea8d | -0.180 ± 0.148 |
| PubMed | DGI | 7be8f89a61ad | -1.020 ± 0.847 |
| PubMed | Edgepred_GPPT | 7b0bb774ae09 | -0.120 ± 0.370 |
| PubMed | Edgepred_Gprompt | fb61b312c4fc | -4.620 ± 5.807 |
| PubMed | GraphCL | dbcbde183804 | -1.820 ± 2.081 |
| PubMed | GraphMAE | 70d698a5a235 | -1.480 ± 2.119 |
| PubMed | GraphMAE2 | f33a0e8ff3a7 | -0.900 ± 1.231 |
| PubMed | S2GAE | dedf45129a74 | +0.220 ± 0.303 |
| PubMed | SimGRACE | 50beb94f14f3 | -3.120 ± 2.213 |

## 概率传播

完整：`A1_scale2_qeps`；消融：`A1_scale2_mean`。

| 数据集 | checkpoint | SHA前缀 | 配对增量均值±标准差pp |
|---|---|---|---:|
| Cornell | DGI | 7b79975af25a | +3.243 ± 4.441 |
| Cornell | Edgepred_GPPT | 72a6b63fbfbf | -2.162 ± 7.500 |
| Cornell | Edgepred_Gprompt | f656050f9497 | +5.405 ± 6.043 |
| Cornell | GraphCL | 2f6b499f430b | +9.730 ± 4.910 |
| Cornell | GraphMAE | efa244a8c256 | +5.405 ± 5.405 |
| Cornell | SimGRACE | 0a9e536877f7 | -1.081 ± 13.323 |
| Chameleon | DGI | 649de463352a | -2.281 ± 2.090 |
| Chameleon | Edgepred_GPPT | 4f7fee992773 | -4.123 ± 3.392 |
| Chameleon | Edgepred_Gprompt | 0f63e6a81dbc | +0.439 ± 1.974 |
| Chameleon | GraphCL | 07058ab2a2fa | -0.044 ± 1.530 |
| Chameleon | GraphMAE | fae19b2ebf82 | -1.009 ± 1.246 |
| Chameleon | SimGRACE | 525eec28fed1 | -0.482 ± 1.205 |
| Cora | Bandana | df27fae92ee3 | -0.120 ± 4.454 |
| Cora | DGI | fb2b361637cb | +0.060 ± 1.088 |
| Cora | Edgepred_GPPT | a279caf82ae2 | +1.240 ± 1.408 |
| Cora | Edgepred_Gprompt | 87bd546f7897 | +1.600 ± 1.678 |
| Cora | GraphCL | 3e74255f6fd2 | +2.100 ± 2.239 |
| Cora | GraphMAE | fb9bb0e2e2bd | +1.740 ± 0.488 |
| Cora | GraphMAE2 | b4ccffb19908 | +0.740 ± 1.004 |
| Cora | S2GAE | 5ed4965565a5 | +1.280 ± 2.350 |
| Cora | SimGRACE | 1e1fb8143d25 | +2.500 ± 0.875 |
| CiteSeer | Bandana | 8e2a1c825227 | +1.760 ± 1.534 |
| CiteSeer | DGI | 460f97d1cd97 | -0.780 ± 0.545 |
| CiteSeer | DGI | 8f4dd5622b98 | +0.060 ± 1.581 |
| CiteSeer | Edgepred_GPPT | 71a6af7c13c8 | +0.260 ± 0.811 |
| CiteSeer | Edgepred_GPPT | e5488f6ef537 | -0.000 ± 0.919 |
| CiteSeer | Edgepred_Gprompt | 5b3b70a23a78 | +0.120 ± 1.928 |
| CiteSeer | Edgepred_Gprompt | 7bf67dc55e3a | +0.920 ± 1.355 |
| CiteSeer | GraphCL | b4708f27c7e5 | -2.260 ± 0.817 |
| CiteSeer | GraphCL | ba86bc2a10e8 | -1.980 ± 2.128 |
| CiteSeer | GraphMAE | c95a7169f993 | +0.220 ± 1.244 |
| CiteSeer | GraphMAE | f4d07b34b0a6 | +0.220 ± 1.244 |
| CiteSeer | GraphMAE2 | cf4ad5bbee38 | +1.320 ± 2.232 |
| CiteSeer | S2GAE | bd209ec5181e | -1.120 ± 2.266 |
| CiteSeer | SimGRACE | 86e4c3c121d2 | +0.200 ± 1.010 |
| CiteSeer | SimGRACE | da8250d13cc2 | -1.440 ± 1.333 |
| Actor | DGI | 8ab886f3fc7e | -0.197 ± 1.670 |
| Actor | Edgepred_GPPT | ade64f6f3b9b | -2.079 ± 2.541 |
| Actor | Edgepred_Gprompt | 30b9c2c89a1d | -0.184 ± 0.907 |
| Actor | GraphCL | 73eda304d4ee | +0.092 ± 0.374 |
| Actor | GraphMAE | 39d2fb2110d1 | +0.237 ± 2.195 |
| Actor | SimGRACE | 049c20350cd6 | +0.092 ± 1.531 |
| Photo | DGI | a110375cbf8a | +0.075 ± 1.298 |
| Photo | Edgepred_GPPT | 021cfd4af878 | +0.288 ± 1.147 |
| Photo | Edgepred_Gprompt | 1c80d00b449e | -0.013 ± 1.521 |
| Photo | GraphCL | 28ba37961798 | +1.162 ± 0.785 |
| Photo | GraphMAE | cf3667a6a9c1 | +1.955 ± 2.358 |
| Photo | SimGRACE | 92300a0c4162 | +2.753 ± 3.362 |
| Computers | DGI | a5f0b042a971 | +1.523 ± 1.027 |
| Computers | Edgepred_GPPT | 37699b780525 | +0.702 ± 0.732 |
| Computers | Edgepred_Gprompt | fd9e216e5d41 | +1.377 ± 0.535 |
| Computers | GraphCL | f821af6f2440 | +3.430 ± 2.134 |
| Computers | GraphMAE | 61e493ff08ad | +1.817 ± 1.328 |
| Computers | SimGRACE | a4ae8c26f33c | +0.980 ± 1.202 |
| PubMed | Bandana | a4e198b2ea8d | +1.240 ± 1.991 |
| PubMed | DGI | 7be8f89a61ad | -0.020 ± 0.148 |
| PubMed | Edgepred_GPPT | 7b0bb774ae09 | -0.120 ± 0.228 |
| PubMed | Edgepred_Gprompt | fb61b312c4fc | -0.440 ± 0.573 |
| PubMed | GraphCL | dbcbde183804 | -0.700 ± 1.181 |
| PubMed | GraphMAE | 70d698a5a235 | -0.380 ± 1.761 |
| PubMed | GraphMAE2 | f33a0e8ff3a7 | -0.440 ± 1.240 |
| PubMed | S2GAE | dedf45129a74 | +4.400 ± 3.106 |
| PubMed | SimGRACE | 50beb94f14f3 | -0.100 ± 0.200 |

## Prompt整体

完整：`A1_scale2_qeps`；消融：`base_qeps`。

| 数据集 | checkpoint | SHA前缀 | 配对增量均值±标准差pp |
|---|---|---|---:|
| Cornell | DGI | 7b79975af25a | +5.405 ± 6.338 |
| Cornell | Edgepred_GPPT | 72a6b63fbfbf | +2.703 ± 6.620 |
| Cornell | Edgepred_Gprompt | f656050f9497 | -1.081 ± 5.921 |
| Cornell | GraphCL | 2f6b499f430b | +0.541 ± 1.209 |
| Cornell | GraphMAE | efa244a8c256 | +1.622 ± 2.417 |
| Cornell | SimGRACE | 0a9e536877f7 | -4.324 ± 12.177 |
| Chameleon | DGI | 649de463352a | -1.886 ± 5.857 |
| Chameleon | Edgepred_GPPT | 4f7fee992773 | +0.482 ± 2.123 |
| Chameleon | Edgepred_Gprompt | 0f63e6a81dbc | +0.965 ± 2.158 |
| Chameleon | GraphCL | 07058ab2a2fa | -5.439 ± 2.564 |
| Chameleon | GraphMAE | fae19b2ebf82 | -4.386 ± 4.510 |
| Chameleon | SimGRACE | 525eec28fed1 | +0.789 ± 5.577 |
| Cora | Bandana | df27fae92ee3 | +12.080 ± 7.686 |
| Cora | DGI | fb2b361637cb | -1.080 ± 6.050 |
| Cora | Edgepred_GPPT | a279caf82ae2 | +32.840 ± 2.641 |
| Cora | Edgepred_Gprompt | 87bd546f7897 | +41.520 ± 6.276 |
| Cora | GraphCL | 3e74255f6fd2 | +15.840 ± 3.606 |
| Cora | GraphMAE | fb9bb0e2e2bd | +11.960 ± 3.397 |
| Cora | GraphMAE2 | b4ccffb19908 | +35.380 ± 13.955 |
| Cora | S2GAE | 5ed4965565a5 | +10.100 ± 10.743 |
| Cora | SimGRACE | 1e1fb8143d25 | +41.960 ± 1.919 |
| CiteSeer | Bandana | 8e2a1c825227 | +3.560 ± 2.260 |
| CiteSeer | DGI | 460f97d1cd97 | -19.620 ± 6.751 |
| CiteSeer | DGI | 8f4dd5622b98 | -19.980 ± 6.657 |
| CiteSeer | Edgepred_GPPT | 71a6af7c13c8 | +21.800 ± 3.050 |
| CiteSeer | Edgepred_GPPT | e5488f6ef537 | +20.220 ± 3.813 |
| CiteSeer | Edgepred_Gprompt | 5b3b70a23a78 | +34.100 ± 3.649 |
| CiteSeer | Edgepred_Gprompt | 7bf67dc55e3a | +34.660 ± 3.802 |
| CiteSeer | GraphCL | b4708f27c7e5 | +6.960 ± 5.204 |
| CiteSeer | GraphCL | ba86bc2a10e8 | +9.360 ± 5.692 |
| CiteSeer | GraphMAE | c95a7169f993 | +31.640 ± 3.294 |
| CiteSeer | GraphMAE | f4d07b34b0a6 | +31.660 ± 3.313 |
| CiteSeer | GraphMAE2 | cf4ad5bbee38 | +33.980 ± 9.243 |
| CiteSeer | S2GAE | bd209ec5181e | +16.400 ± 6.903 |
| CiteSeer | SimGRACE | 86e4c3c121d2 | +19.480 ± 5.664 |
| CiteSeer | SimGRACE | da8250d13cc2 | +12.300 ± 5.885 |
| Actor | DGI | 8ab886f3fc7e | +4.079 ± 4.339 |
| Actor | Edgepred_GPPT | ade64f6f3b9b | +0.171 ± 1.734 |
| Actor | Edgepred_Gprompt | 30b9c2c89a1d | +0.987 ± 0.756 |
| Actor | GraphCL | 73eda304d4ee | -1.289 ± 1.208 |
| Actor | GraphMAE | 39d2fb2110d1 | +4.158 ± 5.530 |
| Actor | SimGRACE | 049c20350cd6 | +0.184 ± 1.583 |
| Photo | DGI | a110375cbf8a | +31.458 ± 7.518 |
| Photo | Edgepred_GPPT | 021cfd4af878 | +18.203 ± 13.267 |
| Photo | Edgepred_Gprompt | 1c80d00b449e | +15.503 ± 6.404 |
| Photo | GraphCL | 28ba37961798 | +25.864 ± 8.822 |
| Photo | GraphMAE | cf3667a6a9c1 | +37.068 ± 6.086 |
| Photo | SimGRACE | 92300a0c4162 | +30.312 ± 6.890 |
| Computers | DGI | a5f0b042a971 | +25.167 ± 9.330 |
| Computers | Edgepred_GPPT | 37699b780525 | +17.389 ± 4.116 |
| Computers | Edgepred_Gprompt | fd9e216e5d41 | +20.273 ± 6.814 |
| Computers | GraphCL | f821af6f2440 | +18.312 ± 4.625 |
| Computers | GraphMAE | 61e493ff08ad | +34.280 ± 6.846 |
| Computers | SimGRACE | a4ae8c26f33c | +21.538 ± 9.681 |
| PubMed | Bandana | a4e198b2ea8d | +28.200 ± 2.162 |
| PubMed | DGI | 7be8f89a61ad | +21.920 ± 6.616 |
| PubMed | Edgepred_GPPT | 7b0bb774ae09 | +5.640 ± 5.183 |
| PubMed | Edgepred_Gprompt | fb61b312c4fc | +0.740 ± 7.142 |
| PubMed | GraphCL | dbcbde183804 | +15.480 ± 4.782 |
| PubMed | GraphMAE | 70d698a5a235 | -2.480 ± 6.307 |
| PubMed | GraphMAE2 | f33a0e8ff3a7 | -0.660 ± 4.799 |
| PubMed | S2GAE | dedf45129a74 | +18.680 ± 4.063 |
| PubMed | SimGRACE | 50beb94f14f3 | +18.020 ± 5.226 |

## KD

完整：`A1_scale2_qeps`；消融：`A1_nokd`。

| 数据集 | checkpoint | SHA前缀 | 配对增量均值±标准差pp |
|---|---|---|---:|
| Cornell | DGI | 7b79975af25a | +0.541 ± 2.261 |
| Cornell | Edgepred_GPPT | 72a6b63fbfbf | -2.162 ± 3.524 |
| Cornell | Edgepred_Gprompt | f656050f9497 | +2.162 ± 6.453 |
| Cornell | GraphCL | 2f6b499f430b | -1.081 ± 3.626 |
| Cornell | GraphMAE | efa244a8c256 | +0.541 ± 1.209 |
| Cornell | SimGRACE | 0a9e536877f7 | +1.081 ± 9.669 |
| Chameleon | DGI | 649de463352a | -2.018 ± 7.637 |
| Chameleon | Edgepred_GPPT | 4f7fee992773 | -0.833 ± 2.265 |
| Chameleon | Edgepred_Gprompt | 0f63e6a81dbc | +1.096 ± 2.417 |
| Chameleon | GraphCL | 07058ab2a2fa | -3.904 ± 2.307 |
| Chameleon | GraphMAE | fae19b2ebf82 | -4.167 ± 4.938 |
| Chameleon | SimGRACE | 525eec28fed1 | -3.772 ± 2.643 |
| Cora | Bandana | df27fae92ee3 | -2.400 ± 5.554 |
| Cora | DGI | fb2b361637cb | +5.020 ± 6.111 |
| Cora | Edgepred_GPPT | a279caf82ae2 | -2.240 ± 1.484 |
| Cora | Edgepred_Gprompt | 87bd546f7897 | +4.860 ± 2.365 |
| Cora | GraphCL | 3e74255f6fd2 | +0.820 ± 3.557 |
| Cora | GraphMAE | fb9bb0e2e2bd | +1.960 ± 1.466 |
| Cora | GraphMAE2 | b4ccffb19908 | +3.820 ± 1.675 |
| Cora | S2GAE | 5ed4965565a5 | -5.100 ± 6.192 |
| Cora | SimGRACE | 1e1fb8143d25 | +3.360 ± 4.340 |
| CiteSeer | Bandana | 8e2a1c825227 | -0.600 ± 2.596 |
| CiteSeer | DGI | 460f97d1cd97 | -2.000 ± 3.465 |
| CiteSeer | DGI | 8f4dd5622b98 | -2.260 ± 3.334 |
| CiteSeer | Edgepred_GPPT | 71a6af7c13c8 | -4.060 ± 4.315 |
| CiteSeer | Edgepred_GPPT | e5488f6ef537 | -3.500 ± 2.357 |
| CiteSeer | Edgepred_Gprompt | 5b3b70a23a78 | +5.920 ± 5.023 |
| CiteSeer | Edgepred_Gprompt | 7bf67dc55e3a | +6.560 ± 2.980 |
| CiteSeer | GraphCL | b4708f27c7e5 | +0.960 ± 5.909 |
| CiteSeer | GraphCL | ba86bc2a10e8 | -4.600 ± 7.001 |
| CiteSeer | GraphMAE | c95a7169f993 | +1.040 ± 2.618 |
| CiteSeer | GraphMAE | f4d07b34b0a6 | +1.020 ± 2.639 |
| CiteSeer | GraphMAE2 | cf4ad5bbee38 | +2.220 ± 1.916 |
| CiteSeer | S2GAE | bd209ec5181e | -0.460 ± 3.956 |
| CiteSeer | SimGRACE | 86e4c3c121d2 | -3.620 ± 3.736 |
| CiteSeer | SimGRACE | da8250d13cc2 | -4.720 ± 3.704 |
| Actor | DGI | 8ab886f3fc7e | +0.987 ± 0.633 |
| Actor | Edgepred_GPPT | ade64f6f3b9b | -0.461 ± 0.713 |
| Actor | Edgepred_Gprompt | 30b9c2c89a1d | -0.263 ± 0.543 |
| Actor | GraphCL | 73eda304d4ee | -0.605 ± 0.584 |
| Actor | GraphMAE | 39d2fb2110d1 | +0.092 ± 0.136 |
| Actor | SimGRACE | 049c20350cd6 | -0.421 ± 0.511 |
| Photo | DGI | a110375cbf8a | +1.933 ± 5.524 |
| Photo | Edgepred_GPPT | 021cfd4af878 | -0.326 ± 7.658 |
| Photo | Edgepred_Gprompt | 1c80d00b449e | -0.032 ± 0.542 |
| Photo | GraphCL | 28ba37961798 | -0.921 ± 5.374 |
| Photo | GraphMAE | cf3667a6a9c1 | +1.057 ± 0.673 |
| Photo | SimGRACE | 92300a0c4162 | +2.019 ± 4.462 |
| Computers | DGI | a5f0b042a971 | +0.818 ± 6.274 |
| Computers | Edgepred_GPPT | 37699b780525 | +6.563 ± 6.671 |
| Computers | Edgepred_Gprompt | fd9e216e5d41 | +0.505 ± 1.606 |
| Computers | GraphCL | f821af6f2440 | -1.460 ± 5.359 |
| Computers | GraphMAE | 61e493ff08ad | +0.308 ± 3.414 |
| Computers | SimGRACE | a4ae8c26f33c | +2.226 ± 5.248 |
| PubMed | Bandana | a4e198b2ea8d | +1.700 ± 1.425 |
| PubMed | DGI | 7be8f89a61ad | -1.800 ± 3.240 |
| PubMed | Edgepred_GPPT | 7b0bb774ae09 | +0.500 ± 1.342 |
| PubMed | Edgepred_Gprompt | fb61b312c4fc | -6.900 ± 7.322 |
| PubMed | GraphCL | dbcbde183804 | -1.580 ± 3.214 |
| PubMed | GraphMAE | 70d698a5a235 | -2.420 ± 4.511 |
| PubMed | GraphMAE2 | f33a0e8ff3a7 | -2.420 ± 3.469 |
| PubMed | S2GAE | dedf45129a74 | +0.440 ± 0.586 |
| PubMed | SimGRACE | 50beb94f14f3 | -3.160 ± 4.057 |

## 语义门控对应

完整：`A1_scale2_qeps`；消融：`A1_permgate`。

| 数据集 | checkpoint | SHA前缀 | 配对增量均值±标准差pp |
|---|---|---|---:|
| Cornell | DGI | 7b79975af25a | +9.189 ± 9.479 |
| Cornell | Edgepred_GPPT | 72a6b63fbfbf | +1.622 ± 9.479 |
| Cornell | Edgepred_Gprompt | f656050f9497 | +0.541 ± 8.198 |
| Cornell | GraphCL | 2f6b499f430b | +3.243 ± 4.835 |
| Cornell | GraphMAE | efa244a8c256 | +2.162 ± 5.859 |
| Cornell | SimGRACE | 0a9e536877f7 | -3.243 ± 9.045 |
| Chameleon | DGI | 649de463352a | -0.965 ± 6.595 |
| Chameleon | Edgepred_GPPT | 4f7fee992773 | +1.272 ± 1.373 |
| Chameleon | Edgepred_Gprompt | 0f63e6a81dbc | +1.974 ± 1.413 |
| Chameleon | GraphCL | 07058ab2a2fa | -1.184 ± 3.832 |
| Chameleon | GraphMAE | fae19b2ebf82 | -3.070 ± 3.643 |
| Chameleon | SimGRACE | 525eec28fed1 | +1.535 ± 5.162 |
| Cora | Bandana | df27fae92ee3 | +9.620 ± 7.165 |
| Cora | DGI | fb2b361637cb | +33.420 ± 4.743 |
| Cora | Edgepred_GPPT | a279caf82ae2 | +37.420 ± 4.034 |
| Cora | Edgepred_Gprompt | 87bd546f7897 | +49.000 ± 4.577 |
| Cora | GraphCL | 3e74255f6fd2 | +18.780 ± 3.751 |
| Cora | GraphMAE | fb9bb0e2e2bd | +27.080 ± 3.163 |
| Cora | GraphMAE2 | b4ccffb19908 | +53.360 ± 3.064 |
| Cora | S2GAE | 5ed4965565a5 | +11.240 ± 9.795 |
| Cora | SimGRACE | 1e1fb8143d25 | +45.940 ± 3.551 |
| CiteSeer | Bandana | 8e2a1c825227 | +5.360 ± 2.205 |
| CiteSeer | DGI | 460f97d1cd97 | +10.220 ± 3.898 |
| CiteSeer | DGI | 8f4dd5622b98 | +9.980 ± 3.893 |
| CiteSeer | Edgepred_GPPT | 71a6af7c13c8 | +23.420 ± 3.339 |
| CiteSeer | Edgepred_GPPT | e5488f6ef537 | +22.620 ± 3.784 |
| CiteSeer | Edgepred_Gprompt | 5b3b70a23a78 | +35.880 ± 4.068 |
| CiteSeer | Edgepred_Gprompt | 7bf67dc55e3a | +36.640 ± 2.901 |
| CiteSeer | GraphCL | b4708f27c7e5 | +6.360 ± 6.219 |
| CiteSeer | GraphCL | ba86bc2a10e8 | +10.920 ± 5.704 |
| CiteSeer | GraphMAE | c95a7169f993 | +37.960 ± 4.817 |
| CiteSeer | GraphMAE | f4d07b34b0a6 | +37.960 ± 4.817 |
| CiteSeer | GraphMAE2 | cf4ad5bbee38 | +40.480 ± 4.330 |
| CiteSeer | S2GAE | bd209ec5181e | +17.860 ± 5.054 |
| CiteSeer | SimGRACE | 86e4c3c121d2 | +21.140 ± 4.275 |
| CiteSeer | SimGRACE | da8250d13cc2 | +13.040 ± 6.590 |
| Actor | DGI | 8ab886f3fc7e | +1.408 ± 2.810 |
| Actor | Edgepred_GPPT | ade64f6f3b9b | +0.184 ± 1.897 |
| Actor | Edgepred_Gprompt | 30b9c2c89a1d | +0.579 ± 1.465 |
| Actor | GraphCL | 73eda304d4ee | -0.303 ± 1.935 |
| Actor | GraphMAE | 39d2fb2110d1 | +3.382 ± 5.834 |
| Actor | SimGRACE | 049c20350cd6 | +0.513 ± 1.741 |
| Photo | DGI | a110375cbf8a | +51.907 ± 7.202 |
| Photo | Edgepred_GPPT | 021cfd4af878 | +33.559 ± 5.665 |
| Photo | Edgepred_Gprompt | 1c80d00b449e | +42.983 ± 5.676 |
| Photo | GraphCL | 28ba37961798 | +30.043 ± 9.991 |
| Photo | GraphMAE | cf3667a6a9c1 | +62.569 ± 4.333 |
| Photo | SimGRACE | 92300a0c4162 | +30.427 ± 7.890 |
| Computers | DGI | a5f0b042a971 | +39.386 ± 6.004 |
| Computers | Edgepred_GPPT | 37699b780525 | +22.766 ± 8.498 |
| Computers | Edgepred_Gprompt | fd9e216e5d41 | +36.151 ± 7.877 |
| Computers | GraphCL | f821af6f2440 | +20.844 ± 5.568 |
| Computers | GraphMAE | 61e493ff08ad | +49.308 ± 10.357 |
| Computers | SimGRACE | a4ae8c26f33c | +22.519 ± 7.690 |
| PubMed | Bandana | a4e198b2ea8d | +23.100 ± 4.666 |
| PubMed | DGI | 7be8f89a61ad | +30.080 ± 5.493 |
| PubMed | Edgepred_GPPT | 7b0bb774ae09 | +16.560 ± 5.677 |
| PubMed | Edgepred_Gprompt | fb61b312c4fc | +1.280 ± 8.014 |
| PubMed | GraphCL | dbcbde183804 | +15.160 ± 1.931 |
| PubMed | GraphMAE | 70d698a5a235 | -3.020 ± 4.847 |
| PubMed | GraphMAE2 | f33a0e8ff3a7 | +2.860 ± 4.612 |
| PubMed | S2GAE | dedf45129a74 | +21.560 ± 3.367 |
| PubMed | SimGRACE | 50beb94f14f3 | +17.780 ± 4.013 |

## 动态CE×概率传播的2×2交互

定义：`A1_scale2_qeps - A1_scale2_mean - A1_token_qeps + A1_token_mean`。描述性交互，正值不代表已证明普遍协同。

| 数据集 | checkpoint | SHA前缀 | 交互pp |
|---|---|---|---:|
| Cornell | DGI | 7b79975af25a | +0.000 |
| Cornell | Edgepred_GPPT | 72a6b63fbfbf | -0.541 |
| Cornell | Edgepred_Gprompt | f656050f9497 | +0.000 |
| Cornell | GraphCL | 2f6b499f430b | -0.541 |
| Cornell | GraphMAE | efa244a8c256 | +0.541 |
| Cornell | SimGRACE | 0a9e536877f7 | -1.622 |
| Chameleon | DGI | 649de463352a | -0.088 |
| Chameleon | Edgepred_GPPT | 4f7fee992773 | -0.614 |
| Chameleon | Edgepred_Gprompt | 0f63e6a81dbc | +0.965 |
| Chameleon | GraphCL | 07058ab2a2fa | -0.044 |
| Chameleon | GraphMAE | fae19b2ebf82 | -0.088 |
| Chameleon | SimGRACE | 525eec28fed1 | -0.219 |
| Cora | Bandana | df27fae92ee3 | -0.020 |
| Cora | DGI | fb2b361637cb | +0.060 |
| Cora | Edgepred_GPPT | a279caf82ae2 | +0.160 |
| Cora | Edgepred_Gprompt | 87bd546f7897 | -0.140 |
| Cora | GraphCL | 3e74255f6fd2 | -0.380 |
| Cora | GraphMAE | fb9bb0e2e2bd | +0.340 |
| Cora | GraphMAE2 | b4ccffb19908 | +0.040 |
| Cora | S2GAE | 5ed4965565a5 | -0.520 |
| Cora | SimGRACE | 1e1fb8143d25 | +0.400 |
| CiteSeer | Bandana | 8e2a1c825227 | -0.060 |
| CiteSeer | DGI | 460f97d1cd97 | +0.040 |
| CiteSeer | DGI | 8f4dd5622b98 | -0.160 |
| CiteSeer | Edgepred_GPPT | 71a6af7c13c8 | +0.160 |
| CiteSeer | Edgepred_GPPT | e5488f6ef537 | -0.420 |
| CiteSeer | Edgepred_Gprompt | 5b3b70a23a78 | -0.140 |
| CiteSeer | Edgepred_Gprompt | 7bf67dc55e3a | -0.140 |
| CiteSeer | GraphCL | b4708f27c7e5 | -0.240 |
| CiteSeer | GraphCL | ba86bc2a10e8 | -1.200 |
| CiteSeer | GraphMAE | c95a7169f993 | -0.160 |
| CiteSeer | GraphMAE | f4d07b34b0a6 | -0.160 |
| CiteSeer | GraphMAE2 | cf4ad5bbee38 | -0.360 |
| CiteSeer | S2GAE | bd209ec5181e | -0.680 |
| CiteSeer | SimGRACE | 86e4c3c121d2 | -0.480 |
| CiteSeer | SimGRACE | da8250d13cc2 | -1.800 |
| Actor | DGI | 8ab886f3fc7e | -0.276 |
| Actor | Edgepred_GPPT | ade64f6f3b9b | -0.184 |
| Actor | Edgepred_Gprompt | 30b9c2c89a1d | -0.342 |
| Actor | GraphCL | 73eda304d4ee | -0.026 |
| Actor | GraphMAE | 39d2fb2110d1 | +0.000 |
| Actor | SimGRACE | 049c20350cd6 | -0.118 |
| Photo | DGI | a110375cbf8a | -0.168 |
| Photo | Edgepred_GPPT | 021cfd4af878 | -0.043 |
| Photo | Edgepred_Gprompt | 1c80d00b449e | -0.056 |
| Photo | GraphCL | 28ba37961798 | -0.043 |
| Photo | GraphMAE | cf3667a6a9c1 | -0.016 |
| Photo | SimGRACE | 92300a0c4162 | +0.545 |
| Computers | DGI | a5f0b042a971 | +0.378 |
| Computers | Edgepred_GPPT | 37699b780525 | +0.339 |
| Computers | Edgepred_Gprompt | fd9e216e5d41 | +0.047 |
| Computers | GraphCL | f821af6f2440 | +0.087 |
| Computers | GraphMAE | 61e493ff08ad | +0.081 |
| Computers | SimGRACE | a4ae8c26f33c | -0.111 |
| PubMed | Bandana | a4e198b2ea8d | -0.080 |
| PubMed | DGI | 7be8f89a61ad | -0.040 |
| PubMed | Edgepred_GPPT | 7b0bb774ae09 | -0.120 |
| PubMed | Edgepred_Gprompt | fb61b312c4fc | -0.420 |
| PubMed | GraphCL | dbcbde183804 | -0.400 |
| PubMed | GraphMAE | 70d698a5a235 | +0.120 |
| PubMed | GraphMAE2 | f33a0e8ff3a7 | -0.340 |
| PubMed | S2GAE | dedf45129a74 | +0.200 |
| PubMed | SimGRACE | 50beb94f14f3 | -0.180 |
scale2是gamma=2的focal梯度系数均值动态缩放CE，绝非固定乘2。
