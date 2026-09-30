# 主线03消融结果

锐化门控宏均 +1.271 pp（42/63格为正），7个数据集均值为正、Cornell为零。Adam步长匹配相对无匹配矩阵更新 +11.718 pp，说明这套矩阵更新对更新尺度高度敏感；不能把这一差值全部解释为矩阵预条件的贡献。完整矩阵相对对角仅 −0.013 pp，25正/1平/37负；匹配Shampoo相对Adam +0.454 pp。输出中心化 −1.726 pp，仅Actor均值为正。保留锐化和匹配步长；完整矩阵必要性证据不足，中心化不宜默认打开。


口径：以下增量均为预定义完整分支减消融分支，单位为准确率百分点；总体采用8数据集等权宏均值。每个checkpoint内配对5个相同支持采样seed，再对checkpoint汇总。5个seed不是5张独立图，未宣称统计显著性或独立外部确认。主线03的tr1与gsh完整分支不同，不能混用消融分母。


详细配对表如下。


每项按同seed、同checkpoint配对。差值=完整分支−消融分支。正值是保留模块收益；并非统计显著性结论。

## 门控锐化

完整：`A3_tr1_qeps`；消融：`A3_temp02`。

| 数据集 | checkpoint | SHA前缀 | 配对增量均值±标准差pp |
|---|---|---|---:|
| Cornell | DGI | 7b79975af25a | -0.541 ± 3.524 |
| Cornell | Edgepred_GPPT | 72a6b63fbfbf | -1.081 ± 1.480 |
| Cornell | Edgepred_Gprompt | f656050f9497 | -1.622 ± 4.099 |
| Cornell | GraphCL | 2f6b499f430b | -1.081 ± 3.626 |
| Cornell | GraphMAE | efa244a8c256 | +2.162 ± 5.539 |
| Cornell | SimGRACE | 0a9e536877f7 | +2.162 ± 11.687 |
| Chameleon | DGI | 649de463352a | +0.614 ± 4.514 |
| Chameleon | Edgepred_GPPT | 4f7fee992773 | +0.088 ± 2.455 |
| Chameleon | Edgepred_Gprompt | 0f63e6a81dbc | +0.439 ± 1.463 |
| Chameleon | GraphCL | 07058ab2a2fa | -0.263 ± 2.674 |
| Chameleon | GraphMAE | fae19b2ebf82 | +0.965 ± 1.468 |
| Chameleon | SimGRACE | 525eec28fed1 | -1.491 ± 1.994 |
| Cora | Bandana | df27fae92ee3 | +2.040 ± 5.948 |
| Cora | DGI | fb2b361637cb | +4.680 ± 1.305 |
| Cora | Edgepred_GPPT | a279caf82ae2 | +2.940 ± 0.820 |
| Cora | Edgepred_Gprompt | 87bd546f7897 | +0.440 ± 0.767 |
| Cora | GraphCL | 3e74255f6fd2 | +2.460 ± 1.193 |
| Cora | GraphMAE | fb9bb0e2e2bd | -0.640 ± 0.456 |
| Cora | GraphMAE2 | b4ccffb19908 | -0.260 ± 0.270 |
| Cora | S2GAE | 5ed4965565a5 | +4.720 ± 11.537 |
| Cora | SimGRACE | 1e1fb8143d25 | +3.560 ± 1.641 |
| CiteSeer | Bandana | 8e2a1c825227 | +6.040 ± 5.474 |
| CiteSeer | DGI | 460f97d1cd97 | +3.280 ± 3.935 |
| CiteSeer | DGI | 8f4dd5622b98 | +3.100 ± 3.787 |
| CiteSeer | Edgepred_GPPT | 71a6af7c13c8 | +6.140 ± 2.212 |
| CiteSeer | Edgepred_GPPT | e5488f6ef537 | +4.780 ± 2.117 |
| CiteSeer | Edgepred_Gprompt | 5b3b70a23a78 | +3.700 ± 2.753 |
| CiteSeer | Edgepred_Gprompt | 7bf67dc55e3a | +3.480 ± 1.073 |
| CiteSeer | GraphCL | b4708f27c7e5 | +8.400 ± 4.439 |
| CiteSeer | GraphCL | ba86bc2a10e8 | +8.340 ± 4.353 |
| CiteSeer | GraphMAE | c95a7169f993 | -0.600 ± 1.838 |
| CiteSeer | GraphMAE | f4d07b34b0a6 | -0.580 ± 1.802 |
| CiteSeer | GraphMAE2 | cf4ad5bbee38 | +0.160 ± 1.262 |
| CiteSeer | S2GAE | bd209ec5181e | +7.580 ± 3.772 |
| CiteSeer | SimGRACE | 86e4c3c121d2 | +4.440 ± 1.893 |
| CiteSeer | SimGRACE | da8250d13cc2 | +6.960 ± 2.414 |
| Actor | DGI | 8ab886f3fc7e | -0.263 ± 1.236 |
| Actor | Edgepred_GPPT | ade64f6f3b9b | -0.711 ± 1.127 |
| Actor | Edgepred_Gprompt | 30b9c2c89a1d | +0.211 ± 0.952 |
| Actor | GraphCL | 73eda304d4ee | +0.763 ± 1.699 |
| Actor | GraphMAE | 39d2fb2110d1 | +0.461 ± 0.512 |
| Actor | SimGRACE | 049c20350cd6 | -0.263 ± 1.172 |
| Photo | DGI | a110375cbf8a | +1.247 ± 2.992 |
| Photo | Edgepred_GPPT | 021cfd4af878 | -0.996 ± 1.977 |
| Photo | Edgepred_Gprompt | 1c80d00b449e | -1.279 ± 2.559 |
| Photo | GraphCL | 28ba37961798 | +2.187 ± 0.989 |
| Photo | GraphMAE | cf3667a6a9c1 | +2.630 ± 3.817 |
| Photo | SimGRACE | 92300a0c4162 | +2.748 ± 5.031 |
| Computers | DGI | a5f0b042a971 | +5.828 ± 3.608 |
| Computers | Edgepred_GPPT | 37699b780525 | -0.040 ± 6.607 |
| Computers | Edgepred_Gprompt | fd9e216e5d41 | +1.606 ± 1.309 |
| Computers | GraphCL | f821af6f2440 | +2.102 ± 5.219 |
| Computers | GraphMAE | 61e493ff08ad | +2.875 ± 1.273 |
| Computers | SimGRACE | a4ae8c26f33c | -1.811 ± 4.538 |
| PubMed | Bandana | a4e198b2ea8d | +3.180 ± 1.763 |
| PubMed | DGI | 7be8f89a61ad | +4.320 ± 7.166 |
| PubMed | Edgepred_GPPT | 7b0bb774ae09 | +0.320 ± 0.554 |
| PubMed | Edgepred_Gprompt | fb61b312c4fc | -0.860 ± 1.150 |
| PubMed | GraphCL | dbcbde183804 | -0.640 ± 1.246 |
| PubMed | GraphMAE | 70d698a5a235 | -0.740 ± 0.913 |
| PubMed | GraphMAE2 | f33a0e8ff3a7 | -0.340 ± 1.074 |
| PubMed | S2GAE | dedf45129a74 | +0.040 ± 3.210 |
| PubMed | SimGRACE | 50beb94f14f3 | +0.700 ± 0.857 |

## 输出中心化

完整：`A3_tr1_qeps`；消融：`A3_nocenter`。

| 数据集 | checkpoint | SHA前缀 | 配对增量均值±标准差pp |
|---|---|---|---:|
| Cornell | DGI | 7b79975af25a | +2.703 ± 7.151 |
| Cornell | Edgepred_GPPT | 72a6b63fbfbf | -1.622 ± 5.604 |
| Cornell | Edgepred_Gprompt | f656050f9497 | -2.703 ± 4.273 |
| Cornell | GraphCL | 2f6b499f430b | -8.108 ± 5.405 |
| Cornell | GraphMAE | efa244a8c256 | -2.703 ± 6.891 |
| Cornell | SimGRACE | 0a9e536877f7 | +1.081 ± 8.461 |
| Chameleon | DGI | 649de463352a | -2.018 ± 3.507 |
| Chameleon | Edgepred_GPPT | 4f7fee992773 | -0.307 ± 1.983 |
| Chameleon | Edgepred_Gprompt | 0f63e6a81dbc | -0.132 ± 3.449 |
| Chameleon | GraphCL | 07058ab2a2fa | +0.526 ± 1.812 |
| Chameleon | GraphMAE | fae19b2ebf82 | -0.351 ± 1.645 |
| Chameleon | SimGRACE | 525eec28fed1 | -0.965 ± 3.813 |
| Cora | Bandana | df27fae92ee3 | +6.940 ± 7.757 |
| Cora | DGI | fb2b361637cb | -16.180 ± 5.965 |
| Cora | Edgepred_GPPT | a279caf82ae2 | -0.780 ± 0.502 |
| Cora | Edgepred_Gprompt | 87bd546f7897 | -0.700 ± 1.475 |
| Cora | GraphCL | 3e74255f6fd2 | -0.160 ± 1.001 |
| Cora | GraphMAE | fb9bb0e2e2bd | -0.420 ± 0.642 |
| Cora | GraphMAE2 | b4ccffb19908 | -0.840 ± 0.976 |
| Cora | S2GAE | 5ed4965565a5 | +9.980 ± 12.944 |
| Cora | SimGRACE | 1e1fb8143d25 | -0.060 ± 0.365 |
| CiteSeer | Bandana | 8e2a1c825227 | +2.320 ± 4.411 |
| CiteSeer | DGI | 460f97d1cd97 | -7.920 ± 7.071 |
| CiteSeer | DGI | 8f4dd5622b98 | -7.260 ± 7.881 |
| CiteSeer | Edgepred_GPPT | 71a6af7c13c8 | -0.100 ± 0.255 |
| CiteSeer | Edgepred_GPPT | e5488f6ef537 | +0.240 ± 0.493 |
| CiteSeer | Edgepred_Gprompt | 5b3b70a23a78 | +3.620 ± 4.387 |
| CiteSeer | Edgepred_Gprompt | 7bf67dc55e3a | +2.720 ± 2.580 |
| CiteSeer | GraphCL | b4708f27c7e5 | +4.800 ± 1.005 |
| CiteSeer | GraphCL | ba86bc2a10e8 | -1.000 ± 0.791 |
| CiteSeer | GraphMAE | c95a7169f993 | +0.780 ± 0.750 |
| CiteSeer | GraphMAE | f4d07b34b0a6 | +0.800 ± 0.787 |
| CiteSeer | GraphMAE2 | cf4ad5bbee38 | +0.240 ± 1.262 |
| CiteSeer | S2GAE | bd209ec5181e | -1.060 ± 0.885 |
| CiteSeer | SimGRACE | 86e4c3c121d2 | -0.200 ± 0.711 |
| CiteSeer | SimGRACE | da8250d13cc2 | +0.060 ± 0.532 |
| Actor | DGI | 8ab886f3fc7e | +1.408 ± 1.987 |
| Actor | Edgepred_GPPT | ade64f6f3b9b | +1.368 ± 1.376 |
| Actor | Edgepred_Gprompt | 30b9c2c89a1d | -0.303 ± 0.768 |
| Actor | GraphCL | 73eda304d4ee | +0.539 ± 0.821 |
| Actor | GraphMAE | 39d2fb2110d1 | +1.645 ± 1.688 |
| Actor | SimGRACE | 049c20350cd6 | +0.724 ± 1.484 |
| Photo | DGI | a110375cbf8a | -12.056 ± 6.917 |
| Photo | Edgepred_GPPT | 021cfd4af878 | -2.363 ± 6.703 |
| Photo | Edgepred_Gprompt | 1c80d00b449e | +0.024 ± 3.013 |
| Photo | GraphCL | 28ba37961798 | -5.338 ± 7.603 |
| Photo | GraphMAE | cf3667a6a9c1 | -3.372 ± 4.091 |
| Photo | SimGRACE | 92300a0c4162 | -9.461 ± 7.298 |
| Computers | DGI | a5f0b042a971 | -13.374 ± 9.570 |
| Computers | Edgepred_GPPT | 37699b780525 | -4.506 ± 9.462 |
| Computers | Edgepred_Gprompt | fd9e216e5d41 | -1.817 ± 2.143 |
| Computers | GraphCL | f821af6f2440 | -3.635 ± 6.299 |
| Computers | GraphMAE | 61e493ff08ad | -11.951 ± 6.672 |
| Computers | SimGRACE | a4ae8c26f33c | +2.826 ± 3.668 |
| PubMed | Bandana | a4e198b2ea8d | -1.120 ± 1.052 |
| PubMed | DGI | 7be8f89a61ad | -5.100 ± 6.317 |
| PubMed | Edgepred_GPPT | 7b0bb774ae09 | -0.140 ± 0.230 |
| PubMed | Edgepred_Gprompt | fb61b312c4fc | +0.900 ± 1.859 |
| PubMed | GraphCL | dbcbde183804 | +0.260 ± 0.602 |
| PubMed | GraphMAE | 70d698a5a235 | -0.100 ± 0.803 |
| PubMed | GraphMAE2 | f33a0e8ff3a7 | -0.020 ± 0.572 |
| PubMed | S2GAE | dedf45129a74 | -4.480 ± 7.757 |
| PubMed | SimGRACE | 50beb94f14f3 | +0.280 ± 0.691 |

## 条件Prompt

完整：`A3_tr1_qeps`；消融：`A3_noprompt`。

| 数据集 | checkpoint | SHA前缀 | 配对增量均值±标准差pp |
|---|---|---|---:|
| Cornell | DGI | 7b79975af25a | -6.486 ± 8.461 |
| Cornell | Edgepred_GPPT | 72a6b63fbfbf | +1.622 ± 5.921 |
| Cornell | Edgepred_Gprompt | f656050f9497 | -8.108 ± 10.978 |
| Cornell | GraphCL | 2f6b499f430b | -2.703 ± 6.891 |
| Cornell | GraphMAE | efa244a8c256 | -4.324 ± 15.121 |
| Cornell | SimGRACE | 0a9e536877f7 | +2.162 ± 11.209 |
| Chameleon | DGI | 649de463352a | +4.737 ± 8.236 |
| Chameleon | Edgepred_GPPT | 4f7fee992773 | +3.114 ± 2.184 |
| Chameleon | Edgepred_Gprompt | 0f63e6a81dbc | +8.246 ± 5.753 |
| Chameleon | GraphCL | 07058ab2a2fa | +1.447 ± 4.189 |
| Chameleon | GraphMAE | fae19b2ebf82 | -0.526 ± 4.186 |
| Chameleon | SimGRACE | 525eec28fed1 | +0.526 ± 5.262 |
| Cora | Bandana | df27fae92ee3 | +27.020 ± 10.905 |
| Cora | DGI | fb2b361637cb | +23.260 ± 6.345 |
| Cora | Edgepred_GPPT | a279caf82ae2 | +41.020 ± 0.832 |
| Cora | Edgepred_Gprompt | 87bd546f7897 | +39.280 ± 4.141 |
| Cora | GraphCL | 3e74255f6fd2 | +20.480 ± 1.844 |
| Cora | GraphMAE | fb9bb0e2e2bd | +13.180 ± 2.621 |
| Cora | GraphMAE2 | b4ccffb19908 | +44.160 ± 6.752 |
| Cora | S2GAE | 5ed4965565a5 | +31.740 ± 12.564 |
| Cora | SimGRACE | 1e1fb8143d25 | +47.920 ± 4.341 |
| CiteSeer | Bandana | 8e2a1c825227 | +9.300 ± 4.838 |
| CiteSeer | DGI | 460f97d1cd97 | +15.400 ± 8.924 |
| CiteSeer | DGI | 8f4dd5622b98 | +14.280 ± 7.477 |
| CiteSeer | Edgepred_GPPT | 71a6af7c13c8 | +42.020 ± 3.542 |
| CiteSeer | Edgepred_GPPT | e5488f6ef537 | +39.840 ± 4.667 |
| CiteSeer | Edgepred_Gprompt | 5b3b70a23a78 | +40.640 ± 3.228 |
| CiteSeer | Edgepred_Gprompt | 7bf67dc55e3a | +38.820 ± 4.006 |
| CiteSeer | GraphCL | b4708f27c7e5 | +31.100 ± 3.650 |
| CiteSeer | GraphCL | ba86bc2a10e8 | +26.000 ± 2.175 |
| CiteSeer | GraphMAE | c95a7169f993 | +45.380 ± 6.796 |
| CiteSeer | GraphMAE | f4d07b34b0a6 | +45.400 ± 6.767 |
| CiteSeer | GraphMAE2 | cf4ad5bbee38 | +47.980 ± 5.845 |
| CiteSeer | S2GAE | bd209ec5181e | +27.580 ± 3.171 |
| CiteSeer | SimGRACE | 86e4c3c121d2 | +36.320 ± 6.035 |
| CiteSeer | SimGRACE | da8250d13cc2 | +29.440 ± 2.066 |
| Actor | DGI | 8ab886f3fc7e | +2.658 ± 2.998 |
| Actor | Edgepred_GPPT | ade64f6f3b9b | +1.263 ± 1.526 |
| Actor | Edgepred_Gprompt | 30b9c2c89a1d | +1.066 ± 1.807 |
| Actor | GraphCL | 73eda304d4ee | -0.579 ± 2.046 |
| Actor | GraphMAE | 39d2fb2110d1 | -0.592 ± 4.431 |
| Actor | SimGRACE | 049c20350cd6 | +0.553 ± 2.654 |
| Photo | DGI | a110375cbf8a | +23.768 ± 4.977 |
| Photo | Edgepred_GPPT | 021cfd4af878 | +2.953 ± 4.923 |
| Photo | Edgepred_Gprompt | 1c80d00b449e | +3.212 ± 5.009 |
| Photo | GraphCL | 28ba37961798 | +19.159 ± 5.741 |
| Photo | GraphMAE | cf3667a6a9c1 | +41.228 ± 3.993 |
| Photo | SimGRACE | 92300a0c4162 | +9.356 ± 4.929 |
| Computers | DGI | a5f0b042a971 | +25.750 ± 9.675 |
| Computers | Edgepred_GPPT | 37699b780525 | +3.145 ± 7.871 |
| Computers | Edgepred_Gprompt | fd9e216e5d41 | +11.821 ± 5.940 |
| Computers | GraphCL | f821af6f2440 | +16.473 ± 7.880 |
| Computers | GraphMAE | 61e493ff08ad | +32.255 ± 6.313 |
| Computers | SimGRACE | a4ae8c26f33c | +7.159 ± 7.850 |
| PubMed | Bandana | a4e198b2ea8d | +29.680 ± 5.480 |
| PubMed | DGI | 7be8f89a61ad | +31.540 ± 4.065 |
| PubMed | Edgepred_GPPT | 7b0bb774ae09 | +6.100 ± 6.320 |
| PubMed | Edgepred_Gprompt | fb61b312c4fc | +2.900 ± 2.442 |
| PubMed | GraphCL | dbcbde183804 | +18.180 ± 6.224 |
| PubMed | GraphMAE | 70d698a5a235 | +0.140 ± 2.819 |
| PubMed | GraphMAE2 | f33a0e8ff3a7 | +3.680 ± 4.430 |
| PubMed | S2GAE | dedf45129a74 | +19.780 ± 10.758 |
| PubMed | SimGRACE | 50beb94f14f3 | +28.060 ± 6.038 |

## Adam步长匹配

完整：`A3_gsh_qeps`；消融：`A3_ungrafted`。

| 数据集 | checkpoint | SHA前缀 | 配对增量均值±标准差pp |
|---|---|---|---:|
| Cornell | DGI | 7b79975af25a | -8.108 ± 6.338 |
| Cornell | Edgepred_GPPT | 72a6b63fbfbf | +2.162 ± 4.441 |
| Cornell | Edgepred_Gprompt | f656050f9497 | -6.486 ± 12.177 |
| Cornell | GraphCL | 2f6b499f430b | -8.108 ± 6.891 |
| Cornell | GraphMAE | efa244a8c256 | -9.189 ± 14.378 |
| Cornell | SimGRACE | 0a9e536877f7 | +1.622 ± 8.242 |
| Chameleon | DGI | 649de463352a | +0.263 ± 4.217 |
| Chameleon | Edgepred_GPPT | 4f7fee992773 | +0.088 ± 2.334 |
| Chameleon | Edgepred_Gprompt | 0f63e6a81dbc | +5.526 ± 3.912 |
| Chameleon | GraphCL | 07058ab2a2fa | -1.886 ± 6.149 |
| Chameleon | GraphMAE | fae19b2ebf82 | -1.711 ± 2.960 |
| Chameleon | SimGRACE | 525eec28fed1 | +1.272 ± 4.535 |
| Cora | Bandana | df27fae92ee3 | +27.860 ± 4.137 |
| Cora | DGI | fb2b361637cb | +27.720 ± 5.978 |
| Cora | Edgepred_GPPT | a279caf82ae2 | +29.720 ± 1.886 |
| Cora | Edgepred_Gprompt | 87bd546f7897 | +22.060 ± 5.290 |
| Cora | GraphCL | 3e74255f6fd2 | +15.000 ± 1.810 |
| Cora | GraphMAE | fb9bb0e2e2bd | +9.680 ± 3.372 |
| Cora | GraphMAE2 | b4ccffb19908 | +27.160 ± 6.430 |
| Cora | S2GAE | 5ed4965565a5 | +35.520 ± 8.394 |
| Cora | SimGRACE | 1e1fb8143d25 | +42.360 ± 2.669 |
| CiteSeer | Bandana | 8e2a1c825227 | +10.220 ± 5.301 |
| CiteSeer | DGI | 460f97d1cd97 | +23.300 ± 4.205 |
| CiteSeer | DGI | 8f4dd5622b98 | +23.140 ± 4.272 |
| CiteSeer | Edgepred_GPPT | 71a6af7c13c8 | +35.140 ± 2.626 |
| CiteSeer | Edgepred_GPPT | e5488f6ef537 | +32.400 ± 3.123 |
| CiteSeer | Edgepred_Gprompt | 5b3b70a23a78 | +25.080 ± 4.027 |
| CiteSeer | Edgepred_Gprompt | 7bf67dc55e3a | +23.980 ± 2.257 |
| CiteSeer | GraphCL | b4708f27c7e5 | +26.220 ± 4.395 |
| CiteSeer | GraphCL | ba86bc2a10e8 | +23.980 ± 3.241 |
| CiteSeer | GraphMAE | c95a7169f993 | +38.160 ± 9.181 |
| CiteSeer | GraphMAE | f4d07b34b0a6 | +38.160 ± 9.181 |
| CiteSeer | GraphMAE2 | cf4ad5bbee38 | +41.280 ± 7.547 |
| CiteSeer | S2GAE | bd209ec5181e | +26.660 ± 2.117 |
| CiteSeer | SimGRACE | 86e4c3c121d2 | +31.340 ± 5.232 |
| CiteSeer | SimGRACE | da8250d13cc2 | +25.980 ± 2.846 |
| Actor | DGI | 8ab886f3fc7e | -0.289 ± 4.665 |
| Actor | Edgepred_GPPT | ade64f6f3b9b | +1.658 ± 0.930 |
| Actor | Edgepred_Gprompt | 30b9c2c89a1d | +0.776 ± 1.114 |
| Actor | GraphCL | 73eda304d4ee | -0.250 ± 1.690 |
| Actor | GraphMAE | 39d2fb2110d1 | -1.092 ± 4.852 |
| Actor | SimGRACE | 049c20350cd6 | +0.737 ± 1.531 |
| Photo | DGI | a110375cbf8a | +22.304 ± 4.517 |
| Photo | Edgepred_GPPT | 021cfd4af878 | +0.299 ± 2.603 |
| Photo | Edgepred_Gprompt | 1c80d00b449e | +1.162 ± 3.830 |
| Photo | GraphCL | 28ba37961798 | +22.553 ± 3.294 |
| Photo | GraphMAE | cf3667a6a9c1 | +38.710 ± 5.210 |
| Photo | SimGRACE | 92300a0c4162 | +10.668 ± 1.932 |
| Computers | DGI | a5f0b042a971 | +16.300 ± 10.866 |
| Computers | Edgepred_GPPT | 37699b780525 | +1.060 ± 8.050 |
| Computers | Edgepred_Gprompt | fd9e216e5d41 | +12.090 ± 4.320 |
| Computers | GraphCL | f821af6f2440 | +13.533 ± 6.164 |
| Computers | GraphMAE | 61e493ff08ad | +32.755 ± 9.653 |
| Computers | SimGRACE | a4ae8c26f33c | +11.204 ± 10.102 |
| PubMed | Bandana | a4e198b2ea8d | +28.180 ± 5.795 |
| PubMed | DGI | 7be8f89a61ad | +21.720 ± 3.399 |
| PubMed | Edgepred_GPPT | 7b0bb774ae09 | +1.800 ± 3.012 |
| PubMed | Edgepred_Gprompt | fb61b312c4fc | -1.720 ± 3.502 |
| PubMed | GraphCL | dbcbde183804 | +14.400 ± 6.899 |
| PubMed | GraphMAE | 70d698a5a235 | -0.100 ± 2.108 |
| PubMed | GraphMAE2 | f33a0e8ff3a7 | +3.060 ± 3.913 |
| PubMed | S2GAE | dedf45129a74 | +23.380 ± 4.907 |
| PubMed | SimGRACE | 50beb94f14f3 | +21.420 ± 5.942 |

## 完整矩阵相对对角

完整：`A3_gsh_qeps`；消融：`A3_diagonal`。

| 数据集 | checkpoint | SHA前缀 | 配对增量均值±标准差pp |
|---|---|---|---:|
| Cornell | DGI | 7b79975af25a | -2.162 ± 2.961 |
| Cornell | Edgepred_GPPT | 72a6b63fbfbf | -1.622 ± 3.082 |
| Cornell | Edgepred_Gprompt | f656050f9497 | -1.081 ± 2.417 |
| Cornell | GraphCL | 2f6b499f430b | -0.541 ± 2.261 |
| Cornell | GraphMAE | efa244a8c256 | -1.081 ± 2.417 |
| Cornell | SimGRACE | 0a9e536877f7 | -4.324 ± 5.269 |
| Chameleon | DGI | 649de463352a | -0.307 ± 0.829 |
| Chameleon | Edgepred_GPPT | 4f7fee992773 | -0.351 ± 1.227 |
| Chameleon | Edgepred_Gprompt | 0f63e6a81dbc | +1.930 ± 2.434 |
| Chameleon | GraphCL | 07058ab2a2fa | -0.439 ± 1.874 |
| Chameleon | GraphMAE | fae19b2ebf82 | -0.000 ± 0.658 |
| Chameleon | SimGRACE | 525eec28fed1 | -0.175 ± 1.301 |
| Cora | Bandana | df27fae92ee3 | +7.660 ± 5.767 |
| Cora | DGI | fb2b361637cb | +0.240 ± 5.829 |
| Cora | Edgepred_GPPT | a279caf82ae2 | -0.400 ± 0.682 |
| Cora | Edgepred_Gprompt | 87bd546f7897 | +0.480 ± 0.460 |
| Cora | GraphCL | 3e74255f6fd2 | -1.300 ± 1.634 |
| Cora | GraphMAE | fb9bb0e2e2bd | +0.620 ± 0.572 |
| Cora | GraphMAE2 | b4ccffb19908 | -0.060 ± 0.207 |
| Cora | S2GAE | 5ed4965565a5 | +5.640 ± 5.650 |
| Cora | SimGRACE | 1e1fb8143d25 | -0.440 ± 0.241 |
| CiteSeer | Bandana | 8e2a1c825227 | +1.860 ± 0.981 |
| CiteSeer | DGI | 460f97d1cd97 | +1.440 ± 4.298 |
| CiteSeer | DGI | 8f4dd5622b98 | +4.100 ± 5.985 |
| CiteSeer | Edgepred_GPPT | 71a6af7c13c8 | -0.640 ± 0.615 |
| CiteSeer | Edgepred_GPPT | e5488f6ef537 | -1.200 ± 0.696 |
| CiteSeer | Edgepred_Gprompt | 5b3b70a23a78 | +0.280 ± 2.092 |
| CiteSeer | Edgepred_Gprompt | 7bf67dc55e3a | -0.480 ± 1.439 |
| CiteSeer | GraphCL | b4708f27c7e5 | -0.060 ± 2.276 |
| CiteSeer | GraphCL | ba86bc2a10e8 | -0.860 ± 1.547 |
| CiteSeer | GraphMAE | c95a7169f993 | +0.160 ± 0.297 |
| CiteSeer | GraphMAE | f4d07b34b0a6 | +0.160 ± 0.297 |
| CiteSeer | GraphMAE2 | cf4ad5bbee38 | +0.060 ± 0.365 |
| CiteSeer | S2GAE | bd209ec5181e | -0.020 ± 0.968 |
| CiteSeer | SimGRACE | 86e4c3c121d2 | -0.720 ± 0.766 |
| CiteSeer | SimGRACE | da8250d13cc2 | -0.500 ± 3.119 |
| Actor | DGI | 8ab886f3fc7e | -1.184 ± 1.673 |
| Actor | Edgepred_GPPT | ade64f6f3b9b | +0.158 ± 0.296 |
| Actor | Edgepred_Gprompt | 30b9c2c89a1d | -0.079 ± 0.468 |
| Actor | GraphCL | 73eda304d4ee | +0.184 ± 0.692 |
| Actor | GraphMAE | 39d2fb2110d1 | -0.487 ± 0.894 |
| Actor | SimGRACE | 049c20350cd6 | +0.303 ± 0.829 |
| Photo | DGI | a110375cbf8a | -3.274 ± 5.872 |
| Photo | Edgepred_GPPT | 021cfd4af878 | -1.957 ± 2.801 |
| Photo | Edgepred_Gprompt | 1c80d00b449e | -1.725 ± 3.548 |
| Photo | GraphCL | 28ba37961798 | +4.820 ± 4.987 |
| Photo | GraphMAE | cf3667a6a9c1 | +2.935 ± 2.769 |
| Photo | SimGRACE | 92300a0c4162 | -0.446 ± 1.772 |
| Computers | DGI | a5f0b042a971 | -0.462 ± 4.934 |
| Computers | Edgepred_GPPT | 37699b780525 | -1.865 ± 3.726 |
| Computers | Edgepred_Gprompt | fd9e216e5d41 | -0.968 ± 2.458 |
| Computers | GraphCL | f821af6f2440 | -1.502 ± 10.992 |
| Computers | GraphMAE | 61e493ff08ad | +2.537 ± 2.472 |
| Computers | SimGRACE | a4ae8c26f33c | +1.632 ± 4.369 |
| PubMed | Bandana | a4e198b2ea8d | -0.300 ± 3.978 |
| PubMed | DGI | 7be8f89a61ad | +2.840 ± 2.459 |
| PubMed | Edgepred_GPPT | 7b0bb774ae09 | +0.100 ± 0.212 |
| PubMed | Edgepred_Gprompt | fb61b312c4fc | +0.260 ± 0.537 |
| PubMed | GraphCL | dbcbde183804 | -0.100 ± 0.846 |
| PubMed | GraphMAE | 70d698a5a235 | +0.120 ± 0.311 |
| PubMed | GraphMAE2 | f33a0e8ff3a7 | +0.060 ± 0.251 |
| PubMed | S2GAE | dedf45129a74 | -1.140 ± 4.851 |
| PubMed | SimGRACE | 50beb94f14f3 | -0.060 ± 0.321 |

## 匹配Shampoo相对Adam

完整：`A3_gsh_qeps`；消融：`A3_tr1_qeps`。

| 数据集 | checkpoint | SHA前缀 | 配对增量均值±标准差pp |
|---|---|---|---:|
| Cornell | DGI | 7b79975af25a | -2.162 ± 5.199 |
| Cornell | Edgepred_GPPT | 72a6b63fbfbf | -1.081 ± 2.417 |
| Cornell | Edgepred_Gprompt | f656050f9497 | -1.081 ± 1.480 |
| Cornell | GraphCL | 2f6b499f430b | -1.622 ± 2.417 |
| Cornell | GraphMAE | efa244a8c256 | -3.243 ± 5.199 |
| Cornell | SimGRACE | 0a9e536877f7 | -1.622 ± 3.082 |
| Chameleon | DGI | 649de463352a | -2.237 ± 5.273 |
| Chameleon | Edgepred_GPPT | 4f7fee992773 | -1.228 ± 1.294 |
| Chameleon | Edgepred_Gprompt | 0f63e6a81dbc | +0.132 ± 2.887 |
| Chameleon | GraphCL | 07058ab2a2fa | -1.842 ± 3.414 |
| Chameleon | GraphMAE | fae19b2ebf82 | -1.535 ± 0.744 |
| Chameleon | SimGRACE | 525eec28fed1 | +1.404 ± 2.802 |
| Cora | Bandana | df27fae92ee3 | +1.000 ± 8.720 |
| Cora | DGI | fb2b361637cb | +11.060 ± 3.552 |
| Cora | Edgepred_GPPT | a279caf82ae2 | +0.380 ± 0.370 |
| Cora | Edgepred_Gprompt | 87bd546f7897 | +1.020 ± 1.240 |
| Cora | GraphCL | 3e74255f6fd2 | -0.780 ± 0.760 |
| Cora | GraphMAE | fb9bb0e2e2bd | +0.840 ± 0.498 |
| Cora | GraphMAE2 | b4ccffb19908 | -0.000 ± 0.652 |
| Cora | S2GAE | 5ed4965565a5 | +4.280 ± 4.735 |
| Cora | SimGRACE | 1e1fb8143d25 | +0.080 ± 0.526 |
| CiteSeer | Bandana | 8e2a1c825227 | +0.940 ± 7.007 |
| CiteSeer | DGI | 460f97d1cd97 | +12.580 ± 9.806 |
| CiteSeer | DGI | 8f4dd5622b98 | +12.820 ± 9.499 |
| CiteSeer | Edgepred_GPPT | 71a6af7c13c8 | -0.080 ± 0.835 |
| CiteSeer | Edgepred_GPPT | e5488f6ef537 | +0.120 ± 0.804 |
| CiteSeer | Edgepred_Gprompt | 5b3b70a23a78 | -0.400 ± 1.548 |
| CiteSeer | Edgepred_Gprompt | 7bf67dc55e3a | +0.040 ± 1.781 |
| CiteSeer | GraphCL | b4708f27c7e5 | -1.020 ± 3.890 |
| CiteSeer | GraphCL | ba86bc2a10e8 | +2.000 ± 2.853 |
| CiteSeer | GraphMAE | c95a7169f993 | +0.340 ± 0.416 |
| CiteSeer | GraphMAE | f4d07b34b0a6 | +0.320 ± 0.409 |
| CiteSeer | GraphMAE2 | cf4ad5bbee38 | +0.340 ± 0.451 |
| CiteSeer | S2GAE | bd209ec5181e | +0.160 ± 1.293 |
| CiteSeer | SimGRACE | 86e4c3c121d2 | -0.480 ± 0.476 |
| CiteSeer | SimGRACE | da8250d13cc2 | -0.360 ± 2.774 |
| Actor | DGI | 8ab886f3fc7e | -0.816 ± 0.861 |
| Actor | Edgepred_GPPT | ade64f6f3b9b | +0.434 ± 0.452 |
| Actor | Edgepred_Gprompt | 30b9c2c89a1d | -0.553 ± 0.969 |
| Actor | GraphCL | 73eda304d4ee | +0.421 ± 1.197 |
| Actor | GraphMAE | 39d2fb2110d1 | -0.724 ± 0.878 |
| Actor | SimGRACE | 049c20350cd6 | +0.316 ± 1.364 |
| Photo | DGI | a110375cbf8a | +0.115 ± 3.356 |
| Photo | Edgepred_GPPT | 021cfd4af878 | -0.993 ± 2.163 |
| Photo | Edgepred_Gprompt | 1c80d00b449e | -1.268 ± 4.225 |
| Photo | GraphCL | 28ba37961798 | +5.124 ± 3.559 |
| Photo | GraphMAE | cf3667a6a9c1 | +4.072 ± 2.774 |
| Photo | SimGRACE | 92300a0c4162 | +2.587 ± 3.969 |
| Computers | DGI | a5f0b042a971 | -8.679 ± 9.673 |
| Computers | Edgepred_GPPT | 37699b780525 | -0.121 ± 3.657 |
| Computers | Edgepred_Gprompt | fd9e216e5d41 | +2.915 ± 6.152 |
| Computers | GraphCL | f821af6f2440 | -1.055 ± 9.160 |
| Computers | GraphMAE | 61e493ff08ad | +2.304 ± 3.515 |
| Computers | SimGRACE | a4ae8c26f33c | +7.596 ± 2.972 |
| PubMed | Bandana | a4e198b2ea8d | +0.500 ± 5.714 |
| PubMed | DGI | 7be8f89a61ad | +5.580 ± 3.854 |
| PubMed | Edgepred_GPPT | 7b0bb774ae09 | +0.240 ± 0.832 |
| PubMed | Edgepred_Gprompt | fb61b312c4fc | -0.780 ± 1.605 |
| PubMed | GraphCL | dbcbde183804 | +0.240 ± 0.688 |
| PubMed | GraphMAE | 70d698a5a235 | -0.140 ± 0.472 |
| PubMed | GraphMAE2 | f33a0e8ff3a7 | +0.360 ± 0.750 |
| PubMed | S2GAE | dedf45129a74 | -1.040 ± 5.944 |
| PubMed | SimGRACE | 50beb94f14f3 | +0.100 ± 0.394 |

## 概率传播

完整：`A3_gsh_qeps`；消融：`A3_gsh_mean`。

| 数据集 | checkpoint | SHA前缀 | 配对增量均值±标准差pp |
|---|---|---|---:|
| Cornell | DGI | 7b79975af25a | -0.541 ± 5.199 |
| Cornell | Edgepred_GPPT | 72a6b63fbfbf | -0.000 ± 3.310 |
| Cornell | Edgepred_Gprompt | f656050f9497 | +0.541 ± 2.261 |
| Cornell | GraphCL | 2f6b499f430b | +0.541 ± 4.441 |
| Cornell | GraphMAE | efa244a8c256 | -2.162 ± 4.009 |
| Cornell | SimGRACE | 0a9e536877f7 | +0.541 ± 10.537 |
| Chameleon | DGI | 649de463352a | -0.570 ± 1.468 |
| Chameleon | Edgepred_GPPT | 4f7fee992773 | -0.965 ± 1.965 |
| Chameleon | Edgepred_Gprompt | 0f63e6a81dbc | +0.088 ± 1.500 |
| Chameleon | GraphCL | 07058ab2a2fa | -0.614 ± 1.441 |
| Chameleon | GraphMAE | fae19b2ebf82 | -1.184 ± 1.965 |
| Chameleon | SimGRACE | 525eec28fed1 | +0.088 ± 1.294 |
| Cora | Bandana | df27fae92ee3 | +3.240 ± 3.506 |
| Cora | DGI | fb2b361637cb | +1.380 ± 2.398 |
| Cora | Edgepred_GPPT | a279caf82ae2 | +0.780 ± 0.785 |
| Cora | Edgepred_Gprompt | 87bd546f7897 | +0.460 ± 0.586 |
| Cora | GraphCL | 3e74255f6fd2 | +0.600 ± 0.583 |
| Cora | GraphMAE | fb9bb0e2e2bd | +0.880 ± 0.522 |
| Cora | GraphMAE2 | b4ccffb19908 | +0.660 ± 0.422 |
| Cora | S2GAE | 5ed4965565a5 | +2.120 ± 1.604 |
| Cora | SimGRACE | 1e1fb8143d25 | +0.640 ± 0.416 |
| CiteSeer | Bandana | 8e2a1c825227 | +5.420 ± 5.020 |
| CiteSeer | DGI | 460f97d1cd97 | +2.960 ± 1.612 |
| CiteSeer | DGI | 8f4dd5622b98 | +2.820 ± 1.521 |
| CiteSeer | Edgepred_GPPT | 71a6af7c13c8 | +0.540 ± 0.522 |
| CiteSeer | Edgepred_GPPT | e5488f6ef537 | +0.720 ± 0.432 |
| CiteSeer | Edgepred_Gprompt | 5b3b70a23a78 | +0.620 ± 0.507 |
| CiteSeer | Edgepred_Gprompt | 7bf67dc55e3a | +0.600 ± 0.640 |
| CiteSeer | GraphCL | b4708f27c7e5 | +0.500 ± 0.815 |
| CiteSeer | GraphCL | ba86bc2a10e8 | -0.320 ± 1.094 |
| CiteSeer | GraphMAE | c95a7169f993 | +0.300 ± 0.339 |
| CiteSeer | GraphMAE | f4d07b34b0a6 | +0.300 ± 0.339 |
| CiteSeer | GraphMAE2 | cf4ad5bbee38 | +0.580 ± 0.390 |
| CiteSeer | S2GAE | bd209ec5181e | -0.160 ± 0.669 |
| CiteSeer | SimGRACE | 86e4c3c121d2 | +1.020 ± 0.476 |
| CiteSeer | SimGRACE | da8250d13cc2 | +0.760 ± 1.085 |
| Actor | DGI | 8ab886f3fc7e | +0.184 ± 0.313 |
| Actor | Edgepred_GPPT | ade64f6f3b9b | -0.053 ± 0.709 |
| Actor | Edgepred_Gprompt | 30b9c2c89a1d | +0.500 ± 0.344 |
| Actor | GraphCL | 73eda304d4ee | +0.711 ± 0.749 |
| Actor | GraphMAE | 39d2fb2110d1 | +0.447 ± 0.791 |
| Actor | SimGRACE | 049c20350cd6 | +0.289 ± 0.391 |
| Photo | DGI | a110375cbf8a | +2.056 ± 1.197 |
| Photo | Edgepred_GPPT | 021cfd4af878 | -0.198 ± 2.212 |
| Photo | Edgepred_Gprompt | 1c80d00b449e | -0.160 ± 1.088 |
| Photo | GraphCL | 28ba37961798 | +1.364 ± 1.813 |
| Photo | GraphMAE | cf3667a6a9c1 | -0.003 ± 0.734 |
| Photo | SimGRACE | 92300a0c4162 | +0.924 ± 1.831 |
| Computers | DGI | a5f0b042a971 | +1.566 ± 1.591 |
| Computers | Edgepred_GPPT | 37699b780525 | +0.474 ± 0.418 |
| Computers | Edgepred_Gprompt | fd9e216e5d41 | +0.449 ± 0.235 |
| Computers | GraphCL | f821af6f2440 | +0.976 ± 1.528 |
| Computers | GraphMAE | 61e493ff08ad | +0.124 ± 0.661 |
| Computers | SimGRACE | a4ae8c26f33c | +3.155 ± 2.967 |
| PubMed | Bandana | a4e198b2ea8d | +0.120 ± 0.148 |
| PubMed | DGI | 7be8f89a61ad | -0.080 ± 0.045 |
| PubMed | Edgepred_GPPT | 7b0bb774ae09 | +0.020 ± 0.045 |
| PubMed | Edgepred_Gprompt | fb61b312c4fc | +0.080 ± 0.249 |
| PubMed | GraphCL | dbcbde183804 | +0.160 ± 0.351 |
| PubMed | GraphMAE | 70d698a5a235 | -0.000 ± 0.436 |
| PubMed | GraphMAE2 | f33a0e8ff3a7 | -0.140 ± 0.508 |
| PubMed | S2GAE | dedf45129a74 | -0.180 ± 0.349 |
| PubMed | SimGRACE | 50beb94f14f3 | -0.080 ± 0.259 |