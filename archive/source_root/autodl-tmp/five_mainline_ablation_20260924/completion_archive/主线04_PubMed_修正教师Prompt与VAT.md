# 主线04消融结果

VAT相对无VAT宏均 +0.132 pp，25正/14平/24负，属于小且不稳定的增益。教师修正 +0.631 pp（46/63格为正），其中 PubMed +1.596 pp；KD −0.035 pp，接近零。VAT相对同epsilon随机噪声 +0.121 pp，但315个checkpoint×seed记录中只有207个首末步RMS匹配，108个不匹配，存在方向和幅度混杂，不能宣称纯方向优势。保留修正教师作为本线较清晰贡献，VAT仅作为有条件小增益。


口径：以下增量均为预定义完整分支减消融分支，单位为准确率百分点；总体采用8数据集等权宏均值。每个checkpoint内配对5个相同支持采样seed，再对checkpoint汇总。5个seed不是5张独立图，未宣称统计显著性或独立外部确认。主线03的tr1与gsh完整分支不同，不能混用消融分母。


详细配对表如下。


每项按同seed、同checkpoint配对。差值=完整分支−消融分支。正值是保留模块收益；并非统计显著性结论。

## VAT一致性

完整：`A4_vat03`；消融：`A4_none`。

| 数据集 | checkpoint | SHA前缀 | 配对增量均值±标准差pp |
|---|---|---|---:|
| Cornell | DGI | 7b79975af25a | +1.081 ± 1.480 |
| Cornell | Edgepred_GPPT | 72a6b63fbfbf | -0.541 ± 1.209 |
| Cornell | Edgepred_Gprompt | f656050f9497 | +0.541 ± 1.209 |
| Cornell | GraphCL | 2f6b499f430b | +1.622 ± 3.626 |
| Cornell | GraphMAE | efa244a8c256 | +0.000 ± 1.911 |
| Cornell | SimGRACE | 0a9e536877f7 | -0.541 ± 1.209 |
| Chameleon | DGI | 649de463352a | +3.377 ± 2.671 |
| Chameleon | Edgepred_GPPT | 4f7fee992773 | +0.395 ± 0.422 |
| Chameleon | Edgepred_Gprompt | 0f63e6a81dbc | -0.088 ± 0.593 |
| Chameleon | GraphCL | 07058ab2a2fa | +0.000 ± 1.074 |
| Chameleon | GraphMAE | fae19b2ebf82 | +0.088 ± 0.651 |
| Chameleon | SimGRACE | 525eec28fed1 | -0.088 ± 0.398 |
| Cora | Bandana | df27fae92ee3 | +0.000 ± 0.000 |
| Cora | DGI | fb2b361637cb | -0.060 ± 0.241 |
| Cora | Edgepred_GPPT | a279caf82ae2 | +0.000 ± 0.255 |
| Cora | Edgepred_Gprompt | 87bd546f7897 | -0.020 ± 0.130 |
| Cora | GraphCL | 3e74255f6fd2 | +0.460 ± 0.397 |
| Cora | GraphMAE | fb9bb0e2e2bd | -0.040 ± 0.055 |
| Cora | GraphMAE2 | b4ccffb19908 | -0.040 ± 0.055 |
| Cora | S2GAE | 5ed4965565a5 | +0.000 ± 0.000 |
| Cora | SimGRACE | 1e1fb8143d25 | +0.000 ± 0.187 |
| CiteSeer | Bandana | 8e2a1c825227 | -0.040 ± 0.089 |
| CiteSeer | DGI | 460f97d1cd97 | +0.040 ± 0.114 |
| CiteSeer | DGI | 8f4dd5622b98 | +0.480 ± 1.190 |
| CiteSeer | Edgepred_GPPT | 71a6af7c13c8 | +0.080 ± 0.084 |
| CiteSeer | Edgepred_GPPT | e5488f6ef537 | -0.000 ± 0.100 |
| CiteSeer | Edgepred_Gprompt | 5b3b70a23a78 | +0.000 ± 0.000 |
| CiteSeer | Edgepred_Gprompt | 7bf67dc55e3a | +0.020 ± 0.084 |
| CiteSeer | GraphCL | b4708f27c7e5 | +1.180 ± 2.286 |
| CiteSeer | GraphCL | ba86bc2a10e8 | +1.180 ± 1.307 |
| CiteSeer | GraphMAE | c95a7169f993 | +0.000 ± 0.000 |
| CiteSeer | GraphMAE | f4d07b34b0a6 | +0.000 ± 0.000 |
| CiteSeer | GraphMAE2 | cf4ad5bbee38 | +0.000 ± 0.000 |
| CiteSeer | S2GAE | bd209ec5181e | +0.580 ± 1.413 |
| CiteSeer | SimGRACE | 86e4c3c121d2 | +0.280 ± 0.277 |
| CiteSeer | SimGRACE | da8250d13cc2 | -0.020 ± 0.563 |
| Actor | DGI | 8ab886f3fc7e | -0.289 ± 0.281 |
| Actor | Edgepred_GPPT | ade64f6f3b9b | -0.000 ± 0.168 |
| Actor | Edgepred_Gprompt | 30b9c2c89a1d | -0.053 ± 0.317 |
| Actor | GraphCL | 73eda304d4ee | -0.066 ± 0.168 |
| Actor | GraphMAE | 39d2fb2110d1 | -0.026 ± 0.075 |
| Actor | SimGRACE | 049c20350cd6 | -0.053 ± 0.225 |
| Photo | DGI | a110375cbf8a | -0.005 ± 0.015 |
| Photo | Edgepred_GPPT | 021cfd4af878 | -0.008 ± 0.040 |
| Photo | Edgepred_Gprompt | 1c80d00b449e | -0.021 ± 0.048 |
| Photo | GraphCL | 28ba37961798 | +0.107 ± 0.195 |
| Photo | GraphMAE | cf3667a6a9c1 | -0.005 ± 0.012 |
| Photo | SimGRACE | 92300a0c4162 | +0.016 ± 0.091 |
| Computers | DGI | a5f0b042a971 | +0.004 ± 0.023 |
| Computers | Edgepred_GPPT | 37699b780525 | +0.001 ± 0.030 |
| Computers | Edgepred_Gprompt | fd9e216e5d41 | +0.012 ± 0.047 |
| Computers | GraphCL | f821af6f2440 | -2.093 ± 4.598 |
| Computers | GraphMAE | 61e493ff08ad | -0.001 ± 0.006 |
| Computers | SimGRACE | a4ae8c26f33c | +0.229 ± 0.235 |
| PubMed | Bandana | a4e198b2ea8d | +0.000 ± 0.000 |
| PubMed | DGI | 7be8f89a61ad | +0.440 ± 0.658 |
| PubMed | Edgepred_GPPT | 7b0bb774ae09 | -0.020 ± 0.045 |
| PubMed | Edgepred_Gprompt | fb61b312c4fc | +1.060 ± 1.045 |
| PubMed | GraphCL | dbcbde183804 | +0.700 ± 1.111 |
| PubMed | GraphMAE | 70d698a5a235 | -0.220 ± 0.832 |
| PubMed | GraphMAE2 | f33a0e8ff3a7 | -0.540 ± 0.336 |
| PubMed | S2GAE | dedf45129a74 | +0.000 ± 0.000 |
| PubMed | SimGRACE | 50beb94f14f3 | +0.140 ± 0.182 |

## KD

完整：`A4_vat03`；消融：`A4_nokd`。

| 数据集 | checkpoint | SHA前缀 | 配对增量均值±标准差pp |
|---|---|---|---:|
| Cornell | DGI | 7b79975af25a | -2.162 ± 4.835 |
| Cornell | Edgepred_GPPT | 72a6b63fbfbf | +1.622 ± 2.417 |
| Cornell | Edgepred_Gprompt | f656050f9497 | -0.000 ± 3.310 |
| Cornell | GraphCL | 2f6b499f430b | +0.541 ± 2.961 |
| Cornell | GraphMAE | efa244a8c256 | +0.541 ± 1.209 |
| Cornell | SimGRACE | 0a9e536877f7 | -2.703 ± 4.273 |
| Chameleon | DGI | 649de463352a | +3.509 ± 4.383 |
| Chameleon | Edgepred_GPPT | 4f7fee992773 | -0.175 ± 1.382 |
| Chameleon | Edgepred_Gprompt | 0f63e6a81dbc | +1.272 ± 1.764 |
| Chameleon | GraphCL | 07058ab2a2fa | -2.018 ± 2.597 |
| Chameleon | GraphMAE | fae19b2ebf82 | +1.184 ± 2.163 |
| Chameleon | SimGRACE | 525eec28fed1 | +0.526 ± 1.601 |
| Cora | Bandana | df27fae92ee3 | +0.000 ± 0.071 |
| Cora | DGI | fb2b361637cb | -3.380 ± 5.249 |
| Cora | Edgepred_GPPT | a279caf82ae2 | +0.020 ± 0.729 |
| Cora | Edgepred_Gprompt | 87bd546f7897 | +0.500 ± 1.729 |
| Cora | GraphCL | 3e74255f6fd2 | -0.360 ± 3.543 |
| Cora | GraphMAE | fb9bb0e2e2bd | +0.840 ± 0.929 |
| Cora | GraphMAE2 | b4ccffb19908 | -0.220 ± 0.383 |
| Cora | S2GAE | 5ed4965565a5 | +0.120 ± 0.239 |
| Cora | SimGRACE | 1e1fb8143d25 | +1.020 ± 1.028 |
| CiteSeer | Bandana | 8e2a1c825227 | +0.000 ± 0.173 |
| CiteSeer | DGI | 460f97d1cd97 | +1.240 ± 2.404 |
| CiteSeer | DGI | 8f4dd5622b98 | +4.940 ± 7.560 |
| CiteSeer | Edgepred_GPPT | 71a6af7c13c8 | +0.700 ± 1.109 |
| CiteSeer | Edgepred_GPPT | e5488f6ef537 | +0.040 ± 0.344 |
| CiteSeer | Edgepred_Gprompt | 5b3b70a23a78 | -0.260 ± 0.950 |
| CiteSeer | Edgepred_Gprompt | 7bf67dc55e3a | +0.600 ± 0.283 |
| CiteSeer | GraphCL | b4708f27c7e5 | -6.340 ± 5.622 |
| CiteSeer | GraphCL | ba86bc2a10e8 | -3.180 ± 2.962 |
| CiteSeer | GraphMAE | c95a7169f993 | +0.320 ± 0.421 |
| CiteSeer | GraphMAE | f4d07b34b0a6 | +0.320 ± 0.421 |
| CiteSeer | GraphMAE2 | cf4ad5bbee38 | -0.200 ± 0.442 |
| CiteSeer | S2GAE | bd209ec5181e | +2.680 ± 3.260 |
| CiteSeer | SimGRACE | 86e4c3c121d2 | -0.380 ± 1.621 |
| CiteSeer | SimGRACE | da8250d13cc2 | -0.760 ± 1.494 |
| Actor | DGI | 8ab886f3fc7e | -1.184 ± 2.264 |
| Actor | Edgepred_GPPT | ade64f6f3b9b | +0.066 ± 0.246 |
| Actor | Edgepred_Gprompt | 30b9c2c89a1d | -0.118 ± 0.911 |
| Actor | GraphCL | 73eda304d4ee | -0.303 ± 1.026 |
| Actor | GraphMAE | 39d2fb2110d1 | -0.224 ± 0.304 |
| Actor | SimGRACE | 049c20350cd6 | -0.671 ± 0.680 |
| Photo | DGI | a110375cbf8a | -0.144 ± 0.392 |
| Photo | Edgepred_GPPT | 021cfd4af878 | +0.024 ± 0.072 |
| Photo | Edgepred_Gprompt | 1c80d00b449e | -0.029 ± 0.347 |
| Photo | GraphCL | 28ba37961798 | -1.610 ± 1.688 |
| Photo | GraphMAE | cf3667a6a9c1 | +0.230 ± 0.167 |
| Photo | SimGRACE | 92300a0c4162 | -0.521 ± 2.711 |
| Computers | DGI | a5f0b042a971 | +0.069 ± 0.177 |
| Computers | Edgepred_GPPT | 37699b780525 | +0.080 ± 0.121 |
| Computers | Edgepred_Gprompt | fd9e216e5d41 | -0.096 ± 0.360 |
| Computers | GraphCL | f821af6f2440 | +1.339 ± 5.790 |
| Computers | GraphMAE | 61e493ff08ad | +0.055 ± 0.050 |
| Computers | SimGRACE | a4ae8c26f33c | -0.409 ± 1.168 |
| PubMed | Bandana | a4e198b2ea8d | -0.100 ± 0.141 |
| PubMed | DGI | 7be8f89a61ad | -0.680 ± 2.286 |
| PubMed | Edgepred_GPPT | 7b0bb774ae09 | +0.180 ± 0.517 |
| PubMed | Edgepred_Gprompt | fb61b312c4fc | +0.300 ± 2.760 |
| PubMed | GraphCL | dbcbde183804 | -1.960 ± 3.827 |
| PubMed | GraphMAE | 70d698a5a235 | +0.840 ± 2.503 |
| PubMed | GraphMAE2 | f33a0e8ff3a7 | +2.500 ± 1.063 |
| PubMed | S2GAE | dedf45129a74 | -0.040 ± 0.167 |
| PubMed | SimGRACE | 50beb94f14f3 | +0.020 ± 0.963 |

## 教师修正

完整：`A4_vat03`；消融：`A4_norect_vat`。

| 数据集 | checkpoint | SHA前缀 | 配对增量均值±标准差pp |
|---|---|---|---:|
| Cornell | DGI | 7b79975af25a | -2.162 ± 5.859 |
| Cornell | Edgepred_GPPT | 72a6b63fbfbf | +0.541 ± 2.261 |
| Cornell | Edgepred_Gprompt | f656050f9497 | +1.081 ± 3.082 |
| Cornell | GraphCL | 2f6b499f430b | +4.324 ± 3.626 |
| Cornell | GraphMAE | efa244a8c256 | -1.081 ± 2.417 |
| Cornell | SimGRACE | 0a9e536877f7 | -2.162 ± 5.199 |
| Chameleon | DGI | 649de463352a | +5.132 ± 3.127 |
| Chameleon | Edgepred_GPPT | 4f7fee992773 | +0.658 ± 1.150 |
| Chameleon | Edgepred_Gprompt | 0f63e6a81dbc | +0.132 ± 2.298 |
| Chameleon | GraphCL | 07058ab2a2fa | +0.658 ± 1.815 |
| Chameleon | GraphMAE | fae19b2ebf82 | +1.053 ± 1.144 |
| Chameleon | SimGRACE | 525eec28fed1 | +0.789 ± 0.704 |
| Cora | Bandana | df27fae92ee3 | +0.020 ± 0.792 |
| Cora | DGI | fb2b361637cb | +4.800 ± 9.195 |
| Cora | Edgepred_GPPT | a279caf82ae2 | +1.420 ± 1.128 |
| Cora | Edgepred_Gprompt | 87bd546f7897 | +2.760 ± 1.491 |
| Cora | GraphCL | 3e74255f6fd2 | +0.520 ± 0.838 |
| Cora | GraphMAE | fb9bb0e2e2bd | -0.320 ± 0.779 |
| Cora | GraphMAE2 | b4ccffb19908 | +2.580 ± 1.625 |
| Cora | S2GAE | 5ed4965565a5 | -1.260 ± 2.079 |
| Cora | SimGRACE | 1e1fb8143d25 | +0.060 ± 0.969 |
| CiteSeer | Bandana | 8e2a1c825227 | +0.200 ± 0.400 |
| CiteSeer | DGI | 460f97d1cd97 | +4.580 ± 6.112 |
| CiteSeer | DGI | 8f4dd5622b98 | +8.740 ± 8.368 |
| CiteSeer | Edgepred_GPPT | 71a6af7c13c8 | +0.640 ± 1.958 |
| CiteSeer | Edgepred_GPPT | e5488f6ef537 | +1.400 ± 0.700 |
| CiteSeer | Edgepred_Gprompt | 5b3b70a23a78 | +2.180 ± 2.357 |
| CiteSeer | Edgepred_Gprompt | 7bf67dc55e3a | +2.240 ± 1.757 |
| CiteSeer | GraphCL | b4708f27c7e5 | +1.540 ± 3.925 |
| CiteSeer | GraphCL | ba86bc2a10e8 | +0.560 ± 3.836 |
| CiteSeer | GraphMAE | c95a7169f993 | +3.240 ± 1.845 |
| CiteSeer | GraphMAE | f4d07b34b0a6 | +3.240 ± 1.845 |
| CiteSeer | GraphMAE2 | cf4ad5bbee38 | +4.240 ± 0.740 |
| CiteSeer | S2GAE | bd209ec5181e | +1.240 ± 3.894 |
| CiteSeer | SimGRACE | 86e4c3c121d2 | -0.480 ± 1.446 |
| CiteSeer | SimGRACE | da8250d13cc2 | +0.500 ± 1.061 |
| Actor | DGI | 8ab886f3fc7e | +0.474 ± 2.052 |
| Actor | Edgepred_GPPT | ade64f6f3b9b | +0.224 ± 0.897 |
| Actor | Edgepred_Gprompt | 30b9c2c89a1d | +0.211 ± 1.053 |
| Actor | GraphCL | 73eda304d4ee | -0.132 ± 1.030 |
| Actor | GraphMAE | 39d2fb2110d1 | -0.776 ± 1.552 |
| Actor | SimGRACE | 049c20350cd6 | -0.276 ± 0.782 |
| Photo | DGI | a110375cbf8a | +0.099 ± 1.373 |
| Photo | Edgepred_GPPT | 021cfd4af878 | -0.457 ± 0.912 |
| Photo | Edgepred_Gprompt | 1c80d00b449e | +0.128 ± 0.663 |
| Photo | GraphCL | 28ba37961798 | -0.579 ± 1.075 |
| Photo | GraphMAE | cf3667a6a9c1 | -3.509 ± 1.568 |
| Photo | SimGRACE | 92300a0c4162 | -0.972 ± 0.594 |
| Computers | DGI | a5f0b042a971 | -2.206 ± 2.121 |
| Computers | Edgepred_GPPT | 37699b780525 | +0.105 ± 1.282 |
| Computers | Edgepred_Gprompt | fd9e216e5d41 | -0.170 ± 0.375 |
| Computers | GraphCL | f821af6f2440 | +3.239 ± 5.244 |
| Computers | GraphMAE | 61e493ff08ad | -2.470 ± 1.311 |
| Computers | SimGRACE | a4ae8c26f33c | -1.871 ± 1.386 |
| PubMed | Bandana | a4e198b2ea8d | +0.500 ± 1.943 |
| PubMed | DGI | 7be8f89a61ad | +4.640 ± 5.473 |
| PubMed | Edgepred_GPPT | 7b0bb774ae09 | +0.540 ± 0.527 |
| PubMed | Edgepred_Gprompt | fb61b312c4fc | +2.340 ± 1.721 |
| PubMed | GraphCL | dbcbde183804 | +1.520 ± 1.502 |
| PubMed | GraphMAE | 70d698a5a235 | +0.680 ± 1.784 |
| PubMed | GraphMAE2 | f33a0e8ff3a7 | +0.420 ± 1.128 |
| PubMed | S2GAE | dedf45129a74 | +2.120 ± 3.451 |
| PubMed | SimGRACE | 50beb94f14f3 | +1.600 ± 1.487 |

## VAT方向相对等epsilon随机噪声

完整：`A4_vat03`；消融：`A4_randomnoise`。

| 数据集 | checkpoint | SHA前缀 | 配对增量均值±标准差pp |
|---|---|---|---:|
| Cornell | DGI | 7b79975af25a | +1.622 ± 2.417 |
| Cornell | Edgepred_GPPT | 72a6b63fbfbf | -0.541 ± 1.209 |
| Cornell | Edgepred_Gprompt | f656050f9497 | +1.081 ± 1.480 |
| Cornell | GraphCL | 2f6b499f430b | +2.162 ± 2.961 |
| Cornell | GraphMAE | efa244a8c256 | +0.000 ± 1.911 |
| Cornell | SimGRACE | 0a9e536877f7 | -0.541 ± 1.209 |
| Chameleon | DGI | 649de463352a | +1.447 ± 2.912 |
| Chameleon | Edgepred_GPPT | 4f7fee992773 | +0.439 ± 0.439 |
| Chameleon | Edgepred_Gprompt | 0f63e6a81dbc | -0.482 ± 1.112 |
| Chameleon | GraphCL | 07058ab2a2fa | -0.044 ± 0.883 |
| Chameleon | GraphMAE | fae19b2ebf82 | +0.219 ± 0.981 |
| Chameleon | SimGRACE | 525eec28fed1 | -0.175 ± 0.609 |
| Cora | Bandana | df27fae92ee3 | +0.000 ± 0.000 |
| Cora | DGI | fb2b361637cb | +0.000 ± 0.141 |
| Cora | Edgepred_GPPT | a279caf82ae2 | +0.020 ± 0.249 |
| Cora | Edgepred_Gprompt | 87bd546f7897 | -0.020 ± 0.130 |
| Cora | GraphCL | 3e74255f6fd2 | +0.400 ± 0.354 |
| Cora | GraphMAE | fb9bb0e2e2bd | -0.040 ± 0.055 |
| Cora | GraphMAE2 | b4ccffb19908 | -0.040 ± 0.055 |
| Cora | S2GAE | 5ed4965565a5 | +0.000 ± 0.000 |
| Cora | SimGRACE | 1e1fb8143d25 | +0.000 ± 0.187 |
| CiteSeer | Bandana | 8e2a1c825227 | -0.020 ± 0.045 |
| CiteSeer | DGI | 460f97d1cd97 | +0.080 ± 0.179 |
| CiteSeer | DGI | 8f4dd5622b98 | +0.420 ± 1.226 |
| CiteSeer | Edgepred_GPPT | 71a6af7c13c8 | +0.040 ± 0.055 |
| CiteSeer | Edgepred_GPPT | e5488f6ef537 | +0.040 ± 0.089 |
| CiteSeer | Edgepred_Gprompt | 5b3b70a23a78 | +0.000 ± 0.000 |
| CiteSeer | Edgepred_Gprompt | 7bf67dc55e3a | +0.020 ± 0.084 |
| CiteSeer | GraphCL | b4708f27c7e5 | +1.180 ± 2.158 |
| CiteSeer | GraphCL | ba86bc2a10e8 | +1.120 ± 1.165 |
| CiteSeer | GraphMAE | c95a7169f993 | +0.000 ± 0.000 |
| CiteSeer | GraphMAE | f4d07b34b0a6 | +0.000 ± 0.000 |
| CiteSeer | GraphMAE2 | cf4ad5bbee38 | +0.000 ± 0.000 |
| CiteSeer | S2GAE | bd209ec5181e | +0.540 ± 1.322 |
| CiteSeer | SimGRACE | 86e4c3c121d2 | +0.260 ± 0.230 |
| CiteSeer | SimGRACE | da8250d13cc2 | -0.060 ± 0.532 |
| Actor | DGI | 8ab886f3fc7e | -0.224 ± 0.257 |
| Actor | Edgepred_GPPT | ade64f6f3b9b | +0.026 ± 0.136 |
| Actor | Edgepred_Gprompt | 30b9c2c89a1d | +0.013 ± 0.299 |
| Actor | GraphCL | 73eda304d4ee | +0.013 ± 0.215 |
| Actor | GraphMAE | 39d2fb2110d1 | -0.026 ± 0.075 |
| Actor | SimGRACE | 049c20350cd6 | -0.039 ± 0.165 |
| Photo | DGI | a110375cbf8a | -0.008 ± 0.015 |
| Photo | Edgepred_GPPT | 021cfd4af878 | -0.011 ± 0.035 |
| Photo | Edgepred_Gprompt | 1c80d00b449e | -0.019 ± 0.045 |
| Photo | GraphCL | 28ba37961798 | +0.101 ± 0.166 |
| Photo | GraphMAE | cf3667a6a9c1 | -0.005 ± 0.012 |
| Photo | SimGRACE | 92300a0c4162 | +0.000 ± 0.081 |
| Computers | DGI | a5f0b042a971 | +0.006 ± 0.016 |
| Computers | Edgepred_GPPT | 37699b780525 | +0.003 ± 0.030 |
| Computers | Edgepred_Gprompt | fd9e216e5d41 | +0.012 ± 0.048 |
| Computers | GraphCL | f821af6f2440 | -2.090 ± 4.595 |
| Computers | GraphMAE | 61e493ff08ad | +0.001 ± 0.008 |
| Computers | SimGRACE | a4ae8c26f33c | +0.220 ± 0.216 |
| PubMed | Bandana | a4e198b2ea8d | -0.020 ± 0.045 |
| PubMed | DGI | 7be8f89a61ad | +0.340 ± 0.445 |
| PubMed | Edgepred_GPPT | 7b0bb774ae09 | -0.020 ± 0.045 |
| PubMed | Edgepred_Gprompt | fb61b312c4fc | +1.060 ± 0.924 |
| PubMed | GraphCL | dbcbde183804 | +0.800 ± 1.185 |
| PubMed | GraphMAE | 70d698a5a235 | -0.260 ± 0.844 |
| PubMed | GraphMAE2 | f33a0e8ff3a7 | -0.540 ± 0.336 |
| PubMed | S2GAE | dedf45129a74 | -0.020 ± 0.045 |
| PubMed | SimGRACE | 50beb94f14f3 | +0.140 ± 0.114 |

## 教师修正×VAT的2×2交互

定义：`A4_vat03 - A4_none - A4_norect_vat + A4_norect_none`。描述性交互，正值不代表已证明普遍协同。

| 数据集 | checkpoint | SHA前缀 | 交互pp |
|---|---|---|---:|
| Cornell | DGI | 7b79975af25a | +1.081 |
| Cornell | Edgepred_GPPT | 72a6b63fbfbf | +0.000 |
| Cornell | Edgepred_Gprompt | f656050f9497 | +0.541 |
| Cornell | GraphCL | 2f6b499f430b | +2.162 |
| Cornell | GraphMAE | efa244a8c256 | -0.541 |
| Cornell | SimGRACE | 0a9e536877f7 | -0.541 |
| Chameleon | DGI | 649de463352a | +3.246 |
| Chameleon | Edgepred_GPPT | 4f7fee992773 | +0.746 |
| Chameleon | Edgepred_Gprompt | 0f63e6a81dbc | +0.175 |
| Chameleon | GraphCL | 07058ab2a2fa | -0.044 |
| Chameleon | GraphMAE | fae19b2ebf82 | -0.132 |
| Chameleon | SimGRACE | 525eec28fed1 | -0.132 |
| Cora | Bandana | df27fae92ee3 | -0.020 |
| Cora | DGI | fb2b361637cb | -0.340 |
| Cora | Edgepred_GPPT | a279caf82ae2 | -0.220 |
| Cora | Edgepred_Gprompt | 87bd546f7897 | -0.020 |
| Cora | GraphCL | 3e74255f6fd2 | +0.080 |
| Cora | GraphMAE | fb9bb0e2e2bd | -0.100 |
| Cora | GraphMAE2 | b4ccffb19908 | -0.060 |
| Cora | S2GAE | 5ed4965565a5 | +0.040 |
| Cora | SimGRACE | 1e1fb8143d25 | -0.020 |
| CiteSeer | Bandana | 8e2a1c825227 | -0.040 |
| CiteSeer | DGI | 460f97d1cd97 | +0.020 |
| CiteSeer | DGI | 8f4dd5622b98 | +0.480 |
| CiteSeer | Edgepred_GPPT | 71a6af7c13c8 | -0.260 |
| CiteSeer | Edgepred_GPPT | e5488f6ef537 | -0.000 |
| CiteSeer | Edgepred_Gprompt | 5b3b70a23a78 | +0.020 |
| CiteSeer | Edgepred_Gprompt | 7bf67dc55e3a | +0.060 |
| CiteSeer | GraphCL | b4708f27c7e5 | +0.380 |
| CiteSeer | GraphCL | ba86bc2a10e8 | +0.180 |
| CiteSeer | GraphMAE | c95a7169f993 | +0.000 |
| CiteSeer | GraphMAE | f4d07b34b0a6 | +0.000 |
| CiteSeer | GraphMAE2 | cf4ad5bbee38 | +0.000 |
| CiteSeer | S2GAE | bd209ec5181e | +0.600 |
| CiteSeer | SimGRACE | 86e4c3c121d2 | +0.360 |
| CiteSeer | SimGRACE | da8250d13cc2 | -0.020 |
| Actor | DGI | 8ab886f3fc7e | +1.026 |
| Actor | Edgepred_GPPT | ade64f6f3b9b | -0.026 |
| Actor | Edgepred_Gprompt | 30b9c2c89a1d | -0.000 |
| Actor | GraphCL | 73eda304d4ee | +0.013 |
| Actor | GraphMAE | 39d2fb2110d1 | -0.039 |
| Actor | SimGRACE | 049c20350cd6 | -0.237 |
| Photo | DGI | a110375cbf8a | +0.011 |
| Photo | Edgepred_GPPT | 021cfd4af878 | -0.008 |
| Photo | Edgepred_Gprompt | 1c80d00b449e | -0.032 |
| Photo | GraphCL | 28ba37961798 | -1.020 |
| Photo | GraphMAE | cf3667a6a9c1 | -0.005 |
| Photo | SimGRACE | 92300a0c4162 | +0.176 |
| Computers | DGI | a5f0b042a971 | +0.003 |
| Computers | Edgepred_GPPT | 37699b780525 | +0.003 |
| Computers | Edgepred_Gprompt | fd9e216e5d41 | +0.006 |
| Computers | GraphCL | f821af6f2440 | +0.790 |
| Computers | GraphMAE | 61e493ff08ad | +0.001 |
| Computers | SimGRACE | a4ae8c26f33c | +0.114 |
| PubMed | Bandana | a4e198b2ea8d | +0.000 |
| PubMed | DGI | 7be8f89a61ad | +2.580 |
| PubMed | Edgepred_GPPT | 7b0bb774ae09 | +0.040 |
| PubMed | Edgepred_Gprompt | fb61b312c4fc | +0.520 |
| PubMed | GraphCL | dbcbde183804 | -0.600 |
| PubMed | GraphMAE | 70d698a5a235 | -0.420 |
| PubMed | GraphMAE2 | f33a0e8ff3a7 | -0.360 |
| PubMed | S2GAE | dedf45129a74 | -0.020 |
| PubMed | SimGRACE | 50beb94f14f3 | +0.180 |
VAT幅度边界：等epsilon不等于等实际RMS；207/315首末步匹配，108不匹配，不能据此单独归因于方向。
