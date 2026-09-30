# 配对结果汇总

种子区间仅反映同一张图上支持采样变化，不是独立图置信区间。

|数据集|权重|方法|n|准确率 %|对 Frozen pp|对 GraphPrompt pp|
|---|---|---|---:|---:|---:|---:|
|CiteSeer|GraphMAE|frozen|5|31.960|+0.000|-1.640|
|CiteSeer|GraphMAE|graphprompt|5|33.600|+1.640|+0.000|
|CiteSeer|GraphMAE|rawprop2|5|61.040|+29.080|+27.440|
|CiteSeer|GraphMAE|ridge_teacher|5|61.120|+29.160|+27.520|
|CiteSeer|GraphMAE|token_kd3|5|40.880|+8.920|+7.280|
|CiteSeer|GraphMAE|token_ridge_kd10|5|52.640|+20.680|+19.040|
|CiteSeer|GraphMAE|token_ridge_kd3|5|49.240|+17.280|+15.640|
|CiteSeer|GraphMAE2|frozen|5|40.680|+0.000|+1.040|
|CiteSeer|GraphMAE2|graphprompt|5|39.640|-1.040|+0.000|
|CiteSeer|GraphMAE2|rawprop2|5|61.040|+20.360|+21.400|
|CiteSeer|GraphMAE2|ridge_teacher|5|61.120|+20.440|+21.480|
|CiteSeer|GraphMAE2|token_kd3|5|50.320|+9.640|+10.680|
|CiteSeer|GraphMAE2|token_ridge_kd10|5|57.800|+17.120|+18.160|
|CiteSeer|GraphMAE2|token_ridge_kd3|5|54.760|+14.080|+15.120|
|Cora|GraphMAE|frozen|5|60.840|+0.000|+0.280|
|Cora|GraphMAE|graphprompt|5|60.560|-0.280|+0.000|
|Cora|GraphMAE|rawprop2|5|69.080|+8.240|+8.520|
|Cora|GraphMAE|ridge_teacher|5|72.360|+11.520|+11.800|
|Cora|GraphMAE|token_kd3|5|66.960|+6.120|+6.400|
|Cora|GraphMAE|token_ridge_kd10|5|74.880|+14.040|+14.320|
|Cora|GraphMAE|token_ridge_kd3|5|72.560|+11.720|+12.000|
|Cora|GraphMAE2|frozen|5|56.400|+0.000|+1.520|
|Cora|GraphMAE2|graphprompt|5|54.880|-1.520|+0.000|
|Cora|GraphMAE2|rawprop2|5|69.080|+12.680|+14.200|
|Cora|GraphMAE2|ridge_teacher|5|72.360|+15.960|+17.480|
|Cora|GraphMAE2|token_kd3|5|65.240|+8.840|+10.360|
|Cora|GraphMAE2|token_ridge_kd10|5|73.240|+16.840|+18.360|
|Cora|GraphMAE2|token_ridge_kd3|5|70.960|+14.560|+16.080|
|PubMed|GraphMAE|frozen|5|71.360|+0.000|-3.160|
|PubMed|GraphMAE|graphprompt|5|74.520|+3.160|+0.000|
|PubMed|GraphMAE|rawprop2|5|69.320|-2.040|-5.200|
|PubMed|GraphMAE|ridge_teacher|5|68.000|-3.360|-6.520|
|PubMed|GraphMAE|token_kd3|5|72.320|+0.960|-2.200|
|PubMed|GraphMAE|token_ridge_kd10|5|70.040|-1.320|-4.480|
|PubMed|GraphMAE|token_ridge_kd3|5|73.080|+1.720|-1.440|
|PubMed|GraphMAE2|frozen|5|68.320|+0.000|-1.840|
|PubMed|GraphMAE2|graphprompt|5|70.160|+1.840|+0.000|
|PubMed|GraphMAE2|rawprop2|5|69.320|+1.000|-0.840|
|PubMed|GraphMAE2|ridge_teacher|5|68.000|-0.320|-2.160|
|PubMed|GraphMAE2|token_kd3|5|72.160|+3.840|+2.000|
|PubMed|GraphMAE2|token_ridge_kd10|5|69.480|+1.160|-0.680|
|PubMed|GraphMAE2|token_ridge_kd3|5|71.720|+3.400|+1.560|
