# 配对结果汇总

种子区间仅反映同一张图上支持采样变化，不是独立图置信区间。

|数据集|权重|方法|n|准确率 %|对 Frozen pp|对 GraphPrompt pp|
|---|---|---|---:|---:|---:|---:|
|Cora|GraphMAE|cons_none_0|5|72.920|+11.720|+11.680|
|Cora|GraphMAE|cons_rpt_0.1|5|72.920|+11.720|+11.680|
|Cora|GraphMAE|cons_rpt_0.3|5|72.920|+11.720|+11.680|
|Cora|GraphMAE|cons_vat_0.1|5|72.880|+11.680|+11.640|
|Cora|GraphMAE|cons_vat_0.3|5|72.840|+11.640|+11.600|
|Cora|GraphMAE|frozen|5|61.200|+0.000|-0.040|
|Cora|GraphMAE|graphprompt|5|61.240|+0.040|+0.000|
|Cora|GraphMAE|rect_teacher_ref50|5|71.360|+10.160|+10.120|
|Cora|GraphMAE|ridge_teacher_plain|5|70.560|+9.360|+9.320|
|Cora|GraphMAE2|cons_none_0|5|70.760|+12.840|+15.040|
|Cora|GraphMAE2|cons_rpt_0.1|5|70.760|+12.840|+15.040|
|Cora|GraphMAE2|cons_rpt_0.3|5|70.760|+12.840|+15.040|
|Cora|GraphMAE2|cons_vat_0.1|5|70.760|+12.840|+15.040|
|Cora|GraphMAE2|cons_vat_0.3|5|70.720|+12.800|+15.000|
|Cora|GraphMAE2|frozen|5|57.920|+0.000|+2.200|
|Cora|GraphMAE2|graphprompt|5|55.720|-2.200|+0.000|
|Cora|GraphMAE2|rect_teacher_ref50|5|71.360|+13.440|+15.640|
|Cora|GraphMAE2|ridge_teacher_plain|5|70.560|+12.640|+14.840|
|PubMed|GraphMAE|cons_none_0|5|73.080|+3.120|-0.480|
|PubMed|GraphMAE|cons_rpt_0.1|5|73.080|+3.120|-0.480|
|PubMed|GraphMAE|cons_rpt_0.3|5|73.080|+3.120|-0.480|
|PubMed|GraphMAE|cons_vat_0.1|5|73.080|+3.120|-0.480|
|PubMed|GraphMAE|cons_vat_0.3|5|73.560|+3.600|+0.000|
|PubMed|GraphMAE|frozen|5|69.960|+0.000|-3.600|
|PubMed|GraphMAE|graphprompt|5|73.560|+3.600|+0.000|
|PubMed|GraphMAE|rect_teacher_ref50|5|67.920|-2.040|-5.640|
|PubMed|GraphMAE|ridge_teacher_plain|5|71.720|+1.760|-1.840|
|PubMed|GraphMAE2|cons_none_0|5|74.760|+8.880|+6.920|
|PubMed|GraphMAE2|cons_rpt_0.1|5|74.720|+8.840|+6.880|
|PubMed|GraphMAE2|cons_rpt_0.3|5|74.760|+8.880|+6.920|
|PubMed|GraphMAE2|cons_vat_0.1|5|74.600|+8.720|+6.760|
|PubMed|GraphMAE2|cons_vat_0.3|5|74.640|+8.760|+6.800|
|PubMed|GraphMAE2|frozen|5|65.880|+0.000|-1.960|
|PubMed|GraphMAE2|graphprompt|5|67.840|+1.960|+0.000|
|PubMed|GraphMAE2|rect_teacher_ref50|5|67.920|+2.040|+0.080|
|PubMed|GraphMAE2|ridge_teacher_plain|5|71.720|+5.840|+3.880|
