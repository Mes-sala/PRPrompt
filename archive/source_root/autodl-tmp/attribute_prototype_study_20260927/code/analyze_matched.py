"""Paired readout/geometry comparisons and dataset-balanced aggregates."""
import json,argparse
from pathlib import Path
import numpy as np

def main(root):
 root=Path(root);rows=[json.loads(f.read_text()) for f in sorted(root.glob('*/*/result.json'))];assert len(rows)==json.loads((root/'complete.json').read_text())['expected']
 groups={}
 for x in rows:groups.setdefault(x.get('checkpoint_id',x['dataset']),{}).setdefault(x['method'],{})[x['seed']]=x
 frozen='raw_refine' in next(iter(groups.values()))
 pairs=([('zca_refine','raw_refine'),('cv_frozen','raw_refine'),('cv_frozen','zca_refine'),('cv_frozen','cv_frozen_no_refine')] if frozen else [('cv_anchored','c0_base')]+[(f'cv_{m}','cv_anchored') for m in ['simpleshot_style','soft_kmeans_adapted','laplacianshot_adapted']]+[(f'cv_{m}',f'anchored_on_{m}') for m in ['simpleshot_style','soft_kmeans_adapted','laplacianshot_adapted']])
 rng=np.random.default_rng(20260927);out=[]
 for unit,methods in groups.items():
  for a,b in pairs:
   assert methods[a].keys()==methods[b].keys();aa=[methods[a][s] for s in sorted(methods[a])];bb=[methods[b][s] for s in sorted(methods[b])];assert all(x['split_sha256']==y['split_sha256'] for x,y in zip(aa,bb))
   dd=np.array([x['accuracy']-y['accuracy'] for x,y in zip(aa,bb)]);ci=np.quantile(dd[rng.integers(0,len(dd),(10000,len(dd)))].mean(1),[.025,.975]);cost=lambda rr:float(np.mean([x['seconds']+x.get('selection_seconds',0) for x in rr]))
   out.append(dict(unit=unit,dataset=aa[0]['dataset'],family=aa[0].get('family'),method=a,reference=b,n=len(dd),accuracy=float(np.mean([x['accuracy'] for x in aa])),reference_accuracy=float(np.mean([x['accuracy'] for x in bb])),delta=float(dd.mean()),ci_low=float(ci[0]),ci_high=float(ci[1]),seconds_inference_plus_selection=cost(aa),reference_seconds=cost(bb),knn_preparation_once_seconds=aa[0].get('knn_preparation_seconds',0)))
 dest=root/'analysis';dest.mkdir(exist_ok=True);(dest/'matched.json').write_text(json.dumps(out,indent=2));lines=['# 匹配对比','', '差值方向为方法减参照。区间为各设置内同支持抽样配对 bootstrap，未校正多重比较；不代表独立图验证。时间包括推断与支持集选择，最近邻缓存单次构建时间单列。','', '|方法|参照|权重/图设置等权差|数据集等权差|区间正/负|','|---|---|---:|---:|---|']
 for a,b in pairs:
  ss=[x for x in out if x['method']==a and x['reference']==b];ds={d:float(np.mean([x['delta'] for x in ss if x['dataset']==d])) for d in sorted({x['dataset'] for x in ss})};lines.append(f"|{a}|{b}|{np.mean([x['delta'] for x in ss]):+.3f}|{np.mean(list(ds.values())):+.3f}|{sum(x['ci_low']>0 for x in ss)}/{sum(x['ci_high']<0 for x in ss)}|");print(a,'minus',b,'dataset balanced',round(float(np.mean(list(ds.values()))),3),ds)
 lines+=['','|设置|方法|参照|Accuracy|差 pp|95%区间|推断+选择秒|近邻预计算秒|','|---|---|---|---:|---:|---|---:|---:|']
 for x in out:lines.append(f"|{x['unit']}|{x['method']}|{x['reference']}|{x['accuracy']:.3f}|{x['delta']:+.3f}|[{x['ci_low']:.3f}, {x['ci_high']:.3f}]|{x['seconds_inference_plus_selection']:.3f}|{x['knn_preparation_once_seconds']:.3f}|")
 (dest/'匹配对比.md').write_text('\n'.join(lines))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('root');main(ap.parse_args().root)
