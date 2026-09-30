import json,hashlib,argparse,csv
from pathlib import Path
import numpy as np

def analyze(root):
 root=Path(root);rows=[json.loads(p.read_text()) for p in sorted(root.glob('*/*/result.json'))]
 groups={}
 for r in rows:groups.setdefault((r.get('checkpoint_id',r['dataset']),r['method']),[]).append(r)
 summaries=[];rng=np.random.default_rng(20260927)
 for (unit,method),rr in sorted(groups.items()):
  rr=sorted(rr,key=lambda r:r['seed'])
  if method.startswith(('h0_c','h2_c')):ref='_'.join(method.split('_')[:2])+'_anchored'
  elif (unit,'raw_refine') in groups:ref='raw_refine'
  else:ref='c'+method[1]+'_base' if method.startswith(('c0_','c1_')) else 'c0_base'
  bb={r['seed']:r for r in groups.get((unit,ref),[])}
  pairs=[(r,bb[r['seed']]) for r in rr if r['seed'] in bb]
  assert all(a['split_sha256']==b['split_sha256'] for a,b in pairs)
  dd=np.array([a['accuracy']-b['accuracy'] for a,b in pairs]);ci=np.quantile(dd[rng.integers(0,len(dd),(10000,len(dd)))].mean(1),[.025,.975]) if len(dd) else [None,None]
  s=dict(unit=unit,dataset=rr[0]['dataset'],method=method,n=len(rr),accuracy=float(np.mean([r['accuracy'] for r in rr])),std=float(np.std([r['accuracy'] for r in rr],ddof=1)) if len(rr)>1 else 0,macro_f1=float(np.mean([r['macro_f1'] for r in rr])),seconds=float(np.mean([r['seconds'] for r in rr])),reference=ref,delta=float(dd.mean()) if len(dd) else None,ci_low=ci[0],ci_high=ci[1]);summaries.append(s)
 out=root/'analysis';out.mkdir(exist_ok=True)
 (out/'summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2))
 with (out/'summary.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
 lines=['# 阶段结果','',f'已读取 {len(rows)} 项。开发轮查询评价不等于独立确认。区间为同seed配对bootstrap，未作多重比较校正。','', '| 数据/权重 | 方法 | n | Accuracy | 相对匹配base | 95%区间 |','|---|---|---:|---:|---:|---|']
 for s in summaries:
  if s['delta'] is not None:lines.append(f"|{s['unit']}|{s['method']}|{s['n']}|{s['accuracy']:.3f}|{s['delta']:+.3f}|[{s['ci_low']:.3f}, {s['ci_high']:.3f}]|")
 (out/'结果.md').write_text('\n'.join(lines))
 print('rows',len(rows),'units',len({s['unit'] for s in summaries}))
 for method in sorted({s['method'] for s in summaries}):
  ss=[s for s in summaries if s['method']==method and s['delta'] is not None];print(method,round(float(np.mean([s['delta'] for s in ss])),3),'wins',sum(s['delta']>0 for s in ss),'/',len(ss))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('root');args=ap.parse_args();analyze(args.root)
