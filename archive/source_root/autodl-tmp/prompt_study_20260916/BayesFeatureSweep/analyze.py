"""Descriptive paired summaries; never chooses a configuration from query scores."""
import json,csv,statistics,sys
from pathlib import Path
root=Path(sys.argv[1])
rows=json.loads((root/'metrics.json').read_text())
cells={}
for r in rows: cells.setdefault((r['dataset'],r['checkpoint'],r['method']),[]).append(r['accuracy']*100)
summary=[dict(dataset=d,checkpoint=c,method=m,mean=statistics.mean(v),support_sd=statistics.stdev(v)) for (d,c,m),v in cells.items()]
with (root/'summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
lookup={(r['dataset'],r['checkpoint'],r['seed'],r['method']):r['accuracy']*100 for r in rows}
paired=[];gate={}
for ds in ['Cora','CiteSeer']:
 checkpoints=sorted({r['checkpoint'] for r in rows if r['dataset']==ds})
 deltas={}
 for control in ['fixed','selected_simple','selected_label_shuffle','selected_feature_permute','diffused_raw','dual_raw']:
  values=[]
  for ck in checkpoints:
   seeds=sorted({r['seed'] for r in rows if r['dataset']==ds and r['checkpoint']==ck})
   v=[lookup[ds,ck,s,'selected_full']-lookup[ds,ck,s,control] for s in seeds]
   values+=v
   paired.append(dict(dataset=ds,checkpoint=ck,control=control,mean=statistics.mean(v),positive=sum(x>1e-9 for x in v),n=len(v),minimum=min(v),maximum=max(v)))
  deltas[control]=statistics.mean(values)
 positive_ck=sum(statistics.mean(cells[ds,c,'selected_full'])>statistics.mean(cells[ds,c,'fixed']) for c in checkpoints)
 passed=deltas['fixed']>=.5 and positive_ck>=3 and all(deltas[c]>0 for c in ['selected_simple','selected_label_shuffle','selected_feature_permute'])
 gate[ds]={'passed':passed,'positive_checkpoints':positive_ck,'paired_mean_pp':deltas}
with (root/'paired.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
(root/'gate.json').write_text(json.dumps(gate,indent=2)+'\n')
print(json.dumps(gate,indent=2))
