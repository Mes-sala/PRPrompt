"""Paired Accuracy and support-only choices for attributes versus frozen encoders."""
from pathlib import Path
import argparse,json,sqlite3,hashlib,csv
from collections import defaultdict
import numpy as np

def csv_save(path,rows):
 with path.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main(project):
 p=Path(project);coverage=json.loads((p/'results/expanded_grid_dispatch/coverage_verified.json').read_text());expected={u['id'] for u in coverage['units']};bank=json.loads((p/'design/expanded_grid_candidate_bank.json').read_text());cfgs={k:{c['id']:c for c in bank[k]} for k in ['attributes','frozen']};units={};data={}
 for folder in (p/'results').glob('backbone_value_*'):
  if not (folder/'complete.json').exists():continue
  for mpath in folder.glob('*.complete.json'):
   m=json.loads(mpath.read_text());db=folder/(mpath.name.removesuffix('.complete.json')+'.sqlite');assert hashlib.sha256(db.read_bytes()).hexdigest()==m['sha256'];con=sqlite3.connect(f'file:{db}?mode=ro',uri=True);u=json.loads(con.execute("SELECT value FROM metadata WHERE key='unit'").fetchone()[0]);units[u['id']]=u;split={s:h for s,h in con.execute('SELECT seed,split_hash FROM splits')}
   for seed,config,val,acc in con.execute('SELECT seed,config,validation,accuracy FROM results'):
    v=json.loads(val);c=cfgs[u['kind']][config];data[u['id'],c['geometry'],c['rounds'],seed]=dict(accuracy=acc,support_accuracy=v['accuracy'],support_nll=v['nll'],split=split[seed],unit=u['id'],geometry=c['geometry'],rounds=c['rounds'])
   con.close()
 assert set(units)==expected,'Comparison incomplete';seeds=list(range(271500,271520));rng=np.random.default_rng(271500);pairs=[];selected=[]
 def contrast(dataset,name,a,b,**extra):
  assert len(a)==len(b)==20 and all(x['split']==y['split'] for x,y in zip(a,b));dd=np.array([x['accuracy']-y['accuracy'] for x,y in zip(a,b)]);ci=np.quantile(dd[rng.integers(0,20,(10000,20))].mean(1),[.025,.975]);return dict(dataset=dataset,comparison=name,accuracy=float(np.mean([x['accuracy'] for x in a])),attribute_accuracy=float(np.mean([x['accuracy'] for x in b])),delta=float(dd.mean()),ci_low=float(ci[0]),ci_high=float(ci[1]),positive_draws=int((dd>0).sum()),negative_draws=int((dd<0).sum()),**extra)
 for dataset in sorted({u['dataset'] for u in units.values()}):
  attr=dataset+'_attributes';subset=[u for u in units.values() if u['dataset']==dataset]
  for u in subset:
   if u['kind']!='frozen':continue
   for geometry in ['raw','center']:
    for rounds in [0,3]:pairs.append(contrast(dataset,'same_readout_frozen_minus_attribute',[data[u['id'],geometry,rounds,s] for s in seeds],[data[attr,geometry,rounds,s] for s in seeds],unit=u['id'],family=u['family'],architecture=u['architecture'],geometry=geometry,rounds=rounds))
  def pick(eligible,seed):
   candidates=[data[u['id'],g,n,seed] for u in eligible for g in ['raw','center'] for n in [0,3]]
   return min(candidates,key=lambda x:(-x['support_accuracy'],x['support_nll'],x['unit'],x['geometry'],x['rounds']))
  attrs=[u for u in subset if u['kind']=='attributes'];baseline=[pick(attrs,s) for s in seeds]
  choices={'GCN':[u for u in subset if u.get('architecture')=='GCN'],'GraphSAGE':[u for u in subset if u.get('architecture')=='GraphSAGE'],'GAT':[u for u in subset if u.get('architecture')=='GAT'],'any_frozen':[u for u in subset if u['kind']=='frozen'],'all_including_attributes':subset}
  for name,eligible in choices.items():
   if not eligible:continue
   aa=[pick(eligible,s) for s in seeds];row=contrast(dataset,name+'_support_selected_minus_attribute_support_selected',aa,baseline,chosen_units=json.dumps([x['unit'] for x in aa]),choices_use_query_labels=False);selected.append(row)
 out=p/'reports/backbone_value';out.mkdir(parents=True,exist_ok=True);csv_save(out/'same_readout_pairs.csv',pairs);csv_save(out/'support_selected_comparison.csv',selected);(out/'support_selected_comparison.json').write_text(json.dumps(selected,indent=2))
 lines=['# 固定属性传播与冻结底座的保留价值','', '14数据集、5-shot、20次新支持抽样。每条路线用支持内5折选择原始/中心化及是否更新原型；固定属性为两步传播。尚不包含后续扩大网格的各自充分优化。区间为同图支持抽样配对bootstrap，未作多重校正。输入预处理遵循各权重契约，比较的是完整表示路线，不单独证明预训练这一因素的因果作用。','', '|数据集|属性路线Acc|支持选GCN Acc|差 pp|95%区间|','|---|---:|---:|---:|---|']
 for x in selected:
  if x['comparison'].startswith('GCN_support'):lines.append(f"|{x['dataset']}|{x['attribute_accuracy']:.3f}|{x['accuracy']:.3f}|{x['delta']:+.3f}|[{x['ci_low']:.3f},{x['ci_high']:.3f}]|")
 lines+=['','同后处理的每个权重配对结果见same_readout_pairs.csv。支持选择的其他架构与允许属性回退的结果见support_selected_comparison.csv。优势不明确的设置不据此宣称冻结底座必要；最终保留建议还需合并各自优化后的确认结果。'];(out/'对比报告.md').write_text('\n'.join(lines));print('units',len(units),'fixed pairs',len(pairs),'selected comparisons',len(selected))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('project');main(ap.parse_args().project)
