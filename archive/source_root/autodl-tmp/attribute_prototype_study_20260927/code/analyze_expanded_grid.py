"""Setting-specific module contrasts; support-ranked shortlist distinct from query oracle."""
import argparse,json,sqlite3,hashlib,csv
from pathlib import Path
from collections import defaultdict
import numpy as np

def write_csv(path,rows):
 if not rows:return
 with path.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def analyze(project,smoke=False):
 project=Path(project);bank=json.loads((project/'design/expanded_grid_candidate_bank.json').read_text());configs={kind:{x['id']:x for x in bank[kind]} for kind in ['attributes','frozen']};groups={};units={};raw={}
 folders=list((project/'results').glob('smoke_expanded_grid*' if smoke else 'grid_*'))
 for folder in folders:
  if not (folder/'complete.json').exists():continue
  for marker in folder.glob('*.complete.json'):
   meta=json.loads(marker.read_text());path=folder/(marker.name.removesuffix('.complete.json')+'.sqlite');assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['sha256']
   con=sqlite3.connect(f'file:{path}?mode=ro',uri=True);u=json.loads(con.execute("SELECT value FROM metadata WHERE key='unit'").fetchone()[0]);unit_id=u['id']
   # Smoke V1/V2 have the same unit and predictions: keep only one set.
   if unit_id in units:
    assert smoke;con.close();continue
   units[unit_id]=u;split={s:h for s,h in con.execute('SELECT seed,split_hash FROM splits')}
   for seed,cfg,val,acc,secs in con.execute('SELECT seed,config,validation,accuracy,seconds FROM results'):
    assert acc is not None;v=json.loads(val);raw[(unit_id,cfg,seed)]=dict(accuracy=acc,validation_accuracy=v['accuracy'],validation_nll=v['nll'],seconds=secs,split=split[seed]);groups.setdefault((unit_id,cfg),[]).append(seed)
   con.close()
 if not smoke:
  coverage=json.loads((project/'results/expanded_grid_dispatch/coverage_verified.json').read_text());assert set(units)=={u['id'] for u in coverage['units']},'Do not finalize an incomplete grid'
 summaries=[];contrasts=[]
 for (unit_id,cfg_id),seeds in groups.items():
  u=units[unit_id];cfg=configs[u['kind']][cfg_id];rr=[raw[(unit_id,cfg_id,s)] for s in sorted(seeds)]
  if not smoke:assert sorted(seeds)==bank['screen_seeds']
  summaries.append(dict(dataset=u['dataset'],unit=unit_id,family=u.get('family','attributes'),architecture=u.get('architecture','none'),config=cfg_id,n=len(rr),accuracy=float(np.mean([r['accuracy'] for r in rr])),support_accuracy=float(np.mean([r['validation_accuracy'] for r in rr])),support_nll=float(np.mean([r['validation_nll'] for r in rr])),seconds=float(np.mean([r['seconds'] for r in rr]))))
  if cfg['module'] in ['confidence','neighbor']:
   same=lambda c:all(c.get(k)==cfg.get(k) for k in ['geometry','hops','beta','tau','rounds'])
   candidates=[c for c in configs[u['kind']].values() if c['module']=='none' and same(c)]
   assert len(candidates)==1;ref=candidates[0]['id']
   if (unit_id,ref) not in groups:assert smoke;continue
   dd=[]
   for seed in seeds:
    a=raw[(unit_id,cfg_id,seed)];b=raw[(unit_id,ref,seed)];assert a['split']==b['split'];dd.append(a['accuracy']-b['accuracy'])
   contrasts.append(dict(dataset=u['dataset'],unit=unit_id,config=cfg_id,reference=ref,module=cfg['module'],n=len(dd),delta=float(np.mean(dd)),minimum_seed_delta=float(min(dd)),maximum_seed_delta=float(max(dd)),paired_deltas=json.dumps(dd),status='exploratory; requires new-support confirmation'))
 dest=project/'reports'/('grid_analysis_smoke' if smoke else 'expanded_grid_analysis');dest.mkdir(parents=True,exist_ok=True);write_csv(dest/'all_configurations.csv',summaries);write_csv(dest/'matched_module_effects.csv',contrasts)
 shortlist={};oracle=[]
 for dataset in sorted({x['dataset'] for x in summaries}):
  ss=[x for x in summaries if x['dataset']==dataset];ranked=sorted(ss,key=lambda x:(-x['support_accuracy'],x['support_nll'],x['unit'],x['config']));shortlist[dataset]=[dict(unit=x['unit'],config=x['config'],support_accuracy=x['support_accuracy'],support_nll=x['support_nll']) for x in ranked[:20]]
  best=max(ss,key=lambda x:x['accuracy']);oracle.append(dict(dataset=dataset,unit=best['unit'],config=best['config'],query_max_accuracy=best['accuracy'],status='query-selected exploratory upper bound; NOT confirmed performance'))
 (dest/'support_only_shortlist.json').write_text(json.dumps(dict(smoke=smoke,selection_uses_query_accuracy=False,datasets=shortlist),indent=2));write_csv(dest/'exploratory_query_maximum.csv',oracle);print(dict(units=len(units),configurations=len(summaries),module_contrasts=len(contrasts),smoke=smoke))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('project');ap.add_argument('--smoke',action='store_true');a=ap.parse_args();analyze(a.project,a.smoke)
