"""Report committed job coverage only, without reading predictions or query labels."""
from pathlib import Path
import json,time,hashlib,os,shutil
R=Path(__file__).resolve().parent
cfg=json.loads((R/'downstream_freeze.json').read_text())
freeze=hashlib.sha256((R/'downstream_freeze.json').read_bytes()).hexdigest()
rows=[];errors=[]
for cell in cfg['cells']:
 d=R/'downstream'/cell['dataset']/cell['encoder'];expected={f'episode{ei:02d}__{m}__lr{lr:g}' for ei in cfg['episode_indices'] for m in cfg['prompt_methods'] for lr in cfg['learning_rates']}
 present={f.parent.name for f in d.glob('episode*/record.json')}
 if present-expected:errors.append({'unexpected':sorted(present-expected),'cell':str(d)})
 size=0
 for n in sorted(present):
  rec=json.loads((d/n/'record.json').read_text())
  if rec['freeze_sha256']!=freeze or rec['checkpoint_sha256']!=cell['checkpoint_sha256'] or rec['query_labels_read'] is not False:errors.append({'record_contract':str(d/n)})
  size+=sum(f.stat().st_size for f in (d/n).iterdir() if f.is_file())
 rows.append({'dataset':cell['dataset'],'encoder':cell['encoder'],'jobs':len(present),'expected':len(expected),'baseline_present':(d/'no_prompt/record.json').exists(),'committed_bytes':size})
result={'status':'coverage_observed' if not errors else 'coverage_errors','time':time.time(),'cells':rows,'committed_jobs':sum(r['jobs'] for r in rows),'complete_cells':sum(r['jobs']==r['expected'] and r['baseline_present'] for r in rows),'total_jobs':cfg['total_jobs'],'errors':errors,'disk_free_bytes':shutil.disk_usage(R).free,'query_labels_read':False,'scope':'Commit records and file sizes only; SHA and numerical verification are separate.'}
t=R/'coverage_summary.tmp';t.write_text(json.dumps(result,indent=2));os.replace(t,R/'coverage_summary.json')
print(json.dumps({k:v for k,v in result.items() if k!='cells'}))
