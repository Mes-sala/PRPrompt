"""Full final contract audit, run with the ProG Python on server, or --transport locally."""
import json,hashlib,argparse,collections,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--transport',action='store_true');args=p.parse_args()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
plan=json.loads((ROOT/'plan.json').read_text());assert sha(ROOT/'grid.py')==plan['runner_sha256'];assert sha(ROOT/'source_manifest.json')==plan['source_manifest_sha256']
for r in json.loads((ROOT/'source_manifest.json').read_text()):assert sha(ROOT/r['path'])==r['sha256']
if not args.transport:
 import sys,torch
 sys.path.insert(0,str(ROOT));import grid
 datasets={n:grid.data_load(n) for n in grid.NODES}
rows=0;bad=[];splits={};modelsha={}
for job in plan['jobs']:
 d=ROOT/'results'/job['id'];methods=plan['direct_methods'] if job['direct'] else plan['methods'];expected=len(methods)*len(plan['seeds'])
 try:complete=json.loads((d/'complete.json').read_text());assert complete['results']==expected
 except Exception as e:bad.append(dict(job=job['id'],error='missing/invalid complete marker '+repr(e)))
 if not args.transport and not job['direct']:
  ck=job['checkpoint'];assert sha(Path(ck['path']))==ck['sha256'];modelsha[job['id']]=ck['sha256']
 for seed in plan['seeds']:
  for method in methods:
   f=d/f'{seed}__{method}'
   try:
    r=json.loads((f/'result.json').read_text());assert r['method']==method and r['seed']==seed and r['dataset']==job['dataset'];assert r['steps']==100 and r['shots']==5 and not r['smoke'];assert r['runner_sha256']==plan['runner_sha256'];assert r['source_manifest_sha256']==plan['source_manifest_sha256'];assert sha(f/'sealed.pt')==r['prediction_sha256']
    assert r['checkpoint_sha256']==(None if job['direct'] else job['checkpoint']['sha256'])
    key=(job['dataset'],seed)
    if key in splits:assert splits[key]==r['split_sha256']
    splits[key]=r['split_sha256']
    for label in ['neighbor','geometry']:
     t=r['trace']
     if label+'_file' in t:assert sha(d/t[label+'_file'])==t[label+'_sha256']
    if not job['direct']:assert r['frozen_verified']
    if not args.transport:
     q=torch.load(f/'sealed.pt',map_location='cpu',weights_only=True);data,pool,query,C,protocol=datasets[job['dataset']]
     assert torch.equal(q['query'],query) and not torch.isin(q['support'],query).any();assert torch.isin(q['support'],pool).all();assert torch.all(torch.bincount(data.y[q['support']],minlength=C)==5)
     gen=torch.Generator().manual_seed(seed);expected_support=torch.cat([ix[torch.randperm(len(ix),generator=gen)[:5]] for c in range(C) for ix in [pool[data.y[pool]==c]]]);assert torch.equal(q['support'],expected_support)
     assert hashlib.sha256((grid.a1.thash(q['support'])+grid.a1.thash(query)).encode()).hexdigest()==r['split_sha256'];assert torch.isfinite(q['scores']).all();assert torch.equal(q['prediction'],q['scores'].argmax(-1));assert abs(float((q['prediction']==data.y[query]).float().mean())-r['accuracy'])<1e-8
    rows+=1
   except Exception as e:bad.append(dict(job=job['id'],seed=seed,method=method,error=repr(e)))
assert rows<=plan['expected_results'];report=dict(passed=not bad and rows==plan['expected_results'],verified_rows=rows,expected_rows=plan['expected_results'],errors=bad,paired_splits=len(splits),checkpoints_verified=len(modelsha),transport_only=args.transport,finished=time.time(),plan_sha256=sha(ROOT/'plan.json'))
(ROOT/('transport_audit.json' if args.transport else 'final_audit.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='errors'}));assert report['passed'],bad[:5]
