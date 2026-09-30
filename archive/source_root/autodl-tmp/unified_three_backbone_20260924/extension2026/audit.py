from pathlib import Path
import argparse,json,hashlib,sys
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'prog_src'))
import torch
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def run(smoke=False,transport=False):
 plan=json.loads((R/'plan.json').read_text());errors=[];count=0;checks=[];cross_splits={};datasets={}
 assert sha(R/'grid.py')==plan['runner_sha256'];assert sha(R/'source_manifest.json')==plan['source_manifest_sha256']
 for f in json.loads((R/'source_manifest.json').read_text()):assert sha(R/f['path'])==f['sha256'],f
 for j in plan['jobs']:
  if not transport and j['dataset'] not in datasets:
   from grid import data_load
   datasets[j['dataset']]=data_load(j['dataset'])[0]
  if not transport:assert sha(Path(j['checkpoint']['path']))==j['checkpoint']['sha256']
  for seed in (plan['seeds'][:1] if smoke else plan['seeds']):
   splits=set();graphs=set();prompts={}
   for m in plan['methods']:
    p=R/('smoke' if smoke else 'results')/j['id']/f'{seed}__{m}'
    try:
     d=json.loads((p/'result.json').read_text());assert sha(p/'sealed.pt')==d['prediction_sha256'];assert d['checkpoint_sha256']==j['checkpoint']['sha256'];assert d['runner_sha256']==plan['runner_sha256'];assert d['source_manifest_sha256']==plan['source_manifest_sha256']
     assert d['steps']==(2 if smoke else 100) and d['frozen_verified'];assert d['method']==m and d['seed']==seed and d['dataset']==j['dataset'];s=torch.load(p/'sealed.pt',map_location='cpu',weights_only=False)
     assert torch.isfinite(s['scores']).all() and torch.equal(s['prediction'],s['scores'].argmax(-1));assert not torch.isin(s['support'],s['query']).any();assert len(s['query'])==1000
     splits.add(d['split_sha256']);graphs.add(d['graph_sha256']);count+=1
     cross_splits.setdefault((j['dataset'],seed),set()).add(d['split_sha256'])
     if not transport:
      data=datasets[j['dataset']];assert abs(float((s['prediction']==data.y[s['query']]).float().mean())-d['accuracy'])<1e-8
      assert torch.all(torch.bincount(data.y[s['support']])==5)
      pool=data.train_mask.nonzero().flatten();expected_query=data.test_mask.nonzero().flatten();assert torch.equal(s['query'],expected_query)
      gen=torch.Generator().manual_seed(seed);expected_support=torch.cat([ix[torch.randperm(len(ix),generator=gen)[:5]] for c in range(int(data.y.max())+1) for ix in [pool[data.y[pool]==c]]]);assert torch.equal(s['support'],expected_support)
     if not m.startswith('F'):
      assert len(d['trace']['training_loss'])==(2 if smoke else 100);assert len(d['trace']['gradient_norm'])==(2 if smoke else 100);assert max(d['trace']['gradient_norm'])>0
      assert d['trace']['query_labels_received'] is False
     if m in ['F110','F111']:prompts[m]=s['prompt']
    except Exception as e:errors.append({'job':j['id'],'seed':seed,'method':m,'error':repr(e)})
   if len(splits)>1 or len(graphs)>1:errors.append({'job':j['id'],'seed':seed,'error':'unpaired inputs'})
   if len(prompts)==2:
    for k in prompts['F110']:
     a,b=prompts['F110'][k],prompts['F111'][k]
     if torch.is_tensor(a):assert torch.equal(a,b),(j['id'],seed,k)
   checks.append({'job':j['id'],'seed':seed,'shared_graph_and_split':len(splits)==len(graphs)==1})
 assert all(len(v)==1 for v in cross_splits.values()),'different support across backbones'
 expected=len(plan['jobs'])*len(plan['methods'])*(1 if smoke else len(plan['seeds']))
 report={'passed':not errors and count==expected,'count':count,'expected':expected,'errors':errors,'pairs':checks,'transport':transport}
 name='smoke_audit.json' if smoke else ('transport_audit.json' if transport else 'final_audit.json');(R/name).write_text(json.dumps(report,indent=2));print({k:v for k,v in report.items() if k!='pairs'});assert report['passed']
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--transport',action='store_true');a=ap.parse_args();run(a.smoke,a.transport)
