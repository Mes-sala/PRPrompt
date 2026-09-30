from pathlib import Path
import json,hashlib,subprocess,time
R=Path(__file__).resolve().parent;B=Path('/root/autodl-tmp/unified_three_backbone_20260924');old=json.loads((B/'plan.json').read_text());wide=json.loads(Path('/root/autodl-tmp/five_mainline_grid_20260924/plan.json').read_text());jobs=[]
for j in old['jobs']:
 if not j['direct']:jobs.append(dict(j,phase='main',seeds=list(range(24092800,24092820))))
for j in wide['jobs']:
 if j.get('direct'):continue
 d=j['dataset'];f=j['checkpoint']['family']
 if (d in ['Actor','Chameleon','PubMed'] and f in ['DGI','Edgepred_GPPT','Edgepred_Gprompt']) or (d in ['Cora','CiteSeer'] and f in ['GraphCL','GraphMAE']):jobs.append(dict(j,phase='extension',seeds=list(range(26092600,26092620))))
assert len(jobs)==21,len(jobs)
for j in jobs:assert hashlib.sha256(Path(j['checkpoint']['path']).read_bytes()).hexdigest()==j['checkpoint']['sha256']
files=[R/'run_core.py']+list((B/'sources').rglob('*.py'))+list((B/'prog_src').rglob('*.py'))+[B/'grid.py',B/'operators.py']
manifest=[dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files]
plan=dict(schema='cgr-validation-v1',created=time.time(),jobs=jobs,shots=5,steps=100,core_runner_sha256=hashlib.sha256((R/'run_core.py').read_bytes()).hexdigest(),description='6 main x 22 controls x20 +15 extension x7x20; other allocation/convergence experiments separate locked plan')
(R/'plan.json').write_text(json.dumps(plan,indent=2));(R/'source_manifest.json').write_text(json.dumps(manifest,indent=2));(R/'environment.txt').write_text(subprocess.check_output(['/root/autodl-tmp/prog_env/bin/python','-m','pip','freeze'],text=True))
print('jobs',len(jobs),'expected core rows',6*22*20+15*7*20)
