from pathlib import Path
import subprocess,json,os,time
R=Path(__file__).resolve().parent;PY='/root/autodl-tmp/prog_env/bin/python';p=json.loads((R/'plan.json').read_text());env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
def run(args):subprocess.run([PY,*args],cwd=R,env=env,check=True)
try:
 if not (R/'smoke_audit.json').exists():
  for j in p['jobs']:
   run(['grid.py','run','--job',j['id'],'--smoke'])
   d=R/'smoke'/j['id'];files=list(d.glob('*/result.json'));assert len(files)==len(p['direct_methods'] if j['direct'] else p['methods'])
   for f in files:
    r=json.loads(f.read_text());assert r['smoke'] and r['steps']==2
    if not j['direct']:assert r['frozen_verified']
  (R/'smoke_audit.json').write_text(json.dumps(dict(passed=True,complete=True,jobs=len(p['jobs']))))
 run(['launch.py']);run(['audit_grid.py']);run(['audit_pairs.py'])
 (R/'pipeline_status.json').write_text(json.dumps(dict(status='server_complete_audited',time=time.time())))
except Exception as e:
 (R/'pipeline_status.json').write_text(json.dumps(dict(status='failed',error=repr(e),time=time.time())));raise
