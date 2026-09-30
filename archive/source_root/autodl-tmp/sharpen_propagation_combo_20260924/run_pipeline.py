from pathlib import Path
import subprocess,json,os,time
R=Path(__file__).resolve().parent;PY='/root/autodl-tmp/prog_env/bin/python';p=json.loads((R/'plan.json').read_text())
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
def run(args):subprocess.run([PY,*args],cwd=R,env=env,check=True)
try:
 if not (R/'smoke_audit.json').exists():
  selected=[next(j for j in p['jobs'] if j['dataset']==ds and j['checkpoint']['family']=='GraphMAE') for ds in p['datasets']]
  for j in selected:run(['grid.py','run','--job',j['id'],'--smoke'])
  for j in selected:
   f=R/'smoke'/j['id']/f"{p['seeds'][0]}__Combo_temp02_mean"/'result.json';r=json.loads(f.read_text());assert r['smoke'] and r['trace']['replay_prediction_identical'] and r['trace']['new_training_steps']==0
  (R/'smoke_audit.json').write_text(json.dumps(dict(complete=True,passed=True,jobs=len(selected),new_training_fits=0)))
 run(['launch.py']);run(['audit_grid.py']);run(['audit_references.py']);run(['analyze_combo.py'])
 (R/'pipeline_status.json').write_text(json.dumps(dict(status='server_complete_audited',time=time.time())))
except Exception as e:
 (R/'pipeline_status.json').write_text(json.dumps(dict(status='failed',error=repr(e),time=time.time())));raise
