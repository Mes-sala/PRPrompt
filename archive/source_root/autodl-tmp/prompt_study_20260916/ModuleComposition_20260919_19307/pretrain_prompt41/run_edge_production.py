from pathlib import Path
import subprocess,os,json,time,sys
p=Path(__file__).resolve().parent
ad=json.loads((p/'edge_smoke/complete.json').read_text());assert ad['status']=='complete' and ad['count']==12
with (p/'edge_production.started').open('x') as f:f.write(str(os.getpid()))
env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
with (p/'edge_production.log').open('a') as log:
 child=subprocess.Popen([sys.executable,str(p/'train_edge41.py')],stdout=log,stderr=subprocess.STDOUT,env=env)
 (p/'edge_production_process.json').write_text(json.dumps({'wrapper_pid':os.getpid(),'child_pid':child.pid,'time':time.time(),'command':[sys.executable,str(p/'train_edge41.py')]}))
 rc=child.wait()
(p/'edge_production_terminal.json').write_text(json.dumps({'status':'complete' if rc==0 else 'failed','exit_code':rc,'time':time.time()}))
