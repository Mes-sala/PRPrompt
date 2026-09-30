"""Wait on the known pretraining processes, bind all inputs, run the fixed grid."""
from downstream41_common import *
import os,time,subprocess,sys


def main():
    with (ROOT/'downstream_coordinator.started').open('x') as f:f.write(str(os.getpid()))
    prereqs=[('masked',9),('edge',12),('multig',3)]
    for name,count in prereqs:
        terminal=ROOT/(name+'_production_terminal.json')
        while not terminal.exists():
            process=json.loads((ROOT/(name+'_production_process.json')).read_text())
            # A status file alone is insufficient: the actual child must still exist.
            os.kill(process['child_pid'],0)
            write(ROOT/'downstream_coordinator_progress.json',{'status':'waiting_on_live_pretraining','pid':os.getpid(),'dependency':name,'child_pid':process['child_pid'],'time':time.time()})
            time.sleep(10)
        assert json.loads(terminal.read_text())['exit_code']==0,name
        cm=json.loads((ROOT/(name+'_training')/'complete.json').read_text())
        assert cm['status']=='complete' and cm['count']==count and not cm['smoke'],name
        for p in (ROOT/(name+'_training')).glob('*/*/record.json'):
            r=json.loads(p.read_text());assert r['epochs']==100 and digest(p.parent/'encoder.pth')==r['checkpoint_sha256']
    assert json.loads((ROOT/'downstream_admission.json').read_text())['count']==12
    cm=json.loads((ROOT/'downstream_smoke/complete.json').read_text());assert cm['status']=='sealed_unscored' and cm['count']==9
    cfg=json.loads((ROOT/'downstream_smoke_freeze.json').read_text());cfg.pop('engineering_recovery',None)
    cfg.update(episode_indices=list(range(20)),learning_rates=[.001,.01],horizons=[100,200],total_jobs=5040)
    cfg['cells']=[]
    for ds in ['ENZYMES','MUTAG','PROTEINS']:
        for name,kind,path in checkpoint_specs41(ds):
            cfg['cells'].append({'dataset':ds,'encoder':name,'kind':kind,'checkpoint':str(path.relative_to(BASE)),'checkpoint_sha256':digest(path)})
    assert len(cfg['cells'])==42
    cfg['production_protocol_sha256']=digest(ROOT/'正式下游比较协议.md')
    cfg['source_sha256']['正式下游比较协议.md']=cfg['production_protocol_sha256']
    cfg['admission_sha256']=digest(ROOT/'downstream_admission.json')
    cfg['prerequisite_complete_sha256']={n:digest(ROOT/(n+'_training')/'complete.json') for n,_ in prereqs}
    with (ROOT/'downstream_freeze.json').open('x') as f:json.dump(cfg,f,indent=2)
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with (ROOT/'downstream_production.log').open('a') as log:
        p=subprocess.Popen([sys.executable,str(ROOT/'run_downstream41.py')],stdout=log,stderr=subprocess.STDOUT,env=env)
        write(ROOT/'downstream_production_process.json',{'wrapper_pid':os.getpid(),'child_pid':p.pid,'time':time.time(),'command':[sys.executable,str(ROOT/'run_downstream41.py')]})
        rc=p.wait()
    write(ROOT/'downstream_production_terminal.json',{'status':'sealed_unscored' if rc==0 else 'failed','exit_code':rc,'time':time.time()})


if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        write(ROOT/'downstream_coordinator_failure.json',{'status':'failed','time':time.time(),'traceback':traceback.format_exc()});raise
