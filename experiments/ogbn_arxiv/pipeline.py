"""Detached staged controller. Never restarts a failed task automatically."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
from prprompt.protocol import save_json,sha_file
from .common import ROOT, load_plan

def pipeline(w):
    lock=w/'pipeline.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    state=dict(pid=os.getpid(),started_at=time.time(),phase='initializing')
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    def update(**kw):
        state.update(kw,updated_at=time.time());save_json(w/'status.json',state)
    def command(module,args,log):
        update(phase=log)
        with (w/'logs'/f'{log}.log').open('x') as stream:
            result=subprocess.run([sys.executable,'-m',module]+args,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        assert result.returncode==0,dict(stage=log,exit_code=result.returncode)
    try:
        plan=load_plan(w);update(plan_sha256=plan['plan_sha256'])
        for device in ['cpu','cuda:0','cuda:1']:
            tag=device.replace(':','_')
            command('experiments.ogbn_arxiv.checks',['--device',device],'chunk_gradient_'+tag)
            command('experiments.f100_boundary_20261002.model',['--device',device],'recipe_parity_'+tag)
        command('experiments.ogbn_arxiv.pretrain',[str(w)],'pretrain')
        command('experiments.ogbn_arxiv.run',[str(w),'--smoke'],'smoke')
        smoke=[json.loads(p.read_text()) for p in (w/'smoke').glob('*/*/*/*/metrics.json')]
        assert len(smoke)==21
        peak=max(r['peak_reserved'] for r in smoke)
        assert peak<28*1024**3
        save_json(w/'memory_calibration.json',dict(smoke_units=21,peak_reserved=peak,workers_per_gpu=1,amp=False,tf32=False))
        active=[];update(phase='formal',active=[])
        for gpu in [0,1]:
            log=(w/'logs'/f'formal_{gpu}.log').open('x')
            child=subprocess.Popen([sys.executable,'-m','experiments.ogbn_arxiv.run',str(w),'--device',f'cuda:{gpu}','--shard',str(gpu)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            active.append((child,log))
        failed=[]
        while active:
            remaining=[]
            for child,log in active:
                rc=child.poll()
                if rc is None:remaining.append((child,log))
                else:
                    log.close()
                    if rc:failed.append(dict(pid=child.pid,exit_code=rc))
            active=remaining
            update(phase='formal_draining_failure' if failed else 'formal',active=[p.pid for p,_ in active],failed=failed,
                   units=len(list((w/'results').glob('*/*/*/*/metrics.json'))),expected=1260)
            if active:time.sleep(10)
        assert not failed,failed
        d=torch.load(w/'input/dataset.pt',map_location='cpu',weights_only=True)
        np.savez_compressed(w/'audit_reference.npz',labels=d['labels'].numpy(),pool=d['pool'].numpy(),query=d['query'].numpy())
        del d
        command('experiments.ogbn_arxiv.summarize',[str(w)],'summarize')
        with (w/'environment.txt').open('w') as f:subprocess.run([sys.executable,'-m','pip','freeze'],stdout=f,check=True)
        update(phase='server_complete',finished_at=time.time(),units=1260,active=[])
        manifest={}
        for p in sorted(w.rglob('*')):
            rel=p.relative_to(w)
            if not p.is_file() or p.is_symlink() or rel.parts[0]=='input' or '__pycache__' in rel.parts:continue
            if p.name in ['delivery_manifest.json','pipeline.log'] or p.suffix in ['.pt','.pth','.lock','.pyc','.pid']:continue
            manifest[str(rel)]=sha_file(p)
        save_json(w/'delivery_manifest.json',manifest)
    except Exception as e:
        update(phase='failed',error=repr(e));raise
    finally:
        lock.unlink()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('workspace',type=Path);pipeline(p.parse_args().workspace)
