"""Bounded GPU workers; durable process status and explicit failure records."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from prprompt.protocol import save_json


def pipeline(w):
    p=json.loads((w/'plan.json').read_text());logs=w/'logs';logs.mkdir(exist_ok=True)
    status=dict(pid=os.getpid(),phase='starting',started_at=time.time(),completed=[])
    def update(**kw):status.update(kw,updated_at=time.time());save_json(w/'status.json',status)
    attempt=str(int(time.time()))
    def stage(smoke):
        phase='smoke' if smoke else 'formal'
        pending=[(j,k,s) for j in range(len(p['jobs'])) for k in p['shots'] for s in ([0] if smoke else range(p['shards']))]
        # Large graph goes first so CPU preparation can overlap small fits.
        pending.sort(key=lambda x: p['jobs'][x[0]]['dataset']!='ogbn-arxiv')
        active=[];failed=[]
        while pending or active:
            if not failed:
                for gpu in [0,1]:
                    while sum(t['gpu']==gpu for t in active)<(1 if smoke else 3):
                        # At most one arxiv worker per GPU; do not crowd existing users.
                        idx=next((i for i,t in enumerate(pending) if p['jobs'][t[0]]['dataset']!='ogbn-arxiv' or not any(a['gpu']==gpu and a['large'] for a in active)),None)
                        if idx is None:break
                        job,k,s=pending.pop(idx);label=f'{phase}_{job}_{k}_{s}'
                        f=(logs/(attempt+'_'+label+'.log')).open('x')
                        cmd=[sys.executable,'-m','experiments.prog_split_20261002.run',str(w),'--job',str(job),'--shots',str(k),'--shard',str(s),'--device',f'cuda:{gpu}']
                        if smoke:cmd+=['--smoke']
                        proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT)
                        active.append(dict(label=label,gpu=gpu,large=p['jobs'][job]['dataset']=='ogbn-arxiv',pid=proc.pid,proc=proc,stream=f))
            for t in list(active):
                code=t['proc'].poll()
                if code is None:continue
                t['stream'].close();active.remove(t)
                record={k:v for k,v in t.items() if k not in ['proc','stream']};record['exit_code']=code
                (failed if code else status['completed']).append(record)
            update(phase=phase,active=[{k:v for k,v in t.items() if k not in ['proc','stream']} for t in active],pending=len(pending),failures=failed)
            if failed and not active:raise RuntimeError(str(failed))
            if active:time.sleep(5)
        if smoke:
            rows=[json.loads(f.read_text()) for f in (w/'smoke').glob('*/*/*/*/*/metrics.json')]
            assert len(rows)==96
            save_json(w/'smoke_acceptance.json',dict(units=len(rows),finite=True,max_saved_readout_error=max(r['saved_readout_max_error'] for r in rows)))
    try:
        if not (w/'smoke_acceptance.json').exists():stage(True)
        stage(False)
        update(phase='summarizing')
        subprocess.run([sys.executable,'-m','experiments.prog_split_20261002.summarize',str(w)],check=True)
        update(phase='complete',finished_at=time.time())
    except Exception as e:update(phase='failed',error=repr(e));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('workspace',type=Path);pipeline(p.parse_args().workspace)
