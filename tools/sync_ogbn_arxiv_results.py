"""Results-only transfer and independent local delivery verification (no checkpoints)."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.metric_utils import sha
from experiments.ogbn_arxiv.summarize import summarize

def save(p,v):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)

def once(a):
    a.local.mkdir(parents=True,exist_ok=True);a.control.mkdir(parents=True,exist_ok=True)
    base=dict(pid=os.getpid(),updated_at=time.time())
    cmd=['rsync','-a','--partial','--timeout=120','--exclude=/input/','--exclude=*.pt','--exclude=*.pth',
         '--exclude=*.pt.gz','--exclude=*.ckpt','--exclude=*.safetensors','--exclude=__pycache__/',
         '--exclude=*.pyc','--exclude=*.lock','--exclude=*.pid','--exclude=*.tmp',
         '-e',f'ssh -S {a.socket} -o BatchMode=yes -o ConnectTimeout=20',a.remote.rstrip('/')+'/',str(a.local)+'/']
    completed=subprocess.run(cmd)
    if completed.returncode:
        save(a.control/'sync_status.json',dict(base,phase='transport_error',exit_code=completed.returncode));return False
    statuspath=a.local/'status.json';status=json.loads(statuspath.read_text()) if statuspath.exists() else {}
    count=len(list((a.local/'results').glob('*/*/*/*/metrics.json')))
    if status.get('phase')!='server_complete' or not (a.local/'delivery_manifest.json').exists():
        save(a.control/'sync_status.json',dict(base,phase='server_failed' if status.get('phase')=='failed' else 'following',
            server=status,units=count,expected=1260));return False
    manifest=json.loads((a.local/'delivery_manifest.json').read_text())
    for name,h in manifest.items():
        p=a.local/name
        assert p.is_file() and sha(p)==h,name
    for pattern in ['*.pt','*.pth','*.pt.gz','*.ckpt','*.safetensors']:
        assert not list(a.local.rglob(pattern)),pattern
    summarize(a.local,local=True)
    result=json.loads((a.local/'local_delivery_status.json').read_text())
    result.update(files_verified=len(manifest),manifest_sha256=sha(a.local/'delivery_manifest.json'))
    save(a.local/'local_delivery_status.json',result)
    save(a.control/'sync_status.json',dict(base,phase='complete',delivery=result))
    return True

def main(a):
    a.control.mkdir(parents=True,exist_ok=True)
    lock=a.control/'sync.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        while True:
            try:
                done=once(a)
                if done or not a.watch:return
            except Exception as e:
                save(a.control/'sync_status.json',dict(pid=os.getpid(),updated_at=time.time(),phase='verification_error',error=repr(e)))
                print(repr(e),flush=True)
                if not a.watch:raise
            time.sleep(300)
    finally:lock.unlink()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--remote',required=True);p.add_argument('--socket',required=True)
    p.add_argument('--local',required=True,type=Path);p.add_argument('--control',required=True,type=Path);p.add_argument('--watch',action='store_true')
    main(p.parse_args())
