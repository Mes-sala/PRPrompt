"""Keep a local result mirror without checkpoints, state tensors or graph snapshots."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def main(a):
    a.local.mkdir(parents=True,exist_ok=True);a.control.mkdir(parents=True,exist_ok=True)
    lock=a.control/'sync.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        while True:
            command=['rsync','-az','--partial','--exclude=*.pt','--exclude=*.pth','--exclude=*.pt.gz',
                     '--exclude=*.ckpt','--exclude=*.tmp','--exclude=state.pt.gz*','--exclude=__pycache__/','--exclude=checkpoints/',
                     '--exclude=dataset_snapshots/','--exclude=input/','--exclude=*.lock',
                     '-e',f'ssh -S {a.socket} -o BatchMode=yes -o ConnectTimeout=15',a.remote.rstrip('/')+'/',str(a.local)+'/']
            with (a.control/'sync.log').open('a') as log:
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            status=dict(pid=os.getpid(),updated_at=time.time(),rsync_exit_code=result.returncode,
                        local=str(a.local),model_binaries_transferred=False)
            remote_status=a.local/'status.json'
            if remote_status.exists():status['experiment']=json.loads(remote_status.read_text()).get('phase')
            (a.control/'sync_status.json').write_text(json.dumps(status,indent=2)+'\n')
            if result.returncode==0 and status.get('experiment')=='complete':break
            time.sleep(300)
    finally:lock.unlink()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--local',type=Path,required=True);p.add_argument('--control',type=Path,required=True)
    p.add_argument('--remote',required=True);p.add_argument('--socket',required=True);main(p.parse_args())
