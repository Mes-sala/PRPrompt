"""Single-instance, non-deleting rsync mirror; no credentials or automatic experiment restart."""
import argparse
import fcntl
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

def run(args):
    args.destination.mkdir(parents=True, exist_ok=True); args.control.mkdir(parents=True, exist_ok=True)
    lock = (args.control/'sync.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (args.control/'sync.pid').write_text(str(os.getpid()))
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', '-S', args.socket]
    def status(**values):
        values.update(pid=os.getpid(), updated_at=time.time())
        p = args.control/'sync_status.json'; tmp = p.with_suffix('.tmp')
        tmp.write_text(json.dumps(values, indent=2)+'\n'); tmp.replace(p)
    while True:
        if shutil.disk_usage(args.destination).free < 10*1024**3:
            status(phase='waiting_for_local_space', reason='less than 10 GiB free; remote files retained')
            time.sleep(args.interval); continue
        start = time.time()
        cmd = ['rsync', '-az', '--partial', '--exclude=__pycache__', '--exclude=*.tmp', '--exclude=*.lock',
               '-e', ' '.join(ssh), args.host+':'+args.remote.rstrip('/')+'/', str(args.destination)+'/']
        with (args.control/'sync.log').open('a') as stream:
            stream.write(f'\n[{time.time()}] start incremental pull\n'); stream.flush()
            result = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
        status(phase='synced' if result.returncode == 0 else 'retry_pending', returncode=result.returncode,
               last_sync_started=start, last_sync_finished=time.time(), destination=str(args.destination))
        if args.once: return result.returncode
        time.sleep(args.interval)

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--host', required=True); p.add_argument('--socket', required=True)
    p.add_argument('--remote', required=True); p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--control', type=Path, required=True); p.add_argument('--interval', type=int, default=300); p.add_argument('--once', action='store_true')
    raise SystemExit(run(p.parse_args()))
