from pathlib import Path
import subprocess,os,time,json,fcntl,shutil,hashlib
R=Path(__file__).resolve().parent;PY='/root/autodl-tmp/prog_env/bin/python'
lock=open(R/'supervisor.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def write(x):
 q=R/'supervisor_status.json';tmp=q.with_suffix('.tmp');tmp.write_text(json.dumps(dict(x,time=time.time()),indent=2));tmp.replace(q)
while not (R/'core_complete.json').exists():
 try:
  s=Path('/proc/2699/stat').read_text();assert s.split()[2]!='Z';assert b'run_core.py' in Path('/proc/2699/cmdline').read_bytes()
 except Exception:write(dict(status='failed',stage='core',reason='core completion absent and original process terminal'));raise SystemExit(1)
 write(dict(status='waiting',stage='core',pid=2699));time.sleep(20)
for stage,script in [('allocation','run_allocation.py'),('tuning','run_tuning.py')]:
 if (R/f'{stage}_complete.json').exists():continue
 assert shutil.disk_usage(R).free>4*1024**3,'less than4GiB free'
 plan=json.loads((R/f'{stage}_plan.json').read_text());assert hashlib.sha256((R/script).read_bytes()).hexdigest()==plan['runner_sha256']
 with open(R/f'{stage}.log','a') as log:
  p=subprocess.Popen([PY,'-u',str(R/script)],cwd=R,stdout=log,stderr=log);write(dict(status='running',stage=stage,pid=p.pid));ret=p.wait()
 if ret:write(dict(status='failed',stage=stage,exit_code=ret));raise SystemExit(ret)
 assert (R/f'{stage}_complete.json').exists()
write(dict(status='complete',stage='core_allocation_tuning',remaining='local audit, dedicated timing, narrative update'))
