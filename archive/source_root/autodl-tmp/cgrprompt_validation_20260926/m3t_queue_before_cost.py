from pathlib import Path
import subprocess,json,time,fcntl,shutil
R=Path(__file__).resolve().parent
lock=open(R/'m3t_queue.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def status(**x):(R/'m3t_queue_status.json').write_text(json.dumps(dict(time=time.time(),**x),indent=2))
try:
 while not (R/'readout_confirmation_complete.json').exists():
  s=json.loads((R/'supervisor_status.json').read_text());assert s['status']!='failed',s
  p=Path('/proc/9526');assert p.exists() and (p/'stat').read_text().split()[2]!='Z' and b'parallel_finish.py' in (p/'cmdline').read_bytes()
  status(status='waiting',wait_for='original readout confirmation after dedicated cost',supervisor_pid=9526);time.sleep(30)
 assert (R/'cost_complete.json').exists() and shutil.disk_usage(R).free>4*1024**3
 assert (R/'m3t_smoke_complete.json').exists()
 with open(R/'m3t_optimization.log','a') as log:
  p=subprocess.Popen(['/root/autodl-tmp/prog_env/bin/python','-u',str(R/'run_m3t_optimization.py')],cwd=R,stdout=log,stderr=log);status(status='running',pid=p.pid);ret=p.wait();assert ret==0,ret
 assert (R/'m3t_optimization_complete.json').exists();status(status='complete',remaining='local audit, candidate comparison and updated narrative')
except Exception as e:status(status='failed',error=repr(e));raise
