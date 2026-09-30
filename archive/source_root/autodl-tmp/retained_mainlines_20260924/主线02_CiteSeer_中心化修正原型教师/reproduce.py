#!/usr/bin/env python3
"""Print or execute sealed historical reproduction jobs in a prepared ProG server environment."""
import argparse,json,subprocess,sys
from pathlib import Path
p=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--python',default='/root/autodl-tmp/prog_env/bin/python');ap.add_argument('--execute',action='store_true');a=ap.parse_args()
plan=json.loads((p/'reproduce_plan.json').read_text());commands=[]
for job in plan['jobs']:
 out=Path(a.out)/job['name'];cmd=[a.python,str(p/'code'/job['script']),'--out',str(out)]+job['args'];commands.append((out,cmd))
print(json.dumps({'execute':a.execute,'commands':[c for _,c in commands]},ensure_ascii=False,indent=2))
if a.execute:
 subprocess.run([sys.executable,str(p/'verify_bundle.py')],check=True)
 assert Path(a.python).is_file(),'ProG interpreter missing'
 assert Path('/root/ProG').is_dir(),'Original /root/ProG runtime/data/checkpoints required'
 for out,_ in commands:assert not out.exists(),f'Refuse to overwrite existing output: {out}'
 for out,cmd in commands:
  out.parent.mkdir(parents=True,exist_ok=True);subprocess.run(cmd,cwd=p/'code',check=True)
