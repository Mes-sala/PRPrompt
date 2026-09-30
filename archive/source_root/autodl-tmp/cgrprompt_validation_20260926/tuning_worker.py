from pathlib import Path
import sys,json,hashlib
R=Path(__file__).resolve().parent
lock=json.loads((R/'tuning_plan.json').read_text())
for name,key in [('run_tuning.py','runner_sha256'),('allocation.py','operator_sha256')]:assert hashlib.sha256((R/name).read_bytes()).hexdigest()==lock[key]
from run_tuning import run
job=next(j for j in json.loads((R/'plan.json').read_text())['jobs'] if j['id']==sys.argv[1]);assert job['phase']=='main'
run(job)
