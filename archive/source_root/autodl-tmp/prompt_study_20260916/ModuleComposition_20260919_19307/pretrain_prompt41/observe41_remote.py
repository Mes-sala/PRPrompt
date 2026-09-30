from pathlib import Path
import json,subprocess,time,shutil,hashlib
p=Path('/root/autodl-tmp/prompt_study_20260916/ModuleComposition_20260919_19307/pretrain_prompt41')
r={'time':time.time(),'pretraining':{},'downstream':{},'space_free_bytes':shutil.disk_usage(p).free}
for stage in ['masked','edge','multig']:
 folder=p/(stage+'_training');records=list(folder.glob('*/*/record.json'))
 r['pretraining'][stage]={'completed_records':len(records)}
 for suffix,file in [('progress',folder/'progress.json'),('complete',folder/'complete.json'),('terminal',p/(stage+'_production_terminal.json'))]:
  if file.exists():r['pretraining'][stage][suffix]=json.loads(file.read_text())
for name in ['downstream_coordinator_progress.json','downstream_production_process.json','downstream_production_terminal.json','downstream/progress.json','downstream/complete.json','downstream_failure.json','downstream_coordinator_failure.json']:
 if (p/name).exists():r['downstream'][name]=json.loads((p/name).read_text())
for name in ['parallel_execution/processes.json','parallel_execution/progress.json','parallel_execution/terminal.json','parallel_execution/driver_failure.json','parallel_execution/final_audit.json','parallel_migration/start.json']:
 if (p/name).exists():r['downstream'][name]=json.loads((p/name).read_text())
pids=[2360,2894]
if (p/'parallel_execution/processes.json').exists():
 current=json.loads((p/'parallel_execution/processes.json').read_text());pids+=[current['driver_pid']]+[w['pid'] for w in current['workers']]
 r['execution_mode']='four_disjoint_cell_workers; old producer termination is intentional'

if (p/'downstream_production_process.json').exists():pids.append(json.loads((p/'downstream_production_process.json').read_text())['child_pid'])
r['actual_processes']=subprocess.run(['ps','-p',','.join(map(str,pids)),'-o','pid,etime,stat,args'],text=True,capture_output=True).stdout
r['gpu']=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True,capture_output=True).stdout
print(json.dumps(r))
