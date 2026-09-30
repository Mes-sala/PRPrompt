from pathlib import Path
import subprocess,sys,json,time
p=Path('/root/autodl-tmp/attribute_prototype_study_20260927')
for name in ['CS','Physics']:
 t=p/'checkpoints'/(name+'__GraphMAE2__GAT__seed42_norm')
 if (t/'ready.json').exists():continue
 with (t/'chunked_repair_train.log').open('w') as log:r=subprocess.run([sys.executable,str(p/'code/repair_gat_memory.py'),str(t/'job.json')],stdout=log,stderr=subprocess.STDOUT)
 if r.returncode:raise RuntimeError('GAT repair failed: '+name)
(p/'provenance/gat_repair_queue_complete.json').write_text(json.dumps({'finished':time.time()}))
with (p/'expanded_grid_dispatch_v6.log').open('w') as log:
 job=subprocess.Popen([sys.executable,str(p/'code/schedule_expanded_grid_v6.py')],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
(p/'provenance/dispatch_after_gat_repair.json').write_text(json.dumps({'pid':job.pid}))
