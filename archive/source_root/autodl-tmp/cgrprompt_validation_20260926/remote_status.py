from pathlib import Path
import json,time,shutil
R=Path(__file__).resolve().parent
s=json.loads((R/'supervisor_status.json').read_text());workers=s.get('workers',[dict(pid=s['pid'])] if 'pid' in s else [])
for w in workers:
 p=Path('/proc')/str(w['pid']);w['alive']=p.exists() and (p/'stat').read_text().split()[2]!='Z';w['command']=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode() if p.exists() else None
 if 'job' in w:w['completed']=len(list((R/'tuning'/w['job']).glob('*/result.json')))
counts={stage:len(list((R/stage).glob('*/*/result.json'))) for stage in ['core','allocation','tuning','readout_confirmation']}
report=dict(observed=time.time(),scheduler=s,workers=workers,counts=counts,free_gib=round(shutil.disk_usage(R).free/2**30,2),complete={stage:(R/(stage+'_complete.json')).exists() for stage in ['core','allocation','tuning','cost','readout_confirmation']});print(json.dumps(report,ensure_ascii=False))
