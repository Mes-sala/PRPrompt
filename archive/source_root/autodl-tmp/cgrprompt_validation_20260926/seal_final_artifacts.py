from pathlib import Path
import hashlib,json,time
R=Path(__file__).resolve().parent
stages=['core','allocation','tuning','cost','readout_confirmation','m3t_optimization','m3t_cost']
assert all((R/(s+'_complete.json')).exists() for s in stages),'Experiment stages incomplete'
expected={'core':4740,'allocation':2160,'tuning':840,'readout_confirmation':420,'m3t_optimization':420}
for stage,n in expected.items():assert len(list((R/stage).glob('*/*/result.json')))==n,(stage,n)
assert len(json.loads((R/'cost_results.json').read_text()))==225
assert len(json.loads((R/'m3t_cost_results.json').read_text()))==495
exclude={'final_artifact_manifest.json','final_transport_receipt.json'}
files=[]
for p in sorted(R.rglob('*')):
 if not p.is_file() or p.is_symlink() or '__pycache__' in p.parts:continue
 if p.name in exclude or p.suffix in ['.log','.lock','.tmp'] or p.name.endswith('_status.json') or 'current' in p.name or p.name=='.DS_Store':continue
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 files.append(dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=h.hexdigest()))
manifest=dict(created=time.time(),required_completed_stages=stages,expected_prediction_packages=expected,file_count=len(files),total_bytes=sum(f['bytes'] for f in files),excluded='volatile status, current pointers, logs, locks, temporary files and bytecode',files=files)
tmp=R/'final_artifact_manifest.tmp';tmp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2));tmp.replace(R/'final_artifact_manifest.json');print(json.dumps({k:v for k,v in manifest.items() if k!='files'}))
