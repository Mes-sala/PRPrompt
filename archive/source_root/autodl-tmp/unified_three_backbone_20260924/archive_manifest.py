from pathlib import Path
import hashlib,json,sys,time
R=Path(__file__).resolve().parent;target=R/'archive_manifest.json'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
if '--verify' in sys.argv:
 m=json.loads(target.read_text());errors=[]
 for r in m['files']:
  p=R/r['path']
  if not p.is_file() or p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:errors.append(r['path'])
 report={'passed':not errors,'files':len(m['files']),'bytes':sum(r['bytes'] for r in m['files']),'errors':errors,'manifest_sha256':sha(target)}
 (R/'archive_transport_verified.json').write_text(json.dumps(report,indent=2));print(report);assert report['passed']
else:
 for n in ['pipeline_status.json','extension2026/pipeline_status.json']:
  assert json.loads((R/n).read_text())['status']=='complete_audited'
 assert json.loads((R/'metrics_pipeline_status.json').read_text())['status']=='complete'
 exclude={'archive_manifest.json','archive_transport_verified.json'}
 rows=[]
 for p in sorted(R.rglob('*')):
  rel=p.relative_to(R)
  if p.is_file() and not p.is_symlink() and not any(c in ['.git','__pycache__'] for c in rel.parts) and str(rel) not in exclude:
   rows.append({'path':str(rel),'bytes':p.stat().st_size,'sha256':sha(p)})
 target.write_text(json.dumps({'created':time.time(),'excludes':['.git','__pycache__','manifest itself','transport receipt'],'files':rows},ensure_ascii=False,indent=2));print('manifest files',len(rows),'bytes',sum(r['bytes'] for r in rows))
