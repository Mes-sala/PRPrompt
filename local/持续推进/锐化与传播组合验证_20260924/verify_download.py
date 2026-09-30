import pathlib,json,hashlib,concurrent.futures,time
p=pathlib.Path(__file__).resolve().parent;rows=json.loads((p/'artifact_manifest.json').read_text())['files']
def check(r):
 f=p/r['path']
 if not f.is_file():return {'path':r['path'],'error':'missing'}
 if f.stat().st_size!=r['bytes']:return {'path':r['path'],'error':'size'}
 h=hashlib.sha256()
 with f.open('rb') as s:
  for b in iter(lambda:s.read(4*1024*1024),b''):h.update(b)
 if h.hexdigest()!=r['sha256']:return {'path':r['path'],'error':'sha256'}
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:bad=[x for x in ex.map(check,rows) if x]
out=dict(passed=not bad,files=len(rows),bytes=sum(r['bytes'] for r in rows),bad=bad,finished=time.time());(p/'artifact_transport_audit.json').write_text(json.dumps(out,indent=2));print({k:v for k,v in out.items() if k!='bad'});assert not bad,bad[:10]
