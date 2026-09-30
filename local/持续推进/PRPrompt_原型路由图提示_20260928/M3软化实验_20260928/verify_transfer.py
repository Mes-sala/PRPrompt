from pathlib import Path
import hashlib,json,tarfile
r=Path(__file__).resolve().parent;a=r/'prprompt_softening_transfer_20260928.tar.gz'
assert hashlib.sha256(a.read_bytes()).hexdigest()=='4b8b6fedea5570c57a4a570cc3bc658222c1761346c9c912482d88d75f84b065'
with tarfile.open(a) as t:t.extractall(r,filter='data')
m=json.loads((r/'transfer_manifest.json').read_text())
for x in m:
 p=r/x['path'];assert p.stat().st_size==x['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256'],x['path']
report=dict(files=len(m),bytes=sum(x['bytes'] for x in m),missing=0,mismatch=0,archive_sha256=hashlib.sha256(a.read_bytes()).hexdigest())
(r/'transfer_verified.json').write_text(json.dumps(report,indent=2));print(report)
