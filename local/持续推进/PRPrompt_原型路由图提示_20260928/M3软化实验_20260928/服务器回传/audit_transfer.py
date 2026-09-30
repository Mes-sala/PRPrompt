from pathlib import Path
import json,hashlib
r=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((r/'transfer_manifest.json').read_text());paths={x['path']:x['sha256'] for x in m}
for x in json.loads((r/'base/source_manifest.json').read_text()):paths['base/'+x['path']]=x['sha256']
for p,h in paths.items():assert sha(r/p)==h,p
out=dict(verified_unique_files=len(paths),verified_bytes=sum((r/p).stat().st_size for p in paths),missing=0,mismatch=0,source_manifest_sha256=sha(r/'base/source_manifest.json'),archives={p.name:sha(p) for p in r.glob('*.tar.gz')})
(r/'transfer_verified_full.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
