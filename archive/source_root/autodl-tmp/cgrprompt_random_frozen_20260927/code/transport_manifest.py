"""Hash completed remote artifacts for byte-identical local acceptance."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parents[1]
assert (R/'complete.json').exists()
manifest={}
for p in sorted(R.rglob('*')):
 if not p.is_file() or p.is_symlink():continue
 rel=p.relative_to(R)
 if '__pycache__' in rel.parts or p.suffix in ['.lock','.tmp'] or p.name=='transport_manifest.json':continue
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 manifest[str(rel)]=dict(bytes=p.stat().st_size,sha256=h.hexdigest())
(R/'transport_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
print(json.dumps(dict(files=len(manifest),bytes=sum(v['bytes'] for v in manifest.values()))))
