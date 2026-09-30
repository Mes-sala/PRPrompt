import json,hashlib,tarfile
from pathlib import Path
root=Path('/root/autodl-tmp');base=root/'unified_three_backbone_20260924';val=root/'cgrprompt_validation_20260926'
plan=json.loads((val/'plan.json').read_text());jobs=[j for j in plan['jobs'] if j['phase']=='main' and j['checkpoint']['family'].startswith('Edgepred')];assert len(jobs)==4
files={}
def add(p,rel):
 if p.is_file():files[rel]=p
for sub in ['sources','prog_src']:
 for p in (base/sub).rglob('*'):
  if '__pycache__' not in p.parts:add(p,'base/'+str(p.relative_to(base)))
for s in ['grid.py','operators.py','source_manifest.json','plan.json']:add(base/s,'base/'+s)
for s in ['run_core.py','plan.json']:add(val/s,'reference/'+s)
for name in ['Cora','CiteSeer']:
 dp=Path('/root/ProG/data/Planetoid')/name
 for p in dp.rglob('*'):
  if '__pycache__' not in p.parts:add(p,'data/Planetoid/'+str(p.relative_to(dp.parent)))
for j in jobs:
 cp=Path(j['checkpoint']['path']);assert hashlib.sha256(cp.read_bytes()).hexdigest()==j['checkpoint']['sha256'];add(cp,'checkpoints/'+j['id']+'.pth')
 for p in (val/'core'/j['id']).glob('*__F111/result.json'):add(p,'reference/core/'+str(p.relative_to(val/'core')))
manifest=[dict(path=s,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),source=str(p)) for s,p in sorted(files.items())]
mp=root/'prprompt_softening_transfer_manifest.json';mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
archive=root/'prprompt_softening_transfer_20260928.tar.gz'
with tarfile.open(archive,'w:gz') as t:
 for s,p in files.items():t.add(p,arcname=s,recursive=False)
 t.add(mp,arcname='transfer_manifest.json')
print(json.dumps(dict(files=len(manifest),unpacked_bytes=sum(x['bytes'] for x in manifest),archive=str(archive),archive_bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())))
