from pathlib import Path
import json,hashlib,shutil
R=Path(__file__).resolve().parent;B=Path('/root/autodl-tmp/unified_three_backbone_20260924');out=R/'reproduction';out.mkdir(exist_ok=True)
for part in ['sources','prog_src']:
 shutil.copytree(B/part,out/part,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for name in ['grid.py','operators.py','source_manifest.json','plan.json']:
 shutil.copy2(B/name,out/('historical_'+name))
index=[]
for j in json.loads((R/'plan.json').read_text())['jobs']:
 ck=j['checkpoint'];src=Path(ck['path']);dest=out/'checkpoints'/j['dataset']/(ck['sha256']+'.pth');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest);assert hashlib.sha256(dest.read_bytes()).hexdigest()==ck['sha256'];index.append(dict(job=j['id'],original_path=str(src),copy=str(dest.relative_to(R)),sha256=ck['sha256'],provenance=ck.get('provenance')))
for name in ['Cora','CiteSeer','PubMed']:
 shutil.copytree(Path('/root/ProG/data/Planetoid')/name,out/'data'/'Planetoid'/name,dirs_exist_ok=True)
for name in ['Actor','WikipediaNetwork']:
 src=Path('/root/autodl-tmp/accuracy_prompt_20260923/assets5090/data')/name
 if src.exists():shutil.copytree(src,out/'data'/name,dirs_exist_ok=True)
(out/'checkpoint_index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2))
files=[dict(path=str(f.relative_to(R)),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in sorted(out.rglob('*')) if f.is_file()];(R/'reproduction_manifest.json').write_text(json.dumps(files,indent=2));print('files',len(files),'bytes',sum(r['bytes'] for r in files))
