from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files=[p for name in ['sources','prog_src','upstream'] for p in (R/name).rglob('*.py') if '.git' not in p.parts]+[R/'operators.py',R/'verify_operators.py']
manifest=[{'path':str(p.relative_to(R)),'sha256':sha(p)} for p in sorted(files)]
(R/'source_manifest.json').write_text(json.dumps(manifest,indent=2));p=json.loads((R/'plan_draft.json').read_text());p['source_manifest_sha256']=sha(R/'source_manifest.json');p['runner_sha256']=sha(R/'grid.py');(R/'plan.json').write_text(json.dumps(p,ensure_ascii=False,indent=2))
print('frozen',len(files),'sources')
