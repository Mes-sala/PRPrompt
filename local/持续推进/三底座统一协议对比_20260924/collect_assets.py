from pathlib import Path
import json,shutil,hashlib,subprocess,sys,platform
R=Path('/root/autodl-tmp/unified_three_backbone_20260924'); assets=R/'reproduction_assets';assets.mkdir(exist_ok=True)
records=[]
for j in json.loads((R/'plan.json').read_text())['jobs']:
 src=Path(j['checkpoint']['path']);dst=assets/'checkpoints'/j['dataset']/(j['checkpoint']['family']+'__'+j['checkpoint']['sha256'][:12]+src.suffix);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
 digest=hashlib.sha256(dst.read_bytes()).hexdigest();assert digest==j['checkpoint']['sha256'];records.append({'source':str(src),'copy':str(dst.relative_to(R)),'sha256':digest})
for ds in ['Cora','CiteSeer']:
 src=Path('/root/ProG/data/Planetoid')/ds;dst=assets/'data'/ds
 shutil.copytree(src,dst,dirs_exist_ok=True)
(assets/'checkpoint_index.json').write_text(json.dumps(records,indent=2))
(assets/'pip_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
(assets/'environment.json').write_text(json.dumps({'python':sys.version,'platform':platform.platform(),'gpu':subprocess.check_output(['nvidia-smi'],text=True)},indent=2))
print('snapshotted',len(records),'checkpoints and two datasets')
