from pathlib import Path
import json,hashlib,concurrent.futures
R=Path(__file__).resolve().parent;p=json.loads((R/'plan.json').read_text())
def sha(f):
 h=hashlib.sha256()
 with f.open('rb') as s:
  for b in iter(lambda:s.read(4194304),b''):h.update(b)
 return h.hexdigest()
assert sha(R/'reference_manifest.json')==p['reference_manifest_sha256']
refs=json.loads((R/'reference_manifest.json').read_text());assert len(refs)==p['reference_count']
def check(r):
 root=Path(p[r['origin']+'_remote']);root=root if root.exists() else Path(p[r['origin']+'_local']);f=root/r['path']
 assert sha(f)==r['sha256'] and sha(f.parent/'sealed.pt')==r['sealed_sha256']
 row=json.loads(f.read_text());assert row['split_sha256']==r['split_sha256'];return row
with concurrent.futures.ThreadPoolExecutor(8) as ex:rows=list(ex.map(check,refs))
old={(r['dataset'],r['checkpoint_sha256'],r['seed'],r['method']):r for r in rows}
count=0
for f in (R/'results').glob('*/*/result.json'):
 r=json.loads(f.read_text());ref=old[r['dataset'],r['checkpoint_sha256'],r['seed'],'A3_gsh_qeps'];assert r['split_sha256']==ref['split_sha256'];assert r['trace']['gate_temperature']==.2 and r['trace']['optimizer82']=='gshampoo' and not r['trace']['kd_active'];count+=1
assert count==630
out=dict(passed=True,reference_rows=len(refs),new_readout_rows=count,new_training_fits=315,reference_manifest_sha256=p['reference_manifest_sha256']);(R/'reference_audit.json').write_text(json.dumps(out,indent=2));print(out)
