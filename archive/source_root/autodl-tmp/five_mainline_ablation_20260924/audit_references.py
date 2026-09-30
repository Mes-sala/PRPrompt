from pathlib import Path
import json,hashlib,concurrent.futures
R=Path(__file__).resolve().parent;plan=json.loads((R/'plan.json').read_text());B=Path(plan['parent_remote']);B=B if B.exists() else Path(plan['parent_local'])
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
assert sha(B/'plan.json')==plan['parent_plan_sha256'];refs=json.loads((R/'reused_reference_manifest.json').read_text());assert len(refs)==5475
lock=json.loads((R/'reference_lock.json').read_text());assert sha(R/'reused_reference_manifest.json')==lock['sha256']
old={}
def check(r):
 f=B/r['path'];assert sha(f)==r['result_sha256'];assert sha(f.parent/'sealed.pt')==r['sealed_sha256'];row=json.loads(f.read_text());assert row['split_sha256']==r['split_sha256'];return row
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
 for row in ex.map(check,refs):old[(row['dataset'],row['checkpoint_sha256'],row['seed'])]=row['split_sha256']
count=0
for f in (R/'results').glob('*/*/result.json'):
 r=json.loads(f.read_text());assert r['split_sha256']==old[(r['dataset'],r['checkpoint_sha256'],r['seed'])];count+=1
assert count==plan['expected_results'];result=dict(passed=True,old_references=len(refs),new_rows_paired=count,reference_manifest_sha256=lock['sha256'],parent_plan_sha256=plan['parent_plan_sha256']);(R/'reference_audit.json').write_text(json.dumps(result,indent=2));print(result)
