from pathlib import Path
import json,hashlib,time
R=Path('/data1/shq/GP/PRPrompt_full_archive_20260929');A=R/'audit'
def read(p):return json.loads(p.read_text())
assert read(A/'status.json')['stage']=='complete'
assert read(A/'supplement_status.json')['stage']=='complete'
reports=[read(A/'verification.json'),read(A/'supplement/verification.json'),read(A/'overflow/verification.json')]
assert all(not x['mismatches'] for x in reports)
assert read(A/'source_snapshot_check.json')['all_unchanged']
assert read(A/'credentials_revoked.json')['revoked']
rows=[]
for p in [A/'source_inventory/manifest.jsonl',A/'supplement/manifest.jsonl',A/'overflow/manifest.jsonl']:
 rows += [json.loads(line) for line in p.read_text().splitlines()]
assert len({x['path'] for x in rows})==len(rows)==268170
assert sum(x['size'] for x in rows)==94217361751
out=dict(status='complete',files=len(rows),bytes=sum(x['size'] for x in rows),missing=0,sha256_mismatches=0,source_unchanged=True,source_deleted=False,credentials_revoked=True,destination=str(R),completed=time.time(),reports=[{'files':x['files'],'bytes':x['bytes'],'manifest_sha256':x['manifest_sha256']} for x in reports])
(A/'migration_complete.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print(json.dumps(out))
