from pathlib import Path
import json,time
r=Path('/root');groups=['prprompt_migration_20260929','prprompt_migration_supplement_20260929','prprompt_migration_overflow_20260929'];reports=[]
for name in groups:
 d=r/name
 assert (d/'inventory_complete.json').exists(),name
 rows=[json.loads(s) for s in (d/'manifest.jsonl').read_text().splitlines()];bad=[]
 for row in rows:
  p=r/row['path']
  if not p.exists():bad.append(dict(path=row['path'],error='missing'));continue
  st=p.stat()
  if (st.st_size,st.st_mtime_ns)!=(row['size'],row['mtime_ns']):bad.append(dict(path=row['path'],error='changed_since_hash'))
 reports.append(dict(group=name,files=len(rows),changed=bad))
out=dict(checked_at=time.time(),reports=reports,all_unchanged=all(not x['changed'] for x in reports));(r/'prprompt_source_snapshot_check_20260929.json').write_text(json.dumps(out,indent=2));print(json.dumps(out));assert out['all_unchanged']
