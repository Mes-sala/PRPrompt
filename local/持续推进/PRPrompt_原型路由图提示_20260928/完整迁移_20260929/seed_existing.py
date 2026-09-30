from pathlib import Path
import json,shutil,os,time
r=Path('/data1/shq/GP/PRPrompt_full_archive_20260929');a=r/'audit';dst=r/'source_root';old=Path('/data1/shq/GP/ModuleComposition_20260919_19307');prefix='autodl-tmp/prompt_study_20260916/ModuleComposition_20260919_19307/'
rows=json.loads((a/'transfer_index.json').read_text());copied=[];skipped=0
for row in rows:
 if not row['path'].startswith(prefix):continue
 p=old/row['path'][len(prefix):];q=dst/row['path']
 if q.exists() or not p.is_file() or p.stat().st_size!=row['size']:skipped+=1;continue
 q.parent.mkdir(parents=True,exist_ok=True);tmp=q.with_name(q.name+'.local_seed_tmp')
 shutil.copy2(p,tmp)
 try:os.link(tmp,q);copied.append(row)
 except FileExistsError:skipped+=1
 finally:tmp.unlink()
(a/'local_seed.json').write_text(json.dumps(dict(files=len(copied),bytes=sum(x['size'] for x in copied),skipped=skipped,finished=time.time(),note='Local reuse is provisional until source SHA256 verification; separate copies, no shared inode with existing project.'),indent=2));print(len(copied))
