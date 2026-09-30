from pathlib import Path
import os,json,re,hashlib,concurrent.futures,time
R=Path('/root');A=R/'autodl-tmp';O=Path('/root/prprompt_migration_overflow_20260929');O.mkdir(exist_ok=True)
roots=[R/'gp_storage_overflow_20260920']
files=[];dirs=[];links=[];refs={};size=0
for base in roots:
 if base.is_file():files.append(base);continue
 for parent,ds,fs in os.walk(base,followlinks=False):
  dirs.append(str(Path(parent).relative_to(R)))
  for name in ds+fs:
   p=Path(parent)/name
   if p.is_symlink():links.append(dict(path=str(p.relative_to(R)),target=os.readlink(p),resolved=str(p.resolve())))
  for name in fs:
   p=Path(parent)/name
   if p.is_file() and not p.is_symlink():files.append(p)
for p in files:
 size+=p.stat().st_size
 if p.suffix in ['.py','.json','.md','.sh'] and p.stat().st_size<400000:
  try:
   for ref in re.findall(r'/root/(?:autodl-tmp/)?[A-Za-z0-9_\-]+',p.read_text(errors='ignore')):refs[ref]=refs.get(ref,0)+1
  except OSError:pass
scope=dict(roots=[str(x) for x in roots],file_count=len(files),bytes=size,links=links,referenced_roots=refs)
(O/'scope.json').write_text(json.dumps(scope,ensure_ascii=False,indent=2));print(json.dumps(scope),flush=True)
def hashfile(p):
 stat=p.stat();h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 stat2=p.stat();assert (stat.st_size,stat.st_mtime_ns)==(stat2.st_size,stat2.st_mtime_ns),str(p)
 return dict(path=str(p.relative_to(R)),size=stat.st_size,mtime_ns=stat.st_mtime_ns,sha256=h.hexdigest())
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
 with (O/'manifest.jsonl').open('w') as out:
  for i,x in enumerate(pool.map(hashfile,files)):
   out.write(json.dumps(x,ensure_ascii=False)+'\n')
   if (i+1)%10000==0:print('hashed',i+1,flush=True)
(O/'files.list').write_bytes(b''.join(str(p.relative_to(R)).encode()+b'\0' for p in files)+b''.join(x['path'].encode()+b'\0' for x in links))
(O/'directories.json').write_text(json.dumps(dirs))
(O/'inventory_complete.json').write_text(json.dumps(dict(files=len(files),bytes=size,finished=time.time(),manifest_sha256=hashlib.sha256((O/'manifest.jsonl').read_bytes()).hexdigest())))
print('COMPLETE',flush=True)
