import json,subprocess,concurrent.futures,time,hashlib,os,fcntl
from pathlib import Path
R=Path('/data1/shq/GP/PRPrompt_full_archive_20260929');A=R/'audit';D=R/'source_root'
lock=open(A/'migration.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
SSH=f'ssh -i {A}/transfer_key -o IdentitiesOnly=yes -o UserKnownHostsFile={A}/known_hosts -o ServerAliveInterval=30 -o ServerAliveCountMax=4 -p 51542'
REMOTE='root@connect.westb.seetacloud.com:'
def status(stage,**kwargs):
 tmp=A/'status.tmp';tmp.write_text(json.dumps(dict(stage=stage,time=time.time(),pid=os.getpid(),**kwargs),indent=2));tmp.replace(A/'status.json')
def fetchmeta():return subprocess.call(['rsync','-a','--timeout=120','-e',SSH,REMOTE+'prprompt_migration_20260929/',str(A/'source_inventory')+'/'],stdout=subprocess.DEVNULL,stderr=open(A/'metadata.log','a'))
status('waiting_inventory_index')
while not (A/'transfer_index.json').exists():time.sleep(10)
index=json.loads((A/'transfer_index.json').read_text());buckets=[[] for _ in range(8)];sizes=[0]*8
for row in sorted(index,key=lambda x:x['size'],reverse=True):
 i=min(range(8),key=lambda i:sizes[i]);buckets[i].append(row['path']);sizes[i]+=row['size']
for i,paths in enumerate(buckets):(A/f'files_{i}.list').write_bytes(b''.join(p.encode()+b'\0' for p in sorted(paths)))
def transfer(i):
 for compressed in [True,False]:
  paths=[p for p in buckets[i] if p.lower().endswith(('.npz','.gz','.zip','.png','.jpg','.jpeg','.xz','.bz2'))==compressed]
  fl=A/f'files_{i}_{compressed}.list';fl.write_bytes(b''.join(p.encode()+b'\0' for p in sorted(paths)))
  cmd=['rsync','-a','--no-owner','--no-group','--timeout=180','--partial-dir=.rsync-partial','--stats','--from0','--files-from='+str(fl),'-e',SSH,REMOTE+'./',str(D)+'/']
  if not compressed:cmd[1:1]=['-z','--compress-level=1']
  for attempt in range(1,4):
   with open(A/f'shard_{i}.log','a') as log:ret=subprocess.call(cmd,stdout=log,stderr=subprocess.STDOUT)
   (A/f'shard_{i}_status.json').write_text(json.dumps(dict(shard=i,compressed_input=compressed,returncode=ret,attempt=attempt,files=len(paths),time=time.time())))
   if ret==0:break
   time.sleep(10)
  if ret:raise RuntimeError(f'shard {i} failed {ret}')
 return i
status('transferring',files=len(index),bytes=sum(sizes),shards=8)
try:
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  pending={pool.submit(transfer,i):i for i in range(8)}
  while pending:
   done,_=concurrent.futures.wait(pending,timeout=20,return_when=concurrent.futures.FIRST_COMPLETED)
   for f in done:f.result();pending.pop(f)
   status('transferring',remaining_shards=list(pending.values()),files=len(index),bytes=sum(sizes))
 status('waiting_source_hashes')
 while True:
  fetchmeta()
  if (A/'source_inventory/inventory_complete.json').exists():break
  time.sleep(20)
 M=A/'source_inventory/manifest.jsonl';complete=json.loads((A/'source_inventory/inventory_complete.json').read_text());assert hashlib.sha256(M.read_bytes()).hexdigest()==complete['manifest_sha256']
 rows=[json.loads(s) for s in M.read_text().splitlines()];assert len(rows)==len(index)
 for dr in json.loads((A/'source_inventory/directories.json').read_text()):(D/dr).mkdir(parents=True,exist_ok=True)
 status('verifying',files=len(rows),bytes=sum(x['size'] for x in rows))
 def verify(row):
  p=D/row['path']
  if not p.exists():return dict(path=row['path'],error='missing')
  if p.stat().st_size!=row['size']:return dict(path=row['path'],error='size')
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
  if h.hexdigest()!=row['sha256']:return dict(path=row['path'],error='sha256')
  return None
 bad=[];n=0
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  for v in pool.map(verify,rows):
   n+=1
   if v:bad.append(v)
   if n%10000==0:status('verifying',verified=n,total=len(rows),bad=len(bad))
 report=dict(files=len(rows),bytes=sum(x['size'] for x in rows),mismatches=bad,manifest_sha256=complete['manifest_sha256'],finished=time.time())
 (A/'verification.json').write_text(json.dumps(report,indent=2));assert not bad,bad[:3]
 status('complete',files=len(rows),bytes=report['bytes'],verification='all_sha256_match')
except Exception as e:
 status('failed',error=repr(e));raise
