import json,hashlib,subprocess,time,os,concurrent.futures
from pathlib import Path
R=Path('/data1/shq/GP/PRPrompt_full_archive_20260929');A=R/'audit';D=R/'source_root'
SSH=f'ssh -i {A}/transfer_key -o IdentitiesOnly=yes -o UserKnownHostsFile={A}/known_hosts -o ServerAliveInterval=30 -p 51542'
def write(x):(A/'supplement_status.json').write_text(json.dumps(dict(time=time.time(),pid=os.getpid(),**x),indent=2))
reports=[]
for name in ['supplement','overflow']:
 out=A/name;out.mkdir(exist_ok=True);write(dict(stage='waiting_inventory',part=name))
 while True:
  with open(out/'metadata.log','a') as log:ret=subprocess.call(['rsync','-az','--timeout=120','-e',SSH,f'root@connect.westb.seetacloud.com:prprompt_migration_{name}_20260929/',str(out)+'/'],stdout=log,stderr=subprocess.STDOUT)
  if ret==0 and (out/'inventory_complete.json').exists():break
  time.sleep(20)
 manifest=out/'manifest.jsonl';meta=json.loads((out/'inventory_complete.json').read_text());assert hashlib.sha256(manifest.read_bytes()).hexdigest()==meta['manifest_sha256']
 rows=[json.loads(x) for x in manifest.read_text().splitlines()]
 if (out/'verification.json').exists():
  prior=json.loads((out/'verification.json').read_text())
  if not prior['mismatches'] and prior['manifest_sha256']==meta['manifest_sha256']:
   reports.append(prior);continue
 write(dict(stage='transferring',part=name,files=len(rows),bytes=sum(x['size'] for x in rows)))
 n=4 if name=='overflow' else 1
 buckets=[[] for _ in range(n)];sizes=[0]*n
 for x in sorted(rows,key=lambda x:x['size'],reverse=True):
  k=min(range(n),key=lambda k:sizes[k]);buckets[k].append(x['path']);sizes[k]+=x['size']
 def transfer(k):
  f=out/f'part_{k}.list';f.write_bytes(b''.join(p.encode()+b'\0' for p in sorted(buckets[k])))
  cmd=['rsync','-a','--no-owner','--no-group','-z','--compress-level=1','--partial-dir=.rsync-partial','--timeout=180','--stats','--from0','--files-from='+str(f),'-e',SSH,'root@connect.westb.seetacloud.com:./',str(D)+'/']
  for attempt in range(3):
   with open(out/f'transfer_{k}.log','a') as log:ret=subprocess.call(cmd,stdout=log,stderr=subprocess.STDOUT)
   if not ret:return
   time.sleep(10)
  raise RuntimeError(f'{name} shard {k} failed {ret}')
 with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:list(pool.map(transfer,range(n)))
 for dr in json.loads((out/'directories.json').read_text()):(D/dr).mkdir(parents=True,exist_ok=True)
 write(dict(stage='verifying',part=name));bad=[]
 for x in rows:
  p=D/x['path'];h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
  if h.hexdigest()!=x['sha256']:bad.append(x['path'])
 report=dict(part=name,files=len(rows),bytes=sum(x['size'] for x in rows),mismatches=bad,manifest_sha256=meta['manifest_sha256']);(out/'verification.json').write_text(json.dumps(report,indent=2));assert not bad
 reports.append(report)
write(dict(stage='complete',reports=reports))
