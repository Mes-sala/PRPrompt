import pathlib,json,hashlib,os,concurrent.futures,shutil,time
D=pathlib.Path('/root/cleanup_five_20260924');m=json.loads((D/'inventory.json').read_text());v=json.loads((D/'archive_verified.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
assert not (D/'deletion_receipt.json').exists()
assert v['verified'] and sha(D/'inventory.json')==v['inventory_sha256']
rows=[r for r in m['files'] if r['action']=='delete'];keeps=[r for r in m['files'] if r['action']=='keep']
assert len(rows)==v['files'] and sum(r['bytes'] for r in rows)==v['bytes']
allowed=['/root/autodl-tmp/accuracy_prompt_20260923/','/root/autodl-tmp/prompt_study_20260916/','/root/gp_storage_overflow_20260920/']
def check(r):
 p=r['path'];return not os.path.islink(p) and os.path.getsize(p)==r['bytes'] and sha(p)==r['sha256']
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:assert all(pool.map(check,m['files'])),'server changed since inventory'
for r in rows:
 assert any(r['path'].startswith(a) for a in allowed)
 assert pathlib.Path(r['path']).suffix not in ['.pth','.ckpt','.safetensors','.bin']
 assert not any(pathlib.Path(r['path']).parents[i].is_symlink() for i in range(len(pathlib.Path(r['path']).parents)))
for r in m['links']:
 if r['action']=='delete':assert os.path.islink(r['path']) and os.readlink(r['path'])==r['target']
before={p:shutil.disk_usage(p)._asdict() for p in ['/root','/root/autodl-tmp']}
print('second full SHA check passed; deleting archived file allowlist',flush=True)
for r in rows:os.unlink(r['path'])
for r in m['links']:
 if r['action']=='delete':os.unlink(r['path'])
for root in [a.rstrip('/') for a in allowed]:
 for p,ds,fs in os.walk(root,topdown=False):
  if p!=root:
   try:os.rmdir(p)
   except OSError:pass
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:assert all(pool.map(check,keeps)),'protected file changed'
assert all(not os.path.lexists(r['path']) for r in rows)
receipt=dict(completed=True,deleted_files=len(rows),deleted_logical_bytes=sum(r['bytes'] for r in rows),protected_files_verified=len(keeps),protected_bytes=sum(r['bytes'] for r in keeps),before=before,after={p:shutil.disk_usage(p)._asdict() for p in before},inventory_sha256=v['inventory_sha256'],time=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
(D/'deletion_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
