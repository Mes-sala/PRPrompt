import os,json,hashlib,re,concurrent.futures,pathlib
OUT=pathlib.Path('/root/cleanup_five_20260924');A='/root/autodl-tmp/accuracy_prompt_20260923';L='/root/autodl-tmp/prompt_study_20260916';O='/root/gp_storage_overflow_20260920'
keep=set(json.loads((OUT/'keep_sources.json').read_text()))
roots=[A,L,O,'/root/ProG','/root/autodl-tmp/pubmed_5090_transfer']
files=[];links=[];protected_parents=set()
for root in roots:
 for p,ds,fs in os.walk(root,followlinks=False):
  for n in list(ds)+fs:
   q=os.path.join(p,n)
   if os.path.islink(q):links.append({'path':q,'target':os.readlink(q),'action':'keep'})
  for n in fs:
   q=os.path.join(p,n)
   if os.path.islink(q):continue
   files.append((root,q))
   if n.endswith(('.pth','.ckpt','.safetensors','.bin')) or (root!=A and n.endswith('.pt')):protected_parents.add(p)
rounds={4,5,7,10,11,14,21,30,32,34,39,41,47,53,63,64,68,70,79,83}
def classify(root,q):
 rel=os.path.relpath(q,root);parts=rel.split('/');top=parts[0]
 if '/archives/lagrangebench/' in q or q.endswith(('.h5','.hdf5')):return 'keep','raw_dataset_archive'
 if any(re.search(r'^weights$|^models$|^model$|^checkpoints?$|^ckpts$',x,re.I) for x in parts[:-1]):return 'keep','additional_model_formats_and_sidecars'
 if root not in (A,L,O):return 'keep','core_runtime_data_checkpoint'
 if any(re.search(r'runtime|vendor|^env$|^venv$|site-packages',x,re.I) for x in parts[:-1]) or q.endswith('.whl'):return 'keep','shared_environment_library'
 if os.path.dirname(q) in protected_parents:return 'keep','weight_or_tensor_parent_and_sidecars'
 if root==A:
  if rel in keep or top=='assets5090':return 'keep','mainline_dependency'
  if len(parts)==1 and pathlib.Path(q).suffix in ('.json','.md','.txt'):return 'keep','historical_audit_metadata'
  m=re.match(r'(?:round|confirm)(\d+)',top)
  if len(parts)>1 and m and int(m[1]) in rounds:return 'keep','mainline_history_and_matched_controls'
 else:
  if any(re.search(r'asset|feature|pretrain|provenance|checkpoint|^ckpts$|^data$|^dataset|^raw$|^processed$',x,re.I) for x in parts[:-1]):return 'keep','shared_data_or_pretraining_asset'
 return 'delete','obsolete_scheme_local_archive_required'
def row(t):
 root,q=t;action,reason=classify(root,q);h=hashlib.sha256()
 with open(q,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 s=os.stat(q)
 return dict(path=q,bytes=s.st_size,sha256=h.hexdigest(),action=action,reason=reason)
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:rows=list(pool.map(row,files))
# Remove legacy experiment symlinks only when archived as links. Their shared targets remain protected.
for e in links:
 if e['path'].startswith(L+'/'):e['action']='delete'
m=dict(roots=roots,files=rows,links=links,policy='explicit file allowlist; preserve all model/tensor parents and shared assets; five mainline historical controls retained')
(OUT/'inventory.json').write_text(json.dumps(m,ensure_ascii=False))
print(json.dumps({a:{'files':sum(r['action']==a for r in rows),'bytes':sum(r['bytes'] for r in rows if r['action']==a)} for a in ['keep','delete']}),flush=True)
