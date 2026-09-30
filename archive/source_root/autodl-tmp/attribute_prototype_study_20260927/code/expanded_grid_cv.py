"""Expanded Accuracy screen; label predictions sealed in resumable per-unit SQLite."""
import argparse,json,sqlite3,zlib,time,os,sys,fcntl
from pathlib import Path
import numpy as np
import torch
import run_study as r
import expanded_data as data

def geometry(base,kind):
 if kind=='raw':return base
 centered=base-base.mean(0,keepdim=True)
 if kind=='center':return centered
 var=centered.square().mean(0);scale=var.mean().clamp_min(1e-12)
 if kind=='diag':return centered/(var+.1*scale).sqrt()
 rho={'zca001':.01,'zca01':.1,'zca1':1.}[kind];cov=centered.T@centered/len(centered);ev,u=torch.linalg.eigh(cov)
 return centered@((u*(ev.clamp_min(0)+rho*scale).rsqrt()[None,:])@u.T)

def support_split(d,pool,C,seed,shots):
 gen=torch.Generator().manual_seed(seed)
 support=torch.cat([ix[torch.randperm(len(ix),generator=gen)[:shots]] for c in range(C) for ix in [pool[d.y[pool]==c]]]);sy=d.y[support]
 vi=torch.tensor([(sy==c).nonzero().flatten()[0] for c in range(C)],dtype=torch.long);mask=torch.ones(len(support),dtype=torch.bool);mask[vi]=False
 return support,sy,vi,mask

def cfg_predict(features,adj,si,sy,C,cfg,seed):
 return r.predict(features,adj,si,sy,C,dict(cfg,center=0,op='prototype'),seed)

def run(plan):
 assert r.sha(__file__)==plan['runner_sha256'];assert r.sha(data.__file__)==plan['loader_sha256'];assert r.sha(r.__file__)==plan['operator_sha256']
 assert len({u['source_root'] for u in plan['units'] if u['kind']=='frozen'})<=1, 'Separate processes required for source snapshots'
 for source_file,digest in plan.get('source_hashes',{}).items():assert r.sha(source_file)==digest
 bankfile=Path(plan['bank_path']);assert r.sha(bankfile)==plan['bank_sha256'];bank=json.loads(bankfile.read_text())
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 out=r.ROOT/'results'/plan['round'];out.mkdir(parents=True,exist_ok=True);lock=open(out/'run.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);r.save(out/'process.json',dict(pid=os.getpid(),started=time.time()));total=0
 for unit in plan['units']:
  done=out/(unit['id']+'.complete.json');path=out/(unit['id']+'.sqlite')
  if done.exists():
   m=json.loads(done.read_text());assert m['plan_sha256']==r.sha(plan['_path']) and m['sha256']==r.sha(path);total+=m['rows'];continue
  torch.cuda.reset_peak_memory_stats()
  d,x,adj,pool,query,C,protocol=data.load_data(unit['dataset']);assert C<256
  if unit['kind']=='frozen':
   assert r.sha(unit['path'])==unit['sha256'];sys.path.insert(0,unit['source_root'])
   if unit.get('legacy',False):
    from prompt_graph.model import GCN
    model=GCN(d.x.shape[1],128,128,num_layer=2).cuda().eval()
   else:
    from prompt_graph.model import build_gnn
    model=build_gnn(unit['architecture'],d.x.shape[1],128,2).cuda().eval()
   model.load_state_dict(torch.load(unit['path'],weights_only=True,map_location='cpu'),strict=True);model.requires_grad_(False);before={k:r.thash(v) for k,v in model.state_dict().items()}
   with torch.no_grad():base=model(data.encoder_input(d,unit),d.edge_index.cuda()).detach()
   configs=bank['frozen']
  else:base=x;configs=bank['attributes']
  if plan.get('config_ids'):configs=[c for c in configs if c['id'] in plan['config_ids']]
  con=sqlite3.connect(path);con.execute('PRAGMA journal_mode=WAL');con.executescript('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY,value TEXT); CREATE TABLE IF NOT EXISTS splits(seed INTEGER PRIMARY KEY,support TEXT,query TEXT,split_hash TEXT); CREATE TABLE IF NOT EXISTS results(seed INTEGER,config TEXT,prediction BLOB,prediction_hash TEXT,validation TEXT,accuracy REAL,seconds REAL,PRIMARY KEY(seed,config));')
  existing=con.execute("SELECT value FROM metadata WHERE key='plan_sha256'").fetchone();assert existing is None or existing[0]==r.sha(plan['_path'])
  for k,v in dict(plan_sha256=r.sha(plan['_path']),unit=json.dumps(unit),protocol=protocol,classes=str(C)).items():con.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',(k,v))
  con.commit();count=con.execute('SELECT COUNT(*) FROM results WHERE accuracy IS NOT NULL').fetchone()[0]
  keys=list(dict.fromkeys((c.get('hops',0),c['geometry']) for c in configs))
  for hops,kind in keys:
   raw=base
   if unit['kind']=='attributes':
    for _ in range(hops):raw=torch.sparse.mm(adj,raw)
   feat=geometry(raw,kind);assert torch.isfinite(feat).all()
   for seed in plan['seeds']:
    support,sy,vi,mask=support_split(d,pool,C,seed,plan['shots']);assert not torch.isin(support,query).any();si=support.cuda();sy=sy.cuda();vi=vi.cuda();mask=mask.cuda();split=r.thash(torch.cat([support,query]))
    con.execute('INSERT OR IGNORE INTO splits VALUES (?,?,?,?)',(seed,json.dumps(support.tolist()),json.dumps(query.tolist()),split));con.commit()
    for cfg in [c for c in configs if (c.get('hops',0),c['geometry'])==(hops,kind)]:
     prior=con.execute('SELECT accuracy FROM results WHERE seed=? AND config=?',(seed,cfg['id'])).fetchone()
     if prior and prior[0] is not None:continue
     assert __import__('shutil').disk_usage(r.ROOT).free>8*1024**3
     r.save(out/'current.json',dict(unit=unit['id'],seed=seed,config=cfg['id'],completed_unit=count,time=time.time()))
     with torch.no_grad():
      torch.cuda.synchronize();t=time.perf_counter();vv=[];yy=[]
      for fold in range(plan.get('validation_folds',1)):
       fv=torch.stack([(sy==c).nonzero().flatten()[fold] for c in range(C)]);fm=torch.ones(len(si),dtype=torch.bool,device=si.device);fm[fv]=False
       vv.append(cfg_predict(feat,adj,si[fm],sy[fm],C,cfg,seed)[si[fv]].softmax(-1));yy.append(sy[fv])
      v=torch.cat(vv);vy_t=torch.cat(yy);score=cfg_predict(feat,adj,si,sy,C,cfg,seed);assert torch.isfinite(score).all() and torch.isfinite(v).all();pred=score[query.cuda()].argmax(-1).cpu().numpy().astype(np.uint8);torch.cuda.synchronize();secs=time.perf_counter()-t
      vp=v.cpu().numpy();vy=vy_t.cpu().numpy();val=dict(accuracy=float((vp.argmax(1)==vy).mean()*100),nll=float(-np.log(np.clip(vp[np.arange(len(vy)),vy],1e-12,None)).mean()),probability=vp.tolist(),labels=vy.tolist())
     blob=zlib.compress(pred.tobytes());digest=__import__('hashlib').sha256(pred.tobytes()).hexdigest()
     # Commit predictions and support-only validation BEFORE consulting query labels.
     con.execute('INSERT OR REPLACE INTO results VALUES (?,?,?,?,?,NULL,?)',(seed,cfg['id'],blob,digest,json.dumps(val),secs));con.commit()
     acc=float((pred==d.y[query].numpy()).mean()*100);con.execute('UPDATE results SET accuracy=? WHERE seed=? AND config=?',(acc,seed,cfg['id']));con.commit();count+=1
   del feat,raw
  expected=len(configs)*len(plan['seeds']);assert con.execute('SELECT COUNT(*) FROM results WHERE accuracy IS NOT NULL').fetchone()[0]==expected
  if unit['kind']=='frozen':
   assert before=={k:r.thash(v) for k,v in model.state_dict().items()} and not model.training and all(p.grad is None for p in model.parameters());del model
  con.execute('PRAGMA wal_checkpoint(TRUNCATE)');con.close();r.save(done,dict(rows=expected,sha256=r.sha(path),plan_sha256=r.sha(plan['_path']),prediction_format='zlib uint8 query argmax, permits independent Accuracy only',peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),finished=time.time()));total+=expected;print(unit['id'],expected,flush=True);del d,x,adj,base;torch.cuda.empty_cache()
 r.save(out/'complete.json',dict(rows=total,units=len(plan['units']),finished=time.time()))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=json.loads(Path(a.plan).read_text());p['_path']=a.plan
 try:run(p)
 except Exception:
  import traceback
  r.save(r.ROOT/'results'/p['round']/'failure.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
