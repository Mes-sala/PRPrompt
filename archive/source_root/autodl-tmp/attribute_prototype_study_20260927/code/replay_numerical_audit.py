"""Paired numerical sensitivity audit. Never substitutes reruns for sealed results."""
from pathlib import Path
import argparse,json,sqlite3,zlib,sys,time,fcntl
import numpy as np
import torch
import run_study as r
import expanded_data as data
import expanded_grid_cv as cv
import inspect,hashlib
# Preserve the original operator, changing only one-hot dtype for double arithmetic.
_src=inspect.getsource(r.predict).replace("def predict(","def predict64(").replace("F.one_hot(sy,C).float()","F.one_hot(sy,C).to(dtype=z.dtype)")
_ns=dict(r.__dict__);exec(_src,_ns);predict64=_ns["predict64"]

def run(plan):
    assert r.sha(__file__)==plan['replay_sha256'];parentfile=Path(plan['parent_plan']);assert r.sha(parentfile)==plan['parent_plan_sha256'];parent=json.loads(parentfile.read_text())
    assert r.sha(cv.__file__)==parent['runner_sha256'] and r.sha(data.__file__)==parent['loader_sha256'] and r.sha(r.__file__)==parent['operator_sha256']
    for f,h in parent.get('source_hashes',{}).items():assert r.sha(f)==h
    bankfile=Path(parent['bank_path']);assert r.sha(bankfile)==parent['bank_sha256'];bank=json.loads(bankfile.read_text());unit=parent['units'][0];assert len(parent['units'])==1
    source=r.ROOT/'results'/parent['round'];marker=json.loads((source/(unit['id']+'.complete.json')).read_text());db=source/(unit['id']+'.sqlite');assert r.sha(db)==marker['sha256'] and marker['plan_sha256']==r.sha(parentfile)
    con=sqlite3.connect(f'file:{db}?mode=ro',uri=True);configs={x['id']:x for x in bank[unit['kind']]};out=r.ROOT/'results'/plan['round']/unit['id'];out.mkdir(parents=True,exist_ok=True);lock=open(out/'run.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    r.save(out/'process.json',dict(pid=__import__('os').getpid(),started=time.time()))
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    d,x,adj,pool,query,C,protocol=data.load_data(unit['dataset']);model=None
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
    else:base=x
    # Seal one representation per unit, shared across baseline/optimized audit plans.
    cache=r.ROOT/'results/numerical_representation_cache';cache.mkdir(exist_ok=True)
    cachefile=cache/(unit['id']+'.pt');metafile=cache/(unit['id']+'.json')
    contract=dict(unit=unit,input_sha256=r.thash(data.encoder_input(d,unit)) if model is not None else r.thash(x),edge_sha256=r.thash(d.edge_index),operator_sha256=r.sha(r.__file__),loader_sha256=r.sha(data.__file__))
    if metafile.exists():
        meta=json.loads(metafile.read_text());assert meta['contract']==contract and r.sha(cachefile)==meta['file_sha256'];base=torch.load(cachefile,map_location='cuda',weights_only=True)
    else:
        torch.save(base.cpu(),cachefile);r.save(metafile,dict(contract=contract,file_sha256=r.sha(cachefile),tensor_sha256=r.thash(base)))
    basehash=r.thash(base);adj64=adj.double()
    assert len({(j['seed'],j['config']) for j in plan['jobs']})==len(plan['jobs'])
    groups={}
    for job in plan['jobs']:
        assert job['seed'] in parent['seeds'] and job['config'] in parent['config_ids'];c=configs[job['config']];groups.setdefault((c.get('hops',0),c['geometry']),[]).append(job)
    for (hops,geometry),jobs in groups.items():
        raw=base
        if unit['kind']=='attributes':
            for _ in range(hops):raw=torch.sparse.mm(adj,raw)
        feat=cv.geometry(raw,geometry)
        raw64=base.double()
        if unit["kind"]=="attributes":
            for _ in range(hops):raw64=torch.sparse.mm(adj64,raw64)
        feat64=cv.geometry(raw64,geometry)
        for job in jobs:
            seed=job['seed'];cfg=configs[job['config']];folder=out/(str(seed)+'__'+cfg['id']);folder.mkdir(exist_ok=True);result=folder/'result.json'
            if result.exists():
                old=json.loads(result.read_text());assert old['plan_sha256']==r.sha(plan['_path']) and r.sha(folder/'prediction.npz')==old['prediction_sha256'];continue
            ss,qq,sh=con.execute('SELECT support,query,split_hash FROM splits WHERE seed=?',(seed,)).fetchone();support=torch.tensor(json.loads(ss));sq=torch.tensor(json.loads(qq));assert torch.equal(query,sq) and not torch.isin(support,query).any();assert r.thash(torch.cat([support,query]))==sh
            old=con.execute('SELECT prediction,prediction_hash FROM results WHERE seed=? AND config=?',(seed,cfg['id'])).fetchone();sealed=zlib.decompress(old[0]);assert __import__('hashlib').sha256(sealed).hexdigest()==old[1];expected=np.frombuffer(sealed,dtype=np.uint8)
            with torch.no_grad():
                scores=cv.cfg_predict(feat,adj,support.cuda(),d.y[support].cuda(),C,cfg,seed);prob=scores[query.cuda()].softmax(-1).cpu().numpy()
                cfg64=dict(cfg,center=0,op='prototype');pp=[]
                for repeat in range(2):pp.append(predict64(feat64,adj64,support.cuda(),d.y[support].cuda(),C,cfg64,seed)[query.cuda()].softmax(-1).cpu().numpy())
                prob64,repeat64=pp
            assert np.isfinite(prob).all() and np.allclose(prob.sum(1),1,atol=1e-6);assert np.isfinite(prob64).all() and np.isfinite(repeat64).all()
            pf=folder/'prediction.npz';np.savez_compressed(pf,probability=prob,probability64=prob64,repeat64=repeat64,sealed_prediction=expected,support=support.numpy(),query=query.numpy())
            r.save(result,dict(dataset=unit['dataset'],unit=unit['id'],seed=seed,config=cfg,routes=job.get('routes',[]),representation=unit['kind'],checkpoint_sha256=unit.get('sha256'),protocol=protocol,split_sha256=sh,prediction_sha256=r.sha(pf),sealed_argmax_sha256=old[1],plan_sha256=r.sha(plan['_path']),parent_database_sha256=marker['sha256'],replay_argmax_exact=bool(np.array_equal(prob.argmax(1),expected)),changed_from_sealed=int((prob.argmax(1)!=expected).sum()),double_changed_from_sealed=int((prob64.argmax(1)!=expected).sum()),double_repeat_changed=int((prob64.argmax(1)!=repeat64.argmax(1)).sum()),double_repeat_max_abs=float(np.max(np.abs(prob64-repeat64))),base_tensor_sha256=basehash,representation_file_sha256=r.sha(cachefile),scope='Frozen selected configurations; numerical sensitivity only, no reselection',double_metrics=r.measures(d.y[query].numpy(),prob64),**r.measures(d.y[query].numpy(),prob)))
        del raw,feat,raw64,feat64
    if model is not None:assert before=={k:r.thash(v) for k,v in model.state_dict().items()} and not model.training and all(v.grad is None for v in model.parameters())
    con.close();r.save(out/'complete.json',dict(rows=len(plan['jobs']),plan_sha256=r.sha(plan['_path']),finished=time.time()));print(unit['id'],len(plan['jobs']),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=json.loads(Path(a.plan).read_text());p['_path']=a.plan;run(p)
