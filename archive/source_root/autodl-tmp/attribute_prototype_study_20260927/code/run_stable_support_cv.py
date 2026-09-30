"""Support-only reselection audit on sealed representations using float64 arithmetic."""
from pathlib import Path
import argparse,json,sqlite3,sys,time,fcntl,inspect,os
import numpy as np
import torch
import run_study as r
import expanded_data as data
import expanded_grid_cv as cv
_src=inspect.getsource(r.predict).replace("def predict(","def predict64(").replace("F.one_hot(sy,C).float()","F.one_hot(sy,C).to(dtype=z.dtype)")
_ns=dict(r.__dict__);exec(_src,_ns);predict64=_ns['predict64']

def run(plan):
    assert r.sha(__file__)==plan['runner_sha256']
    parentfile=Path(plan['parent_plan']);assert r.sha(parentfile)==plan['parent_plan_sha256'];parent=json.loads(parentfile.read_text());unit=parent['units'][0]
    assert r.sha(cv.__file__)==parent['runner_sha256'] and r.sha(data.__file__)==parent['loader_sha256'] and r.sha(r.__file__)==parent['operator_sha256']
    for f,h in parent.get('source_hashes',{}).items():assert r.sha(f)==h
    bankfile=Path(parent['bank_path']);assert r.sha(bankfile)==parent['bank_sha256'];bank=json.loads(bankfile.read_text());configs={x['id']:x for x in bank[unit['kind']]};assert set(plan['config_ids'])<=set(configs)
    source=r.ROOT/'results'/parent['round'];marker=json.loads((source/(unit['id']+'.complete.json')).read_text());db=source/(unit['id']+'.sqlite');assert r.sha(db)==marker['sha256'] and marker['plan_sha256']==r.sha(parentfile)
    con=sqlite3.connect(f'file:{db}?mode=ro',uri=True);out=r.ROOT/'results'/plan['round']/unit['id'];out.mkdir(parents=True,exist_ok=True);lock=open(out/'run.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    r.save(out/'process.json',dict(pid=os.getpid(),started=time.time()))
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
    groups={}
    for cid in plan['config_ids']:
        c=configs[cid];groups.setdefault((c.get('hops',0),c['geometry']),[]).append(c)
    count=0
    for (hops,geometry),cc in groups.items():
        raw64=base.double()
        if unit['kind']=='attributes':
            for _ in range(hops):raw64=torch.sparse.mm(adj64,raw64)
        feat=cv.geometry(raw64,geometry);assert feat.dtype==torch.float64 and torch.isfinite(feat).all()
        for seed in plan['seeds']:
            ss,qq,sh=con.execute('SELECT support,query,split_hash FROM splits WHERE seed=?',(seed,)).fetchone();support=torch.tensor(json.loads(ss));assert torch.equal(query,torch.tensor(json.loads(qq))) and not torch.isin(support,query).any();assert r.thash(torch.cat([support,query]))==sh
            si=support.cuda();sy=d.y[support].cuda()
            for cfg in cc:
                folder=out/(str(seed)+'__'+cfg['id']);folder.mkdir(exist_ok=True);result=folder/'result.json'
                if result.exists():
                    old=json.loads(result.read_text());assert old['plan_sha256']==r.sha(plan['_path']) and r.sha(folder/'prediction.npz')==old['prediction_sha256'];count+=1;continue
                assert __import__('shutil').disk_usage(r.ROOT).free>8*1024**3
                r.save(out/'current.json',dict(seed=seed,config=cfg['id'],completed=count,time=time.time()))
                cfg64=dict(cfg,center=0,op='prototype');vv=[];yy=[]
                with torch.no_grad():
                    for fold in range(5):
                        vi=torch.stack([(sy==c).nonzero().flatten()[fold] for c in range(C)]);mask=torch.ones(len(si),dtype=torch.bool,device=si.device);mask[vi]=False
                        vv.append(predict64(feat,adj64,si[mask],sy[mask],C,cfg64,seed)[si[vi]].softmax(-1));yy.append(sy[vi])
                    vp=torch.cat(vv).cpu().numpy();vy=torch.cat(yy).cpu().numpy();prob=predict64(feat,adj64,si,sy,C,cfg64,seed)[query.cuda()].softmax(-1).cpu().numpy();repeat=predict64(feat,adj64,si,sy,C,cfg64,seed)[query.cuda()].softmax(-1).cpu().numpy()
                assert np.isfinite(prob).all() and np.isfinite(vp).all() and np.allclose(prob.sum(1),1,atol=1e-10)
                pf=folder/'prediction.npz';np.savez_compressed(pf,probability=prob,repeat_probability=repeat,support=support.numpy(),query=query.numpy(),validation_probability=vp,validation_labels=vy)
                val=dict(accuracy=float((vp.argmax(1)==vy).mean()*100),nll=float(-np.log(np.clip(vp[np.arange(len(vy)),vy],1e-12,None)).mean()))
                # Probability and support validation are sealed before query labels are used.
                r.save(result,dict(dataset=unit['dataset'],unit=unit['id'],seed=seed,config=cfg,validation=val,plan_sha256=r.sha(plan['_path']),split_sha256=sh,prediction_sha256=r.sha(pf),base_tensor_sha256=basehash,representation_file_sha256=r.sha(cachefile),repeat_changed=int((prob.argmax(1)!=repeat.argmax(1)).sum()),repeat_probability_max_abs=float(np.max(np.abs(prob-repeat))),numeric_protocol='float64 geometry and readout; sealed float32 encoder representation',**r.measures(d.y[query].numpy(),prob)));count+=1
        del raw64,feat
    assert count==len(plan['config_ids'])*len(plan['seeds'])
    if model is not None:assert before=={k:r.thash(v) for k,v in model.state_dict().items()} and not model.training and all(v.grad is None for v in model.parameters())
    con.close();r.save(out/'complete.json',dict(rows=count,plan_sha256=r.sha(plan['_path']),finished=time.time()));print(unit['id'],count,flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=json.loads(Path(a.plan).read_text());p['_path']=a.plan;run(p)
