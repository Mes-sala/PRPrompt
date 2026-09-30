"""Frozen full grid producer. No query labels are read or metrics computed here."""
from downstream41_common import *
import argparse,os,time,shutil,traceback


def atomic_npz(path,**arrays):
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
    tmp.replace(path)


def save_tensor(path,obj):
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:torch.save(obj,f)
    tmp.replace(path)


def state_cpu(m):return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()} if m is not None else None


def native_scores(blocks,ep,c,head):
    h=blocks[3];s=ep['support'];q=ep['query'];y=ep['support_labels']
    with torch.no_grad():
        if c is None:return head.cpu().double()(h[q]).numpy()
        centers=torch.stack([h[s][torch.tensor(y==i)].mean(0) for i in range(head.out_features)])
        return F.cosine_similarity(h[q,None,:],centers[None,:,:],dim=-1).numpy()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    freeze=ROOT/('downstream_smoke_freeze.json' if a.smoke else 'downstream_freeze.json')
    cfg=json.loads(freeze.read_text())
    for n,h in cfg['source_sha256'].items():assert digest(ROOT/n)==h,n
    for n,h in cfg['inputs'].items():assert digest(BASE/n)==h,n
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    out=ROOT/('downstream_smoke' if a.smoke else 'downstream');out.mkdir(exist_ok=True);start=time.time();done=0
    for cell in cfg['cells']:
        ds,enc,kind=cell['dataset'],cell['encoder'],cell['kind'];path=BASE/cell['checkpoint']
        assert digest(path)==cell['checkpoint_sha256']
        graphs,dim=graphs41(ds);eps=[episode(episodes(ds)[i]) for i in cfg['episode_indices']]
        statics=[static41(ds,ep) for ep in eps]
        model=model41(dim,path,kind);original=fingerprint(model)
        baseline_dir=out/ds/enc/'no_prompt';baseline_dir.mkdir(parents=True,exist_ok=True)
        if not (baseline_dir/'record.json').exists():
            blocks=pooled41(model,graphs);ks=kernels41(blocks);baseline=[];errs=[]
            for ep,st in zip(eps,statics):
                score,err=readout41(ks,st,ep);baseline.append(np.stack([score[k] for k in ['four','HM','WA','H','M']]));errs.append(err)
            atomic_npz(baseline_dir/'sealed.npz',scores=np.stack(baseline),predictions=np.round(np.stack(baseline),10).argmax(-1).astype(np.int16),supports=np.stack([e['support'] for e in eps]),query=eps[0]['query'])
            write(baseline_dir/'record.json',{'status':'sealed_unscored','methods':['four','HM','WA','H','M'],'freeze_sha256':digest(freeze),'checkpoint_sha256':digest(path),'sealed_sha256':digest(baseline_dir/'sealed.npz'),'max_algebra_error':max(errs),'query_labels_read':False})
            del blocks,ks
        else:
            rec=json.loads((baseline_dir/'record.json').read_text());assert rec['freeze_sha256']==digest(freeze) and digest(baseline_dir/'sealed.npz')==rec['sealed_sha256']
        for ei,(ep,st) in enumerate(zip(eps,statics)):
            for method in cfg['prompt_methods']:
                for lr in cfg['learning_rates']:
                    job=out/ds/enc/(f'episode{cfg["episode_indices"][ei]:02d}__{method}__lr{lr:g}');job.mkdir(parents=True,exist_ok=True)
                    if (job/'record.json').exists():
                        rec=json.loads((job/'record.json').read_text());assert rec['freeze_sha256']==digest(freeze)
                        assert digest(job/'sealed.npz')==rec['sealed_sha256'] and digest(job/'prompt_states.pt')==rec['state_sha256'];done+=1;continue
                    assert shutil.disk_usage(ROOT).free>3*1024**3,'Storage lower guard; preserve completed work before resuming'
                    begin=time.time();model=model41(dim,path,kind).cuda()
                    assert fingerprint(model)==original
                    nc=int(ep['support_labels'].max())+1;f,c,head=prompts41(dim,nc,method)
                    for module in [f,c,head]:
                        if module is not None:module.cuda()
                    params=(list(f.parameters()) if f is not None else [])+(list(c.parameters()) if c is not None else [])+(list(head.parameters()) if c is None else [])
                    opt=torch.optim.Adam(params,lr=lr,weight_decay=.0005)
                    assert not {id(p) for p in params}&{id(p) for p in model.parameters()}
                    b=Batch.from_data_list([graphs[int(i)] for i in ep['support']]).to('cuda');assert b.y is None
                    y=torch.tensor(ep['support_labels'],dtype=torch.long,device='cuda');states={0:{'feature':state_cpu(f),'channel':state_cpu(c),'head':state_cpu(head)}};losses=[]
                    for step in range(1,max(cfg['horizons'])+1):
                        losses.append(train_step41(model,b,y,f,c,head,opt))
                        if step in cfg['horizons']:states[step]={'feature':state_cpu(f),'channel':state_cpu(c),'head':state_cpu(head)}
                    assert fingerprint(model)==original
                    outputs=[];errors=[];methods=['four','HM','WA','H','M','native']
                    for step in [0,*cfg['horizons']]:
                        ff,cc,hh=prompts41(dim,nc,method)
                        for key,module in [('feature',ff),('channel',cc),('head',hh)]:
                            if module is not None:module.load_state_dict(states[step][key],strict=True)
                        blocks=pooled41(model,graphs,ff,cc);score,err=readout41(kernels41(blocks),st,ep)
                        score['native']=native_scores(blocks,ep,cc,hh);outputs.append(np.stack([score[k] for k in methods]));errors.append(err)
                    scores=np.stack(outputs);pred=np.round(scores,10).argmax(-1).astype(np.int16)
                    pred[:,-1]=scores[:,-1].argmax(-1).astype(np.int16)
                    atomic_npz(job/'sealed.npz',scores=scores,predictions=pred,support=ep['support'],support_labels=ep['support_labels'],query=ep['query'],steps=np.array([0,*cfg['horizons']]))
                    save_tensor(job/'prompt_states.pt',states)
                    write(job/'record.json',{'status':'sealed_unscored','dataset':ds,'encoder':enc,'kind':kind,'method':method,'lr':lr,'support_seed':episodes(ds)[cfg['episode_indices'][ei]]['seed'],'steps':[0,*cfg['horizons']],'methods':methods,'support_losses':losses,'frozen_encoder_state':original,'checkpoint_sha256':digest(path),'freeze_sha256':digest(freeze),'sealed_sha256':digest(job/'sealed.npz'),'state_sha256':digest(job/'prompt_states.pt'),'max_algebra_error':max(errors),'seconds':time.time()-begin,'query_labels_read':False,'native_score_precision':'FP64 evaluation of FP32 trained parameters; ordinary argmax; kernel readouts round(10)'})
                    done+=1
                    write(out/'progress.json',{'status':'running','pid':os.getpid(),'dataset':ds,'encoder':enc,'method':method,'episode':cfg['episode_indices'][ei],'lr':lr,'completed_jobs':done,'total_jobs':cfg['total_jobs'],'elapsed':time.time()-start})
                    print('DOWNSTREAM41_SEALED',done,cfg['total_jobs'],ds,enc,method,ei,lr,flush=True)
                    del opt,b,y,f,c,head,states,blocks;torch.cuda.empty_cache()
    assert done==cfg['total_jobs']
    write(out/'complete.json',{'status':'sealed_unscored','count':done,'cells':len(cfg['cells']),'freeze_sha256':digest(freeze),'query_labels_read':False,'seconds':time.time()-start})


if __name__=='__main__':
    try:main()
    except Exception:
        write(ROOT/('downstream_smoke_failure.json' if '--smoke' in sys.argv else 'downstream_failure.json'),{'status':'failed','time':time.time(),'traceback':traceback.format_exc()});raise
