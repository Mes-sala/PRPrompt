"""Protocol-preserving runner derived from audited AnchorPrompt scaffold."""
from experiment import *

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--prog',default='/root/ProG')
    ap.add_argument('--output',required=True)
    ap.add_argument('--seed-count',type=int,default=4)
    ap.add_argument('--seed-start',type=int,default=16092401)
    args=ap.parse_args()
    root=Path(args.output)
    assert not root.exists(), 'Fresh output required'
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    prog=Path(args.prog)
    sys.path.insert(0,str(prog))
    from prompt_graph.model import GCN
    files=[Path(__file__).resolve().parents[1]/'BayesFeaturePrompt/experiment.py',Path(__file__),Path(__file__).with_name('experiment.py'),Path(__file__).resolve().parents[1]/'AnchorPrompt/run.py',Path(__file__).parent/'docs/01_protocol.md',prog/'prompt_graph/model/gnn.py',prog/'prompt_graph/model/GCN.py',prog/'prompt_graph/utils/act.py']
    history_path=Path(__file__).resolve().parents[1]/'AnchorPrompt/provenance/historical_supports.json'
    if history_path.exists():
        files.append(history_path)
    save(root/'environment.json',{'python':sys.executable,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),
        'cuda':torch.version.cuda,'sources':{str(f):sha(f) for f in files},'args':vars(args),
        'lambdas':LAMBDAS,'grid':GRID,'selection':'joint support CV','optimizer_steps':0,'evidence':'development on historically reused graphs/query'})
    started=time.time()
    count=0
    for ds in ['Cora','CiteSeer']:
        data=Planetoid(str(prog/'data/Planetoid'),ds,transform=NormalizeFeatures())[0]
        # Data loader contains all labels; below only train labels are passed into adaptation.
        train_nodes=torch.where(data.train_mask)[0]
        train_y=data.y[train_nodes].clone()
        c=int(train_y.max())+1
        query=torch.where(data.test_mask)[0]
        pools=[train_nodes[train_y==k] for k in range(c)]
        data.y[~data.train_mask]=-1  # Poison all non-support-pool labels until independent scoring reload.
        x,edge=data.x.cuda(),data.edge_index.cuda()
        p,q=transition(edge,len(x)),attribute_graph(x)
        save(root/f'{ds}_data.json',{'x_sha256':digest(data.x),'edge_sha256':digest(data.edge_index),
            'query_sha256':digest(query),'train_mask_sha256':digest(data.train_mask),'n':len(x),'classes':c,
            'processed_file_sha256':sha(prog/'data/Planetoid'/ds/'processed/data.pt'),'transform':'NormalizeFeatures',
            'non_train_labels_poisoned':bool((data.y[~data.train_mask]==-1).all())})
        old=set()
        for file in (prog/'Experiment').rglob('train_idx.pt'):
            if ds.lower() not in str(file).lower():
                continue
            idx=torch.load(file,map_location='cpu',weights_only=True)
            if isinstance(idx,torch.Tensor):
                old.add(tuple(sorted(idx.reshape(-1).tolist())))
        history=Path(__file__).resolve().parents[1]/'AnchorPrompt/provenance/historical_supports.json'
        if history.exists():
            old.update(tuple(a) for a in json.loads(history.read_text()).get(ds,[]))
        for previous in ['AnchorPrompt','CovariancePrompt','BayesFeaturePrompt','CyclePrompt']:
            for f in (Path(__file__).resolve().parents[1]/previous/'results/stage_a/splits'/ds).glob('*.json'):
                old.add(tuple(sorted(json.loads(f.read_text())['support'])))
        splits=[]
        seen=set()
        for seed in range(args.seed_start,args.seed_start+args.seed_count):
            gen=generator(seed,'cpu')
            s=torch.cat([a[torch.randperm(len(a),generator=gen)[:5]] for a in pools])
            labels=data.y[s].clone()
            key=tuple(sorted(s.tolist()))
            assert key not in old and key not in seen
            seen.add(key)
            assert data.train_mask[s].all() and not torch.isin(s,query).any()
            splits.append((seed,s,labels))
            save(root/'splits'/ds/f'{seed}.json',{'support':s.tolist(),'support_labels':labels.tolist(),
                 'query':query.tolist(),'support_sha256':digest(s),'support_labels_sha256':digest(labels),
                 'query_sha256':digest(query),'excluded_known_support_sets':len(old),'duplicate':False})
        for ck in ['DGI','GraphMAE','GraphCL','GraphMAE2']:
            ckpath=prog/'Experiment/pre_trained_model'/ds/f'{ck}.GCN.128hidden_dim.pth'
            model=GCN(x.shape[1],128,128,num_layer=2).cuda().eval().requires_grad_(False)
            import shutil
            snapshot=Path(__file__).parent/'provenance'/ds/ckpath.name
            snapshot.parent.mkdir(parents=True,exist_ok=True)
            if snapshot.exists(): assert sha(snapshot)==sha(ckpath)
            else: shutil.copy2(ckpath,snapshot)
            meta=Path(str(ckpath)+'.json')
            if meta.exists(): shutil.copy2(meta,Path(str(snapshot)+'.json'))
            model.load_state_dict(torch.load(ckpath,map_location='cpu',weights_only=True),strict=True)
            rm=None
            before={k:digest(v) for k,v in model.state_dict().items()}
            with torch.no_grad():
                ex=Experiment(model,rm,x,edge,p,q)
                for seed,sc,yc in splits:
                    scores,heads,diag,folds=ex.predict(sc.cuda(),yc.cuda(),c)
                    score_cpu={name:t[query.cuda()].double().cpu() for name,t in scores.items()}
                    record={'dataset':ds,'checkpoint':ck,'checkpoint_sha256':sha(ckpath),'seed':seed,
                        'support':sc.tolist(),'support_labels':yc.tolist(),'support_sha256':digest(sc),
                        'query':query.tolist(),'query_sha256':digest(query),'heads':heads,'diagnostics':diag,'folds':folds,
                        'scores':score_cpu,'prediction_sha256':{k:digest(v) for k,v in score_cpu.items()}}
                    dest=root/'sealed'/ds/ck/f'{seed}.pt'
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    torch.save(record,dest)
                    count+=len(score_cpu)
                    print('SEALED',ds,ck,seed,len(score_cpu),round(time.time()-started,1),flush=True)
                after={k:digest(v) for k,v in model.state_dict().items()}
                assert before==after
                save(root/f'{ds}_{ck}_audit.json',{'frozen':True,'before':before,'after':after,
                    **ex.audit})
    # Only after ALL models and splits are sealed do we access query labels for scoring.
    save(root/'sealed_complete.json',{'records':count,'seconds':time.time()-started,
        'files':{str(f.relative_to(root)):sha(f) for f in sorted((root/'sealed').rglob('*.pt'))}})
    rows=[]
    for ds in ['Cora','CiteSeer']:
        data=Planetoid(str(prog/'data/Planetoid'),ds,transform=NormalizeFeatures())[0]
        qi=torch.where(data.test_mask)[0]
        save(root/f'{ds}_evaluation_truth.json',{'query':qi.tolist(),'truth':data.y[qi].tolist(),'truth_sha256':digest(data.y[qi])})
        for f in sorted((root/'sealed'/ds).rglob('*.pt')):
            obj=torch.load(f,map_location='cpu',weights_only=True)
            truth=data.y[obj['query']]
            for name,score in obj['scores'].items():
                pred=score.argmax(1)
                f1=[]
                for k in range(score.shape[1]):
                    tp=((pred==k)&(truth==k)).sum().item()
                    den=(pred==k).sum().item()+(truth==k).sum().item()
                    f1.append(2*tp/den if den else 0)
                rows.append({'dataset':ds,'checkpoint':obj['checkpoint'],'seed':obj['seed'],'method':name,
                    'accuracy':float((pred==truth).double().mean()),'macro_f1':float(np.mean(f1)),
                    'prediction_changed_vs_fixed':float((pred!=obj['scores']['fixed'].argmax(1)).double().mean()),
                    'lambda':obj['heads'][name].get('lambda'),'support_mse':obj['heads'][name].get('support_mse'),
                    'prediction_sha256':obj['prediction_sha256'][name]})
    save(root/'metrics.json',rows)
    save(root/'complete.json',{'records':count,'seconds':time.time()-started,'state':'complete'})


if __name__=='__main__':
    main()
