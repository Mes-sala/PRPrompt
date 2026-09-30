"""Explicit repaired GraphMultiGprompt pretraining with fixed unlabeled views."""
from pathlib import Path
import sys,os,time,json,random,argparse,traceback,inspect
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
sys.path.insert(0,str(BASE));sys.path.insert(0,str(ROOT/'runtime'))
import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data,Batch
from torch_geometric.datasets import TUDataset
from prompt_graph.pretrain.MultiGPrompt import GraphPrePrompt,tu_prompt_pretrain_sample
from prompt_graph.utils import process
from repair31_common import admission,fingerprint
from run_transfer5 import full_key
from train_masked41 import digest,write


def augment(adj):
    # Uniform undirected edge deletion/addition, without materializing O(n^2) nonedges.
    pairs=sorted({(min(int(a),int(b)),max(int(a),int(b))) for a,b in zip(*adj.nonzero())})
    count=int(len(pairs)*.1/4);remaining=set(pairs)-set(random.sample(pairs,count))
    added=set();n=adj.shape[0]
    while len(added)<count:
        i=random.randrange(n);j=random.randrange(n)
        if i==j:continue
        pair=(min(i,j),max(i,j))
        if pair not in remaining:added.add(pair)
    pairs=remaining|added
    rr=[];cc=[]
    for i,j in sorted(pairs):
        rr.append(i);cc.append(j)
        if i!=j:rr.append(j);cc.append(i)
    return sp.csr_matrix((np.ones(len(rr)),(rr,cc)),shape=adj.shape)


def tensor_adj(adj):
    a=(process.normalize_adj(adj+sp.eye(adj.shape[0]))+sp.eye(adj.shape[0])).tocoo()
    return torch.sparse_coo_tensor(torch.tensor(np.stack([a.row,a.col]),dtype=torch.long),torch.tensor(a.data,dtype=torch.float32),a.shape).coalesce()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    cfg=json.loads((ROOT/'multig_pretrain_freeze.json').read_text());assert digest(Path(__file__))==cfg['script_sha256']
    for n,h in cfg['runtime_sha256'].items():assert digest(ROOT/'runtime'/n)==h,n
    native=inspect.getsource(GraphPrePrompt.pretrain)
    assert '.backward(' not in native and '.step(' not in native
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    out=ROOT/('multig_smoke' if a.smoke else 'multig_training');out.mkdir(exist_ok=True);done=[];start=time.time()
    for ds in cfg['datasets']:
        job=out/ds/'GraphMultiGprompt_repaired_s101';job.mkdir(parents=True,exist_ok=True)
        if (job/'record.json').exists():
            r=json.loads((job/'record.json').read_text());assert r['freeze_sha256']==digest(ROOT/'multig_pretrain_freeze.json')
            assert digest(job/'encoder.pth')==r['checkpoint_sha256'];done.append(ds);continue
        random.seed(101);np.random.seed(101);torch.manual_seed(101);torch.cuda.manual_seed_all(101)
        data=TUDataset(str(BASE/'graph_assets/raw_datasets'),ds,use_node_attr=True)
        graphs=[Data(x=g.x.clone(),edge_index=g.edge_index.clone()) for g in data]
        assert [full_key(g) for g in graphs]==admission()['datasets'][ds]['keys']
        task=GraphPrePrompt(graphs,data.num_features,1,ds,128,'prelu',.9,.9,.1,2,.3,0).cuda()
        initial=fingerprint(task.gcn)
        with (job/'random_encoder.pth').open('xb') as f:torch.save({k:v.detach().cpu().clone() for k,v in task.gcn.state_dict().items()},f)
        order=torch.randperm(len(graphs),generator=torch.Generator().manual_seed(50101)).tolist();views=[]
        for start_i in range(0,len(order),32):
            ids=order[start_i:start_i+32];b=Batch.from_data_list([graphs[i] for i in ids]);assert b.y is None
            n=b.num_nodes;adj=sp.csr_matrix((np.ones(b.num_edges),(b.edge_index[0].numpy(),b.edge_index[1].numpy())),shape=(n,n))
            sample=tu_prompt_pretrain_sample(adj.copy(),50)
            views.append({'ids':ids,'x':b.x.unsqueeze(0),'adj':tensor_adj(adj),'aug1':tensor_adj(augment(adj)),'aug2':tensor_adj(augment(adj)),'sample':sample})
        with (job/'fixed_views.pt').open('xb') as f:torch.save(views,f)
        opt=torch.optim.Adam(task.parameters(),lr=.0001,weight_decay=0)
        losses=[];begin=time.time();epochs=1 if a.smoke else cfg['epochs'];updates=0
        for epoch in range(1,epochs+1):
            epoch_losses=[]
            for vi,v in enumerate(views):
                x=v['x'].cuda();n=x.shape[1];shuf=x[:,torch.randperm(n,device='cuda'),:]
                adj=v['adj'].cuda();aa=v['aug1'].cuda();ab=v['aug2'].cuda()
                labels=torch.cat([torch.ones(1,n,device='cuda'),torch.zeros(1,n,device='cuda')],1)
                task.train();opt.zero_grad()
                before_params={k:p.detach().clone() for k,p in task.gcn.named_parameters()} if updates==0 else None
                loss=task(x,shuf,x,x,adj,aa,ab,True,None,None,None,labels,v['sample'])
                assert torch.isfinite(loss)
                if updates==0:
                    assert all(torch.equal(before_params[k],p) for k,p in task.gcn.named_parameters())
                loss.backward();opt.step();updates+=1
                if updates==1:assert any(not torch.equal(before_params[k],p) for k,p in task.gcn.named_parameters())
                epoch_losses.append(float(loss.detach()))
                if a.smoke:break
            losses.append(float(np.mean(epoch_losses)))
            if epoch==1 or epoch%10==0:
                write(out/'progress.json',{'status':'running','pid':os.getpid(),'dataset':ds,'epoch':epoch,'epochs':epochs,'updates':updates,'completed_jobs':len(done),'total_jobs':3,'elapsed':time.time()-start})
                print('MULTIG41',ds,epoch,losses[-1],flush=True)
        final=fingerprint(task.gcn);assert initial!=final
        assert all(torch.isfinite(v).all() for v in task.gcn.state_dict().values())
        with (job/'encoder.pth').open('xb') as f:torch.save({k:v.detach().cpu() for k,v in task.gcn.state_dict().items()},f)
        write(job/'record.json',{'dataset':ds,'method':'GraphMultiGprompt','variant':'explicit_backward_step_persistent_optimizer_fixed_sparse_views','seed':101,'epochs':epochs,'updates':updates,'losses':losses,'initial_state':initial,'final_state':final,'seconds':time.time()-begin,'checkpoint_sha256':digest(job/'encoder.pth'),'random_checkpoint_sha256':digest(job/'random_encoder.pth'),'views_sha256':digest(job/'fixed_views.pt'),'freeze_sha256':digest(ROOT/'multig_pretrain_freeze.json'),'class_labels_used':False,'native_no_parameter_update_before_backward_checked':True,'scope':'Repaired training loop; original multi-task loss and GcnLayers, fixed augmented views, sparse adjacency, no process_tu labels; not native unmodified pretrain.'})
        done.append(ds);del task,opt,views;torch.cuda.empty_cache()
    write(out/'complete.json',{'status':'complete','count':len(done),'jobs':done,'smoke':a.smoke,'seconds':time.time()-start,'freeze_sha256':digest(ROOT/'multig_pretrain_freeze.json')})


if __name__=='__main__':
    try:main()
    except Exception:
        write(ROOT/('multig_smoke_failure.json' if '--smoke' in sys.argv else 'multig_training_failure.json'),{'status':'failed','traceback':traceback.format_exc(),'time':time.time()});raise
