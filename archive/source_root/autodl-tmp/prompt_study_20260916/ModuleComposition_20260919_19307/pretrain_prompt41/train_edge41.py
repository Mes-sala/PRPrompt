"""Native edge objectives on audited graph inputs, with explicit optimizer variants."""
from pathlib import Path
import sys,os,time,json,hashlib,random,importlib,argparse,traceback
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
sys.path.insert(0,str(BASE));sys.path.insert(0,str(ROOT/'runtime'))
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data,Batch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import negative_sampling
from repair31_common import admission,weight,fingerprint
from run_transfer5 import full_key
from train_masked41 import digest,write


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    cfg=json.loads((ROOT/'edge_pretrain_freeze.json').read_text())
    assert digest(Path(__file__))==cfg['script_sha256']
    for n,h in cfg['runtime_sha256'].items():assert digest(ROOT/'runtime'/n)==h,n
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    out=ROOT/('edge_smoke' if a.smoke else 'edge_training');out.mkdir(exist_ok=True);done=[];start=time.time()
    for ds in cfg['datasets']:
        data=TUDataset(str(BASE/'graph_assets/raw_datasets'),ds,use_node_attr=True)
        graphs=[Data(x=g.x.clone(),edge_index=g.edge_index.clone()) for g in data]
        assert [full_key(g) for g in graphs]==admission()['datasets'][ds]['keys']
        for method in cfg['methods']:
            for variant in cfg['optimizer_variants']:
                job=out/ds/(method+'_'+variant+'_s101');job.mkdir(parents=True,exist_ok=True)
                if (job/'record.json').exists():
                    r=json.loads((job/'record.json').read_text());assert r['freeze_sha256']==digest(ROOT/'edge_pretrain_freeze.json')
                    assert digest(job/'encoder.pth')==r['checkpoint_sha256'];done.append(ds+'/'+job.name);continue
                random.seed(101);np.random.seed(101);torch.manual_seed(101);torch.cuda.manual_seed_all(101)
                union=Batch.from_data_list(graphs);assert union.y is None
                neg=negative_sampling(union.edge_index,num_nodes=union.num_nodes,num_neg_samples=union.num_edges)
                edge=torch.cat([union.edge_index,neg],1)
                labels=torch.cat([torch.ones(union.num_edges),torch.zeros(neg.shape[1])])
                mod=importlib.import_module('prompt_graph.pretrain.'+method)
                original=mod.load4link_prediction_multi_graph
                mod.load4link_prediction_multi_graph=lambda *args,**kwargs:(union,labels,edge,data.num_features,2)
                try:task=getattr(mod,method)(dataset_name=ds,gnn_type='GCN',hid_dim=128,gln=2,num_epoch=100,device=0,num_workers=0)
                finally:mod.load4link_prediction_multi_graph=original
                # The objective implementation is native; full sample batches provide an explicit bounded budget.
                task.dataloader=DataLoader(task.dataloader.dataset,batch_size=len(task.dataloader.dataset),shuffle=False,num_workers=0)
                w=weight(ds,'random_s101');assert digest(w)==cfg['initial_weights'][ds]
                task.gnn.load_state_dict(torch.load(w,map_location='cpu',weights_only=True),strict=True)
                if variant=='full':
                    task.optimizer=torch.optim.Adam(task.parameters(),lr=task.learning_rate,weight_decay=task.weight_decay)
                else:
                    # Native optimizer only covers the GNN. Keep this behavior explicit and avoid accumulating unused head gradients.
                    task.graph_pred_linear.requires_grad_(False)
                initial=fingerprint(task.gnn);head_before=fingerprint(task.graph_pred_linear)
                coverage={n:any(id(p)==id(q) for g in task.optimizer.param_groups for q in g['params']) for n,p in task.named_parameters()}
                losses=[];begin=time.time();steps=1 if a.smoke else cfg['epochs']
                for epoch in range(1,steps+1):
                    loss=task.pretrain_one_epoch();assert np.isfinite(loss),(ds,method,variant,loss);losses.append(loss)
                    if epoch==1 or epoch%10==0:
                        write(out/'progress.json',{'status':'running','pid':os.getpid(),'dataset':ds,'method':method,'variant':variant,'epoch':epoch,'epochs':steps,'completed_jobs':len(done),'total_jobs':12,'elapsed':time.time()-start})
                        print('EDGE41',ds,method,variant,epoch,loss,flush=True)
                final=fingerprint(task.gnn);assert initial!=final
                assert (head_before!=fingerprint(task.graph_pred_linear))==(variant=='full')
                assert all(torch.isfinite(v).all() for v in task.gnn.state_dict().values())
                with (job/'encoder.pth').open('xb') as f:torch.save({k:v.detach().cpu() for k,v in task.gnn.state_dict().items()},f)
                write(job/'record.json',{'dataset':ds,'method':method,'variant':variant,'seed':101,'epochs':steps,'losses':losses,'seconds':time.time()-begin,'initial_state':initial,'final_state':final,'optimizer_coverage':coverage,'head_changed':head_before!=fingerprint(task.graph_pred_linear),'checkpoint_sha256':digest(job/'encoder.pth'),'freeze_sha256':digest(ROOT/'edge_pretrain_freeze.json'),'graph_class_labels_used':False,'scope':'Native ProG edge loss, all admitted unlabeled graphs, fixed full edge/triplet batch per update; native encoder-only and full optimizer variants kept separate; not original mini-batch budget.'})
                done.append(ds+'/'+job.name);del task,union;torch.cuda.empty_cache()
    write(out/'complete.json',{'status':'complete','count':len(done),'jobs':done,'smoke':a.smoke,'seconds':time.time()-start,'freeze_sha256':digest(ROOT/'edge_pretrain_freeze.json')})


if __name__=='__main__':
    try:main()
    except Exception:
        write(ROOT/('edge_smoke_failure.json' if '--smoke' in sys.argv else 'edge_training_failure.json'),{'status':'failed','traceback':traceback.format_exc(),'time':time.time()});raise
