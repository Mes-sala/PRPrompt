"""Auditable disconnected-graph adaptation of three node-pretraining ProG ports."""
from pathlib import Path
import sys,os,json,hashlib,time,random,importlib,argparse,traceback
ROOT=Path(__file__).resolve().parent; BASE=ROOT.parent
sys.path.insert(0,str(BASE));sys.path.insert(0,str(ROOT/'runtime'))
import numpy as np
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data,Batch
from repair31_common import admission,weight,fingerprint
from run_transfer5 import full_key


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p,d):
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(p)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    cfg=json.loads((ROOT/'masked_pretrain_freeze.json').read_text())
    assert digest(Path(__file__))==cfg['script_sha256']
    for n,h in cfg['runtime_sha256'].items():assert digest(ROOT/'runtime'/n)==h,n
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    out=ROOT/('masked_smoke' if a.smoke else 'masked_training');out.mkdir(exist_ok=True)
    done=[];start=time.time()
    for ds in cfg['datasets']:
        data=TUDataset(str(BASE/'graph_assets/raw_datasets'),ds,use_node_attr=True)
        graphs=[Data(x=g.x.clone(),edge_index=g.edge_index.clone()) for g in data]
        assert [full_key(g) for g in graphs]==admission()['datasets'][ds]['keys']
        for method in cfg['methods']:
            job=out/ds/(method+'_s101');job.mkdir(parents=True,exist_ok=True)
            if (job/'record.json').exists():
                r=json.loads((job/'record.json').read_text());assert r['freeze_sha256']==digest(ROOT/'masked_pretrain_freeze.json')
                assert digest(job/'encoder.pth')==r['checkpoint_sha256'];done.append(ds+'/'+method);continue
            random.seed(101);np.random.seed(101);torch.manual_seed(101);torch.cuda.manual_seed_all(101)
            union=Batch.from_data_list(graphs);assert union.y is None
            cls=getattr(importlib.import_module('prompt_graph.pretrain.'+method),method)
            task=cls(dataset_name=ds,gnn_type='GCN',hid_dim=128,gln=2,num_epoch=100,device=0,data=union,num_workers=0)
            w=weight(ds,'random_s101');assert digest(w)==cfg['initial_weights'][ds]
            state=torch.load(w,map_location='cpu',weights_only=True);task.gnn.load_state_dict(state,strict=True)
            if hasattr(task,'encoder_ema'):task.encoder_ema.load_state_dict(state,strict=True)
            initial=fingerprint(task.gnn);losses=[];begin=time.time()
            steps=1 if a.smoke else cfg['epochs']
            for epoch in range(1,steps+1):
                loss=task.pretrain_one_epoch();assert np.isfinite(loss),(ds,method,loss);losses.append(loss)
                if epoch==1 or epoch%10==0:
                    write(out/'progress.json',{'status':'running','pid':os.getpid(),'dataset':ds,'method':method,'epoch':epoch,'epochs':steps,'completed_jobs':len(done),'total_jobs':9,'elapsed':time.time()-start})
                    print('MASKED41',ds,method,epoch,loss,flush=True)
            final=fingerprint(task.gnn);assert initial!=final
            assert all(torch.isfinite(v).all() for v in task.gnn.state_dict().values())
            with (job/'encoder.pth').open('xb') as f:torch.save({k:v.detach().cpu() for k,v in task.gnn.state_dict().items()},f)
            record={'dataset':ds,'method':method,'seed':101,'epochs':steps,'losses':losses,'seconds':time.time()-begin,'initial_state':initial,'final_state':final,'checkpoint_sha256':digest(job/'encoder.pth'),'initial_checkpoint_sha256':digest(w),'freeze_sha256':digest(ROOT/'masked_pretrain_freeze.json'),'query_labels_used':False,'graph_count':len(graphs),'nodes':union.num_nodes,'edges':union.num_edges,'scope':'ProG node-method port applied to disconnected union of original graph inputs; transductive unlabeled graphs; S2GAE/Bandana negatives may cross graph components; not author-native graph benchmark.'}
            write(job/'record.json',record);done.append(ds+'/'+method);del task,union;torch.cuda.empty_cache()
    write(out/'complete.json',{'status':'complete','jobs':done,'count':len(done),'smoke':a.smoke,'freeze_sha256':digest(ROOT/'masked_pretrain_freeze.json'),'seconds':time.time()-start,'query_labels_used':False})


if __name__=='__main__':
    try:main()
    except Exception:
        p=ROOT/('masked_smoke_failure.json' if '--smoke' in sys.argv else 'masked_training_failure.json')
        write(p,{'status':'failed','time':time.time(),'pid':os.getpid(),'traceback':traceback.format_exc()})
        raise
