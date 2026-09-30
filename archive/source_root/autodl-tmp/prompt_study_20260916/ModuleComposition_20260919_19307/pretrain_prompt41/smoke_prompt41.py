"""Support-only encoder/Prompt admission; never evaluates query labels."""
from pathlib import Path
import sys,json,hashlib,time
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
sys.path.insert(0,str(BASE))
sys.path.insert(0,str(ROOT/'runtime'))
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data,Batch
from torch_geometric.nn import global_mean_pool
from prompt_graph.model import GCN
from prompt_graph.prompt.GPF import GPF_plus
from prompt_graph.prompt.GPrompt import Gprompt
from stable_random_graph import stats
from repair31_common import episodes,episode,weight,fingerprint


def nodes(model,batch,feature=None,channel=None):
    x=batch.x if feature is None else feature.add(batch.x)
    hs=model(x,batch.edge_index,prompt_type='DAGPrompT')[1:]
    if channel is not None:
        hs=list(hs)
        hs[-1]=channel(hs[-1])
    return hs


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    before=time.time(); results=[];baseline=[]
    for ds in ['ENZYMES','MUTAG','PROTEINS']:
        data=TUDataset(str(BASE/'graph_assets/raw_datasets'),ds,use_node_attr=True)
        ep=episode(episodes(ds)[0]); y=torch.tensor(ep['support_labels'],device='cuda',dtype=torch.long)
        graphs=[Data(x=data[int(i)].x.clone(),edge_index=data[int(i)].edge_index.clone()) for i in ep['support']]
        b=Batch.from_data_list(graphs); assert b.y is None
        state=torch.load(weight(ds,'random_s101'),map_location='cpu',weights_only=True)
        model=GCN(data.num_features,128,128,num_layer=2).eval().requires_grad_(False)
        model.load_state_dict(state,strict=True); model.double()
        with torch.no_grad():
            b.x=b.x.double(); hs=nodes(model,b)
            pooled=[v for h in hs for v in stats(h,b.batch)]
            with np.load(BASE/'repair31_features'/ds/'random_s101_pooled.npz') as saved:
                err=max(float(np.max(np.abs(a.numpy()-saved[str(i)][ep['support']]))) for i,a in enumerate(pooled))
            assert err<1e-10,(ds,err)
        baseline.append({'dataset':ds,'all_six_pools_max_error_to_original':err,'support_only':True})
        for method in ['GPF-plus','Gprompt','GPF-plus+Gprompt']:
            model.float().cuda();model.load_state_dict(state,strict=True)
            initial=fingerprint(model)
            torch.manual_seed(4242)
            feature=GPF_plus(data.num_features,20).cuda() if 'GPF-plus' in method else None
            channel=Gprompt(128).cuda() if 'Gprompt' in method else None
            torch.manual_seed(42);head=torch.nn.Linear(128,len(set(ep['support_labels']))).cuda()
            params=(list(feature.parameters()) if feature is not None else [])+(list(channel.parameters()) if channel is not None else [])
            if channel is None:params+=list(head.parameters())
            originals=[p.detach().clone() for p in params]
            opt=torch.optim.Adam(params,lr=.001,weight_decay=.0005)
            assert not any(id(p) in {id(q) for q in params} for p in model.parameters())
            bb=b.clone().to('cuda');bb.x=bb.x.float();losses=[]
            for step in range(2):
                opt.zero_grad();hs=nodes(model,bb,feature,channel)
                z=global_mean_pool(hs[-1],bb.batch)
                if channel is None:logits=head(z)
                else:
                    center=torch.stack([z[y==c].mean(0) for c in range(head.out_features)])
                    logits=F.cosine_similarity(z[:,None,:],center[None,:,:],dim=-1)/.1
                loss=F.cross_entropy(logits,y);assert torch.isfinite(loss)
                loss.backward()
                assert all(p.grad is None for p in model.parameters())
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
                opt.step();losses.append(float(loss.detach()))
            assert initial==fingerprint(model)
            changed=sum(not torch.equal(a,p) for a,p in zip(originals,params));assert changed
            with torch.no_grad():
                hs=nodes(model,bb,feature,channel)
                manual=global_mean_pool(hs[-1],bb.batch)
                x=bb.x if feature is None else feature.add(bb.x)
                native=model(x,bb.edge_index,bb.batch,prompt=channel,prompt_type='Gprompt' if channel is not None else None)
                err=float((manual-native).abs().max());assert err<1e-6
            results.append({'dataset':ds,'method':method,'steps':2,'losses':losses,'changed_parameter_tensors':changed,'frozen_encoder':True,'native_final_pool_error':err,'labels':'sealed support only'})
            print('PROMPT41_SMOKE',ds,method,flush=True)
    output={'status':'support_only_admission_passed','baseline_pool_parity':baseline,'prompt_cases':results,'query_predictions_generated':False,'new_pretraining_methods_admitted':False,'seconds':time.time()-before,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    dest=ROOT/'smoke_prompt_admission.json'
    with dest.open('x') as f:json.dump(output,f,indent=2)


if __name__=='__main__':main()
