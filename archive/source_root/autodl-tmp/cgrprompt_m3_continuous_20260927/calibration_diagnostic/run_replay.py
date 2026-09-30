from pathlib import Path
import sys
OLD=Path('/root/autodl-tmp/cgrprompt_validation_20260926');sys.path.insert(0,str(OLD))
from run_core import *
ROOT=Path(__file__).resolve().parent;PARENT=ROOT.parent
@torch.no_grad()
def run():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 manifest=json.loads((PARENT/'transport_round02_06.json').read_text());jobs=json.loads((PARENT/'round05/confirm_plan.json').read_text())['jobs'];assert len(jobs)==4
 assert g.verify_sources()==json.loads((BASE/'plan.json').read_text())['source_manifest_sha256']
 for job in jobs:
  data,pool,query,C,protocol=g.data_load(job['dataset']);x=data.x.cuda();edges=data.edge_index.cuda();ei,ew=g.gcn_norm(edges,num_nodes=len(x),add_self_loops=True);adj=torch.sparse_coo_tensor(ei.flip(0),ew,(len(x),len(x))).coalesce();r=torch.sparse.mm(adj,torch.sparse.mm(adj,x));r-=r.mean(0,keepdim=True);op,_,_=knn(r);ix=op.indices()
  for cohort in ['round05','round06']:
   for folder in sorted((PARENT/cohort/'confirm'/job['id']).iterdir()):
    if not folder.is_dir():continue
    seed=int(folder.name);out=ROOT/cohort/job['id']/str(seed);out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():continue
    cached=folder/'initial_scores.npz';assert sha(cached)==manifest[str(cached.relative_to(PARENT))];old=json.loads((folder/'result.json').read_text());assert sha(folder/'prediction.npz')==old['prediction_sha256'];prev=np.load(folder/'prediction.npz');support=torch.from_numpy(prev['support']);assert np.array_equal(query.numpy(),prev['query']) and not torch.isin(support,query).any();si=support.cuda();sy=data.y[support].cuda();scores=np.load(cached);arrays={};parity={};temps={}
    for source in scores.files:
     score=torch.from_numpy(scores[source]).cuda();labels=score.argmax(1);agree=torch.zeros(len(score),device=x.device).scatter_add_(0,ix[0],op.values()*(labels[ix[0]]==labels[ix[1]]).float());local=(1+3*(1-agree))[:,None];temps[source]=float(local.mean())
     for method,T in [('fixed',1),('temp2',2),('temp4',4),('localtemp',local),('meanT',local.mean())]:
      p=propagate(score/T,op,si,sy,C)[query.cuda()].double().softmax(-1).cpu().numpy();key=source+'__'+method;arrays[key]=p
      if method in ['fixed','localtemp']:parity[key]=float(abs(p-prev[key]).max());assert parity[key]<2e-6
    np.savez_compressed(out/'prediction.npz',support=support.numpy(),query=query.numpy(),**arrays);ph=sha(out/'prediction.npz');y=data.y[query].numpy();np.save(out.parent/'query_labels.npy',y)
    save(out/'result.json',dict(job=job['id'],cohort=cohort,seed=seed,source_score_sha256=sha(cached),runner_sha256=sha(__file__),prediction_sha256=ph,parity=parity,temperatures=temps,metrics={k:metrics(y,p) for k,p in arrays.items()}));print(json.dumps(dict(cohort=cohort,job=job['id'],seed=seed)),flush=True)
 save(ROOT/'complete.json',dict(records=len(list(ROOT.glob('round*/*/*/result.json')))))
if __name__=='__main__':run()
