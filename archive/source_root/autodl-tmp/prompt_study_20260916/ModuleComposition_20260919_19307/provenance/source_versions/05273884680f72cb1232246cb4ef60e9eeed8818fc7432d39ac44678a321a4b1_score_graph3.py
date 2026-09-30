import argparse,csv,json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score,f1_score
from torch_geometric.datasets import TUDataset
import run_graph1 as r
def main(a):
 torch.set_num_threads(2);root=Path(a.root);files=sorted((root/'sealed').glob('*/*.pt'));assert len(files)==json.loads((root/'complete.json').read_text())['count']==60;audit=json.loads((root/'audit.json').read_text());rows=[];truth={}
 for p in files:
  assert r.sha(p)==audit['sealed_sha'][str(p.relative_to(root))];rec=torch.load(p,weights_only=True);ds=rec['dataset'];s,y,q=rec['support'],rec['support_labels'],rec['query']
  if ds not in truth:truth[ds]=torch.tensor([int(g.y) for g in TUDataset(root=str(Path(a.assets)/'raw_datasets'),name=ds,use_node_attr=True)])
  assert torch.equal(y,truth[ds][s]);yt=truth[ds][q].numpy()
  for n,sc in rec['scores'].items():
   assert r.digest(sc)==rec['scores_sha'][n];pred=sc.argmax(1).numpy();rows.append({'dataset':ds,'seed':rec['seed'],'method':n,'accuracy':accuracy_score(yt,pred)*100,'macro_f1':f1_score(yt,pred,average='macro')*100})
 with (root/'metrics.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 methods=sorted(set(v['method'] for v in rows));datasets=list(truth);means={n:{ds:float(np.mean([v['accuracy'] for v in rows if v['method']==n and v['dataset']==ds])) for ds in datasets} for n in methods}
 raw_methods=[n for n in methods if (n.startswith('direct_') or n.startswith('svm_') or n.startswith('ridge_') or n.startswith('prototype_')) and not any(v in n for v in ['RH','RM','PH','PM'])]+['multiscale_direct','wl_bank','attribute_bank','structure_bank']
 best_raw={ds:max(raw_methods,key=lambda n:means[n][ds]) for ds in datasets};paired={};rng=np.random.default_rng(19307)
 for n in ['six','five','multiscale_full','multiscale_half','expanded_flat']:
  paired[n]={}
  for b in ['best_fixed_direct','svm_direct_selected','ridge_direct_selected','multiscale_direct','direct_02','direct_0123','direct6','permuted_six','six']:
   paired[n][b]={}
   for ds in datasets:
    bn=best_raw[ds] if b=='best_fixed_direct' else b;mr={v['seed']:v for v in rows if v['method']==n and v['dataset']==ds};br={v['seed']:v for v in rows if v['method']==bn and v['dataset']==ds};d=np.array([mr[s]['accuracy']-br[s]['accuracy'] for s in sorted(mr)]);ci=np.percentile(d[rng.integers(0,len(d),(10000,len(d)))].mean(1),[2.5,97.5]);paired[n][b][ds]={'baseline':bn,'mean_pp':float(d.mean()),'ci95':ci.tolist(),'wtl':[int((d>1e-9).sum()),int((abs(d)<=1e-9).sum()),int((d< -1e-9).sum())],'f1_pp':float(np.mean([mr[s]['macro_f1']-br[s]['macro_f1'] for s in mr]))}
 r.save(root/'analysis.json',{'means':means,'best_fixed_direct':best_raw,'paired':paired,'rows':len(rows),'boundary':'Best fixed direct is a conservative query-informed comparator, never a fitted candidate or deployment router. NCI tasks related; ENZYMES previously developed.','source_sha':r.sha(__file__)})
 lines=['# 图分类第三轮：新数据与强直接控制','','|方法|'+'|'.join(datasets)+'|','|---|'+'|'.join(['---:']*len(datasets))+'|']
 for n in methods:lines.append('|'+n+'|'+'|'.join(f'{means[n][ds]:.3f}' for ds in datasets)+'|')
 (root/'summary.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'main':{n:means[n] for n in ['six','five','multiscale_full','multiscale_direct','multiscale_half']},'best_raw':best_raw,'six':paired['six']['best_fixed_direct'],'five':paired['five']['best_fixed_direct'],'multi':paired['multiscale_full']['best_fixed_direct']},indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--assets',required=True);main(ap.parse_args())
