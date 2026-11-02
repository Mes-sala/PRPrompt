"""Run the locked graph recipe; checkpoints and outputs are never repository assets."""
import argparse,csv,json,hashlib,statistics
from pathlib import Path
import torch
from sklearn.metrics import balanced_accuracy_score,f1_score
from prprompt import TwoLayerGCN,prepare_graph_reference,fit_graph
from prprompt.data import load_tu,sample_graph_task


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True);p.add_argument('--data-root',default='data')
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--split-dir',help='Optional dataset-specific directory: SHOT/TASK/{train,test}_idx.pt')
    p.add_argument('--shots',type=int,nargs='+',default=[1,3,5]);p.add_argument('--tasks',type=int,default=20)
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2]);p.add_argument('--device',default='cpu')
    p.add_argument('--epochs',type=int,default=1000,help='Use 1000 for the locked protocol; smaller values are smoke checks')
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    graphs,labels,c=load_tu(args.dataset,args.data_root)
    backbone=TwoLayerGCN.from_checkpoint(args.checkpoint,graphs[0].x.shape[1])
    ref=prepare_graph_reference(graphs);torch.save(ref.state_dict(),out/'reference.pt')
    (out/'selection.json').write_text(json.dumps(ref.selection,indent=2))
    rows=[]
    for k in args.shots:
        for t in range(args.tasks):
            if args.split_dir:
                folder=Path(args.split_dir)/str(k)/str(t)
                si=torch.load(folder/'train_idx.pt',weights_only=True).long();qi=torch.load(folder/'test_idx.pt',weights_only=True).long()
            else:si,qi=sample_graph_task(labels,k,t)
            if si.unique().numel()!=si.numel() or qi.unique().numel()!=qi.numel() or torch.isin(si,qi).any():raise ValueError('Invalid overlapping or duplicate split')
            if not qi.numel() or qi.min()<0 or qi.max()>=len(graphs):raise ValueError('Invalid query IDs')
            if not torch.equal(torch.bincount(labels[si],minlength=c),torch.full((c,),k)):raise ValueError('Not exact k-shot')
            for s in args.seeds:
                dest=out/str(k)/str(t)/str(s);dest.mkdir(parents=True)
                fit=fit_graph(backbone,graphs,si,labels[si],c,reference=ref,seed=s,device=args.device,epochs=args.epochs)
                torch.save(fit.state_dict(),dest/'state.pt')
                scores=fit.scores(graphs,qi);truth=labels[qi];pred=scores.softmax(1).argmax(1)
                torch.save(dict(scores=scores,predictions=pred,truth=truth,support=si,query=qi),dest/'predictions.pt')
                row=dict(dataset=args.dataset,shot=k,task=t,seed=s,accuracy=float((pred==truth).double().mean()),balanced_accuracy=float(balanced_accuracy_score(truth,pred)),macro_f1=float(f1_score(truth,pred,average='macro',zero_division=0)),steps=len(fit.trace),**fit.config)
                rows.append(row);print(row,flush=True)
    with (out/'per_run.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[]
    for k in args.shots:
        cell=dict(shot=k)
        for metric in ['accuracy','balanced_accuracy','macro_f1']:
            taskmeans=[statistics.mean(x[metric] for x in rows if x['shot']==k and x['task']==t) for t in range(args.tasks)]
            cell[metric]=dict(mean=100*statistics.mean(taskmeans),task_sd=100*statistics.stdev(taskmeans) if len(taskmeans)>1 else None)
        summary.append(cell)
    (out/'summary.json').write_text(json.dumps(dict(rows=summary,arguments=vars(args),checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),aggregation='seed mean within task, then task mean/sample SD',smoke=args.epochs!=1000,protocol='historical 80% query; unlabelled full-pool transductive statistics'),indent=2))

if __name__=='__main__':main()
