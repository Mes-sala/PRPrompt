"""No pooled conditions; task-block statistics and paired effects only."""
import argparse
import collections
import csv
import json
from pathlib import Path
import numpy as np
from prprompt.protocol import save_json

METRICS=['accuracy','macro_f1','micro_f1','roc_auc_macro','average_precision_macro']


def summarize(w):
    p=json.loads((w/'plan.json').read_text());out=w/'analysis';out.mkdir(exist_ok=True)
    expected={(j['dataset'],k,s,t,m) for j in p['jobs'] for k in p['shots'] for s in p['split_seeds'] for t in p['train_seeds'] for m in p['methods']}
    seen=set();groups=collections.defaultdict(lambda:collections.defaultdict(list));rows=[]
    for f in sorted((w/'results').glob('*/*/*/*/*/metrics.json')):
        r=json.loads(f.read_text());key=tuple(r[k] for k in ['dataset','shots','split_seed','train_seed','method'])
        assert key in expected and key not in seen and not r['smoke'];seen.add(key);rows.append(r)
        assert all(np.isfinite(r['metrics'][m]) for m in METRICS)
        groups[(r['dataset'],r['shots'],r['method'])][r['split_seed']].append(r['metrics'])
    assert seen==expected,(len(seen),len(expected))
    blocks={};table=[];jobs={j['dataset']:j for j in p['jobs']}
    for (d,k,m),draws in sorted(groups.items()):
        assert len(draws)==20 and all(len(v)==3 for v in draws.values())
        values=np.array([[np.mean([r[metric] for r in draws[s]]) for metric in METRICS] for s in p['split_seeds']])
        blocks[d,k,m]=values
        row=dict(dataset=d,shots=k,method=m,label=m,checkpoint_id=jobs[d]['id'],family=jobs[d]['family'],support_draws=20,training_repeats=3)
        for i,metric in enumerate(METRICS):
            row[metric+'_mean']=float(values[:,i].mean());row[metric+'_std_over_support_draws']=float(values[:,i].std(ddof=1))
        table.append(row)
    paired=[];rng=np.random.default_rng(p['bootstrap_seed'])
    for d in jobs:
        for k in p['shots']:
            for ref in ['FrozenPrototype','FullFineTune','NoRouting']:
                delta=blocks[d,k,'PRPrompt']-blocks[d,k,ref]
                boot=delta[rng.integers(0,20,size=(10000,20))].mean(1)
                for i,metric in enumerate(METRICS):
                    lo,hi=np.quantile(boot[:,i],[.025,.975])
                    paired.append(dict(dataset=d,shots=k,method='PRPrompt',reference=ref,metric=metric,
                        mean_delta=float(delta[:,i].mean()),ci95_low=float(lo),ci95_high=float(hi),interval_source='original_paired_interval'))
    save_json(out/'comparison.json',table);save_json(out/'paired.json',paired)
    for name,data in [('comparison',table),('paired',paired)]:
        with (out/(name+'.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    inventory=json.loads((w/'split_inventory.json').read_text())
    save_json(out/'coverage.json',dict(expected=len(expected),completed=len(seen),missing=0,
        task_splits=len(inventory),nominal_shot_shortages=[t for t in inventory if not t['nominal_shot_complete']],
        overlapping_tasks=sum(t['support_query_overlap']>0 for t in inventory),
        saved_readout_max_error=max(r['saved_readout_max_error'] for r in rows),
        protocol=p['protocol'],unit_hash_rescans=False))
    lines=['# 原 ProG 实现复算结果','',p['protocol'],'',
           '均值±样本标准差，单位为百分数。先平均同一任务的3个训练seed，再统计20个任务；不跨数据集或shot平均。','']
    for d in jobs:
        lines += ['## '+d,'','| 方法 | 1-shot | 3-shot | 5-shot |','|---|---:|---:|---:|']
        for m in p['methods']:
            cells=[]
            for k in p['shots']:
                r=next(r for r in table if (r['dataset'],r['shots'],r['method'])==(d,k,m))
                cells.append(f"{100*r['accuracy_mean']:.2f} ± {100*r['accuracy_std_over_support_draws']:.2f}")
            lines.append('| '+m+' | '+' | '.join(cells)+' |')
        subset=[t for t in inventory if t['dataset']==d]
        lines+=['',f"支持／测试重叠节点范围：{min(t['support_query_overlap'] for t in subset)}–{max(t['support_query_overlap'] for t in subset)}；每类未达到名义shot的任务：{sum(not t['nominal_shot_complete'] for t in subset)}/60。",'']
    (out/'结果总结.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('workspace',type=Path);summarize(p.parse_args().workspace)
