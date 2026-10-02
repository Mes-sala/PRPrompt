"""Publish completed original-ProG results; never change training or source values."""
from pathlib import Path
import collections
import csv
import hashlib
import json
import shutil
import time
import numpy as np

REPO=Path(__file__).resolve().parents[1]
W=REPO/'experiments/outputs/prog_original_5090_20261002'
P=REPO/'docs/paper/PRPrompt_LaTeX_双栏_20261002'
DOC=REPO/'docs/results/prog_original_20261002'
D=['Cora','CiteSeer','PubMed']
M=['accuracy','macro_f1','micro_f1','roc_auc_macro','average_precision_macro']

def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def replace(p,a,b):
    s=p.read_text();assert a in s,(str(p),a[:100]);p.write_text(s.replace(a,b))

def write_report():
    """Keep condition-specific effects and counterexamples in the Chinese report."""
    report=(W/'analysis/结果总结.md').read_text().replace(
        'Exact ProG algorithm; overlap retained at explicit user request; not a held-out-only evaluation',
        '复用旧 checkpoint，按原 ProG 的两次独立排列实现复算，保留支持／测试重叠，不作为严格未标注节点泛化评估。5760/5760 单元完成，96 个 smoke 单元单独保存。下表为 Accuracy；五指标完整统计见 comparison.csv，配对统计见 paired.csv。')
    rows=read(DOC/'paired.json')
    index={(r['dataset'],r['shots'],r['metric']):r for r in rows if r['reference']=='NoRouting'}
    report+='\n## 原型路由相对共享提示的配对效果\n\n每格为百分点差值 [95% 区间]。以任务为块做 10000 次 bootstrap，未作多重比较校正；区间描述固定图及 checkpoint 下的任务波动。\n\n| 数据集 | Shot | Accuracy | Macro-F1 |\n|---|---:|---:|---:|\n'
    for d in ['Cora','CiteSeer','PubMed','Photo','Computers','Actor','Chameleon','ogbn-arxiv']:
        for shot in [1,3,5]:
            cells=[]
            for metric in ['accuracy','macro_f1']:
                r=index[d,shot,metric]
                cells.append(f"{100*r['mean_delta']:+.2f} [{100*r['ci95_low']:+.2f}, {100*r['ci95_high']:+.2f}]")
            report+=f'| {d} | {shot} | '+ ' | '.join(cells)+' |\n'
    report+='''
## 结果解释与边界

正文三个数据集的九个条件中，PRPrompt 相对共享提示的 Accuracy 与 Macro-F1 配对区间下界均为正，支持原型路由整体设计的价值。它不隔离原型校正、中心化和图聚合的单独贡献，且共享提示的有效自由度更小。

不支持所有图上普遍占优：Photo、Computers 的 3/5-shot Accuracy 低于全量微调；ogbn-arxiv 的三个 shot Accuracy 均低于全量微调，其中 5-shot 均值非常接近。Actor 的 1-shot Macro-F1 相对共享提示均值差为负，多个条件区间覆盖零；Chameleon 和 ogbn-arxiv 的 1-shot 路由效应也不确定。完整负结果均保留。

ogbn-arxiv 的 7 个 3-shot、13 个 5-shot 任务存在少数类支持不足，实际类别支持数在不足类别中为 1–4，保留原实现不补抽样。四个任务的最小类别数为 1，沿用既有包含自身的原型训练损失，其余按实际数量使用 LOO。全部实验使用一个既有 checkpoint／图，不能估计预训练运行方差。

外部文献 Accuracy 原值不变。虽然本轮采用原 ProG 实现，引用论文的具体划分索引不可得，不能断言其实际存在重叠；不同预训练选择、选参预算和重复次数使文献结果仅作为描述性参照，不做跨来源配对检验。
'''
    (DOC/'结果总结.md').write_text(report)


def main():
    plan=read(W/'plan.json');coverage=read(W/'analysis/coverage.json')
    assert coverage['completed']==coverage['expected']==5760 and coverage['missing']==0
    assert read(W/'status.json')['phase']=='complete'
    table=read(W/'analysis/comparison.json');paired=read(W/'analysis/paired.json')
    idx={(r['dataset'],r['shots'],r['method']):r for r in table}
    rows=[read(f) for f in sorted((W/'results').glob('*/*/*/*/*/metrics.json'))]
    expected={(j['dataset'],k,s,t,m) for j in plan['jobs'] for k in plan['shots'] for s in plan['split_seeds'] for t in plan['train_seeds'] for m in plan['methods']}
    keys=[tuple(r[k] for k in ['dataset','shots','split_seed','train_seed','method']) for r in rows]
    assert len(keys)==len(set(keys))==len(expected)==5760 and set(keys)==expected
    assert all(np.isfinite(r['metrics'][metric]) for r in rows for metric in M)
    groups=collections.defaultdict(lambda:collections.defaultdict(list))
    for r in rows:groups[r['dataset'],r['shots'],r['method']][r['split_seed']].append(r['metrics'])
    maximum=0.;blocks={}
    for key,draws in groups.items():
        values=np.array([[np.mean([r[m] for r in draws[s]]) for m in M] for s in plan['split_seeds']])
        blocks[key]=values
        for i,m in enumerate(M):
            maximum=max(maximum,abs(values[:,i].mean()-idx[key][m+'_mean']),abs(values[:,i].std(ddof=1)-idx[key][m+'_std_over_support_draws']))
    assert maximum<1e-12
    rng=np.random.default_rng(plan['bootstrap_seed']);effects={(r['dataset'],r['shots'],r['reference'],r['metric']):r for r in paired}
    for j in plan['jobs']:
        d=j['dataset']
        for k in plan['shots']:
            for ref in ['FrozenPrototype','FullFineTune','NoRouting']:
                delta=blocks[d,k,'PRPrompt']-blocks[d,k,ref]
                boot=delta[rng.integers(0,20,size=(10000,20))].mean(1)
                for i,m in enumerate(M):
                    r=effects[d,k,ref,m];lo,hi=np.quantile(boot[:,i],[.025,.975])
                    assert max(abs(delta[:,i].mean()-r['mean_delta']),abs(lo-r['ci95_low']),abs(hi-r['ci95_high']))<1e-12
    # Only read support/query members of each archive, not full prediction matrices.
    for j in plan['jobs']:
        for k in plan['shots']:
            for seed in plan['split_seeds']:
                with np.load(W/'splits'/j['dataset']/f'{k}_{seed}.npz') as f:s,q=f['support'],f['query']
                for train in plan['train_seeds']:
                    for method in plan['methods']:
                        with np.load(W/'results'/j['dataset']/str(k)/str(seed)/str(train)/method/'predictions.npz') as f:
                            assert np.array_equal(s,f['support']) and np.array_equal(q,f['query'])
    binaries=[str(f.relative_to(W)) for f in W.rglob('*') if f.is_file() and (f.name.endswith(('.pt','.pth','.pt.gz','.ckpt','.safetensors')) or f.name.startswith('state.pt'))]
    assert not binaries,binaries[:3]
    audit=dict(status='passed',completed_units=5760,missing_units=0,finite_metrics=True,
               paired_support_query_indices_verified=True,mean_std_and_bootstrap_recomputed=True,
               max_summary_error=maximum,saved_readout_max_error=coverage['saved_readout_max_error'],
               exhaustive_file_hashes=False,fresh_cuda_bitwise_audit=False,local_model_binary_files=0)
    save(REPO/'experiments/local/prog_split_20261002/local_delivery_status.json',audit)
    DOC.mkdir(exist_ok=True)
    for f in (W/'analysis').iterdir():
        if f.suffix=='.csv': (DOC/f.name).write_text(f.read_text())
        else: shutil.copy2(f,DOC/f.name)
    write_report()
    save(DOC/'delivery_status.json',audit)
    save(DOC/'split_inventory.json',read(W/'split_inventory.json'))
    save(DOC/'checkpoint_records.json',[{k:j[k] for k in ['dataset','checkpoint_sha256','family','pretrain_seed']} for j in plan['jobs']])
    save(P/'data/main.json',[r for r in table if r['dataset'] in D and r['method']!='NoRouting'])
    save(P/'data/ablation.json',[r for r in table if r['dataset'] in D and r['method']!='FullFineTune'])
    save(P/'data/paired.json',[r for r in paired if r['dataset'] in D])
    save(P/'data/supplementary.json',[r for r in table if r['dataset'] not in D])
    save(P/'data/supplementary_paired.json',[r for r in paired if r['dataset'] not in D])
    inventory=read(W/'split_inventory.json')
    save(P/'data/split_inventory.json',inventory)
    save(P/'data/datasets.json',{j['dataset']:dict(nodes=j['nodes'],input_dim=j['input_dim'],classes=j['classes'],split_policy=plan['schema']) for j in plan['jobs']})
    save(P/'data/numerical_audit.json',audit)
    save(P/'data/provenance.json',dict(schema='original-ProG-eight-dataset-rerun',source_commit='844d0a5443272bf50fdb8591386d040022be7022',
        protocol=plan['protocol'],upstream_commit=plan['upstream_commit'],main_datasets=D,
        supplementary_datasets=[j['dataset'] for j in plan['jobs'] if j['dataset'] not in D],
        units=5760,split_tasks=480,methods=plan['methods'],shots=plan['shots'],task_seeds=plan['split_seeds'],train_seeds=plan['train_seeds'],
        original_checkpoint_reused=True,new_pretraining=False,checkpoint_records=read(DOC/'checkpoint_records.json'),
        old_results_policy='Previous official-mask data remain in Git and original experiment directories; no pooling',
        numeric_revision='Only filter row-sum diagnostic assertion changed to finite error recording; operator arithmetic unchanged',
        reporting_scope='Eight datasets specified before rerun; main three retained by user request; descriptive evaluation on previously studied graphs',
        evaluation_limit='Support/test overlap retained; not strict held-out generalization; external paper split identities unavailable'))
    # Paper package integrity metadata is generated once; no experiment-file hash sweep.
    names=[f for f in (P/'data').glob('*.json') if f.name!='source_manifest.json']
    save(P/'data/source_manifest.json',{f.name:dict(sha256=hashlib.sha256(f.read_bytes()).hexdigest(),source='original-ProG completed rerun') for f in names})
    print(json.dumps(audit))


if __name__=='__main__':main()
