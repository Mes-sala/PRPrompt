"""No torch dependency: independently verify predictions and condition-wise summaries."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from tools.metric_utils import numpy_metrics, sha

METHODS=['Frozen','FullFineTune','GPF','GPFPlus','GraphPromptDiag','PRPrompt','SharedPrompt']
METRICS=['accuracy','macro_f1','micro_f1','roc_auc_macro','average_precision_macro']

def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')

def summarize(w,local=False):
    plan=json.loads((w/'plan.json').read_text());rows=[];blocks={};signatures=set();split_ids={};max_error=0.
    with np.load(w/'audit_reference.npz') as ref:labels=ref['labels'];query=ref['query'];pool=ref['pool']
    expected=set(itertools.product(plan['shots'],plan['split_seeds'],plan['train_seeds'],METHODS))
    seen=set();diagnostic_changes=0
    for f in sorted((w/'results').glob('*/*/*/*/metrics.json')):
        r=json.loads(f.read_text());folder=f.parent
        k=(r['shots'],r['split_seed'],r['train_seed'],r['method']);assert k in expected and k not in seen
        seen.add(k);assert not r['smoke'] and r['plan_sha256']==plan['plan_sha256']
        assert r['signature'] not in signatures;signatures.add(r['signature'])
        for fn,h in r['files'].items():
            if local and fn=='state.pt':assert not (folder/fn).exists();continue
            assert sha(folder/fn)==h,(f,fn)
        seal=json.loads((folder/'seal.json').read_text());assert seal['predictions_sha256']==r['files']['predictions.npz']
        assert seal['signature']==r['signature']
        with np.load(folder/'predictions.npz') as p:
            assert np.array_equal(query,p['query']) and np.isin(p['support'],pool).all()
            assert not np.isin(p['support'],query).any()
            assert np.array_equal(np.bincount(labels[p['support']],minlength=40),np.full(40,r['shots']))
            split=(r['shots'],r['split_seed']);support=p['support'].tobytes()
            if split in split_ids:assert split_ids[split]==support
            split_ids[split]=support
            got=numpy_metrics(labels[query],p['probabilities'])
        error=max(abs(got[m]-r['metrics'][m]) for m in METRICS);assert error<1e-12;max_error=max(max_error,error)
        audit=json.loads((folder/'unit_audit.json').read_text())
        assert audit['serialized_probability_exact'] and audit['saved_readout_class_changes']==0
        diagnostic_changes+=audit['fresh_forward']['query_class_changes']
        blocks.setdefault((r['shots'],r['method']),{}).setdefault(r['split_seed'],[]).append(got)
    assert seen==expected,dict(found=len(seen),expected=len(expected),missing=list(expected-seen)[:10])
    for shot in plan['shots']:
        for method in METHODS:
            group=blocks[shot,method];record=dict(dataset='ogbn-arxiv',shots=shot,method=method,support_draws=20,training_repeats=3)
            for metric in METRICS:
                values=np.array([np.mean([r[metric] for r in group[s]]) for s in plan['split_seeds']])
                assert all(len(group[s])==3 for s in plan['split_seeds'])
                record[metric+'_mean']=float(values.mean());record[metric+'_std']=float(values.std(ddof=1))
            rows.append(record)
    effects=[]
    for shot in plan['shots']:
        for method in METHODS:
            if method=='PRPrompt':continue
            for metric in METRICS:
                delta=np.array([np.mean([r[metric] for r in blocks[shot,'PRPrompt'][s]])-np.mean([r[metric] for r in blocks[shot,method][s]]) for s in plan['split_seeds']])
                rng=np.random.default_rng(plan['bootstrap_seed']);samples=delta[rng.integers(0,20,(10000,20))].mean(1)
                lo,hi=np.quantile(samples,[.025,.975]);effects.append(dict(dataset='ogbn-arxiv',shots=shot,reference=method,metric=metric,
                    mean_delta=float(delta.mean()),ci95_low=float(lo),ci95_high=float(hi)))
    if local:
        assert rows==json.loads((w/'analysis/summary.json').read_text())
        assert effects==json.loads((w/'analysis/paired.json').read_text())
        write(w/'local_delivery_status.json',dict(status='complete',units=len(seen),maximum_metric_error=max_error,
            source_and_prediction_hashes_verified=True,summary_and_paired_recomputed=True,local_checkpoint_files=0,
            fresh_forward_query_changes=diagnostic_changes,plan_sha256=plan['plan_sha256']))
        return
    analysis=w/'analysis';write(analysis/'summary.json',rows);write(analysis/'paired.json',effects)
    with (analysis/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (analysis/'paired.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(effects[0]));writer.writeheader();writer.writerows(effects)
    write(analysis/'coverage.json',dict(expected=1260,complete=len(seen),maximum_metric_error=max_error,
        prediction_state_selection_hashes=True,support_query_disjoint=True,paired_support_identical=True,
        fresh_forward_query_changes=diagnostic_changes))
    lines=['# PRPrompt：ogbn-arxiv 实验结果','',
        '独立新增数据集；官方时间划分、同图传导式少样本设置。固定新建 EdgePred 风格全量边预训练；并非 OGB 全监督排行榜成绩。',
        '每个 shot/方法/权重条件内，先平均三个训练 seed，再汇总 20 个支持抽样，单位为百分比。无跨条件平均。','']
    for shot in plan['shots']:
        lines += [f'## {shot}-shot','', '| 方法 | Accuracy | Macro-F1 | Micro-F1 | Macro ROC-AUC | Macro AP |','|---|---:|---:|---:|---:|---:|']
        for r in rows:
            if r['shots']==shot:lines.append('| '+r['method']+' | '+' | '.join(f"{100*r[m+'_mean']:.2f} ± {100*r[m+'_std']:.2f}" for m in METRICS)+' |')
        lines+=['','PRPrompt 相对对照的配对差值与未校正 95% 支持 bootstrap 区间（百分点）：','',
            '| 对照 | Accuracy | Macro-F1 |','|---|---:|---:|']
        for method in METHODS:
            if method=='PRPrompt':continue
            cells=[]
            for metric in METRICS[:2]:
                e=next(e for e in effects if (e['shots'],e['reference'],e['metric'])==(shot,method,metric))
                cells.append(f"{100*e['mean_delta']:+.2f} [{100*e['ci95_low']:+.2f}, {100*e['ci95_high']:+.2f}]")
            lines.append('| '+method+' | '+' | '.join(cells)+' |')
        lines+=['']
    lines+=['## 核验与边界','',f'完整覆盖 {len(seen)}/1260；独立 NumPy 五指标最大误差 {max_error:.3g}。新 CUDA 前向在查询集累计 {diagnostic_changes} 个类别变化，属于单独数值诊断；正式结果来自封存预测。',
        '预训练 seed=42，固定 1000 步最终权重；未使用节点标签选择。所有下游方法共享该权重。1-shot PRPrompt 使用包含自身的原型训练损失。',
        '对照方法均为项目统一协议实现；不等同于各论文原始配置。结果不与历史三个数据集求平均。完整五指标配对见 paired.csv。']
    (analysis/'结果总结.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('workspace',type=Path);p.add_argument('--local',action='store_true')
    a=p.parse_args();summarize(a.workspace,a.local)
