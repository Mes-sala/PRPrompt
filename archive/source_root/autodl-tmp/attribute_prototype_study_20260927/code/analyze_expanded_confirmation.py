"""Analyze frozen support-selected routes and paired module ablations on new draws."""
from pathlib import Path
import json,sqlite3,hashlib,csv,argparse
import numpy as np

def write_csv(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def pick(candidates,data,seed):
    return min((data[x['unit'],x['config'],seed] for x in candidates),key=lambda x:(-x['support_accuracy'],x['support_nll'],x['unit'],x['config']))

def main(project):
    p=Path(project);manifest=json.loads((p/'design/expanded_confirmation/manifest.json').read_text());assert manifest['selection_uses_query_labels'] is False
    seeds=manifest['seeds'];assert seeds==list(range(271400,271420));data={};groups={(g['dataset'],g['route']):g['candidates'] for g in manifest['groups']}
    for item in manifest['plans']:
        uid=item['unit'];folder=p/'results'/('expanded_confirmation_'+uid);assert (folder/'complete.json').exists()
        marker=json.loads((folder/(uid+'.complete.json')).read_text());db=folder/(uid+'.sqlite');assert hashlib.sha256(db.read_bytes()).hexdigest()==marker['sha256']
        con=sqlite3.connect(f'file:{db}?mode=ro',uri=True);splits=dict(con.execute('SELECT seed,split_hash FROM splits'));rows=con.execute('SELECT seed,config,validation,accuracy FROM results').fetchall();assert len(rows)==item['rows']
        for seed,cfg,val,acc in rows:
            v=json.loads(val);data[uid,cfg,seed]=dict(unit=uid,config=cfg,accuracy=acc,support_accuracy=v['accuracy'],support_nll=v['nll'],split=splits[seed])
        con.close()
    rng=np.random.default_rng(271400)
    def compare(a,b,**kw):
        assert len(a)==len(b)==20 and all(x['split']==y['split'] for x,y in zip(a,b))
        d=np.array([x['accuracy']-y['accuracy'] for x,y in zip(a,b)]);lo,hi=np.quantile(d[rng.integers(0,20,(10000,20))].mean(1),[.025,.975])
        return dict(**kw,accuracy=float(np.mean([x['accuracy'] for x in a])),reference_accuracy=float(np.mean([x['accuracy'] for x in b])),delta_pp=float(d.mean()),ci_low=float(lo),ci_high=float(hi),positive_draws=int((d>0).sum()),negative_draws=int((d<0).sum()))
    results=[];chosen=[]
    for dataset in sorted({k[0] for k in groups}):
        attr=[pick(groups[dataset,'attributes'],data,s) for s in seeds]
        eligible={route:groups[dataset,route] for route in ('attributes','GCN','GraphSAGE','GAT')}
        eligible['any_frozen']=[x for r in ('GCN','GraphSAGE','GAT') for x in eligible[r]]
        eligible['all_including_attributes']=eligible['attributes']+eligible['any_frozen']
        for route,candidates in eligible.items():
            selected=[pick(candidates,data,s) for s in seeds]
            results.append(compare(selected,attr,dataset=dataset,route=route,selection_uses_query_labels=False))
            chosen.extend(dict(dataset=dataset,route=route,seed=s,unit=x['unit'],config=x['config'],support_accuracy=x['support_accuracy'],support_nll=x['support_nll'],query_accuracy=x['accuracy']) for s,x in zip(seeds,selected))
    ablations=[]
    for item in manifest['matched_controls']:
        uid=item['unit'];candidate=item['candidate'];a=[data[uid,candidate,s] for s in seeds]
        for reference,role in item['controls'].items():
            if reference==candidate:continue
            b=[data[uid,reference,s] for s in seeds]
            ablations.append(compare(a,b,dataset=item['dataset'],route=item['route'],unit=uid,candidate=candidate,reference=reference,control_role=role))
    dest=p/'reports/expanded_confirmation';dest.mkdir(parents=True,exist_ok=True);write_csv(dest/'route_accuracy.csv',results);write_csv(dest/'chosen_configurations.csv',chosen)
    if ablations:write_csv(dest/'matched_ablations.csv',ablations)
    lines=['# 各自优化后的表示路线比较','','14数据集，5-shot，20次新支持抽样。每条路线先按开发支持验证筛选20项，再在每次新抽样内以支持5折选择。查询标签不参与候选和配置选择。区间为同图支持抽样的配对bootstrap，未作多重校正。','', '|数据集|属性Acc|GCN Acc|差 pp|95%区间|','|---|---:|---:|---:|---|']
    for x in results:
        if x['route']=='GCN':lines.append(f"|{x['dataset']}|{x['reference_accuracy']:.3f}|{x['accuracy']:.3f}|{x['delta_pp']:+.3f}|[{x['ci_low']:.3f}, {x['ci_high']:.3f}]|")
    lines+=['','GraphSAGE/GAT以及跨路线支持选择见route_accuracy.csv；逐配置匹配消融见matched_ablations.csv。这里是预定候选预算内的优化表现，不是数学意义的全局最优。新支持抽样仍不是独立图泛化。']
    (dest/'各自优化对比.md').write_text('\n'.join(lines));print(dict(route_comparisons=len(results),paired_ablations=len(ablations),verified_rows=len(data)))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('project');main(ap.parse_args().project)
