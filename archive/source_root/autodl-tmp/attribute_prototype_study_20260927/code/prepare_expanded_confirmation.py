"""Freeze route-stratified support-only candidates and matched ablation closure.
Run after analyze_expanded_grid.py on a fully sealed broad grid.
This prepares plans; it does not launch experiments or select by query Accuracy.
"""
from pathlib import Path
from collections import defaultdict
import argparse,csv,json,hashlib

ROUTES=('attributes','GCN','GraphSAGE','GAT')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def choose(rows,n=20):
    groups=defaultdict(list)
    for x in rows:
        route='attributes' if x['family']=='attributes' else x['architecture']
        assert route in ROUTES
        groups[x['dataset'],route].append(x)
    return {k:sorted(v,key=lambda x:(-float(x['support_accuracy']),float(x['support_nll']),x['unit'],x['config']))[:n] for k,v in sorted(groups.items())}

def closure(cfg,bank):
    samegeo=lambda c:c.get('hops')==cfg.get('hops') and c['geometry']==cfg['geometry']
    no=[c for c in bank if samegeo(c) and c['rounds']==0]
    assert len(no)==1
    out={cfg['id']: 'candidate',no[0]['id']:'no_refinement'}
    if cfg['rounds']:
        ref=[c for c in bank if samegeo(c) and all(c[k]==cfg[k] for k in ('beta','tau','rounds')) and c['module']=='none']
        assert len(ref)==1
        out[ref[0]['id']]='no_extra_module'
    return out

def main(root):
    p=Path(root);inp=p/'reports/expanded_grid_analysis/all_configurations.csv';rows=list(csv.DictReader(inp.open()))
    coverage=json.loads((p/'results/expanded_grid_dispatch/coverage_verified.json').read_text());units={u['id']:u for u in coverage['units']}
    assert {x['unit'] for x in rows}==set(units)
    bankpath=p/'design/expanded_grid_candidate_bank.json';bank=json.loads(bankpath.read_text());configs={k:{c['id']:c for c in bank[k]} for k in ('attributes','frozen')}
    assert all(int(x['n'])==len(bank['screen_seeds']) for x in rows)
    assert len(rows)==sum(len(bank[u['kind']]) for u in units.values())
    selectable=[x for x in rows if not configs[units[x['unit']]['kind']][x['config']]['module'].endswith('_shuffle')]
    candidates=choose(selectable);assert len(candidates)==coverage['datasets']*4
    requested=defaultdict(set);groups=[];links=[]
    for (dataset,route),ss in candidates.items():
        groups.append(dict(dataset=dataset,route=route,candidates=[{k:x[k] for k in ('unit','config','support_accuracy','support_nll')} for x in ss]))
        for x in ss:
            unit=units[x['unit']];cfg=configs[unit['kind']][x['config']]
            cc=closure(cfg,bank[unit['kind']]);requested[x['unit']].update(cc)
            links.append(dict(dataset=dataset,route=route,unit=x['unit'],candidate=x['config'],controls=cc))
    dest=p/'design/expanded_confirmation';dest.mkdir(parents=True,exist_ok=True)
    manifest=dict(selection_uses_query_labels=False,top_k_per_route=20,seeds=bank['confirmation_seeds'],validation_folds=5,source_csv_sha256=sha(inp),bank_sha256=sha(bankpath),groups=groups,matched_controls=links)
    # Reuse source/hash contracts of the completed matched comparison, replacing only round/configs/seeds.
    plans=[]
    for uid,ids in sorted(requested.items()):
        old=p/'design/grid_units'/('backbone_value_'+uid+'.json');q=json.loads(old.read_text());assert len(q['units'])==1 and q['units'][0]['id']==uid
        q.update(round='expanded_confirmation_'+uid,config_ids=sorted(ids),seeds=bank['confirmation_seeds'],validation_folds=5)
        out=dest/(q['round']+'.json')
        if out.exists():assert json.loads(out.read_text())==q
        else:out.write_text(json.dumps(q,indent=2))
        plans.append(dict(plan=out.name,unit=uid,configs=len(ids),rows=len(ids)*len(bank['confirmation_seeds'])))
    manifest['plans']=plans;manifest['rows']=sum(x['rows'] for x in plans)
    f=dest/'manifest.json'
    if f.exists():assert json.loads(f.read_text())==manifest
    else:f.write_text(json.dumps(manifest,indent=2))
    print(dict(groups=len(groups),units=len(plans),rows=manifest['rows']))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('project');main(ap.parse_args().project)
