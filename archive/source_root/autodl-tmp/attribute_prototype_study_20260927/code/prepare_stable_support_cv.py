"""Freeze the pre-existing Amazon candidate union; query scores never read."""
from pathlib import Path
import json,hashlib,argparse

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(root):
 p=Path(root);dataset='Amazon-ratings';coverage=json.loads((p/'results/expanded_grid_dispatch/coverage_verified.json').read_text());units=[u for u in coverage['units'] if u['dataset']==dataset];old=json.loads((p/'design/expanded_confirmation/manifest.json').read_text());groups={g['route']:{(c['unit'],c['config']) for c in g['candidates']} for g in old['groups'] if g['dataset']==dataset};folder=p/'design/stable_support_cv_Amazon';folder.mkdir(exist_ok=True);plans=[]
 for u in units:
  uid=u['id'];parent=p/'design/default_anchor_confirmation'/('default_anchor_'+uid+'.json');base=json.loads(parent.read_text());cfgs=set(base['config_ids']);route='attributes' if u['kind']=='attributes' else u['architecture'];groups[route].update((uid,c) for c in cfgs)
  other=p/'design/expanded_confirmation'/('expanded_confirmation_'+uid+'.json')
  if other.exists():cfgs.update(json.loads(other.read_text())['config_ids'])
  q=dict(round='stable_support_cv_Amazon',parent_plan=str(parent),parent_plan_sha256=sha(parent),runner_sha256=sha(p/'code/run_stable_support_cv.py'),config_ids=sorted(cfgs),seeds=list(range(271400,271420)),selection_uses_query_labels=False)
  f=folder/(uid+'.json')
  if f.exists():assert json.loads(f.read_text())==q
  else:f.write_text(json.dumps(q,indent=2))
  plans.append(dict(unit=uid,plan=f.name,rows=len(cfgs)*20))
 m=dict(dataset=dataset,plans=plans,rows=sum(x['rows'] for x in plans),groups={r:[dict(unit=u,config=c) for u,c in sorted(items)] for r,items in groups.items()},scope='Numerical correction on original support draws, not independent confirmation',selection_uses_query_labels=False)
 (folder/'manifest.json').write_text(json.dumps(m,indent=2));print({'units':len(plans),'rows':m['rows'],'selection_candidates':{k:len(v) for k,v in groups.items()}})
 # Four operator smoke cases, separate output scope; same prespecified seed.
 attr=json.loads((folder/'Amazon-ratings_attributes.json').read_text());bank=json.loads((p/'design/expanded_grid_candidate_bank.json').read_text())['attributes'];smoke=[]
 for mod,rounds in [('none',0),('none',3),('confidence',3),('neighbor',3)]:
  candidates=[c for c in bank if c.get('hops')==2 and c['geometry']=='raw' and c['rounds']==rounds and c.get('module','none')==mod and (rounds==0 or (c['beta']==.5 and c['tau']==.05))];assert candidates;smoke.append(candidates[0]['id'])
 attr.update(round='smoke_stable_support_cv',seeds=[271400],config_ids=smoke);(folder/'smoke.json').write_text(json.dumps(attr,indent=2));print(smoke)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('root');main(ap.parse_args().root)
