"""Freeze replay jobs for support-selected union of defaults and optimized candidates."""
from pathlib import Path
from collections import defaultdict
import json,csv,hashlib,argparse

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(root):
 p=Path(root);csvpath=p/'reports/default_anchor/chosen_with_default_fallback.csv';rows=list(csv.DictReader(csvpath.open()));assert len(rows)==1680
 assert len({(x['dataset'],x['route'],x['seed']) for x in rows})==1680
 jobs=defaultdict(dict)
 for x in rows:
  assert x['source'] in ['baseline','optimized'];key=(int(x['seed']),x['config']);item=jobs[x['source'],x['unit']].setdefault(key,dict(seed=key[0],config=key[1],routes=[]));item['routes'].append(x['route'])
 folder=p/'design/numerical_probability_audit';folder.mkdir(parents=True,exist_ok=True);plans=[]
 for (source,uid),jj in sorted(jobs.items()):
  parent=p/'design'/('default_anchor_confirmation' if source=='baseline' else 'expanded_confirmation')/(('default_anchor_' if source=='baseline' else 'expanded_confirmation_')+uid+'.json');assert parent.exists()
  q=dict(round='numerical_probability_'+source,parent_plan=str(parent),parent_plan_sha256=sha(parent),replay_sha256=sha(p/'code/replay_numerical_audit.py'),selection_csv_sha256=sha(csvpath),jobs=list(jj.values()))
  name=source+'__'+uid+'.json';f=folder/name
  if f.exists():assert json.loads(f.read_text())==q
  else:f.write_text(json.dumps(q,indent=2))
  plans.append(dict(source=source,unit=uid,plan=name,predictions=len(jj)))
 manifest=dict(selection_rows=1680,unique_predictions=sum(x['predictions'] for x in plans),source_csv_sha256=sha(csvpath),selection_uses_query_labels=False,plans=plans);f=folder/'manifest.json'
 if f.exists():assert json.loads(f.read_text())==manifest
 else:f.write_text(json.dumps(manifest,indent=2))
 print({k:v for k,v in manifest.items() if k!='plans'})
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('root');main(ap.parse_args().root)
