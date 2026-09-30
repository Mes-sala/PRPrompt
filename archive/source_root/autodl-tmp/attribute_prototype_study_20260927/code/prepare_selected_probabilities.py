"""Freeze exact replays of support-selected confirmation predictions, without reselection."""
from pathlib import Path
from collections import defaultdict
import argparse,json,csv,hashlib

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(project):
 root=Path(project);source=root/'reports/expanded_confirmation/chosen_configurations.csv';rows=list(csv.DictReader(source.open()));assert len(rows)==14*20*6
 grouped=defaultdict(dict)
 for row in rows:
  key=(int(row['seed']),row['config']);job=grouped[row['unit']].setdefault(key,dict(seed=key[0],config=key[1],routes=[]));job['routes'].append(row['route'])
 folder=root/'design/selected_probability_replay';folder.mkdir(parents=True,exist_ok=True);plans=[]
 for uid,jobs in sorted(grouped.items()):
  parent=root/'design/expanded_confirmation'/('expanded_confirmation_'+uid+'.json');assert parent.exists()
  plan=dict(round='selected_probability_replay',parent_plan=str(parent),parent_plan_sha256=sha(parent),selection_csv_sha256=sha(source),replay_sha256=sha(root/'code/replay_selected_probabilities.py'),jobs=list(jobs.values()))
  target=folder/(uid+'.json')
  if target.exists():assert json.loads(target.read_text())==plan
  else:target.write_text(json.dumps(plan,indent=2))
  plans.append(dict(unit=uid,file=target.name,jobs=len(jobs)))
 manifest=dict(selection_rows=len(rows),unique_predictions=sum(x['jobs'] for x in plans),selection_uses_query_labels=False,source_csv_sha256=sha(source),plans=plans);dest=folder/'manifest.json'
 if dest.exists():assert json.loads(dest.read_text())==manifest
 else:dest.write_text(json.dumps(manifest,indent=2))
 print({'selection_rows':len(rows),'unique_predictions':manifest['unique_predictions']})
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('project');main(ap.parse_args().project)
