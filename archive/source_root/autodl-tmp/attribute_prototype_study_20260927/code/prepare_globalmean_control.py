from pathlib import Path
import json,hashlib,argparse

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(root):
 p=Path(root);folder=p/'design/globalmean_control';folder.mkdir(exist_ok=True);datasets=['Cora','CiteSeer','PubMed','Computers','Photo','Actor','Chameleon','Cornell','Squirrel','WikiCS','CS','Physics','Roman-empire','Amazon-ratings'];plans=[]
 for ds in datasets:
  q=dict(round='globalmean_control_r14',dataset=ds,seeds=list(range(271700,271720)),runner_sha256=sha(p/'code/run_globalmean_control.py'),loader_sha256=sha(p/'code/expanded_data.py'),operator_sha256=sha(p/'code/run_study.py'))
  f=folder/(ds+'.json')
  if f.exists():assert json.loads(f.read_text())==q
  else:f.write_text(json.dumps(q,indent=2))
  plans.append(dict(dataset=ds,plan=f.name,rows=40))
 (folder/'manifest.json').write_text(json.dumps(dict(plans=plans,rows=560,scope='fixed attributes;mechanism diagnostics,no pretrained encoder'),indent=2))
 q=json.loads((folder/'Cornell.json').read_text());q.update(round='smoke_globalmean_control',seeds=[271700]);(folder/'smoke.json').write_text(json.dumps(q,indent=2));print('14 datasets;560 rows;2-row Cornell smoke')
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('root');main(ap.parse_args().root)
