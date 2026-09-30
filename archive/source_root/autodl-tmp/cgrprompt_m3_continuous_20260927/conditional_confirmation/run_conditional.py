from pathlib import Path
import sys,importlib.util,json,hashlib,time
ROOT=Path(__file__).resolve().parent;PARENT=ROOT.parent
for round_name,dataset,variant in [('round07','Cora','oof50l01'),('round09','CiteSeer','proto_decor')]:
 path=PARENT/round_name/'run_structure.py';spec=importlib.util.spec_from_file_location('candidate_'+round_name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 sourceplan=json.loads((PARENT/round_name/'screen_plan.json').read_text());assert hashlib.sha256(path.read_bytes()).hexdigest()==sourceplan['runner_sha256'];jobs=[j for j in sourceplan['jobs'] if j['dataset']==dataset];assert len(jobs)==2
 plan=dict(jobs=jobs,variants=[variant],conditional_scope=dataset,runner_sha256=sourceplan['runner_sha256'],core_runner_sha256=sourceplan['core_runner_sha256'],design_sha256=sourceplan['design_sha256'],conditional_design_sha256=hashlib.sha256((ROOT/'设计与判定.md').read_bytes()).hexdigest())
 (PARENT/round_name/'conditional_confirm_plan.json').write_text(json.dumps(plan,indent=2))
 for job in jobs:module.run(job,'conditional_confirm',[variant])
 (PARENT/round_name/'conditional_confirm_complete.json').write_text(json.dumps(dict(jobs=2,episodes=40,finished=time.time())))
(ROOT/'complete.json').write_text(json.dumps(dict(episodes=80,prompt_fits=280,finished=time.time())))
