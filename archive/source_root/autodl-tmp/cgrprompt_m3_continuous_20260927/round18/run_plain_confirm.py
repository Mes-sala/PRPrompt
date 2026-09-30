from pathlib import Path
import run_structure as m
ROOT=Path(__file__).resolve().parent
jobs=[j for j in m.json.loads((m.OLD/'plan.json').read_text())['jobs'] if j['phase']=='main' and j['dataset']=='CiteSeer' and j['checkpoint']['family'] in ['Edgepred_Gprompt','Edgepred_GPPT']]
assert len(jobs)==2
stage='plain_confirm'
assert not (ROOT/(stage+'_complete.json')).exists(), 'already completed'
m.save(ROOT/(stage+'_plan.json'),dict(jobs=jobs,variants=['ens50correct'],candidate_key='Prompt__ens50correct_plain',runner_sha256=m.sha(ROOT/'run_structure.py'),wrapper_sha256=m.sha(__file__),core_runner_sha256=m.sha(m.OLD/'run_core.py'),design_sha256=m.sha(ROOT/'设计与判定.md'),conditional_design_sha256=m.sha(ROOT/'纯集成确认设计.md')))
try:
 for j in jobs:m.run(j,stage,['ens50correct'])
 m.save(ROOT/(stage+'_complete.json'),dict(finished=m.time.time(),jobs=len(jobs)))
except Exception as ex:m.save(ROOT/(stage+'_failure.json'),dict(error=repr(ex),traceback=m.traceback.format_exc()));raise
