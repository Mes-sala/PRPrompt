import json,time,collections,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def dump(p,x):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(p)
def summarize():
 plan=json.loads((ROOT/'plan.json').read_text());jobs=[];rows=[]
 for job in plan['jobs']:
  d=ROOT/'results'/job['id'];rr=[json.loads(f.read_text()) for f in d.glob('*/result.json')];rows+=rr
  status='complete' if (d/'complete.json').exists() else ('failed' if (ROOT/'failures'/f"{job['id']}.json").exists() else ('running_or_partial' if (d/'current.json').exists() else 'pending'))
  jobs.append(dict(id=job['id'],status=status,rows=len(rr)))
 counts=dict(collections.Counter(j['status'] for j in jobs));s=dict(updated=time.time(),completed_rows=len(rows),expected_rows=plan['expected_results'],total_jobs=len(jobs),jobs=counts,complete=len(rows)==plan['expected_results'] and counts.get('complete',0)==len(jobs),job_status=jobs)
 dump(ROOT/'summary.json',s);(ROOT/'进展.md').write_text(f"# 三模块完整因子筛查进展\n\n更新时间：{time.strftime('%Y-%m-%d %H:%M:%S %z')}\n\n新增结果 {len(rows)}/{plan['expected_results']}；任务 {counts}。共63个任务、315次新增训练、630条新增读出；历史引用不算新训练。\n\n实验定义与scale2机制纠正见[阶段设计](阶段设计.md)。完成仍需新预测和旧参照引用审计，当前结果齐全标记为{s['complete']}。\n");return s
if __name__=='__main__':print(summarize())
