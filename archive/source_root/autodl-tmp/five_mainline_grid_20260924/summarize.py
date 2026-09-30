import json,math,time,statistics,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def dump(p,x):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(p)
def summarize():
 plan=json.loads((ROOT/'plan.json').read_text());rows=[];jobs=[]
 for job in plan['jobs']:
  d=ROOT/'results'/job['id'];rr=[json.loads(p.read_text()) for p in d.glob('*/result.json')];rows+=rr
  status='complete' if (d/'complete.json').exists() else ('failed' if (ROOT/'failures'/f"{job['id']}.json").exists() else ('running_or_partial' if rr or (d/'current.json').exists() else 'pending'))
  jobs.append(dict(id=job['id'],status=status,rows=len(rr)))
 groups=collections.defaultdict(list);direct={}
 for r in rows:
  groups[(r['dataset'],r['checkpoint'],r['checkpoint_sha256'],r['method'])].append(r)
  if r['checkpoint']=='Direct':direct[(r['dataset'],r['seed'],r['method'])]=r
 grouped=[]
 for (ds,ck,sha,method),rr in sorted(groups.items(),key=lambda x:str(x[0])):
  vals=[r['accuracy']*100 for r in rr];grouped.append(dict(dataset=ds,checkpoint=ck,checkpoint_sha256=sha,method=method,n=len(rr),mean=statistics.mean(vals),std=statistics.stdev(vals) if len(vals)>1 else None))
 pairs=[];comparisons=[('A1_scale2_qeps','A1_token_qeps'),('A1_scale2_qeps','base_qeps'),('A3_tr1_qeps','frozen'),('A3_gsh_qeps','A3_tr1_qeps'),('A4_vat03','A4_none'),('A5_self100','frozen'),('A5_selfonly','frozen'),('A5_center','A5_selfonly'),('A5_diag10','A5_selfonly'),('A2_center','rect_uncentered')]
 indexed={(r['dataset'],r['checkpoint_sha256'],r['seed'],r['method']):r for r in rows}
 for ds,ck,sha in sorted({(r['dataset'],r['checkpoint'],r['checkpoint_sha256']) for r in rows},key=str):
  for method,base in comparisons:
   diff=[]
   for seed in plan['seeds']:
    a=indexed.get((ds,sha,seed,method));b=indexed.get((ds,sha,seed,base))
    if a and b:
     assert a['split_sha256']==b['split_sha256'];diff.append(dict(seed=seed,delta_pp=100*(a['accuracy']-b['accuracy'])))
   if diff:pairs.append(dict(dataset=ds,checkpoint=ck,checkpoint_sha256=sha,method=method,baseline=base,n=len(diff),delta_pp=statistics.mean(x['delta_pp'] for x in diff),per_seed=diff))
  if ck!='Direct':
   dif=[]
   for seed in plan['seeds']:
    a=direct.get((ds,seed,'A2_center'));b=indexed.get((ds,sha,seed,'frozen'))
    if a and b:assert a['split_sha256']==b['split_sha256'];dif.append(dict(seed=seed,delta_pp=100*(a['accuracy']-b['accuracy'])))
   if dif:pairs.append(dict(dataset=ds,checkpoint=ck,checkpoint_sha256=sha,method='A2_center_shared',baseline='frozen',n=len(dif),delta_pp=statistics.mean(x['delta_pp'] for x in dif),per_seed=dif))
 counts=dict(collections.Counter(j['status'] for j in jobs));summary=dict(updated=time.time(),expected_rows=plan['expected_results'],completed_rows=len(rows),jobs=counts,total_jobs=len(jobs),checkpoint_coverage=dict(collections.Counter(r['status'] for r in plan['checkpoints'])),complete=len(rows)==plan['expected_results'] and counts.get('complete',0)==len(jobs),groups=grouped,paired=pairs,job_status=jobs)
 dump(ROOT/'summary.json',summary)
 text=f"# 五主线网格进展\n\n更新时间：{time.strftime('%Y-%m-%d %H:%M:%S %z')}\n\n正式结果：{len(rows)} / {plan['expected_results']}；任务：{counts}，总数{len(jobs)}。\n\n覆盖状态：{summary['checkpoint_coverage']}。主线02共享直接结果不重复计为独立实验。完整方案和限制见方案与协议.md。\n\n| 数据集 | checkpoint | SHA前缀 | 方法 | 对照 | seeds | 配对提升pp |\n|---|---|---|---|---|---:|---:|\n"
 for p in pairs:text+=f"| {p['dataset']} | {p['checkpoint']} | {(p['checkpoint_sha256'] or 'direct')[:12]} | {p['method']} | {p['baseline']} | {p['n']} | {p['delta_pp']:+.3f} |\n"
 (ROOT/'进展.md').write_text(text);return summary
if __name__=='__main__':
 s=summarize();print(json.dumps({k:s[k] for k in ['completed_rows','expected_rows','jobs','total_jobs','checkpoint_coverage','complete']}))
