from pathlib import Path
import json,statistics,random,hashlib
R=Path(__file__).resolve().parent;p=json.loads((R/'plan.json').read_text())
for name in ['final_audit.json','pair_audit.json']:assert json.loads((R/name).read_text())['passed']
rows={}
for f in (R/'results').glob('*/*/result.json'):
 r=json.loads(f.read_text());rows[r['dataset'],r['checkpoint_sha256'],r['seed'],r['method']]=r
assert len(rows)==p['expected_results']
# One fixed seed-index resampling matrix reused for every paired contrast.
gen=random.Random(20260924);n=len(p['seeds']);resamples=[[gen.randrange(n) for _ in range(n)] for _ in range(10000)]
def stat(v):
 bs=sorted(sum(v[i] for i in ix)/n for ix in resamples)
 return dict(mean_pp=statistics.mean(v),std_pp=statistics.stdev(v),ci95=[bs[249],bs[9749]],positive=sum(x>1e-5 for x in v),tie=sum(abs(x)<=1e-5 for x in v),negative=sum(x< -1e-5 for x in v),per_seed=v)
cells=[]
for job in p['jobs']:
 if job['direct']:continue
 ds=job['dataset'];sha=job['checkpoint']['sha256'];acc={m:[] for m in p['methods']+p['direct_methods']}
 for seed in p['seeds']:
  rr=[rows[ds,None if m in p['direct_methods'] else sha,seed,m] for m in acc];assert len({r['split_sha256'] for r in rr})==1
  for m,r in zip(acc,rr):acc[m].append(100*r['accuracy'])
 comparisons={'remove_M1':'F011','remove_M2':'F101','remove_M3':'F110','full_vs_base':'F000'}
 for m in ['M1_normmatched','M1_amplitude_only','M2_diagonal','M2_ungrafted','frozen','graphprompt']+p['direct_methods']:comparisons['full_minus_'+m]=m
 contrasts={label:stat([a-b for a,b in zip(acc['F111'],acc[m])]) for label,m in comparisons.items()}
 weights={'interaction12_M3off':{'F110':1,'F100':-1,'F010':-1,'F000':1},'interaction12_M3on':{'F111':1,'F101':-1,'F011':-1,'F001':1},'interaction13_M2off':{'F101':1,'F100':-1,'F001':-1,'F000':1},'interaction13_M2on':{'F111':1,'F110':-1,'F011':-1,'F010':1},'interaction23_M1off':{'F011':1,'F010':-1,'F001':-1,'F000':1},'interaction23_M1on':{'F111':1,'F110':-1,'F101':-1,'F100':1},'interaction123':{'F111':1,'F110':-1,'F101':-1,'F011':-1,'F100':1,'F010':1,'F001':1,'F000':-1}}
 for label,terms in weights.items():contrasts[label]=stat([sum(acc[m][i]*w for m,w in terms.items()) for i in range(n)])
 cells.append(dict(dataset=ds,checkpoint=job['checkpoint']['family'],sha256=sha,accuracy={m:dict(mean=statistics.mean(v),std=statistics.stdev(v),per_seed=v) for m,v in acc.items()},contrasts=contrasts,three_positive_means=all(contrasts['remove_M'+str(i)]['mean_pp']>1e-5 for i in [1,2,3]),three_positive_ci_lower=all(contrasts['remove_M'+str(i)]['ci95'][0]>0 for i in [1,2,3])))
out=dict(plan_sha256=hashlib.sha256((R/'plan.json').read_bytes()).hexdigest(),bootstrap_seed=20260924,bootstrap_replicates=10000,ci_scope='paired support-sampling conditional on fixed graph and query set; no multiplicity adjustment',cells=cells);(R/'confirmation_pairs.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
t='# 三模块固定范围确认结果\n\n20个预先固定新支持seed；每份权重8格因子及机制对照。Bootstrap区间描述同图支持采样波动，不是独立图泛化证据，未作多重比较校正。不根据结果换seed/权重。\n'
for c in cells:
 t+=f"\n## {c['dataset']} / {c['checkpoint']} / {c['sha256'][:12]}\n\n三个移除差均值全正：{c['three_positive_means']}；三个区间下界全正：{c['three_positive_ci_lower']}。二者不是同一个主张。\n\n| 方法 | 准确率均值±标准差% |\n|---|---:|\n"
 for m,r in c['accuracy'].items():t+=f"| {m} | {r['mean']:.3f} ± {r['std']:.3f} |\n"
 t+='\n| 比较 | 配对增量pp | 条件bootstrap95%区间 | 正/平/负seed |\n|---|---:|---|---|\n'
 for m,r in c['contrasts'].items():t+=f"| {m} | {r['mean_pp']:+.3f} | [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['positive']}/{r['tie']}/{r['negative']} |\n"
t+='\n本报告保留所有负结果。若三模块并未共同成立，研究目标仍未完成；应保留固定确认结果，基于新的独立假设设计下一轮，不能事后挑seed变成成功。\n';(R/'确认结果.md').write_text(t);print([(c['dataset'],c['three_positive_means'],{k:round(c['contrasts'][k]['mean_pp'],3) for k in ['remove_M1','remove_M2','remove_M3']}) for c in cells])
