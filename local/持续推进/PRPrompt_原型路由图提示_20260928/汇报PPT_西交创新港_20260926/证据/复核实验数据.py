from pathlib import Path
import json,csv,hashlib,random,statistics
import numpy as np
OUT=Path(__file__).resolve().parent
BASE=OUT.parents[1]
U=BASE.parent/'三底座统一协议对比_20260924'
load=lambda p: json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
raw=load(U/'all_metrics.json')['rows'];pairs=load(U/'all_metric_pairs.json');plan=load(U/'plan.json');ext=load(U/'extension2026/plan.json');state=load(U/'state.json')
methods=plan['methods']+ext['methods'];metrics=['accuracy','macro_f1','roc_auc_macro_ovr','auprc_macro']
assert len(raw)==1680 and len(methods)==14 and len(set(methods))==14
assert state['expected_results']==1680 and state['completed_results']==1680
rng=random.Random(20260924);draws=np.array([[rng.randrange(20) for _ in range(20)] for _ in range(10000)])
idx={(x['dataset'],x['checkpoint'],x['method'],x['metric']):x for x in pairs}; arrays={};means=[];errors=[];split_groups=0
for job in plan['jobs']:
 ds,fam=job['dataset'],job['checkpoint']['family']; splits=[]
 for method in methods:
  rr=sorted([r for r in raw if (r['dataset'],r['checkpoint'],r['method'])==(ds,fam,method)],key=lambda x:x['seed'])
  assert [r['seed'] for r in rr]==plan['seeds']
  assert all(r['checkpoint_sha256']==job['checkpoint']['sha256'] for r in rr)
  splits.append([r['split_sha256'] for r in rr])
  for metric in rr[0]['metrics']:
   v=np.array([r['metrics'][metric]*100 for r in rr]); arrays[ds,fam,method,metric]=v
   means.append(dict(dataset=ds,checkpoint=fam,method=method,metric=metric,mean_pct=float(v.mean()),sd_pct=float(v.std(ddof=1)),n=len(v)))
 assert all(x==splits[0] for x in splits);split_groups+=20
for key,p in idx.items():
 ds,fam,method,metric=key;v=arrays[key];diff=arrays[ds,fam,'F111',metric]-v;bs=np.sort(diff[draws].mean(1))
 vals={'mean':v.mean(),'sd':v.std(ddof=1),'full_minus_method':diff.mean(),'ci_low':bs[249],'ci_high':bs[9749]}
 errors.extend(abs(float(vals[k])-p[k]) for k in vals)
assert max(errors)<1e-10
cells=[]
for j in plan['jobs']:
 ds,fam=j['dataset'],j['checkpoint']['family']
 def stat(m,k):
  p=idx[ds,fam,m,k]
  return {'mean_pct':p['mean'],'sd_pct':p['sd'],'full_minus_method_pp':p['full_minus_method'],'ci95_pp':[p['ci_low'],p['ci_high']],'positive_seeds':p['positive'],'tie_seeds':p['tie'],'negative_seeds':p['negative']}
 externals=[m for m in methods if not m.startswith('F0') and not m.startswith('F1')]
 best=max(externals,key=lambda m:idx[ds,fam,m,'accuracy']['mean'])
 cells.append({'dataset':ds,'checkpoint':fam,'checkpoint_sha256':j['checkpoint']['sha256'],'full':{k:stat('F111',k) for k in metrics},'base':{k:stat('F000',k) for k in metrics},'ablation':{name:{k:stat(m,k) for k in metrics} for name,m in [('M1','F011'),('M2','F101'),('M3','F110')]},'best_external_by_accuracy':{'method':best,**stat(best,'accuracy')},'external_methods':externals})
# Verify early confirmation rows directly, keep their seed blocks separate.
weights={'interaction12_M3on':{'F111':1,'F101':-1,'F011':-1,'F001':1}}
early=[];early_error=0.;early_count=0
for stage in ['stage2_confirmation','stage2b_gppt_confirmation','stage2c_dgi_confirmation']:
 P=BASE/stage; pp=load(P/'plan.json'); dd=load(P/'confirmation_pairs.json')
 rows=[load(f) for f in (P/'results').glob('*/*/result.json')];early_count+=len(rows)
 ix={(r['dataset'],r.get('checkpoint_sha256'),r['seed'],r['method']):r for r in rows}
 for c in dd['cells']:
  ds=c['dataset'];vs={}
  for m,a in c['accuracy'].items():
   rr=[ix[ds,None if m in pp['direct_methods'] else c['sha256'],s,m] for s in pp['seeds']]
   v=np.array([r['accuracy']*100 for r in rr]);vs[m]=v
   early_error=max(early_error,float(np.max(np.abs(v-np.array(a['per_seed'])))))
  # Include all existing contrast records; independently verify all reported per-seed means and CIs.
  contrast_out={}
  for name,v in c['contrasts'].items():
   if name=='interaction12_M3on': z=sum(vs[m]*w for m,w in weights[name].items())
   elif name=='full_vs_base':z=vs['F111']-vs['F000']
   elif name.startswith('remove_M'):z=vs['F111']-vs[{'remove_M1':'F011','remove_M2':'F101','remove_M3':'F110'}[name]]
   elif name.startswith('full_minus_'):z=vs['F111']-vs[name[len('full_minus_'):]]
   else:continue
   bs=np.sort(z[draws].mean(1));early_error=max(early_error,float(np.max(np.abs(z-np.array(v['per_seed'])))),abs(float(z.mean())-v['mean_pp']),abs(float(bs[249])-v['ci95'][0]),abs(float(bs[9749])-v['ci95'][1]))
   contrast_out[name]={k:v[k] for k in ['mean_pp','std_pp','ci95','positive','tie','negative']}
  early.append({'stage':stage,'dataset':ds,'checkpoint':c['checkpoint'],'seeds':pp['seeds'],'accuracy':{m:{k:a[k] for k in ['mean','std']} for m,a in c['accuracy'].items()},'contrasts':contrast_out,'source':str(P/'confirmation_pairs.json')})
assert early_error<1e-10
sources=[U/'all_metrics.json',U/'all_metric_pairs.json',U/'plan.json',U/'extension2026/plan.json',U/'metrics_transport_verified.json',U/'state.json',U/'final_audit.json',U/'extension2026/final_audit.json']+[BASE/s/'confirmation_pairs.json' for s in ['stage2_confirmation','stage2b_gppt_confirmation','stage2c_dgi_confirmation']]
out={'date':'2026-09-26','scope':'latest unified 1680-row protocol; early confirmation kept separate','metric_units':'means and SD in percent; differences and CI in percentage points (pp)','metric_definition':load(U/'all_metrics.json')['definition'],'protocol':{'datasets':2,'checkpoints':6,'methods':methods,'support_seeds':plan['seeds'],'shots_per_class':5,'updates':100,'query_nodes_per_dataset':1000,'formal_results':len(raw),'fits':1560,'trainable_parameters':{'Cora':896,'CiteSeer':768},'bootstrap_replicates':10000,'bootstrap_seed':20260924,'ci_scope':'same graph, fixed query set, paired support resampling; no multiple-testing correction'},'audit':{'passed':True,'raw_unified_rows':len(raw),'metric_pairs_recomputed':len(pairs),'max_abs_error_pct_or_pp':max(errors),'matched_split_groups':split_groups,'early_raw_results_read':early_count,'early_max_abs_error_pct_or_pp':early_error,'core_audit_rows':load(U/'final_audit.json')['count'],'extension_audit_rows':load(U/'extension2026/final_audit.json')['count'],'local_metrics_verified_rows':load(U/'metrics_transport_verified.json')['rows'],'no_GPU_training':True,'note':'means, SDs, deltas and bootstrap intervals recomputed from stored raw result/metric rows; no fresh logits inference'},'main_cells':cells,'early_confirmation':early,'sources':[{'path':str(p),'sha256':sha(p)} for p in sources]}
(OUT/'汇报实验数据_已复核.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
with (OUT/'统一协议全部四指标均值.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(means[0]));w.writeheader();w.writerows(x for x in means if x['metric'] in metrics)
with (OUT/'主结果六格.csv').open('w') as f:
 fields=['dataset','checkpoint','full_accuracy','full_macro_f1','full_auc','full_ap','base_accuracy','overall_delta_pp','overall_ci_low','overall_ci_high','best_external','best_external_accuracy','gap_vs_best_external_pp','M1_delta_pp','M1_ci_low','M1_ci_high','M2_delta_pp','M2_ci_low','M2_ci_high','M3_delta_pp','M3_ci_low','M3_ci_high']
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for c in cells:
  row={'dataset':c['dataset'],'checkpoint':c['checkpoint'],'base_accuracy':c['base']['accuracy']['mean_pct'],'overall_delta_pp':c['base']['accuracy']['full_minus_method_pp'],'overall_ci_low':c['base']['accuracy']['ci95_pp'][0],'overall_ci_high':c['base']['accuracy']['ci95_pp'][1],'best_external':c['best_external_by_accuracy']['method'],'best_external_accuracy':c['best_external_by_accuracy']['mean_pct'],'gap_vs_best_external_pp':c['best_external_by_accuracy']['full_minus_method_pp']}
  for tag,m in [('accuracy','accuracy'),('macro_f1','macro_f1'),('auc','roc_auc_macro_ovr'),('ap','auprc_macro')]:row['full_'+tag]=c['full'][m]['mean_pct']
  for n,a in c['ablation'].items():row.update({n+'_delta_pp':a['accuracy']['full_minus_method_pp'],n+'_ci_low':a['accuracy']['ci95_pp'][0],n+'_ci_high':a['accuracy']['ci95_pp'][1]})
  w.writerow(row)
print(json.dumps(out['audit'],indent=2))
