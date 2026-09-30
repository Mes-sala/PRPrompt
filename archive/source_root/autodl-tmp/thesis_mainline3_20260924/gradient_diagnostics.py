from pathlib import Path
import json,torch,math,statistics
R=Path(__file__).resolve().parent;torch.set_num_threads(4);records=[]
for stage in ['stage2_confirmation','stage2b_gppt_confirmation','stage2c_dgi_confirmation']:
 p=R/stage;plan=json.loads((p/'plan.json').read_text());assert json.loads((p/'final_audit.json').read_text())['passed']
 for j in plan['jobs']:
  if j['direct']:continue
  for seed in plan['seeds']:
   d=p/'results'/j['id']/f'{seed}__F111';r=json.loads((d/'result.json').read_text());a=torch.load(d/'sealed.pt',map_location='cpu',weights_only=True);g=a['prompt']['optimizer_gradients'].double();assert g.shape[0]==100
   left=g.permute(1,0,2).reshape(g.shape[1],-1);right=g.reshape(-1,g.shape[2]);L=1e-4*torch.eye(g.shape[1],dtype=torch.float64)+left@left.T;Q=1e-4*torch.eye(g.shape[2],dtype=torch.float64)+right.T@right
   el=torch.linalg.eigvalsh(L);er=torch.linalg.eigvalsh(Q);assert el.min()>0 and er.min()>0
   nr=er/er.sum();rank=float((-(nr*nr.log()).sum()).exp())
   diag=r['trace']['optimizer_diagnostics'];ratios=[x['adam_norm']/x['direction_norm'] for x in diag if not x['zero_direction']]
   records.append(dict(stage=stage,dataset=j['dataset'],checkpoint=j['checkpoint']['family'],sha256=j['checkpoint']['sha256'],seed=seed,log10_condition_left=math.log10(float(el.max()/el.min())),log10_condition_right=math.log10(float(er.max()/er.min())),right_effective_rank=rank,median_graft_scale=statistics.median(ratios)))
out=R/'chapter_assets_final';out.mkdir(exist_ok=True);(out/'gradient_diagnostics.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
t='# 支持梯度结构诊断\n\n在完整F111实际训练轨迹上重建累积左右梯度二阶矩（包含1e-4稳定项），计算log10条件数、右矩阵有效秩和每步步长匹配倍率的中位数。以下先按seed统计再取均值。这是描述性诊断；不同底座的训练预算、表示质量和梯度轨迹均有差异，不能据此单独归因为预训练目标。\n\n| 数据集/底座 | log10左条件数 | log10右条件数 | 右有效秩 | 匹配倍率中位数的均值 |\n|---|---:|---:|---:|---:|\n'
keys=list(dict.fromkeys((r['stage'],r['dataset'],r['checkpoint']) for r in records))
for stage,ds,family in keys:
 rr=[r for r in records if (r['stage'],r['dataset'],r['checkpoint'])==(stage,ds,family)];v=[statistics.mean(r[k] for r in rr) for k in ['log10_condition_left','log10_condition_right','right_effective_rank','median_graft_scale']];t+='| '+ds+'/'+family+' | '+' | '.join(f'{x:.3f}' for x in v)+' |\n'
(out/'梯度诊断.md').write_text(t);print('gradient_diagnostics_complete',len(records))
