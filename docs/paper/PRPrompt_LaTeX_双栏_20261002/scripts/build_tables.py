#!/usr/bin/env python3
"""Generate shared EN/ZH LaTeX tables from sealed, condition-wise JSON only."""
from pathlib import Path
import json, hashlib
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'
OUT=ROOT/'tables'
D=['Cora','CiteSeer','PubMed'];SHOTS=[1,3,5]
METHODS=['FrozenPrototype','FullFineTune','GPF','GPFPlus','GraphPromptDiag','PRPrompt']
ABLATION=['FrozenPrototype','NoRouting','PRPrompt']
LABELS={'FrozenPrototype':r'\frozenname','FullFineTune':r'\ftname','GPF':'GPF','GPFPlus':'GPF-plus','GraphPromptDiag':r'\gpromptname','PRPrompt':r'\method','NoRouting':r'\sharedname'}
METRICS={'accuracy':'Accuracy','macro_f1':'Macro-F1','micro_f1':'Micro-F1','roc_auc_macro':'Macro ROC-AUC','average_precision_macro':'Macro AP'}
def read(name):return json.loads((DATA/name).read_text())
def table(caption,label,heads,rows,small=True):
 static=label in ['tab:main-micro_f1','tab:main-roc_auc_macro','tab:main-average_precision_macro']
 env='table*' if small else 'table'
 opening=r'\par\medskip\noindent\begin{minipage}{\textwidth}' if static else r'\begin{'+env+'}'+('[t]' if small else '[H]')
 cap=r'\captionof{table}{' if static else r'\caption{'
 lines=[opening,r'\linespread{1}\selectfont' if label.startswith('tab:paired-') else '',r'\centering',r'\captionsetup{font=small}' if small else r'\captionsetup{font=footnotesize}',cap+caption+'}',r'\label{'+label+'}',r'\small' if small else r'\footnotesize',r'\setlength{\tabcolsep}{4pt}' if small else r'\setlength{\tabcolsep}{2pt}']
 if label.startswith('tab:paired-'):lines.extend([r'\fontsize{8}{9}\selectfont',r'\setlength{\tabcolsep}{2pt}'])
 if small:lines.append(r'\resizebox{\linewidth}{!}{%')
 lines += [r'\begin{tabular}{l'+'r'*(len(heads)-1)+'}',r'\toprule',' & '.join(heads)+r' \\',r'\midrule']
 for r in rows:lines.append(' & '.join(r)+r' \\')
 lines.extend([r'\bottomrule',r'\end{tabular}'])
 if small:lines.append('}')
 lines.append(r'\end{minipage}\par' if static else r'\end{'+env+'}')
 return '\n'.join(lines)+'\n'
def render():
 main=read('main.json');abl=read('ablation.json');effects=read('paired.json')
 outputs={};cells=[]
 for group,data,methods,metrics in [('main',main,METHODS,METRICS),('ablation',abl,ABLATION,{k:METRICS[k] for k in ['accuracy','macro_f1']})]:
  idx={(r['dataset'],r['shots'],r['method']):r for r in data}
  assert set(idx)=={(d,s,m) for d in D for s in SHOTS for m in methods}
  for metric,title in metrics.items():
   rows=[]
   for d in D:
    for s in SHOTS:
     row=[f'{d} / {s}'];best=max(idx[d,s,m][metric+'_mean'] for m in methods)
     for m in methods:
      r=idx[d,s,m];mean=100*r[metric+'_mean'];std=100*r[metric+'_std_over_support_draws']
      macro='bestmeanstd' if r[metric+'_mean']==best else 'meanstd'
      cell='\\'+macro+f'{{{mean:.2f}}}{{{std:.2f}}}';row.append(cell)
      cells.append(dict(table=group,metric=metric,dataset=d,shots=s,method=m,mean=mean,std=std,latex=cell))
     rows.append(row)
   en=f'{title} (\\%, higher is better). '+('Main comparison.' if group=='main' else 'Prototype-routing ablation.')+' Within each dataset/shot/method, training seeds are averaged within each support draw; mean and sample standard deviation are then computed over 20 support draws. Bold marks the largest mean in that row.'
   zh=f'{title}（\\%，越高越好）。'+('主比较。' if group=='main' else '原型路由消融。')+'固定数据集、shot 与方法，在每个支持抽样内先平均训练 seed，再对 20 个支持抽样计算均值与样本标准差。粗体为该行最大均值。'
   if group=='main':en+=' $\\dagger$: common-protocol adaptation.';zh+=' $\\dagger$：统一协议适配实现。'
   if metric not in ['accuracy','macro_f1']:
    en=f'{title} (\\%, higher is better). Main comparison; seed aggregation follows Appendix~\\ref{{app:protocol}}.'
    zh=f'{title}（\\%，越高越好）。主比较；seed 统计沿用附录~\\ref{{app:protocol}}。'
   outputs[f'{group}_{metric}.tex']=table(r'\lang{'+en+'}{'+zh+'}',f'tab:{group}-{metric}',[r'\lang{Dataset / shot}{数据集 / shot}']+[LABELS[m] for m in methods],rows,small=(group=='main'))
 pairs=[]
 for ref in METHODS[:-1]+['NoRouting']:
  rows=[]
  for d in D:
   for s in SHOTS:
    row=[f'{d} / {s}']
    for metric in ['accuracy','macro_f1']:
     r=next(r for r in effects if (r['dataset'],r['shots'],r['reference'],r['metric'])==(d,s,ref,metric))
     assert r['interval_source']=='original_paired_interval'
     cell=rf"\shortstack{{${100*r['mean_delta']:+.2f}$\\$[{100*r['ci95_low']:+.2f},\,{100*r['ci95_high']:+.2f}]$}}"
     row.append(cell);cells.append(dict(table='paired',metric=metric,dataset=d,shots=s,reference=ref,latex=cell))
    rows.append(row)
  caption=r'\lang{\method{} minus '+LABELS[ref]+r': paired differences and 95\% intervals (percentage points).}{\method{} 相对 '+LABELS[ref]+r' 的配对差值与 95\% 区间（百分点）。}'
  pairs.append(table(caption,'tab:paired-'+ref,[r'\lang{Dataset / shot}{数据集 / shot}','Accuracy','Macro-F1'],rows,False))
 outputs['paired_all.tex']='\n'.join(pairs[:3])+r'\newpage'+'\n'+'\n'.join(pairs[3:])
 stats=read('datasets.json')
 outputs['datasets.tex']=table(r'\lang{Dataset dimensions and trainable prompt counts from the recorded protocol.}{记录协议中的数据维度及可学习提示参数量。}','tab:datasets',[r'\lang{Dataset}{数据集}',r'\lang{Nodes}{节点}',r'\lang{Features}{特征}',r'\lang{Classes}{类别}',r'\lang{Prompt parameters}{提示参数}'],[[d,str(stats[d]['nodes']),str(stats[d]['input_dim']),str(stats[d]['classes']),str(stats[d]['classes']*128)] for d in D],False)
 return outputs,cells
if __name__=='__main__':
 outputs,cells=render();OUT.mkdir(exist_ok=True)
 for name,content in outputs.items():(OUT/name).write_text(content)
 (ROOT/'table_cell_manifest.json').write_text(json.dumps(cells,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(dict(tables=len(outputs),numeric_cells=len(cells)),ensure_ascii=False))
