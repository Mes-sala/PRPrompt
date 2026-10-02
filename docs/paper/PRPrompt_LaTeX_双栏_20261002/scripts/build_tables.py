#!/usr/bin/env python3
"""Typeset condition-wise results with dataset/shot column groups; no aggregation."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'; OUT=ROOT/'tables'
D=['Cora','CiteSeer','PubMed']; SHOTS=[1,3,5]
METHODS=['FrozenPrototype','FullFineTune','GPF','GPFPlus','GraphPromptDiag','PRPrompt']
ABLATION=['FrozenPrototype','NoRouting','PRPrompt']
LABELS={'FrozenPrototype':r'\frozenname','FullFineTune':r'\ftname','GPF':'GPF','GPFPlus':'GPF-plus','GraphPromptDiag':r'\gpromptname','PRPrompt':r'\textbf{\method}','NoRouting':r'\sharedname'}
METRICS={'accuracy':'Accuracy','macro_f1':'Macro-F1','micro_f1':'Micro-F1','roc_auc_macro':'Macro ROC-AUC','average_precision_macro':'Macro AP'}
def read(name):return json.loads((DATA/name).read_text())
def lang(en,zh):return r'\lang{'+en+'}{'+zh+'}'

def start(caption,label,static=False,narrow=False):
 opening=r'\par\medskip\noindent\begin{minipage}{\textwidth}' if static else (r'\begin{table}[H]' if narrow else r'\begin{table*}[t]')
 cap=r'\captionof{table}{' if static else r'\caption{'
 return [opening,r'\centering',r'\linespread{1}\selectfont',r'\captionsetup{font=small,skip=5pt}',cap+caption+'}',r'\label{'+label+'}',r'\fontsize{9}{11}\selectfont',r'\setlength{\tabcolsep}{1pt}',r'\renewcommand{\arraystretch}{1.18}',r'\setlength{\heavyrulewidth}{0.8pt}',r'\setlength{\lightrulewidth}{0.4pt}']

def grouped_header(first=None):
 if first is None:first=lang('Method','方法')
 return [r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}l*{9}{r}@{}}',r'\toprule',first+r' & \multicolumn{3}{c}{\textbf{Cora}} & \multicolumn{3}{c}{\textbf{CiteSeer}} & \multicolumn{3}{c}{\textbf{PubMed}} \\',r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(l){8-10}',r' & 1-shot & 3-shot & 5-shot & 1-shot & 3-shot & 5-shot & 1-shot & 3-shot & 5-shot \\',r'\midrule']

def end(lines,note='',static=False,narrow=False):
 lines.extend([r'\bottomrule',r'\end{tabular*}'])
 if note:lines.extend([r'\par\vspace{3pt}',r'\begin{minipage}{\linewidth}\fontsize{8}{9.5}\selectfont\raggedright '+note+r'\end{minipage}'])
 lines.append(r'\end{minipage}\par' if static else (r'\end{table}' if narrow else r'\end{table*}'))
 return '\n'.join(lines)+'\n'

STATS=lang(r'Mean $\pm$ sample standard deviation (\%). Within each fixed condition, three training seeds are averaged per support draw, then 20 support draws are summarized. \textbf{Bold}: best; \underline{underlined}: second-best mean in each column. No averaging across columns.',r'均值 $\pm$ 样本标准差（\%）。每个固定条件内，先平均支持抽样内的 3 个训练 seed，再统计 20 次支持抽样。每列\textbf{粗体}为最优均值，\underline{下划线}为次优；不跨列平均。')
SHORT=lang(r'Statistics and emphasis follow Table~\ref{tab:main-accuracy}. Each column is a separate dataset/shot condition.',r'统计与强调规则同表~\ref{tab:main-accuracy}；每列对应独立的数据集／shot 条件。')

def render():
 main=read('main.json');abl=read('ablation.json');effects=read('paired.json');outputs={};cells=[]
 def metric_rows(group,data,methods,metric):
  idx={(r['dataset'],r['shots'],r['method']):r for r in data};rows=[]
  for m in methods:
   if group=='main' and m=='GPF':rows.append(r'\midrule')
   if m=='PRPrompt':rows.append(r'\midrule\rowcolor{black!5}')
   row=[LABELS[m]]
   for d in D:
    for s in SHOTS:
     r=idx[d,s,m];means=sorted({idx[d,s,k][metric+'_mean'] for k in methods},reverse=True)
     rank=means.index(r[metric+'_mean']);macro='bestmeanstd' if rank==0 else 'secondmeanstd' if rank==1 else 'meanstd'
     mean=100*r[metric+'_mean'];std=100*r[metric+'_std_over_support_draws'];cell='\\'+macro+f'{{{mean:.2f}}}{{{std:.2f}}}'
     row.append(cell);cells.append(dict(table=group,metric=metric,dataset=d,shots=s,method=m,mean=mean,std=std,rank=rank+1,latex=cell))
   rows.append(' & '.join(row)+r' \\')
  return rows
 for metric,title in METRICS.items():
  static=metric not in ['accuracy','macro_f1']
  caption=lang(r'\textbf{'+title+r' on few-shot node classification.} Higher is better ($\uparrow$).',r'\textbf{少样本节点分类的 '+title+r'。}越高越好（$\uparrow$）。')
  lines=start(caption,'tab:main-'+metric,static)+grouped_header()+metric_rows('main',main,METHODS,metric)
  note=STATS if metric=='accuracy' else SHORT
  if not static:note+=lang(r' $\dagger$: GraphPrompt under the common-protocol adaptation.',r' $\dagger$：GraphPrompt 的统一协议适配实现。')
  outputs['main_'+metric+'.tex']=end(lines,note,static)
 # One ablation table with two metric panels, sharing the condition header.
 lines=start(lang(r'\textbf{Prototype-routing ablation.} Matched prompt recipe and readout.',r'\textbf{原型路由消融。}提示训练配方与读出保持一致。'),'tab:ablation')+grouped_header()
 for i,metric in enumerate(['accuracy','macro_f1']):
  if i:lines.append(r'\midrule')
  lines.extend([r'\multicolumn{10}{@{}l}{\textit{'+METRICS[metric]+r' (\%) $\uparrow$}} \\[2pt]'])
  lines.extend(metric_rows('ablation',abl,ABLATION,metric))
 outputs['ablation_combined.tex']=end(lines,SHORT+lang(r' Frozen: no prompt; shared prompt: uniform allocation; PRPrompt: prototype routing.',r' 冻结预训练：无提示；共享提示：均匀分配；PRPrompt：原型路由。'))
 # One full-width paired table per metric instead of six miniature tables.
 pairs=[]
 for metric in ['accuracy','macro_f1']:
  title=METRICS[metric];caption=lang(r'\textbf{Paired '+title+r' gains of \method.} Differences in percentage points.',r'\textbf{\method 的 '+title+r' 配对增益。}差值单位为百分点。')
  lines=start(caption,'tab:paired-'+metric,True)+grouped_header(lang('Reference','对照方法'))
  lines.extend([r'\renewcommand{\arraystretch}{1.35}'])
  for m in METHODS[:-1]+['NoRouting']:
   if m in ['GPF','NoRouting']:lines.append(r'\midrule')
   row=[LABELS[m]]
   for d in D:
    for s in SHOTS:
     r=next(r for r in effects if (r['dataset'],r['shots'],r['reference'],r['metric'])==(d,s,m,metric));assert r['interval_source']=='original_paired_interval'
     delta,lo,hi=[100*r[k] for k in ['mean_delta','ci95_low','ci95_high']]
     cell=rf'\pairedcell{{{delta:+.2f}}}{{{lo:.2f}}}{{{hi:.2f}}}'
     row.append(cell);cells.append(dict(table='paired',metric=metric,dataset=d,shots=s,reference=m,mean_delta=delta,ci95_low=lo,ci95_high=hi,latex=cell))
   lines.append(' & '.join(row)+r' \\[2pt]')
  note=lang(r'Each cell: PRPrompt minus the reference (top); unadjusted 95\% paired support-bootstrap interval (below). All values preserve the original condition-wise statistics.',r'每格上行为 PRPrompt 减去对照的差值，下行为未经多重比较校正的 95\% 配对支持 bootstrap 区间。数值沿用原始逐条件统计。')
  pairs.append(end(lines,note,True))
 outputs['paired_all.tex']='\n\n'.join(pairs)
 stats=read('datasets.json');lines=start(lang(r'\textbf{Dataset and prompt sizes.}',r'\textbf{数据集规模与提示参数量。}'),'tab:datasets',narrow=True)
 lines += [r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrr@{}}',r'\toprule',lang('Dataset','数据集')+' & '+lang('Nodes','节点')+' & '+lang('Features','特征')+' & '+lang('Classes','类别')+' & '+lang(r'\shortstack{Prompt\\params.}',r'\shortstack{提示\\参数}')+r' \\',r'\midrule']
 for d in D:lines.append(' & '.join([d,f"{stats[d]['nodes']:,}",f"{stats[d]['input_dim']:,}",str(stats[d]['classes']),str(stats[d]['classes']*128)])+r' \\')
 outputs['datasets.tex']=end(lines,narrow=True)
 return outputs,cells
if __name__=='__main__':
 outputs,cells=render();OUT.mkdir(exist_ok=True)
 for name,content in outputs.items():(OUT/name).write_text(content)
 (ROOT/'table_cell_manifest.json').write_text(json.dumps(cells,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(dict(tables=len(outputs),numeric_cells=len(cells))))
