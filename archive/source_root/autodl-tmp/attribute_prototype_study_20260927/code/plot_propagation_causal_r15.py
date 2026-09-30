from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent.parent
rows=list(csv.DictReader((R/'reports/propagation_causal_r15/summary.csv').open()));idx={(t['dataset'],t['mode'],int(t['hops'])):t for t in rows};names=sorted(set(t['dataset'] for t in rows));signed=[n for n in names if (n,'remove_shift_same_denominator',0) in idx]
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,axes=plt.subplots(1,3,figsize=(16,6),gridspec_kw={'width_ratios':[1.3,1,1]});colors=['#536B8E','#C66037','#35978F','#9262A6']
a=axes[0]
for i,n in enumerate(names):
 x=[float(idx[n,'original',h]['rho_estimate']) for h in [0,2]];a.plot(x,[i,i],color='#aaaaaa',lw=1);a.scatter(x[0],i,color=colors[0],s=28);a.scatter(x[1],i,color=colors[1],s=28,marker='s')
a.set_yticks(range(len(names)),names);a.invert_yaxis();a.set_xscale('log');a.axvline(1,color='black',ls='--',lw=1);a.set_title('A  Propagation does not always increase consensus');a.set_xlabel('Estimated local spectral radius (log scale)');a.scatter([],[],color=colors[0],label='Original, 0 hops');a.scatter([],[],color=colors[1],marker='s',label='Original, 2 hops')
conds=[('original',0),('original',2),('remove_shift_same_denominator',0),('remove_shift_same_denominator',2)];labels=['Original, 0 hops','Original, 2 hops','Remove shift, 0 hops','Remove shift, 2 hops']
for ax,metric,title in [(axes[1],'rho_estimate','B  Removing the preprocessing shift'),(axes[2],'unanchored_ratio','C  Actual three-step class contrast')]:
 for j,((m,h),label) in enumerate(zip(conds,labels)):
  vals=[max(float(idx[n,m,h][metric]),1e-22) for n in signed];ax.bar(np.arange(len(signed))+(j-1.5)*.19,vals,width=.18,color=colors[j],label=label)
 ax.set_yscale('log');ax.set_xticks(range(len(signed)),[n.replace('-','-\n') for n in signed]);ax.set_title(title);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
axes[1].axhline(1,color='black',ls='--',lw=1);axes[1].set_ylabel('Estimated spectral radius');axes[2].set_ylabel('Class-prototype dispersion / initial dispersion')
fig.suptitle('Fixed attributes: preprocessing, graph propagation, and prototype consensus',fontsize=15,y=.99)
fig.text(.5,.015,'14 datasets; 20 matched support draws; temperature 0.05; no accuracy tuning. Removing shift keeps the original row denominators.',ha='center',fontsize=10)
handles,legend_labels=axes[1].get_legend_handles_labels();fig.legend(handles,legend_labels,loc='upper center',bbox_to_anchor=(.5,.94),ncol=4,frameon=False,fontsize=10)
fig.tight_layout(rect=[0,.06,1,.86],w_pad=2.3)
out=R/'reports/propagation_causal_r15';out.mkdir(exist_ok=True)
for ext in ['png','pdf','svg']:fig.savefig(out/('mechanism_interventions.'+ext),dpi=170,bbox_inches='tight')
