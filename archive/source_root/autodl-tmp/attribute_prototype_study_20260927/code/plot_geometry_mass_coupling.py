"""Scientific figures from verified R22 and 80-digit counterexample artifacts."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'reports/geometry_mass_coupling';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
 'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
blue='#1769aa';orange='#d97724';dark='#263244'
def save(fig,name):
 for ext in ['png','svg','pdf']:fig.savefig(OUT/f'{name}.{ext}',dpi=180,bbox_inches='tight')
 plt.close(fig)

p=json.loads((ROOT/'theory/池质量_方向集中_80位反例.json').read_text())
a=[r for r in p['rows'] if r['policy']=='fixed_pool'];b=[r for r in p['rows'] if r['policy']=='joint_pool'];x=[1/r['epsilon'] for r in a]
fig,ax=plt.subplots(1,3,figsize=(13.2,4.4))
ax[0].loglog(x,[r['soft_location_residual'] for r in a],'-o',color=dark,ms=4)
ax[0].set(title='A  Absolute location feedback',ylabel='Soft mean minus pool mean (norm)')
ax[0].text(.05,.07,'Identical in both pool policies',transform=ax[0].transAxes,fontsize=9,color=dark)
for rows,color,label in [(a,orange,'Fixed pool'),(b,blue,r'Repeated pool: $r=\epsilon^{-2}$')]:
 ax[1].loglog(x,[r['feedback_anchor_ratio'] for r in rows],'-o',color=color,ms=4,label=label)
 ax[2].semilogx(x,[r['full_common_disagreement_pct'] for r in rows],'-o',color=color,ms=4,label=label)
ax[1].set(title='B  Feedback relative to anchor',ylabel='Feedback / anchor displacement')
ax[1].legend(fontsize=8,loc='lower left',frameon=False)
ax[2].set(title='C  Full vs. common-pool decisions',ylabel='Prediction disagreement (%)',ylim=(-1,28))
ax[2].axhline(25,ls=':',lw=1,color='#777777');ax[2].text(480,25.8,'25% limit',ha='right',fontsize=9)
for axis in ax:
 axis.set_xlabel(r'Concentration factor $1/\epsilon$ (larger = closer)');axis.grid(alpha=.15)
fig.suptitle('Vanishing absolute feedback can retain a nonzero decision effect',fontsize=14,fontweight='bold',y=.97)
fig.subplots_adjust(left=.065,right=.99,bottom=.22,top=.80,wspace=.32)
fig.text(.5,.015,'Fixed synthetic construction; 80-digit arithmetic. No class-accuracy labels are used.',ha='center',fontsize=9,color='#555555')
save(fig,'counterexample')

names=['WikiCS','Roman-empire','Amazon-ratings'];methods=['common_true_mass__r1','common_true_mass__r8','moment_mass2__r8']
labels=['Common pool with exact mass, r=1','Common pool with exact mass, r=8','Position + mass approximation, r=8']
fig,ax=plt.subplots(figsize=(9,4.1),layout='constrained');width=.24
summary=[]
for j,(key,label,color) in enumerate(zip(methods,labels,[orange,blue,dark])):
 values=[]
 for name in names:
  rows=[json.loads((ROOT/'results/mass_anchor_r22'/name/f'{seed}.json').read_text())['endpoint'][key]['disagreement_pct'] for seed in range(271700,271705)]
  values.append(rows);summary.append(dict(dataset=name,method=key,mean=float(np.mean(rows)),minimum=min(rows),maximum=max(rows)))
 mean=np.mean(values,axis=1);lo=mean-np.min(values,axis=1);hi=np.max(values,axis=1)-mean;pos=np.arange(3)+(j-1)*width
 ax.bar(pos,mean,width,color=color,label=label,alpha=.9)
 ax.errorbar(pos,mean,yerr=[lo,hi],fmt='none',color=color,capsize=3,lw=1)
 for xx,yy in zip(pos,mean):ax.text(xx,yy+.2,f'{yy:.3f}' if yy<.01 else f'{yy:.2f}',ha='center',fontsize=8)
ax.set_xticks(np.arange(3),names);ax.set_ylim(0,11);ax.set_ylabel('Disagreement with full Ren-style inference (%)')
ax.set_title('Repeating the same pool reveals the effect of small location feedback',fontweight='bold',pad=12)
ax.legend(frameon=False,ncol=1,loc='upper left',fontsize=8);ax.grid(axis='y',alpha=.15)
fig.get_layout_engine().set(rect=(0,.12,1,1))
fig.text(.5,.01,'R22: fixed-feature inference operators, 5 development support draws per graph. Whiskers: min–max.\nDeclared local domain only; these are prediction differences, not classification error rates.',ha='center',fontsize=9,color='#555555')
save(fig,'r22_location_feedback')
(OUT/'figure_data.json').write_text(json.dumps(dict(counterexample_source='theory/池质量_方向集中_80位反例.json',r22_summary=summary),ensure_ascii=False,indent=2))
print(OUT)
