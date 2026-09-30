from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent.parent;O=R/'reports/spectral_temperature_r18'
rows=list(csv.DictReader((O/'summary.csv').open()));names=['Computers','Photo','Squirrel','WikiCS','Roman-empire','Amazon-ratings']
u=[next(r for r in rows if r['dataset']==n and r['branch']=='unanchored') for n in names]
a=[next(r for r in rows if r['dataset']==n and r['branch']=='anchor') for n in names]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,ax=plt.subplots(1,2,figsize=(12,4.8),gridspec_kw={'width_ratios':[1.18,1]});x=np.arange(6)
ax[0].bar(x-.18,[float(r['fixed_v_ratio']) for r in u],width=.36,color='#9CA3AF',label='Fixed temperature 0.05')
ax[0].bar(x+.18,[float(r['spectral_v_ratio']) for r in u],width=.36,color='#2675B4',label='Geometry-calibrated temperature')
ax[0].set_yscale('log');ax[0].set_ylim(1e-21,3);ax[0].set_xticks(x,names,rotation=30,ha='right');ax[0].set_ylabel('Prototype dispersion / initial dispersion')
ax[0].set_title('A  Unanchored: finite-step contrast recovers',loc='left',fontweight='bold');ax[0].grid(axis='y',alpha=.12)
d=np.array([float(r['delta_accuracy_pp']) for r in a]);lo=np.array([float(r['delta_ci_low']) for r in a]);hi=np.array([float(r['delta_ci_high']) for r in a]);colors=['#BA584B' if v<0 else '#2675B4' for v in d]
ax[1].barh(x,d,color=colors,alpha=.9);ax[1].errorbar(d,x,xerr=np.stack([d-lo,hi-d]),fmt='none',ecolor='#303846',capsize=3,lw=1)
ax[1].set_yticks(x,names);ax[1].invert_yaxis();ax[1].axvline(0,color='#303846',lw=.9);ax[1].set_xlabel('Accuracy change from fixed temperature (pp)');ax[1].set_title('B  Half-anchored: accuracy need not improve',loc='left',fontweight='bold');ax[1].grid(axis='x',alpha=.12)
fig.suptitle('Controlling feedback strength is different from improving classification',x=.04,ha='left',fontsize=14,fontweight='bold')
fig.legend(*ax[0].get_legend_handles_labels(),loc='upper left',bbox_to_anchor=(.055,.915),ncol=2,frameon=False,fontsize=9)
fig.text(.04,.02,'Fixed graph and inputs; 5-shot, 20 paired support draws, 3 rounds. Temperature from geometry; no accuracy selection.\nPanel B: 95% paired bootstrap intervals, no multiple-comparison correction. The two panels intentionally test different anchor settings.',fontsize=8.5,color='#4B5563')
fig.subplots_adjust(left=.07,right=.99,top=.78,bottom=.23,wspace=.55)
for ext in ['png','pdf','svg']:fig.savefig(O/f'feedback_intervention.{ext}',dpi=180,bbox_inches='tight')
