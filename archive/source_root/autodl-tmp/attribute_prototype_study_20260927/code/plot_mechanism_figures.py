"""Publication-style summary of fully verified mechanism observations."""
from pathlib import Path
import json,csv,argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main(root):
 p=Path(root)
 for sub in ['attribute_mechanism','globalmean_control']:assert json.loads((p/'reports'/sub/'verification.json').read_text())['complete']
 rows=list(csv.DictReader((p/'reports/attribute_mechanism/mechanism_summary.csv').open()));data={(r['dataset'],r['condition']):r for r in rows};names=sorted({r['dataset'] for r in rows});assert len(names)==14;contrasts=list(csv.DictReader((p/'reports/globalmean_control/mechanism_contrasts.csv').open()));out=p/'reports/mechanism_figures';out.mkdir(exist_ok=True)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42});fig,ax=plt.subplots(2,2,figsize=(14,10));x=np.arange(14)
 ratios=[float(data[n,'graph_anchor']['support_draw_variance'])/float(data[n,'graph_no_update']['support_draw_variance']) for n in names];ax[0,0].bar(x,ratios,color='#33658A');ax[0,0].axhline(1,color='gray',lw=1,ls='--');ax[0,0].set_ylabel('Variance ratio (anchored / no update)');ax[0,0].set_title('A. Support-sampling variability')
 mr=[float(data[n,'graph_anchor']['prototype_mse'])/float(data[n,'graph_no_update']['prototype_mse']) for n in names];ad=[float(data[n,'graph_anchor']['accuracy'])-float(data[n,'graph_no_update']['accuracy']) for n in names];ax[0,1].scatter(mr,ad,c='#33658A');ax[0,1].axhline(0,color='gray',lw=1);ax[0,1].axvline(1,color='gray',lw=1,ls='--');ax[0,1].set(xlabel='Prototype MSE ratio',ylabel='Accuracy change (pp)',title='B. Estimation error versus classification')
 offsets={'Amazon-ratings':(-8,-25),'Physics':(15,-45),'PubMed':(48,-22),'Squirrel':(-2,18),'Roman-empire':(8,-35)}
 for n,a,b in zip(names,mr,ad):
  off=offsets.get(n,(3,3));ax[0,1].annotate(n,(a,b),xytext=off,textcoords='offset points',fontsize=7,bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=.3),arrowprops=dict(arrowstyle='-',color='gray',lw=.5) if n in offsets else None)
 def get(n,t,c):return next(r for r in contrasts if r['dataset']==n and r['treatment']==t and r['control']==c and r['metric']=='accuracy_delta_pp')
 for offset,t,c,color,label in [(-.2,'graph_globalmean','graph_no_update','#E2A14A','Common shrinkage - no update'),(.2,'graph_anchor','graph_globalmean','#33658A','Soft anchored - common shrinkage')]:
  rr=[get(n,t,c) for n in names];v=np.array([float(r['delta']) for r in rr]);lo=np.array([float(r['ci_low']) for r in rr]);hi=np.array([float(r['ci_high']) for r in rr]);ax[1,0].bar(x+offset,v,.4,color=color,label=label);ax[1,0].errorbar(x+offset,v,yerr=np.maximum(0,np.stack([v-lo,hi-v])),fmt='none',ecolor='#333333',lw=.7,capsize=2)
 ax[1,0].axhline(0,color='gray',lw=1);ax[1,0].set_title('C. Common shrinkage and class-conditioned correction');ax[1,0].set_ylabel('Accuracy difference (pp)');ax[1,0].legend(fontsize=7)
 v=[float(data[n,'graph_anchor']['accuracy'])-float(data[n,'graph_unanchored']['accuracy']) for n in names];ax[1,1].bar(x,v,color='#60977A');ax[1,1].axhline(0,color='gray',lw=1);ax[1,1].set_title('D. Retaining versus removing the anchor');ax[1,1].set_ylabel('Accuracy difference (pp)')
 for a in [ax[0,0],ax[1,0],ax[1,1]]:a.set_xticks(x,names,rotation=55,ha='right')
 fig.suptitle('Fixed-attribute mechanism study: 20 paired support draws per dataset',fontsize=14);fig.tight_layout(rect=(0,.03,1,.96));fig.text(.02,.012,'MSE uses empirical query-class means. Error bars: paired bootstrap 95% intervals, unadjusted. No checkpoint or parameter selection.',fontsize=8)
 for ext in ['png','pdf','svg']:fig.savefig(out/('mechanism_summary.'+ext),dpi=180,bbox_inches='tight')
 print(out)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('root');main(ap.parse_args().root)
