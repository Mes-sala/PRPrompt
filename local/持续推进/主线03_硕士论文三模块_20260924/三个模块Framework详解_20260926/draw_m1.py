from pathlib import Path
import json, itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Rectangle,FancyBboxPatch,FancyArrowPatch,Polygon
from matplotlib.font_manager import FontProperties
from matplotlib.colors import to_rgb
OUT=Path(__file__).resolve().parent
FONT=FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc')
plt.rcParams.update({'pdf.fonttype':42,'svg.fonttype':'path','mathtext.fontset':'stix'})
INK='#24394B';GRAY='#667D90';LINE='#A1B4C1';BLUE='#5683A9';TEAL='#238D89';PURPLE='#926FB1';ORANGE='#B56B3D';COL=[TEAL,PURPLE]
fig=plt.figure(figsize=(20,12.5),facecolor='white');ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,2000),ylim=(1250,0));ax.axis('off')
def txt(x,y,s,fs=23,col=INK,ha='center',math=False):return ax.text(x,y,s,fontsize=fs,color=col,ha=ha,va='center',fontproperties=None if math else FONT,zorder=15)
def rr(x,y,w,h,fc,ec='none',r=18):ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',fc=fc,ec=ec,lw=1,zorder=0))
def line(pts,c=LINE,lw=1.5,dash=False,z=2):ax.plot(*np.array(pts).T,color=c,lw=lw,ls=(0,(4,3)) if dash else '-',zorder=z)
def arrow(a,b,c=GRAY,lw=1.8,rad=0,dash=False):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=18,linewidth=lw,color=c,linestyle=(0,(4,3)) if dash else '-',connectionstyle=f'arc3,rad={rad}',zorder=3,shrinkA=2,shrinkB=2))
def route(pts,c=GRAY,lw=1.8,dash=False):line(pts[:-1],c,lw,dash);arrow(pts[-2],pts[-1],c,lw,dash=dash)
def dot(x,y,c=GRAY,r=10,ring=False):
 ax.add_patch(Circle((x,y),r,fc=c,ec='white',lw=1,zorder=6))
 if ring:ax.add_patch(Circle((x,y),r+5,fc='none',ec=INK,lw=1,zorder=7))
def star(x,y,c=TEAL,size=500,outline=False):ax.scatter([x],[y],s=size,c=['white' if outline else c],marker='*',edgecolors=c,lw=1.5,zorder=9)
def bar(x,y,p,w=180,h=30,numeric=False):
 start=x
 for k,v in enumerate(p):
  ax.add_patch(Rectangle((start,y),w*v,h,fc=COL[k],ec='white',lw=.8,zorder=5))
  if numeric and v>.1:txt(start+w*v/2,y+h/2,f'{v:.2f}',18,'white')
  start+=w*v

def mat(x,y,arr=None,rows=3,cols=5,cell=16,c=BLUE):
 if arr is None:arr=np.array([[.25+.65*((i*3+j*2)%7)/6 for j in range(cols)] for i in range(rows)])
 rows,cols=arr.shape
 for i in range(rows):
  for j in range(cols):
   color=COL[i%2] if c=='class' else c;rgb=to_rgb(color);a=float(abs(arr[i,j]));fc=tuple(1-a+a*t for t in rgb)
   ax.add_patch(Rectangle((x+j*cell,y+i*cell),cell,cell,fc=fc,ec='white',lw=.65,zorder=5))
 ax.add_patch(Rectangle((x,y),cols*cell,rows*cell,fc='none',ec=LINE,lw=.7,zorder=6))
def snow(x,y):
 for a in [0,np.pi/3,2*np.pi/3]:
  d=10*np.array([np.cos(a),np.sin(a)]);line([[x-d[0],y-d[1]],[x+d[0],y+d[1]]],BLUE,1.6,z=6)
def encoder(x,y,label):
 rr(x,y,82,100,'#DFECF5',BLUE,9)
 for i in range(3):
  for j in range(3):dot(x+20+i*21,y+27+j*24,'white',3)
 for i in range(2):
  for j,k in itertools.product(range(3),range(3)):line([[x+20+i*21,y+27+j*24],[x+41+i*21,y+27+k*24]],'#AAC5DA',.55)
 snow(x+41,y-20);txt(x+41,y+123,label,19,BLUE)

# A mathematically consistent illustrative assignment from unit feature vectors.
A=np.array([[.8,.2],[.6,.4],[.2,.8]])
delta=.05*np.log(A[:,0]/A[:,1]);summ=np.array([1.1,.65,.9]);u=(summ+delta)/2;v=(summ-delta)/2
R=np.column_stack([u,v,np.sqrt(1-u*u-v*v)])
S=R[:,:2]/.05;calc=np.exp(S-S.max(1,keepdims=True));calc/=calc.sum(1,keepdims=True)
assert np.allclose(A,calc)
soft=A.T@R/A.sum(0)[:,None]
# Display is an affine 2D projection of the same three vectors; weighted means commute with it.
B=np.linalg.solve(np.column_stack([R[:,0],R[:,2],np.ones(3)]),np.array([[0,0],[225,15],[80,135]]))
D=np.column_stack([R[:,0],R[:,2],np.ones(3)])@B

# Header and prototype construction.
txt(45,45,'M1  类别语义引导：软原型怎样变成节点提示',31,INK,ha='left')
txt(45,86,'一套固定教师生成节点到类别的系数；共享类别提示据此形成不同节点的隐藏层修正。',20,GRAY,ha='left')
rr(35,117,1930,563,'#F6FAFA','#DCEAE8')
txt(73,151,'构建与修正类别原型',25,TEAL,ha='left')
# 1: original graph features and support means.
pos=np.array([[92,243],[184,227],[148,285],[85,328],[199,328]])
for i,j in [(0,1),(0,2),(1,2),(2,3),(2,4)]:line([pos[i],pos[j]])
for i,(x,y) in enumerate(pos):dot(x,y,COL[0] if i==0 else COL[1] if i==4 else '#B8C6D1',9,i in [0,4])
arrow((220,277),(262,277));mat(278,246,rows=4,cols=4,cell=16)
txt(276,205,'两跳属性传播',20,BLUE)
txt(286,345,'中心化、归一化',18,GRAY)
txt(186,386,'同类支持向量求平均',21)
for k,y in enumerate([445,530]):
 for x in [83,118,153]:dot(x,y,COL[k],9,True)
 arrow((180,y),(271,y),COL[k]);star(308,y,COL[k],520)
 txt(346,y,f'o{k+1}',19,COL[k])
txt(203,603,'支持原型始终作参照',19,GRAY)
# 2: assignment table.
arrow((365,280),(417,280),BLUE)
txt(662,201,'与当前类别原型比较',23)
star(738,251,TEAL,360);star(828,251,PURPLE,360)
txt(783,289,'初始取支持原型',16,GRAY)
txt(510,251,'无标签向量',18,GRAY)
for i,y in enumerate([313,385,457]):
 txt(440,y+16,f'u{i+1}',18,GRAY)
 mat(469,y,arr=R[i:i+1],cell=26,c=BLUE)
 arrow((562,y+16),(626,y+16),GRAY)
 bar(630,y,A[i],260,32,True)
txt(544,516,'相似度 → softmax',20,GRAY)
txt(780,516,'温度 0.05',19,TEAL)
txt(656,549,'每一行和为 1',20,TEAL)
txt(656,583,'一个节点可同时贡献给两类',18,GRAY)
# 3: same nodes, different weights and therefore different centers.
arrow((903,394),(934,394),TEAL)
txt(1174,201,'按类归一化，再求加权中心',23)
for k in range(2):
 base=np.array([990.,260.+k*210]);pp=D+base
 weight=A[:,k]/A[:,k].sum();center=(weight[:,None]*pp).sum(0)
 for i,p in enumerate(pp):
  arrow(p,center,COL[k],1.0+weight[i]*4)
  dot(*p,COL[k],np.sqrt(weight[i])*25)
  txt(p[0]-5,p[1]-29,f'u{i+1}',16,GRAY)
 star(*center,COL[k],540)
 txt(1327,base[1]+39,f'权重和 {A[:,k].sum():.1f}',18,COL[k])
 txt(1327,base[1]+78,'除以权重和',18,GRAY)
 txt(1327,base[1]+115,f'软原型 p{k+1}',19,COL[k])
 txt(1327,base[1]+155,' / '.join(f'{w*100:g}%' for w in weight.round(3)),16,GRAY)
# 4: blend original and soft prototype in equal proportions.
arrow((1439,394),(1480,394),TEAL)
txt(1730,201,'保留支持锚点，修正类中心',23)
txt(1550,263,'支持原型',19,GRAY);txt(1890,263,'软原型',19,GRAY)
for k,y in enumerate([347,505]):
 star(1550,y,COL[k],560,True);star(1890,y,COL[k],560)
 arrow((1580,y),(1694,y),COL[k]);arrow((1860,y),(1747,y),COL[k])
 txt(1627,y-32,'50%',20,COL[k]);txt(1820,y-32,'50%',20,COL[k])
 ax.scatter([1721],[y],s=550,c=[COL[k]],marker='D',edgecolors='white',lw=1.4,zorder=9)
 txt(1721,y+49,f'修正原型 c{k+1}',20,COL[k])
route([(1721,568),(1721,662),(405,662),(405,583),(435,583)],TEAL,1.5,True)
txt(1840,633,'重新分配 · 3 轮',18,TEAL)
# Bottom: softened/strong assignments -> centered signed local propagation -> composition -> insertion.
rr(35,719,1930,467,'#F7F9FC','#E0E7EF')
txt(73,755,'固定门控驱动共享提示',25,BLUE,ha='left')
# Refined class anchors, all-node probabilities.
for k,y in enumerate([882,968]):ax.scatter([105],[y],s=490,c=[COL[k]],marker='D',edgecolors='white',lw=1,zorder=6)
txt(120,1041,'3 轮后的\n固定原型',20,TEAL)
arrow((140,925),(205,925),TEAL)
txt(325,813,'让类别分配更集中',23)
base=np.array([[.70,.30],[.60,.40],[.20,.80]]);sharp=base**2/(base**2).sum(1,keepdims=True)
for i in range(3):bar(213,857+i*60,base[i],105,28);bar(370,857+i*60,sharp[i],105,28)
arrow((324,930),(363,930),TEAL)
txt(265,1050,'T = 0.2',18,GRAY);txt(423,1050,'T = 0.1',18,TEAL)
txt(340,1100,'原设置与锐化设置的对照',18,GRAY)
# signed before/after an original-graph aggregate
arrow((494,930),(534,930),TEAL)
txt(740,813,'去掉共同偏移，沿原图传播',23)
center=sharp-sharp.mean(0);op=np.array([[.7,.2,.1],[.2,.6,.2],[.1,.2,.7]]);q=op@center
for which,x0 in [(center,590),(q,858)]:
 for j in range(2):
  line([(x0+j*57,850),(x0+j*57,1007)],'#B5C2CC',.9)
  txt(x0+j*57,1038,'0',17,GRAY)
  for i in range(3):
   v=which[i,j]*95
   ax.add_patch(Rectangle((x0+j*57+min(v,0),860+i*60),abs(v),23,fc=COL[j],ec='none',zorder=5))
# Four-node miniature is the operator between signed matrices.
pp=np.array([[720,886],[755,919],[714,962],[787,965]])
for i,j in [(0,1),(1,2),(1,3),(2,3)]:line([pp[i],pp[j]],LINE,1.1)
for x,y in pp:dot(x,y,BLUE,6)
arrow((669,930),(699,930));arrow((805,930),(834,930))
txt(748,1100,'门控有正负，不再是概率',18,GRAY)
# Weighted shared-token composition; residual rescaling explicitly shown.
arrow((940,930),(984,930),TEAL)
txt(1134,813,'按节点组合类别提示',23)
mat(1005,862,rows=2,cols=7,cell=20,c='class');txt(1076,845,'可学习 P',16,ORANGE)
for j in range(2):line([(1050+j*57,919),(1159,979)],COL[j],1.7+j)
mat(1170,957,rows=3,cols=5,cell=16,c='class')
txt(1118,1057,'tanh 限幅 · 按隐藏尺度缩放',18,GRAY)
txt(1118,1100,'每个节点获得不同的修正',18,PURPLE)
# actual injection into same encoder, not three separate models.
txt(1603,813,'在第一层激活前改变隐藏表示',23)
encoder(1326,918,'第 1 层');mat(1441,936,rows=3,cols=3,cell=17)
arrow((1412,970),(1432,970));mat(1538,936,rows=3,cols=3,cell=17,c='class');arrow((1498,970),(1527,970),PURPLE)
route([(1210,957),(1210,874),(1563,874),(1563,932)],PURPLE,1.7)
line([(1611,990),(1624,975),(1639,942)],PURPLE,2.2)
arrow((1595,970),(1607,970));arrow((1644,970),(1674,970));encoder(1686,918,'第 2 层')
arrow((1779,970),(1818,970));mat(1835,936,rows=3,cols=4,cell=17,c='class')
txt(1630,1100,'两层主干权重冻结；P 由 M2 更新',18,BLUE)
# two compact exact formulas reserved for technical verification, not flow glyphs.
txt(1498,1156,r'$\Delta u_i=10\left(\sum_c Q_{ic}\tanh(P_c)\right)\odot s_U$',23,PURPLE,math=True)
txt(40,1221,'图中点数与维度为局部示意；软原型只使用无标签属性，查询节点真实标签不参与任何计算。',20,GRAY,ha='left')
fig.canvas.draw();rend=fig.canvas.get_renderer();(OUT/'M1_layout.json').write_text(json.dumps([{'text':t.get_text(),'bounds':list(t.get_window_extent(rend).bounds)} for t in ax.texts],ensure_ascii=False,indent=2))
(OUT/'M1_example.json').write_text(json.dumps({'unit_unlabeled_vectors':R.tolist(),'original_prototypes':[[1,0,0],[0,1,0]],'assignments':A.tolist(),'class_weight_sums':A.sum(0).tolist(),'soft_prototypes':soft.tolist(),'normalized_contributions':(A/A.sum(0)).tolist()},indent=2))
stem=OUT/'M1_类别语义引导与软原型'
for e in ['pdf','svg']:fig.savefig(stem.with_suffix('.'+e),facecolor='white')
fig.savefig(stem.with_suffix('.png'),dpi=300,facecolor='white');fig.savefig(OUT/(stem.name+'_preview.png'),dpi=100,facecolor='white');plt.close(fig)
print(stem)
