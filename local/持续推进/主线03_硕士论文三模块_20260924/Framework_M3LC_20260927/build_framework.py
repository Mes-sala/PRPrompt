from pathlib import Path
import math, json, xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.font_manager import FontProperties
OUT=Path(__file__).resolve().parent
FONT=FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc')
plt.rcParams.update({'svg.fonttype':'none','font.family':'DejaVu Sans','mathtext.fontset':'stix'})
INK='#162F43'; MUTED='#567084'; TEAL='#008D87'; BLUE='#267AC0'; ORANGE='#C57A20'; PURPLE='#8952BB'; LINE='#A2B8C7'
class Canvas:
 def __init__(self,name,lang):
  self.name=name;self.lang=lang;self.cells=[];self.texts=[]
  self.fig=plt.figure(figsize=(19,10.5),facecolor='white');self.ax=self.fig.add_axes([0,0,1,1]);self.ax.set(xlim=(0,1900),ylim=(1050,0));self.ax.axis('off')
 def tr(self,a,b):return a if self.lang=='CN' else b
 def box(self,x,y,w,h,fill='white',color=LINE,r=16):
  self.ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',fc=fill,ec=color,lw=1.2,zorder=1));self.cells.append(('box',x,y,w,h,fill,color))
 def text(self,x,y,s,size=20,color=INK,ha='center',bold=False):
  t=self.ax.text(x,y,s,fontsize=size,color=color,ha=ha,va='center',fontproperties=FONT if self.lang=='CN' and '$' not in s else None,weight='bold' if bold else 'normal',linespacing=1.5,zorder=5)
  self.texts.append(t);self.cells.append(('text',x,y,s,size,color,ha,bold))
 def arrow(self,pts,color=MUTED,dash=False):
  if len(pts)>2:self.ax.plot(*zip(*pts[:-1]),color=color,lw=1.8,ls='--' if dash else '-',zorder=3)
  self.ax.add_patch(FancyArrowPatch(pts[-2],pts[-1],arrowstyle='-|>',mutation_scale=17,lw=1.8,color=color,ls='--' if dash else '-',shrinkA=0,shrinkB=2,zorder=3));self.cells.append(('arrow',pts,color,dash))
 def panel(self,x,w,n,title,color,fill):
  self.box(x,160,w,785,fill,color);self.box(x+20,180,48,40,color,color,8);self.text(x+44,200,n,17,'white',bold=True);self.text(x+85,200,title,23,color,ha='left',bold=True)
 def node(self,x,y,c,r=11):
  self.ax.add_patch(Circle((x,y),r,fc=c,ec='white',lw=1.5,zorder=4));self.cells.append(('circle',x-r,y-r,2*r,2*r,c))
 def graph(self,x,y):
  pos=[(x-65,y-25),(x,y-60),(x+65,y-20),(x+45,y+45),(x-30,y+45)]
  for a,b in [(0,1),(1,2),(2,3),(3,4),(4,0),(1,4)]:
   self.ax.plot([pos[a][0],pos[b][0]],[pos[a][1],pos[b][1]],color=LINE,lw=2,zorder=2);self.cells.append(('edge',[pos[a],pos[b]],LINE,False))
  for k,(xx,yy) in enumerate(pos):self.node(xx,yy,[TEAL,TEAL,PURPLE,PURPLE,TEAL][k])
 def bar(self,x,y,p,w=160):
  self.box(x,y,w*p,24,TEAL,TEAL,0);self.box(x+w*p,y,w*(1-p),24,PURPLE,PURPLE,0)
 def save(self):
  self.fig.canvas.draw();renderer=self.fig.canvas.get_renderer();outside=[]
  for text in self.texts:
   x,y=text.get_position();boxes=[v for v in self.cells if v[0]=='box' and v[1]<=x<=v[1]+v[3] and v[2]<=y<=v[2]+v[4]]
   if boxes:
    box=min(boxes,key=lambda v:v[3]*v[4]);pad=4 if box[3]<70 else 15;left,right=box[1]+pad,box[1]+box[3]-pad
    available=right-x if text.get_ha()=='left' else 2*min(x-left,right-x)
    b=text.get_window_extent(renderer).transformed(self.ax.transData.inverted())
    if b.width>available and available>0:text.set_fontsize(text.get_fontsize()*available/b.width)
  self.fig.canvas.draw()
  rendered=iter(self.texts)
  for index,item in enumerate(self.cells):
   if item[0]=='text':
    text=next(rendered);v=list(item);v[4]=text.get_fontsize();self.cells[index]=tuple(v)
  for t in self.texts:
   b=t.get_window_extent(renderer).transformed(self.ax.transData.inverted());
   if b.x0<0 or b.x1>1900 or min(b.y0,b.y1)<0 or max(b.y0,b.y1)>1050:outside.append(t.get_text())
  assert not outside,outside
  stem=OUT/f'{self.name}_{self.lang}'
  self.fig.savefig(str(stem)+'.svg');self.fig.savefig(str(stem)+'.png',dpi=160);self.fig.savefig(str(stem)+'_preview.png',dpi=80)
  root=ET.Element('mxfile',host='app.diagrams.net');dia=ET.SubElement(root,'diagram',name=self.name+' '+self.lang);model=ET.SubElement(dia,'mxGraphModel',pageWidth='1900',pageHeight='1050',math='1');rr=ET.SubElement(model,'root');ET.SubElement(rr,'mxCell',id='0');ET.SubElement(rr,'mxCell',id='1',parent='0')
  for i,item in enumerate(self.cells,2):
   typ=item[0];style='';value='';geo={};edge=False
   if typ=='box':
    _,x,y,w,h,fc,ec=item;style=f'rounded=1;arcSize=12;fillColor={fc};strokeColor={ec};';geo=dict(x=x,y=y,width=w,height=h)
   elif typ=='circle':
    _,x,y,w,h,fc=item;style=f'ellipse;fillColor={fc};strokeColor=#FFFFFF;';geo=dict(x=x,y=y,width=w,height=h)
   elif typ=='text':
    _,x,y,s,sz,col,ha,bold=item
    value=s.replace('\n','<br>');import re
    value=re.sub(r'\$([^$]+)\$',lambda m:r'\('+m.group(1)+r'\)',value)
    w=1800 if ha=='left' else 1600;h=sz*2.2*(s.count('\n')+1);geo=dict(x=x if ha=='left' else x-w/2,y=y-h/2,width=w,height=h)
    style=f'text;html=1;whiteSpace=nowrap;overflow=visible;fillColor=none;strokeColor=none;align={ha};fontFamily={"Heiti SC" if self.lang=="CN" else "Helvetica"};fontSize={sz*100/72};fontColor={col};fontStyle={1 if bold else 0};'
   else:
    _,pts,col,dash=item;edge=True;style=f'edgeStyle=none;endArrow={"block" if typ=="arrow" else "none"};strokeColor={col};strokeWidth=2;dashed={1 if dash else 0};'
   cell=ET.SubElement(rr,'mxCell',id=str(i),parent='1',value=value,style=style,**({'edge':'1'} if edge else {'vertex':'1'}));g=ET.SubElement(cell,'mxGeometry',**{k:str(v) for k,v in geo.items()},**{'as':'geometry'})
   if edge:
    g.set('relative','1');ET.SubElement(g,'mxPoint',x=str(pts[0][0]),y=str(pts[0][1]),**{'as':'sourcePoint'});ET.SubElement(g,'mxPoint',x=str(pts[-1][0]),y=str(pts[-1][1]),**{'as':'targetPoint'});arr=ET.SubElement(g,'Array',**{'as':'points'})
    for x,y in pts[1:-1]:ET.SubElement(arr,'mxPoint',x=str(x),y=str(y))
  ET.ElementTree(root).write(str(stem)+'.drawio',encoding='utf-8',xml_declaration=True)
  plt.close(self.fig)

def m3(lang):
 c=Canvas('M3_LC',lang);t=c.tr
 c.text(55,49,t('M3-LC  |  局部一致性校准读出','M3-LC  |  Local-consistency calibrated readout'),31,ha='left',bold=True)
 c.text(55,106,t('分类分数来自 Prompt；近邻关系来自教师属性；温度由初始预测的一致性决定','Prompt scores + teacher-attribute neighbors + temperatures from initial prediction agreement'),20,MUTED,ha='left')
 c.panel(45,470,'01',t('分数与关系分工','Scores and relations'),BLUE,'#F1F7FC')
 c.panel(555,700,'02',t('一次性局部温度调整','One-time local temperature'),TEAL,'#EFF9F6')
 c.panel(1295,560,'03',t('支持锚定传播','Support-clamped diffusion'),PURPLE,'#F7F3FC')
 # input
 c.box(75,252,410,157);c.text(280,282,t('Prompt 适配表示','Prompt representations'),22,BLUE);c.text(280,330,r'$Z_P\ \longrightarrow\ \mathrm{prototype\ scores}\ s$',24);c.text(280,379,t('支持原型给出类别分数','Class scores from support prototypes'),18,MUTED)
 c.box(75,462,410,397);c.text(280,496,t('教师构图属性','Teacher graph features'),22,BLUE);c.text(280,540,t('中心化两跳传播属性','Centered two-hop attributes'),20);c.graph(280,636);c.text(280,742,t('余弦 50-NN · 排除自身','Cosine 50-NN · exclude self'),19);c.text(280,787,t('邻居 softmax 温度 0.2','Neighbor softmax temperature 0.2'),18);c.text(280,830,r'$W\mathbf{1}=\mathbf{1}$',24)
 # agree
 c.box(590,252,630,180);c.text(905,285,t('属性邻居是否支持当前预测？','Do attribute neighbors agree?'),22,TEAL)
 c.text(905,337,r'$c_i=\arg\max_c s_{ic}$',25);c.text(905,392,r'$a_i=\sum_j W_{ij}\,\mathbf{1}[c_j=c_i]$',27)
 c.arrow([(485,320),(530,320),(530,313),(588,313)],BLUE);c.arrow([(485,795),(535,795),(535,390),(588,390)],BLUE)
 c.arrow([(905,432),(905,477)],TEAL)
 c.box(590,480,630,139);c.text(905,515,t('邻域支持越少，概率越软','Less agreement → softer probabilities'),22,TEAL);c.text(905,571,r'$T_i=1+3(1-a_i),\qquad T_i\in[1,4]$',27)
 c.arrow([(905,619),(905,658)],TEAL)
 c.box(590,660,630,109);c.text(905,692,t('只调整置信度，初始类别不变','Change confidence, preserve initial class'),20);c.text(905,739,r'$P_i^{(0)}=\operatorname{softmax}(s_i/T_i)$',27)
 c.text(905,815,t('示意：相同初始概率 [0.90, 0.10]','Illustration: same initial probabilities [0.90, 0.10]'),17,MUTED)
 c.text(615,860,'a = 1.0  →  T = 1.0',17,ha='left');c.bar(997,848,.9,185)
 c.text(615,909,'a = 0.0  →  T = 4.0',17,ha='left');c.bar(997,897,.634,185)
 # propagation
 c.box(1330,252,490,135);c.text(1575,282,t('初始化与支持标签锚定','Initialize and clamp support'),22,PURPLE);c.text(1575,329,r'$P^{[0]}=P^{(0)},\quad P_S^{[0]}=Y_S$',25);c.text(1575,366,t('初始概率副本固定已知标签','Clamp a copy of the initial probabilities'),17,MUTED)
 c.arrow([(1220,713),(1275,713),(1275,326),(1328,326)],TEAL)
 c.arrow([(1575,387),(1575,455)],PURPLE)
 c.box(1330,458,490,220);c.text(1575,493,t('同一 W 传播 + 初始概率回注','Diffuse on W + reinject probabilities'),20,PURPLE);c.text(1575,550,r'$P^{[t+1]}=0.1P^{(0)}+0.9WP^{[t]}$',25);c.text(1575,608,r'$P_S^{[t+1]}=Y_S$',27);c.text(1575,650,t('每一步均重新固定支持标签','Re-clamp support labels after every step'),17,MUTED)
 c.arrow([(1820,630),(1838,630),(1838,430),(1760,430),(1760,456)],PURPLE);c.text(1723,422,t('重复 20 步','20 steps'),17,PURPLE)
 c.arrow([(1575,678),(1575,753)],PURPLE)
 c.box(1330,756,490,125,'#EEE5F9',PURPLE);c.text(1575,792,t('查询节点最终预测','Final query predictions'),22,PURPLE);c.text(1575,844,r'$\widehat y_i=\arg\max_c P_{ic}^{[20]}$',27)
 c.text(55,991,t('固定读出规则  •  不新增可训练参数  •  温度只计算一次  •  仅使用支持标签','Fixed readout  •  No added trainable parameters  •  Temperature computed once  •  Support labels only'),20,MUTED,ha='left');c.save()

def overall(lang):
 c=Canvas('CGRPrompt_Framework',lang);t=c.tr
 c.text(50,48,t('CGRPrompt：类别引导残差提示','CGRPrompt: Class-Guided Residual Prompt'),32,ha='left',bold=True)
 c.text(50,104,t('同一个提示结构：类别条件组织残差，矩阵更新学习残差，局部一致性约束预测读出','One prompt: class-conditioned residuals, matrix-aware learning, and consistency-calibrated readout'),20,MUTED,ha='left')
 c.panel(40,480,'M1',t('构造类别条件','Class conditioning'),TEAL,'#EFF9F6')
 c.panel(565,695,'P',t('共享类别残差与冻结主干','Shared residuals + frozen GNN'),BLUE,'#F1F7FC')
 c.panel(1305,555,'M3',t('局部一致性读出','Consistency readout'),PURPLE,'#F7F3FC')
 c.box(70,250,420,110);c.text(280,282,t('图属性、原图、支持标签','Features, graph, support labels'),21,TEAL);c.text(280,330,r'$X,\ E,\ Y_S$',26)
 c.arrow([(280,360),(280,395)],TEAL)
 c.box(70,398,420,107);c.text(280,430,t('两跳属性传播与中心化','Two-hop propagation + centering'),20);c.text(280,475,t('固定教师属性 R','Fixed teacher attributes R'),20,TEAL)
 c.arrow([(280,505),(280,551)],TEAL)
 c.box(70,554,420,135);c.text(280,590,t('支持原型 + 无标签软原型','Support + unlabeled soft prototypes'),19);c.text(280,642,t('精炼类别原型 → 教师分配','Refine prototypes → assignments'),20)
 c.arrow([(280,689),(280,732)],TEAL)
 c.box(70,735,420,145);c.text(280,770,t('锐化 → 中心化 → 原图传播','Sharpen → center → graph propagation'),18,TEAL);c.text(280,823,r'$Q=\widehat A(\Pi_T-\overline{\Pi}_T)$',27);c.text(280,858,t('固定的节点—类别组合系数','Fixed node–class mixture coefficients'),17,MUTED)
 # backbone main
 c.box(600,250,625,82);c.text(912,276,t('冻结 GCN 第一层','Frozen GCN layer 1'),21,BLUE);c.text(912,310,r'$U=\operatorname{GCN}_1(X,E)$',23)
 c.arrow([(490,282),(598,282)],BLUE)
 c.arrow([(912,332),(912,368)],BLUE)
 c.box(600,371,625,148,'#ECE7F7',PURPLE);c.text(912,405,t('一个可学习类别残差矩阵 P','One trainable class-residual matrix P'),22,PURPLE);c.text(912,454,r'$\Delta U=10[Q\tanh(P)]\odot s_U$',28);c.text(912,497,r'$P\in\mathbb{R}^{C\times d}\ ;\quad P_{\rm init}=0$',23)
 c.arrow([(490,812),(550,812),(550,448),(598,448)],TEAL)
 c.arrow([(912,519),(912,554)],BLUE)
 c.box(600,557,625,105);c.text(912,586,t('激活 + 冻结 GCN 第二层','Activation + frozen GCN layer 2'),21,BLUE);c.text(912,632,r'$Z_P=\operatorname{GCN}_2(\sigma(U+\Delta U),E)$',25)
 # M2
 c.box(600,711,625,208,'#FFF4E5',ORANGE);c.text(623,742,t('M2  提示学习：方向与步幅分工','M2  Prompt learning: direction + scale'),22,ORANGE,ha='left')
 c.text(912,789,t('支持集损失 → 同一 P 的梯度','Support loss → gradient of the same P'),20)
 c.text(912,836,r'$\widetilde D=\frac{\|A_{\mathrm{Adam}}\|_F}{\|D_{\mathrm{Shampoo}}\|_F}D_{\mathrm{Shampoo}}$',26)
 c.text(912,889,r'$P\leftarrow P-\eta\widetilde D$',25,ORANGE)
 c.arrow([(912,662),(912,710)],ORANGE,True)
 c.arrow([(600,870),(579,870),(579,493),(598,493)],ORANGE,True)
 # relation route above center panel via title-safe gutter
 c.arrow([(490,466),(530,466),(530,143),(1840,143),(1840,280),(1832,280)],TEAL)
 # M3
 c.box(1335,250,495,113);c.text(1582,280,t('教师属性近邻图 W','Teacher-attribute neighbor graph W'),20,PURPLE);c.text(1582,324,t('50-NN · 余弦 · softmax 0.2','50-NN · cosine · softmax 0.2'),19);c.text(1582,349,t('去自环，行归一化','No self-edges; row-normalized'),16,MUTED)
 c.box(1335,409,495,90);c.text(1582,438,t('Prompt 支持原型分类分数','Prompt support-prototype scores'),20,BLUE);c.text(1582,479,r'$s,\quad c_i=\arg\max_c s_{ic}$',24)
 c.arrow([(1225,610),(1281,610),(1281,453),(1333,453)],BLUE)
 c.arrow([(1582,363),(1850,363),(1850,599),(1832,599)],PURPLE)
 c.arrow([(1582,499),(1582,544)],PURPLE)
 c.box(1335,547,495,182);c.text(1582,577,t('局部一致性 → 节点温度','Local agreement → node temperature'),20,PURPLE);c.text(1582,624,r'$a_i=\sum_jW_{ij}\mathbf{1}[c_j=c_i]$',25);c.text(1582,669,r'$T_i=1+3(1-a_i)$',25);c.text(1582,707,r'$P_i^{(0)}=\operatorname{softmax}(s_i/T_i)$',24)
 c.arrow([(1582,729),(1582,764)],PURPLE)
 c.box(1335,767,495,146,'#EEE5F9',PURPLE);c.text(1582,798,t('支持锚定传播 → 最终预测','Clamped diffusion → predictions'),21,PURPLE);c.text(1582,842,r'$P^{[t+1]}=0.1P^{(0)}+0.9WP^{[t]}$',24);c.text(1582,884,t('初始化及每步固定支持标签 · 20 步','Clamp support initially and each step · 20 steps'),17)
 c.text(50,991,t('实线：前向数据流    虚线：学习与参数更新    仅 P 可训练；教师条件、读出与预训练主干保持固定','Solid: forward flow    Dashed: learning / updates    Only P is trainable; teacher, readout rule and backbone are fixed'),19,MUTED,ha='left');c.save()

for lang in ['CN','EN']:m3(lang);overall(lang)
print('Built 4 figures: PNG, SVG, draw.io and previews.')
