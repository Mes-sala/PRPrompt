from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, PathPatch
from matplotlib.path import Path as MPath
from matplotlib.font_manager import FontProperties
import numpy as np

OUT=Path(__file__).resolve().parent
FONT=FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc')
plt.rcParams.update({'mathtext.fontset':'stix','font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'path','axes.unicode_minus':False})
INK='#172C42';GRAY='#607187';EDGE='#ADBAC9';TEAL='#087F8C';BLUE='#426C9B';PURPLE='#7551A3';ORANGE='#AD632B';GREEN='#348361'
W,H=2100,1250

def draw(lang):
 fig=plt.figure(figsize=(21,12.5),facecolor='white');ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,W),ylim=(H,0));ax.axis('off')
 check=[]
 def tr(c,e):return c if lang=='zh' else e
 def text(x,y,s,size=28,color=INK,ha='center',weight='normal',math=False):
  t=ax.text(x,y,s,fontsize=size,color=color,ha=ha,va='center',fontproperties=None if math or lang=='en' else FONT,weight=weight,zorder=5)
  return t
 def rect(x,y,w,h,fc='white',ec=EDGE,r=15,lw=1.4):
  ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',facecolor=fc,edgecolor=ec,linewidth=lw,zorder=1))
 def panel(x,y,w,h,label,title,color,fill):
  rect(x,y,w,h,fill,color,20,1.6)
  rect(x+20,y+20,66,48,color,color,10)
  text(x+53,y+44,label,20 if label=='GNN' else 23,'white',math=True,weight='bold')
  text(x+106,y+45,title,29,color,ha='left')
 def arrow(points,color=GRAY,dashed=False,width=1.9):
  ax.plot([p[0] for p in points[:-1]],[p[1] for p in points[:-1]],color=color,lw=width,ls=(0,(5,4)) if dashed else '-',zorder=3)
  ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=19,linewidth=width,color=color,linestyle=(0,(5,4)) if dashed else '-',zorder=3,shrinkA=0,shrinkB=0))
 def formula(x,y,s,size=30,color=INK):return text(x,y,'$'+s+'$',size,color,math=True)
 def box(x,y,w,h,title,sub=None,color=GRAY,fill='white'):
  rect(x,y,w,h,fill,color)
  text(x+w/2,y+35,title,27,color)
  if sub: text(x+w/2,y+h-32,sub,23,GRAY)
 def nodes(cx,cy,scale=1):
  pts=np.array([[-45,-10],[-10,-38],[35,-15],[45,30],[-5,38],[-43,26]])*scale+[cx,cy]
  for i,j in [(0,1),(1,2),(0,4),(1,4),(2,3),(3,4),(4,5),(5,0),(2,4)]:
   ax.plot(pts[[i,j],0],pts[[i,j],1],color='#A1B8C7',lw=1.4,zorder=2)
  for i,p in enumerate(pts): ax.add_patch(Circle(p,8*scale,facecolor=['#248E9B','#DC9956','#8267B4','#D4E2E8','#248E9B','#D4E2E8'][i],edgecolor='white',lw=1.2,zorder=4))
 # Input banner
 rect(40,30,1310,100,'#F3F6FA','#CFD8E2',15)
 nodes(115,80,.73)
 text(215,64,tr('图属性与原图结构','Input graph'),29,ha='left')
 formula(710,66,r'G=(V,E,X)',31)
 text(1010,64,tr('支持标签','Support labels'),28)
 formula(1225,65,r'Y_S',32)
 text(790,105,tr('仅使用支持标签；查询标签不参与适配','Support labels only; no query labels'),23,GRAY)
 rect(1420,30,640,100,'#FAFBFD','#CFD8E2',15)
 text(1740,65,tr('一个冻结主干 · 一组可学习 token','Frozen backbone + task tokens'),26)
 text(1740,106,tr('转导少样本节点分类','Transductive few-shot classification'),23,GRAY)
 # Panels
 panel(40,180,540,1000,'M1',tr('语义门控锐化','Semantic gating'),TEAL,'#F0F9F9')
 panel(650,180,700,550,'GNN',tr('共享冻结主干','Frozen backbone'),BLUE,'#F3F7FC')
 panel(650,785,700,395,'M2',tr('更新方向与尺度匹配','Direction–scale matching'),ORANGE,'#FFF7EF')
 panel(1420,180,640,1000,'M3',tr('支持约束概率传播','Clamped propagation'),GREEN,'#F2F9F5')
 arrow([(310,130),(310,175)],TEAL);arrow([(1000,130),(1000,175)],BLUE)
 # M1
 box(80,280,460,130,tr('两跳属性传播与中心化','Propagate + center'),color=TEAL)
 formula(310,364,r'R=\widehat{A}^{\,2}X',33)
 arrow([(310,410),(310,448)],TEAL)
 box(80,450,460,180,tr('修正原型教师','Prototype refinement'),color=TEAL)
 text(310,523,tr('支持原型 + 无标签软原型','Support + soft prototypes'),24)
 formula(310,579,r'c^{(j+1)}=0.5\,o+0.5\,p^{(j)}',29)
 arrow([(310,630),(310,668)],TEAL)
 box(80,670,460,220,tr('锐化 → 中心化 → 传播','Sharpen + propagate'),color=TEAL)
 formula(310,757,r'\Pi_T=\operatorname{softmax}(0.2B/T)',29)
 formula(310,831,r'Q=\widehat{A}(\Pi_T-\overline{\Pi}_T)',33)
 text(310,933,tr('门控温度：0.2 → 0.1','Temperature: 0.2 → 0.1'),26,TEAL)
 # matrix icon signed gate
 colors=['#AEDAD7','#E9F4F3','#E7D3C3','#68BBB6','#F6E8DC']
 x0,y0=174,984
 for i in range(4):
  for j in range(5):ax.add_patch(Rectangle((x0+j*24,y0+i*23),20,19,facecolor=colors[(i*3+j*2)%5],edgecolor='none',zorder=2))
 formula(382,1027,r'Q\in\mathbb{R}^{N\times C}',30)
 text(310,1134,tr('固定教师与门控，不额外训练','Fixed teacher and gate'),24,TEAL)
 # Backbone
 box(690,280,620,95,tr('冻结 GCN 第一层','Frozen GCN layer 1'),color=BLUE)
 formula(1000,355,r'U=\mathrm{GCN}_1(X,E)',30)
 arrow([(1000,375),(1000,411)],BLUE)
 rect(690,415,620,170,'#F6F0FC',PURPLE,14,1.8)
 text(1000,450,tr('类别 token 注入与激活','Task-token injection'),28,PURPLE)
 formula(1000,508,r'\Delta U=10\,[Q\tanh(P)]\odot s_U',33)
 formula(1000,557,r'H=\mathrm{LeakyReLU}(U+\Delta U)',29)
 arrow([(1000,585),(1000,603)],BLUE)
 box(690,605,620,105,tr('冻结 GCN 第二层','Frozen GCN layer 2'),color=BLUE)
 formula(1000,690,r'Z_P=\mathrm{GCN}_2(H,E)',30)
 # Gate and train loop in separate gutters
 arrow([(540,831),(610,831),(610,500),(688,500)],TEAL,False,2.2)
 # M2
 rect(690,870,620,60,'white','#D8B494',10)
 formula(1000,899,r'\mathcal{L}=\mathcal{L}_{\mathrm{LOO}}+\lambda\,\mathrm{mean}(P\odot P)',29)
 arrow([(1000,711),(1000,780)],ORANGE,True)
 # clear title gap, gradient arrows loss to directions
 rect(690,963,325,85,'white','#D8B494',10)
 rect(1035,963,275,85,'white','#D8B494',10)
 formula(852,1005,r'D_t=L_t^{-1/4}G_tR_t^{-1/4}',28)
 formula(1172,1005,r's_t=\frac{\|A_t^{\mathrm{Adam}}\|_F}{\|D_t\|_F}',30)
 arrow([(850,930),(850,960)],ORANGE,True);arrow([(1170,930),(1170,960)],ORANGE,True)
 arrow([(850,1048),(850,1070)],ORANGE,True);arrow([(1170,1048),(1170,1070)],ORANGE,True)
 formula(1000,1093,r'P_{t+1}=P_t-\eta\,s_tD_t',35,ORANGE)
 text(1000,1148,tr('仅更新 P：零初始化，100 次更新','P only · zero init. · 100 steps'),24,ORANGE)
 arrow([(690,1093),(633,1093),(633,555),(688,555)],ORANGE,True,2.2)
 # M3
 box(1460,280,560,130,tr('最终表示中心化','Center embeddings'),color=GREEN)
 formula(1740,362,r'\overline{Z}=Z_P-\mathrm{mean}_V(Z_P)',31)
 arrow([(1310,668),(1385,668),(1385,345),(1458,345)],BLUE,False,2.2)
 # fork
 arrow([(1740,410),(1740,443),(1588,443),(1588,480)],GREEN)
 arrow([(1740,443),(1885,443),(1885,480)],GREEN)
 box(1460,485,260,190,tr('原型概率','Prototypes'),color=GREEN)
 formula(1590,571,r'F_0',43,GREEN)
 text(1590,630,tr('支持标签','Support labels'),24,GRAY)
 box(1750,485,270,190,tr('表示近邻图','50-NN graph'),color=GREEN)
 nodes(1885,569,.61)
 formula(1813,635,r'W',32,GREEN)
 text(1924,635,'50-NN',24,GREEN,math=True)
 arrow([(1590,675),(1590,714),(1740,714),(1740,751)],GREEN)
 arrow([(1885,675),(1885,714),(1740,714)],GREEN)
 rect(1460,755,560,245,'white',GREEN,14,1.8)
 text(1740,792,tr('传播并固定支持标签','Clamped diffusion'),27,GREEN)
 formula(1740,858,r'F^{(t+1)}=0.1F_0+0.9WF^{(t)}',33)
 formula(1740,920,r'F_S^{(t+1)}=Y_S',35,GREEN)
 text(1740,973,tr('初始化及每轮钳制 · 20 轮','Clamp from initialization · 20 steps'),22,GRAY)
 arrow([(1740,1000),(1740,1041)],GREEN)
 rect(1460,1045,560,95,'#E3F1E9',GREEN,12)
 text(1554,1092,tr('预测','Predict'),27,GREEN)
 formula(1814,1092,r'\widehat y_i=\arg\max_c F_{ic}^{(20)}',33,GREEN)
 # Legend clean separated
 arrow([(65,1220),(143,1220)],BLUE)
 text(165,1220,tr('前向数据流','Forward flow'),23,GRAY,ha='left')
 arrow([(470,1220),(550,1220)],ORANGE,True)
 text(573,1220,tr('训练更新','Update'),23,GRAY,ha='left')
 text(990,1220,tr('P：唯一可学习参数','P: trainable tokens'),23,PURPLE)
 text(1740,1220,tr('M3 仅用于最终读出，无反向传播','Readout only; no backpropagation'),23,GREEN)
 # Exports: keep vector paths; provide PNG at 300 dpi + compact preview.
 fig.canvas.draw()
 import json
 renderer=fig.canvas.get_renderer()
 text_bounds=[{'text':t.get_text(),'bounds':list(t.get_window_extent(renderer).bounds)} for t in ax.texts]
 (OUT/('layout_'+lang+'.json')).write_text(json.dumps(text_bounds,ensure_ascii=False,indent=2))
 prefix=OUT/('主架构图_中文' if lang=='zh' else 'Main_architecture_EN')
 for ext in ['svg','pdf']:fig.savefig(prefix.with_suffix('.'+ext),facecolor='white')
 fig.savefig(prefix.with_suffix('.png'),dpi=300,facecolor='white')
 fig.savefig(prefix.with_name(prefix.name+'_预览').with_suffix('.png'),dpi=100,facecolor='white')
 plt.close(fig)
 return str(prefix)
if __name__=='__main__':
 print(draw('zh'));print(draw('en'))
