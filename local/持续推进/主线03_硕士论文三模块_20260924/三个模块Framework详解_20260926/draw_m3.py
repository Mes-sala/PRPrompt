from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Rectangle, Wedge, FancyArrowPatch, FancyBboxPatch, Polygon
from matplotlib.font_manager import FontProperties
from matplotlib.colors import to_rgb

OUT = Path(__file__).resolve().parent
FONT = FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc')
plt.rcParams.update({'pdf.fonttype': 42, 'svg.fonttype': 'path', 'font.family': 'DejaVu Sans', 'mathtext.fontset': 'stix'})
INK='#24394B'; GRAY='#667D90'; LINE='#A2B5C2'; BLUE='#5683A9'; TEAL='#238D89'; PURPLE='#926FB1'; ORANGE='#B56B3D'
COL=[TEAL,PURPLE]; W=2000; H=1250

# A numerical illustration only: five local nodes and k=3, not the full k=50 experiment.
E=np.array([[.8,.6,0,0,0],[.6,0,.8,0,0],[1,0,0,0,0],[.4,0,0,np.sqrt(.84),0],[.1,0,0,0,np.sqrt(.99)]])
SIM=E@E.T; np.fill_diagonal(SIM,-np.inf)
IX=np.argsort(-SIM,axis=1)[:,:3]
SV=np.take_along_axis(SIM,IX,axis=1)
WW=np.exp(SV/.2-SV.max(1,keepdims=True)/.2);WW/=WW.sum(1,keepdims=True)
OP=np.zeros((5,5));OP[np.arange(5)[:,None],IX]=WW
RAW=np.array([[.70,.30],[.40,.60],[.55,.45],[.20,.80],[.45,.55]])
SUP=[0,3]; INIT=RAW.copy();INIT[SUP]=[[1,0],[0,1]]
AVG=OP@INIT; FIRST=.1*RAW+.9*AVG;FIRST[SUP]=[[1,0],[0,1]]
FINAL=INIT.copy()
for _ in range(20):
    FINAL=.1*RAW+.9*OP@FINAL;FINAL[SUP]=[[1,0],[0,1]]
assert np.allclose(WW.sum(1),1) and np.allclose(FINAL.sum(1),1)
assert np.array_equal(FINAL[SUP],INIT[SUP])

fig=plt.figure(figsize=(20,12.5),facecolor='white')
ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,W),ylim=(H,0));ax.axis('off')

def txt(x,y,s,fs=23,col=INK,ha='center',z=12):
    return ax.text(x,y,s,fontsize=fs,color=col,ha=ha,va='center',fontproperties=FONT,zorder=z)
def line(pts,col=LINE,lw=1.6,dash=False,z=2):
    ax.plot(*np.array(pts).T,color=col,lw=lw,ls=(0,(4,3)) if dash else '-',zorder=z)
def arrow(a,b,col=GRAY,lw=1.8,dash=False,rad=0,z=5,ms=17):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=ms,linewidth=lw,color=col,linestyle=(0,(5,4)) if dash else '-',connectionstyle=f'arc3,rad={rad}',shrinkA=0,shrinkB=2,zorder=z))
def route(pts,col=GRAY,lw=1.8,dash=False,z=3):
    line(pts[:-1],col,lw,dash,z);arrow(pts[-2],pts[-1],col,lw,dash,z=z)
def rr(x,y,w,h,fc,ec='none',r=18,lw=1):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',fc=fc,ec=ec,lw=lw,zorder=0))
def pie(x,y,p,r=16,support=False,ec=LINE):
    a=90
    for k,v in enumerate(p):
        if v>0:ax.add_patch(Wedge((x,y),r,a,a+360*v,fc=COL[k],ec='white',lw=.7,zorder=6))
        a+=360*v
    ax.add_patch(Circle((x,y),r,fc='none',ec=ec,lw=.8,zorder=7))
    if support:ax.add_patch(Circle((x,y),r+5,fc='none',ec=INK,lw=1.3,zorder=7))
def bar(x,y,p,w=160,h=24,numbers=False,fs=17):
    cum=0
    for k,v in enumerate(p):
        if v>0:
            ax.add_patch(Rectangle((x+cum*w,y),v*w,h,fc=COL[k],ec='white',lw=.6,zorder=5))
            if numbers and v>.10:txt(x+(cum+v/2)*w,y+h/2,f'{v:.2f}',fs,'white')
        cum+=v
    ax.add_patch(Rectangle((x,y),w,h,fc='none',ec=LINE,lw=.7,zorder=6))
def snow(x,y,r=10):
    for a in [0,np.pi/3,2*np.pi/3]:
        v=r*np.array([np.cos(a),np.sin(a)]);line([[x-v[0],y-v[1]],[x+v[0],y+v[1]]],BLUE,1.6,z=4)
def star(x,y,col,s=210):ax.scatter([x],[y],marker='*',s=s,c=[col],edgecolors=INK,lw=.6,zorder=7)
def cloud(cx,cy,normalized=False):
    p=np.array([[-47,-31],[-25,20],[0,-40],[23,-20],[53,17],[-6,44],[24,35],[-52,6]])
    if normalized:
        ax.add_patch(Circle((cx,cy),62,fc='none',ec='#CAD8E2',lw=1,zorder=1))
        line([(cx-71,cy),(cx+72,cy)],'#CAD8E2',.8);line([(cx,cy-69),(cx,cy+68)],'#CAD8E2',.8)
        p=p/np.linalg.norm(p,axis=1,keepdims=True)*54
    else:
        line([(cx-71,cy+60),(cx+72,cy+60)],'#CAD8E2',.8);line([(cx-71,cy+60),(cx-71,cy-65)],'#CAD8E2',.8)
    for i,(dx,dy) in enumerate(p):ax.add_patch(Circle((cx+dx,cy+dy),6.7,fc=COL[0] if i<4 else COL[1],ec='white',lw=.6,zorder=5))
    if normalized:ax.add_patch(Circle((cx,cy),3,fc=INK,zorder=5))
def fixed_row(x,y,label,p,raw=None):
    txt(x,y+12,label,19,GRAY,ha='right');bar(x+17,y,p,155,24,True)
    if raw is not None:
        bar(x+17,y-30,raw,155,15,False)
        arrow((x+95,y-12),(x+95,y-2),ORANGE,1.4,ms=12)

# Main heading and compact legend.
txt(60,52,'M3  支持约束概率传播',31,INK,ha='left')
txt(60,98,'训练后的读出：以表示建立关系，让类别概率沿近邻传递',20,GRAY,ha='left')
pie(1620,49,[1,0],11,True);txt(1644,49,'支持节点',18,GRAY,ha='left')
pie(1790,49,[.55,.45],13);txt(1812,49,'类别概率',18,GRAY,ha='left')
line([(60,127),(1940,127)],'#DCE5EB',1)

# One representation source, two preparation paths.
cloud(145,295);txt(145,188,'最终适配表示',23);txt(145,402,'最终提示重新前向',18,GRAY)
arrow((220,293),(281,293),BLUE)
cloud(370,293,True);txt(370,189,'共同中心化',23);txt(370,402,'逐节点归一化',20,BLUE)
txt(257,449,'两路共用同一份表示',19,GRAY)
route([(449,293),(480,293),(480,247),(548,247)],BLUE,1.7)
route([(449,293),(480,293),(480,550),(548,550)],BLUE,1.7)

# Upper path: supports --> prototype --> cosine scores --> original probabilities.
txt(698,172,'支持均值原型',23,TEAL)
for k,yy in enumerate([227,298]):
    for dx,dy in [(0,-10),(22,14),(45,-4)]:pie(575+dx,yy+dy,[1,0] if k==0 else [0,1],8,True)
    arrow((646,yy),(691,yy),COL[k],1.5)
    star(719,yy,COL[k],300)
txt(704,372,'每类平均后归一化',18,GRAY)
arrow((763,261),(828,261),TEAL)
txt(965,172,'余弦相似度',23)
# Small matching picture: a node tests similarity against both prototypes.
ax.add_patch(Circle((856,261),14,fc='#B8C9D5',ec=BLUE,lw=1,zorder=6))
for k,yy in enumerate([223,301]):
    line([(873,261),(915,yy)],COL[k],1.5)
    star(925,yy,COL[k],225)
    ax.add_patch(Rectangle((953,yy-10),150*(.7 if k==0 else .4),20,fc=COL[k],alpha=.55,ec='none',zorder=4))
txt(975,372,'softmax 温度 0.2',18,GRAY)
arrow((1115,261),(1170,261),TEAL)
txt(1301,172,'原始概率 $F_0$',24,TEAL)
for r,p in enumerate(RAW):
    txt(1187,208+r*30,['a','b','v','c','d'][r],18,GRAY,ha='right');bar(1202,195+r*30,p,197,24)
txt(1303,372,'保留原值，每轮回注',18,TEAL)
route([(1416,261),(1472,261),(1472,334),(1524,334)],ORANGE,1.6)
txt(1701,171,'初始化传播状态',23,ORANGE)
fixed_row(1585,235,'a',[1,0],RAW[0]);fixed_row(1585,331,'c',[0,1],RAW[3])
txt(1707,395,'复制 $F_0$；支持行换成已知标签',18,GRAY)
for r,p in enumerate(INIT):bar(1828,205+r*28,p,104,20)
txt(1880,365,r'$F^{(0)}$',23,ORANGE)
arrow((1776,277),(1815,277),ORANGE,1.7)

# Lower preparation: local search, removal of self, 50 neighbors, W.
txt(693,476,'从表示找近邻',23,BLUE)
loc=np.array([[579,550],[641,510],[698,575],[753,526],[779,610]],float)
for k,(x,y) in enumerate(loc):
    ax.add_patch(Circle((x,y),10 if k==2 else 7,fc=INK if k==2 else '#AFC5D5',ec='white',lw=.8,zorder=4))
    if k!=2:line([loc[2],(x,y)],BLUE,.9,dash=True,z=1)
txt(702,612,'v',18,INK);txt(677,657,'原图边不参与选择',18,GRAY)
arrow((814,550),(854,550),BLUE)
txt(1010,476,'排自身 · 余弦排序',23,BLUE)
ranking=[('v',1.0),('a',.8),('b',.6),('c',.4)]
for r,(lab,val) in enumerate(ranking):
    yy=512+r*34
    txt(885,yy+8,lab,18,GRAY)
    ax.add_patch(Rectangle((906,yy),130*val,17,fc='#CAD4DC' if r==0 else BLUE,alpha=.55 if r==0 else .8,ec='none',zorder=4))
    txt(1075,yy+8,f'{val:.2f}',18,GRAY)
    if r==0:
        line([(890,yy-1),(1106,yy+22)],ORANGE,1.5,z=8)
        txt(1131,yy+8,'×',23,ORANGE)
txt(1020,657,'实际取前 50；示例仅取 3',18,GRAY)
arrow((1162,565),(1217,565),BLUE)
txt(1353,476,'相似度转权重 W',23,BLUE)
for r,(lab,v) in enumerate(zip(['a','b','c'],WW[2])):
    yy=528+r*43
    txt(1237,yy+10,lab,18,GRAY)
    ax.add_patch(Rectangle((1261,yy),190*v,22,fc=BLUE,ec='none',zorder=4))
    txt(1491,yy+11,f'{v:.3f}',18,BLUE)
txt(1370,657,'softmax 0.2；每行和为 1',18,GRAY)
arrow((1529,565),(1588,565),BLUE)
# The directed neighborhood is a computed data object, not the input graph.
gp=np.array([[1620,526],[1755,518],[1692,587],[1783,622],[1608,637]],float)
for j,lw in zip([0,1,3],[3.3,2.,1.]):
    a=gp[j];b=gp[2];unit=(b-a)/np.linalg.norm(b-a);arrow(a+unit*11,b-unit*14,BLUE,lw,ms=13)
for k,(x,y) in enumerate(gp):ax.add_patch(Circle((x,y),10,fc='#D4E3EE' if k!=2 else BLUE,ec='white',lw=1,zorder=5))
txt(1861,561,'有向\n近邻图',21,BLUE)
txt(1718,680,'建立后保持固定',18,GRAY)

# A single magnifying inset explains the repeated probability operation.
rr(60,712,1880,456,'#F7FAFC','#D6E3EB',18,1)
txt(91,751,'放大一次更新',25,INK,ha='left')
txt(388,752,'下面始终传播类别概率，表示与 W 均不再更新',19,GRAY,ha='left')

# Neighbor distributions and exact weighted average for target v.
txt(192,816,'邻居当前概率',22)
for r,j in enumerate([0,1,3]):
    yy=860+r*75
    pie(112,yy,INIT[j],17,j in SUP)
    txt(146,yy,['a','b','c'][r],18,GRAY)
    bar(174,yy-13,INIT[j],147,26,True,17)
    txt(368,yy,f'{WW[2,r]:.3f}',18,BLUE)
    route([(407,yy),(437,yy),(485,935)],BLUE,1.1+WW[2,r]*2.8)
bar(507,921,AVG[2],201,30,True,18)
txt(607,988,'邻居加权概率',20,BLUE)
txt(607,855,'W 的权重',19,BLUE)

# Restart mixture: preserved F0 and evolving neighbor probabilities merge visually.
txt(816,822,'原始 $F_0(v)$',21,TEAL)
bar(735,853,RAW[2],162,25,True,17)
txt(809,898,'始终不变',18,TEAL)
route([(896,865),(934,865),(980,929)],TEAL,1.3)
txt(965,863,'10%',19,TEAL)
route([(722,936),(823,936),(978,944)],BLUE,3.3)
txt(847,968,'90%',20,BLUE)
bar(1000,922,FIRST[2],177,31,True,18)
txt(1087,993,'v 的新概率',21)
txt(1087,1040,'0.742 / 0.258',18,GRAY)

# Support reset applies to the full probability state after the mixed update.
arrow((1191,937),(1244,937),ORANGE,1.8)
txt(1382,821,'支持行恢复标签',22,ORANGE)
for k,yy in enumerate([885,984]):
    j=SUP[k];p=.1*RAW[j]+.9*AVG[j]
    pie(1267,yy,p,15,True)
    arrow((1294,yy),(1324,yy),ORANGE,1.4)
    pie(1348,yy,INIT[j],15,True)
    bar(1383,yy-12,INIT[j],111,24,True,17)
txt(1377,1060,'非支持行保留混合结果',18,GRAY)

# Final five-node state from the same toy weights, after exactly 20 iterations.
arrow((1514,937),(1571,937),TEAL,1.9)
txt(1741,821,'20 轮后概率',24,TEAL)
pos=np.array([[1650,899],[1790,887],[1714,957],[1816,1019],[1630,1036]],float)
for j,lw in zip([0,1,3],[2.6,1.8,1.1]):
    a=pos[j];b=pos[2];unit=(b-a)/np.linalg.norm(b-a);arrow(a+unit*20,b-unit*19,LINE,lw,ms=12)
for j,(x,y) in enumerate(pos):pie(x,y,FINAL[j],15,j in SUP)
for j in range(5):
    ax.add_patch(Circle((1647+j*44,1090),8,fc=COL[int(FINAL[j].argmax())],ec='white',lw=.6,zorder=4))
txt(1741,1132,'最大概率 → 类别',19,GRAY)

# Loop returns the clamped current state; F0 and W remain fixed.
route([(1375,1090),(1375,1138),(101,1138),(101,1070)],ORANGE,1.7,dash=True)
txt(724,1110,'每轮先混合，再恢复支持标签；重复 20 次',18,ORANGE)

# Two formulas only, and an explicit boundary at the bottom.
txt(61,1200,'仅作读出，不反向更新提示或主干',20,GRAY,ha='left')
txt(936,1200,r'$F^{(t+1)}=0.1F_0+0.9WF^{(t)},\quad F^{(t+1)}_S=Y_S$',24,INK)
txt(1909,1200,'局部图为示意',18,GRAY,ha='right')

fig.canvas.draw()
# Prevent silent canvas clipping when editing labels.
renderer=fig.canvas.get_renderer()
for item in ax.texts:
    b=item.get_window_extent(renderer)
    assert b.x0>=-1 and b.x1<=fig.bbox.x1+1 and b.y0>=-1 and b.y1<=fig.bbox.y1+1,item.get_text()
stem=OUT/'M3_支持约束概率传播'
for ext in ['pdf','svg']:fig.savefig(stem.with_suffix('.'+ext),facecolor='white')
fig.savefig(stem.with_suffix('.png'),dpi=300,facecolor='white')
fig.savefig(OUT/'M3_支持约束概率传播_preview.png',dpi=100,facecolor='white')
plt.close(fig)
print('weights(v):',WW[2].tolist())
print('neighbor average:',AVG[2].tolist())
print('first update:',FIRST[2].tolist())
print('F20:',FINAL.tolist())
print(stem)
