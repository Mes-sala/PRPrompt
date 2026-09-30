from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, FancyArrowPatch
from matplotlib.font_manager import FontProperties
from matplotlib.colors import to_rgb

OUT=Path(__file__).resolve().parent
FONT=FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc')
plt.rcParams.update({'pdf.fonttype':42,'svg.fonttype':'path','font.family':'DejaVu Sans','mathtext.fontset':'stix'})
W,H=2000,1250
INK='#24394B';GRAY='#667D90';LINE='#9AAEBE';BLUE='#5683A9';TEAL='#238D89';PURPLE='#926FB1';ORANGE='#B56B3D'
fig=plt.figure(figsize=(20,12.5),facecolor='white')
ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,W),ylim=(H,0));ax.axis('off')

def txt(x,y,s,fs=24,col=INK,ha='center',math=False):
    return ax.text(x,y,s,fontsize=fs,color=col,ha=ha,va='center',fontproperties=None if math else FONT,zorder=12)
def rr(x,y,w,h,fc,ec='none',r=20,lw=1):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',fc=fc,ec=ec,lw=lw,zorder=0))
def line(points,c=LINE,lw=1.6,dash=False,z=2):
    ax.plot(*np.asarray(points).T,c=c,lw=lw,ls=(0,(4,4)) if dash else '-',zorder=z)
def arr(a,b,c=GRAY,lw=2,style='-|>',dash=False,z=4):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle=style,mutation_scale=18,lw=lw,color=c,linestyle=(0,(4,3)) if dash else '-',shrinkA=0,shrinkB=1,zorder=z))
def route(points,c=GRAY,lw=2,dash=False):
    line(points[:-1],c,lw,dash);arr(points[-2],points[-1],c,lw,dash=dash)
def matrix(x,y,rows=3,cols=7,cell=21,color=BLUE,phase=0,shade=True):
    for i in range(rows):
        for j in range(cols):
            a=.22+.57*((i*3+j*2+phase)%9)/8 if shade else .42
            rgb=to_rgb(color);fc=tuple(1-a+a*v for v in rgb)
            ax.add_patch(Rectangle((x+j*cell,y+i*cell),cell,cell,fc=fc,ec='white',lw=.7,zorder=4))
    ax.add_patch(Rectangle((x,y),cols*cell,rows*cell,fc='none',ec=LINE,lw=.7,zorder=6))
def snow(x,y,size=12):
    for a in [0,np.pi/3,2*np.pi/3]:
        v=size*np.array([np.cos(a),np.sin(a)]);line([(x-v[0],y-v[1]),(x+v[0],y+v[1])],BLUE,1.7,z=7)
def network(x,y):
    rr(x,y,128,118,'#ECF3F9','#BDD0DF',r=9)
    nodes=[(x+23,y+31),(x+64,y+23),(x+104,y+32),(x+25,y+85),(x+65,y+66),(x+102,y+87)]
    for i,j in [(0,1),(0,3),(0,4),(1,2),(1,4),(2,4),(2,5),(3,4),(4,5)]:line([nodes[i],nodes[j]],'#B6CADB',1)
    for p in nodes:ax.add_patch(Circle(p,5,fc='white',ec=BLUE,lw=1,zorder=6))
    snow(x+64,y-20)
def history_stack(x,y,c,rows=3,cols=5,cell=14):
    for off,a in [(16,.08),(8,.12)]:
        ax.add_patch(Rectangle((x-off,y-off),cols*cell,rows*cell,fc='white',ec='#C4D1DB',lw=1,zorder=2))
    matrix(x,y,rows,cols,cell,c)

# One parameter set generates the single gradient used by both branches.
txt(62,49,'M2  更新方向与尺度匹配',34,INK,ha='left')
txt(63,92,'同一个提示矩阵、同一条训练轨迹；用矩阵统计定方向，用 Adam 定步长',22,GRAY,ha='left')
matrix(87,167,3,7,25,TEAL)
txt(172,144,'类别提示 '+r'$P_t$',23)
txt(172,274,'唯一可学习参数',21,ORANGE)
arr((280,208),(342,208),ORANGE,2.2)
network(358,149)
txt(422,291,'冻结图编码器',22,BLUE)
arr((501,208),(567,208),GRAY)
# Three support rows: target circles and measured predictions, feeding the loss.
for i,p in enumerate([.80,.35,.63]):
    yy=164+31*i;co=[TEAL,PURPLE,TEAL][i]
    ax.add_patch(Circle((597,yy+9),8,fc=co,ec=INK,lw=.8,zorder=6))
    ax.add_patch(Circle((597,yy+9),11,fc='none',ec=INK,lw=.7,zorder=6))
    ax.add_patch(Rectangle((620,yy),100*p,18,fc=TEAL,ec='white',lw=.7,zorder=6))
    ax.add_patch(Rectangle((620+100*p,yy),100*(1-p),18,fc=PURPLE,ec='white',lw=.7,zorder=6))
txt(654,274,'支持集损失',23)
arr((739,208),(808,208),ORANGE)
txt(775,172,'求导',18,ORANGE)
matrix(832,162,3,7,25,BLUE)
ax.add_patch(Rectangle((832,162),175,25,fc='none',ec=TEAL,lw=2.4,zorder=7))
ax.add_patch(Rectangle((932,162),25,75,fc='none',ec=PURPLE,lw=2.4,zorder=8))
txt(921,131,'当前梯度 '+r'$G_t$',25)
txt(921,273,'行 = 类别；列 = 隐藏维',21,GRAY)
# Top-right gives the tensor axes a spatial legend rather than another parameter box.
matrix(1202,158,3,7,25,BLUE)
arr((1202,250),(1377,250),PURPLE,1.5,style='<->')
txt(1289,280,'隐藏维 d = 128',21,PURPLE)
arr((1183,158),(1183,234),TEAL,1.5,style='<->')
txt(1137,198,'C\n类别',20,TEAL)
txt(1585,176,'两条分支读取',24)
txt(1585,214,'同一个梯度',24,ORANGE)
txt(1815,193,'不再训练\n第二套模型',22,GRAY)
line([(1450,150),(1450,250)],'#DEE6ED',1.2)
line([(1725,150),(1725,250)],'#DEE6ED',1.2)

# The fork remains outside the mechanism panels.
line([(921,301),(921,331)],ORANGE,2)
ax.add_patch(Circle((921,331),4,fc=ORANGE,ec='none',zorder=5))
route([(921,331),(491,331),(491,389)],BLUE,2)
route([(921,331),(1497,331),(1497,389)],PURPLE,2)
txt(1000,365,'在每一步维护对应的历史状态',20,GRAY)

# Matrix preconditioning. Row and column highlights are repeated in their corresponding statistics.
rr(50,403,935,343,'#F1F6FA','#DAE5EE')
txt(79,432,'类别轴与特征轴分别统计，重新调节方向',25,BLUE,ha='left')
history_stack(116,509,TEAL,3,3,25)
txt(153,617,'类别关联',22,TEAL)
txt(153,648,'L：C × C',21,TEAL)
matrix(324,517,3,7,20,BLUE)
ax.add_patch(Rectangle((324,517),140,20,fc='none',ec=TEAL,lw=2.5,zorder=7))
ax.add_patch(Rectangle((404,517),20,60,fc='none',ec=PURPLE,lw=2.5,zorder=8))
txt(394,484,'梯度行 / 列',22)
arr((312,533),(211,533),TEAL,2)
txt(257,584,'行配对',18,TEAL)
arr((476,548),(562,548),PURPLE,2)
txt(520,603,'列配对',18,PURPLE)
history_stack(577,491,PURPLE,7,7,15)
txt(630,617,'特征关联',22,PURPLE)
txt(630,648,r'$R_t: d\times d$',21,PURPLE,math=True)
arr((703,551),(766,551),BLUE,2)
origin=np.array([817.,581.])
for ang,le in [(-1.05,62),(-.28,75)]:
    v=le*np.array([np.cos(ang),np.sin(ang)]);arr(origin,origin+v,BLUE,2.2)
txt(851,629,'双侧重标定',20,BLUE)
txt(851,658,'Shampoo 方向',18,BLUE)
txt(514,709,r'$D_t=L_t^{-1/4}\,G_t\,R_t^{-1/4}$',28,INK,math=True)

# Adam uses the same trajectory, but only its scalar norm is passed onward.
rr(1014,403,936,343,'#F7F3FA','#E7E0ED')
txt(1044,432,'平滑历史梯度，形成 Adam 参照',25,PURPLE,ha='left')
history_stack(1080,529,BLUE,3,7,11)
txt(1120,619,'历次梯度',22,GRAY)
route([(1166,551),(1215,551),(1215,511),(1261,511)],PURPLE,1.8)
route([(1166,551),(1215,551),(1215,632),(1261,632)],PURPLE,1.8)
matrix(1273,482,3,7,13,PURPLE)
txt(1475,509,'一阶平均 '+r'$m_t$',22,PURPLE)
matrix(1273,603,3,7,13,ORANGE,phase=4)
txt(1475,630,'平方平均 '+r'$v_t$',22,ORANGE)
route([(1584,510),(1640,510),(1640,554),(1684,554)],PURPLE,1.7)
route([(1584,630),(1640,630),(1640,554),(1684,554)],PURPLE,1.7)
arr((1731,608),(1821,536),GRAY,2.4)
txt(1788,499,'候选方向 '+r'$A_t$',22)
txt(1766,650,'偏差校正 · 坐标缩放',18,GRAY)
txt(1490,709,'只传递整体长度，不采用 Adam 的方向',24,PURPLE)

# The lower construction visually separates directions from the negative parameter step.
route([(849,746),(849,777),(405,777),(405,822)],BLUE,2)
route([(1790,746),(1790,777),(1089,777),(1089,822)],PURPLE,2)
rr(50,835,1900,355,'#FFF9F3','#F0E4D8')
txt(81,866,'方向保持，长度匹配',27,ORANGE,ha='left')

o=np.array([344.,1070.]);rad=122.
ax.add_patch(Circle(o,rad,fc='none',ec='#B2BEC8',lw=1.5,ls=(0,(4,3)),zorder=2))
uD=np.array([np.cos(-2.60),np.sin(-2.60)])
uA=np.array([np.cos(-.36),np.sin(-.36)])
full=o+rad*uD;raw=o+(rad/3)*uD;adam=o+rad*uA
arr(o,full,ORANGE,3,z=5)
arr(o,raw,BLUE,4,z=7)
arr(o,adam,GRAY,2.2,dash=True)
for p,c in [(full,ORANGE),(adam,GRAY)]:ax.add_patch(Circle(p,4,fc=c,ec='white',lw=.7,zorder=8))
ax.add_patch(Circle(o,4,fc=INK,ec='none',zorder=8))
txt(143,959,'放大 3 倍',23,ORANGE)
txt(127,994,'沿同一射线',20,ORANGE)
line([(199,981),(235,981),(full[0]-4,full[1])],ORANGE,1.3)
txt(228,1104,'原方向 D：2',20,BLUE)
txt(487,1002,'Adam：6',21,GRAY)
txt(493,1120,'圆半径 = Adam 范数',19,GRAY)
line([(480,1102),(455,1095)],GRAY,1.1)

# Only the necessary second equation; norm matching is whole-matrix Frobenius matching.
txt(1010,883,'整体范数匹配',24,ORANGE)
txt(1015,948,r'$\widetilde D_t=D_t\,\frac{\|A_t\|_F}{\|D_t\|_F}$',31,INK,math=True)
txt(1007,1014,'方向来自 D；长度变为 6',23)
txt(1007,1056,'学习率 0.01 → 实际步长 0.06',24,ORANGE)
txt(1007,1100,'范数覆盖整个提示矩阵，不逐元素匹配',20,GRAY)
txt(1007,1138,'D 为零时不更新',19,GRAY)
line([(1430,907),(1430,1141)],'#E6D7C8',1.2)

txt(1692,877,'沿负方向更新同一个 P',24,ORANGE)
matrix(1496,957,3,7,17,TEAL,phase=0)
txt(1553,925,r'$P_t$',24,INK,math=True)
matrix(1784,1060,3,7,17,TEAL,phase=3)
txt(1840,1152,r'$P_{t+1}$',24,INK,math=True)
arr((1629,1000),(1765,1082),ORANGE,3.1)
txt(1720,970,'实际参数移动',21,ORANGE)
txt(1633,1137,'与上图 D 方向相反',20,GRAY)

txt(57,1226,'示意向量表示整个矩阵；Shampoo / Adam 是已有更新方法。主干冻结，优化器只修改提示参数。',21,GRAY,ha='left')
stem=OUT/'M2_更新方向与尺度匹配'
fig.savefig(str(stem)+'.png',dpi=300,facecolor='white')
fig.savefig(str(stem)+'_preview.png',dpi=100,facecolor='white')
fig.savefig(str(stem)+'.pdf',facecolor='white')
fig.savefig(str(stem)+'.svg',facecolor='white')
plt.close(fig)
print(stem)
