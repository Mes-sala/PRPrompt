from pathlib import Path
import json
OUT=Path(__file__).resolve().parent
BASE=OUT.parent
src=(BASE/'三个模块Framework详解_20260926/draw_m3.py').read_text()
prefix=src[:src.index('# Main heading')]
prefix=prefix.replace("OUT = Path(__file__).resolve().parent",'OUT = Path(__file__).resolve().parent').replace('W=2000; H=1250','W=2000; H=1570').replace('figsize=(20,12.5)','figsize=(20,15.7)')
prefix=prefix.replace('SUP=[0,3]; INIT=RAW.copy()', '''INPUT=RAW.copy()
LABEL=INPUT.argmax(1)
AGREE=(OP*(LABEL[:,None]==LABEL[None,:])).sum(1)
TEMP=1+3*(1-AGREE)
RAW=np.exp(np.log(INPUT)/TEMP[:,None]);RAW/=RAW.sum(1,keepdims=True)
SUP=[0,3]; INIT=RAW.copy()''')
exec(compile(prefix,'original_primitives','exec'))
top=src[src.index('# Main heading'):src.index('# A single magnifying inset')]
top=top.replace('M3  支持约束概率传播','M3-LC  局部一致性校准与支持锚定传播').replace('训练后的读出：以表示建立关系，让类别概率沿近邻传递','Prompt 决定分类分数，教师属性决定近邻，局部一致性调整进入传播的置信度')
top=top.replace("txt(257,449,'两路共用同一份表示',19,GRAY)","")
top=top.replace('route([(449,293),(480,293),(480,550),(548,550)],BLUE,1.7)',"cloud(260,555,True);txt(260,475,'教师构图属性',23,BLUE);txt(260,650,'中心化两跳传播属性',19,GRAY);arrow((345,550),(548,550),BLUE,1.7)")
top=top.replace('从表示找近邻','从教师属性找近邻').replace("for r,p in enumerate(RAW):","for r,p in enumerate(INPUT):").replace("'原始概率 $F_0$'","'初始概率与预测'").replace('保留原值，每轮回注','由初始分数 s 产生')
a=top.index('route([(1416,261)');b=top.index('# Lower preparation')
top=top[:a]+'''arrow((1416,261),(1520,261),TEAL)
txt(1720,172,'初始预测类别',23,TEAL)
for j in range(5):
    pie(1550+j*78,265,[1,0] if LABEL[j]==0 else [0,1],15)
    txt(1550+j*78,313,['a','b','v','c','d'][j],18,GRAY)
txt(1720,373,r'$c_i=\\arg\\max_c s_{ic}$',24,INK)
''' +top[b:]
exec(compile(top,'updated_inputs','exec'))
# New module uses the original circle, bar, arrow, and typography primitives.
rr(60,713,1880,290,'#F1F9F6','#C9E2DB',18,1)
txt(88,748,'局部一致性 → 一次性温度调整',25,INK,ha='left')
txt(975,748,'近邻预测越不支持，温度越高；不改变初始 argmax',19,GRAY,ha='left')
txt(208,798,'邻居是否同类？',22,BLUE)
for r,j in enumerate([0,1,3]):
 yy=843+r*53
 pie(110,yy,[1,0] if LABEL[j]==0 else [0,1],12)
 txt(147,yy,['a','b','c'][r],18,GRAY)
 txt(215,yy,f'{WW[2,r]:.3f}',18,BLUE)
 txt(308,yy,'1' if LABEL[j]==LABEL[2] else '0',22,TEAL if LABEL[j]==LABEL[2] else GRAY)
arrow((355,884),(410,884),BLUE)
txt(565,807,'加权一致性',22,BLUE)
txt(565,856,r'$a_v=\sum_j W_{vj}\mathbf{1}[c_j=c_v]$',23)
txt(565,913,f'$a_v={AGREE[2]:.3f}$',26,BLUE)
txt(565,962,'使用 Prompt 的初始类别',18,GRAY)
arrow((751,884),(800,884),TEAL)
txt(972,798,'节点温度',22,TEAL)
txt(972,843,r'$T_v=1+3(1-a_v)$',24,TEAL)
txt(972,884,f'$T_v={TEMP[2]:.3f}$',25,TEAL)
bar(848,925,INPUT[2],100,24,True,16);arrow((962,937),(1004,937),TEAL);bar(1017,925,RAW[2],120,24,True,16)
txt(985,977,'同一类别，较软的概率',18,GRAY)
arrow((1153,884),(1200,884),TEAL)
txt(1340,798,'校准概率 $F_0$',22,TEAL)
for j,p in enumerate(RAW):bar(1240,831+j*28,p,188,21)
txt(1340,983,r'$F_{0,i}=\mathrm{softmax}(s_i/T_i)$',20)
arrow((1458,884),(1508,884),ORANGE)
txt(1720,798,'初始化：固定支持标签',22,ORANGE)
for j,p in enumerate(INIT):
 txt(1580,840+j*28,['a','b','v','c','d'][j],17,GRAY);bar(1607,831+j*28,p,185,21)
txt(1720,983,'复制校准概率；覆盖支持行',18,GRAY)
# Preserve original one-step zoom-in, recomputed using calibrated probabilities.
old=set(ax.get_children())
lower=src[src.index('# A single magnifying inset'):src.index('fig.canvas.draw()')]
lower=lower.replace('原始 $F_0(v)$','校准 $F_0(v)$').replace("'0.742 / 0.258'","f'{FIRST[2,0]:.3f} / {FIRST[2,1]:.3f}'").replace('两 formulas','两 formulas')
exec(compile(lower,'original_diffusion_zoom','exec'))
from matplotlib.transforms import Affine2D
for a in set(ax.get_children())-old:a.set_transform(Affine2D().translate(0,320)+ax.transData)
fig.canvas.draw()
for item in ax.texts:
 b=item.get_window_extent(fig.canvas.get_renderer())
 assert b.x0>=-1 and b.x1<=fig.bbox.x1+1 and b.y0>=-1 and b.y1<=fig.bbox.y1+1,item.get_text()
stem=OUT/'M3_LC_机制详解'
for ext in ['pdf','svg']:fig.savefig(stem.with_suffix('.'+ext),facecolor='white')
fig.savefig(stem.with_suffix('.png'),dpi=250,facecolor='white');fig.savefig(OUT/'M3_LC_机制详解_preview.png',dpi=85,facecolor='white')
(OUT/'M3_数值算例.json').write_text(json.dumps({'weights':OP.tolist(),'input':INPUT.tolist(),'agreement':AGREE.tolist(),'temperature':TEMP.tolist(),'calibrated':RAW.tolist(),'initial_clamped':INIT.tolist(),'first':FIRST.tolist(),'final':FINAL.tolist()},indent=2))
print('M3 complete; first(v)=',FIRST[2])
