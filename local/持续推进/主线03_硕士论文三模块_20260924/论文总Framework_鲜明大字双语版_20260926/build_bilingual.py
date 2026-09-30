from pathlib import Path
import sys,re,xml.etree.ElementTree as ET,json,itertools
import numpy as np
from matplotlib.colors import to_rgb,to_hex
from matplotlib.patches import Circle,Rectangle
OUT=Path(__file__).resolve().parent
BASE=OUT.parent
# Reuse coordinate and operator definitions, replacing the component drawing primitives.
old=(BASE/'论文总Framework_Drawio组件版_20260926/build_drawio.py').read_text()
exec(compile(old[:old.index('source=SOURCE.read_text()')],str(BASE/'论文总Framework_Drawio组件版_20260926/build_drawio.py'),'exec'))
LANG=sys.argv[1] if len(sys.argv)>1 else 'CN'
INK='#182C41';GRAY='#44586D';LINE='#8EA8BE';BLUE='#2B75B9';TEAL='#008E87';PURPLE='#9056C3';ORANGE='#D47A23';C=[TEAL,PURPLE]
EN={
'修正类别原型':'Prototype refinement','循环更新':'Iterative refinement','支持原型':'Support prototypes','无标签属性':'Unlabeled features','软分配':'Soft assignments','加权软原型':'Weighted soft\nprototypes','加权均值':'Weighted mean','修正原型':'Refined\nprototypes','融合':'Blend','保留支持参照':'Retain support anchors','锐化':'Sharpening','固定有符号门控':'Fixed signed\ngates','类别提示 P':'Class prompts P','节点修正':'Node-wise\ncorrection','同一冻结图编码器':'Shared frozen graph encoder','输入图':'Input graph','第 1 层':'Layer 1','第 2 层':'Layer 2','隐藏特征修正':'Hidden-feature correction','适配表示':'Adapted representations','提示学习：方向与长度分别适配':'Prompt learning: direction and norm adaptation','支持集损失':'Support loss','同一梯度 G':'Shared gradient\nG','同一起点':'Common origin','适配后方向 $\\widetilde D$':'Rescaled direction $\\widetilde D$','与 D 同方向，仅沿射线缩放':'Same ray; only the length changes','Shampoo 原方向 D':'Shampoo direction D','Adam 参照方向 A':'Adam reference A','只取整体长度，不采用其方向':'Use its norm, not its direction','虚线圆半径 = Adam 的整体范数':'Dashed-circle radius = Adam norm','$\\|\\widetilde D\\|_F=\\|A\\|_F$':'$\\|\\widetilde D\\|_F=\\|A\\|_F$','匹配整个提示矩阵的长度':'Match the whole-matrix norm','$P\\leftarrow P-\\eta\\widetilde D$':'$P\\leftarrow P-\\eta\\widetilde D$','参数实际沿负方向更新':'Parameters move in\nthe negative direction','训练后：近邻构建与传播':'Post-training: neighbors and propagation','表示相似度':'Embedding\nsimilarity','从表示找邻居':'Retrieve by similarity','选择近邻':'Neighbor\nselection','排除自身':'Exclude self','表示近邻图':'Representation\nneighbor graph','相似度赋权':'Similarity weights','支持原型分类':'Prototype\nclassifier','原始概率 $F_0$':'Original probabilities $F_0$','保留原值':'Keep fixed','固定\nW':'Fixed\nW','当前概率':'Current probabilities','更新概率':'Updated probabilities','邻居概率传播':'Neighbor\npropagation','迭代更新':'Iterative update','原始 $F_0$':'Original $F_0$','邻居加权概率':'Weighted neighbor\nprobabilities','新的类别概率':'Updated class probabilities','支持标签保持固定':'Clamp support labels','类别预测':'Predicted classes','冻结主干':'Frozen backbone','支持节点':'Support node','仅更新提示 P':'Only P is updated','局部机制示意':'Local mechanism illustration'}
MAXWIDTH={'提示学习：方向与长度分别适配':1200,'支持原型':250,'软分配':220,'加权软原型':220,'修正原型':160,'锐化':140,'固定有符号门控':260,'类别提示 P':290,'节点修正':280,'支持集损失':220,'同一梯度 G':265,'表示相似度':210,'从表示找邻居':250,'选择近邻':180,'表示近邻图':230,'相似度赋权':245,'支持原型分类':265,'原始概率 $F_0$':355,'当前概率':290,'更新概率':290,'邻居概率传播':300,'邻居加权概率':280,'新的类别概率':325,'类别预测':255,'冻结主干':205,'支持节点':185,'局部机制示意':340,'隐藏特征修正':315,'适配表示':325}

def txt(x,y,s,fs=24,col=INK,ha='center',math=False):
 orig=s
 if LANG=='EN':s=EN.get(s,s)
 if orig not in ['Shampoo 原方向 D','Adam 参照方向 A','适配后方向 $\\widetilde D$','与 D 同方向，仅沿射线缩放']:col=INK if fs>=21 else GRAY
 font='Times New Roman' if LANG=='EN' else 'Songti SC'
 size=fs*1.30
 if LANG=='EN':
  size=fs*1.20
  if orig=='训练后：近邻构建与传播':size=31
  if orig=='提示学习：方向与长度分别适配':size=35
  if orig in ['加权软原型','修正原型','固定有符号门控','表示相似度','选择近邻','表示近邻图']:y-=3
  if orig=='邻居加权概率':y=1140;size=22
  if orig=='节点修正':y=433
  if orig=='从表示找邻居':x+=18
  if orig=='无标签属性':x+=24
  if orig=='邻居概率传播':size=26
  if orig=='原始概率 $F_0$':size=27
  if orig=='锐化':size=25
  if orig=='隐藏特征修正':size=24
  if orig in ['当前概率','更新概率']:size=28
 # Short lines and fixed-size formula containers keep both native editor and raster export stable.
 s=re.sub(r'\$([^$]+)\$',lambda m:'\\('+m.group(1)+'\\)',s)
 plain=re.sub(r'\\\(.+?\\\)','D',s)
 maxw=MAXWIDTH.get(orig,650 if fs>=27 else 560)
 width=max(sum(1 if ord(c)>255 else .54 for c in p)*size for p in plain.split('\n'))+16
 if LANG=='EN' and width>maxw:size*=maxw/width;width=maxw
 # Increase type after language fitting; preserve a modest increase in the tightest slots.
 gain=1.12
 if LANG=='EN' and orig in ['支持原型','软分配','加权软原型','修正原型','固定有符号门控','表示相似度','选择近邻','表示近邻图','原始概率 $F_0$','当前概率','更新概率','训练后：近邻构建与传播','隐藏特征修正']:gain=1.08
 if orig=='虚线圆半径 = Adam 的整体范数':gain=1.06
 if LANG=='EN' and orig=='邻居加权概率':gain=1.08
 size*=gain;width*=gain
 height=size*(2.35 if '\n' in s else 1.6)
 if math:width=370
 xx=x-width/2 if ha=='center' else x if ha=='left' else x-width
 style=f'text;html=1;strokeColor=none;fillColor=none;align={ha};verticalAlign=middle;whiteSpace={"wrap" if math else "nowrap"};overflow={"fill" if math else "visible"};fontFamily={font};fontSize={size};fontColor={col};fontStyle={1 if fs>=27 and not math else 0};spacing=0;'
 cell(style,xx,y-height/2,width,height,s.replace('\n','<br>'),12)
 texts.append({'source':orig,'text':s,'x':x,'y':y,'size':size,'width':width,'height':height})

def box(x,y,w,h,fc,ec='none',r=20):
 if w>500:
  fills={900:'#F0FAF7',755:'#EFF5FC',1380:'#FFF8EF',695:'#F4F8FD',659:'#EAF3FC'}
  result=vertex(x,y,w,h,fc=fills.get(int(w),'#FFFFFF'),ec='#CED7DE',sw=1.1,extra='rounded=1;absoluteArcSize=1;arcSize=18;',z=0)
  if w!=659:line([(x+18,y+17),(x+18,y+44)],TEAL if w==900 else ORANGE if w==1380 else BLUE,3.2,z=1)
  return result
 return vertex(x,y,w,h,fc='#FFFFFF',ec='#B8C8D1',sw=1.2,extra='rounded=1;absoluteArcSize=1;arcSize=10;',z=0)

def mat(x,y,rows=3,cols=5,cell=16,kind='blue',phase=0):
 # A tiny offset backing sheet conveys a feature array; front cells remain independently editable.
 width=cols*cell;height=rows*cell
 vertex(x+5,y-5,width,height,fc='#EEF3F6',ec='#9EB3C3',sw=.8,z=3)
 vertex(x+2.5,y-2.5,width,height,fc='#F8FAFC',ec='#9EB3C3',sw=.8,z=3.1)
 for i in range(rows):
  for j in range(cols):
   color=C[i%2] if kind=='class' else BLUE
   rgb=to_rgb(color);amount=.38+.52*((j+2*i+phase)%5)/4
   fc=to_hex(tuple(1-amount+amount*v for v in rgb))
   vertex(x+j*cell,y+i*cell,cell,cell,fc=fc,ec='#617B8B',sw=.55,z=4)
 vertex(x,y,width,height,fc='none',ec='#577385',sw=1.05,z=6)

def bar(x,y,p,w=150,h=27,numbers=False):
 start=x
 for k,v in enumerate(p):
  if v>0:vertex(start,y,w*v,h,fc=C[k],ec='#FFFFFF',sw=.6,z=5)
  start+=w*v
 vertex(x,y,w,h,fc='none',ec='#798F9F',sw=.65,z=6)

def dot(x,y,c=GRAY,r=10,ring=False):
 # Visible contours survive reduction to paper width.
 vertex(x-r,y-r,2*r,2*r,'ellipse',c,INK if ring else '#668093',.85,z=6)
 if ring:
  vertex(x-r-5,y-r-5,2*r+10,2*r+10,'ellipse','none',INK,1.25,z=7)

def star(x,y,c=TEAL,size=410,outline=False):
 r=np.sqrt(size)*.73
 vertex(x-r-5,y-r-5,2*r+10,2*r+10,'ellipse',c,'none',0,extra='fillOpacity=13;',z=3)
 vertex(x-r,y-r,2*r,2*r,'mxgraph.basic.star','#FFFFFF' if outline else c,c,1.4,z=8)

def diamond(x,y,c=TEAL,size=420):
 r=np.sqrt(size)*.77
 vertex(x-r-4,y-r-4,2*r+8,2*r+8,'ellipse',c,'none',0,extra='fillOpacity=13;',z=3)
 vertex(x-r,y-r,r*2,r*2,'rhombus',c,'#FFFFFF',1.1,z=7)

def encoder(x,y,label):
 # Graph convolution is represented by graph nodes, not a fully connected MLP icon.
 vertex(x+8,y-8,84,108,fc='#E3EBF2',ec='#7E97AD',sw=1.0,extra='rounded=1;arcSize=10;',z=1)
 vertex(x+4,y-4,84,108,fc='#ECF1F6',ec='#7E97AD',sw=1.0,extra='rounded=1;arcSize=10;',z=1.1)
 vertex(x,y,84,108,fc='#F9FBFD',ec=BLUE,sw=1.5,extra='rounded=1;arcSize=10;',z=1.2)
 # Upper band denotes a frozen compute block.
 vertex(x,y,84,8,fc=BLUE,ec='none',sw=0,z=2)
 ps=np.array([[19,36],[60,31],[41,55],[21,80],[65,78]])+[x,y]
 for i,j in [(0,1),(0,2),(1,2),(2,3),(2,4),(3,4)]:line([ps[i],ps[j]],'#869DAF',1.1,z=2)
 for i,p in enumerate(ps):vertex(p[0]-4.3,p[1]-4.3,8.6,8.6,'ellipse','#FFFFFF',BLUE,1.1,z=5)
 snow(x+42,y-30);txt(x+42,y+132,label,21,BLUE)

# The same numerical operator and semantic layout feed both language variants.
source=SOURCE.read_text()
body=source[source.index('# Local probability example'):source.index('# Persist text positions')]
body=body.replace('[(1298,386),(1298,542),(700,542),(700,613)]','[(1298,386),(1298,391),(1396,391),(1396,542),(700,542),(700,613)]')
body=body.replace('[(1827,600),(1827,663),(1615,663),(1615,713)]','[(1827,600),(1827,604),(1942,604),(1942,663),(1615,663),(1615,713)]')
exec(compile(body,str(SOURCE),'exec'))
assert np.isclose(np.linalg.norm(adapt-o),np.linalg.norm(adam-o))
assert np.allclose((adapt-o)/np.linalg.norm(adapt-o),(raw-o)/np.linalg.norm(raw-o))
model=ET.Element('mxGraphModel',dx='2200',dy='1440',grid='1',gridSize='10',guides='1',tooltips='1',connect='1',arrows='1',fold='1',page='1',pageScale='1',pageWidth='2200',pageHeight='1440',math='1',shadow='0',background='#ffffff')
root=ET.SubElement(model,'root');ET.SubElement(root,'mxCell',id='0');ET.SubElement(root,'mxCell',id='1',parent='0',value='Framework '+LANG)
for z,n,c in sorted(items,key=lambda v:(v[0],v[1])):root.append(c)
xml=ET.tostring(model,encoding='unicode');stem='Framework_'+LANG
path=OUT/(stem+'.drawio')
if not path.exists():path.write_text('<mxfile host="draw.io"><diagram id="framework" name="Framework '+LANG+'"><mxGraphModel><root><mxCell id="0"/><mxCell id="1" parent="0"/></root></mxGraphModel></diagram></mxfile>')
(OUT/(stem+'_mcp.json')).write_text(json.dumps({'path':str(path),'page':'framework','content':xml},ensure_ascii=False))
(OUT/(stem+'_labels.json')).write_text(json.dumps(texts,ensure_ascii=False,indent=2))
print(stem,len(items),'editable objects',len(texts),'labels')
