from pathlib import Path
import xml.etree.ElementTree as E
import math,copy
out=Path('/Users/shq/Research/GP/持续推进/PRPrompt_原型路由图提示_20260928/PRPrompt_Framework_连续计算图_20260928')
ink='#292929'; orange='#EBA166'; blue='#B6CDDC'; red='#B14E3A'; gray='#E8E8E5'
r=E.Element('root'); E.SubElement(r,'mxCell',id='0'); E.SubElement(r,'mxCell',id='1',parent='0')
idx=0

def box(id,x,y,w,h,label='',fill='none',extra='',parent='1'):
 c=E.SubElement(r,'mxCell',id=id,value=label,style=f'html=1;whiteSpace=wrap;fontFamily=Arial;fontSize=18;fontColor={ink};strokeColor={ink};strokeWidth=1.5;fillColor={fill};'+extra,vertex='1',parent=parent)
 E.SubElement(c,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),**{'as':'geometry'});return id

def text(id,x,y,w,h,label,size=18,extra=''):
 return box(id,x,y,w,h,label,extra=f'strokeColor=none;fontSize={size};'+extra)

def line(id,points,color=ink,dash=False,arrow=True,source=None,target=None):
 c=E.SubElement(r,'mxCell',id=id,style=f'html=1;strokeColor={color};strokeWidth=1.7;endArrow={"block" if arrow else "none"};endSize=7;rounded=0;'+('dashed=1;dashPattern=5 4;' if dash else ''),edge='1',parent='1',**({'source':source} if source else {}),**({'target':target} if target else {}))
 g=E.SubElement(c,'mxGeometry',relative='1',**{'as':'geometry'});
 E.SubElement(g,'mxPoint',x=str(points[0][0]),y=str(points[0][1]),**{'as':'sourcePoint'});E.SubElement(g,'mxPoint',x=str(points[-1][0]),y=str(points[-1][1]),**{'as':'targetPoint'})
 if len(points)>2:
  a=E.SubElement(g,'Array',**{'as':'points'})
  for x,y in points[1:-1]:E.SubElement(a,'mxPoint',x=str(x),y=str(y))

def node(id,x,y,fill='white',size=18,label='',support=False):
 box(id,x-size/2,y-size/2,size,size,label,fill,'ellipse;aspect=fixed;fontSize='+('25;' if size>=35 else '12;')+('strokeWidth=3;' if support else ''))

def graph(id,x,y,scale=1,mode=0):
 p=[(0,45),(38,10),(48,74),(88,39),(120,5),(133,85),(167,47)]
 edges=[(0,1),(0,2),(1,3),(2,3),(2,5),(3,4),(3,6),(4,6),(5,6)] if mode!=2 else [(0,2),(1,4),(2,5),(3,1),(4,1),(5,2),(6,3),(3,6)]
 for k,(a,b) in enumerate(edges):line(id+'e'+str(k),[(x+p[a][0]*scale,y+p[a][1]*scale),(x+p[b][0]*scale,y+p[b][1]*scale)],color='#787878',arrow=mode==2)
 for k,(a,b) in enumerate(p):
  f=('#EBA166' if k in [0,2,5] else '#B6CDDC') if mode==3 else ('#EBA166' if k==0 else '#B6CDDC' if k==4 else 'white')
  node(id+'n'+str(k),x+a*scale,y+b*scale,f,19*scale,support=k in [0,4])

def lock(id,x,y):
 box(id+'arc',x+4,y,12,14,'','none','rounded=1;arcSize=60;strokeWidth=1.6;')
 box(id+'body',x,y+9,20,16,'',blue,'rounded=0;');node(id+'dot',x+10,y+17,ink,3)

def vector(id,x,y,colors):
 for j,c in enumerate(colors):box(id+str(j),x+j*14,y,14,22,'',c,'strokeWidth=1;')

# upper route: symbolic prototypes and assignments, no framing cards
text('m1',50,18,810,35,'<b>M1　原型路由：构造节点相关的提示残差</b>',23,'align=left;')
text('attr',50,66,170,44,'传播属性空间<br><i>X̄</i>',18)
# clusters two distinct shapes + centroids
for cl,(cx,cy,col) in enumerate([(105,157,orange),(188,140,blue)]):
 for j,(dx,dy) in enumerate([(-26,-18),(-24,19),(22,-22),(27,14)]):node(f'cluster{cl}-{j}',cx+dx,cy+dy,col,12)
 box(f'proto{cl}',cx-11,cy-11,22,22,'',col,'shape=rhombus;strokeWidth=2;')
text('refine',43,207,194,54,'支持原型 ↔ 软分配<br>融合与迭代',17)
line('protoToRoute',[(242,153),(287,153)])
# routing bipartite as node to class with signed edge cue
for j,y in enumerate([118,157,196]):node('routeNode'+str(j),310,y,'white',18)
for j,y in enumerate([127,187]):box('routeProto'+str(j),405,y-12,24,24,'',[orange,blue][j],'shape=rhombus;')
for k,(s,t,d) in enumerate([(118,127,False),(118,187,True),(157,127,False),(157,187,False),(196,127,True),(196,187,False)]):line('routeEdge'+str(k),[(321,s),(403,t)],'#929292',d,False)
text('routeLabel',270,66,195,40,'固定有符号路由 <i>Q</i>',21)
text('routeOps',256,218,221,53,'分配锐化 → 中心化<br>原图传播',17)
line('routeToMix',[(443,153),(537,153)])
text('mixMath',487,89,145,36,'<i>Q tanh(P)</i>',22,'fontFamily=Times New Roman;')
node('mix',563,153,'white',37,'×')
line('pToMix',[(563,290),(563,174)],orange)
# residual bank orange vector strips diagonal offsets
vector('p0',515,298,[orange,'#F3C7A4',orange,'#E9B48C','#F7DEC9'])
vector('p1',527,326,['#F3C7A4',orange,'#F7DEC9',orange,'#E9B48C'])
text('bankLabel',445,356,248,30,'共享残差库 <i>P</i> · 唯一可学习参数',16)
line('mixToDelta',[(584,153),(694,153)])
vector('delta',708,141,['#F2CAA9',orange,'#FAE8D8','#D68D53',orange])
text('deltaName',676,86,159,42,'节点残差 <i>ΔU</i>',20)
text('deltaScale',660,181,157,30,'有界变换 · 尺度匹配',15)
line('deltaToAdd',[(783,153),(825,153),(825,407)],orange)
# main computation spine y435
text('inputLab',28,334,195,35,'输入图 <i>G = (X, E)</i>',20)
graph('input',39,391,0.9)
line('inputToGCN',[(209,434),(274,434)])
# encoder as two blue trapezoids + addition circle inside an open dashed enclosure
box('encoderBoundary',259,390,828,124,'','none','rounded=1;arcSize=12;dashed=1;strokeColor=#B1B1AD;')
box('gcn1',290,410,155,59,'冻结 GCN₁',blue,'shape=trapezoid;direction=north;fontSize=21;')
lock('lock1',350,377)
line('gcn1ToU',[(446,434),(493,434)])
vector('u',506,423,['#E6E6E3','#B6CDDC','#E6E6E3','#B6CDDC','#E6E6E3'])
text('uLabel',501,470,105,30,'预激活 <i>U</i>',17)
line('uToPlus',[(579,434),(804,434)])
node('add',825,434,'white',40,'+')
line('addToActivation',[(846,434),(880,434)])
box('activation',883,406,53,59,'σ','white','fontFamily=Times New Roman;fontSize=31;')
text('activationLabel',854,477,114,25,'LeakyReLU',14)
line('actToGCN2',[(937,434),(962,434)])
box('gcn2',969,406,99,59,'GCN₂',blue,'shape=trapezoid;fontSize=21;')
lock('lock2',1007,373)
line('gcn2ToZ',[(1069,434),(1122,434)])
text('backboneLabel',290,521,367,31,'冻结主干 · 两层均使用原图 E',16,'fontColor=#555555;')
# adapted scatter graph at branch centre
for j,(dx,dy) in enumerate([(0,0),(23,-22),(30,20),(64,5),(78,-26),(94,22)]):node('embed'+str(j),1139+dx,431+dy,[orange,orange,orange,blue,blue,blue][j],14)
text('adaptLab',1095,476,177,49,'适配表示 <i>Z̃ₚ</i><br>中心化 · 归一化',17)
# M3 on right, fork probability upper and graph lower
text('m3',1143,18,505,35,'<b>M3　支持约束读出</b>',23,'align=left;')
line('zToF0',[(1189,391),(1189,122),(1277,122)])
text('protoProb',1169,49,225,39,'支持均值原型 → 初始概率',17)
# probability horizontal small barchart
for k,vals in enumerate([(46,12),(20,38),(33,25)]):
 for j,v in enumerate(vals):box(f'fbar{k}{j}',1300+sum(vals[:j]),99+k*23,v,14,'',[orange,blue][j],'strokeWidth=0.7;')
text('f0Label',1360,104,45,42,'<i>F₀</i>',24,'fontFamily=Times New Roman;')
line('f0ToReadout',[(1403,122),(1530,122),(1530,317)])
text('retainF0',1400,160,127,36,'初始概率回注',16)
line('zToW',[(1192,407),(1262,302)])
text('knnlabel',1151,251,242,37,'余弦近邻 · 行归一化',17)
graph('knn',1232,304,0.68,2)
text('wLabel',1280,377,129,30,'读出图 <i>W</i>',19)
line('wToIter',[(1364,337),(1431,337)])
# visual probability mixing as circle instead of card
node('iterate',1502,357,'#FAF5EC',99,'融合<br>与迭代')
# anchor labels above/below not long prose
line('iterLoop',[(1540,389),(1581,389),(1581,288),(1502,288),(1502,305)],ink)
text('clamp',1408,421,214,44,'每轮恢复支持标签<br><i>Fₛ ← Yₛ</i>',17)
line('clampToIter',[(1497,419),(1497,409)],red)
line('inputToAttr',[(39,413),(18,413),(18,152),(70,152)],'#777777',True)
text('supportCue',14,281,199,35,'属性 X 与支持标签 Yₛ',15)
line('iterToPred',[(1557,357),(1631,357),(1631,518),(1502,518),(1502,528)])
graph('prediction',1438,536,0.75,3)
text('outputLab',1380,624,245,32,'查询节点预测 <i>ŷ</i>',20)
text('readoutNote',1135,683,507,25,'训练后固定表示与读出图，仅迭代类别概率',16,'fontColor=#555555;')
# M2 compact below spine, entirely different auxiliary feedback typography
text('m2',44,567,390,36,'<b>M2　残差矩阵的结构化更新</b>',23,'align=left;')
text('loss',1029,574,172,48,'支持集损失<br><i>ℒₛ</i>',19)
line('zToLoss',[(1248,458),(1248,596),(1203,596)],red,True)
line('lossToGrad',[(1027,604),(973,604),(973,650),(900,650)],red,True)
text('grad',825,629,64,34,'∇<sub>P</sub>ℒₛ',22,'fontFamily=Times New Roman;')
line('gradToD',[(818,645),(752,612)],red,True)
line('gradToA',[(818,661),(752,698)],red,True)
text('shampoo',479,581,272,59,'类别轴 × 特征轴统计<br><b>Shampoo 方向 D</b>',18)
text('adam',503,674,247,50,'同轨迹 Adam 候选<br>仅取范数 ‖A‖',17)
line('dToMatch',[(482,610),(398,651)],red,True)
line('aToMatch',[(503,700),(398,669)],red,True)
text('matched',222,634,187,48,'方向保留 · 长度匹配<br><i>D̃ = D · ‖A‖ / ‖D‖</i>',18)
line('updateP',[(279,628),(438,628),(438,563),(710,563),(710,337),(602,337)],red,True)
text('updateLabel',711,562,156,26,'更新同一个 P',15,'fontColor='+red+';')
# small legend only, no huge empty caption
node('legendSup',59,754,orange,15,support=True);text('legendSupT',77,740,131,28,'已标注支持节点',15,'align=left;')
node('legendQry',247,754,'white',15);text('legendQryT',265,740,102,28,'未标注节点',15,'align=left;')
box('legendP',406,746,20,17,'',orange);text('legendPt',434,740,110,28,'可学习提示',15,'align=left;')
lock('legendLock',582,739);text('legendLockT',611,740,100,28,'冻结参数',15,'align=left;')
line('legendGrad',[(769,754),(822,754)],red,True);text('legendGradT',832,740,130,28,'训练反馈',15,'align=left;')

def write(path,root,width=1680,height=790):
 f=E.Element('mxfile',host='app.diagrams.net'); d=E.SubElement(f,'diagram',name='PRPrompt',id='prprompt-continuous');m=E.SubElement(d,'mxGraphModel',dx=str(width),dy=str(height),grid='0',page='1',pageWidth=str(width),pageHeight=str(height),math='0',background='#FFFFFF');m.append(root);E.indent(f);E.ElementTree(f).write(path,encoding='utf-8',xml_declaration=True)
write(out/'PRPrompt_总框架.drawio',r)
print(out)
