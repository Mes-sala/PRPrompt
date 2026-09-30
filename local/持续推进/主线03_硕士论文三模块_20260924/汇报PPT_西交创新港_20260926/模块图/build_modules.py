from pathlib import Path
import xml.etree.ElementTree as E
import math,json,re
OUT=Path(__file__).resolve().parent
W,H=1800,900
INK='#182C41';GRAY='#536779';LINE='#9EB1C0';BLUE='#2B75B9';TEAL='#008E87';PURPLE='#9056C3';ORANGE='#D47A23';C=[TEAL,PURPLE]
items=[];labels=[];seq=0

def cell(style,x,y,w,h,value='',z=5):
 global seq
 seq+=1;c=E.Element('mxCell',id=f'c{seq}',value=value,style=style,vertex='1',parent='1')
 E.SubElement(c,'mxGeometry',x=str(round(x,3)),y=str(round(y,3)),width=str(round(w,3)),height=str(round(h,3)),attrib={'as':'geometry'})
 items.append((z,seq,c));return c

def shape(x,y,w,h,s='rectangle',fill='none',stroke='none',sw=3,extra='',z=5):
 return cell(f'shape={s};html=1;fillColor={fill};strokeColor={stroke};strokeWidth={sw};shadow=0;{extra}',x,y,w,h,z=z)

def text(x,y,s,size=36,color=INK,bold=False,width=None,z=15):
 lines=s.split('\n');width=width or max(sum(1 if ord(c)>255 else .56 for c in l)*size for l in lines)+18
 h=size*(1.35*len(lines)+.2)
 val=s.replace('\n','<br>')
 cell(f'text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=nowrap;overflow=visible;fontFamily=PingFang SC;fontSize={size};fontColor={color};fontStyle={int(bold)};spacing=0;',x-width/2,y-h/2,width,h,val,z)
 labels.append({'text':s,'size':size,'x':x,'y':y})

def mathtext(x,y,s,size=42,color=ORANGE,width=480,height=90):
 cell(f'text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;overflow=fill;fontFamily=Times New Roman;fontSize={size};fontColor={color};spacing=0;',x-width/2,y-height/2,width,height,s,15)
 labels.append({'text':s,'size':size,'x':x,'y':y,'mathjax':True})

def edge(points,color=GRAY,sw=4,head=True,dash=False,z=3):
 global seq
 seq+=1;c=E.Element('mxCell',id=f'c{seq}',value='',style=f'html=1;strokeColor={color};strokeWidth={sw};endArrow={"block" if head else "none"};endSize=15;endFill=1;rounded=1;arcSize=16;dashed={int(dash)};dashPattern=8 5;',edge='1',parent='1')
 g=E.SubElement(c,'mxGeometry',relative='1',attrib={'as':'geometry'})
 for p,k in [(points[0],'sourcePoint'),(points[-1],'targetPoint')]:E.SubElement(g,'mxPoint',x=str(p[0]),y=str(p[1]),attrib={'as':k})
 if len(points)>2:
  a=E.SubElement(g,'Array',attrib={'as':'points'})
  for p in points[1:-1]:E.SubElement(a,'mxPoint',x=str(p[0]),y=str(p[1]))
 items.append((z,seq,c));return c

def arrow(a,b,color=GRAY,sw=4,dash=False):return edge([a,b],color,sw,True,dash)
def line(a,b,color=LINE,sw=3,dash=False):return edge([a,b],color,sw,False,dash)
def blockarrow(a,b,color=ORANGE,h=30,stroke=None,z=8):
 dx=b[0]-a[0];dy=b[1]-a[1];L=math.hypot(dx,dy);angle=math.degrees(math.atan2(dy,dx));cx=(a[0]+b[0])/2;cy=(a[1]+b[1])/2
 shape(cx-L/2,cy-h/2,L,h,'mxgraph.arrows2.arrow',color,stroke or color,3.5,f'direction=east;dy=0.4;dx={min(40,L*.3)};notch=0;rotation={angle};',z)

def dot(x,y,c=BLUE,r=13,ring=False):
 shape(x-r,y-r,2*r,2*r,'ellipse',c,'#FFFFFF',3,z=8)
 if ring:shape(x-r-6,y-r-6,2*r+12,2*r+12,'ellipse','none',INK,3.5,z=9)

def star(x,y,c=TEAL,r=25,outline=False):shape(x-r,y-r,2*r,2*r,'mxgraph.basic.star','#FFFFFF' if outline else c,c,3.5,z=10)
def diamond(x,y,c=TEAL,r=20):shape(x-r,y-r,2*r,2*r,'rhombus',c,'#FFFFFF',3,z=10)
def pie(x,y,p,r=22,ring=False):
 if p<=0 or p>=1:shape(x-r,y-r,r*2,r*2,'ellipse',C[0] if p>=1 else C[1],'#FFFFFF',2.5,z=8)
 else:shape(x-r,y-r,r*2,r*2,'mxgraph.mockup.graphics.pieChart','none','#FFFFFF',2.5,f'partsCount=2;parts={round(p*1000000)},{round((1-p)*1000000)};partColors={C[0]},{C[1]};',8)
 shape(x-r,y-r,2*r,2*r,'ellipse','none',INK,2.5,z=9)
 if ring:shape(x-r-6,y-r-6,2*r+12,2*r+12,'ellipse','none',INK,3.5,z=10)

def bar(x,y,p,w=180,h=32,number=False):
 shape(x,y,w*p,h,fill=C[0],stroke='#FFFFFF',sw=2.2)
 shape(x+w*p,y,w*(1-p),h,fill=C[1],stroke='#FFFFFF',sw=2.2)
 shape(x,y,w,h,fill='none',stroke=INK,sw=2.8)
 if number:
  text(x+w*p/2,y+h/2,f'{p:.1f}',27,'#FFFFFF');text(x+w*p+w*(1-p)/2,y+h/2,f'{1-p:.1f}',27,'#FFFFFF')

def vector(x,y,w=170,rows=2,color=BLUE,classes=False):
 for i in range(rows):
  c=C[i%2] if classes else color
  shape(x,y+42*i,w,31,'rectangle','#F3F8FC',c,3,'rounded=1;absoluteArcSize=1;arcSize=28;',4)
  for j in range(5):
   d=14 if (j+i)%3 else 18
   dot(x+22+j*(w-44)/4,y+15+42*i,c,d/2)

def lock(x,y):shape(x-14,y-17,28,34,'mxgraph.ios7.icons.locked','#FFFFFF',BLUE,3,z=12)
def encoder(cx,cy,label):
 shape(cx-58,cy-63,116,126,'hexagon','#EAF3FA',BLUE,3.8,'fixedSize=1;size=23;',2)
 pts=[(cx-30,cy-22),(cx+28,cy-30),(cx,cy+4),(cx-30,cy+35),(cx+32,cy+33)]
 for i,j in [(0,1),(0,2),(1,2),(2,3),(2,4),(3,4)]:line(pts[i],pts[j],BLUE,2.8)
 for p in pts:dot(*p,BLUE,7)
 lock(cx,cy-92);text(cx,cy+96,label,34,BLUE,True)

def graph(cx,cy,scale=1,probs=None,directed=False):
 pts=[(cx-75*scale,cy-35*scale),(cx+40*scale,cy-60*scale),(cx+78*scale,cy+48*scale),(cx-27*scale,cy+68*scale),(cx-2*scale,cy-3*scale)]
 pairs=[(0,1),(0,4),(1,2),(1,4),(2,3),(3,4)] if not directed else [(0,4),(1,4),(2,4),(3,4)]
 for i,j in pairs:
  a,b=pts[i],pts[j]
  if directed:
   dx=b[0]-a[0];dy=b[1]-a[1];L=math.hypot(dx,dy);a=(a[0]+dx/L*24,a[1]+dy/L*24);b=(b[0]-dx/L*27,b[1]-dy/L*27)
  edge([a,b],LINE,4,head=directed)
 for i,p in enumerate(pts):
  if probs is None:dot(*p,(C[0] if i==0 else C[1]) if i in [0,2] else BLUE,14,ring=i in [0,2])
  else:pie(*p,probs[i],r=24,ring=i in [0,2])
 return pts

def save(name):
 model=E.Element('mxGraphModel',dx=str(W),dy=str(H),page='1',pageScale='1',pageWidth=str(W),pageHeight=str(H),background='#ffffff',math='1',grid='1',gridSize='10')
 root=E.SubElement(model,'root');E.SubElement(root,'mxCell',id='0');E.SubElement(root,'mxCell',id='1',parent='0')
 # Explicit invisible page extent preserves the same aspect ratio in native exports.
 shape(0,0,W,H,fill='#FFFFFF',stroke='none',z=-10)
 for _,_,c in sorted(items):root.append(c)
 xml=E.tostring(model,encoding='unicode');p=OUT/(name+'.drawio')
 if not p.exists():p.write_text('<mxfile host="draw.io"><diagram id="framework" name="'+name+'"><mxGraphModel><root><mxCell id="0"/><mxCell id="1" parent="0"/></root></mxGraphModel></diagram></mxfile>')
 (OUT/(name+'_mcp.json')).write_text(json.dumps({'path':str(p),'page':'framework','content':xml},ensure_ascii=False))
 (OUT/(name+'_检查.json')).write_text(json.dumps({'native_cells':len(items),'labels':labels,'palette':[BLUE,TEAL,PURPLE,ORANGE],'canvas':[W,H],'minimum_main_font':32},ensure_ascii=False,indent=2))
 print(name,len(items))

def reset():
 global items,labels,seq
 items=[];labels=[];seq=0

# M1: two simultaneous class-weighted means and an anchored prototype loop.
text(162,66,'原始支持原型',40,bold=True)
for cy,c in [(195,TEAL),(330,PURPLE)]:
 for dx,dy in [(-70,-25),(-70,25)]:dot(150+dx,cy+dy,c,15,True)
 arrow((114,cy),(171,cy),c);star(214,cy,c,28)
text(155,430,'无标签特征',37,bold=True)
for xx,yy in [(90,493),(215,500),(155,564)]:dot(xx,yy,BLUE,16)
text(450,66,'软分配',40,bold=True)
for i,p in enumerate([.8,.6,.2]):
 text(339,220+i*80,'u'+str(i+1),32,GRAY);bar(368,201+i*80,p,185,38)
text(460,447,'每行是类别权重',32,GRAY)
arrow((261,244),(341,244),BLUE)
edge([(248,514),(300,514),(300,414),(350,414)],BLUE)
text(780,72,'按类别加权\n求软原型',38,bold=True)
pts=[(-68,-45),(74,-23),(0,68)]
weights=[[.50,.375,.125],[.142857,.285714,.571429]]
for q,(cy,c) in enumerate([(216,TEAL),(416,PURPLE)]):
 center=(sum(p[0]*v for p,v in zip(pts,weights[q]))+780,sum(p[1]*v for p,v in zip(pts,weights[q]))+cy)
 for p,v in zip(pts,weights[q]):
  pp=(780+p[0],cy+p[1]);line(pp,center,c,3+v*7);dot(*pp,c,10+v*22)
 star(*center,c,26)
text(780,544,'除以该类权重总和',32,GRAY)
arrow((573,283),(656,283),BLUE)
text(1130,72,'支持锚定\n修正类别原型',38,bold=True)
for cy,c in [(216,TEAL),(416,PURPLE)]:
 line((1030,cy),(1240,cy),c,5);star(1030,cy,c,26,True);star(1240,cy,c,26);diamond(1135,cy,c,23)
text(1030,528,'原支持',32,GRAY);text(1238,528,'软原型',32,GRAY)
text(1135,305,'各占 50%',34,GRAY)
arrow((894,290),(988,290),BLUE)
edge([(1140,469),(1140,594),(447,594),(447,465)],BLUE,4,True,True)
text(794,594,'循环更新',34,BLUE,bold=True)
shape(683,567,222,54,fill='#FFFFFF',stroke='none',z=13)
# keep loop label above the white knockout
text(794,594,'循环更新',34,BLUE,bold=True,z=16)
text(1554,66,'固定有符号门控',40,bold=True)
text(1392,169,'锐化',34,ORANGE,bold=True)
arrow((1278,225),(1466,225),ORANGE)
for j,c in enumerate(C):
 x=1510+j*125
 line((x,190),(x,420),GRAY,3)
 for i,v in enumerate(([.70,-.28,.20],[-.42,.66,-.12])[j]):
  yy=222+i*75;shape(x+min(0,v*100),yy-15,abs(v*100),30,fill=c,stroke='none')
 text(x,448,'0',30,GRAY)
text(1556,510,'类别提示 P',37,bold=True)
vector(1480,552,160,2,classes=True)
edge([(1685,414),(1725,414),(1725,683),(1380,683)],BLUE)
edge([(1535,652),(1535,683),(1094,683)],BLUE)
text(1365,651,'按节点组合',32,GRAY)
text(999,623,'节点修正',35,bold=True)
vector(913,660,172,1)
edge([(999,701),(999,718),(1000,718),(1000,747)],BLUE)
# Shared frozen encoder pathway below.
graph(150,746,.78)
text(151,864,'输入图',34,GRAY)
arrow((240,755),(362,755),BLUE)
encoder(430,755,'冻结第 1 层')
arrow((493,755),(651,755),BLUE)
vector(672,722,167,2)
text(758,866,'隐藏特征',34,GRAY)
# Same feature with correction injection, no decorative plus/multiply glyph.
arrow((845,755),(1134,755),BLUE)
text(982,835,'激活前注入',32,ORANGE,bold=True)
encoder(1200,755,'冻结第 2 层')
arrow((1268,755),(1387,755),BLUE)
vector(1410,722,174,2)
text(1496,866,'适配表示',34,GRAY)
text(1700,755,'仅 P\n可训练',32,BLUE,bold=True)
text(1685,829,'门控固定',30,GRAY)
save('M1_类别语义引导与软原型');reset()

# M2: optimizer names are explicit; geometry carries norm/direction semantics.
text(258,55,'唯一提示矩阵 P',40,bold=True)
vector(164,104,188,2,classes=True)
arrow((258,200),(258,238),BLUE)
shape(160,252,196,100,'hexagon','#EAF3FA',BLUE,3.6,'fixedSize=1;size=24;',3)
text(258,302,'支持集损失',35,BLUE,bold=True)
arrow((258,365),(258,409),BLUE)
text(258,450,'同一梯度 G',40,bold=True)
vector(164,500,188,2)
edge([(160,552),(92,552),(92,683),(278,683)],BLUE,4)
edge([(352,552),(696,552),(696,611)],BLUE,4)
shape(290,625,245,118,'hexagon','#EAF3FA',BLUE,3.6,'fixedSize=1;size=23;',3)
text(412,684,'Shampoo',40,BLUE,bold=True)
shape(597,625,198,118,'hexagon','#F3F5F7',GRAY,3.6,'fixedSize=1;size=23;',3)
text(696,684,'Adam',40,GRAY,bold=True)
text(412,788,'提供方向 D',36,BLUE,bold=True)
text(696,788,'提供长度参照 A',34,GRAY,bold=True)
text(570,856,'两条统计分支来自同一参数轨迹',32,GRAY)
# Same-origin vectors. Direction endpoints satisfy exact geometric constraints.
o=(1180,446);R=238;theta=-2.42
D=(o[0]+94*math.cos(theta),o[1]+94*math.sin(theta))
Dt=(o[0]+R*math.cos(theta),o[1]+R*math.sin(theta))
A=(o[0]+R*math.cos(-.31),o[1]+R*math.sin(-.31))
shape(o[0]-R,o[1]-R,2*R,2*R,'ellipse','none',LINE,4,'dashed=1;dashPattern=9 6;',1)
blockarrow(o,Dt,ORANGE,38,stroke='#B65F16',z=7)
blockarrow(o,D,BLUE,24,stroke=BLUE,z=8)
blockarrow(o,A,'#CBD6DE',31,stroke=GRAY,z=6)
dot(*o,INK,8)
text(1158,71,'方向保持 · 全矩阵范数匹配',42,bold=True)
text(1035,139,'适配后方向',38,ORANGE,bold=True)
mathtext(1170,149,r'\(\widetilde{D}\)',38,width=92,height=80)
text(1110,186,'与 D 同射线，仅缩放长度',34,ORANGE)
edge([(966,214),(1009,280)],ORANGE,3,False)
text(936,503,'原方向 D',36,BLUE,bold=True)
line((1008,493),(1120,401),BLUE,3)
text(1470,288,'Adam 参照 A',38,GRAY,bold=True)
text(1548,435,'同一起点',33,GRAY)
line((1460,435),(1198,446),LINE,3,True)
text(1224,715,'圆半径 = Adam 的整体范数',33,GRAY)
mathtext(1208,782,r'\(\lVert\widetilde{D}\rVert_F=\lVert A\rVert_F\)',44,width=520,height=100)
# parameter movement is opposite to the update-direction candidate.
text(1590,560,'实际参数更新',38,bold=True)
vector(1492,609,193,2,classes=True)
blockarrow((1552,710),(1552-120*math.cos(theta),710-120*math.sin(theta)),ORANGE,28)
mathtext(1570,836,r'\(P\leftarrow P-\eta\widetilde{D}\)',37,width=410,height=90)
text(1600,888,'沿负方向移动',28,GRAY)
assert abs(math.dist(o,Dt)-math.dist(o,A))<1e-6
assert abs((D[0]-o[0])*(Dt[1]-o[1])-(D[1]-o[1])*(Dt[0]-o[0]))<1e-6
save('M2_更新方向与尺度匹配');reset()

# M3: representation branch fixes both W and F0; propagation itself changes probabilities.
text(190,63,'最终适配表示',40,bold=True)
vector(95,129,190,3)
text(180,318,'同一份表示',33,GRAY)
text(655,63,'支持原型分类',40,bold=True)
for cy,c in [(182,TEAL),(298,PURPLE)]:
 dot(463,cy-19,c,14,True);dot(463,cy+19,c,14,True);arrow((495,cy),(558,cy),c);star(610,cy,c,25)
text(820,182,'原始概率 F₀',37,bold=True)
for i,p in enumerate([.70,.40,.20]):bar(729,233+i*48,p,190,31)
text(827,419,'始终保留原值',33,BLUE,bold=True)
arrow((926,247),(1005,247),BLUE)
text(976,194,'初始化',30,BLUE)
edge([(302,167),(364,167),(364,229),(420,229)],BLUE)
arrow((647,236),(708,236),BLUE)
# Representation retrieval is visible and distinct from the original graph.
text(292,491,'相似度检索近邻',40,bold=True)
pts=[(110,576),(264,567),(350,633),(251,705),(91,693)]
for i,p in enumerate(pts):dot(*p,(C[0] if i==0 else C[1]) if i in [0,2] else BLUE,17,ring=i in [0,2])
shape(195,602,93,93,'ellipse','none',BLUE,3.4,'dashed=1;dashPattern=7 5;',3)
for i in [0,1,2]:edge([pts[i],(243,648)],LINE,4,False)
dot(243,648,ORANGE,17)
text(238,791,'排除自身后选近邻',33,GRAY)
edge([(91,218),(40,218),(40,576),(76,576)],BLUE)
text(580,490,'相似度赋权',38,bold=True)
for i,(name,v) in enumerate([('a',.8),('b',.6),('c',.4)]):
 text(472,579+i*74,name,34,GRAY);shape(505,562+i*74,170*v,29,fill=BLUE,stroke='none')
arrow((369,642),(448,642),BLUE)
text(837,490,'固定近邻图 W',38,bold=True)
graph(828,646,1.00,directed=True)
text(836,796,'与原图边不同',33,GRAY)
arrow((672,642),(720,642),BLUE)
# Right half concentrates the probability iteration and anchored readout.
text(1342,63,'传播类别概率',42,bold=True)
graph(1113,243,.80,probs=[1,.4,0,.45,.55],directed=True)
text(1111,376,'当前概率',36,bold=True)
graph(1557,243,.80,probs=[1,.62,0,.59,.69],directed=True)
text(1556,376,'更新概率',36,bold=True)
blockarrow((1240,240),(1420,240),BLUE,28)
text(1326,177,'邻居加权',33,BLUE,bold=True)
edge([(1591,326),(1678,326),(1678,431),(1075,431),(1075,339)],BLUE,4,True,True)
shape(1228,406,289,51,fill='#FFFFFF',stroke='none',z=13)
text(1373,431,'循环更新',35,BLUE,bold=True,z=16)
edge([(916,641),(976,641),(976,301),(1021,301)],BLUE)
# F0 anchor and neighbor mix are represented by two strips converging into the output.
text(1108,503,'原始 F₀',35,bold=True)
bar(1020,544,.55,196,37);text(1275,529,'10%',35,GRAY)
text(1108,640,'邻居加权概率',35,bold=True)
bar(1020,681,.763,196,37);text(1275,663,'90%',35,GRAY)
edge([(923,284),(960,284),(960,562),(1005,562)],BLUE)
edge([(1219,561),(1350,561),(1350,630),(1400,630)],BLUE)
edge([(1219,700),(1350,700),(1350,630),(1400,630)],BLUE)
bar(1430,611,.742,241,40)
text(1550,569,'新的类别概率',37,bold=True)
text(1494,757,'支持标签恢复为已知类别',34,bold=True)
pie(1440,823,1,24,True);pie(1540,823,0,24,True)
shape(1609,797,31,39,'mxgraph.ios7.icons.locked','#FFFFFF',INK,3.5,z=12)
edge([(1550,668),(1550,720)],BLUE)
edge([(1668,823),(1740,823),(1740,204),(1660,204)],BLUE,4,True)
save('M3_支持约束概率传播')
