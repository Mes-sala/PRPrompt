from pathlib import Path
import xml.etree.ElementTree as E
import base64
out=Path(__file__).parent
r=E.Element('root');E.SubElement(r,'mxCell',id='0');E.SubElement(r,'mxCell',id='1',parent='0')
blue='#8AB5CD';orange='#ECA075';ink='#384047'
def shape(id,x,y,w,h,fill='none',style='',value=''):
 c=E.SubElement(r,'mxCell',id=id,value=value,vertex='1',parent='1',style=f'html=1;whiteSpace=wrap;fontFamily=Arial;fontSize=20;fontColor={ink};strokeColor={ink};strokeWidth=1.5;fillColor={fill};'+style);E.SubElement(c,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),**{'as':'geometry'});return c

def label(id,x,y,w,s,size=20):shape(id,x,y,w,32,style=f'strokeColor=none;fontSize={size};',value=s)
def pic(id,key,x,y,w,h):
 data=base64.b64encode((out/'组件'/f'{key}.png').read_bytes()).decode();shape(id,x,y,w,h,style='shape=image;imageAspect=0;aspect=fixed;strokeColor=none;image=data:image/png,'+data+';')
def edge(id,pts,color=ink,dash=False,arrow=True):
 c=E.SubElement(r,'mxCell',id=id,edge='1',parent='1',style=f'html=1;strokeWidth=1.8;strokeColor={color};endArrow={"block" if arrow else "none"};endSize=7;'+('dashed=1;dashPattern=5 4;' if dash else ''))
 g=E.SubElement(c,'mxGeometry',relative='1',**{'as':'geometry'});E.SubElement(g,'mxPoint',x=str(pts[0][0]),y=str(pts[0][1]),**{'as':'sourcePoint'});E.SubElement(g,'mxPoint',x=str(pts[-1][0]),y=str(pts[-1][1]),**{'as':'targetPoint'})
 if len(pts)>2:
  a=E.SubElement(g,'Array',**{'as':'points'})
  for x,y in pts[1:-1]:E.SubElement(a,'mxPoint',x=str(x),y=str(y))
def dot(id,x,y,col,size=14):shape(id,x,y,size,size,col,'ellipse;')
def vector(id,x,y,cols):
 for j,c in enumerate(cols):shape(id+str(j),x+j*12,y,12,15,c,'strokeWidth=0.8;')
# All image objects first so arrows/labels never disappear behind white backgrounds.
pic('input','flat_encoder',25,275,175,117)
pic('proto','simple_proto',167,13,330,165)
pic('route','flat_route',616,51,177,118)
pic('encoder','simple_encoder_clean',742,255,296,197)
pic('update','simple_update_clean',449,477,170,114)
pic('prop','simple_prop',1210,261,336,168)
# M1 pictorial ribbon
label('m1',184,5,305,'M1  原型路由提示',23)
label('protoName',244,155,186,'原型凝聚 · 迭代',19)
edge('inputToProto',[(111,271),(111,98),(229,98)])
label('source',24,202,171,'属性与支持标签',17)
edge('protoToSharp',[(455,101),(486,101)])
# confidence sharpening as signed columns straddling baseline
edge('baseline',[(498,114),(575,114)],'#ADB2B4',False,False)
for j,(v,c) in enumerate([(17,blue),(38,orange),(-13,blue),(-24,orange)]):shape('sharp'+str(j),505+j*17,114-max(v,0),10,abs(v),c,'strokeColor=none;')
label('sharpLabel',465,155,146,'锐化 · 中心化',18)
edge('sharpToRoute',[(582,104),(609,104)])
label('routeLabel',611,183,190,'原图传播 · 固定路由',18)
# shared bank depicted as separate colored residual strips
vector('bankA',640,259,[orange,'#F7D4BA',orange,'#F7D4BA'])
vector('bankB',650,283,['#F7D4BA',orange,'#F7D4BA',orange])
label('bankLabel',569,310,196,'共享残差',19)
edge('routeToPrompt',[(801,111),(856,111)],orange)
edge('bankToPrompt',[(708,280),(822,280),(822,148),(856,111)],orange)
for j,cols in enumerate([[orange,blue,orange,blue],[blue,orange,orange,blue],[blue,blue,orange,orange]]):vector('nodePrompt'+str(j),867+j*6,87+j*23,cols)
label('promptName',830,173,172,'节点提示',19)
edge('promptInject',[(900,211),(900,251)],orange)
label('injectLabel',932,210,156,'隐藏层注入',17)
# continuous main computation
label('inputLabel',16,400,187,'输入图',21)
edge('inputToEncode',[(208,338),(771,338)])
label('frozen',782,452,221,'冻结图编码器',21)
edge('encodeToEmbed',[(1021,376),(1056,376)])
# adapted representation simple cluster
for j,(x,y,c) in enumerate([(1068,352,orange),(1092,337,orange),(1097,369,orange),(1129,357,blue),(1143,336,blue),(1153,379,blue)]):dot('z'+str(j),x,y,c)
label('zLabel',1026,404,181,'适配表示',21)
# M3 two inputs, initial probability and nearest-neighbor graph
label('m3',1236,5,278,'M3  支持约束读出',23)
edge('embedToProb',[(1110,329),(1110,104),(1220,104)])
for j in range(3):vector('prob'+str(j),1240,73+j*21,([orange,orange,blue,blue] if j==0 else [blue,orange,blue,blue] if j==1 else [orange,orange,orange,blue]))
label('probName',1207,144,170,'原型初始概率',19)
edge('probToProp',[(1300,105),(1468,105),(1468,297)])
label('reinject',1384,168,133,'回注',18)
edge('embedToProp',[(1172,370),(1254,345)])
label('knn',1173,393,192,'适配表示近邻',18)
label('propName',1289,428,199,'邻域融合 · 迭代',20)
# support constraints little double rings with fixed color
shape('anchorA',1335,508,24,24,orange,'ellipse;strokeWidth=3;');shape('anchorB',1375,508,24,24,blue,'ellipse;strokeWidth=3;')
edge('supportClamp',[(1405,520),(1490,520),(1490,407),(1450,393)],'#777777',True)
label('clampName',1260,543,215,'支持标签固定',18)
edge('propToOutput',[(1505,343),(1545,343)])
for j,c in enumerate([orange,blue,orange]):dot('out'+str(j),1570,305+j*33,c,21)
label('outLabel',1530,413,112,'节点预测',20)
# M2 auxiliary update around same parameter bank
label('m2',247,474,199,'M2  结构化更新',22)
label('direction',434,597,189,'矩阵方向',19)
edge('supportLoss',[(1090,435),(1090,637),(413,637),(413,537),(440,537)],'#A27059',True)
label('supportLossLabel',940,595,155,'支持监督',18)
# two same-length arrows represent matched magnitude, not arithmetic
edge('normA',[(780,516),(847,516)],blue)
edge('normB',[(780,546),(847,546)],orange)
label('normName',739,597,163,'步幅匹配',19)
# Gradient goes to matrix-direction icon, then magnitude matching.
edge('directionToMatch',[(627,543),(758,543)],'#A27059',True)
edge('updateToBank',[(792,502),(779,460),(779,278),(708,278)],'#A27059',True)
label('updateSame',651,420,122,'更新残差',17)
# write no mathematical operator glyphs and no explanatory text boxes
f=E.Element('mxfile',host='app.diagrams.net');d=E.SubElement(f,'diagram',id='prprompt-illustrated',name='PRPrompt');m=E.SubElement(d,'mxGraphModel',pageWidth='1670',pageHeight='655',grid='0',background='#FFFFFF');m.append(r);E.indent(f);E.ElementTree(f).write(out/'PRPrompt_图案框架.drawio',encoding='utf-8',xml_declaration=True)
print('image components',sum('shape=image' in c.get('style','') for c in r),'native cells',len(r))
