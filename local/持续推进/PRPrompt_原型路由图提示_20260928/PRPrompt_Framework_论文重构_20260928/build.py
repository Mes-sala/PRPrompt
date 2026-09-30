from pathlib import Path
import xml.etree.ElementTree as E
import math,copy,json
O=Path(__file__).resolve().parent
INK='#263445'; GRAY='#8291A2'; LINE='#B9C4CF'; BLUE='#477AA5'; TEAL='#278F8A'; PURPLE='#9873B1'; ORANGE='#C88439'
class Diagram:
 def __init__(self,w,h,name):
  self.file=E.Element('mxfile',host='drawio',version='26.0.0');self.diagram=E.SubElement(self.file,'diagram',id=name,name=name)
  self.model=E.SubElement(self.diagram,'mxGraphModel',page='1',pageWidth=str(w),pageHeight=str(h),grid='1',gridSize='10',math='0',background='#FFFFFF')
  self.root=E.SubElement(self.model,'root');E.SubElement(self.root,'mxCell',id='0');E.SubElement(self.root,'mxCell',id='1',parent='0');self.n=0
 def vid(self,p):self.n+=1;return p+'_'+str(self.n)
 def v(self,id,x,y,w,h,value='',style='',parent='1'):
  c=E.SubElement(self.root,'mxCell',id=id,value=value,style='html=1;whiteSpace=wrap;fontFamily=Arial;fontSize=20;fontColor='+INK+';spacing=0;'+style,vertex='1',parent=parent)
  E.SubElement(c,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),**{'as':'geometry'});return id
 def text(self,id,x,y,w,h,t,size=20,color=INK,bold=False,align='center',parent='1'):
  return self.v(id,x,y,w,h,t,f'text;fillColor=none;strokeColor=none;fontSize={size};fontColor={color};fontStyle={1 if bold else 0};align={align};verticalAlign=middle;',parent)
 def rect(self,id,x,y,w,h,t='',fill='#FFFFFF',stroke=LINE,parent='1',size=20):
  return self.v(id,x,y,w,h,t,f'rounded=1;arcSize=10;fillColor={fill};strokeColor={stroke};strokeWidth=1.5;fontSize={size};',parent)
 def group(self,id,x,y,w,h):return self.v(id,x,y,w,h,'','group;pointerEvents=0;','1')
 def line(self,id,points,color=INK,dash=False,arrow=True,parent='1',width=1.8):
  c=E.SubElement(self.root,'mxCell',id=id,value='',style=f'html=1;edgeStyle=none;rounded=1;orthogonalLoop=1;jettySize=auto;strokeColor={color};strokeWidth={width};endArrow={"block" if arrow else "none"};endSize=7;endFill=1;dashed={1 if dash else 0};dashPattern=5 4;',edge='1',parent=parent)
  g=E.SubElement(c,'mxGeometry',relative='1',**{'as':'geometry'})
  for as_,pt in [('sourcePoint',points[0]),('targetPoint',points[-1])]:E.SubElement(g,'mxPoint',x=str(pt[0]),y=str(pt[1]),**{'as':as_})
  if len(points)>2:
   a=E.SubElement(g,'Array',**{'as':'points'})
   for pt in points[1:-1]:E.SubElement(a,'mxPoint',x=str(pt[0]),y=str(pt[1]))
 def dot(self,id,x,y,r=7,color=GRAY,support=False,parent='1'):
  if support:self.v(id+'_ring',x-r-4,y-r-4,2*r+8,2*r+8,'',f'ellipse;fillColor=white;strokeColor={color};strokeWidth=1.8;',parent)
  self.v(id,x-r,y-r,2*r,2*r,'',f'ellipse;fillColor={color};strokeColor=white;strokeWidth=1;',parent)
 def diamond(self,id,x,y,color,parent):self.v(id,x-11,y-11,22,22,'',f'rhombus;fillColor={color};strokeColor=white;strokeWidth=1;',parent)
 def matrix(self,id,x,y,rows=4,cols=3,cw=19,ch=17,palette=None,parent='1',border=None):
  colors=palette or [[TEAL,PURPLE,'#DCE3E9'][j%3] for j in range(cols)]
  for i in range(rows):
   for j in range(cols):
    co=colors[i%len(colors)] if id.endswith('P') else colors[j%len(colors)]
    self.v(f'{id}_{i}_{j}',x+j*cw,y+i*ch,cw-2,ch-2,'',f'rounded=0;fillColor={co};strokeColor=none;opacity={55+((i*7+j*13)%45)};',parent)
  if border:self.v(id+'_border',x-5,y-5,cols*cw+8,rows*ch+8,'',f'rounded=0;fillColor=none;strokeColor={border};strokeWidth=1.6;',parent)
 def bars(self,id,x,y,vals,parent='1',width=85,height=12,gap=9):
  for i,v in enumerate(vals):
   for j,(xx,ww,c) in enumerate([(x,width*v,TEAL),(x+width*v,width*(1-v),PURPLE)]):
    self.v(f'{id}_{i}_{j}',xx,y+i*(height+gap),ww,height,'',f'rounded=0;fillColor={c};strokeColor=none;',parent)
 def graph(self,id,x,y,s=1,parent='1',colored=False,prob=False):
  pts=[(10,40),(65,10),(120,45),(42,90),(106,112),(165,85)]
  for n,(a,b) in enumerate([(0,1),(0,3),(1,2),(1,3),(2,4),(2,5),(3,4),(4,5)]):self.line(f'{id}_e{n}',[(x+pts[a][0]*s,y+pts[a][1]*s),(x+pts[b][0]*s,y+pts[b][1]*s)],LINE,False,False,parent,1.5)
  for i,(xx,yy) in enumerate(pts):
   co=PURPLE if i==4 else TEAL if i%2==0 else PURPLE
   self.dot(f'{id}_n{i}',x+xx*s,y+yy*s,7*s,co if colored or i in [0,4] else '#C8D3DC',i in [0,4],parent)
   if prob and i not in [0,4]:self.bars(f'{id}_p{i}',x+(xx-14)*s,y+(yy+12)*s,[.75 if i%2==0 else .25],parent,28*s,5*s,0)
 def save(self,p):E.ElementTree(self.file).write(p,encoding='utf-8',xml_declaration=True)

# Every component is native and has local coordinates. The composition imports it as a group.
def prototype(d,p,cn=True):
 T=lambda a,b:a if cn else b
 d.text('r_title',15,5,660,35,T('M1  原型路由与共享残差','M1  Prototype routing & shared residuals'),25,bold=True,align='left',parent=p)
 d.text('r_stage',20,53,660,27,T('支持类别确定组合条件，P 学习修正内容','Support classes define routing; P learns the residual'),18,GRAY,align='left',parent=p)
 d.text('r_support_label',18,99,135,28,T('支持原型','Support prototypes'),18,parent=p)
 for i,c in enumerate([TEAL,PURPLE]):
  for j in range(2):d.dot(f'r_sup{i}{j}',35+j*24,153+i*40,6,c,True,p)
  d.line(f'r_sup_arr{i}',[(78,153+i*40),(106,153+i*40)],GRAY,parent=p)
  d.diamond(f'r_proto{i}',126,153+i*40,c,p)
 d.matrix('r_attr',30,240,3,4,18,15,palette=['#A9BBCB','#CAD5DF'],parent=p)
 d.text('r_attrlabel',12,294,153,24,T('传播属性','Propagated attributes'),17,GRAY,parent=p)
 d.line('r_toassign',[(152,168),(197,168)],GRAY,parent=p)
 d.line('r_attrsassign',[(107,263),(176,263),(176,194),(197,194)],GRAY,parent=p)
 d.bars('r_soft',206,151,[.72,.25,.60],p,91,16,11)
 d.text('r_softlbl',190,111,125,28,T('软分配','Soft assignment'),18,parent=p)
 d.line('r_torefine',[(309,181),(351,181)],GRAY,parent=p)
 for i,c in enumerate([TEAL,PURPLE]):
  for j,(dx,dy) in enumerate([(-24,-9),(26,-13),(-7,20)]):d.dot(f'r_cloud{i}{j}',388+dx,150+i*75+dy,5,c,False,p)
  d.diamond(f'r_refine{i}',388,151+i*75,c,p)
 d.text('r_refinelbl',329,103,130,35,T('原型融合','Prototype fusion'),18,parent=p)
 d.line('r_iter',[(427,222),(455,222),(455,269),(248,269),(248,223)],TEAL,True,parent=p)
 d.text('r_iterlbl',308,276,140,26,T('保留支持参照','Support anchored'),17,TEAL,parent=p)
 d.line('r_toQ',[(443,171),(533,171)],GRAY,parent=p)
 d.matrix('r_Q',548,140,4,2,29,21,[TEAL,PURPLE],p)
 d.text('r_Qlbl',491,98,181,32,T('固定路由 Q','Fixed routing Q'),20,TEAL,True,parent=p)
 d.text('r_Qops',486,240,180,60,T('锐化 · 中心化<br>原图传播','Sharpen · Center<br>Graph propagation'),17,GRAY,parent=p)
 d.line('r_separator',[(20,326),(663,326)],'#E0E6EB',False,False,p,1)
 d.matrix('r_mixQ',55,380,4,2,24,17,[TEAL,PURPLE],p)
 d.text('r_mixQlabel',32,347,95,27,'Q',22,TEAL,True,parent=p)
 d.text('r_times',139,391,33,35,'×',28,parent=p)
 d.matrix('r_P',196,380,2,5,22,28,[TEAL,PURPLE],p,border=ORANGE)
 d.text('r_Plabel',174,342,160,30,T('共享残差 P','Residual bank P'),20,ORANGE,True,parent=p)
 d.text('r_Plearn',183,448,150,27,T('唯一可学习参数','Only trainable parameter'),16,ORANGE,parent=p)
 d.line('r_mixarrow',[(337,409),(394,409)],TEAL,parent=p)
 d.matrix('r_delta',424,380,4,5,19,17,[TEAL,PURPLE,'#A6B8C9'],p)
 d.text('r_deltalbl',391,341,180,30,T('节点修正 ΔU','Node residual ΔU'),19,TEAL,True,parent=p)
 d.text('r_eq',30,491,620,39,'ΔU = γ [Q tanh(P)] ⊙ s<sub>U</sub>',22,parent=p)
 d.text('r_Pnote',555,362,105,99,T('类别共享<br>节点相关','Shared<br>by class'),18,GRAY,parent=p)

def optimizer(d,p,cn=True):
 T=lambda a,b:a if cn else b
 d.text('o_title',12,5,420,35,T('M2  矩阵更新','M2  Matrix update'),25,bold=True,align='left',parent=p)
 d.text('o_sub',12,53,420,27,T('同一梯度，分开确定方向与步幅','One gradient; direction and step size'),18,GRAY,align='left',parent=p)
 d.matrix('o_G',25,164,3,4,17,17,['#A8B7C8','#CED7DF'],p)
 d.text('o_Glbl',17,221,89,28,'G = ∇<sub>P</sub>ℒ',20,parent=p)
 d.line('o_g1',[(103,181),(127,181),(127,146),(155,146)],ORANGE,parent=p)
 d.line('o_g2',[(103,192),(127,192),(127,247),(155,247)],ORANGE,parent=p)
 d.rect('o_shampoo',159,107,229,76,'Shampoo<br>'+T('矩阵方向 D','Matrix direction D'),'#F3F6FA','#A6B8CB',p,20)
 d.rect('o_adam',159,210,229,76,'Adam<br>'+T('范数参照 ‖A‖','Norm reference ‖A‖'),'#FCF6ED','#D8B78F',p,20)
 d.line('o_d',[(389,145),(410,145),(410,327),(283,327),(283,347)],ORANGE,parent=p)
 d.line('o_a',[(258,287),(258,347)],ORANGE,parent=p)
 d.rect('o_norm',125,350,270,76,T('方向保持 · 范数匹配','Preserve direction · Match norm'),'#FBF4E9','#D8B78F',p,20)
 d.text('o_form',25,442,386,34,'D̃ = D · ‖A‖<sub>F</sub> / (‖D‖<sub>F</sub> + ε)',22,parent=p)
 d.text('o_update',25,489,386,36,'P ← P − ηD̃',25,ORANGE,True,parent=p)

def readout(d,p,cn=True):
 T=lambda a,b:a if cn else b
 d.text('d_title',12,5,512,35,T('M3  支持约束读出','M3  Support-constrained readout'),25,bold=True,align='left',parent=p)
 d.text('d_sub',12,53,510,27,T('最终表示固定，仅迭代类别概率','Final embeddings fixed; iterate probabilities'),18,GRAY,align='left',parent=p)
 d.text('d_z',123,94,282,28,T('适配表示：中心化、归一化','Adapted embeddings: center & normalize'),18,BLUE,parent=p)
 d.line('d_fork1',[(257,131),(257,147),(117,147),(117,167)],GRAY,parent=p)
 d.line('d_fork2',[(257,147),(401,147),(401,167)],GRAY,parent=p)
 for i,c in enumerate([TEAL,PURPLE]):d.diamond(f'd_proto{i}',88+i*48,193,c,p)
 d.text('d_protoLbl',28,214,180,28,T('支持类别原型','Support prototypes'),18,parent=p)
 # A directed local neighbor graph, intentionally distinct from the input edges.
 for j,(xx,yy) in enumerate([(343,174),(459,171),(459,229)]):
  d.line(f'd_knn_edge{j}',[(xx,yy),(396 if xx<400 else 407,204 if yy<210 else 215)],BLUE,False,True,p,1.3+j*.6)
  d.dot(f'd_knn_n{j}',xx,yy,6,[TEAL,PURPLE,TEAL][j],False,p)
 d.dot('d_knn_target',401,209,8,INK,False,p)
 d.text('d_knnLbl',302,248,214,28,T('余弦近邻 · 相似度赋权','Cosine kNN · Similarity weights'),17,parent=p)
 d.bars('d_f0',74,259,[.8,.3,.55],p,84,11,6)
 d.line('d_proto_to_f0',[(117,245),(117,255)],TEAL,parent=p)
 d.text('d_f0label',24,317,188,26,T('固定初始概率 F₀','Fixed initial scores F₀'),18,TEAL,parent=p)
 d.text('d_wlabel',320,307,187,28,T('固定近邻权重 W','Fixed neighbor weights W'),18,BLUE,parent=p)
 d.line('d_f0tomix',[(117,344),(117,368),(185,368)],TEAL,parent=p)
 d.line('d_wtomix',[(400,341),(400,369),(371,369)],BLUE,parent=p)
 d.rect('d_iteration',188,347,180,79,T('融合 + 标签钳制','Mix + Label clamp'),'#F2F7F9','#AABFC9',p,20)
 d.line('d_loop',[(370,403),(431,403),(431,456),(276,456),(276,429)],BLUE,True,parent=p)
 d.text('d_loopLbl',339,452,103,27,T('迭代','Iterate'),17,BLUE,parent=p)
 d.text('d_formula',15,496,515,35,'F ← (1 − α)F₀ + αWF ;  F<sub>S</sub> ← Y<sub>S</sub>',20,parent=p)

# Reusable illustrated assets are standalone native diagrams, not crops from the total figure.
components=[('01_原型路由与残差',prototype,700,548),('02_矩阵更新',optimizer,442,548),('03_支持约束读出',readout,540,548)]
for name,fn,w,h in components:
 d=Diagram(w+40,h+40,name);p=d.group(name,20,20,w,h);fn(d,p,True);d.save(O/'组件'/f'{name}.drawio')
# Additional small glyphs provide a consistent visual vocabulary.
for name,kind in [('04_输入图','input'),('05_适配表示','embedding'),('06_预测图','output')]:
 d=Diagram(240,190,name);p=d.group(name,20,20,200,150)
 if kind=='embedding':d.matrix('asset_z',47,27,5,5,22,19,[TEAL,PURPLE,'#9FB5C8'],p)
 else:d.graph('asset_graph',15,10,1,p,kind=='output',kind=='output')
 d.save(O/'组件'/f'{name}.drawio')

def total(cn):
 T=lambda a,b:a if cn else b
 d=Diagram(1820,950,'PRPrompt CN' if cn else 'PRPrompt EN')
 # Compact left-to-right forward pass. Only the learned residual enters the frozen encoder.
 d.text('name',32,5,260,47,'PRPrompt',32,bold=True,align='left')
 d.text('context',252,13,900,34,T('原型路由图提示','Prototype-routed graph prompting'),23,GRAY,align='left')
 d.graph('input',34,108,.83)
 d.text('input_l',25,221,180,34,T('输入图 (X, E)','Input graph (X, E)'),21)
 d.line('toenc1',[(191,166),(243,166)],GRAY)
 d.rect('enc1',247,128,163,76,T('冻结 GCN₁','Frozen GCN₁'),'#EEF3F7','#A5B6C6',size=23)
 d.line('toadd',[(412,166),(472,166)],GRAY)
 d.v('add',479,143,48,48,'+','ellipse;fillColor=white;strokeColor='+TEAL+';strokeWidth=2;fontSize=31;')
 d.line('encnext',[(530,166),(632,166)],GRAY)
 d.text('activation',527,196,112,24,'LeakyReLU',16,GRAY)
 d.rect('enc2',639,128,163,76,T('冻结 GCN₂','Frozen GCN₂'),'#EEF3F7','#A5B6C6',size=23)
 d.line('toZ',[(805,166),(861,166)],GRAY)
 d.matrix('Z',884,122,5,4,20,18,[TEAL,PURPLE,'#A6B8C9'])
 d.text('Z_l',840,224,183,30,T('适配表示 Z<sub>P</sub>','Adapted embeddings Z<sub>P</sub>'),21)
 d.line('toRead',[(977,166),(1127,166)],GRAY)
 d.rect('readout',1136,128,255,76,T('支持约束概率传播','Support-constrained<br>probability propagation'),'#F0F6F8','#99B5C6',size=22)
 d.text('readout_stage',1134,215,260,25,T('训练后读出','Post-adaptation readout'),17,GRAY)
 d.line('toout',[(1395,166),(1464,166)],GRAY)
 d.graph('output',1484,110,.83,colored=True)
 d.text('output_l',1460,226,215,29,T('节点类别预测','Node predictions'),21)
 # Residual is the only trainable insertion, its origin is expanded in M1.
 d.rect('delta',416,265,173,55,'ΔU', '#EEF7F4',TEAL,size=25)
 d.line('inject',[(502,262),(502,194)],TEAL,width=2.4)
 d.text('delta_l',208,277,195,30,T('原型路由残差','Prototype-routed residual'),19,TEAL)
 # Training branch is visually subordinate and stops at P, never at readout.
 d.line('training_loss',[(978,178),(1003,178),(1003,241),(1074,241),(1074,257)],ORANGE,True)
 d.rect('loss',996,258,157,62,T('支持损失 ℒ<br>仅更新 P','Support loss ℒ<br>Update P only'),'#FEF8F0','#DAB991',size=18)
 d.line('to_optimizer',[(1074,323),(1074,350)],ORANGE,True)
 # Detail strips use a thin top rule and a very light tint rather than heavy boxed panels.
 for id,x,w,c in [('m1',25,700,TEAL),('m2',753,442,ORANGE),('m3',1223,570,BLUE)]:
  d.v(id+'_background',x,361,w,567,'','rounded=0;fillColor=#FFFFFF;strokeColor=#DDE4E9;strokeWidth=1;')
  d.line(id+'_rule',[(x,362),(x+w,362)],c,False,False,width=3)
 p=d.group('m1',32,375,688,548);prototype(d,p,cn)
 p=d.group('m2',760,375,430,548);optimizer(d,p,cn)
 p=d.group('m3',1231,375,550,548);readout(d,p,cn)
 d.line('M1_residual_detail',[(502,360),(502,323)],TEAL,True)
 d.line('M3_detail',[(1263,245),(1263,350)],BLUE,True,False)
 return d
for cn in [True,False]:total(cn).save(O/('PRPrompt_Framework_CN.drawio' if cn else 'PRPrompt_Framework_EN.drawio'))
# One multi-page library, with independent mechanism and glyph pages.
lib=E.Element('mxfile',host='drawio')
for f in sorted((O/'组件').glob('*.drawio')):lib.append(copy.deepcopy(E.parse(f).find('diagram')))
E.ElementTree(lib).write(O/'PRPrompt_组件库.drawio',encoding='utf-8',xml_declaration=True)
print('Built fresh CN/EN compositions and 6 component files')
