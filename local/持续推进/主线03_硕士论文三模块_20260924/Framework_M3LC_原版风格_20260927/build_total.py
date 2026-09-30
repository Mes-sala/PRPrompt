from pathlib import Path
import xml.etree.ElementTree as E
import copy, re
OUT=Path(__file__).resolve().parent;BASE=OUT.parent
TEAL='#008E87';PURPLE='#9056C3';BLUE='#2B75B9';INK='#182C41';GRAY='#44586D';ORANGE='#D47A23'
for lang in ['CN','EN']:
 doc=E.parse(BASE/'论文总Framework_鲜明大字双语版_20260926'/f'Framework_{lang}.drawio');model=doc.find('.//mxGraphModel');root=model.find('root');model.set('pageHeight','1460')
 def tr(a,b):return a if lang=='CN' else b
 def get(n):return next(c for c in root if c.get('id')==n)
 def replace(n,v,size=None,w=None):
  c=get(n);c.set('value',v)
  if size:c.set('style',re.sub(r'fontSize=[^;]+',f'fontSize={size}',c.get('style')))
  if w:
   g=c.find('mxGeometry');cx=float(g.get('x'))+float(g.get('width'))/2;g.set('x',str(cx-w/2));g.set('width',str(w))
 def route(n,pts):
  g=get(n).find('mxGeometry');g.clear();g.attrib.update({'relative':'1','as':'geometry'})
  E.SubElement(g,'mxPoint',x=str(pts[0][0]),y=str(pts[0][1]),**{'as':'sourcePoint'});E.SubElement(g,'mxPoint',x=str(pts[-1][0]),y=str(pts[-1][1]),**{'as':'targetPoint'});a=E.SubElement(g,'Array',**{'as':'points'})
  for x,y in pts[1:-1]:E.SubElement(a,'mxPoint',x=str(x),y=str(y))
 # Preserve all existing M1/M2/encoder primitives; only shift the readout zoom down.
 for c in root:
  m=re.fullmatch(r'c(\d+)',c.get('id',''));n=int(m[1]) if m else 0
  if 429<=n<=500:
   g=c.find('mxGeometry')
   if g is not None:
    if g.get('y'):g.set('y',str(float(g.get('y'))+320))
    for pt in g.iter('mxPoint'):
     if pt.get('y'):pt.set('y',str(float(pt.get('y'))+320))
 get('c373').find('mxGeometry').set('height','1690')
 replace('c375',tr('训练后：局部一致性校准读出','Post-training: calibrated readout'),36,650)
 replace('c376',tr('教师属性相似度','Attribute similarity'),29,240)
 replace('c386',tr('中心化传播属性','Centered propagated features'),24,265)
 replace('c405',tr('教师属性近邻图','Attribute graph'),29,275)
 replace('c424',tr('初始预测概率','Initial probabilities'),31,270)
 replace('c425',tr('由 Prompt 分数 s 产生','From Prompt scores s'),26,360)
 replace('c466',tr('校准后的 \\(F_0\\)','Calibrated \\(F_0\\)'),26,250)
 replace('c465',tr('迭代','Iterate'),29,230)
 replace('c428',tr('固定<br>W','Fixed<br>W'),28)
 # No adapted-representation-to-neighbor path remains.
 route('c318',[(1319,651),(1445,651),(1445,532),(1497,532)])
 root.remove(get('c319'))
 route('c426',[(1930,935),(1960,935),(1960,983),(1615,983),(1615,1033)])
 route('c427',[(2059,388),(2160,388),(2160,976),(2059,976)])
 counter=0
 def cell(style,x,y,w,h,value=''):
  global counter
  counter+=1;c=E.SubElement(root,'mxCell',id='lc_'+str(counter),parent='1',vertex='1',value=value,style=style);E.SubElement(c,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),**{'as':'geometry'});return c
 def text(x,y,v,sz=30,col=INK,w=400):
  return cell(f'text;html=1;align=center;verticalAlign=middle;whiteSpace=wrap;overflow=fill;fontFamily={"Songti SC" if lang=="CN" else "Times New Roman"};fontSize={sz};fontColor={col};strokeColor=none;fillColor=none;',x-w/2,y-25,w,50,v)
 def box(x,y,w,h,fc,ec='none'):
  return cell(f'rounded=1;absoluteArcSize=1;arcSize=18;fillColor={fc};strokeColor={ec};',x,y,w,h)
 def arrow(pts,col=TEAL,dash=False):
  global counter
  counter+=1;n='lc_'+str(counter);c=E.SubElement(root,'mxCell',id=n,parent='1',edge='1',style=f'html=1;endArrow=block;endFill=1;endSize=8;strokeWidth=2.2;strokeColor={col};rounded=1;dashed={int(dash)};');E.SubElement(c,'mxGeometry',relative='1',**{'as':'geometry'});route(n,pts)
 def circle(x,y,col,r=10):return cell(f'ellipse;fillColor={col};strokeColor=#FFFFFF;strokeWidth=1;',x-r,y-r,2*r,2*r)
 def bar(x,y,p,w=130,h=25):
  box(x,y,w*p,h,TEAL);box(x+w*p,y,w*(1-p),h,PURPLE)
 # New calibration module, using original class colors, circles, bars, and open layout.
 box(1490,673,640,283,'#F0FAF7','#CED7DE')
 text(1810,708,tr('局部一致性 → 节点温度','Local agreement → node temperature'),32,w=620)
 text(1600,756,tr('初始类别','Initial classes'),26,w=210)
 for x,y,col in [(1540,800,TEAL),(1645,793,PURPLE),(1590,853,TEAL)]:circle(x,y,col,12)
 arrow([(1548,810),(1580,844)],BLUE);arrow([(1636,804),(1600,844)],BLUE)
 text(1590,901,tr('按 W 加权','Weighted by W'),25,w=190)
 arrow([(1664,837),(1700,837)],TEAL)
 text(1850,769,r'\(a_i=\sum_j W_{ij}\mathbf{1}[c_j=c_i]\)',26,w=430)
 text(1850,817,r'\(T_i=1+\lambda(1-a_i)\)',28,w=390)
 text(1865,864,r'\(F_{0,i}=\operatorname{softmax}(s_i/T_i)\)',27,w=460)
 bar(1735,924,.9,115,21)
 arrow([(1862,934),(1900,934)],ORANGE)
 bar(1912,924,.65,115,21)
 text(1865,899,tr('预测概率软化','Prediction softening'),25,ORANGE,w=420)
 arrow([(1827,646),(1827,670)],TEAL)
 arrow([(2059,388),(2115,388),(2115,660),(1660,660),(1660,679)],BLUE)
 # Tie independent teacher-attribute input explicitly to M1's source, without crossing encoder flow.
 text(1570,385,tr('与 M1 同源的属性','Teacher features from M1'),23,GRAY,w=290)
 # Match sharpening and softening with the same before/after visual grammar.
 replace('c85',tr('锐化','Sharpening'))
 get('c85').set('style',re.sub(r'fontColor=[^;]+','fontColor='+ORANGE,get('c85').get('style')))
 text(1080,332,tr('类别分配锐化','Assignment sharpening'),25,ORANGE,w=300)
 bar(953,372,.65,104,23)
 arrow([(1070,383),(1103,383)],ORANGE)
 bar(1116,372,.9,104,23)
 # Fusion / iteration denote operations without fixing experimental settings.
 replace('c476',tr('融合','Fuse'),28,140)
 root.remove(get('c477'))
 # Reflow the right readout region to align with the left model / learning panels.
 # Preserve the horizontal component layout and readable type while reducing vertical gaps.
 for c in root:
  g=c.find('mxGeometry')
  if g is None:continue
  if c.get('vertex')=='1':
   x=float(g.get('x','0'));w=float(g.get('width','0'))
   if x+w/2>=1470:
    g.set('y',str(24+(float(g.get('y','0'))-24)*.8))
    if g.get('height'):
     height=float(g.get('height'));g.set('height',str(height*.8))
     if abs(w-height)<.01 and 'text;' not in c.get('style',''):
      g.set('width',str(w*.8));g.set('x',str(x+w*.1))
    c.set('style',re.sub(r'fontSize=([0-9.]+)',lambda m:'fontSize='+str(float(m[1])*.9),c.get('style','')))
  elif c.get('edge')=='1':
   # Readout-internal edges, plus the classification input endpoint.
   for point in g.iter('mxPoint'):
    if float(point.get('x','0'))>=1445:
     point.set('y',str(24+(float(point.get('y','0'))-24)*.8))
 route('c318',[(1319,651),(1445,651),(1445,24+(532-24)*.8),(1497,24+(532-24)*.8)])
 doc.write(OUT/f'Framework_{lang}.drawio',encoding='utf-8',xml_declaration=True)
 print(lang,'objects',len(root))
