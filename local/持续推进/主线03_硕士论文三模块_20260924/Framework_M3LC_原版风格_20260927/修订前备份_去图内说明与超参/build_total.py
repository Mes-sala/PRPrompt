from pathlib import Path
import xml.etree.ElementTree as E
import copy, re
OUT=Path(__file__).resolve().parent;BASE=OUT.parent
TEAL='#008E87';PURPLE='#9056C3';BLUE='#2B75B9';INK='#182C41';GRAY='#44586D';ORANGE='#D47A23'
for lang in ['CN','EN']:
 doc=E.parse(BASE/'论文总Framework_鲜明大字双语版_20260926'/f'Framework_{lang}.drawio');model=doc.find('.//mxGraphModel');root=model.find('root');model.set('pageHeight','1780')
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
 replace('c386',tr('中心化两跳属性','Centered two-hop features'),24,265)
 replace('c405',tr('教师属性近邻图','Attribute graph'),29,275)
 replace('c424',tr('初始预测概率','Initial probabilities'),31,270)
 replace('c425',tr('由 Prompt 分数 s 产生','From Prompt scores s'),26,360)
 replace('c466',tr('校准后的 \\(F_0\\)','Calibrated \\(F_0\\)'),26,250)
 replace('c465',tr('迭代 20 次','20 iterations'),29,230)
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
 text(1850,817,r'\(T_i=1+3(1-a_i)\)',28,w=390)
 text(1865,864,r'\(F_{0,i}=\operatorname{softmax}(s_i/T_i)\)',27,w=460)
 bar(1765,922,.68,165,21)
 text(1850,899,tr('低一致性 → 软化','Low agreement → soften'),23,w=360)
 arrow([(1827,646),(1827,670)],TEAL)
 arrow([(2059,388),(2115,388),(2115,660),(1660,660),(1660,679)],BLUE)
 # Tie independent teacher-attribute input explicitly to M1's source, without crossing encoder flow.
 text(1570,385,tr('与 M1 同源的属性','Teacher features from M1'),23,GRAY,w=290)
 # A method label uses the spare area below preserved M2; no extra mechanism added.
 text(720,1550,tr('CGRPrompt：类别引导残差提示','CGRPrompt: Class-Guided Residual Prompt'),38,w=1320)
 text(720,1610,tr('同一组类别残差：构造、学习与读出','One class-residual prompt: construction, learning, readout'),28,GRAY,w=1300)
 text(720,1660,tr('读出温度只计算一次；主干和教师属性保持固定','Temperatures are computed once; backbone and teacher features stay fixed'),25,GRAY,w=1320)
 doc.write(OUT/f'Framework_{lang}.drawio',encoding='utf-8',xml_declaration=True)
 print(lang,'objects',len(root))
