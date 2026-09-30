from pathlib import Path
import xml.etree.ElementTree as E
import re
OUT=Path(__file__).resolve().parent
SRC=OUT.parent/'论文总Framework_鲜明大字双语版_20260926/Framework_CN.drawio'
t=E.parse(SRC); m=t.find('.//mxGraphModel'); root=m.find('root')
t.getroot().set('host','drawio'); t.find('diagram').set('name','PRPrompt 基础M3总框架')
m.set('pageHeight','1510'); m.set('dy','1510')
for c in list(root):
 ident=c.get('id','')
 if ident.startswith('c') and ident[1:].isdigit() and 501<=int(ident[1:])<=521:
  root.remove(c);continue
 g=c.find('mxGeometry')
 if g is not None:
  if c.get('vertex')=='1':g.set('y',str(float(g.get('y',0))+100))
  for pt in g.findall('.//mxPoint'):
   if pt.get('as')!='offset':pt.set('y',str(float(pt.get('y',0))+100))
idx={c.get('id'):c for c in root}
def label(i,value,x=None,y=None,w=None,h=None,fs=None):
 c=idx[i];c.set('value',value);g=c.find('mxGeometry')
 for key,v in [('x',x),('y',y),('width',w),('height',h)]:
  if v is not None:g.set(key,str(v))
 if fs:c.set('style',re.sub(r'fontSize=[^;]+',f'fontSize={fs}',c.get('style')))
label('c3','M1  原型路由构造',w=450,fs=42)
label('c4','迭代修正')
label('c86','固定路由 Q<br>中心化<br>原图传播',x=1016,y=301,w=182,h=102,fs=29)
label('c114','共享残差 P',x=1168,w=245)
label('c323','M2  残差矩阵学习：方向与步幅',w=760)
label('c375','M3  支持约束概率传播',x=1480,w=640)
label('c476','融合',x=1718,y=1150,w=95,fs=31)
root.remove(idx['c477'])
label('c137','冻结图编码器',x=535,w=350)
label('c386','适配表示构图')
# A single title identifies the whole prompt; module labels remain subordinate.
c=E.SubElement(root,'mxCell',{'id':'prprompt_title','value':'PRPrompt：面向少样本节点分类的原型路由图提示','style':'text;html=1;strokeColor=none;fillColor=none;fontFamily=Songti SC;fontSize=47;fontStyle=1;fontColor=#182C41;align=left;verticalAlign=middle;whiteSpace=nowrap;spacing=0;','vertex':'1','parent':'1'})
E.SubElement(c,'mxGeometry',{'x':'50','y':'15','width':'2090','height':'70','as':'geometry'})
t.write(OUT/'PRPrompt_总框架图.drawio',encoding='utf-8',xml_declaration=True)
print(OUT/'PRPrompt_总框架图.drawio')
