from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import html, math
OUT=Path(__file__).resolve().parent
W,H=1800,1550
im=Image.new('RGB',(W,H),'#f5f7fb');d=ImageDraw.Draw(im)
font='/System/Library/Fonts/STHeiti Light.ttc'
svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><rect width="100%" height="100%" fill="#f5f7fb"/>']
def rect(x,y,w,h,fill,stroke='#d6deea',r=18):
 d.rounded_rectangle((x,y,x+w,y+h),radius=r,fill=fill,outline=stroke,width=2)
 svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
def txt(x,y,s,size=25,color='#26354b',center=False):
 f=ImageFont.truetype(font,size);box=d.textbbox((0,0),s,font=f);ww=box[2]-box[0]
 xx=x-ww/2 if center else x
 d.text((xx,y),s,font=f,fill=color)
 svg.append(f'<text x="{x}" y="{y+size}" font-family="STHeiti, PingFang SC, sans-serif" font-size="{size}" fill="{color}" text-anchor="{"middle" if center else "start"}">{html.escape(s)}</text>')
def box(x,y,w,h,title,lines,fill='#ffffff',stroke='#d6deea'):
 rect(x,y,w,h,fill,stroke)
 txt(x+w/2,y+18,title,28,center=True)
 for j,line in enumerate(lines):txt(x+w/2,y+59+34*j,line,23,center=True)
def arrow(points,color='#556c8b',dash=False):
 for a,b in zip(points,points[1:]):
  if dash:
   dx,dy=b[0]-a[0],b[1]-a[1];length=math.hypot(dx,dy)
   for t in range(0,int(length),18):
    u=min(t+10,length);d.line((a[0]+dx*t/length,a[1]+dy*t/length,a[0]+dx*u/length,a[1]+dy*u/length),fill=color,width=4)
  else:d.line([a,b],fill=color,width=4)
 a,b=points[-2:];angle=math.atan2(b[1]-a[1],b[0]-a[0]);p=[b,(b[0]-14*math.cos(angle-.45),b[1]-14*math.sin(angle-.45)),(b[0]-14*math.cos(angle+.45),b[1]-14*math.sin(angle+.45))]
 d.polygon(p,fill=color)
 pts=' '.join(f'{x},{y}' for x,y in points)
 svg.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="4"'+(' stroke-dasharray="10 8"' if dash else '')+'/>')
 svg.append('<polygon points="'+' '.join(f'{x},{y}' for x,y in p)+f'" fill="{color}"/>')
txt(60,35,'三模块图提示适配：同一冻结主干上的生成、优化与读出',39)
txt(60,93,'实线：前向数据流    橙色虚线：训练梯度与参数更新    只有类别 token P 可学习',25,'#5b6d84')
box(60,150,1680,105,'输入：节点属性 X、原图边 E、支持节点 S 及其标签', ['全图属性和结构可见；查询节点的真实标签不进入教师、训练或概率传播'],fill='#eaf0fa')
for x,w,title,color in [(60,460,'M1｜语义条件与门控锐化','#e8f4f5'),(610,580,'共享的冻结两层 GCN','#eaf0fa'),(1280,460,'M2｜方向与尺度匹配','#fff0e2')]:
 rect(x,305,w,750,color);txt(x+w/2,323,title,29,center=True)
box(90,395,400,135,'属性准备',['原图两跳传播 → 全图中心化','节点向量归一化'])
box(90,575,400,150,'支持原型 → 修正教师',['支持原型 + 无标签软原型','混合比例 0.5；修正 3 次'])
box(90,775,400,170,'门控 Q（训练期间固定）',['类别概率温度 0.2 → 0.1','按类别中心化 → 原图传播','输出：N × C 的门控系数'],fill='#ffffff',stroke='#52a7ae')
box(650,395,500,120,'冻结 GCN 第一层',['输出预激活 U；计算各维标准差'])
box(650,590,500,165,'提示注入与激活',['ΔU = 10 · [Q tanh(P)] · std(U)','P：C × 128，零初始化','H = LeakyReLU(U + ΔU)'],stroke='#7d91d7')
box(650,805,500,105,'冻结 GCN 第二层',['利用原图 E 得到适配表示 Z'])
box(650,960,500,75,'最终更新后重新前向',[],fill='#ffffff')
box(1310,395,400,165,'支持集训练损失',['留一原型交叉熵','+ 0.001 × token 均方值','只使用支持标签'])
box(1310,595,400,120,'token 梯度 G',['左右矩阵统计 → Shampoo方向'])
box(1310,785,400,165,'尺度匹配并更新 P',['同轨迹 Adam 候选方向的范数','匹配矩阵方向范数；学习率 0.01','重复 100 次，只更新 P'],stroke='#d79654')
arrow([(290,255),(290,300)]);arrow([(900,255),(900,300)])
arrow([(290,370),(290,395)]);arrow([(900,370),(900,395)])
arrow([(290,530),(290,575)]);arrow([(290,725),(290,775)])
arrow([(490,860),(555,860),(555,670),(650,670)],'#32939d')
arrow([(900,515),(900,590)]);arrow([(900,755),(900,805)]);arrow([(900,910),(900,960)])
arrow([(1150,858),(1235,858),(1235,470),(1310,470)])
arrow([(1510,560),(1510,595)],'#c47c37',True);arrow([(1510,715),(1510,785)],'#c47c37',True)
arrow([(1310,865),(1260,865),(1260,1080),(585,1080),(585,720),(650,720)],'#c47c37',True)
rect(60,1150,1680,255,'#eaf5ed');txt(85,1166,'训练结束后的读出｜M3 使用适配表示，不向 token 或主干反传',29)
for x,w,title,lines in [(90,330,'共同中心化',['Z 减去全图均值','按节点归一化']),(500,340,'两项并行准备',['支持原型 → 初始概率 F₀','余弦相似度 → 50 近邻 W']),(920,380,'M3：支持约束传播',['0.1 F₀ + 0.9 W F，20 次','初始化及每步固定支持标签']),(1380,330,'输出预测',['log 概率 / argmax','封存后才用查询标签评分'])]:box(x,1220,w,155,title,lines,stroke='#80b28d')
arrow([(900,1035),(900,1115),(255,1115),(255,1145)])
arrow([(255,1205),(255,1220)])
arrow([(420,1300),(500,1300)]);arrow([(840,1300),(920,1300)]);arrow([(1300,1300),(1380,1300)])
txt(60,1440,'两种关系图：原图用于 GCN 与门控传播；适配表示的近邻图仅用于最终概率读出。',27)
txt(60,1490,'M1 调整 token 的使用方式，M2 学习这些 token，M3 利用所得表示；不是三个独立模型的投票。',25,'#5b6d84')
im.save(OUT/'三模块方案架构图.png');svg.append('</svg>');(OUT/'三模块方案架构图.svg').write_text('\n'.join(svg))
print(OUT/'三模块方案架构图.png')
