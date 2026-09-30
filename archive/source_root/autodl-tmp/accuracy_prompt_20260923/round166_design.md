# R166 LoRA-RITE来源与优化器核查

来源：[LoRA Done RITE，ICLR2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/bcbc0f660d2dde42f9d1d0ecb14a6f9a-Abstract-Conference.html)，完整arxiv v2 HTML保存rite166_paper.html（HF markdown404已记录，未以摘要替代）；[作者PyTorch重实现](https://github.com/gkevinyen5418/LoRA-RITE)，提交d4186b6fedb39300d23c00ce0334db09719da9fc，rite166_official.py原样保留。README明确原论文JAX实验与PyTorch复现实验设置不同，不混称全论文复现。

源码机制：对配对A.T/B作QR，用对方R的伪逆消除因子尺度；用新旧Q的重叠矩阵变换矩阵二阶矩及一阶动量；可加入escape mass；在小矩阵特征分解逆平方根预条件后再积累一阶动量，最终变回因子坐标。并非R158全权重Adam再映射，也非对原Adam梯度简单缩放。

已做rank1/4、escape开/关各五步独立NumPy回放，包括QR/pinv、偏差修正衰减、矩阵矩变换、escape、eigh和因子更新，最大参数差1.3323e−15；零B起点首步仅B更新、随后两因子更新。A乘2/B除2五步物理乘积相同（仅该测试，不概括所有奇异初始化和dtype）。证据round166_replay_probe.json、round166_source_probe.json。

源码边界：relative_epsilon=True引用未定义p_new，会NameError；默认False可运行，后续不启用。不擅自修复原源码。inverse_sqrt还会对数值负特征值产生NaN，再由nan_to_num掩掉，现有小测试未触发；真实图实现应检测并记录，不能把最终有限输出当作中间全部有限。默认update_skipping=1与clip=1，拟显式设0禁用以匹配旧未裁剪协议；escape开/关分开，wd0，eps1e−6，balanceFalse。作者源码保持原样，任何保护适配用新封装和独立审计，不覆盖原件。

拟rank1/4配合原中间Prompt，factor lr.0025及.00025、escape开/关、同初始化同rank Adam控制及旧l1/l4。Prompt维持Adam.01，teacher/scale2/KD/k50保持；没有终端预测融合。RITE双因子用CPUdouble状态更新再回float32，梯度保存前后明确，不与Prompt混入成对列表。此dtype路线仍待实现及验证。尚未实际GCN或query实验。


## 正式预注册

20条两checkpoint真实GCN三步轨迹已全部通过。封装不改作者优化器代码；CPUdouble proxy更新后每步量化回float32再继续double状态，独立NumPy QR/矩变换/escape/特征分解与量化回放覆盖。inverse_sqrt前检查全部移位特征值非负且矩阵有限，后检查逆平方根有限，所有检查值保存；失败即报错，绝不将nan_to_num掩盖作为成功。

e1/e4=rank1/4 escapeTrue lr.0025；n1/n4=escapeFalse lr.0025；k1/k4=escapeTrue lr.00025；a1/a4=同初始化同rank Adam.0025；l1/l4=旧LoRA。初始A kaiming,B零，CPUgen seed+166000，Prompt Adam.01，其他loss/readout不变。候选顺序如上，每个GraphMAE优先，整数正确数排序解平局。新适配参数256/1024加Prompt896，RITE状态开销不等于可训练参数。

Cora5shot236100..104，100步，dev前500，固定k50，先61条12步smoke后285正式。完整110轨迹220快照140邻域审计后分析，最高者对两ck旧CE/scale2最强根及Direct均至少+.20才确认。旧l4不第三次追确认，新配置锁定20新支持，原型保留不切换。
