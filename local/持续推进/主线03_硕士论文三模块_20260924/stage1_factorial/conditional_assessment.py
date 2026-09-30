from pathlib import Path
import json,collections
R=Path(__file__).resolve().parent;j=json.loads((R/'factorial_pairs.json').read_text());groups=collections.defaultdict(list)
text='# 条件性三模块证据：探索阶段\n\n用户已明确不要求普适有效。以下完整展示各具体dataset×checkpoint的三个完整分支移除差；候选是探索发现，不是确认结论。不以全数据集宏均作为否决条件。单位pp。\n\n| 数据集 | 底座 | SHA | M1 | M2 | M3 | 三者均正 | 相对未中心化教师 |\n|---|---|---|---:|---:|---:|---|---:|\n'
for c in j['cells']:
 v=c['means'];ok=all(v['remove_M'+str(i)]>1e-5 for i in [1,2,3]);text+=f"| {c['dataset']} | {c['checkpoint']} | {c['sha256'][:12]} | {v['remove_M1']:+.3f} | {v['remove_M2']:+.3f} | {v['remove_M3']:+.3f} | {ok} | {v['full_minus_rect_uncentered']:+.3f} |\n"
 if ok:groups[c['checkpoint']].append(dict(dataset=c['dataset'],sha256=c['sha256'],effects={k:v[k] for k in ['remove_M1','remove_M2','remove_M3','full_vs_base','full_minus_rect_uncentered','full_minus_A2_center']}))
text+='\n## 按预训练家族归类\n\n同一家族不同SHA仍是不同权重；CiteSeer多个版本不能当多个数据集。\n'
for family,cells in sorted(groups.items()):text+=f"\n- {family}：{len(set(c['dataset'] for c in cells))}个数据集，{len(cells)}份权重；"+'、'.join(sorted(set(c['dataset'] for c in cells)))+'。\n'
text+='\n下一步应冻结候选及底座版本后，使用预定新支持seed确认全部移除对照；探索中挑中的范围不冒充独立确认。未形成三模块正证据的格保留为适用边界。\n'
(R/'条件性三模块评估.md').write_text(text);(R/'conditional_candidates.json').write_text(json.dumps(dict(status='exploratory_only',groups=dict(groups)),ensure_ascii=False,indent=2));print({k:len(set(c['dataset'] for c in v)) for k,v in groups.items()})
