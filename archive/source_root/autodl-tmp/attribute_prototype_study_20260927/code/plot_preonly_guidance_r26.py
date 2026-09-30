"""Plot frozen R26 guidance without selecting a better moment rule."""
from pathlib import Path
import csv
import json
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'reports/preonly_certificate_r26/branches.csv'
OUT = ROOT / 'reports/geometry_mass_coupling'
OUT.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader(SOURCE.open()))
rows = [r for r in rows if r['method'] == 'moment0']
inside = [r for r in rows if r['declared_domain'] == 'True']
colors = {'ignorable': '#198f80', 'must_keep': '#ca6528', 'abstain': '#a6afba'}
labels = {'ignorable': 'May omit', 'must_keep': 'Must retain', 'abstain': 'Abstain'}
counts = {}
for rep in [1, 8]:
    group = [r for r in inside if int(r['replication']) == rep]
    assert len(group) == 90
    counts[rep] = {action: sum(r['action'] == action for r in group) for action in colors}
case = [r for r in rows if r['dataset'] == 'WikiCS' and int(r['hops']) == 3]
assert len(case) == 20
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'svg.fonttype': 'none', 'pdf.fonttype': 42})
fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.2, 6.3), gridspec_kw={'width_ratios': [1, 1.3]})
fig.subplots_adjust(left=.125, right=.98, bottom=.25, top=.75, wspace=.35)
for y, rep in enumerate([1, 8]):
    left = 0
    for action in colors:
        n = counts[rep][action]
        ax.barh(y, n, left=left, color=colors[action], height=.42)
        if n:
            ax.text(left+n/2, y, str(n), ha='center', va='center', color='white', weight='bold', fontsize=12)
        left += n
ax.set_yticks([0, 1], ['Original count', 'Count weight ×8'])
ax.invert_yaxis()
ax.set_xlim(0, 90)
ax.set_xticks([0, 30, 60, 90])
ax.set_xlabel('Support/depth conditions (90 per count setting)')
ax.set_title('A   Three-way decisions within the declared domain', loc='left', fontsize=11, pad=17)
fig.legend(handles=[Patch(color=colors[a], label=labels[a]) for a in colors], loc='upper left', bbox_to_anchor=(.12, .86), ncol=3, frameon=False, fontsize=9)
ax.set_ylim(1.55, -.55)
for rep in [1, 8]:
    group = sorted((r for r in case if int(r['replication']) == rep), key=lambda r: int(r['seed']))
    for j, r in enumerate(group):
        lo, hi, actual = [100*float(r[k]) for k in ['lower', 'upper', 'actual_disagreement']]
        y = j + (-.12 if rep == 1 else .12)
        color = colors['ignorable' if rep == 1 else 'must_keep']
        bx.plot([lo, hi], [y, y], color=color, lw=3, solid_capstyle='round')
        bx.scatter([actual], [y], s=15, color=color, edgecolor='white', linewidth=.4, zorder=3)
bx.axvline(1, color='#343c47', ls='--', lw=1)
bx.text(1.05, 9.8, '1% budget', fontsize=9, color='#343c47')
bx.set_yticks(range(10), [str(i) for i in range(10)])
bx.set_ylim(-.65, 10.5)
bx.set_xlabel('Full vs. same-count common-pool predictions: disagreement (%)')
bx.set_ylabel('Support draw (seed 271900 + index)')
bx.set_title('B   WikiCS, 3 hops: same geometry and support', loc='left', fontsize=11, pad=17)
bx.grid(axis='x', alpha=.15)
fig.suptitle('Count weight changes whether class-specific location is dispensable', x=.125, ha='left', fontsize=15, weight='bold', y=.98)
fig.text(.125, .905, 'Pre-only bounds; 1% prediction-preservation budget. This is not an Accuracy comparison.', fontsize=10, color='#4e5966')
fig.text(.125, .055,
         'Scope: WikiCS, Roman-empire and Amazon-ratings; 3 depths × 10 new support draws per graph.\n'
         'All 330 conditions from the other 11 graphs abstain. Moment0 shown; moment2 gives the same group decisions.\n'
         'Panel B: segments are pre-only disagreement bounds; dots are later verified disagreements. Ordinary float64 audit.',
         fontsize=9, color='#4e5966', linespacing=1.5)
for ext in ['png', 'pdf', 'svg']:
    fig.savefig(OUT / f'r26_preonly_guidance.{ext}', dpi=190)
data = {'source': str(SOURCE.relative_to(ROOT)), 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'rule': 'moment0', 'in_domain_counts': counts, 'out_of_domain_units': 330,
        'selected_example': 'WikiCS h3, all ten prescribed support draws', 'example_rows': case,
        'meaning': 'Prediction disagreement, not Accuracy; original rule and thresholds unchanged.'}
(OUT / 'r26_figure_data.json').write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')
print(json.dumps({'counts': counts, 'files': ['r26_preonly_guidance.png', 'r26_preonly_guidance.pdf', 'r26_preonly_guidance.svg']}))
