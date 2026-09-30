import argparse,json
from pathlib import Path
import run_study as r
ap=argparse.ArgumentParser();ap.add_argument('round');ap.add_argument('tag');a=ap.parse_args();base=r.ROOT/'results'/a.round
proof=json.loads((r.ROOT/'provenance'/f'{a.tag.upper()}本地五指标独立复算.json').read_text());assert max(proof['metric_max_errors'].values())<1e-8
manifest=json.loads((r.ROOT/'provenance'/f'{a.tag}_remote_manifest.json').read_text());items=[x for x in manifest if x['path'].endswith('prediction.npz')];assert len(items)==proof['predictions_independently_rescored']
marker=base/'REMOTE_PREDICTIONS_ARCHIVED.json'
if marker.exists():print('Already archived',a.round);raise SystemExit(0)
for x in items:
 f=base/x['path'];assert f.resolve().is_relative_to(base.resolve()) and r.sha(f)==x['sha256']
for x in items:(base/x['path']).unlink()
r.save(marker,dict(predictions=len(items),bytes_freed=sum(x['bytes'] for x in items),local_authority='/Users/shq/Research/GP/持续推进/属性原型分类独立研究_20260927/results/'+a.round,verification=f'{a.tag.upper()}本地五指标独立复算.json'))
print(a.round,sum(x['bytes'] for x in items))
