import argparse
import run_study as r
ap=argparse.ArgumentParser();ap.add_argument('round');ap.add_argument('tag');a=ap.parse_args();base=r.ROOT/'results'/a.round
assert (base/'complete.json').exists()
r.save(r.ROOT/'provenance'/f'{a.tag}_remote_manifest.json',[dict(path=str(f.relative_to(base)),sha256=r.sha(f),bytes=f.stat().st_size) for f in sorted(base.rglob('*')) if f.is_file()])
