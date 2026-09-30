#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
p=Path(__file__).resolve().parent;m=json.loads((p/'manifest.json').read_text())
for f in m['files']:
 x=p/f['destination'];assert x.is_file(),x
 assert x.stat().st_size==f['bytes'],x
 assert hashlib.sha256(x.read_bytes()).hexdigest()==f['sha256'],x
print(f"verified {len(m['files'])} source snapshots")
