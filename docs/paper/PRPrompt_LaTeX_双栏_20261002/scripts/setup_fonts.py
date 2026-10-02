#!/usr/bin/env python3
"""Fetch pinned open-source CJK fonts to an external cache, never the repository."""
from pathlib import Path
import hashlib
import subprocess

FONT_DIR=Path.home()/'.cache/prprompt-latex/fonts/noto-serif-cjk-2.003'
BASE='https://raw.githubusercontent.com/notofonts/noto-cjk/Serif2.003/Serif/OTF/SimplifiedChinese/'
HASHES={
 'NotoSerifCJKsc-Regular.otf':'2a2eae2628df83556c54018c41e20fa532c1b862c5256ae8b3f23feb918d12ca',
 'NotoSerifCJKsc-Bold.otf':'8af07d4b6c2e82bcc72a30e066eaf295f11b9424f4aad2eaa9fe0e9c3b38fc73',
}
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def ensure_fonts():
 FONT_DIR.mkdir(parents=True,exist_ok=True)
 for name,digest in HASHES.items():
  path=FONT_DIR/name
  if path.exists() and sha(path)==digest:continue
  temp=path.with_suffix('.download')
  subprocess.run(['curl','--fail','--location','--retry','2',BASE+name,'--output',str(temp)],check=True)
  if sha(temp)!=digest:raise ValueError('Font checksum mismatch: '+name)
  temp.replace(path)
 return FONT_DIR
if __name__=='__main__':print(ensure_fonts())
