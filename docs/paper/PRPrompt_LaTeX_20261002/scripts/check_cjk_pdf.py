#!/usr/bin/env python3
"""Reject CJK PDFs dependent on reader-supplied mappings or missing Chinese text."""
from pathlib import Path
import argparse,json,re,subprocess

def check_cjk_pdf(pdf,environment='prprompt-latex'):
 prefix=['conda','run','--no-capture-output','-n',environment]
 fonts=subprocess.run(prefix+['pdffonts',str(pdf)],check=True,capture_output=True,text=True)
 cid_rows=[line.split() for line in fonts.stdout.splitlines()[2:] if 'CID' in line.split()]
 if not cid_rows:raise ValueError('No embedded CID text fonts found')
 for row in cid_rows:
  if row[-5]!='yes' or row[-3]!='yes':raise ValueError('Missing embedded font or ToUnicode: '+row[0])
 extracted=subprocess.run(prefix+['pdftotext',str(pdf),'-'],check=True,capture_output=True,text=True)
 if fonts.stderr.strip() or extracted.stderr.strip():raise ValueError('PDF reader reported errors: '+(fonts.stderr+extracted.stderr).splitlines()[0])
 count=len(re.findall(r'[\u4e00-\u9fff]',extracted.stdout))
 if count<5000 or '原型路由' not in extracted.stdout:raise ValueError('Chinese manuscript text is missing or incomplete')
 return {'cid_fonts_with_embedded_unicode_maps':len(cid_rows),'poppler_han_characters':count,'reader_errors':False}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('pdf',type=Path);p.add_argument('--environment',default='prprompt-latex');a=p.parse_args()
 print(json.dumps(check_cjk_pdf(a.pdf,a.environment),ensure_ascii=False))
