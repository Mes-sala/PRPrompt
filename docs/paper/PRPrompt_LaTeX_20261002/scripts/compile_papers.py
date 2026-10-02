#!/usr/bin/env python3
"""Compile both manuscripts with the dedicated Conda environment.

Intermediate files stay outside the source package. A successful compiler exit
is not a substitute for reviewing the generated pages.
"""
import argparse
from pathlib import Path
import subprocess
import shutil
from setup_fonts import ensure_fonts
from check_cjk_pdf import check_cjk_pdf

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[2]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--language',choices=['en','zh','both'],default='both')
    parser.add_argument('--environment',default='prprompt-latex')
    parser.add_argument('--build-dir',type=Path,default=REPO/'experiments/local/latex_build')
    parser.add_argument('--output-dir',type=Path,default=REPO/'output/pdf/PRPrompt_20261002')
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for language in (['en','zh'] if args.language=='both' else [args.language]):
        build=(args.build_dir/language).resolve();build.mkdir(parents=True,exist_ok=True)
        cmd=['conda','run','--no-capture-output','-n',args.environment,'tectonic',f'main_{language}.tex','--keep-logs','--keep-intermediates','--outdir',str(build)]
        if language=='zh':cmd.extend(['-Z','search-path='+str(ensure_fonts())])
        subprocess.run(cmd,cwd=ROOT,check=True)
        if language=='zh':
            print(check_cjk_pdf(build/f'main_{language}.pdf',args.environment))
        dest=args.output_dir/f'PRPrompt_{language}.pdf'
        shutil.copy2(build/f'main_{language}.pdf',dest)
        if language=='zh':shutil.copy2(dest,args.output_dir/'PRPrompt_zh_CJK_fixed.pdf')
        print(dest.resolve())

if __name__=='__main__':main()
