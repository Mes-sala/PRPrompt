#!/usr/bin/env python3
"""Compile both manuscripts with the dedicated Conda environment.

Intermediate files stay outside the source package. A successful compiler exit
is not a substitute for reviewing the generated pages.
"""
import argparse
from pathlib import Path
import subprocess
import shutil

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[2]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--environment',default='prprompt-latex')
    parser.add_argument('--build-dir',type=Path,default=REPO/'experiments/local/latex_build')
    parser.add_argument('--output-dir',type=Path,default=REPO/'output/pdf/PRPrompt_20261002')
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for language in ['en','zh']:
        build=(args.build_dir/language).resolve();build.mkdir(parents=True,exist_ok=True)
        cmd=['conda','run','--no-capture-output','-n',args.environment,'tectonic',f'main_{language}.tex','--keep-logs','--keep-intermediates','--outdir',str(build)]
        subprocess.run(cmd,cwd=ROOT,check=True)
        dest=args.output_dir/f'PRPrompt_{language}.pdf'
        shutil.copy2(build/f'main_{language}.pdf',dest)
        print(dest.resolve())

if __name__=='__main__':main()
