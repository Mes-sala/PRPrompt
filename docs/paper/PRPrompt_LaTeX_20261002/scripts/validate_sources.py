#!/usr/bin/env python3
"""Static manuscript checks only. Does not invoke TeX, BibTeX, or a renderer."""
from pathlib import Path
import collections, hashlib, json, re, ast
from build_tables import render, ROOT, D, SHOTS, METHODS, ABLATION

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text())
def strip_comments(text): return re.sub(r'(?<!\\)%[^\n]*','',text)
def flat(path, stack=()):
 assert path not in stack, f'Input cycle: {path}'
 text=path.read_text()
 def sub(m):
  target=ROOT/(m[1]+('.tex' if not m[1].endswith('.tex') else ''))
  assert target.exists(),f'Missing input: {target}'
  return flat(target,stack+(path,))
 return re.sub(r'\\input\{([^}]+)\}',sub,text)
def braces(text):
 depth=0
 for token in re.finditer(r'(?<!\\)[{}]',strip_comments(text)):
  depth+=1 if token[0]=='{' else -1
  assert depth>=0,'Unbalanced closing brace'
 assert depth==0,'Unbalanced opening brace'

# Source equivalence and table values are checked independently of TeX expansion.
for name,record in read(ROOT/'data/source_manifest.json').items():assert sha(ROOT/'data'/name)==record['sha256']
main=read(ROOT/'data/main.json');abl=read(ROOT/'data/ablation.json');paired=read(ROOT/'data/paired.json')
assert len(main)==54 and len(abl)==27 and len(paired)==270
idx={(r['dataset'],r['shots'],r['method']):r for r in main+abl}
assert len(idx)==63
for rows,methods in [(main,METHODS),(abl,ABLATION)]:
 assert {(r['dataset'],r['shots'],r['method']) for r in rows}=={(d,s,m) for d in D for s in SHOTS for m in methods}
 assert all(r['support_draws']==20 and r['training_repeats']==3 for r in rows)
metrics=['accuracy','macro_f1','micro_f1','roc_auc_macro','average_precision_macro']
assert {(r['dataset'],r['shots'],r['reference'],r['metric']) for r in paired}=={(d,s,m,k) for d in D for s in SHOTS for m in METHODS[:-1]+['NoRouting'] for k in metrics}
for r in paired:
 d,s,m,k=r['dataset'],r['shots'],r['reference'],r['metric']
 assert abs(idx[d,s,'PRPrompt'][k+'_mean']-idx[d,s,m][k+'_mean']-r['mean_delta'])<1e-12
 assert r['interval_source']=='original_paired_interval'
 if k in metrics[:2]: assert r['ci95_low']>0
expected,cells=render()
assert len(cells)==432
for name,content in expected.items(): assert (ROOT/'tables'/name).read_text()==content,name
assert read(ROOT/'table_cell_manifest.json')==cells

# Check the narrative's highlighted numbers against exactly identified conditions.
for d,numbers in {'Cora':['73.02','33.56','67.50'],'CiteSeer':['66.17','27.38','47.89'],'PubMed':['70.98','68.37','63.96']}.items():
 for m,value in zip(['PRPrompt','FrozenPrototype','FullFineTune'],numbers):
  assert f"{100*idx[d,5,m]['accuracy_mean']:.2f}"==value
  for lang in ['en','zh']: assert value+r'\%' in (ROOT/f'sections/paper_{lang}.tex').read_text()
for d,mean,lo,hi in [('Cora','36.94','34.87','39.06'),('CiteSeer','40.45','39.06','41.84'),('PubMed','3.02','1.39','4.51')]:
 r=next(r for r in paired if (r['dataset'],r['shots'],r['reference'],r['metric'])==(d,5,'NoRouting','accuracy'))
 assert [f"{100*r[k]:.2f}" for k in ['mean_delta','ci95_low','ci95_high']]==[mean,lo,hi]
 for lang in ['en','zh']:
  text=(ROOT/f'sections/paper_{lang}.tex').read_text();assert mean in text and f'[{lo},{hi}]' in text
for d,c in zip(D,[7,6,3]): assert read(ROOT/'data/datasets.json')[d]['classes']==c
for r in read(ROOT/'data/numerical_audit.json')['methods']:
 assert r['units']==540 and r['saved_readout_exact'] and r['gate_forward_failed_units']==r['fresh_query_class_changes']==0

bib=(ROOT/'references.bib').read_text();bibkeys=re.findall(r'@\w+\{([^,]+),',bib)
normalized=[]
for path in sorted((ROOT/'bibliography').glob('*.bib')):
 raw=path.read_text().strip()
 raw=re.sub(r'(@\w+\{)[^,]+,',lambda m:m[1]+path.stem+',',raw,count=1)
 raw=raw.replace('–','--').replace('’',"'")
 if path.stem=='gpf2023':raw=raw.replace('YANG, YANG','Yang, Yang')
 raw=re.sub(r'(title\s*=\s*)\{([^{}]*)\}',lambda m:m[1]+'{{'+m[2]+'}}',raw,flags=re.I)
 normalized.append('% Source: bibliography/'+path.name+'\n'+raw)
assert bib=='\n\n'.join(normalized)+'\n'
braces(bib)
assert len(bibkeys)==len(set(bibkeys))==8
v=read(ROOT/'bibliography/verification.json')
assert not v['unverified_citations']
assert {r['key'] for r in v['entries']}==set(bibkeys)
for r in v['entries']:
 assert len(r['verification_sources'])>=2 and r['status']=='verified' and not r['placeholder']
 assert sha(ROOT/'bibliography'/r['bibliography_source']['file'])==r['bibliography_source']['sha256']

reports={}
for lang in ['en','zh']:
 text=flat(ROOT/f'main_{lang}.tex');braces(text)
 env=[]
 for token in re.finditer(r'\\(begin|end)\{([^}]+)\}',strip_comments(text)):
  if token[1]=='begin':env.append(token[2])
  else: assert env and env.pop()==token[2],token[0]
 assert not env
 labels=re.findall(r'\\label\{([^}]+)\}',text);assert len(labels)==len(set(labels))
 refs=re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',text);assert set(refs)<=set(labels)
 cites={key.strip() for group in re.findall(r'\\cite\w*\{([^}]+)\}',text) for key in group.split(',')}
 assert cites==set(bibkeys)
 assert len(re.findall(r'\\section(?:\*)?\{',text))==11
 assert r'\begin{abstract}' in text and r'\label{fig:architecture}' in text
 for ban in ['FrozenLinear','RoutedLinear','AttributeRidge','AttributePrototype','F100','F111','HardMatched','M3_original','Chameleon','Computers','Photo','TODO','CITATION NEEDED']:
  assert ban not in text,(lang,ban)
 for stale in ['63.46','61.80','25.06']:
  assert stale not in (ROOT/f'sections/paper_{lang}.tex').read_text()
 reports[lang]=dict(labels=len(labels),references=len(refs),citations=sorted(cites),environments_balanced=True,braces_balanced=True)
# Formula bodies and label order must match between the translations.
def math_body(p):return [re.sub(r'\s+','',m[1]) for m in re.finditer(r'\\begin\{(?:equation|align)\}([\s\S]*?)\\end\{(?:equation|align)\}',p.read_text())]
for part in ['paper','appendix']:
 assert math_body(ROOT/f'sections/{part}_en.tex')==math_body(ROOT/f'sections/{part}_zh.tex')
 assert re.findall(r'\\label\{([^}]+)\}',(ROOT/f'sections/{part}_en.tex').read_text())==re.findall(r'\\label\{([^}]+)\}',(ROOT/f'sections/{part}_zh.tex').read_text())
for p in ROOT.rglob('*.tex'):braces(p.read_text())
for p in (ROOT/'scripts').glob('*.py'):ast.parse(p.read_text())
for p in ROOT.glob('*.md'):
 for link in re.findall(r'\[[^\]]+\]\(([^)]+)\)',p.read_text()):
  if '://' not in link and link!='static_validation.json': assert (p.parent/link.split('#')[0]).exists(),link
for ext in ['*.pdf','*.aux','*.bbl','*.blg','*.log','*.pt','*.pth','*.ckpt','*.safetensors']:
 assert not list(ROOT.rglob(ext)),f'Unexpected compiled or model artifact: {ext}'
report=dict(status='passed',scope='static source, bibliography and data validation; no compilation or rendering',languages=reports,shared_generated_table_files=len(expected),numeric_cells_verified=432,primary_and_additional_metric_cells=324,paired_cells=108,bibliographic_records_verified=8,formulas_identical_between_languages=True,source_json_hashes_verified=True,within_condition_seed_statistics_only=True,no_classifier_comparisons=True,main_architecture='intentional placeholder',compilation_performed=False,visual_layout_verified=False,submission_ready=False)
(ROOT/'static_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False))
