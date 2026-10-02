import hashlib
import json
from pathlib import Path
from prprompt.protocol import sha_file, save_json

ROOT = Path(__file__).resolve().parents[2]
METHODS = ['Frozen', 'FullFineTune', 'GPF', 'GPFPlus', 'GraphPromptDiag', 'PRPrompt', 'SharedPrompt']
METRICS = ['accuracy', 'macro_f1', 'micro_f1', 'roc_auc_macro', 'average_precision_macro']

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def load_plan(workspace):
    plan = json.loads((Path(workspace)/'plan.json').read_text())
    assert digest({k:v for k,v in plan.items() if k != 'plan_sha256'}) == plan['plan_sha256']
    for name, h in plan['source'].items():
        assert sha_file(ROOT/name) == h, name
    return plan

def setup_torch():
    import torch
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def source_hashes():
    result = {}
    for name in ['prprompt', 'experiments/v1_f100', 'experiments/boundary_20261002',
                 'experiments/f100_boundary_20261002', 'experiments/ogbn_arxiv']:
        for p in sorted((ROOT/name).rglob('*')):
            if p.is_file() and p.suffix in ['.py', '.md']:
                result[str(p.relative_to(ROOT))] = sha_file(p)
    result['tools/metric_utils.py'] = sha_file(ROOT/'tools/metric_utils.py')
    return result
