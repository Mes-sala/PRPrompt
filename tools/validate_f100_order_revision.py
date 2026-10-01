"""Fit only the failed support cell for engineering validation; no query scoring."""
import argparse
import json
from pathlib import Path
import torch
from prprompt.model import TwoLayerGCN
from prprompt.protocol import sample_support, sha_file
from experiments.boundary_20261002.model import fit, saved_readout
from experiments.boundary_20261002.order_guard import save_order_diagnostic, verify_order_diagnostic
from tools.audit_boundary_delivery import check_order_guard


def validate(args):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    plan = json.loads(args.parent_plan.read_text())
    spec = plan['datasets']['Chameleon']
    job = next(j for j in plan['jobs'] if j['dataset'] == 'Chameleon')
    assert sha_file(spec['snapshot']) == spec['snapshot_sha256']
    assert sha_file(job['checkpoint']) == job['checkpoint_sha256']
    data = torch.load(spec['snapshot'], map_location='cpu', weights_only=True)
    x, edges = data['x'].to(args.device), data['edges'].to(args.device)
    support = sample_support(data['pool'], data['labels'][data['pool']], spec['classes'], 1, 30093009)
    labels = data['labels'][support]
    model = TwoLayerGCN(x.shape[1]).to(args.device).eval().requires_grad_(False)
    model.load_state_dict(torch.load(job['checkpoint'], map_location='cpu', weights_only=True))
    scores, z, state, audit = fit('F100OneShotAdapt', model, x, edges, support.to(args.device),
                                labels.to(args.device), spec['classes'], {'steps':100}, 0)
    assert torch.isfinite(scores).all() and torch.isfinite(z).all()
    state['method'] = 'F100OneShotAdapt'
    assert torch.equal(scores, saved_readout(state, z, support.to(args.device), labels.to(args.device), spec['classes'], args.device))
    assert audit['backbone_frozen'] and not audit['query_labels_received']
    assert len(state['prompt']['optimizer_gradients']) == 100
    assert not args.output.exists()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_order_diagnostic(args.output.parent, state, audit)
    assert verify_order_diagnostic(args.output.parent, state, audit) == audit['prefit_order_guard']
    assert check_order_guard(args.output.parent, {'audit':audit}, {'policy':audit['prefit_order_guard']['policy']}) == audit['prefit_order_guard']
    root = Path(__file__).resolve().parents[1]
    sources = {name:sha_file(root/name) for name in ['experiments/boundary_20261002/order_guard.py',
        'experiments/boundary_20261002/model.py', 'experiments/boundary_20261002/filters.py',
        'experiments/v1_f100/historical.py', 'experiments/v1_f100/reference/run_round83_graft.py',
        'experiments/v1_f100/reference/gate79.py', 'tools/validate_f100_order_revision.py']}
    report = dict(device=args.device, parent_plan_sha256=plan['plan_sha256'], split_seed=30093009,
        train_seed=0, shots=1, steps=100, checkpoint_sha256=job['checkpoint_sha256'],
        snapshot_sha256=spec['snapshot_sha256'], source_sha256=sources, original_failed_forward_not_bitwise_reconstructed=True,
        no_query_scoring=True, not_a_formal_unit=True, all_scores_and_gradients_finite=True,
        saved_readout_exact=True, torch_and_independent_numpy_order_replay=True, audit=audit)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent-plan', type=Path, required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--output', type=Path, required=True)
    validate(parser.parse_args())
