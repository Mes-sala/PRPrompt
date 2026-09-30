"""RSP runner reusing NodeTask splits, checkpoint loading, audit and scoring."""
import time

import numpy as np
import torch

from prompt_graph.data.load4data import audit_node_indices
from prompt_graph.evaluation.node_scores import classification_scores
from prompt_graph.prompt.RSP import RSPOptions, cached_bank, prototype_logits, select_prompt
from prompt_graph.utils import seed_everything

from .audit import FoldAudit


def run_rsp_node(task):
    if not task.protocol_options:
        raise ValueError('RSP requires the explicit protocol runner')
    if task.epochs != 0 or task.protocol_options['support_passes'] != 0:
        raise ValueError('RSP is discrete adaptation: support_passes must be 0')
    if getattr(task.data, 'edge_weight', None) is not None:
        raise ValueError('RSP MVP supports unweighted input graphs only')
    options = RSPOptions.parse(task.prompt_options)
    # No +fold offset: fixed train_seed means fixed scratch encoder across splits.
    seed_everything(task.protocol_options['train_seed'])
    task.initialize_gnn()
    task.prompt = task.answering = task.optimizer = task.pg_opi = None
    task.data = task.data.to(task.device)
    task.run_diagnostics = []
    metrics, losses = [], []
    bank = None
    for fold in range(1, task.task_num + 1):
        audit = FoldAudit(task, fold)
        folder = task.fold_dir(fold)
        support, support_y, query, query_y = [
            torch.load(folder / name, map_location='cpu', weights_only=True).long().view(-1)
            for name in ('train_idx.pt', 'train_labels.pt', 'test_idx.pt', 'test_labels.pt')]
        y = task.data.y.detach().cpu().view(-1)
        audit_node_indices(support, query, y, task.shot_num, task.output_dim)
        if not torch.equal(support_y, y[support]) or not torch.equal(query_y, y[query]):
            raise ValueError('Saved labels do not match dataset')
        if bank is None:
            bank, cache = cached_bank(task.gnn, task.data.x, task.data.edge_index, options)
        else:
            cache = dict(cache, cache_hit=True, cache_seconds=0., encoder_forward_count=0)
        selection_start = time.perf_counter()
        z, selection = select_prompt(bank, support, support_y, task.output_dim, options)
        logits = prototype_logits(z, support, support_y, query, task.output_dim, options.temperature)
        selection_seconds = time.perf_counter() - selection_start
        # Query labels enter only here, after all adaptation and prediction.
        scores = classification_scores(logits, query_y)
        record = audit.finish(0, scores, len(support), len(query))
        record.update(adaptation_mode='discrete_search', backward_count=0,
                      selection_label_source='support_only', extra_validation_labels=0,
                      stop_reason='discrete_selection_complete', **cache, **selection,
                      selection_seconds=selection_seconds, candidate_count=len(bank['candidates']),
                      candidate_stats=bank['candidate_stats'],
                      original_bank_build_seconds=bank['build_seconds'],
                      original_retrieval_seconds=bank['retrieval_seconds'],
                      retrieval_graphs=[{k: v for k, v in item.items() if k != 'degree_mass'}
                                        for item in bank['retrieval']],
                      encoder_seed=task.protocol_options['train_seed'])
        task.run_diagnostics.append(record)
        metrics.append(scores)
        losses.append(selection['mean_inner_losses'][selection['selected_index']])
        print(f'RSP fold={fold} selected={selection["selected_amplitudes"]} acc={scores[0]:.4f} cache_hit={cache["cache_hit"]}')
    values = np.asarray(metrics)
    result = [float(np.mean(losses))]
    for column in values.T:
        result.extend([float(np.mean(column)), float(np.std(column))])
    return tuple(result)
