"""Runtime evidence for fixed-budget NodeTask runs (no benchmark selection)."""
import hashlib
import time

import torch


def state_hash(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class FoldAudit:
    def __init__(self, task, fold):
        self.task, self.fold = task, fold
        self.start = time.perf_counter()
        self.encoder_before = state_hash(task.gnn)
        self.steps, self.handles = {}, []
        self.modules = {name: getattr(task, name, None) for name in
                        ('gnn', 'prompt', 'answering', 'DownPrompt', 'feature_prompt')}
        self.before = {name: {k: p.detach().cpu().clone() for k, p in m.named_parameters()}
                       for name, m in self.modules.items() if m is not None}
        for name in ('optimizer', 'pg_opi', 'answer_opi'):
            optimizer = getattr(task, name, None)
            if optimizer is not None:
                self.steps[name] = 0
                self.handles.append(optimizer.register_step_post_hook(self._hook(name)))
                encoder_ids = {id(p) for p in task.gnn.parameters()}
                if task.prompt_type != 'None' and any(
                    id(p) in encoder_ids for g in optimizer.param_groups for p in g['params']
                ):
                    raise RuntimeError('Frozen encoder appears in optimizer')
        self.optimizer_ids = {id(p) for name in self.steps
                              for group in getattr(task, name).param_groups for p in group['params']}
        self.gradient_seen = set()
        for module in self.modules.values():
            if module is not None:
                for parameter in module.parameters():
                    if parameter.requires_grad:
                        self.handles.append(parameter.register_hook(self._gradient_hook(id(parameter))))
        self.optimizer_groups = {
            name: [{'lr': g['lr'], 'weight_decay': g.get('weight_decay', 0),
                    'parameters': sum(p.numel() for p in g['params'])}
                   for g in getattr(task, name).param_groups] for name in self.steps
        }
        if task.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(task.device)

    def _gradient_hook(self, parameter_id):
        def seen(gradient):
            if not torch.isfinite(gradient).all():
                raise FloatingPointError('Nonfinite parameter gradient')
            if torch.count_nonzero(gradient):
                self.gradient_seen.add(parameter_id)
            return gradient
        return seen

    def _hook(self, name):
        def count(optimizer, args, kwargs):
            self.steps[name] += 1
        return count

    def finish(self, completed_epochs, metrics, support_count, query_count):
        for handle in self.handles:
            handle.remove()
        after = state_hash(self.task.gnn)
        if self.task.prompt_type != 'None' and after != self.encoder_before:
            raise RuntimeError('Frozen encoder state changed')
        parameters = {}
        for name, module in self.modules.items():
            if module is None:
                continue
            parameters[name] = {
                'total': sum(p.numel() for p in module.parameters()),
                'optimizer_managed': sum(p.numel() for p in module.parameters() if id(p) in self.optimizer_ids),
                'received_nonzero_gradient': sum(p.numel() for p in module.parameters() if id(p) in self.gradient_seen),
                'requires_grad': sum(p.numel() for p in module.parameters() if p.requires_grad),
                'changed_elements': sum(int((p.detach().cpu() != self.before[name][k]).sum())
                                        for k, p in module.named_parameters()),
            }
        passes = completed_epochs
        if self.task.prompt_type == 'All-in-one':
            passes *= self.task.answer_epoch + self.task.prompt_epoch
        if self.task.prompt_type in {'FrozenGraphPrototype', 'FrozenNodePrototype', 'FrozenIdentityPrototype'}:
            passes = 0
        return dict(fold=self.fold, completed_epochs=completed_epochs,
                    requested_support_passes=self.task.protocol_options['support_passes'],
                    support_passes=passes, optimizer_steps=self.steps,
                    optimizer_groups=self.optimizer_groups, parameters=parameters,
                    encoder_hash_before=self.encoder_before, encoder_hash_after=after,
                    stop_reason='fixed_last' if passes else 'no_training',
                    manual_structure_updates=completed_epochs if self.task.prompt_type == 'GPPT' else 0,
                    support_count=support_count, query_count=query_count,
                    metrics=list(metrics), seconds=time.perf_counter()-self.start,
                    peak_memory_bytes=torch.cuda.max_memory_allocated(self.task.device)
                    if self.task.device.type == 'cuda' else None)
