import torch
import torchmetrics

from prompt_graph.utils import get_logger

logger = get_logger(__name__)


@torch.no_grad()
def GNNNodeEva(data, idx_test, gnn, answering, num_class, device, prompt=None, prompt_type=None):
    from .node_scores import classification_scores
    gnn.eval()
    answering.eval()
    if prompt is not None:
        prompt.eval()
    kwargs = dict(prompt=prompt, prompt_type=prompt_type) if prompt is not None else {}
    out = answering(gnn(data.x, data.edge_index, batch=None, **kwargs))
    return classification_scores(out[idx_test], data.y[idx_test])


def GNNGraphEva(loader, gnn, answering, num_class, device, prompt=None, prompt_type=None):
    gnn.eval()
    if prompt is not None:
        prompt.eval()
    accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_class).to(
        device
    )
    macro_f1 = torchmetrics.classification.F1Score(
        task="multiclass", num_classes=num_class, average="macro"
    ).to(device)
    auroc = torchmetrics.classification.AUROC(task="multiclass", num_classes=num_class).to(device)
    auprc = torchmetrics.classification.AveragePrecision(
        task="multiclass", num_classes=num_class
    ).to(device)

    accuracy.reset()
    macro_f1.reset()
    auroc.reset()
    auprc.reset()
    if answering:
        answering.eval()
    with torch.no_grad():
        for batch_id, batch in enumerate(loader):
            batch = batch.to(device)
            kwargs = dict(prompt=prompt, prompt_type=prompt_type) if prompt is not None else {}
            out = gnn(batch.x, batch.edge_index, batch.batch, **kwargs)
            if answering:
                out = answering(out)
            pred = out.argmax(dim=1)
            acc = accuracy(pred, batch.y)
            ma_f1 = macro_f1(pred, batch.y)
            roc = auroc(out, batch.y)
            prc = auprc(out, batch.y)
            if len(loader) > 20:
                logger.info(
                    f"Batch {batch_id}/{len(loader)} Acc: {acc.item():.4f} | Macro-F1: {ma_f1.item():.4f}| AUROC: {roc.item():.4f}| AUPRC: {prc.item():.4f}"
                )

            # print(acc)
    acc = accuracy.compute()
    ma_f1 = macro_f1.compute()
    roc = auroc.compute()
    prc = auprc.compute()

    return acc.item(), ma_f1.item(), roc.item(), prc.item()
