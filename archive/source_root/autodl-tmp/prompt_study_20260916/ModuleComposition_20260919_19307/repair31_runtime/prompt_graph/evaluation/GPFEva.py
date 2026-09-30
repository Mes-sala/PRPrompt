import torch
from .node_scores import classification_scores


@torch.no_grad()
def GPFEva(loader, gnn, prompt, answering, num_class, device):
    gnn.eval()
    prompt.eval()
    if answering is not None:
        answering.eval()
    logits, labels = [], []
    for batch in loader:
        batch = batch.to(device)
        out = gnn(prompt.add(batch.x), batch.edge_index, batch.batch)
        logits.append(answering(out) if answering is not None else out)
        labels.append(batch.y)
    return classification_scores(torch.cat(logits), torch.cat(labels))
