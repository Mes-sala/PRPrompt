import torch
import torch.nn.functional as F
from .node_scores import classification_scores


@torch.no_grad()
def GpromptEva(loader, gnn, prompt, center_embedding, num_class, device):
    gnn.eval()
    prompt.eval()
    logits, labels = [], []
    for batch in loader:
        batch = batch.to(device)
        out = gnn(batch.x, batch.edge_index, batch.batch, prompt, "Gprompt")
        logits.append(F.cosine_similarity(out.unsqueeze(1), center_embedding.unsqueeze(0), dim=-1))
        labels.append(batch.y)
    return classification_scores(torch.cat(logits), torch.cat(labels))
