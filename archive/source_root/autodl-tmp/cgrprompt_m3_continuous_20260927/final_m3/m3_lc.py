"""CGRPrompt final M3: frozen local-consistency calibrated readout."""
import torch
import torch.nn.functional as F

@torch.no_grad()
def m3_readout(score, teacher_graph, support_index, support_labels, num_classes):
    """Return log probabilities; teacher_graph is row-normalized sparse COO.

    Build teacher_graph from centered two-hop attributes using cosine 50-NN,
    excluding self, with neighbor softmax temperature 0.2. Scores come from
    the original trained Prompt, not from the direct teacher classifier.
    """
    labels = score.argmax(1)
    ix = teacher_graph.indices()
    agree = torch.zeros(len(score), device=score.device).scatter_add_(
        0, ix[0], teacher_graph.values() * (labels[ix[0]] == labels[ix[1]]).float())
    temperature = 1 + 3 * (1 - agree)
    p0 = (score / temperature[:, None]).softmax(-1)
    p = p0.clone()
    target = F.one_hot(support_labels, num_classes).float()
    p[support_index] = target
    alpha = 0.9
    for _ in range(20):
        p = (1 - alpha) * p0 + alpha * torch.sparse.mm(teacher_graph, p)
        p[support_index] = target
    return p.clamp_min(1e-12).log()
