import torch
import torchmetrics

from prompt_graph.utils import get_logger

logger = get_logger(__name__)


@torch.no_grad()
def AllInOneEva(loader, prompt, gnn, answering, num_class, device):
    from .node_scores import classification_scores
    gnn.eval()
    prompt.eval()
    answering.eval()
    logits, labels = [], []
    for batch in loader:
        batch = batch.to(device)
        graph = prompt(batch)
        logits.append(answering(gnn(graph.x, graph.edge_index, graph.batch)))
        labels.append(batch.y)
    return classification_scores(torch.cat(logits), torch.cat(labels))


def AllInOneEvaWithoutAnswer(loader, prompt, gnn, num_class, device):
    prompt.eval()
    accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_class).to(
        device
    )
    macro_f1 = torchmetrics.classification.F1Score(
        task="multiclass", num_classes=num_class, average="macro"
    ).to(device)
    accuracy.reset()
    macro_f1.reset()
    for batch_id, test_batch in enumerate(loader):
        test_batch = test_batch.to(device)
        emb0 = gnn(test_batch.x, test_batch.edge_index, test_batch.batch)
        pg_batch = prompt.token_view()
        pg_batch = pg_batch.to(device)
        pg_emb = gnn(pg_batch.x, pg_batch.edge_index, pg_batch.batch)
        dot = torch.mm(emb0, torch.transpose(pg_emb, 0, 1))
        pre = torch.softmax(dot, dim=1)

        y = test_batch.y
        pre_cla = torch.argmax(pre, dim=1)

        acc = accuracy(pre_cla, y)
        macro_f1(pre_cla, y)

    acc = accuracy.compute()
    return acc
