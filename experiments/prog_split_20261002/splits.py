"""Original ProG algorithm, including independent permutations and overlap."""
import torch


def split(labels, classes, shots, seed):
    labels = labels.cpu().view(-1)
    g = torch.Generator().manual_seed(seed)
    n = len(labels)
    size = int(.9 * n)
    if size < 1000:
        size = int(.7 * n)
    query = torch.randperm(n, generator=g)[:size]
    pool = torch.randperm(n, generator=g)[size:]
    pieces = [pool[labels[pool] == c][:shots] for c in range(classes)]
    support = torch.cat(pieces)
    support = support[torch.randperm(len(support), generator=g)]
    return support, pool, query
