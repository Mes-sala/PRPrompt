import hashlib
import random
import numpy as np
import torch

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def state_hash(model):
    h = hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def support_vectors(support, labels, classes, size, device):
    si = torch.as_tensor(support, dtype=torch.long, device=device)
    sy = torch.as_tensor(labels, dtype=torch.long, device=device)
    if classes < 1 or si.ndim != 1 or sy.shape != si.shape or not si.numel():
        raise ValueError('Aligned nonempty support IDs and labels required')
    if int(si.min()) < 0 or int(si.max()) >= size or len(si.unique()) != len(si):
        raise ValueError('Support IDs must be unique and in range')
    if int(sy.min()) < 0 or int(sy.max()) >= classes or (torch.bincount(sy, minlength=classes) == 0).any():
        raise ValueError('Support labels must cover classes 0..C-1')
    return si, sy
