"""Independent NumPy metrics and file hashing for sealed experiment results."""
import hashlib
from pathlib import Path
import numpy as np

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def numpy_metrics(y, p):
    c = p.shape[1]
    pred = p.argmax(1)
    cm = np.bincount(y*c+pred, minlength=c*c).reshape(c, c)
    den = cm.sum(0)+cm.sum(1)
    f1 = np.divide(2*np.diag(cm), den, out=np.zeros(c, dtype=np.float64), where=den>0)
    aucs, aps = [], []
    for category in range(c):
        truth = (y == category).astype(np.float64)
        _, inverse, counts = np.unique(p[:, category], return_inverse=True, return_counts=True)
        positives = np.bincount(inverse, weights=truth, minlength=len(counts))
        negatives = counts-positives
        npos, nneg = positives.sum(), negatives.sum()
        assert npos>0 and nneg>0
        neg_before = np.cumsum(negatives)-negatives
        aucs.append(float(np.sum(positives*(neg_before+0.5*negatives))/(npos*nneg)))
        positives, counts = positives[::-1], counts[::-1]
        precision = np.cumsum(positives)/np.cumsum(counts)
        aps.append(float(np.sum(positives*precision)/npos))
    accuracy = float(np.mean(y == pred))
    return {"accuracy": accuracy, "macro_f1": float(f1.mean()), "micro_f1": accuracy,
            "roc_auc_macro": float(np.mean(aucs)), "average_precision_macro": float(np.mean(aps))}


