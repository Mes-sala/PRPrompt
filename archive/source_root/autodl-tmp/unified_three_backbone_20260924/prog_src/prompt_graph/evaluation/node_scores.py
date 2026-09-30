"""Common node scoring boundary: labels enter only after prediction."""
from contextvars import ContextVar

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

PREDICTION_SINK = ContextVar("prediction_sink", default=None)


@torch.no_grad()
def classification_scores(logits, labels):
    probabilities = logits.softmax(-1).detach().cpu().numpy()
    labels = labels.view(-1).detach().cpu().numpy()
    sink = PREDICTION_SINK.get()
    if sink is not None:
        sink.append({"probabilities": probabilities.tolist(), "labels": labels.tolist()})
    classes = probabilities.shape[1]
    prediction = probabilities.argmax(-1)
    accuracy = float(accuracy_score(labels, prediction))
    f1 = float(f1_score(labels, prediction, labels=np.arange(classes), average="macro", zero_division=0))
    if len(np.unique(labels)) != classes:
        return accuracy, f1, float("nan"), float("nan")
    targets = np.eye(classes)[labels]
    return (accuracy, f1, float(roc_auc_score(targets, probabilities, average="macro")),
            float(average_precision_score(targets, probabilities, average="macro")))
