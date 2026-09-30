import torch
import torch.nn as nn
import torch.nn.functional as F


class Gprompt_tuning_loss(nn.Module):
    def __init__(self, tau=0.1):
        super().__init__()
        self.tau = tau

    def forward(self, embedding, center_embedding, labels):
        similarity_matrix = F.cosine_similarity(
            embedding.unsqueeze(1), center_embedding.unsqueeze(0), dim=-1
        ) / self.tau
        return F.cross_entropy(similarity_matrix, labels, reduction='mean')


def Gprompt_link_loss(node_emb, pos_emb, neg_emb, temperature=0.2):
    r"""Refer to GraphPrompt original codes"""
    x = torch.exp(F.cosine_similarity(node_emb, pos_emb, dim=-1) / temperature)
    y = torch.exp(F.cosine_similarity(node_emb, neg_emb, dim=-1) / temperature)

    loss = -1 * torch.log(x / (x + y))
    return loss.mean()
