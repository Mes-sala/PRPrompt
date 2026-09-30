import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from torch_geometric.loader import DataLoader
from sklearn import metrics

from load_data import load_node_data, NodeDownstream, get_subgraphs
from model import GCN
from prompt import Rewiring
from utils import compute_entropy


class NodeTask():
    def __init__(self, dataset_name, shots, hidden_dim, device, pretrain_task, logger):
        self.dataset_name = dataset_name
        self.hidden_dim = hidden_dim
        self.device = device
        self.pretrain_task = pretrain_task
        self.logger = logger
        if dataset_name in ['Cora', 'PubMed', 'Amazon-ratings', 'Minesweeper', 'Flickr']:
            self.data, self.input_dim, self.output_dim = load_node_data(dataset_name, data_folder='./data')
            self.train_node_list, self.test_node_list = NodeDownstream(self.data, shots, test_node_num=2000)
            self.train_data = get_subgraphs(self.data, self.train_node_list)
            self.test_data = get_subgraphs(self.data, self.test_node_list)
        else:
            raise ValueError('Error: invalid dataset name! Supported datasets: [Cora, PubMed, Amazon-ratings, Minesweeper, Flickr]')

        self.initialize_model()
        self.prompt = Rewiring(hidden_dim=hidden_dim).to(self.device)
        self.h = self.gnn(self.data.to(device)).detach()

    def initialize_model(self):
        self.gnn = GCN(input_dim=self.input_dim, hidden_dim=self.hidden_dim, output_dim=self.hidden_dim)
        if self.pretrain_task is not None:
            pretrained_gnn_file = f'./pretrained_gnns/{self.dataset_name}_{self.pretrain_task}_GCN_1.pth'
            self.gnn.load_state_dict(torch.load(pretrained_gnn_file))
        self.gnn.to(self.device)
        self.classifier = nn.Linear(self.hidden_dim, self.output_dim).to(self.device)

    def train(self, batch_size, lr=0.001, decay=0, epochs=100, lambda_e=0., lambda_s=0.):
        eps = 1e-10
        t = 0.2
        train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(self.test_data, batch_size=batch_size*5, shuffle=False)

        learnable_parameters = list(self.classifier.parameters()) + list(self.prompt.parameters())
        optimizer = torch.optim.Adam(learnable_parameters, lr=lr, weight_decay=decay)

        for epoch in range(1, 1 + epochs):
            tau = 0.97 * (1 - epoch / epochs) + 0.03
            self.gnn.train()
            pred_list = []
            label_list = []
            total_loss = []
            for i, data in enumerate(train_loader):
                data = data.to(self.device)
                optimizer.zero_grad()

                prob_all = torch.sigmoid(self.prompt(self.h[data.original_idx][data.batch] + self.h[data.node_idx])/ t)
                prob = prob_all[data.neighbor_mask]

                u0, u1 = torch.rand_like(prob, device=prob.device), torch.rand_like(prob, device=prob.device)
                noise = - torch.log(torch.log(u1 + eps) / torch.log(u0 + eps) + eps)
                m = torch.sigmoid((torch.log(prob + eps) - torch.log(1. - prob + eps) + noise) / tau)
                trainable_edge_weight = torch.repeat_interleave(m, repeats=2)

                edge_weight = deepcopy(data.edge_weight)
                edge_weight[data.trainable_edge] = trainable_edge_weight
                emb = self.gnn(data, edge_weight, pooling='target')

                out = self.classifier(emb)
                cross_entropy = F.cross_entropy(out, data.y.squeeze())

                d_tilde = global_add_pool(prob_all, data.batch) - prob_all[data.target_node_index]
                node_count = data.ptr[1:] - data.ptr[:-1]
                sparsity = d_tilde.squeeze() / (node_count - 1)
                loss_s = (sparsity - 0.5).pow(2).mean()

                loss_e = compute_entropy(prob)

                loss = cross_entropy + lambda_e * loss_e + lambda_s * loss_s
                loss.backward()
                optimizer.step()
                pred_list.extend(out.argmax(1).tolist())
                label_list.extend(data.y.squeeze().tolist())
                total_loss.append(loss.item())
            train_accuracy = metrics.accuracy_score(y_true=label_list, y_pred=pred_list)
            train_loss = np.mean(total_loss)

            if epoch % 1 == 0:
                self.gnn.eval()
                pred_list = []
                label_list = []
                total_loss = []
                for i, data in enumerate(test_loader):
                    data = data.to(self.device)
                    prob_all = torch.sigmoid(self.prompt(self.h[data.original_idx][data.batch] + self.h[data.node_idx])/ t)
                    prob = prob_all[data.neighbor_mask]
                    m = (prob > torch.rand_like(prob)).float()
                    trainable_edge_weight = torch.repeat_interleave(m, repeats=2)

                    edge_weight = deepcopy(data.edge_weight)
                    edge_weight[data.trainable_edge] = trainable_edge_weight
                    emb = self.gnn(data, edge_weight, pooling='target')
                    out = self.classifier(emb)
                    loss = F.cross_entropy(out, data.y.squeeze())
                    pred_list.extend(out.argmax(1).tolist())
                    label_list.extend(data.y.squeeze().tolist())
                    total_loss.append(loss.item())
                test_accuracy = metrics.accuracy_score(y_true=label_list, y_pred=pred_list)
                test_loss = np.mean(total_loss)

                log_info = ''.join(['| epoch: {:3d} '.format(epoch),
                                    '| train_loss: {:7.5f}'.format(train_loss),
                                    '| test_loss: {:7.5f}'.format(test_loss),
                                    '| train_acc: {:7.5f}'.format(train_accuracy),
                                    '| test_acc: {:7.5f}'.format(test_accuracy),
                                    '|'])
                self.logger.info(log_info)
