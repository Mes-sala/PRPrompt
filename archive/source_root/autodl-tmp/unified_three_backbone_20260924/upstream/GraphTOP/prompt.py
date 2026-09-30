import torch.nn as nn
import torch.nn.functional as F


class Rewiring(nn.Module):
    def __init__(self, hidden_dim):
        super(Rewiring, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.fc2(F.relu(self.fc1(x)))
        return x
