import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from models.base import BaseGNN


class GCN(BaseGNN):
    def __init__(self, in_dim, hidden_dim, out_dim, num_classes, drop=0.5):
        super(GCN, self).__init__()
        self.gcn_layers = nn.ModuleList([
            GraphConv(in_dim, hidden_dim, activation=F.relu, allow_zero_in_degree=True),
            GraphConv(hidden_dim, out_dim, activation=None, allow_zero_in_degree=True)
        ])
        self.fc = nn.Linear(out_dim, num_classes)
        self.in_dim = in_dim
        self.loss = nn.CrossEntropyLoss()
        self.epsilon = nn.Parameter(torch.FloatTensor([1e-12]), requires_grad=False)
        self.g = None
        self.drop = nn.Dropout(drop)

    def set_graph(self, g):
        self.g = g

    def forward(self, h, offset=None):
        g = self.g
        with g.local_scope():
            if offset is not None:
                h = h[offset]
            logits = self.fc(h)
            logits = logits / (torch.max(torch.norm(logits, dim=1, keepdim=True), self.epsilon))
            return logits

    def get_emb(self, x, edge_weight=None):
        edge_weight = edge_weight if edge_weight is not None else [None, None]
        g = self.g
        with g.local_scope():
            h = self.gcn_layers[0](g, x, edge_weight=edge_weight[0])
            h = self.drop(h)
            h = self.gcn_layers[1](g, h, edge_weight=edge_weight[1])
            h = self.drop(h)
            return h

    def get_all_emb(self, x, edge_weight=None):
        edge_weight = edge_weight if edge_weight is not None else [None, None]
        g = self.g
        with g.local_scope():
            h1 = self.gcn_layers[0](g, x, edge_weight=edge_weight[0])
            h = self.gcn_layers[1](g, h1, edge_weight=edge_weight[1])
            return [h1, h]
