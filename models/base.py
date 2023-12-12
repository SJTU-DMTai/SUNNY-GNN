import torch.nn as nn


class BaseGNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super(BaseGNN, self).__init__()

    def set_graph(self, g):
        raise NotImplementedError

    def get_emb(self, x, **kwargs):
        raise NotImplementedError

    def forward(self, h, **kwargs):
        raise NotImplementedError
