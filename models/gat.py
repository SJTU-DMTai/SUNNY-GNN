import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.functional import edge_softmax
from dgl.utils import expand_as_pair
from models.base import BaseGNN


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class GATConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer('bias', None)
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, edge_weight, get_attention=False):

        with graph.local_scope():
            if isinstance(feat, tuple):
                src_prefix_shape = feat[0].shape[:-1]
                dst_prefix_shape = feat[1].shape[:-1]
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    feat_src = self.fc(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats)
                else:
                    feat_src = self.fc_src(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc_dst(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats)
            else:
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    *src_prefix_shape, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    h_dst = h_dst[:graph.number_of_dst_nodes()]
                    dst_prefix_shape = (graph.number_of_dst_nodes(),) + dst_prefix_shape[1:]

            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            if edge_weight is None:
                e_w = edge_softmax(graph, e)
            else:
                edge_weight = edge_weight.unsqueeze(1).repeat(1, self._num_heads, 1)
                e_w = edge_softmax(graph, e) * edge_weight
            graph.edata['a'] = self.attn_drop(e_w)
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(*dst_prefix_shape, -1, self._out_feats)
                rst = rst + resval
            if self.bias is not None:
                rst = rst + self.bias.view(
                    *((1,) * len(dst_prefix_shape)), self._num_heads, self._out_feats)
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst


class GAT(BaseGNN):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, num_classes, drop=0.5):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        self.in_dim = in_dim
        self.num_classes = num_classes

        self.gat_layers.append(
            GATConv(in_dim, hidden_dim, num_heads[0], feat_drop=drop, attn_drop=drop, activation=F.elu)
        )
        self.gat_layers.append(
            GATConv(hidden_dim * num_heads[0], out_dim, num_heads[1], feat_drop=drop, attn_drop=drop, activation=None)
        )
        self.g = None
        self.fc = nn.Linear(out_dim, num_classes)
        self.epsilon = nn.Parameter(torch.FloatTensor([1e-12]), requires_grad=False)
        self.loss = nn.CrossEntropyLoss()

    def set_graph(self, g):
        self.g = g

    def forward(self, h, offset=None, get_att=False):
        g = self.g
        with g.local_scope():
            if get_att:
                _, att = h
                return att
            if offset is not None:
                h = h[offset]
            logits = self.fc(h).view(-1, self.num_classes)
            logits = logits / (torch.max(torch.norm(logits, dim=1, keepdim=True), self.epsilon))
            return logits

    def get_emb(self, x, edge_weight=None, get_att=False):
        edge_weight = edge_weight if edge_weight is not None else [None, None]
        g = self.g
        with g.local_scope():
            h = self.gat_layers[0](g, x, edge_weight[0])
            h = h.flatten(1)
            h = self.gat_layers[1](g, h, edge_weight[0], get_attention=get_att)
            if get_att:
                h, attn = h
                return h.mean(1), attn.flatten(0)
            return h.mean(1)

    def get_all_emb(self, x, edge_weight=None):
        g = self.g
        edge_weight = edge_weight if edge_weight is not None else [None, None]
        with g.local_scope():
            h1 = self.gat_layers[0](g, x, edge_weight[0])
            h = h1.flatten(1)
            h2 = self.gat_layers[1](g, h, edge_weight[1])
            return [h1.mean(1), h2.mean(1)]