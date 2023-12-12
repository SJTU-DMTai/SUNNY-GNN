import torch
import torch.nn.functional as F
from torch import nn
import dgl
from dgl import function as fn
from dgl.nn.pytorch import edge_softmax
from dgl.nn.pytorch.utils import Identity
from models.base import BaseGNN


class Attention(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(Attention, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)

        self.softmax = nn.Softmax(dim=-1)
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

    def forward(self, embeds):
        beta = []
        attn_curr = self.attn_drop(self.att)
        for embed in embeds:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)
        z = 0
        for i in range(len(embeds)):
            z += embeds[i] * beta[i]
        return z


class myHeteroGATConv(nn.Module):
    def __init__(
            self,
            edge_feats,
            num_etypes,
            in_feats,
            out_feats,
            num_heads,
            feat_drop=0.0,
            attn_drop=0.0,
            negative_slope=0.2,
            residual=False,
            activation=None,
            allow_zero_in_degree=False,
            bias=False,
            alpha=0.0,
            share_weight=False,
    ):
        super(myHeteroGATConv, self).__init__()
        self.edge_feats = edge_feats
        self.num_heads = num_heads
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.in_src_feats = self._in_dst_feats = in_feats
        self.allow_zero_in_degree = allow_zero_in_degree
        self.shared_weight = share_weight
        self.edge_emb = nn.Parameter(torch.FloatTensor(size=(num_etypes, edge_feats)))
        if not share_weight:
            self.fc = self.weight = nn.ModuleDict({
                name: nn.Linear(in_feats[name], out_feats * num_heads, bias=False) for name in in_feats
            })
        else:
            in_dim = None
            for name in in_feats:
                if in_dim:
                    assert in_dim == in_feats[name]
                else:
                    in_dim = in_feats[name]
            self.fc = nn.Linear(in_dim, out_feats * num_heads, bias=False)
        self.fc_e = nn.Linear(edge_feats, edge_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_e = nn.Parameter(torch.FloatTensor(size=(1, num_heads, edge_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self.shared_weight:
                in_dim = None
                for name in in_feats:
                    if in_dim:
                        assert in_dim == in_feats[name]
                    else:
                        in_dim = in_feats[name]
                if in_dim != num_heads * out_feats:
                    self.res_fc = nn.Linear(in_dim, num_heads * out_feats, bias=False)
                else:
                    self.res_fc = Identity()
            else:
                self.res_fc = nn.ModuleDict()
                for ntype in in_feats.keys():
                    if self._in_dst_feats[ntype] != num_heads * out_feats:
                        self.res_fc[ntype] = nn.Linear(self._in_dst_feats[ntype], num_heads * out_feats, bias=False)
                    else:
                        self.res_fc[ntype] = Identity()
        else:
            self.register_buffer("res_fc", None)
        self.reset_parameters()
        self.activation = activation
        self.bias = bias
        if bias:
            self.bias_param = nn.Parameter(torch.zeros((1, num_heads, out_feats)))
        self.alpha = alpha

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        if self.shared_weight:
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            for name in self.fc:
                nn.init.xavier_normal_(self.fc[name].weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_e.weight, gain=gain)
        nn.init.normal_(self.edge_emb, 0, 1)

    def set_allow_zero_in_degree(self, set_value):
        self.allow_zero_in_degree = set_value

    def forward(self, graph, nfeat, res_attn=None, edge_weight=None):
        with graph.local_scope():
            funcs = {}

            for ntype in graph.ntypes:
                h = self.feat_drop(nfeat[ntype])
                if self.shared_weight:
                    feat = self.fc(h).view(-1, self.num_heads, self.out_feats)
                else:
                    feat = self.fc[ntype](h).view(-1, self.num_heads, self.out_feats)
                graph.nodes[ntype].data['ft'] = feat
                if self.res_fc is not None:
                    graph.nodes[ntype].data['h'] = h

            for src, etype, dst in graph.canonical_etypes:
                feat_src = graph.nodes[src].data['ft']
                feat_dst = graph.nodes[dst].data['ft']
                el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
                graph.nodes[src].data['el'] = el
                er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
                graph.nodes[dst].data['er'] = er
                e_feat = self.edge_emb[int(etype)].unsqueeze(0)
                e_feat = self.fc_e(e_feat).view(-1, self.num_heads, self.edge_feats)
                ee = (e_feat * self.attn_e).sum(dim=-1).unsqueeze(-1).expand(graph.number_of_edges(etype),
                                                                             self.num_heads, 1)
                graph.apply_edges(fn.u_add_v("el", "er", "e"), etype=etype)
                graph.edges[etype].data["a"] = self.leaky_relu(graph.edges[etype].data.pop("e") + ee)
                if edge_weight is not None:
                    edge_mask = edge_weight[etype].unsqueeze(1).repeat(1, self.num_heads, 1)
                    graph.edges[etype].data["a"] = edge_mask * graph.edges[etype].data["a"]

            hg = dgl.to_homogeneous(graph, edata=["a"])
            a = self.attn_drop(edge_softmax(hg, hg.edata.pop("a")))
            e_t = hg.edata['_TYPE']

            for src, etype, dst in graph.canonical_etypes:
                t = graph.get_etype_id(etype)
                graph.edges[etype].data["a"] = a[e_t == t]
                if res_attn is not None:
                    graph.edges[etype].data["a"] = graph.edges[etype].data["a"] * (1 - self.alpha) + res_attn[
                        etype] * self.alpha
                funcs[etype] = (fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))

            graph.multi_update_all(funcs, 'sum')
            rst = graph.ndata.pop('ft')
            if type(rst) != dict:
                rst = {graph.ntypes[0]: rst}

            graph.edata.pop("el")
            graph.edata.pop("er")
            if self.res_fc is not None:
                for ntype in graph.ntypes:
                    if self.shared_weight:
                        rst[ntype] = self.res_fc(graph.nodes[ntype].data['h']).view(
                            graph.nodes[ntype].data['h'].shape[0], self.num_heads, self.out_feats) + rst[ntype]
                    else:
                        rst[ntype] = self.res_fc[ntype](graph.nodes[ntype].data['h']).view(
                            graph.nodes[ntype].data['h'].shape[0], self.num_heads, self.out_feats) + rst[ntype]

            if self.bias:
                for ntype in graph.ntypes:
                    rst[ntype] = rst[ntype] + self.bias_param

            if self.activation:
                for ntype in graph.ntypes:
                    rst[ntype] = self.activation(rst[ntype])
            res_attn = {e: graph.edges[e].data["a"].detach() for e in graph.etypes}
            graph.edata.pop("a")
            return rst, res_attn


class SimpleHeteroHGN(BaseGNN):
    def __init__(
            self,
            edge_dim,
            num_etypes,
            in_dims,
            num_hidden,
            num_classes,
            num_layers,
            heads,
            feat_drop,
            attn_drop,
            negative_slope,
            residual,
            alpha,
            shared_weight=False,
    ):
        super(SimpleHeteroHGN, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.g = None
        self.g_cs = []
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = F.elu

        self.gat_layers.append(
            myHeteroGATConv(
                edge_dim,
                num_etypes,
                in_dims,
                num_hidden,
                heads[0],
                feat_drop,
                attn_drop,
                negative_slope,
                False,
                self.activation,
                alpha=alpha,
            )
        )
        for l in range(1, num_layers):
            in_dims = {n: num_hidden * heads[l - 1] for n in in_dims}
            self.gat_layers.append(
                myHeteroGATConv(
                    edge_dim,
                    num_etypes,
                    in_dims,
                    num_hidden,
                    heads[l],
                    feat_drop,
                    attn_drop,
                    negative_slope,
                    residual,
                    self.activation,
                    alpha=alpha,
                    share_weight=shared_weight,
                )
            )

        in_dims = num_hidden * heads[0]
        self.fc = nn.Linear(in_dims, num_classes)
        self.epsilon = nn.Parameter(torch.FloatTensor([1e-12]), requires_grad=False)

    def set_graph(self, g):
        self.g = g

    def forward(self, h, target_ntype=None, offset=None, get_att=False, pooling=False):
        g = self.g
        with g.local_scope():
            if get_att:
                _, att = h
                return att
            if g.batch_size == 1:
                h = h[target_ntype]
            else:
                if type(h) != dict:
                    h = h[offset]
                else:
                    h = h[target_ntype][offset]
            logits = self.fc(h)
            logits = logits / (torch.max(torch.norm(logits, dim=1, keepdim=True), self.epsilon))
            return logits

    def get_emb(self, x, edge_weight=None, get_att=False, pooling=False):
        edge_weight = [{k: v[i,:,:] for k, v in edge_weight.items()} for i in range(2)] if edge_weight is not None else [None, None]
        g = self.g
        with g.local_scope():
            h = x
            res_attn = None
            for l in range(self.num_layers):
                h, res_attn = self.gat_layers[l](g, h, res_attn=res_attn, edge_weight=edge_weight[l])
                h = {n: h[n].flatten(1) for n in h}
            if get_att:
                return h, res_attn
            if pooling:
                h = {
                    ntype: self.mlp[ntype](self.activation(h[ntype])) for ntype in h.keys()
                }
            return h

    def get_all_emb(self, x, edge_weight=None):
        edge_weight = edge_weight if edge_weight is not None else [None, None]
        g = self.g
        with g.local_scope():
            hs = []
            h = x
            for l in range(self.num_layers):
                h, _ = self.gat_layers[l](g, h, edge_weight=edge_weight[l])
                h = {n: h[n].flatten(1) for n in h}
                hs.append(h)
            return hs

    def loss(self, x, target_ntype, target_node, label):
        h = self.get_emb(x)
        logits = self.forward(h, target_ntype)
        y = logits[target_node]
        return self.cross_entropy_loss(y, label)
