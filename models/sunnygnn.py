import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import aug_mask
import dgl


class ExtractorMLP(nn.Module):
    def __init__(self, in_dim, bias=True):
        super().__init__()
        self.in_dim = in_dim
        hid_dim = in_dim * 2
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_dim, hid_dim, bias), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(hid_dim, 1, bias)
        )

    def forward(self, emb):
        att_log_logits = self.feature_extractor(emb)
        return att_log_logits


class SunnyGNN(nn.Module):
    def __init__(self, pret_encoder, encoder, extractor, in_dim, n_heads=1, dropout=0.5):
        super().__init__()
        self.pret_encoder = pret_encoder
        self.encoder = encoder
        self.extractor = extractor
        hid_dim = 32
        relu = nn.ReLU()
        drop = nn.Dropout(dropout)
        self.f = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, hid_dim, bias=False), relu, drop),
            nn.Sequential(nn.Linear(256, hid_dim, bias=False), relu, drop),
            nn.Sequential(nn.Linear(64, hid_dim, bias=False), relu, drop)])

        self.proj_head = nn.Sequential(nn.Linear(64, 32, bias=False), relu, drop)

        self.sparsity_mask_coef = 1e-4
        self.sparsity_ent_coef = 1e-2
        self.MIN_WEIGHT = -1e5

    def set_config(self, config):
        self.max_topk = config['max_topk']
        self.min_topk = config['min_topk']
        self.max_epoch = config['max_epoch']
        self.n_pos = config['n_pos']
        self.n_neg = config['n_neg']
        self.k = config['k']
        self.temp = config['temp']
        self.tau = config['tau']

    def sampling(self, att_log_logit, training):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            gated_input = ((att_log_logit + random_noise) / self.temp).sigmoid()
        else:
            gated_input = att_log_logit.sigmoid()
        return gated_input

    def sparsity(self, edge_mask, eps=1e-6):
        sparsity = 0.
        ent = -edge_mask * torch.log(edge_mask + eps) - (1 - edge_mask) * torch.log(1 - edge_mask + eps)
        sparsity += self.sparsity_mask_coef * edge_mask.mean()
        sparsity += self.sparsity_ent_coef * ent.mean()
        return sparsity

    def get_cts_mask(self, g, topk, k):
        with torch.no_grad():
            pos_edge_mask, neg_edge_mask, cts_index = aug_mask(g, self.n_pos, self.n_neg, topk, k)
        return pos_edge_mask, neg_edge_mask, cts_index

    def cts_loss(self, anchor_emb, pos_emb, neg_emb, pos_logits, neg_logits, labels):
        anchor_emb = anchor_emb.unsqueeze(1)
        num_graphs = anchor_emb.shape[0]
        if num_graphs == 0:
            return torch.tensor(0.).to(anchor_emb.device)

        def sim_matrix(anchor, aug):
            for i in range(num_graphs):
                sim = torch.exp(torch.cosine_similarity(anchor[i], aug, dim=2) / self.tau)
                if i == 0:
                    sim_tensor = sim.unsqueeze(0)
                else:
                    sim_tensor = torch.cat((sim_tensor, sim.unsqueeze(0)), dim=0)
            return sim_tensor

        pos_sim = sim_matrix(anchor_emb, pos_emb)
        neg_sim = sim_matrix(anchor_emb, neg_emb)

        cts_loss = 0.
        for i in range(num_graphs):
            l = labels[i]
            same_label_idx = torch.nonzero(labels == l).view(-1)
            diff_label_idx = torch.nonzero(labels != l).view(-1)
            pos_conf_coef = pos_logits[i, :, l]
            neg_conf_coef = (1 - neg_logits[i, :, l])
            pos_sim_i = ( pos_conf_coef * pos_sim[i, i, :]).sum() + (
                    pos_logits[same_label_idx, :, l] * pos_sim[i, same_label_idx, :]).sum()
            neg_sim_i = ( neg_conf_coef * neg_sim[i, i, :]).sum() \
                        + ((1 - pos_logits[diff_label_idx, :, l]) * pos_sim[i, diff_label_idx, :]).sum() \
                        + ((1 - neg_logits[same_label_idx, :, l]) * neg_sim[i, same_label_idx, :]).sum()
            denominator = pos_sim_i + neg_sim_i
            cts_loss_i = -torch.log(torch.cat([pos_sim[i, i, :]]) / denominator).mean() \
                         - torch.log(torch.cat([pos_sim[i, same_label_idx, :]]) / denominator).mean()
            cts_loss += cts_loss_i

        return cts_loss / num_graphs

    def loss(self, edge_mask, logits, labels):
        pred_loss = F.cross_entropy(logits, labels)
        sparsity_loss = self.sparsity(edge_mask)
        return pred_loss + sparsity_loss

    def batched_emb(self, g, x, batched_edge_mask, idx):
        n_samples = batched_edge_mask.shape[1]
        g.ndata['x'] = x
        gs = dgl.batch([g] * n_samples)
        x = gs.ndata.pop('x')
        self.encoder.set_graph(gs)
        h = self.encoder.get_emb(x, batched_edge_mask.view(2, -1, 1))
        h = h.view(g.number_of_nodes(), n_samples, -1, h.shape[-1]).mean(2)
        offset = torch.cat([torch.tensor([0], device=gs.device),
                            gs.batch_num_nodes().cumsum(dim=0)[:int(gs.batch_size / n_samples) - 1]])
        proj = self.proj_head(h[offset])
        logits = self.encoder(h, offset).view(g.batch_size, n_samples, -1)
        logits = torch.softmax(logits, dim=2)
        del gs
        return proj[idx], logits[idx]

    def get_edge_att(self, g, all_emb, e_batch, h_target):
        def calc_att(mask, hop_batch, k=1):
            n_map = g.ndata['_ID']
            e = g.edges()[0][mask], g.edges()[1][mask]
            emb = torch.cat([self.f[k - 1](all_emb[k - 1][n_map[e[0]]]),
                             self.f[k](all_emb[k][n_map[e[1]]]),
                             h_target[hop_batch]], dim=1)
            att = self.extractor(emb)
            return att

        e_h_mask = g.edata['e_h_mask'].T
        one_hop_mask = torch.nonzero(e_h_mask[0]).view(-1)
        one_hop_att = calc_att(one_hop_mask, e_batch[one_hop_mask], k=2)

        two_hop_mask = torch.nonzero(e_h_mask[1]).view(-1)
        two_hop_att = calc_att(two_hop_mask, e_batch[two_hop_mask], k=1)

        edge_att = torch.zeros((2, g.number_of_edges(), 1), device=g.device)
        edge_att[:, :, :] = self.MIN_WEIGHT
        edge_att[0][two_hop_mask] = two_hop_att
        edge_att[1][one_hop_mask] = one_hop_att

        return edge_att

    def forward(self, g, all_emb, labels, training=False, explain=False, epoch=0):
        x = all_emb[0][g.ndata['_ID']]
        with g.local_scope():
            offset_node = torch.cat([torch.tensor([0], device=g.device),
                                     g.batch_num_nodes().cumsum(dim=0)[:-1]])
            h_target = self.f[2](all_emb[2][g.ndata['_ID'][offset_node]])
            e_batch = torch.repeat_interleave(torch.arange(g.batch_size, device=g.device),
                                              g.batch_num_edges())
            edge_att = self.get_edge_att(g, all_emb, e_batch, h_target)
            if explain:
                return edge_att

            edge_att = self.sampling(edge_att, training)
            self.encoder.set_graph(g)
            enc_emb = self.encoder.get_emb(x, edge_att)
            enc_logits = self.encoder(enc_emb, offset_node)

            if training:
                pred_loss = self.loss(edge_att, enc_logits, labels)
                g.edata['e_att'] = edge_att.view(2, g.number_of_edges()).T
                enc_proj = self.proj_head(enc_emb[offset_node])
                topk = (self.max_topk - (self.max_topk - self.min_topk) * epoch / self.max_epoch)
                pos_edge_att, neg_edge_att, cts_idxs = self.get_cts_mask(g, topk, self.k)
                pos_enc_proj, pos_enc_logits = self.batched_emb(g, x, pos_edge_att.to(g.device), cts_idxs)
                neg_enc_proj, neg_enc_logits = self.batched_emb(g, x, neg_edge_att.to(g.device), cts_idxs)
                cts_loss = self.cts_loss(enc_proj[cts_idxs], pos_enc_proj, neg_enc_proj, pos_enc_logits,
                                          neg_enc_logits, labels[cts_idxs])
                return enc_logits, [pred_loss, cts_loss]

            return enc_logits, None

