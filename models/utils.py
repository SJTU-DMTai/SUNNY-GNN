import torch
import dgl
import math


def distance_coef(d, c=0.15):
    return c * math.exp(d)


def aug_mask(g, n_pos, n_neg, topk, k):
    edge_mask = g.edata['e_att']
    pos_masks = None
    neg_masks = None
    cts_idxs = torch.tensor([]).long()

    e_d_mask = torch.zeros_like(edge_mask)
    e_d_mask[:, 1] = distance_coef(1) * edge_mask[:, 1]
    e_d_mask[:, 0] = distance_coef(2) * edge_mask[:, 0]
    e_d_mask[e_d_mask > 1] = 1.
    g.edata['e_att'] = e_d_mask

    gs = dgl.unbatch(g)
    for i in range(len(gs)):
        e_h_mask = gs[i].edata['e_h_mask'].T.cpu()
        relevant_edges = torch.nonzero(e_h_mask[0]).shape[0] + torch.nonzero(e_h_mask[1]).shape[0]
        mask = gs[i].edata['e_att'].T.cpu()
        mask = mask.view(-1)
        pos_mask, neg_mask, cts_idxs = \
            perturb_mask(mask, n_pos, n_neg, topk, k, cts_idxs, i, relevant_edges)
        if pos_masks is None:
            pos_masks = pos_mask.view(n_pos, 2, -1).transpose(0, 1)
            neg_masks = neg_mask.view(n_neg, 2, -1).transpose(0, 1)
        else:
            pos_masks = torch.cat((pos_masks, pos_mask.view(n_pos, 2, -1).transpose(0, 1)), dim=2)
            neg_masks = torch.cat((neg_masks, neg_mask.view(n_neg, 2, -1).transpose(0, 1)), dim=2)
    del gs

    return pos_masks, neg_masks, cts_idxs


def aug_mask_hetero(g, n_pos, n_neg, topk, k):
    edge_mask = g.edata['e_att']
    pos_masks = {etp: [] for etp in g.etypes}
    neg_masks = {etp: [] for etp in g.etypes}
    cts_idxs = torch.tensor([]).long()

    for etp in edge_mask.keys():
        e_d_mask = torch.zeros_like(edge_mask[etp])
        e_d_mask[:, 1] = distance_coef(1) * edge_mask[etp][:, 1]
        e_d_mask[:, 0] = distance_coef(2) * edge_mask[etp][:, 0]
        g.edata['e_att'][etp] = e_d_mask

    gs = dgl.unbatch(g)

    for i in range(len(gs)):
        mask = torch.tensor([])
        e_num = torch.tensor([], dtype=torch.uint8)
        m = gs[i].edata['e_att']
        e_h_m = gs[i].edata['e_h_mask']
        relevant_edges = 0
        for i, etp in enumerate(m):
            mask = torch.cat((mask, m[etp].T.cpu().view(-1)))
            relevant_edges += torch.nonzero(e_h_m[etp]).shape[0]
            e_num = torch.cat((e_num, torch.tensor([i]*m[etp].shape[0]*m[etp].shape[1])))

        pos_mask, neg_mask, cts_idxs = \
            perturb_mask(mask, n_pos, n_neg, topk, k, cts_idxs, i, relevant_edges)

        for i, etp in enumerate(pos_masks):
            pos_masks[etp].append(pos_mask[:, e_num==i])
            neg_masks[etp].append(neg_mask[:, e_num==i])
    del gs

    for etp in pos_masks:
        pos_masks[etp] = torch.cat(pos_masks[etp], dim=1)
        neg_masks[etp] = torch.cat(neg_masks[etp], dim=1)

    return pos_masks, neg_masks, cts_idxs


def perturb_mask(mask, n_pos, n_neg, topk, k, cts_idxs, i, relevant_edges):
    pos_loc = torch.arange(n_pos)
    neg_loc = torch.arange(n_neg)

    partition = int(relevant_edges * topk)
    sample_prob, sample_idxs = torch.sort(mask, descending=True)

    pos_idxs = sample_idxs[partition:]
    pos_prob = sample_prob[partition:]
    num_pos_sample = int(len(pos_idxs) * k)
    neg_idxs = sample_idxs[:partition]
    neg_prob = sample_prob[:partition]
    num_neg_sample = int(len(neg_idxs) * k)

    if num_neg_sample == 0 or num_pos_sample == 0:
        pos_mask = torch.stack([mask] * n_pos)
        neg_mask = torch.stack([mask] * n_neg)

    else:
        pos_weight = mask[neg_idxs].mean()
        neg_weight = mask[pos_idxs].mean()

        pos_probs = torch.stack([pos_prob] * n_pos)
        if pos_probs.sum() <= 0:
            pos_sample = torch.randint(0, len(pos_idxs), (n_pos, num_pos_sample))
        else:
            pos_sample = torch.multinomial(pos_probs, num_pos_sample, replacement=False)
        pos_mask = mask.clone()
        pos_mask = torch.stack([pos_mask] * n_pos)
        for j in range(num_pos_sample):
            pos_mask[pos_loc, pos_idxs[pos_sample][:, j]] = pos_weight

        neg_probs = torch.stack([neg_prob] * n_neg)
        if neg_probs.sum() <= 0:
            neg_sample = torch.randint(0, len(neg_idxs), (n_neg, num_neg_sample))
        else:
            neg_sample = torch.multinomial(neg_probs, num_neg_sample, replacement=False)
        neg_mask = mask.clone()
        neg_mask = torch.stack([neg_mask] * n_neg)
        for j in range(num_neg_sample):
            neg_mask[neg_loc, neg_idxs[neg_sample][:, j]] = neg_weight
        cts_idxs = torch.cat((cts_idxs, torch.tensor([i])))
    return pos_mask, neg_mask, cts_idxs