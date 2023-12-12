import torch


def fidelity(model, g, x, edge_mask, labels):
    y = labels
    model.eval()
    model.set_graph(g)
    offset = torch.cat([torch.tensor([0], device=g.device), g.batch_num_nodes().cumsum(dim=0)[:-1]])
    logits = model(model.get_emb(x), offset)
    y_hat = logits.argmax(dim=-1)
    indexs = (y == y_hat)
    y = y[indexs]
    y_hat = y_hat[indexs]

    explain_logits = model(model.get_emb(x, edge_weight=edge_mask), offset)
    explain_y_hat = explain_logits.argmax(dim=-1)[indexs]

    complement_logits = model(model.get_emb(x, edge_weight=(1.-edge_mask)), offset)
    complement_y_hat = complement_logits.argmax(dim=-1)[indexs]

    pos_fidelity = ((y_hat == y).float() - (complement_y_hat == y).float()).abs()
    neg_fidelity = ((y_hat == y).float() - (explain_y_hat == y).float()).abs()

    return pos_fidelity, neg_fidelity


def fidelity_with_logits(model, g, x, logits, labels):
    explain_logits, complement_logits = logits
    y = labels
    model.eval()
    model.set_graph(g)
    offset = torch.cat([torch.tensor([0], device=g.device), g.batch_num_nodes().cumsum(dim=0)[:-1]])
    logits = model(model.get_emb(x), offset)
    y_hat = logits.argmax(dim=-1)
    indexs = (y == y_hat)
    y = y[indexs]
    y_hat = y_hat[indexs]

    explain_y_hat = explain_logits.argmax(dim=-1)[indexs]
    complement_y_hat = complement_logits.argmax(dim=-1)[indexs]

    pos_fidelity = ((y_hat == y).float() - (complement_y_hat == y).float()).abs()
    neg_fidelity = ((y_hat == y).float() - (explain_y_hat == y).float()).abs()

    return pos_fidelity, neg_fidelity


def fidelity_x(model, g, x, g_explains, labels):
    y = labels
    model.eval()
    model.set_graph(g)
    offset = torch.cat([torch.tensor([0], device=g.device), g.batch_num_nodes().cumsum(dim=0)[:-1]])
    logits = model(model.get_emb(x), offset)
    y_hat = logits.argmax(dim=-1)
    if y != y_hat:
        return None, None

    g_x, g_c = g_explains
    model.set_graph(g_x)
    explain_logits = model(model.get_emb(g_x.ndata['nfeat']))
    explain_y_hat = explain_logits.argmax(dim=-1)

    model.set_graph(g_c)
    complement_logits = model(model.get_emb(g_c.ndata['nfeat']))
    complement_y_hat = complement_logits.argmax(dim=-1)

    pos_fidelity = ((y_hat == y).float() - (complement_y_hat == y).float()).abs()
    neg_fidelity = ((y_hat == y).float() - (explain_y_hat == y).float()).abs()

    return pos_fidelity, neg_fidelity


def characterization_score(pos_fidelity, neg_fidelity, pos_weight=0.5, neg_weight=0.5):
    if (pos_weight + neg_weight) != 1.0:
        raise ValueError(f"The weights need to sum up to 1 "
                         f"(got {pos_weight} and {neg_weight})")

    denom = (pos_weight / pos_fidelity) + (neg_weight / (1. - neg_fidelity))
    return 1. / denom


if __name__ == '__main__':
    print(characterization_score(0.5, 0.5))