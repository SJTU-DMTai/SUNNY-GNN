import copy
import numpy as np
from train.utils import *
from tools.fidelity import *
from dgl.dataloading import GraphDataLoader


def train(cfg):
    method = cfg.method
    common_cfg = cfg.hyparams['common']
    method_cfg = cfg.hyparams[method][cfg.encoder_type]

    n_epochs = common_cfg['n_epochs']
    batch_size = common_cfg['batch_size']
    lr = common_cfg['learning_rate']

    debug_epoch = 10
    best_acc = 0.

    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    gw, model, info = get_model(cfg)
    xw = gw.ndata.pop('nfeat')

    model.to(device)
    if type(xw) == dict:
        for key in xw.keys():
            xw[key] = xw[key].to(device)
    else:
        xw = xw.to(device)

    train_set, valid_set, test_set, _ = construct_dataset(gw, info, cfg)
    train_loader = GraphDataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = GraphDataLoader(valid_set, batch_size=batch_size, shuffle=False)
    test_loader = GraphDataLoader(test_set, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-5)

    for epoch in range(n_epochs):
        train_pred_loss = 0.
        train_cts_loss = 0.
        preds, labels = torch.tensor([]), torch.tensor([])

        model.train()

        with torch.no_grad():
            model.pret_encoder.set_graph(gw.to(device))
            all_emb = model.pret_encoder.get_all_emb(xw)
            all_emb = [xw, *all_emb]

        for g, label in train_loader:
            torch.cuda.empty_cache()
            g = g.to(device)
            label = label.to(device)
            pred, loss = model(g, all_emb, label, training=True, epoch=epoch)

            pred_loss, cts_loss = loss
            pred_loss = pred_loss + method_cfg['cts_coef'] * cts_loss
            train_cts_loss += cts_loss.item()

            optimizer.zero_grad()
            pred_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
            train_pred_loss += pred_loss.item()
            preds = torch.cat([preds, pred.detach().cpu()])
            labels = torch.cat([labels, label.detach().cpu()])

        train_acc = accuracy(preds, labels)

        model.eval()
        preds, labels = torch.tensor([]), torch.tensor([])
        for g, label in valid_loader:
            torch.cuda.empty_cache()
            g = g.to(device)
            label = label.to(device)
            with torch.no_grad():
                pred, _ = model(g, all_emb, label, training=False)
            preds = torch.cat([preds, pred.detach().cpu()])
            labels = torch.cat([labels, label.detach().cpu()])
        valid_acc = accuracy(preds, labels)
        if epoch % debug_epoch == 0:
            print(f"Epoch: {epoch}\t Train pred Loss: {train_pred_loss:.4f}, "
                   f"Train Acc: {train_acc:.4f}, Validation Acc: {valid_acc:.4f}, ")

        if valid_acc >= best_acc:
            best_acc = valid_acc
            best_model = copy.deepcopy(model)

    best_model.eval()
    with torch.no_grad():
        best_model.pret_encoder.set_graph(gw.to(device))
        all_emb = best_model.pret_encoder.get_all_emb(xw)
        all_emb = [xw, *all_emb]

    preds, labels = torch.tensor([]), torch.tensor([])
    for g, label in test_loader:
        torch.cuda.empty_cache()
        g = g.to(device)
        label = label.to(device)
        with torch.no_grad():
            pred, _ = best_model(g, all_emb, label, training=False)
        preds = torch.cat([preds, pred.detach().cpu()])
        labels = torch.cat([labels, label.detach().cpu()])

    test_acc = accuracy(preds, labels)
    metrics = {'test_acc': test_acc}
    if not os.path.exists(f'ckpt/{cfg.dataset}'):
        os.makedirs(f'ckpt/{cfg.dataset}')
    torch.save(model.state_dict(), f'ckpt/{cfg.dataset}/{method}-{cfg.encoder_type}-seed-{cfg.seed}.pt')
    for k, v in metrics.items():
        print(f"{k}: {np.mean(v)}")

    return metrics


def train_explain(cfg):
    method = cfg.method
    common_cfg = cfg.hyparams['common']
    batch_size = common_cfg['batch_size']

    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    gw, explainer, info = get_model(cfg)
    explainer.to(device)
    xw = gw.ndata.pop('nfeat').to(device)

    cfg_cp = copy.deepcopy(cfg)
    cfg_cp.method = cfg_cp.encoder_type
    _, gnn_model, _ = get_model(cfg_cp)
    gnn_model.load_state_dict(
        torch.load(f'{cfg.ckpt_dir}/{cfg.dataset}/{cfg_cp.method}-seed-{cfg.seed}-pretrain.pt'))
    gnn_model.to(device)
    gnn_model.eval()

    if type(xw) == dict:
        for key in xw.keys():
            xw[key] = xw[key].to(device)
    else:
        xw = xw.to(device)

    _, _, test_set, _ = construct_dataset(gw, info, cfg)
    test_loader = GraphDataLoader(test_set, batch_size=batch_size, shuffle=False)

    explainer.load_state_dict(
        torch.load(f'{cfg.ckpt_dir}/{cfg.dataset}/{method}-{cfg.encoder_type}-seed-{cfg.seed}.pt'))

    explainer.eval()
    gnn_model.eval()

    with torch.no_grad():
        explainer.pret_encoder.set_graph(gw.to(device))
        all_emb = explainer.pret_encoder.get_all_emb(xw)
        all_emb = [xw, *all_emb]

    metrics = {}

    fidelity_all = [[], []]
    for g, label in test_loader:
        torch.cuda.empty_cache()
        g = g.to(device)
        label = label.to(device)
        x = all_emb[0][g.ndata['_ID']]
        with torch.no_grad():
            explanation = explainer(g, all_emb, label, training=False, explain=True)
            explanation = explanation.sigmoid()
        fidelity_pos, fidelity_neg = fidelity(
                gnn_model, g, x, explanation, label)
        fidelity_all[0].extend(fidelity_pos.detach().cpu().numpy())
        fidelity_all[1].extend(fidelity_neg.detach().cpu().numpy())
    metrics['fidelity_pos'] = np.mean(fidelity_all[0])
    metrics['fidelity_neg'] = np.mean(fidelity_all[1])

    for k, v in metrics.items():
        print(f"{k}: {np.mean(v)}")

    return metrics
