
import copy
from train.utils import *
from dgl.dataloading import GraphDataLoader


def train(cfg):
    target_ntype = cfg.hyparams['data']['target_ntype']
    lr = cfg.hyparams[cfg.method]['learning_rate']
    n_epochs = cfg.hyparams[cfg.method]['n_epochs']
    batch_size = cfg.hyparams[cfg.method]['batch_size']

    debug_epoch = 5
    best_acc = 0.

    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    gw, model, info = get_model(cfg)
    xw = gw.ndata.pop('nfeat')
    model.to(device)

    train_set, valid_set, test_set, _ = construct_dataset(gw, info, cfg)
    train_loader = GraphDataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = GraphDataLoader(valid_set, batch_size=batch_size, shuffle=False)
    test_loader = GraphDataLoader(test_set, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-5)

    for epoch in range(n_epochs):
        train_pred_loss = 0.
        preds, labels = torch.tensor([]), torch.tensor([])

        model.train()
        for g, label in train_loader:
            torch.cuda.empty_cache()
            g = g.to(device)
            label = label.to(device)
            model.set_graph(g)
            x = {}
            for k in g.ndata['_ID']:
                x[k] = xw[k][g.ndata['_ID'][k]].to(device)
            offset = torch.cat([torch.tensor([0], device=device),
                                g.batch_num_nodes(target_ntype).cumsum(dim=0)[:-1]])
            pred = model(model.get_emb(x), target_ntype, offset)
            loss = torch.nn.functional.cross_entropy(pred, label)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()

            train_pred_loss += loss.item()

            preds = torch.cat([preds, pred.detach().cpu()])
            labels = torch.cat([labels, label.detach().cpu()])

        train_acc = accuracy(preds, labels)

        model.eval()
        preds, labels = torch.tensor([]), torch.tensor([])
        for g, label in valid_loader:
            torch.cuda.empty_cache()
            g = g.to(device)
            label = label.to(device)
            model.set_graph(g)
            x = {}
            for k in g.ndata['_ID']:
                x[k] = xw[k][g.ndata['_ID'][k]].to(device)
            offset = torch.cat([torch.tensor([0], device=device),
                                g.batch_num_nodes(target_ntype).cumsum(dim=0)[:-1]])
            pred = model(model.get_emb(x), target_ntype, offset)

            preds = torch.cat([preds, pred.detach().cpu()])
            labels = torch.cat([labels, label.detach().cpu()])

        valid_acc = accuracy(preds, labels)
        if epoch % debug_epoch == 0:
            print(f"Epoch: {epoch}\t Train pred Loss: {train_pred_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, Validation Acc: {valid_acc:.4f}, ")

        if valid_acc > best_acc:
            best_acc = valid_acc
            best_model = copy.deepcopy(model)

    model = copy.deepcopy(best_model)
    model.eval()
    preds, labels = torch.tensor([]), torch.tensor([])
    for g, label in test_loader:
        torch.cuda.empty_cache()
        g = g.to(device)
        label = label.to(device)
        model.set_graph(g)
        x = {}
        for k in g.ndata['_ID']:
            x[k] = xw[k][g.ndata['_ID'][k]].to(device)
        offset = torch.cat([torch.tensor([0], device=device),
                            g.batch_num_nodes(target_ntype).cumsum(dim=0)[:-1]])
        with torch.no_grad():
            pred = model(model.get_emb(x), target_ntype, offset)
        preds = torch.cat([preds, pred.detach().cpu()])
        labels = torch.cat([labels, label.detach().cpu()])

    test_acc = accuracy(preds, labels)
    print(f"Test Acc: {test_acc:.4f}")
    if not os.path.exists(f'{cfg.ckpt_dir}/{cfg.dataset}'):
        os.makedirs(f'{cfg.ckpt_dir}/{cfg.dataset}')
    torch.save(model.state_dict(), f'{cfg.ckpt_dir}/{cfg.dataset}/{cfg.method}-seed-{cfg.seed}-pretrain.pt')
    metrics = {'test_acc': test_acc}
    return metrics