import os
import torch
import dgl
from tqdm import tqdm
from models import sunnygnn, gat, gcn


def edge_hop_mask(sg, k=2):
    src_target = 0
    e_h_mask = torch.tensor([], dtype=torch.bool)
    src = [[src_target]]
    for i in range(k):
        one_hop_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        one_hop_loader = dgl.dataloading.DataLoader(sg, src[i],
                                                   one_hop_sampler, batch_size=1, shuffle=False)
        neighbors = []
        h_mask = torch.zeros(sg.number_of_edges(), dtype=torch.bool)
        for j, (ng, _, _) in enumerate(one_hop_loader):
            ng_lst = ng.numpy().tolist()
            neighbors.extend(ng_lst)
            edge_ids = sg.edge_ids(ng, [src[i][j]]*len(ng))
            h_mask[edge_ids] = 1
        src.append(list(set(neighbors)))
        e_h_mask = torch.cat((e_h_mask, h_mask.unsqueeze(0)), dim=0)

    return e_h_mask.T


def accuracy(y_pred, y_true):
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    correct = preds.eq(y_true).double()
    correct = correct.sum().item()
    return correct / len(y_true)


def get_model(cfg):
    graph_path = cfg.graph_path
    index_path = cfg.index_path
    method = cfg.method
    data_hyparams = cfg.hyparams['data']

    dataset = cfg.dataset
    ckpt_dir = cfg.ckpt_dir
    encoder_type = cfg.encoder_type
    num_classes = data_hyparams['num_classes']

    gs, _ = dgl.load_graphs(graph_path)
    g = gs[0]
    if g.is_homogeneous:
        g = dgl.add_self_loop(g)
    in_dim = g.ndata['nfeat'].shape[1]
    info = torch.load(index_path)

    if method == 'gat':
        model = gat.GAT(in_dim, 256, 64, [8, 1], num_classes)

    elif method == 'gcn':
        model = gcn.GCN(in_dim, 256, 64, num_classes)

    elif method == 'sunny-gnn':
        method_cfg = cfg.hyparams[method][cfg.encoder_type]
        if encoder_type == 'gat':
            pret_encoder = gat.GAT(in_dim, 256, 64, [8, 1], num_classes)
            encoder = gat.GAT(in_dim, 256, 64, [8, 1], num_classes)
        elif encoder_type == 'gcn':
            pret_encoder = gcn.GCN(in_dim, 256, 64, num_classes)
            encoder = gcn.GCN(in_dim, 256, 64, num_classes)

        pret_encoder.load_state_dict(torch.load(f'{ckpt_dir}/{dataset}/{encoder_type}-seed-{cfg.seed}-pretrain.pt'))
        # for param in pret_encoder.parameters():
        #     param.requires_grad = False

        if cfg.eval_explanation:
            encoder.load_state_dict(torch.load(f'{ckpt_dir}/{dataset}/{encoder_type}-seed-{cfg.seed}-pretrain.pt'))
            for param in encoder.parameters():
                param.requires_grad = False

        extractor = sunnygnn.ExtractorMLP(96, False)
        model = sunnygnn.SunnyGNN(pret_encoder, encoder, extractor, in_dim, dropout=method_cfg['dropout'])
        model.set_config(cfg.hyparams[method][cfg.encoder_type])
    else:
        raise NotImplementedError

    return g, model, info


def construct_dataset(g, info, cfg):
    if cfg.index is not None:
        train_node = info["train_index"].long()[: cfg.index]
    else:
        train_node = info["train_index"].long()
    valid_node = info["valid_index"].long()
    test_node = info["test_index"].long()
    labels = info['label'].type(torch.int64)
    nodes = torch.arange(g.number_of_nodes())

    dataset = []
    k = 2
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(k)
    sg_loader = dgl.dataloading.DataLoader(g, nodes, sampler, batch_size=1, shuffle=False)
    print('loading dataset...')
    sg_path = f'{cfg.data_dir}/{cfg.dataset}_sg.bin'
    if os.path.exists(sg_path):
        dataset = dgl.load_graphs(sg_path)[0]
    else:
        i = 0
        for ng, _, _ in tqdm(sg_loader, leave=False):
            sg = dgl.node_subgraph(g, ng)
            sg.edata['e_h_mask'] = edge_hop_mask(sg, k=2)
            dataset.append(sg)
            i += 1
        dgl.save_graphs(sg_path, dataset)

    dataset = [[dataset[i], labels[i]] for i in range(len(dataset))]
    print('load dataset done!')
    train_set = torch.utils.data.Subset(dataset, train_node)
    valid_set = torch.utils.data.Subset(dataset, valid_node)
    test_set = torch.utils.data.Subset(dataset, test_node)
    return train_set, valid_set, test_set, dataset
