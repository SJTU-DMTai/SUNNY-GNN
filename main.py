import argparse
import yaml
import os
import torch
import random
import copy
import dgl
import numpy as np
from train import train_baseline, train_gnn


def parse_args():
    parser = argparse.ArgumentParser(description='Self-explainable GNN')
    parser.add_argument('--method', type=str, default='sunny-gnn', help='self-explainable GNN type',
                        choices=['sunny-gnn', 'gat', 'gcn'])
    parser.add_argument('--encoder', type=str, default='gat', help='GNN encoder type',
                        choices=['gat', 'gcn'])
    parser.add_argument('--dataset', type=str, default='cora', help='dataset name',
                        choices=['citeseer', 'cora', 'pubmed',
                                 'amazon-photo', 'coauthor-physics', 'coauthor-cs'])
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--num_seeds', type=int, default=1, help='number of random seeds')
    parser.add_argument('--eval_explanation', type=bool, default=False,
                            help='whether to evaluate explanation fidelity')
    return parser.parse_args()


class Config(object):
    def __init__(self, args):
        abs_dir = os.path.dirname(os.path.realpath(__file__))
        log_dir = os.path.join(abs_dir, 'log')
        os.makedirs(log_dir, exist_ok=True)
        data_dir = os.path.join(abs_dir, 'dataset', args.dataset)
        self.method = args.method
        self.encoder_type = args.encoder
        self.dataset = args.dataset
        self.abs_dir = abs_dir
        self.data_dir = data_dir
        self.gpu = args.gpu
        self.index = None
        self.graph_path = f'{data_dir}/{args.dataset}_graph.bin'
        self.index_path = f'{data_dir}/{args.dataset}_index.bin'
        self.check_dataset()
        self.ckpt_dir = os.path.join(abs_dir, 'ckpt')
        self.hyparams = self.load_hyperparams(args)
        self.eval_explanation = args.eval_explanation

    def check_dataset(self):
        if not os.path.exists(self.graph_path):
            from tools.get_data import get_dataset
            get_dataset(self.dataset, self.data_dir)

    def load_hyperparams(self, args):
        yml_path = os.path.join(self.abs_dir, 'configs', f'{args.dataset}.yml')
        with open(yml_path, 'r') as f:
            hyperparams = yaml.load(f, Loader=yaml.FullLoader)
            return hyperparams

    def set_seed(self, seed):
        self.seed = seed
        self.encoder_path = f'{self.ckpt_dir}/{self.dataset}/{self.encoder_type}-seed-{seed}-pretrain.pt'


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.seed(seed)


def main():
    results = {}

    for seed in range(args.num_seeds):
        setup_seed(seed)

        cfg.set_seed(seed)
        print(f'===========seed: {seed}===========')

        if cfg.method == 'sunny-gnn':
            print(f"Dataset: {cfg.dataset}, Method: {cfg.method}-{cfg.encoder_type}")

            if not os.path.exists(cfg.encoder_path):
                print(f"Pretrain {cfg.encoder_type}...")
                cfg_cp = copy.deepcopy(cfg)
                cfg_cp.method = cfg_cp.encoder_type
                train_gnn.train(cfg_cp)

            if cfg.eval_explanation:
                metrics = train_baseline.train_explain(cfg)
            else:
                print(f"Train {cfg.method}...")
                metrics = train_baseline.train(cfg)
        elif cfg.method in ['gat', 'gcn']:
            print(f"Dataset: {cfg.dataset}, Method: {cfg.method}")
            metrics = train_gnn.train(cfg)
        else:
            raise NotImplementedError

        if results == {}:
            for k, v in metrics.items():
                results[k] = [v]
        else:
            for k, v in metrics.items():
                results[k].append(v)

    print(f'===========results===========')
    for k, v in results.items():
        results[k] = [v, sum(v) / len(v), np.std(v)]
        print(f'>>> {k}: {sum(v) / len(v)}, {np.std(v)}')
    return results


if __name__ == '__main__':
    args = parse_args()
    cfg = Config(args)
    main()
