import dgl
import torch
import os
import shutil
import dgl.data as data
from sklearn.model_selection import train_test_split


def get_dataset(data_name, dir):
    if data_name == 'cora':
        dataset = data.CoraGraphDataset(raw_dir=dir)
    elif data_name == 'citeseer':
        dataset = data.CiteseerGraphDataset(raw_dir=dir)
    elif data_name == 'pubmed':
        dataset = data.PubmedGraphDataset(raw_dir=dir)
    elif data_name == 'amazon-photo':
        dataset = data.AmazonCoBuyPhotoDataset(raw_dir=dir)
    elif data_name == 'coauthor-physics':
        dataset = data.CoauthorPhysicsDataset(raw_dir=dir)
    elif data_name == 'coauthor-cs':
        dataset = data.CoauthorCSDataset(raw_dir=dir)
    else:
        raise NotImplementedError

    for _, _, filenames in os.walk(dataset.save_path):
        filenames[:] = [f for f in filenames if f.endswith(".bin")]
        os.rename(f'{dataset.save_path}/{filenames[0]}',
                    f'{dataset.save_path}/{data_name}_graph.bin')
    shutil.move(f'{dataset.save_path}/{data_name}_graph.bin', dir)
    shutil.rmtree(dataset.save_path)
    for _, _, filenames in os.walk(dir):
        zip_files = [f for f in filenames if f.endswith(".zip")]
        os.remove(os.path.join(dir, zip_files[0]))

    g = dgl.load_graphs(f'{dir}/{data_name}_graph.bin')[0][0]
    info = {'label': g.ndata['label']}
    if 'train_mask' not in g.ndata:
        train_index, test_index = train_test_split(
            torch.arange(0, len(info['label'])), train_size=600, test_size=1000, random_state=2023)
        info["train_index"] = train_index[:100]
        info["valid_index"] = train_index[100:600]
        info["test_index"] = test_index
    else:
        info['train_index'] = torch.nonzero(g.ndata['train_mask']).view(-1)
        info['valid_index'] = torch.nonzero(g.ndata['val_mask']).view(-1)
        info['test_index'] = torch.nonzero(g.ndata['test_mask']).view(-1)
        del g.ndata['train_mask'], g.ndata['test_mask'], g.ndata['val_mask']
    torch.save(info, os.path.join(dir, data_name + '_index.bin'))

    g.ndata['nfeat'] = g.ndata['feat']
    del g.ndata['feat'], g.ndata['label']

    dgl.save_graphs(f'{dir}/{data_name}_graph.bin', g)

    return dataset