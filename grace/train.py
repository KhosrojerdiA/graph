

import sys
import os

prn_project_path = '/mnt/data/khosro/Graph_v2'
sys.path.append(prn_project_path)

import argparse
import os.path as osp
import random
from time import perf_counter as t
import yaml
from yaml import SafeLoader

import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GCNConv
import yaml
from grace.model import Encoder, Model, drop_feature
from grace.eval import label_classification

#____________________________________________________________________________________________________________________________

def train(model: Model, x, edge_index):
    model.train()
    optimizer.zero_grad()
    edge_index_1 = dropout_adj(edge_index, p=drop_edge_rate_1)[0]
    edge_index_2 = dropout_adj(edge_index, p=drop_edge_rate_2)[0]
    x_1 = drop_feature(x, drop_feature_rate_1)
    x_2 = drop_feature(x, drop_feature_rate_2)
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    loss = model.loss(z1, z2, batch_size=0)
    loss.backward()
    optimizer.step()

    return loss.item()


def test(model: Model, x, edge_index, y, final=False):
    model.eval()
    z = model(x, edge_index)

    label_classification(z, y, ratio=0.1)


def get_dataset(path, name):
    assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP']
    name = 'dblp' if name == 'DBLP' else name

    if name == 'dblp':
        return CitationFull(path, name, transform=T.NormalizeFeatures())
    else:
        return Planetoid(path, name, transform=T.NormalizeFeatures())



def load_trained_model(model_class, encoder, num_hidden, num_proj_hidden, tau, device, load_path):
    """
    Load a trained GRACE model from disk.

    Args:
        model_class: the GRACE model class (e.g., Model)
        encoder: the encoder architecture (must match training setup)
        num_hidden: hidden dimension size
        num_proj_hidden: projection hidden dimension
        tau: temperature parameter
        device: torch.device ('cuda' or 'cpu')
        load_path: path to saved .pt model file

    Returns:
        model (nn.Module): loaded and ready for inference
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Model file not found at: {load_path}")

    print(f"Loading model from {load_path} ...")
    model = model_class(encoder, num_hidden, num_proj_hidden, tau).to(device)
    model.load_state_dict(torch.load(load_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")
    return model

#____________________________________________________________________________________________________________________________

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = "Cora"  # "Cora", "CiteSeer", "PubMed", "DBLP"


    config_path = "/mnt/data/khosro/Graph_v2/grace/config.yaml"
    # Load full YAML and then select dataset-specific section
    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)

    # Extract only the section for the chosen dataset
    if dataset not in full_config:
        raise KeyError(f"Dataset '{dataset}' not found in {config_path}")
    config = full_config[dataset]
    print(f"Loaded config for dataset '{dataset}':")
    print(config)


    torch.manual_seed(config['seed'])
    random.seed(12345)

    learning_rate = config['learning_rate']
    num_hidden = config['num_hidden']
    num_proj_hidden = config['num_proj_hidden']
    activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[config['activation']]
    base_model = ({'GCNConv': GCNConv})[config['base_model']]
    num_layers = config['num_layers']

    drop_edge_rate_1 = config['drop_edge_rate_1']
    drop_edge_rate_2 = config['drop_edge_rate_2']
    drop_feature_rate_1 = config['drop_feature_rate_1']
    drop_feature_rate_2 = config['drop_feature_rate_2']
    tau = config['tau']
    num_epochs = config['num_epochs']
    weight_decay = config['weight_decay']

    path = osp.join(osp.expanduser('~'), 'datasets', dataset)
    dataset = get_dataset(path, dataset)
    data = dataset[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    encoder = Encoder(dataset.num_features, num_hidden, activation,
                      base_model=base_model, k=num_layers).to(device)
    model = Model(encoder, num_hidden, num_proj_hidden, tau).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    start = t()
    prev = start
    for epoch in range(1, num_epochs + 1):
        loss = train(model, data.x, data.edge_index)

        now = t()
        print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, '
              f'this epoch {now - prev:.4f}, total {now - start:.4f}')
        prev = now

    print("=== Final ===")
    test(model, data.x, data.edge_index, data.y, final=True)

#___________________________________________________________________________________________

    # ---------------- Save trained model and embeddings ----------------
    save_dir = "/mnt/data/khosro/Graph_v2/grace/embeddings"
    os.makedirs(save_dir, exist_ok=True)

    file_prefix = f"{dataset.name}_grace"
    model_path = os.path.join(save_dir, f"{file_prefix}_model.pt")
    emb_path = os.path.join(save_dir, f"{file_prefix}_embeddings.pt")

    # Save the full model (architecture + parameters)
    torch.save(model, model_path)

    # Save learned embeddings for downstream use
    with torch.no_grad():
        z = model(data.x, data.edge_index)
    torch.save(z, emb_path)

    print(f"✅ Full GRACE model saved at: {model_path}")
    print(f"✅ Embeddings saved at: {emb_path}")

    # ---------------- Load model again (for verification) ----------------
    model_path = os.path.join(save_dir, f"{dataset.name}_grace_model.pt")
    emb_path = os.path.join(save_dir, f"{dataset.name}_grace_embeddings.pt")

    # Directly load full model (no need to rebuild)
    model = torch.load(model_path, map_location=device)
    model.eval()

    # Load embeddings
    dataset_embeddings = torch.load(emb_path, map_location="cpu")

    print(f"✅ GRACE model and embeddings for {dataset.name} successfully loaded.")


