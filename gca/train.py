import sys
import os

prn_project_path = '/mnt/data/khosro/Graph_v2'
sys.path.append(prn_project_path)

import yaml
import numpy
import argparse
import os.path as osp
import random

from time import time as t
import torch
from torch_geometric.utils import dropout_adj, degree, to_undirected
from gca.dataset import get_dataset
from gca.eval import log_regression, MulticlassEvaluator
from gca.model import Encoder, GRACE
from gca.utils import get_base_model, get_activation, \
    generate_split, compute_pr, eigenvector_centrality
from gca.functional import drop_feature, drop_edge_weighted, \
    degree_drop_weights, \
    evc_drop_weights, pr_drop_weights, \
    feature_drop_weights, drop_feature_weighted_2, feature_drop_weights_dense


# ====================================================================================================
# Training and evaluation
# ====================================================================================================

def train(model, data, optimizer, param, feature_weights, drop_weights):
    model.train()
    optimizer.zero_grad()

    drop_scheme = param.get('drop_scheme', 'degree')  # default to degree

    def drop_edge(idx: int):
        if drop_scheme == 'uniform':
            return dropout_adj(data.edge_index, p=param[f'drop_edge_rate_{idx}'])[0]
        elif drop_scheme in ['degree', 'evc', 'pr']:
            return drop_edge_weighted(
                data.edge_index, drop_weights,
                p=param[f'drop_edge_rate_{idx}'], threshold=0.7
            )
        else:
            raise ValueError(f"Undefined drop scheme: {drop_scheme}")

    edge_index_1 = drop_edge(1)
    edge_index_2 = drop_edge(2)

    if drop_scheme in ['pr', 'degree', 'evc']:
        x_1 = drop_feature_weighted_2(data.x, feature_weights, param['drop_feature_rate_1'])
        x_2 = drop_feature_weighted_2(data.x, feature_weights, param['drop_feature_rate_2'])
    else:
        x_1 = drop_feature(data.x, param['drop_feature_rate_1'])
        x_2 = drop_feature(data.x, param['drop_feature_rate_2'])

    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    loss = model.loss(z1, z2)
    loss.backward()
    optimizer.step()

    return loss.item()


def test(model, data, dataset, split, dataset_name, final=False):
    model.eval()
    z = model(data.x, data.edge_index)
    evaluator = MulticlassEvaluator()

    if dataset_name == 'WikiCS':
        accs = [
            log_regression(z, dataset, evaluator, split=f'wikics:{i}', num_epochs=800)['acc']
            for i in range(20)
        ]
        acc = sum(accs) / len(accs)
    else:
        acc = log_regression(
            z, dataset, evaluator, split='rand:0.1', num_epochs=3000, preload_split=split
        )['acc']

    if final:
        print(f"Final Accuracy: {acc:.4f}")
    return acc


# ====================================================================================================
# Main
# ====================================================================================================

if __name__ == '__main__':

    dataset_name = "Cora"  # "Cora", "CiteSeer", "PubMed", "DBLP"

    # ---------------- Load Config ----------------
    config_path = "/mnt/data/khosro/Graph_v2/gca/config.yaml"
    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)

    if dataset_name not in full_config:
        raise KeyError(f"Dataset '{dataset_name}' not found in {config_path}")

    param = full_config[dataset_name]
    print(f"Loaded config for dataset '{dataset_name}':")
    print(param)

    # ---------------- Device ----------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---------------- Dataset ----------------
    path = osp.join(osp.expanduser('~'), 'datasets', dataset_name)
    dataset = get_dataset(path, dataset_name)
    data = dataset[0].to(device)

    # ---------------- Split ----------------
    split = generate_split(data.num_nodes, train_ratio=0.1, val_ratio=0.1)

    # ---------------- Model & Optimizer ----------------
    encoder = Encoder(
        dataset.num_features,
        param['num_hidden'],
        get_activation(param['activation']),
        k=2  # fixed number of layers (removed base_model)
    ).to(device)

    model = GRACE(encoder, param['num_hidden'], param['num_proj_hidden'], param['tau']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=param['learning_rate'])

    # ---------------- Drop Weights ----------------
    drop_scheme = param.get('drop_scheme', 'degree')

    if drop_scheme == 'degree':
        drop_weights = degree_drop_weights(data.edge_index).to(device)
        node_metric = degree(to_undirected(data.edge_index)[1])
    elif drop_scheme == 'pr':
        drop_weights = pr_drop_weights(data.edge_index, aggr='sink', k=200).to(device)
        node_metric = compute_pr(data.edge_index)
    elif drop_scheme == 'evc':
        drop_weights = evc_drop_weights(data).to(device)
        node_metric = eigenvector_centrality(data)
    else:
        drop_weights = None
        node_metric = torch.ones(data.num_nodes)

    node_metric = node_metric.to(device)
    feature_weights = (
        feature_drop_weights_dense(data.x, node_c=node_metric).to(device)
        if dataset_name == 'WikiCS'
        else feature_drop_weights(data.x, node_c=node_metric).to(device)
    )

    # ---------------- Training ----------------
    start = t()
    prev = start
    for epoch in range(1, param['num_epochs'] + 1):
        loss = train(model, data, optimizer, param, feature_weights, drop_weights)
        now = t()
        print(f"(T) | Epoch={epoch:03d}, loss={loss:.4f}, "
              f"this epoch {now - prev:.4f}, total {now - start:.4f}")
        prev = now

        if epoch % 100 == 0:
            acc = test(model, data, dataset, split, dataset_name)
            print(f"(E) | Epoch={epoch:03d}, Accuracy={acc:.4f}")

    print("=== Final ===")
    final_acc = test(model, data, dataset, split, dataset_name, final=True)

    # ---------------- Save Model and Embeddings ----------------
    save_dir = "/mnt/data/khosro/Graph_v2/gca/embeddings"
    os.makedirs(save_dir, exist_ok=True)

    file_prefix = f"{dataset_name}_gca"
    model_path = os.path.join(save_dir, f"{file_prefix}_model.pt")
    emb_path = os.path.join(save_dir, f"{file_prefix}_embeddings.pt")

    # Save the full model (includes architecture + parameters)
    torch.save(model, model_path)

    # Save learned embeddings for downstream use
    with torch.no_grad():
        z = model(data.x, data.edge_index)
    torch.save(z, emb_path)

    print(f"✅ Full model (architecture + weights) saved at: {model_path}")
    print(f"✅ Embeddings saved at: {emb_path}")

    # ---------------- Verification Load ----------------
    ''' 
    encoder = Encoder(
        dataset.num_features,
        param['num_hidden'],
        get_activation(param['activation']),
        k=2
    ).to(device)

    model = GRACE(
        encoder,
        param['num_hidden'],
        param['num_proj_hidden'],
        param['tau']
    ).to(device)
    '''
    
    model_path = os.path.join(save_dir, f"{dataset_name}_gca_model.pt")
    emb_path = os.path.join(save_dir, f"{dataset_name}_gca_embeddings.pt")

    # Load model directly (no need to rebuild architecture)
    model = torch.load(model_path, map_location=device)
    model.eval()

    # Load embeddings
    dataset_embeddings = torch.load(emb_path, map_location="cpu")

    print(f"✅ Model and embeddings for {dataset_name} successfully loaded.")


