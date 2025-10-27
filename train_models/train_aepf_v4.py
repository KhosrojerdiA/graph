


import sys
import os

te_path = '/mnt/data/khosro/Graph_v2'
sys.path.append(te_path)


from EPAGCL.epagcl import * 
from retrieval.retrieval_epaglc_v4 import * 
from utils.utils import *
from utils.utils_scorer import *
import copy
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.datasets import Planetoid

#________________________________________________________________________________________________________________________

def train_edge_scorer_v3_from_simplified_performance(
    data,
    simplified_performance,
    selected_nodes,
    dataset_embeddings,
    *,
    epochs=250,
    lr=1e-3,
    hidden=128,
    dropout=0.5,
    rank_weight=1.0,          # 1.0 = rank-only (recommended)
    mse_weight=0.0,           # 0.0 = disable regression
    max_edges_per_node=80,
    val_ratio=0.2,
    batch_size=4096,
    weight_decay=3e-4,
    patience=20,
    b_list=(1,2,3,4,5),
    feature_fn=None,
    early_key="mean_nDCG",    # early stop on mean across b_list
    verbose=True,
    seed=42,
):
    assert feature_fn is not None, "Pass your feature function."
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    # ----- Split nodes (exclude attack nodes completely) -----
    all_nodes = set(simplified_performance.keys())
    selected = set(selected_nodes.tolist() if isinstance(selected_nodes, torch.Tensor) else selected_nodes)
    non_attack_nodes = list(all_nodes - selected)
    random.shuffle(non_attack_nodes)
    val_count = max(1, int(len(non_attack_nodes) * val_ratio))
    val_nodes = non_attack_nodes[:val_count]
    train_nodes = non_attack_nodes[val_count:]
    if verbose:
        print(f"[Split] train_nodes={len(train_nodes)}, val_nodes={len(val_nodes)}, excluded(selected)={len(selected)}")

    # ----- Build DFs -----
    train_df = build_df_for_nodes(simplified_performance, train_nodes)
    val_df   = build_df_for_nodes(simplified_performance, val_nodes)
    train_df = zscore_per_node(train_df)
    val_df   = zscore_per_node(val_df)
    train_df = cap_per_node(train_df, max_edges_per_node)

    # ----- Features / tensors -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_dim, target_aware = infer_in_dim(feature_fn, dataset_embeddings, data)
    model = EdgeScorer_v2(in_dim, hidden=hidden, dropout=dropout).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    Xtr, ytr, ntr = tensorize_df(train_df, dataset_embeddings, data, feature_fn, in_dim, device, target_aware)
    Xv,  yv,  nv  = tensorize_df(val_df,   dataset_embeddings, data, feature_fn, in_dim, device, target_aware)

    # ----- Train -----
    best_ckpt, best_key, stale = None, -1.0, 0
    idxs = torch.arange(Xtr.size(0), device=device)
    for epoch in range(1, epochs+1):
        model.train()
        perm = idxs[torch.randperm(idxs.numel())]
        total_loss = total_mse = total_rank = 0.0
        steps = 0
        for start in range(0, perm.numel(), batch_size):
            b = perm[start:start+batch_size]
            xb, yb, nb = Xtr[b], ytr[b], ntr[b]
            opt.zero_grad()
            pred = model(xb)
            rank = per_node_logistic_rank_loss(pred, yb, nb, max_pairs=8192)
            mse  = F.mse_loss(pred, yb) if mse_weight > 0 else pred.new_tensor(0.0)
            loss = rank_weight * rank + mse_weight * mse
            loss.backward()
            opt.step()
            total_loss += float(loss.item()); total_mse += float(mse.item()); total_rank += float(rank.item()); steps += 1

        sched.step()

        # ----- Validation nDCG for budgets -----
        model.eval()
        with torch.no_grad():
            pv = model(Xv).squeeze(1).detach().cpu().numpy()
        val_pred = val_df.copy()
        val_pred["pred"] = pv
        metrics = ndcg_by_node(val_pred, b_list=b_list, rel_col="delta")
        mean_ndcg = float(np.mean([metrics[f"nDCG@{b}"] for b in b_list]))
        if early_key == "mean_nDCG":
            current_key = mean_ndcg
        else:
            # e.g. early_key="nDCG@1"
            current_key = metrics.get(early_key, mean_ndcg)

        if verbose:
            meter = " ".join([f"{k}={metrics[k]:.4f}" for k in sorted(metrics)])
            print(f"[Epoch {epoch:03d}/{epochs}] Loss={total_loss/max(steps,1):.4f} | "
                  f"MSE={total_mse/max(steps,1):.4f} | Rank={total_rank/max(steps,1):.4f} | "
                  f"{meter} | Best({early_key})={best_key if best_key>0 else 0:.4f} | stale={stale}")

        if current_key > best_key:
            best_key = current_key
            best_ckpt = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} (patience={patience}).")
                break

    # restore best
    if best_ckpt is not None:
        model.load_state_dict(best_ckpt)
    if verbose:
        print("✅ Training complete (best checkpoint restored).")
        print(f"Best {early_key}: {best_key:.4f}")

    return model


#____________________________________________________________ Inputs ____________________________________________________________

data_name = 'Cora'
#['Cora', 'CiteSeer', 'PubMed'] 


graph_model = 'greet'
#['gcn', 'sage', 'graphpatcher', 'gat2', 'epagcl_gcn', 'epagcl_sage', 'grace', 'gca', 'greet']

 

embedding_version = 'v1'

embeddings_save_dir = f"{te_path}/{graph_model}/embeddings"

saving_path = f'{te_path}/trained_scorer/{data_name}_{graph_model}_scorer_model_v1.pt'


#____________________________________________________________ Inputs ____________________________________________________________

ep_path = f"{te_path}/edge_performance_dataset"
dataset_subgraph_path = f"{te_path}/data/pubmed_subgraph.pt"



min_number_edges = 10
#Minimum number of edges connected to selected nodes (query) [5,10]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#____________________________________________________________ Data ___________________________________________________________

data = load_data(data_name, dataset_subgraph_path)              
#____________________________________________________________ Loop ____________________________________________________________

                                               
main_seed = 3708

#Retrieval

(
dataset_embeddings, model, selected_nodes, selected_node_embeddings, 
top_k_indice_at_20, top_k_indice_at_100, top_k_indice_at_500, top_k_indice_at_1000,top_k_indice_at_4000, 
found_count_20, found_count_100, found_count_500, found_count_1000, found_count_4000, 
recall_20, recall_100, recall_500, recall_1000, recall_4000, 
avg_position_20, avg_position_100, avg_position_500, avg_position_1000, avg_position_4000 

) = retrieval_v4(data, data_name, graph_model, min_number_edges, embeddings_save_dir, embedding_version, main_seed)



#________________________________________________________________________________________________________________________

simplified_performance = load_edge_performance_change_position(graph_model, data_name, ep_path)

#________________________________________________________________________________________________________________________

# Train

# --- inputs already prepared in your script ---
# data, simplified_performance, selected_nodes, dataset_embeddings
# edge feature function:
#   def edge_features_v4(graph_embeddings, edge, data): ...

scorer_model = train_edge_scorer_v3_from_simplified_performance(
    data=data,
    simplified_performance=simplified_performance,
    selected_nodes=selected_nodes,              # all excluded (no leak)
    dataset_embeddings=dataset_embeddings,
    epochs=250,
    lr=1e-3,
    hidden=64,                                  # often better on Cora/Citeseer
    dropout=0.5,
    rank_weight=1.0,                            # rank-only
    mse_weight=0.0,
    max_edges_per_node=80,
    val_ratio=0.2,
    batch_size=4096,
    weight_decay=3e-4,
    patience=20,
    b_list=(1,2,3,4,5),                         # <-- your budgets
    feature_fn=edge_features_v5_targeted,       # or your legacy edge_features_v4
    early_key="mean_nDCG",                      # mean over budgets
    verbose=True,
)

# Save your best checkpoint
torch.save(scorer_model.state_dict(), saving_path)
#________________________________________________________________________________________________________________________

