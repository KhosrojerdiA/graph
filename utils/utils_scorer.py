
import sys
import os


project_path = './'
sys.path.append(project_path)

from utils.utils import gold_remove_edges
import torch
import numpy as np
import torch.nn as nn
import pandas as pd
from torch_geometric.utils import degree, remove_self_loops

#____________________________________________________________________________________________________________________________

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
#____________________________________________________________________________________________________________________________

''' 
def edge_features_v4(graph_embeddings, edge, data):
    """
    Feature = (u - v)
    """
    u, v = int(edge[0]), int(edge[1])
    emb_u = graph_embeddings[u]
    emb_v = graph_embeddings[v]

    # Ensure numpy arrays
    if isinstance(emb_u, torch.Tensor):
        emb_u = emb_u.detach().cpu().numpy()
        emb_v = emb_v.detach().cpu().numpy()

    # Only difference vector
    feat = emb_u - emb_v

    return feat.astype(np.float32)
'''
#____________________________________________________________________________________________________________________________

def _neighbors_list(data):
    ei = data.edge_index
    N = data.num_nodes
    adj = [set() for _ in range(N)]
    for u, v in ei.t().tolist():
        adj[u].add(v)
        adj[v].add(u)
    return adj
#____________________________________________________________________________________________________________________________

def _jaccard(adj, a, b):
    A, B = adj[a], adj[b]
    if not A or not B:
        return 0.0
    inter = len(A & B)
    uni   = len(A | B)
    return float(inter) / float(uni)
#____________________________________________________________________________________________________________________________

def edge_features_v5_targeted(graph_embeddings, edge, node, data, cached=None):
    """
    Target-aware features for (u,v) w.r.t. 'node'.
    Self-initializes cache keys: 'deg' and 'adj'.
    """
    # ---------- cache ----------
    if cached is None:
        cached = {}
    if ("deg" not in cached) or ("adj" not in cached):
        # fill once
        cached["deg"] = degree(data.edge_index[0], num_nodes=data.num_nodes).cpu().numpy()
        cached["adj"] = _neighbors_list(data)

    deg_vec = cached["deg"]
    adj = cached["adj"]

    # ---------- orient ----------
    u, v = int(edge[0]), int(edge[1])
    t = int(node)
    if t == u:
        n = v
        is_u_target, is_v_target = 1.0, 0.0
    elif t == v:
        n = u
        is_u_target, is_v_target = 0.0, 1.0
    else:
        # not incident: still produce a deterministic vector
        n = v
        is_u_target, is_v_target = 1.0, 0.0

    emb_t = graph_embeddings[t]
    emb_n = graph_embeddings[n]
    if isinstance(emb_t, torch.Tensor):
        emb_t = emb_t.detach().cpu().numpy()
        emb_n = emb_n.detach().cpu().numpy()

    deg_t = float(deg_vec[t])
    deg_n = float(deg_vec[n])
    mean_deg = (deg_t + deg_n) / 2.0

    denom = (np.linalg.norm(emb_t) * np.linalg.norm(emb_n) + 1e-8)
    cos_tn = float(np.dot(emb_t, emb_n) / denom)
    jac_tn = _jaccard(adj, t, n)

    diff = emb_t - emb_n
    l1   = np.abs(diff)
    had  = emb_t * emb_n
    avgg = (emb_t - emb_n)/2

    extra = np.array([deg_t, deg_n, mean_deg, cos_tn, jac_tn,
                      is_u_target, is_v_target], dtype=np.float32)

    return np.concatenate([emb_t, emb_n, diff, l1, had, extra]).astype(np.float32)

#original: 
#ex8: np.concatenate([emb_t, emb_n, diff, l1, had, extra]).astype(np.float32)

#ex_8_v1: np.concatenate([diff]).astype(np.float32) check 
#ex_8_v2: np.concatenate([emb_t, emb_n]).astype(np.float32) check 
#ex_8_v3: np.concatenate([had]).astype(np.float32)  check
#ex_8_v4: np.concatenate([avgg]).astype(np.float32)  check
#ex_8_v5: np.concatenate([extra]).astype(np.float32)  

#____________________________________________________________________________________________________________________________

# --------- nDCG@k for grouped nodes ----------

def _dcg(rels):

    # rels is relevance list ordered by prediction-desc
    rels = np.asarray(rels, dtype=np.float32)
    denom = np.log2(np.arange(2, len(rels) + 2))
    return float(np.sum(rels / denom))


#float(np.sum((2**rels - 1) / denom)) #good for cora and citeseer and not good for pubmed
#float(np.sum(rels / denom)) 

#____________________________________________________________________________________________________________________________

def tr_remove_highest_n_edges_from_graph(data, predictions, n_edges, promo_mode): 
    """
    Removes the top n edges with the highest or lowest predicted change position from the Citeseer graph.
    
    Args:
        data (torch_geometric.data.Data): Citeseer graph data.
        predictions (dict): Sorted predicted change positions.
        n_edges (int): Number of edges to remove.
        promo_mode (bool): If True, removes edges with the lowest predicted change position instead.
     
    Returns:
        torch_geometric.data.Data, list: Updated graph with edges removed and a list of removed edges.
    """
    edges_sorted = sorted(predictions.keys(), key=lambda x: predictions[x], reverse=not promo_mode)
    edges_to_remove = set(edges_sorted[:n_edges])
    mask = torch.tensor([tuple(edge.tolist()) not in edges_to_remove for edge in data.edge_index.t()], dtype=torch.bool)
    
    updated_data = data.clone()
    updated_data.edge_index = data.edge_index[:, mask]
    
    removed_edges = list(edges_to_remove)
    
    return removed_edges
#____________________________________________________________________________________________________________________________

# Scorer model

class EdgeScorer_v2(nn.Module):
    def __init__(self, in_dim, hidden=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x)  # [B, 1]
 
#____________________________________________________________________________________________________________________________

# --------- Rank loss (logistic) ----------
def per_node_logistic_rank_loss(scores, deltas, nodes, max_pairs=8192):
    scores = scores.squeeze(1)
    deltas = deltas.squeeze(1)
    uniq = nodes.unique()
    total = scores.new_tensor(0.0)
    count = 0
    for nid in uniq.tolist():
        idx = (nodes == nid).nonzero(as_tuple=True)[0]
        if idx.numel() < 2:
            continue
        s, d = scores[idx], deltas[idx]
        diff_d = d.unsqueeze(1) - d.unsqueeze(0)
        mask = (diff_d > 0)
        pairs = mask.nonzero(as_tuple=False)
        if pairs.size(0) == 0:
            continue
        if pairs.size(0) > (max_pairs or 10**9):
            perm = torch.randperm(pairs.size(0), device=pairs.device)[:max_pairs]
            pairs = pairs[perm]
        ds = s[pairs[:, 0]] - s[pairs[:, 1]]
        w  = diff_d[pairs[:, 0], pairs[:, 1]].abs().clamp_min(1e-6)
        node_loss = (torch.log1p(torch.exp(-ds)) * w).mean()
        total += node_loss
        count += 1
    return total / max(count, 1)
#____________________________________________________________________________________________________________________________


def _ndcg_at_k_for_group(true_rel, pred_scores, k):
    # rank by pred desc; take top-k
    order = np.argsort(-pred_scores)[:k]
    rel_topk = true_rel[order]
    dcg_k = _dcg(rel_topk)
    # ideal order by true rel desc
    ideal = np.sort(true_rel)[::-1][:k]
    idcg_k = _dcg(ideal)
    return dcg_k / idcg_k if idcg_k > 0 else 0.0
#____________________________________________________________________________________________________________________________

def ndcg_by_node(df, b_list=(1,2,3,4,5), rel_col="delta"):  # fall back to 'delta_z' if needed
    if rel_col not in df.columns:
        rel_col = "delta_z"
    out = {f"nDCG@{b}": [] for b in b_list}
    for nid, g in df.groupby("node"):
        if len(g) == 0:
            continue
        true_rel = g[rel_col].to_numpy(dtype=np.float32)
        pred     = g["pred"].to_numpy(dtype=np.float32)
        for b in b_list:
            k = min(b, len(g))
            out[f"nDCG@{b}"].append(_ndcg_at_k_for_group(true_rel, pred, k))
    # average over nodes
    return {k: (np.mean(v) if len(v) else 0.0) for k, v in out.items()}
#____________________________________________________________________________________________________________________________

# --------- DF builders / preprocessing ----------
def build_df_for_nodes(simplified_performance, nodes):
    rows = []
    node_set = set(nodes)
    for node, edge_dict in simplified_performance.items():
        if node in node_set:
            for (src, dst), delta in edge_dict.items():
                rows.append({"src": int(src), "dst": int(dst), "delta": float(delta), "node": int(node)})
    return pd.DataFrame(rows)
#____________________________________________________________________________________________________________________________

def zscore_per_node(df):
    df = df.copy()
    df["delta_z"] = df.groupby("node")["delta"].transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))
    return df
#____________________________________________________________________________________________________________________________

def cap_per_node(df, max_edges_per_node=80):
    if max_edges_per_node is None:
        return df
    parts = []
    for nid, g in df.groupby("node"):
        if len(g) > max_edges_per_node:
            parts.append(g.sample(n=max_edges_per_node, random_state=42))
        else:
            parts.append(g)
    return pd.concat(parts, ignore_index=True)
#____________________________________________________________________________________________________________________________

# --------- Feature tensorizer (legacy or target-aware) ----------
def infer_in_dim(feature_fn, dataset_embeddings, data):
    try:
        eg = feature_fn(dataset_embeddings, (0, 1), 0, data, None)  # target-aware
        return len(eg), True
    except TypeError:
        eg = feature_fn(dataset_embeddings, (0, 1), data)            # legacy
        return len(eg), False
#____________________________________________________________________________________________________________________________

def tensorize_df(df, dataset_embeddings, data, feature_fn, in_dim, device, target_aware):
    if len(df) == 0:
        return (
            torch.zeros((0, in_dim), dtype=torch.float32, device=device),
            torch.zeros((0, 1), dtype=torch.float32, device=device),
            torch.zeros((0,), dtype=torch.long, device=device),
        )
    cached = {}  # feature fn self-inits
    X, y, n = [], [], []
    for r in df.itertuples(index=False):
        if target_aware:
            f = feature_fn(dataset_embeddings, (r.src, r.dst), r.node, data, cached)
        else:
            f = feature_fn(dataset_embeddings, (r.src, r.dst), data)
        X.append(f); y.append(r.delta_z); n.append(r.node)
    X_t = torch.tensor(np.asarray(X, dtype=np.float32), device=device)
    y_t = torch.tensor(np.asarray(y, dtype=np.float32), device=device).unsqueeze(1)
    n_t = torch.tensor(np.asarray(n, dtype=np.int64), device=device)
    return X_t, y_t, n_t

#____________________________________________________________________________________________________________________________


def load_scorer_model(dataset_embeddings, load_path, data, feature_fn, hidden=None, dropout=None):
    """
    Robust loader for EdgeScorer_v2 that infers the architecture from the checkpoint.

    - Supports checkpoints saved as raw state_dict or as {"state_dict": ..., "meta": {...}}.
    - Infers `hidden` from the first Linear weight shape if not provided.
    - Verifies input dim matches the feature function.
    """
    if feature_fn is None:
        raise ValueError("Please provide the same feature_fn used in training.")

    # --- Infer input dimension from feature_fn (handles target-aware or legacy) ---
    try:
        eg_feat = feature_fn(dataset_embeddings, (0, 1), 0, data, None)  # target-aware signature
    except TypeError:
        eg_feat = feature_fn(dataset_embeddings, (0, 1), data)           # legacy signature
    in_dim = int(len(eg_feat))

    # --- Load checkpoint ---
    ckpt = torch.load(load_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        meta = ckpt.get("meta", {})
    else:
        state_dict = ckpt
        meta = {}

    # --- Try to find the first layer weight to infer hidden & verify in_dim ---
    # Common key: "net.0.weight" (first Linear in Sequential)
    linear0_key = None
    for k in ["net.0.weight", "module.net.0.weight", "model.net.0.weight"]:
        if k in state_dict:
            linear0_key = k
            break
    if linear0_key is None:
        # Fallback: find any *.0.weight that looks like Linear
        for k in state_dict:
            if k.endswith(".0.weight"):
                linear0_key = k
                break
    if linear0_key is None:
        raise RuntimeError("Could not locate first Linear layer weight in checkpoint.")

    W0 = state_dict[linear0_key]             # shape [hidden_chkpt, in_dim_chkpt]
    hidden_chkpt, in_dim_chkpt = W0.shape

    # If user didn't pass hidden/dropout, infer from checkpoint/meta
    if hidden is None:
        hidden = int(hidden_chkpt)
    if dropout is None:
        dropout = float(meta.get("dropout", 0.5))

    # Sanity check: input dim mismatch means wrong feature_fn or different feature config.
    if in_dim_chkpt != in_dim:
        raise RuntimeError(
            f"Input dim mismatch: checkpoint expects in_dim={in_dim_chkpt}, "
            f"but feature_fn produced in_dim={in_dim}. Make sure you pass the same "
            f"feature_fn (and config) used at training."
        )

    # --- Build model with inferred sizes and load ---
    model = EdgeScorer_v2(in_dim, hidden=hidden, dropout=dropout)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print(f"✅ Loaded scorer from {load_path}")
    #print(f"    in_dim={in_dim} (ckpt {in_dim_chkpt}), hidden={hidden} (ckpt {hidden_chkpt}), dropout={dropout}")
    return model

#____________________________________________________________________________________________________________________________

# ==================================================================================================
# Predict edge scores for a given node
# ==================================================================================================
def predict_edges_for_node_scorer_model(scorer_model, dataset_embeddings, data, node_id, feature_fn):

    if isinstance(node_id, torch.Tensor):
        node_id = int(node_id.item())

    scorer_model.eval()
    ei = data.edge_index
    mask = (ei[0] == node_id) | (ei[1] == node_id)
    connected = ei[:, mask]
    if connected.size(1) == 0:
        return {}

    device = next(scorer_model.parameters()).device
    cached = {}  # self-inits inside feature_fn

    X, edge_list = [], []
    for i in range(connected.size(1)):
        u = int(connected[0, i])
        v = int(connected[1, i])
        try:
            feat = feature_fn(dataset_embeddings, (u, v), node_id, data, cached)  # target-aware
        except TypeError:
            feat = feature_fn(dataset_embeddings, (u, v), data)                   # legacy
        X.append(feat)
        edge_list.append((u, v))

    X_t = torch.tensor(np.asarray(X, dtype=np.float32), device=device)
    with torch.no_grad():
        scores = scorer_model(X_t).squeeze().detach().cpu().numpy()

    return {edge: float(score) for edge, score in zip(edge_list, scores)}

#____________________________________________________________________________________________________________________________

def build_new_target_node_attack(data, selected_nodes, budget, scorer_model, dataset_embeddings, promotion_mode):
    
    test_nodes =  selected_nodes
    test_nodes_int = test_nodes.item()

    #predictions = predict_edges_for_node_scorer_model(scorer_model, dataset_embeddings, data, test_nodes)
    predictions = predict_edges_for_node_scorer_model(scorer_model, dataset_embeddings, data, test_nodes, feature_fn = edge_features_v5_targeted)
    edges_to_remove = tr_remove_highest_n_edges_from_graph(data, predictions, budget, promotion_mode)
    updated_data = gold_remove_edges(data, edges_to_remove, budget)


    return updated_data

#____________________________________________________________________________________________________________________________
