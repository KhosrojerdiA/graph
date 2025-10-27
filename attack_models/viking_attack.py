
import torch
import numpy as np
import scipy.linalg as spl
from torch_geometric.utils import to_scipy_sparse_matrix

#____________________________________________________________________________________________________________________________________________

# -------------------------
# Loss estimators (exactly like repo)
# -------------------------
def _sum_of_powers(x: np.ndarray, power: int):
    n = x.shape[0]
    sum_powers = np.zeros((power, n))
    for i, p in enumerate(range(1, power + 1)):
        sum_powers[i] = np.power(x, p)
    return sum_powers.sum(0)
#____________________________________________________________________________________________________________________________________________

def _estimate_loss_unsupervised(candidates, flip_indicator, vals_org, vecs_org, n_nodes, dim, window_size):
    loss_est = np.zeros(len(candidates))
    for idx in range(len(candidates)):
        i, j = candidates[idx]
        vals_est = vals_org + flip_indicator[idx] * (
            2 * vecs_org[i] * vecs_org[j] - vals_org * (vecs_org[i] ** 2 + vecs_org[j] ** 2)
        )
        vals_sum_powers = _sum_of_powers(vals_est, window_size)
        loss_ij = np.sqrt(np.sum(np.sort(vals_sum_powers ** 2)[: n_nodes - dim]))
        loss_est[idx] = loss_ij
    return loss_est
#____________________________________________________________________________________________________________________________________________

def _estimate_loss_supervised(candidates, flip_indicator, A, L, n_nodes):
    assert L is not None, "L (label matrix) required for supervised loss"
    loss_val = np.zeros(len(candidates))
    tempA = A.todense()
    midval = tempA @ L
    numerator = np.sum(np.multiply(midval, L), axis=1)
    denominator = np.sum(midval, axis=1)
    nmu = np.linalg.norm(numerator / denominator)
    for idx, (i, j) in enumerate(candidates):
        midval[i] = midval[i] + flip_indicator[idx] * L[j]
        midval[j] = midval[j] + flip_indicator[idx] * L[i]
        numerator = np.sum(np.multiply(midval, L), axis=1)
        denominator = np.sum(midval, axis=1)
        muvec = numerator / denominator
        loss_val[idx] = (nmu - np.linalg.norm(muvec))
        # revert
        midval[i] = midval[i] - flip_indicator[idx] * L[j]
        midval[j] = midval[j] - flip_indicator[idx] * L[i]
    return loss_val
#____________________________________________________________________________________________________________________________________________
# -------------------------
# Main per-node VIKING function
# -------------------------
def viking_attack_per_node(
    data,
    target_node,
    budget,          
    dim,
    window_size,
    supervised,
    seed=0,
    verbose=True,
    device="cuda"
):

    """
    Directed VIKING-style per-node attack (removal-only).
    Considers both incoming and outgoing edges of the target node,
    then removes the top-`budget` edges with the highest estimated loss impact.
    """

    data = data.clone().to(device)
    edge_index = data.edge_index
    src, dst = edge_index
    E = edge_index.size(1)

    # convert target to int
    target = int(target_node.item()) if torch.is_tensor(target_node) else int(target_node)

    # directed adjacency (no symmetrization)
    A = to_scipy_sparse_matrix(edge_index, num_nodes=data.num_nodes).tocsr()

    # find both incoming & outgoing directed edges
    mask = (src == target) | (dst == target)
    incident_edges = edge_index[:, mask]
    if incident_edges.size(1) == 0:
        if verbose:
            print(f"[VIKING-directed] Node {target}: no incident edges.", flush=True)
        return data, []

    # candidate directed edges
    edges_np = incident_edges.cpu().numpy().T  # shape [num_edges, 2]
    n_nodes = A.shape[0]
    delta_w = np.full(len(edges_np), -1)  # all removals

    # supervised/unsupervised loss
    if supervised and getattr(data, "y", None) is not None:
        labels = data.y.cpu().numpy()
        L = (labels == np.unique(labels)[:, None]).astype(int).T
        losses = _estimate_loss_supervised(edges_np, delta_w, A, L, n_nodes)
    else:
        D = np.diag(A.sum(1).A1 + 1e-8)
        try:
            vals, vecs = spl.eigh(A.toarray(), D)
        except Exception:
            vals, vecs = np.linalg.eigh(A.toarray())
        losses = _estimate_loss_unsupervised(edges_np, delta_w, vals, vecs, n_nodes, dim, window_size)

    # rank by loss (model decides which direction to remove)
    k = min(budget, len(losses))
    ranked_local = np.argsort(losses)[-k:]
    chosen_edges = edges_np[ranked_local]

    # remove chosen edges from edge_index
    edge_pairs = edge_index.t().tolist()
    remove_set = {tuple(pair) for pair in chosen_edges.tolist()}
    keep_mask = torch.tensor(
        [tuple(pair) not in remove_set for pair in edge_pairs],
        dtype=torch.bool,
        device=device
    )

    new_edge_index = edge_index[:, keep_mask]
    attacked = data.clone()
    attacked.edge_index = new_edge_index

    if verbose:
        removed_dirs = [("out" if s == target else "in") for (s, d) in chosen_edges]
        dir_summary = ", ".join(removed_dirs)
        print(f"[VIKING-directed] target={target}, removed {len(chosen_edges)} edges ({dir_summary})", flush=True)
        print(f"Original edges: {E}, Updated edges: {new_edge_index.size(1)}", flush=True)

    return attacked








