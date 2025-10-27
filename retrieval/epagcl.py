import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.utils import degree, to_undirected, dropout_edge
from torch_geometric.datasets import Planetoid
from torch_geometric import transforms as T
from torch_geometric.data import Data

# ============================== Model.py (verbatim structure) ==============================
class MyModel(nn.Module):
    def __init__(self, in_channels, out_channels, num_proj_hidden, temperature=0.3, activation='PReLU', conv_type="gcn"):
        super().__init__()
        if activation == 'PReLU':
            self.activation = nn.PReLU()
        else:
            self.activation = nn.ReLU()
        self._build_up(in_channels, out_channels, num_proj_hidden, temperature)

    def forward(self, x, edge_index):
        out = self.activation(self.conv1(x, edge_index))
        out = self.activation(self.conv2(out, edge_index))
        proj = F.relu(self.fc1(out))
        proj = self.fc2(proj)
        return out, proj

    def loss(self, z1, z2, batch_compute=False, batch_size=256):
        l1 = self.cal_loss(z1, z2, batch_compute, batch_size)
        l2 = self.cal_loss(z2, z1, batch_compute, batch_size)
        l = (l1 + l2) * 0.5
        return l.mean()

    def cal_sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        sim = torch.mm(z1, z2.t())
        return torch.exp(sim / self.t)

    def cal_loss(self, z1, z2, batch_compute=False, batch_size=256):
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        if not batch_compute:
            indices = torch.arange(0, num_nodes, device=device)
        else:
            indices = torch.randperm(num_nodes, device=device)
        losses = []
        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            if not batch_compute:
                refl_sim = self.cal_sim(z1[mask], z1)
                between_sim = self.cal_sim(z1[mask], z2)
                losses.append(
                    -torch.log(
                        between_sim[:, i * batch_size:(i + 1) * batch_size].diag() /
                        (refl_sim.sum(1) + between_sim.sum(1)
                         - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())
                    )
                )
            else:
                refl_sim = self.cal_sim(z1[mask], z1[mask])
                between_sim = self.cal_sim(z1[mask], z2[mask])
                losses.append(-torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag())))
        return torch.cat(losses)

    def _build_up(self, in_channels, out_channels, num_proj_hidden, temperature, conv_type="gcn"):
        if conv_type == "sage":
            self.conv1 = SAGEConv(in_channels, 2 * out_channels)
            self.conv2 = SAGEConv(2 * out_channels, out_channels)
        else:  # Default to GCN
            self.conv1 = GCNConv(in_channels, 2 * out_channels)
            self.conv2 = GCNConv(2 * out_channels, out_channels)

        self.fc1 = nn.Linear(out_channels, num_proj_hidden)
        self.fc2 = nn.Linear(num_proj_hidden, out_channels)
        self.t = temperature


    def freeze(self, flag=True):
        for p in self.parameters():
            p.requires_grad = not flag


class LinearClassifier(nn.Module):
    def __init__(self, feat_dim, classes_num):
        super().__init__()
        self.fc = nn.Linear(feat_dim, classes_num)
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
    def forward(self, x):
        return self.fc(x)


# ============================== Augmentation.py (key functions) ==============================
def _get_index(node_num, node_list):
    return (node_list == node_num).nonzero()

def _get_edge_index(target, edge_index):
    edge_index = to_undirected(edge_index)
    deg = degree(edge_index[0]).to(torch.int32)
    output = torch.tensor([], device=edge_index.device)
    for t in target:
        m = deg[:t].sum()
        d = deg[t]
        output = torch.cat((output, edge_index[:, m:m + d]), dim=-1)
    return output

def _reverse(edge_index):
    # Produces the complement edges of the undirected graph (sparse trick), with a
    # heavy-path fast path for large graphs; mirrors repo logic.
    edge_index = to_undirected(edge_index)
    node_num = edge_index[0].max() + 1
    if node_num > 30000:
        print("Too many nodes, processing")
        t1 = time()
        deg = degree(edge_index[1])
        length = edge_index.size(1)
        candidate_num = min(node_num, math.ceil(math.sqrt(2 * length)))
        candidate_nodes = torch.topk(deg, candidate_num)[1]
        edge_index_sub = _get_edge_index(candidate_nodes, edge_index)
        mask = torch.tensor([k in candidate_nodes for k in edge_index_sub[1]], device=edge_index.device)
        edge_index_sub = edge_index_sub[:, mask]
        edge_index_1 = torch.tensor([_get_index(u, candidate_nodes) for u in edge_index_sub[0]]).view(1, -1)
        edge_index_2 = torch.tensor([_get_index(u, candidate_nodes) for u in edge_index_sub[1]]).view(1, -1)
        edge_index_ = torch.cat((edge_index_1, edge_index_2))
        device = edge_index_.device
        aux = torch.ones(edge_index_.shape[1], device=device)
        reverse_index_ = (1 - torch.sparse_coo_tensor(edge_index_, aux).to_dense()).to_sparse().indices()
        reverse_index = candidate_nodes[reverse_index_]
        t2 = time()
        print(f"Completed, processing time:{t2 - t1:.4f}")
        return reverse_index
    else:
        device = edge_index.device
        aux = torch.ones(edge_index.shape[1], device=device)
        return (1 - torch.sparse_coo_tensor(edge_index, aux).to_dense()).to_sparse().indices()

def add_edge_weights(edge_index, p):
    device = edge_index.device
    edge_index = to_undirected(edge_index)
    deg = degree(edge_index[1])
    l = math.ceil(edge_index.size(1) * p)          # number of edges to consider adding
    add_index = _reverse(edge_index).to(device)    # candidate edges (complement)
    w = 1 / ((deg[add_index[0]] + 1) * (deg[add_index[1]] + 1)) ** 0.5
    weights = (w.max() - w) / (w.max() - w.mean())
    # Keep top-l only (acceleration trick)
    sel_mask = torch.topk(weights, l)[1]
    edge_weights = weights[sel_mask]
    add_edge = add_index[:, sel_mask]
    return edge_weights, add_edge

def add_edges(edge_weights, edge_index, former_index, p=0.5, threshold=0.9):
    # p: edge adding rate; normalize, cap at threshold, Bernoulli sample
    edge_weights = edge_weights / edge_weights.mean() * p
    edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
    sel_mask = torch.bernoulli(edge_weights).to(torch.bool)
    add_index = edge_index[:, sel_mask]
    return torch.cat((former_index, add_index), dim=1)

def add_edge_random(edge_index, p=0.1):
    index = to_undirected(edge_index)
    l = index.size(1)
    add_index = torch.tensor(_reverse(index)).to(index.device)
    l2 = add_index.size(1)
    p_adj = p * l / l2
    weights = (torch.ones(l2, device=index.device) * p_adj)
    sel_mask = torch.bernoulli(weights).to(torch.bool)
    add_index = add_index[:, sel_mask]
    return torch.cat((edge_index, add_index), dim=1)

def drop_edge_weights(edge_index):
    edge_index = to_undirected(edge_index)
    deg = degree(edge_index[1])
    w = 1 / (deg[edge_index[0]] * deg[edge_index[1]]) ** 0.5
    weights = (w - w.min()) / (w.mean() - w.min())
    return weights

def drop_edges(edge_index, edge_weights, p, threshold=1.0):
    edge_weights = edge_weights / edge_weights.mean() * p
    edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
    sel_mask = torch.bernoulli(1.0 - edge_weights).to(torch.bool)
    return edge_index[:, sel_mask]

def drop_feature_random(x, drop_prob):
    drop_mask = torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0
    return x


# ============================== Optim (Utils.get_optim minimal) ==============================
def _get_optim(params, lr=1e-3, weight_decay=1e-4, optim='Adam'):
    if optim.lower() == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optim.lower() == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optim}")


# ============================== Public API: train + embed (main.py mirrored) ==============================
class EPAGCLConfig:
    # Mirrors defaults from Arguments.py
    def __init__(
        self,
        device='cuda:0',
        epoch=500,
        repeat=1,
        add_single=False,
        not_add_edge=False,
        not_drop_edge=False,
        add_edge_random=False,
        batch_compute=False,
        optim='Adam',
        lr=1e-3,
        weight_decay=1e-4,
        loss_log=1,
        eval=50,
        edge_add_rate=0.1,
        edge_drop_rate_1=0.2,
        edge_drop_rate_2=0.3,
        feat_drop_rate_1=0.1,
        feat_drop_rate_2=0.1,
        feat_dim=256,
        proj_hidden_dim=256,
        temperature=0.3,
    ):
        self.device = device
        self.epoch = epoch
        self.repeat = repeat
        self.add_single = add_single
        self.not_add_edge = not_add_edge
        self.not_drop_edge = not_drop_edge
        self.add_edge_random = add_edge_random
        self.batch_compute = batch_compute
        self.optim = optim
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_log = loss_log
        self.eval = eval
        self.edge_add_rate = edge_add_rate
        self.edge_drop_rate_1 = edge_drop_rate_1
        self.edge_drop_rate_2 = edge_drop_rate_2
        self.feat_drop_rate_1 = feat_drop_rate_1
        self.feat_drop_rate_2 = feat_drop_rate_2
        self.feat_dim = feat_dim
        self.proj_hidden_dim = proj_hidden_dim
        self.temperature = temperature

    @property
    def add_edge(self):  # as in Arguments.py
        return not self.not_add_edge

    @property
    def drop_edge(self):
        return not self.not_drop_edge


def _setup_views(data, cfg: EPAGCLConfig):
    # Precompute weights for add/drop ops (once), like main.py -> setup()
    if cfg.not_add_edge:
        adding_edge_weights, adding_edge = None, None
    else:
        adding_edge_weights, adding_edge = add_edge_weights(data.edge_index, cfg.edge_add_rate)
    if cfg.not_drop_edge:
        dropping_edge_weights = None
    else:
        dropping_edge_weights = drop_edge_weights(data.edge_index)
    return adding_edge_weights, adding_edge, dropping_edge_weights


def _train_once(data, add_w, add_idx, graph_model, drop_w, cfg: EPAGCLConfig):
    model = MyModel(data.num_features, cfg.feat_dim, cfg.proj_hidden_dim, cfg.temperature, conv_type="sage" if graph_model == "epagcl_sage" else "gcn").to(cfg.device)
    optimizer = _get_optim(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, optim=cfg.optim)

    for epoch in range(cfg.epoch):
        model.train()
        optimizer.zero_grad()

        # --- Build two edge views (drop, then add depending on flags)
        if cfg.drop_edge and drop_w is not None:
            e1 = drop_edges(data.edge_index, drop_w, cfg.edge_drop_rate_1)
            e2 = drop_edges(data.edge_index, drop_w, cfg.edge_drop_rate_2)
        else:
            e1, e2 = data.edge_index, data.edge_index

        if cfg.add_edge_random:
            edge_index_1 = add_edge_random(e1)
            edge_index_2 = add_edge_random(e2)
        elif cfg.add_edge and (add_w is not None) and (add_idx is not None):
            edge_index_1 = add_edges(add_w, add_idx, e1)
            edge_index_2 = add_edges(add_w, add_idx, e2)
        else:
            edge_index_1, edge_index_2 = e1, e2

        if cfg.add_single:
            edge_index_2 = e2

        # --- Feature dropout (two augmented views)
        x1 = drop_feature_random(data.x, cfg.feat_drop_rate_1)
        x2 = drop_feature_random(data.x, cfg.feat_drop_rate_2)

        # --- Contrastive learning forward
        _, z1 = model(x1.to(cfg.device), edge_index_1.to(cfg.device))
        _, z2 = model(x2.to(cfg.device), edge_index_2.to(cfg.device))
        loss = model.loss(z1, z2, batch_compute=cfg.batch_compute)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % cfg.eval == 0 or (epoch + 1) == cfg.epoch:
            print(f"Epoch {epoch+1}/{cfg.epoch} | loss: {loss.item():.4f}")

    # --- Final embeddings (encoder output only, no projection)
    model.eval()
    with torch.no_grad():
        h, _ = model(data.x.to(cfg.device), data.edge_index.to(cfg.device))

    # Return both embeddings and trained model
    return h.detach().cpu(), model


def get_epagcl_embeddings(
    data: Data,
    device: str = "cuda",
    epochs: int = 500,
    batch_compute: bool = False,
    add_single: bool = False,
    add_edge_random: bool = False,
    not_add_edge: bool = False,
    not_drop_edge: bool = False,
    edge_add_rate: float = 0.1,
    edge_drop_rate_1: float = 0.2,
    edge_drop_rate_2: float = 0.3,
    feat_drop_rate_1: float = 0.1,
    feat_drop_rate_2: float = 0.1,
    feat_dim: int = 256,
    proj_hidden_dim: int = 256,
    temperature: float = 0.3,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    optim: str = "Adam",
    graph_model: str = "epagcl_gcn",   # NEW
):
    """
    1:1 EPAGCL training that returns both node embeddings (encoder outputs)
    and the trained model.
    """
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    cfg = EPAGCLConfig(
        device=str(device),
        epoch=epochs,
        batch_compute=batch_compute,
        add_single=add_single,
        not_add_edge=not_add_edge,
        not_drop_edge=not_drop_edge,
        add_edge_random=add_edge_random,
        edge_add_rate=edge_add_rate,
        edge_drop_rate_1=edge_drop_rate_1,
        edge_drop_rate_2=edge_drop_rate_2,
        feat_drop_rate_1=feat_drop_rate_1,
        feat_drop_rate_2=feat_drop_rate_2,
        feat_dim=feat_dim,
        proj_hidden_dim=proj_hidden_dim,
        temperature=temperature,
        lr=lr,
        weight_decay=weight_decay,
        optim=optim,
        repeat=1,
        eval=50,
    )

    add_w, add_idx, drop_w = _setup_views(data, cfg)
    h, model = _train_once(data, add_w, add_idx, graph_model, drop_w, cfg)

    # Return embeddings + trained model
    return h, model




