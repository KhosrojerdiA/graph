
from greet.utils import *
from torch.nn import Sequential, Linear, ReLU
from sklearn.neighbors import kneighbors_graph
from scipy import sparse
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


EOS = 1e-10



class GCL(nn.Module):
    def __init__(self, nlayers, nlayers_proj, in_dim, emb_dim, proj_dim, dropout, sparse, batch_size):
        super(GCL, self).__init__()

        self.encoder1 = SGC(nlayers, in_dim, emb_dim, dropout, sparse)
        self.encoder2 = SGC(nlayers, in_dim, emb_dim, dropout, sparse)

        if nlayers_proj == 1:
            self.proj_head1 = Sequential(Linear(emb_dim, proj_dim))
            self.proj_head2 = Sequential(Linear(emb_dim, proj_dim))
        elif nlayers_proj == 2:
            self.proj_head1 = Sequential(Linear(emb_dim, proj_dim), ReLU(inplace=True), Linear(proj_dim, proj_dim))
            self.proj_head2 = Sequential(Linear(emb_dim, proj_dim), ReLU(inplace=True), Linear(proj_dim, proj_dim))

        self.batch_size = batch_size


    def get_embedding(self, x, a1, a2, source='all'):
        emb1 = self.encoder1(x, a1)
        emb2 = self.encoder2(x, a2)
        return torch.cat((emb1, emb2), dim=1)


    def get_projection(self, x, a1, a2):
        emb1 = self.encoder1(x, a1)
        emb2 = self.encoder2(x, a2)
        proj1 = self.proj_head1(emb1)
        proj2 = self.proj_head2(emb2)
        return torch.cat((proj1, proj2), dim=1)


    def forward(self, x1, a1, x2, a2):
        emb1 = self.encoder1(x1, a1)
        emb2 = self.encoder2(x2, a2)
        proj1 = self.proj_head1(emb1)
        proj2 = self.proj_head2(emb2)
        loss = self.batch_nce_loss(proj1, proj2)
        return loss


    def set_mask_knn(self, X, k, dataset, metric='cosine'):
        if k != 0:
            path = '../data/knn/{}'.format(dataset)
            if not os.path.exists(path):
                os.makedirs(path)
            file_name = path + '/{}_{}.npz'.format(dataset, k)
            if os.path.exists(file_name):
                knn = sparse.load_npz(file_name)
                # print('Load exist knn graph.')
            else:
                print('Computing knn graph...')
                knn = kneighbors_graph(X, k, metric=metric)
                sparse.save_npz(file_name, knn)
                print('Done. The knn graph is saved as: {}.'.format(file_name))
            knn = torch.tensor(knn.toarray()) + torch.eye(X.shape[0])
        else:
            knn = torch.eye(X.shape[0])
        self.pos_mask = knn
        self.neg_mask = 1 - self.pos_mask


    def batch_nce_loss(self, z1, z2, temperature=0.2, pos_mask=None, neg_mask=None):
        if pos_mask is None and neg_mask is None:
            pos_mask = self.pos_mask
            neg_mask = self.neg_mask

        nnodes = z1.shape[0]
        if (self.batch_size == 0) or (self.batch_size > nnodes):
            loss_0 = self.infonce(z1, z2, pos_mask, neg_mask, temperature)
            loss_1 = self.infonce(z2, z1, pos_mask, neg_mask, temperature)
            loss = (loss_0 + loss_1) / 2.0
        else:
            node_idxs = list(range(nnodes))
            random.shuffle(node_idxs)
            batches = split_batch(node_idxs, self.batch_size)
            loss = 0
            for b in batches:
                weight = len(b) / nnodes
                loss_0 = self.infonce(z1[b], z2[b], pos_mask[:,b][b,:], neg_mask[:,b][b,:], temperature)
                loss_1 = self.infonce(z2[b], z1[b], pos_mask[:,b][b,:], neg_mask[:,b][b,:], temperature)
                loss += (loss_0 + loss_1) / 2.0 * weight
        return loss


    def infonce(self, anchor, sample, pos_mask, neg_mask, tau):
        pos_mask = pos_mask.cuda()
        neg_mask = neg_mask.cuda()
        sim = self.similarity(anchor, sample) / tau
        exp_sim = torch.exp(sim) * neg_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = log_prob * pos_mask
        loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
        return -loss.mean()


    def similarity(self, h1: torch.Tensor, h2: torch.Tensor):
        h1 = F.normalize(h1)
        h2 = F.normalize(h2)
        return h1 @ h2.t()






def edge_weight_norm_both(edge_index, edge_weight, num_nodes):
    """
    Equivalent of DGL's EdgeWeightNorm(norm='both')
    """
    row, col = edge_index
    deg = torch.zeros(num_nodes, device=edge_weight.device)
    deg.scatter_add_(0, row, edge_weight)
    deg_inv_sqrt = (deg + EOS).pow(-0.5)
    norm_weight = edge_weight * deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return norm_weight


def add_self_loops(edge_index, edge_weight, num_nodes, fill_value=1.0):
    """
    Add self-loops to adjacency (edge_index, edge_weight)
    """
    device = edge_index.device
    loop_index = torch.arange(num_nodes, device=device)
    loop_edges = torch.stack([loop_index, loop_index])
    loop_weight = torch.full((num_nodes,), fill_value, device=device)
    new_edges = torch.cat([edge_index, loop_edges], dim=1)
    new_weights = torch.cat([edge_weight, loop_weight])
    return new_edges, new_weights


class Edge_Discriminator(nn.Module):
    def __init__(self, nnodes, input_dim, alpha, sparse,
                 hidden_dim=128, temperature=1.0, bias=0.0001):
        super(Edge_Discriminator, self).__init__()

        self.embedding_layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
        self.edge_mlp = nn.Linear(hidden_dim * 2, 1)

        self.temperature = temperature
        self.bias = bias
        self.nnodes = nnodes
        self.sparse = sparse
        self.alpha = alpha

    def get_node_embedding(self, h):
        for layer in self.embedding_layers:
            h = F.relu(layer(h))
        return h

    def get_edge_weight(self, embeddings, edges):
        s1 = self.edge_mlp(torch.cat((embeddings[edges[0]], embeddings[edges[1]]), dim=1)).flatten()
        s2 = self.edge_mlp(torch.cat((embeddings[edges[1]], embeddings[edges[0]]), dim=1)).flatten()
        return (s1 + s2) / 2

    def gumbel_sampling(self, edges_weights_raw):
        eps = (self.bias - (1 - self.bias)) * torch.rand(edges_weights_raw.size(), device=edges_weights_raw.device) + (1 - self.bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = (gate_inputs + edges_weights_raw) / self.temperature
        return torch.sigmoid(gate_inputs).squeeze()

    def weight_forward(self, features, edges):
        embeddings = self.get_node_embedding(features)
        edges_weights_raw = self.get_edge_weight(embeddings, edges)
        weights_lp = self.gumbel_sampling(edges_weights_raw)
        weights_hp = 1 - weights_lp
        return weights_lp, weights_hp

    def weight_to_adj(self, edges, weights_lp, weights_hp):
        if not self.sparse:
            adj_lp = get_adj_from_edges(edges, weights_lp, self.nnodes)
            adj_lp += torch.eye(self.nnodes, device=adj_lp.device)
            adj_lp = normalize_adj(adj_lp, 'sym', self.sparse)

            adj_hp = get_adj_from_edges(edges, weights_hp, self.nnodes)
            adj_hp += torch.eye(self.nnodes, device=adj_hp.device)
            adj_hp = normalize_adj(adj_hp, 'sym', self.sparse)

            mask = torch.zeros_like(adj_lp)
            mask[edges[0], edges[1]] = 1.
            mask.requires_grad = False
            adj_hp = torch.eye(self.nnodes, device=adj_hp.device) - adj_hp * mask * self.alpha
        else:
            # DGL-free sparse mode
            device = edges[0].device
            num_nodes = self.nnodes

            # --- Low-pass adjacency ---
            edges_lp, weights_lp = add_self_loops(edges, weights_lp, num_nodes)
            weights_lp = weights_lp + EOS
            weights_lp = edge_weight_norm_both(edges_lp, weights_lp, num_nodes)
            adj_lp = {"edge_index": edges_lp, "edge_weight": weights_lp}

            # --- High-pass adjacency ---
            edges_hp, weights_hp = add_self_loops(edges, weights_hp, num_nodes)
            weights_hp = weights_hp + EOS
            weights_hp = edge_weight_norm_both(edges_hp, weights_hp, num_nodes)
            weights_hp *= -self.alpha
            weights_hp[edges.shape[1]:] = 1  # self-loop weights = +1
            adj_hp = {"edge_index": edges_hp, "edge_weight": weights_hp}

        return adj_lp, adj_hp

    def forward(self, features, edges):
        weights_lp, weights_hp = self.weight_forward(features, edges)
        adj_lp, adj_hp = self.weight_to_adj(edges, weights_lp, weights_hp)
        return adj_lp, adj_hp, weights_lp, weights_hp




class SGC(nn.Module):
    def __init__(self, nlayers, in_dim, emb_dim, dropout, sparse):
        super(SGC, self).__init__()
        self.dropout = dropout
        self.sparse = sparse
        self.linear = nn.Linear(in_dim, emb_dim)
        self.k = nlayers

    def forward(self, x, g):
        """
        g:
            - if sparse: a dict {"edge_index": [2, num_edges], "edge_weight": [num_edges]}
            - if dense: adjacency matrix [N, N]
        """
        x = F.relu(self.linear(x))

        if self.sparse:
            edge_index, edge_weight = g["edge_index"], g["edge_weight"]
            row, col = edge_index
            num_nodes = x.size(0)

            # Propagate K times (like g.update_all)
            for _ in range(self.k):
                out = torch.zeros_like(x)
                # Weighted message passing: out_i += w_ij * x_j
                out.index_add_(0, row, x[col] * edge_weight.unsqueeze(1))
                x = out
            return x

        else:
            # Dense adjacency case
            for _ in range(self.k):
                x = torch.matmul(g, x)
            return x


class GREETWrapper(nn.Module):
    """
    Wrapper to hold both cl_model and discriminator
    so they can be saved/loaded as a single torch.nn.Module.
    Compatible with load_embedding_model() and other GCN-style calls.
    """
    def __init__(self, cl_model, discriminator):
        super(GREETWrapper, self).__init__()
        self.cl_model = cl_model
        self.discriminator = discriminator

    def forward(self, features, edges, str_encodings=None):
        # backward compatibility: if no str_encodings provided, use zeros
        if str_encodings is None:
            # Match expected dimension: if model trained with input_dim = features + str_encodings
            # Infer str_encodings dim from discriminator first Linear layer
            expected_total_dim = self.discriminator.embedding_layers[0].in_features
            feat_dim = features.shape[1]
            extra_dim = max(expected_total_dim - feat_dim, 0)
            if extra_dim > 0:
                str_encodings = torch.zeros((features.shape[0], extra_dim), device=features.device)
            else:
                # fallback if already matches
                str_encodings = torch.zeros_like(features)


        adj_1, adj_2, _, _ = self.discriminator(torch.cat((features, str_encodings), 1), edges)
        embedding = self.cl_model.get_embedding(features, adj_1, adj_2)
        return embedding