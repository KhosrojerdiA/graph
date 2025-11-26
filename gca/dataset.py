import os.path as osp
import torch
from torch_geometric.datasets import Planetoid, CitationFull, WikiCS, Coauthor, Amazon
import torch_geometric.transforms as T
from torch_geometric.utils import subgraph, degree, to_undirected

def remove_isolated_nodes(data):
    """
    Removes nodes with zero degree and rebuilds the edge_index so that:
    - Node indices are contiguous [0, num_nodes_new)
    - No degree-0 nodes remain
    - Ensures edge_index and all weights align perfectly
    """
    print(">> Checking for isolated or invalid nodes...")

    # --- Step 1: Ensure edge_index is undirected
    edge_index = to_undirected(data.edge_index)

    # --- Step 2: Identify connected nodes (degree > 0)
    deg = degree(edge_index[0], num_nodes=data.num_nodes)
    connected_nodes = (deg > 0).nonzero(as_tuple=True)[0]
    num_isolated = data.num_nodes - connected_nodes.numel()

    if num_isolated > 0:
        print(f"⚠️  Found {num_isolated} isolated nodes. Removing them...")

        # --- Step 3: Build new subgraph containing only connected nodes
        new_edge_index, mapping = subgraph(connected_nodes, edge_index, relabel_nodes=True)

        # --- Step 4: Reindex features and labels
        data.x = data.x[connected_nodes]
        if hasattr(data, "y"):
            data.y = data.y[connected_nodes]
        data.edge_index = new_edge_index
        data.num_nodes = data.x.size(0)

        # --- Step 5: Remove duplicate edges & self-loops
        data.edge_index = to_undirected(data.edge_index)
        data.edge_index = data.edge_index[:, data.edge_index[0] != data.edge_index[1]]

        # --- Step 6: Validate alignment
        num_edges = data.edge_index.size(1)
        print(f"✅ Removed isolated nodes. New num_nodes={data.num_nodes}, edges={num_edges}")

    else:
        print("✅ No isolated nodes found. Data is consistent.")

    # --- Final check for downstream consistency
    assert data.edge_index.max() < data.num_nodes, "❌ edge_index has invalid node ids!"
    assert data.edge_index.size(0) == 2, "❌ edge_index must be shape [2, num_edges]!"
    assert data.edge_index.size(1) > 0, "❌ Graph has no edges after cleanup!"
    return data


def get_dataset(path, name):
    assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP', 'Karate', 'WikiCS', 'Coauthor-CS', 'Coauthor-Phy',
                    'Amazon-Computers', 'Amazon-Photo', 'ogbn-arxiv', 'ogbg-code']
    name = 'dblp' if name == 'DBLP' else name
    root_path = osp.expanduser('~/datasets')

    if name == 'Coauthor-CS':
        return Coauthor(root=path, name='cs', transform=T.NormalizeFeatures())

    if name == 'Coauthor-Phy':
        return Coauthor(root=path, name='physics', transform=T.NormalizeFeatures())

    if name == 'WikiCS':
        return WikiCS(root=path, transform=T.NormalizeFeatures())

    if name == 'Amazon-Computers':
        return Amazon(root=path, name='computers', transform=T.NormalizeFeatures())

    if name == 'Amazon-Photo':
        return Amazon(root=path, name='photo', transform=T.NormalizeFeatures())

    return (CitationFull if name == 'dblp' else Planetoid)(osp.join(root_path, 'Citation'), name, transform=T.NormalizeFeatures())


def get_path(base_path, name):
    if name in ['Cora', 'CiteSeer', 'PubMed']:
        return base_path
    else:
        return osp.join(base_path, name)
    


def load_data(ep_path, dataset_name):

    if dataset_name == 'CiteSeer': 
        dataset = Planetoid(root=f'{ep_path}/data/Planetoid', name='CiteSeer')
        data = dataset[0]
    elif dataset_name == 'Cora': 
        dataset = Planetoid(root=f'{ep_path}/data/Planetoid', name='Cora')
        data = dataset[0]
    elif dataset_name == 'PubMed':
        #dataset = torch.load(dataset_subgraph_path)
        dataset = Planetoid(root=f'{ep_path}/data/Planetoid', name='PubMed')
        data = torch.load(f'{ep_path}/data/pubmed_subgraph.pt', weights_only=False) #torch.load(dataset_subgraph_path)
        data = remove_isolated_nodes(dataset)

    return dataset, data