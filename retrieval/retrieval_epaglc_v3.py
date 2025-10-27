

import sys, types
import os

rt_project_path = './'
sys.path.append(rt_project_path)


import EPAGCL.epagcl as model_3
sys.modules['EPAGCL.model_2'] = model_3

import sys, types


from utils.utils import *
from EPAGCL.epagcl import *  

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GATv2Conv
from torch.nn import Linear
from torch_geometric.utils import degree
import random
 


def retrieval_v3(data, data_name, graph_model, min_number_edges, save_dir, main_seed):

    torch.manual_seed(main_seed)
    torch.cuda.manual_seed(main_seed)
    np.random.seed(main_seed)
    random.seed(main_seed)
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    data = data.to(device)  # Move data to the appropriate device 
    data_num_nodes = data.num_nodes


    if graph_model.lower() == "epagcl_gcn" or graph_model.lower() == "epagcl_sage":

        file_prefix = f"{data_name}_{graph_model}_epochs_2000"
        emb_path = os.path.join(save_dir, f"{file_prefix}_embeddings.pt")
        model_path = os.path.join(save_dir, f"{file_prefix}_model.pt")


        if os.path.exists(emb_path) and os.path.exists(model_path):
            print(f">> Found cached EPAGCL {graph_model} model and embeddings for {data_name}! Loading from disk...", flush=True)
            print(f"Location: {emb_path}", flush=True)
            dataset_embeddings = torch.load(emb_path, map_location="cpu")
            model = torch.load(model_path, map_location=device)
            model.eval()
            print(f"- Loaded existing EPAGCL {graph_model} for {data_name} embeddings and model successfully!", flush=True)
            print("____________________________________________")

        else:
            print(">> Could not find cached EPAGCL model and embeddings for {data_name}.", flush=True)
 
            
    #____________________________________________________________________________________________________________________________________

    num_nodes = data.x.size(0)
    edge_index = data.edge_index
    node_degrees = degree(edge_index[0], num_nodes=num_nodes).to(device)


    nodes_with_min_edges = (node_degrees >= min_number_edges).nonzero(as_tuple=True)[0]

    if data_name in ['Cora', 'CiteSeer']:
        selected_nodes = nodes_with_min_edges
        print(f"Number of all edges with at least {min_number_edges}: {len(selected_nodes)}")
        selected_nodes = selected_nodes[:100]  #NEW     First 100 elements
    elif data_name == 'PubMed':
        max_nodes = int(0.05 * num_nodes)
        selected_nodes = nodes_with_min_edges[:max_nodes]
        print(f"Number of all edges with at least {min_number_edges}: {len(selected_nodes)}")
        selected_nodes = selected_nodes[:100]  #NEW     First 100 elements
 

    #Creaing Queries. 
    selected_nodes_set = set(selected_nodes.tolist())
    mask = [(edge[0].item() not in selected_nodes_set) and (edge[1].item() not in selected_nodes_set) for edge in data.edge_index.t()]
    mask = torch.tensor(mask, dtype=torch.bool, device=device)
    isolated_edge_index = data.edge_index[:, mask]


    if graph_model.lower() in ["epagcl_gcn", "epagcl_sage"]:
        model.eval()
        with torch.no_grad():
            # Use the trained EPAGCL encoder on the isolated graph
            h_iso, _ = model(data.x.to(device), isolated_edge_index.to(device))
            selected_node_embeddings = h_iso[selected_nodes.to(device)]
    else:
        model.eval()
        with torch.no_grad():
            h, z = model(data.x, isolated_edge_index)
        selected_node_embeddings = h[selected_nodes]
        

    print("- Query Embedding is Generated!", flush=True)
    print("____________________________________________", flush=True)

    #____________________________________________________________________________________________________________________________________

    top_k = 20
    top_k_indice_at_20 = similarity(top_k, selected_node_embeddings, dataset_embeddings)
    found_count_20, recall_20, avg_position_20 = summary(selected_nodes, top_k_indice_at_20)

    top_k = 100
    top_k_indice_at_100 = similarity(top_k, selected_node_embeddings, dataset_embeddings)
    found_count_100, recall_100, avg_position_100 = summary(selected_nodes, top_k_indice_at_100)

    top_k = 500
    top_k_indice_at_500 = similarity(top_k, selected_node_embeddings, dataset_embeddings)
    found_count_500, recall_500, avg_position_500 = summary(selected_nodes, top_k_indice_at_500)

    top_k = 1000
    top_k_indice_at_1000 = similarity(top_k, selected_node_embeddings, dataset_embeddings)
    found_count_1000, recall_1000, avg_position_1000 = summary(selected_nodes, top_k_indice_at_1000)

    top_k = data_num_nodes
    top_k_indice_at_4000 = similarity(top_k, selected_node_embeddings, dataset_embeddings)
    found_count_4000, recall_4000, avg_position_4000 = summary(selected_nodes, top_k_indice_at_4000)


    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    torch.seed()           # re-randomize global torch RNG
    np.random.seed(None)   # reset numpy RNG to random state
    random.seed()          # reset Python RNG to random state



    #____________________________________________________________________________________________________________________________________

    return (
        dataset_embeddings, model, selected_nodes, selected_node_embeddings,
        top_k_indice_at_20, top_k_indice_at_100, top_k_indice_at_500,
        top_k_indice_at_1000, top_k_indice_at_4000,
        found_count_20, found_count_100, found_count_500, found_count_1000, found_count_4000,
        recall_20, recall_100, recall_500, recall_1000, recall_4000,
        avg_position_20, avg_position_100, avg_position_500, avg_position_1000, avg_position_4000, 
        
    )


