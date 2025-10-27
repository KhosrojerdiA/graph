import sys
import os


ep_path = '/mnt/data/khosro/Graph_v2'
sys.path.append(ep_path)

from utils.utils import *
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected
from tqdm.notebook import tqdm
import torch
import pickle
from tqdm import tqdm
from tqdm.notebook import tqdm
from torch_geometric.utils import subgraph

#____________________________________________________________________________________________________________________________

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 

#________________________________________________________________________________________________________________________________________________

def calculate_edge_performance_v2(data, graph_model, data_name, ep_save_path, embeddings_save_dir, embedding_version): 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)  # Move data to the appropriate device
    num_nodes = data.x.size(0)

    model, dataset_embeddings = load_embedding_model(data, graph_model, data_name, embeddings_save_dir, embedding_version)
    edge_performance = {}
#________________________

    #node = 12
 
    for node in tqdm(range(num_nodes)):

        #Start of Loop
        edge_performance[node] = {}

        edges = data.edge_index.t().tolist()
        selected_edges = [edge for edge in edges if edge[0] == node or edge[1] == node]

        #Isolate the node by removing all its edges (Query)
        mask = (data.edge_index[0] != node) & (data.edge_index[1] != node)
        isolated_edge_index = data.edge_index[:, mask]

        # Get query embedding
        #model.eval()
        with torch.no_grad():
            #isolated_embeddings, _ = model(data.x, isolated_edge_index)
            output = model(data.x, isolated_edge_index)
            isolated_embeddings = output[0] if isinstance(output, tuple) else output

        target_node_embedding = isolated_embeddings[node].unsqueeze(0)

        top_k_indices = similarity(num_nodes, target_node_embedding, dataset_embeddings) #20 or num_nodes

        print("______________________________________________________________", flush=True)
        print("Node: ", node)
        print("______________________________________________________________", flush=True)


        for se in selected_edges: #se is edge to be removed

            new_data = data.clone()
            #Remove se edge from the graph
            new_edges = [edge for edge in edges if edge != se]
            new_edges = torch.tensor(new_edges, dtype=torch.long, device=device).t().contiguous()
            new_data.edge_index = new_edges

            attacked_dataset_embeddings, attacked_one_node_selected_node_embeddings = attacked_embedding_v2(new_data, torch.tensor([node]), graph_model, model, device=device) 
            new_top_k_indices = similarity(num_nodes, attacked_one_node_selected_node_embeddings, attacked_dataset_embeddings)

            if node in top_k_indices:
                position_before = torch.nonzero(top_k_indices[0] == node, as_tuple=True)[0].item()
                if node in new_top_k_indices:
                    position_after = torch.nonzero(new_top_k_indices[0] == node, as_tuple=True)[0].item()
                    target_demote = position_after - position_before
                else:
                    target_demote = len(new_top_k_indices[0]) + 1 - position_before
            else:
                target_demote = 0

            avg_promoted, avg_demoted, avg_changed, avg_unchanged, total_items = compare_positions(top_k_indices, new_top_k_indices)
            position_change = compute_position_change(node, top_k_indices, new_top_k_indices)
            edge_performance[node].update({tuple(se): [top_k_indices, new_top_k_indices, [position_change, avg_promoted, avg_demoted, avg_changed, avg_unchanged, total_items]]})
            
    file_path = os.path.join(ep_save_path, f'change_position_{data_name}_{graph_model}_edge_performance_{embedding_version}.pkl') 

    with open(file_path, "wb") as file:
        pickle.dump(edge_performance, file)
    
    print(f"Edge performance Dataset saved at: {file_path}", flush=True)
    
#________________________________________________________________________________________________________________________________________________


data_name_list = ['Cora']
#['Cora', 'CiteSeer', 'PubMed']


graph_model_list = ['epagcl_gcn', 'grace', 'gca', 'greet']
#['epagcl_gcn', 'epagcl_sage', 'grace', 'gca', 'greet']

embedding_version = 'v1'

#____________________________________________________________________________________________________________________________________

ep_save_path = f"{ep_path}/edge_performance_dataset"


os.makedirs(os.path.dirname(ep_save_path), exist_ok=True)
dataset_subgraph_path = f"{ep_path}/data/pubmed_subgraph.pt"

main_seed = 3708

#________________________________________________________________________________________________________________________________________________


for data_name in data_name_list:
    for graph_model in graph_model_list:

        if graph_model.lower() == "epagcl_gcn" or graph_model.lower() == "epagcl_sage":
            embeddings_save_dir = f"{ep_path}/EPAGCL/embeddings"
        else:   
            embeddings_save_dir = f"{ep_path}/{graph_model}/embeddings"

        data = load_data(data_name, dataset_subgraph_path)
        calculate_edge_performance_v2(data, graph_model, data_name, ep_save_path, embeddings_save_dir, embedding_version) 





#edge_performance[node].update({tuple(se): [top_k_indices, new_top_k_indices, [position_change, avg_promoted, avg_demoted, avg_changed, avg_unchanged, total_items]]})

#{0: 

# {(628, 0): 
# [tensor([[   0,  628, 2326,  ...,  925,  995, 1690]], device='cuda:0'), tensor([[   0,  628, 2326,  ...,  925,  995, 1690]], device='cuda:0'), 
# [0, 69.37957051654092, 75.65886075949368, 72.38328792007266, 0.007213706041478809, 3327]], 

#(0, 628): 
# [tensor([[   0,  628, 2326,  ...,  925,  995, 1690]], device='cuda:0'), tensor([[   0, 2326,  496,  ...,  995,  358, 1690]], device='cuda:0'), 
# [0, 84.51393188854489, 80.24103468547914, 82.3220747889023, 0.0033062819356777877, 3327]]}, 
# 
# 1: ...
            

#____________________________________________________________________________________________________________________________________
