
import sys
import os

project_path = './'
sys.path.append(project_path)

import torch
import math
import numpy as np
from torch_geometric.utils import to_networkx
from torch.nn import Linear
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import VGAE, APPNP
from torch_geometric.utils import negative_sampling
from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
import math
from torch_geometric.data import Data
import pandas as pd
from torch_geometric.utils import degree
import pickle
from torch_geometric.transforms import RandomLinkSplit, NormalizeFeatures
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GCNConv, GATConv
import community.community_louvain as community
from torch_geometric.utils import to_networkx, subgraph
from community import community_louvain 
from torch_geometric.datasets import Planetoid
import random

#____________________________________________________________________________________________________________________________

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
#____________________________________________________________________________________________________________________________

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

#____________________________________________________________________________________________________________________________

def similarity(top_k, new_node_embeddings, dataset_embeddings):

    new_node_embeddings = new_node_embeddings.to(device)
    dataset_embeddings = dataset_embeddings.to(device)

    cosine_sim = torch.nn.functional.cosine_similarity(
        new_node_embeddings.unsqueeze(1),  # Add dimension for pairwise comparison
        dataset_embeddings.unsqueeze(0),   # Add dimension for pairwise comparison
        dim=-1                             # Specify the dimension for reduction
    )

    # Get the indices of the top-k most similar vectors
    top_k_indices = torch.argsort(cosine_sim, dim=1, descending=True)[:, :top_k]

    return top_k_indices




#____________________________________________________________________________________________________________________________

 
def similarity_1D(data_num_nodes, one_node_selected_node_embedding, dataset_embeddings): #similarity_1D(top_k, query_embedding, attacked_dataset_embeddings)

    #torch.cuda.empty_cache()
    
    #one_node_selected_node_embedding = one_node_selected_node_embedding.to(device)
    one_node_selected_node_embedding = one_node_selected_node_embedding
    #dataset_embeddings = dataset_embeddings.to(device)

    if one_node_selected_node_embedding.dim() == 1:
        one_node_selected_node_embedding = one_node_selected_node_embedding.unsqueeze(0)

    cosine_sim = torch.nn.functional.cosine_similarity(
        one_node_selected_node_embedding.unsqueeze(1),
        dataset_embeddings.unsqueeze(0),
        dim=-1
    )

    top_k_indices = torch.argsort(cosine_sim, dim=1, descending=True)[:, :data_num_nodes]

    # Ensure the result is dense
    top_k_indices = top_k_indices.to_dense() if top_k_indices.is_sparse else top_k_indices

    return top_k_indices



#____________________________________________________________________________________________________________________________

def evaluation(selected_nodes, top_k_indices, top_k):
    precision_list = []
    recall_list = []
    average_precisions = []
    unique_nodes = set()
    ndcg_list = []
    retrieval_positions = []

    for i, query_node in enumerate(selected_nodes):
        predicted_nodes = top_k_indices[i]
        relevance = 1 if query_node in predicted_nodes else 0

        precision = relevance / top_k
        recall = relevance
        average_precision = precision

        dcg = 0.0
        for rank, predicted_node in enumerate(predicted_nodes):
            if predicted_node == query_node:
                dcg += 1 / math.log2(rank + 2)
                retrieval_positions.append(rank + 1)

        idcg = 1 / math.log2(1 + 1) if relevance else 1.0
        ndcg = dcg / idcg if idcg > 0 else 0.0

        precision_list.append(precision)
        recall_list.append(recall)
        average_precisions.append(average_precision)
        ndcg_list.append(ndcg)
        unique_nodes.update(predicted_nodes)

    MAP_score = torch.mean(torch.tensor(average_precisions, dtype=torch.float32, device=device))
    diversity = len(unique_nodes) / (len(selected_nodes) * top_k)
    mean_precision = torch.mean(torch.tensor(precision_list, dtype=torch.float32, device=device))
    mean_recall = torch.mean(torch.tensor(recall_list, dtype=torch.float32, device=device))
    mean_ndcg = torch.mean(torch.tensor(ndcg_list, dtype=torch.float32, device=device))

    avg_retrieval_position = sum(retrieval_positions) / len(retrieval_positions) if retrieval_positions else 0.0

    print(f"\nMean Precision: {mean_precision:.4f}")
    print(f"Mean Recall: {mean_recall:.4f}")
    print(f"Mean Average Precision (MAP): {MAP_score:.4f}")
    print(f"Mean NDCG: {mean_ndcg:.4f}")
    print(f"Average Retrieval Position: {avg_retrieval_position:.4f}")
#____________________________________________________________________________________________________________________________

def query_result(selected_nodes, top_k_indice):
    for i, node in enumerate(selected_nodes):
        top_k_values = top_k_indice[i].tolist()
        if node.item() in top_k_values:
            position = top_k_values.index(node.item())
            print(f"Node {node.item()}: {top_k_values} - Found at position {position}")
        else:
            print(f"Node {node.item()}: {top_k_values} - Not found")
#____________________________________________________________________________________________________________________________

def compare_original_vs_updated(original_data, updated_data, budget):
    original_num_nodes = original_data.num_nodes
    updated_num_nodes = updated_data.num_nodes

    original_num_edges = original_data.edge_index.size(1)
    updated_num_edges = updated_data.edge_index.size(1)
    print("____________________________________________")
    print("- Compare")
    print("____________________________________________")
    #print("Number of nodes (Original):", original_num_nodes)
    #print("Number of nodes (Updated):", updated_num_nodes)
    print("Number of edges (Original):", original_num_edges)
    print("Number of edges (Updated):", updated_num_edges)
    print("\nDifference in the number of edges:", original_num_edges - updated_num_edges)
    print("____________________________________________")

    if original_num_edges - updated_num_edges != budget:
        print("\nError")
    

    # Find removed edges
    original_edges = original_data.edge_index.t().tolist()
    updated_edges = updated_data.edge_index.t().tolist()

    # Convert to sets for comparison
    original_edges_set = set(map(tuple, original_edges))
    updated_edges_set = set(map(tuple, updated_edges))

    removed_edges = original_edges_set - updated_edges_set

    if removed_edges:
        print("\nRemoved edges:")
        for edge in removed_edges:
            print(edge)
    else:
        print("\nNo edges have been removed.")

    print("____________________________________________")
#____________________________________________________________________________________________________________________________
     
def get_edges_for_node(data, node_id):
    edge_index = data.edge_index
    edges = edge_index[:, (edge_index[0] == node_id) | (edge_index[1] == node_id)]
    return edges
#____________________________________________________________________________________________________________________________

def check_node_positions(at_5_dict):
    found_count = 0
    not_found_count = 0
    total_nodes = len(at_5_dict)
    found_positions = []

    for node_id, tensor_vals in at_5_dict.items():
        tensor_list = tensor_vals[0].tolist()
        if node_id in tensor_list:
            found_count += 1
            found_positions.append(tensor_list.index(node_id) + 1)
        else:
            not_found_count += 1

    avg_found_position = sum(found_positions) / len(found_positions) if found_positions else 0
    return {
        "total_nodes": total_nodes,
        "found_count": found_count,
        "not_found_count": not_found_count,
        "avg_found_position": avg_found_position
    }
#____________________________________________________________________________________________________________________________

def count_mutual_items(origin, attacked):
    mutual_counts = []
    for i in range(len(origin)):
        set1 = set(origin[i])
        set2 = set(attacked[i])
        mutual_items = set1.intersection(set2)
        mutual_counts.append(len(mutual_items))
    return mutual_counts
#____________________________________________________________________________________________________________________________

#Target 5
#R {1,2,3,4,5}
#A {5,6,1,3}, 2

def compare_positions(top_k_indice_at_k_origin, attacked_at_num_node_dict_tensor_only):  

    total_promoted = 0
    total_demoted = 0
    total_unchanged = 0
    total_items_in_both_lists = 0
    total_items_in_promoted_list = 0
    total_items_in_demoted_list = 0

    for i in range(len(top_k_indice_at_k_origin)):
        list1 = top_k_indice_at_k_origin[i]
        list2 = attacked_at_num_node_dict_tensor_only[i]

        for idx, item in enumerate(list1):
            # Check if item is in list2 and find its index using PyTorch
            matches = (list2 == item).nonzero(as_tuple=True)[0]
            if len(matches) > 0:
                total_items_in_both_lists += 1
                index_in_list2 = matches.item()

                if index_in_list2 < idx:
                    total_items_in_promoted_list += 1
                    total_promoted += (idx - index_in_list2)
                elif index_in_list2 > idx:
                    total_items_in_demoted_list += 1
                    total_demoted += (index_in_list2 - idx)
                else:
                    total_unchanged += 1

    if total_items_in_both_lists > 0:
        avg_promoted = total_promoted / total_items_in_promoted_list if total_items_in_promoted_list != 0 else 0
        avg_demoted = total_demoted / total_items_in_demoted_list if total_items_in_demoted_list != 0 else 0
        avg_changed = (total_promoted + total_demoted) / (total_items_in_promoted_list + total_items_in_demoted_list) if (total_items_in_promoted_list + total_items_in_demoted_list) != 0 else 0
        avg_unchanged = total_unchanged / total_items_in_both_lists
    else:
        avg_promoted = avg_demoted = avg_changed = avg_unchanged = 0

    return avg_promoted, avg_demoted, avg_changed, avg_unchanged, total_items_in_both_lists
#____________________________________________________________________________________________________________________________
 
def summary(selected_nodes, top_k_indice):
    found_count = 0
    total_position = 0
    not_found_count = 0

    for i, node in enumerate(selected_nodes):
        top_k_values = top_k_indice[i]

        # Check if node is in top_k_values using PyTorch
        matches = (top_k_values == node).nonzero(as_tuple=True)[0]
        if len(matches) > 0:
            position = matches.item()
            found_count += 1
            total_position += (position + 1)  # Position is 0-indexed, so add 1 for 1-based indexing
        else:
            not_found_count += 1

    avg_position = total_position / found_count if found_count > 0 else 0
    recall = found_count / (found_count + not_found_count)

    return found_count, recall, avg_position

#____________________________________________________________________________________________________________________________

def viz_selected_nodes_edges(data, selected_nodes):

    viz_edge_index = data.edge_index 
    viz_connected_nodes = {node.item(): [] for node in selected_nodes}
    for i in range(viz_edge_index.shape[1]):
        src, dst = viz_edge_index[0, i].item(), viz_edge_index[1, i].item()
        
        if src in viz_connected_nodes:
            viz_connected_nodes[src].append(dst)
        if dst in viz_connected_nodes:  # Because the graph is undirected
            viz_connected_nodes[dst].append(src)

    # Print formatted output
    for node, neighbors in viz_connected_nodes.items():
        print(f"Node {node}: ({', '.join(map(str, neighbors))})")

#____________________________________________________________________________________________________________________________

def subraph_pubmed(data, subgraph_nodes, subgraph_edges):

    G = to_networkx(data, to_undirected=True)

    # Step 1: Apply Louvain Community Detection
    partition = community.best_partition(G)

    # Step 2: Count Nodes in Each Community
    community_counts = {}
    for node, comm in partition.items():
        community_counts[comm] = community_counts.get(comm, 0) + 1

    # Step 3: Select Communities Until We Reach 4000 Nodes
    selected_nodes = []
    selected_communities = sorted(community_counts.keys(), key=lambda k: -community_counts[k])  # Sort largest first
    
    for comm in selected_communities:
        nodes_in_comm = [node for node, c in partition.items() if c == comm]
        selected_nodes.extend(nodes_in_comm)
        if len(selected_nodes) >= subgraph_nodes:
            break

    # Trim to exactly `subgraph_nodes`
    selected_nodes = selected_nodes[:subgraph_nodes]
    selected_nodes = torch.tensor(selected_nodes, dtype=torch.long)

    # Step 4: Extract Induced Subgraph
    sub_edge_index, _ = subgraph(selected_nodes, data.edge_index, relabel_nodes=True)

    # Step 5: Ensure Exactly 12000 Edges
    if sub_edge_index.shape[1] > subgraph_edges:
        selected_edges = torch.randperm(sub_edge_index.shape[1])[:subgraph_edges]
        sub_edge_index = sub_edge_index[:, selected_edges]

    # Step 6: Update Features, Labels, and Masks
    new_x = data.x[selected_nodes]
    new_y = data.y[selected_nodes]

    new_train_mask = data.train_mask[selected_nodes]
    new_val_mask = data.val_mask[selected_nodes]
    new_test_mask = data.test_mask[selected_nodes]

    # Step 7: Create the new subgraph data object
    sub_data = data.clone()
    sub_data.x = new_x
    sub_data.y = new_y
    sub_data.edge_index = sub_edge_index
    sub_data.train_mask = new_train_mask
    sub_data.val_mask = new_val_mask
    sub_data.test_mask = new_test_mask

    #print(f"Subgraph Created: {sub_data.x.shape[0]} nodes, {sub_data.edge_index.shape[1]} edges")
    return sub_data
#____________________________________________________________________________________________________________________________

def attacked_found_recall_avg_position(selected_nodes, top_k_indice_at_k, attacked_at_num_node_dict_tensor_only, n):

    found_count = 0
    not_found_count = 0
    total_position = 0
    position_count = 0
    

    recall_count = 0  # Counter for recall calculation

    for i, node in enumerate(selected_nodes):
        top_k_values = attacked_at_num_node_dict_tensor_only[i]

        # Recall logic: check if the node is in the first n elements
        recall_matches = (top_k_values[:n] == node).nonzero(as_tuple=True)[0]
        if len(recall_matches) > 0:
            recall_count += 1

        # Check if node exists in top_k_indice_at_k
        if node in top_k_indice_at_k[i]:
            # Check if node is within the first n items of attacked_at_num_node_dict_tensor_only
            matches = (top_k_values[:n] == node).nonzero(as_tuple=True)[0]
            if len(matches) > 0:
                found_count += 1
            else:
                not_found_count += 1

            # Calculate position of node in attacked_at_num_node_dict_tensor_only
            all_matches = (top_k_values == node).nonzero(as_tuple=True)[0]
            if len(all_matches) > 0:
                total_position += all_matches.float().mean().item() + 1  # Convert to 1-based index
                position_count += 1
    

    attacked_recall = recall_count / len(selected_nodes) if len(selected_nodes) > 0 else 0  # % of selected nodes that appeared in attacked_top_k (attacked recall)
    attacked_retrieval_node_found_count = found_count #if selected node is in retrieval and attack
    attacked_retrieval_node_avg_position = total_position / position_count if position_count > 0 else 0 #what is the avg position of selected node position that appeared in top k
    
    
    return attacked_recall, attacked_retrieval_node_found_count, attacked_retrieval_node_avg_position 
           

#____________________________________________________________________________________________________________________________

def safe_convert_to_float(value):
    return value.item() if isinstance(value, torch.Tensor) else value

#____________________________________________________________________________________________________________________________

def stat_result(top_k_indice_at_k, attacked_at_num_node_dict, selected_nodes): # top_k_indice_at_20, attacked_at_num_node_dict, selected_nodes
    
    # Combine tensors from the attacked_at_num_node_dict
    attacked_at_num_node_dict_tensor_only = torch.cat(list(attacked_at_num_node_dict.values()), dim=0).to('cuda')


    found_count, recall, avg_position = summary(selected_nodes, attacked_at_num_node_dict_tensor_only)

    top_k_indice_at_k_origin = top_k_indice_at_k.clone().detach().to('cuda')
    avg_promoted, avg_demoted, avg_changed, avg_unchanged, total_items = compare_positions(top_k_indice_at_k_origin, attacked_at_num_node_dict_tensor_only)


    return (found_count, recall, avg_position,
            safe_convert_to_float(avg_promoted),
            safe_convert_to_float(avg_demoted),
            safe_convert_to_float(avg_changed),
            100 * safe_convert_to_float(avg_unchanged))


#____________________________________________________________________________________________________________________________


def aggregate_result(top_k_indice_at_k, attacked_at_num_node_dict, selected_nodes, n): # top_k_indice_at_20, attacked_at_num_node_dict, selected_nodes
    
    # Combine tensors from the attacked_at_num_node_dict
    attacked_at_num_node_dict_tensor_only = torch.cat(list(attacked_at_num_node_dict.values()), dim=0)


    attacked_recall, attacked_retrieval_node_found_count, attacked_retrieval_node_avg_position = attacked_found_recall_avg_position(selected_nodes, top_k_indice_at_k, attacked_at_num_node_dict_tensor_only, n)

    top_k_indice_at_k_origin = top_k_indice_at_k.clone().detach()
    attacked_avg_promoted, attacked_avg_demoted, attacked_avg_changed, atacked_avg_unchanged, total_items = compare_positions(top_k_indice_at_k_origin, attacked_at_num_node_dict_tensor_only)


    return (attacked_recall, attacked_retrieval_node_found_count, attacked_retrieval_node_avg_position,
            safe_convert_to_float(attacked_avg_promoted),
            safe_convert_to_float(attacked_avg_demoted),
            safe_convert_to_float(attacked_avg_changed),
            100 * safe_convert_to_float(atacked_avg_unchanged))

#____________________________________________________________________________________________________________________________


def get_avg_position(top_k_tensor, attacked_dict):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # If the dictionary is empty, return NaN
    if not attacked_dict:
        return float('nan')

    # Concatenate all values from attacked_dict into a single tensor and move to device
    attacked_tensor = torch.cat(list(attacked_dict.values()), dim=0).to(device)

    # Move top_k_tensor to device and flatten it for element-wise comparison
    top_k_tensor = top_k_tensor.to(device).flatten()

    positions = []

    for value in top_k_tensor:
        # Find the position of the value in the attacked_tensor
        indices = torch.where(attacked_tensor == value)[0]  # Get positions

        if indices.numel() > 0:
            positions.append(indices[0].item())  # Store the first found position

    # If no positions were found, return NaN
    return sum(positions) / len(positions) if positions else float('nan')


#____________________________________________________________________________________________________________________________

def stat_result_per_node_retrieval(after_attack_dict, selected_nodes):

    attacked_top_k_indice_at_2631_tensor_only = torch.cat(list(after_attack_dict.values()), dim=0).to('cuda')
    found_count, recall, avg_position = summary(selected_nodes, attacked_top_k_indice_at_2631_tensor_only)

    return (found_count, recall, avg_position)           

 
#____________________________________________________________________________________________________________________________

def stat_restult_non_loop(top_k, selected_nodes, selected_node_embeddings_attacked, degree_embedding, top_k_indice_at_20):
    # Calculate the top-k indices for the attacked degree embeddings
    degree_attacked_top_k_indice_at_20 = similarity(top_k, selected_node_embeddings_attacked, degree_embedding)

    # Call the summary function (assuming it supports GPU tensors)
    found_count, recall, avg_position = summary(selected_nodes, degree_attacked_top_k_indice_at_20)

    # Ensure that top_k_indice_at_20 and degree_attacked_top_k_indice_at_20 are processed correctly
    origin = top_k_indice_at_20.clone().detach().to('cuda')
    attacked =  degree_attacked_top_k_indice_at_20.to('cuda')

    # Perform comparison
    avg_promoted, avg_demoted, avg_changed, avg_unchanged, total_items = compare_positions(origin, attacked)

    # Safe conversion to float if results are PyTorch tensors
    def safe_convert_to_float(value):
        return value.item() if isinstance(value, torch.Tensor) else value

    return (found_count, recall, avg_position,
            safe_convert_to_float(avg_promoted),
            safe_convert_to_float(avg_demoted),
            safe_convert_to_float(avg_changed),
            100 * safe_convert_to_float(avg_unchanged))
#____________________________________________________________________________________________________________________________________________

def custom_link_split(data, selected_nodes, val_ratio=0.1):
    # Ensure selected_nodes is a tensor and on the correct 
    selected_nodes = selected_nodes.to(data.edge_index.device)

    # Get all edges as a list of tuples
    edge_index = data.edge_index
    edges = edge_index.t().tolist()

    # Get edges involving the selected nodes
    selected_edges = [edge for edge in edges if edge[0] in selected_nodes or edge[1] in selected_nodes]

    # Remaining edges for training that do not involve selected nodes
    remaining_edges = [edge for edge in edges if edge not in selected_edges]

    # Convert edge lists to tensors
    test_edge_index = torch.tensor(selected_edges, dtype=torch.long, device=data.edge_index.device).t().contiguous()
    train_edge_index = torch.tensor(remaining_edges, dtype=torch.long, device=data.edge_index.device).t().contiguous()

    # Create train data and apply RandomLinkSplit for validation
    train_data = Data(edge_index=train_edge_index, x=data.x)

    # Apply the RandomLinkSplit to generate a validation split
    transform = RandomLinkSplit(num_test=val_ratio, num_val=0, is_undirected=True, add_negative_train_samples=False)
    train_data, _, val_data = transform(train_data)

    # Create test data
    test_data = Data(edge_index=test_edge_index, x=data.x)

    return train_data, val_data, test_data

#____________________________________________________________________________________________________________________________________________


def retrieval_store_to_excel(data_name, graph_model, min_number_edges, model_name, retrieval_found_count, retrieval_recall, retrieval_avg_position, result_path):

    with pd.ExcelWriter(f"{result_path}/{data_name}_{model_name}_{graph_model}_{min_number_edges}_retrieval.xlsx") as writer:

        data_list = [
            (retrieval_found_count, "retrieval_found_count"),
            (retrieval_recall, "retrieval_recall"),
            (retrieval_avg_position, "retrieval_avg_position"),
        ]


        for data, sheet_name in data_list:
            df = pd.DataFrame(data, columns=[sheet_name])
            runs = len(df) // 5  # Calculate the number of runs
            reshaped_data = {
                'Run Number': range(1, runs + 1),
                '@20': df[sheet_name][::5].reset_index(drop=True),
                '@100': df[sheet_name][1::5].reset_index(drop=True),
                '@500': df[sheet_name][2::5].reset_index(drop=True),
                '@1000': df[sheet_name][3::5].reset_index(drop=True),
                '@4000': df[sheet_name][4::5].reset_index(drop=True),
            }
            result_df = pd.DataFrame(reshaped_data)
             

            mean_row = result_df.mean().to_frame().T
            std_row = result_df.std().to_frame().T
            mean_row['Run Number'] = 'Mean'
            std_row['Run Number'] = 'Std'


            result_df = pd.concat([result_df, mean_row, std_row], ignore_index=True)
            
            result_df.to_excel(writer, sheet_name=sheet_name, index=False)


#____________________________________________________________________________________________________________________________

def attacked_store_to_excel(data_name, graph_model, min_number_edges, model_name, budget, 
                            attacked_found_count, attacked_recall, attacked_avg_position, 
                            attacked_avg_promoted, attacked_avg_demoted, attacked_avg_changed, attacked_avg_unchanged, 
                            result_path):

    with pd.ExcelWriter(f"{result_path}/{data_name}_{model_name}_{graph_model}_{min_number_edges}_attacked.xlsx") as writer:

        data_list = [
            (attacked_found_count, "attacked_found_count"),
            (attacked_recall, "attacked_recall"),
            (attacked_avg_position, "attacked_avg_position"),
            (attacked_avg_promoted, "attacked_avg_promoted"),
            (attacked_avg_demoted, "attacked_avg_demoted"),
            (attacked_avg_changed, "attacked_avg_changed"),
            (attacked_avg_unchanged, "attacked_avg_unchanged"),
        ]


        for data, sheet_name in data_list:
            df = pd.DataFrame(data, columns=[sheet_name])
            runs = len(df) // 5  
            reshaped_data = {
                'Run Number': range(1, runs + 1),
                '@20': df[sheet_name][::5].reset_index(drop=True),
                '@100': df[sheet_name][1::5].reset_index(drop=True),
                '@500': df[sheet_name][2::5].reset_index(drop=True),
                '@1000': df[sheet_name][3::5].reset_index(drop=True),
                '@4000': df[sheet_name][4::5].reset_index(drop=True),
            }
            result_df = pd.DataFrame(reshaped_data)
            
  
            mean_row = result_df.mean().to_frame().T
            std_row = result_df.std().to_frame().T
            mean_row['Run Number'] = 'Mean'
            std_row['Run Number'] = 'Std'


            result_df = pd.concat([result_df, mean_row, std_row], ignore_index=True)
            

            result_df.to_excel(writer, sheet_name=sheet_name, index=False)


#____________________________________________________________________________________________________________________________


def store_to_excel(data_name, graph_model, min_number_edges, model_name, budget,
                                   retrieval_found_count, retrieval_recall, retrieval_avg_position, 
                                   attacked_recall, attacked_retrieval_node_found_count, attacked_retrieval_node_avg_position,
                                   attacked_avg_promoted, attacked_avg_demoted, 
                                   attacked_avg_changed, attacked_avg_unchanged, duration_per_run, result_path, promotion_mode, text):
    
    # Define the file path for the specific data_name
    file_path = f"{result_path}/results/{data_name}_{promotion_mode}_{text}_report.xlsx"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Prepare data for all tabs
    data_list = [
        (retrieval_found_count, "retrieval_found_count"),
        (retrieval_recall, "retrieval_recall"),
        (retrieval_avg_position, "retrieval_avg_position"),
        (attacked_retrieval_node_found_count, "attacked_retrieval_node_found_count"),
        (attacked_retrieval_node_avg_position, "attacked_retrieval_node_avg_pos"),
        (attacked_recall, "attacked_recall"),
        (attacked_avg_promoted, "attacked_avg_promoted"),
        (attacked_avg_demoted, "attacked_avg_demoted"),
        (attacked_avg_changed, "attacked_avg_changed"),
        (attacked_avg_unchanged, "attacked_avg_unchanged"),
        (duration_per_run, "duration_per_run"),
                ]

    # Check if the file already exists
    if os.path.exists(file_path):
        excel_writer = pd.ExcelWriter(file_path, engine="openpyxl", mode="a", if_sheet_exists="overlay")
    else:
        excel_writer = pd.ExcelWriter(file_path, engine="openpyxl")

    for data, sheet_name in data_list:
        if not data:
            continue  # Skip if the data is empty

        # Create DataFrame from the data
        df = pd.DataFrame(data, columns=[sheet_name])
        reshaped_data = {
            'Run Identifier': f"{data_name}_{graph_model}_{min_number_edges}_{budget}_{model_name}_{promotion_mode}", 
            '@20': df[sheet_name][::5].reset_index(drop=True),
            '@100': df[sheet_name][1::5].reset_index(drop=True),
            '@500': df[sheet_name][2::5].reset_index(drop=True),
            '@1000': df[sheet_name][3::5].reset_index(drop=True),
            '@4000': df[sheet_name][4::5].reset_index(drop=True),
        }
        result_df = pd.DataFrame(reshaped_data)

        # Calculate mean and standard deviation
        mean_row = result_df.mean(numeric_only=True).to_frame().T
        std_row = result_df.std(numeric_only=True).to_frame().T
        mean_row['Run Identifier'] = 'Mean'
        std_row['Run Identifier'] = 'Std'

        # Combine original data with mean and std rows
        result_df = pd.concat([result_df, mean_row, std_row], ignore_index=True)

        # Write to the appropriate sheet in the Excel file
        if sheet_name in excel_writer.sheets:
            # Read existing sheet if it exists
            existing_df = pd.read_excel(file_path, sheet_name=sheet_name)
            updated_df = pd.concat([existing_df, result_df], ignore_index=True)
            updated_df.to_excel(excel_writer, sheet_name=sheet_name, index=False)
        else:
            result_df.to_excel(excel_writer, sheet_name=sheet_name, index=False)

    # Save and close the Excel writer
    excel_writer.close()
#____________________________________________________________________________________________________________________________

class GraphPatcherLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.2):
        super(GraphPatcherLayer, self).__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)  # 🔹 GAT with multiple heads
        self.patch_weight = Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)  # 🔹 Normalize features

    def forward(self, x, edge_index):
        # 🔹 **Apply feature augmentation only during training**
        if self.training:
            x = self.augment_features(x)

        h1 = self.conv1(x, edge_index).relu()
        h2 = self.conv2(h1, edge_index).relu()
        h_patch = self.patch_weight(h1)  

        # 🔹 **Residual Connection & Normalization**
        h = self.norm(h1 + h2 + h_patch)  
        return self.dropout(h)

    def augment_features(self, x):
        """GraphPatcher's TTA mechanism"""
        noise = torch.randn_like(x) * 0.1  
        mask = torch.rand(x.shape[0], 1, device=x.device) > 0.9  
        return x + noise * mask  
    

#____________________________________________________________________________________________________________________________
 

def attacked_embedding_v2(updated_data, one_node_selected_nodes, graph_model, dataset_model, device="cuda"): 

    """
    Generate attacked embeddings without retraining the model.

    Args:
        updated_data (torch_geometric.data.Data): Graph after edge removals.
        one_node_selected_nodes (torch.Tensor): Node(s) to isolate and compute query embeddings for.
        dataset_model (torch.nn.Module): Pre-trained model (already trained before attack).
        device (str): 'cuda' or 'cpu'.
    
    Returns:
        attacked_dataset_embeddings (torch.Tensor): Embeddings for all nodes in the attacked graph.
        one_node_selected_node_embeddings (torch.Tensor): Embeddings for selected nodes.
    """

    if graph_model == 'gat2':
        torch.use_deterministic_algorithms(True, warn_only=True)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    updated_data = updated_data.to(device)
    dataset_model = dataset_model.to(device)
    dataset_model.eval()

    # Full attacked graph
    with torch.no_grad():
        #attacked_dataset_embeddings, _ = dataset_model(updated_data.x, updated_data.edge_index)
        output = dataset_model(updated_data.x, updated_data.edge_index)
        attacked_dataset_embeddings = output[0] if isinstance(output, tuple) else output

    print("- Attacked Embedding is Generated!", flush=True)
    print("____________________________________________", flush=True)

    # --- Isolate selected node(s) by removing edges connected to them
    selected_nodes_set = set(one_node_selected_nodes.tolist())
    mask = [(e[0].item() not in selected_nodes_set) and (e[1].item() not in selected_nodes_set) 
            for e in updated_data.edge_index.t()]
    mask = torch.tensor(mask, dtype=torch.bool, device=device)
    isolated_edge_index = updated_data.edge_index[:, mask]

    # --- Embeddings for isolated nodes (query nodes)
    with torch.no_grad():
        #isolated_embeddings, _ = dataset_model(updated_data.x, isolated_edge_index)
        output = dataset_model(updated_data.x, isolated_edge_index)
        isolated_embeddings = output[0] if isinstance(output, tuple) else output


    one_node_selected_node_embeddings = isolated_embeddings[one_node_selected_nodes]

    print("- Attacked Query Embedding is Generated!", flush=True)
    print("____________________________________________", flush=True)

    return attacked_dataset_embeddings, one_node_selected_node_embeddings



''' 
#This is from edge_performance.py
def attacked_embedding_v2(updated_data, one_node_selected_nodes, graph_model, dataset_model, device="cuda"):
    """
    Generate attacked embeddings without retraining the model.

    Args:
        updated_data (torch_geometric.data.Data): Graph after edge removals.
        one_node_selected_nodes (torch.Tensor): Node(s) to isolate and compute query embeddings for.
        dataset_model (torch.nn.Module): Pre-trained model (already trained before attack).
        device (str): 'cuda' or 'cpu'.
    
    Returns:
        attacked_dataset_embeddings (torch.Tensor): Embeddings for all nodes in the attacked graph.
        one_node_selected_node_embeddings (torch.Tensor): Embeddings for selected nodes.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    updated_data = updated_data.to(device)
    dataset_model = dataset_model.to(device)
    dataset_model.eval()

    # --- Full graph embeddings (after attack)
    with torch.no_grad():
        attacked_dataset_embeddings, _ = dataset_model(updated_data.x, updated_data.edge_index)

    print("- Attacked Embedding is Generated!", flush=True)
    print("____________________________________________", flush=True)

    # --- Isolate selected node(s) by removing edges connected to them
    selected_nodes_set = set(one_node_selected_nodes.tolist())
    mask = [(edge[0].item() not in selected_nodes_set) and (edge[1].item() not in selected_nodes_set)
            for edge in updated_data.edge_index.t()]
    mask = torch.tensor(mask, dtype=torch.bool, device=device)
    isolated_edge_index = updated_data.edge_index[:, mask]

    # --- Embeddings for isolated nodes (query nodes)
    with torch.no_grad():
        isolated_embeddings, _ = dataset_model(updated_data.x, isolated_edge_index)
    one_node_selected_node_embeddings = isolated_embeddings[one_node_selected_nodes]

    print("- Attacked Query Embedding is Generated!", flush=True)
    print("____________________________________________", flush=True)

    return attacked_dataset_embeddings, one_node_selected_node_embeddings

'''


#____________________________________________________________________________________________________________________________

def attacked_return(selected_nodes, selected_node_embeddings_attacked, attacked_dataset_embeddings, 
                    top_k_indice_at_20, top_k_indice_at_100, top_k_indice_at_500, top_k_indice_at_1000, top_k_indice_at_4000):

    found_count_20, recall_20, avg_position_20, avg_promoted_20, avg_demoted_20, avg_changed_20, avg_unchanged_20 = stat_restult_non_loop(20, selected_nodes, selected_node_embeddings_attacked, attacked_dataset_embeddings, top_k_indice_at_20)
    found_count_100, recall_100, avg_position_100, avg_promoted_100, avg_demoted_100, avg_changed_100, avg_unchanged_100 = stat_restult_non_loop(100, selected_nodes, selected_node_embeddings_attacked, attacked_dataset_embeddings, top_k_indice_at_100)
    found_count_500, recall_500, avg_position_500, avg_promoted_500, avg_demoted_500, avg_changed_500, avg_unchanged_500 = stat_restult_non_loop(500, selected_nodes, selected_node_embeddings_attacked, attacked_dataset_embeddings, top_k_indice_at_500)
    found_count_1000, recall_1000, avg_position_1000, avg_promoted_1000, avg_demoted_1000, avg_changed_1000, avg_unchanged_1000 = stat_restult_non_loop(1000, selected_nodes, selected_node_embeddings_attacked, attacked_dataset_embeddings, top_k_indice_at_1000)
    found_count_4000, recall_4000, avg_position_4000, avg_promoted_4000, avg_demoted_4000, avg_changed_4000, avg_unchanged_4000 = stat_restult_non_loop(4000, selected_nodes, selected_node_embeddings_attacked, attacked_dataset_embeddings, top_k_indice_at_4000)

    print("- Attacked Result Returned!")
    print("____________________________________________")

    return (found_count_20, found_count_100, found_count_500, found_count_1000, found_count_4000,
            recall_20, recall_100, recall_500, recall_1000, recall_4000,
            avg_position_20, avg_position_100, avg_position_500, avg_position_1000, avg_position_4000,
            avg_promoted_20, avg_promoted_100, avg_promoted_500, avg_promoted_1000, avg_promoted_4000,
            avg_demoted_20, avg_demoted_100, avg_demoted_500, avg_demoted_1000, avg_demoted_4000,
            avg_changed_20, avg_changed_100, avg_changed_500, avg_changed_1000, avg_changed_4000,
            avg_unchanged_20, avg_unchanged_100, avg_unchanged_500, avg_unchanged_1000, avg_unchanged_4000
          )
#____________________________________________________________________________________________________________________________
 
def per_node_attacked_return(data_num_nodes, attacked_dataset_embeddings, one_node_selected_node_embedding, node_id, attacked_at_num_node_dict):

    #top_k = 20
    #attacked_top_k_indice_at_20_per_node = similarity_1D(top_k, one_node_selected_node_embedding, attacked_dataset_embeddings)
    #attacked_at_20_dict[node_id] = attacked_top_k_indice_at_20_per_node

    #top_k = 4000
    attacked_top_num_nodes_indice_per_node = similarity_1D(data_num_nodes, one_node_selected_node_embedding, attacked_dataset_embeddings)
    attacked_at_num_node_dict[node_id] = attacked_top_num_nodes_indice_per_node

    #print("- Attacked Dictionary Generated!")
    #print("____________________________________________")

    return attacked_at_num_node_dict

#____________________________________________________________________________________________________________________________

def per_node_dictionary_return(selected_nodes, top_k_indice_at_20, top_k_indice_at_100, top_k_indice_at_500, top_k_indice_at_1000, top_k_indice_at_4000, attacked_at_num_node_dict, data_num_nodes):

    attacked_recall_20, attacked_retrieval_node_found_count_20, attacked_retrieval_node_avg_position_20, attacked_avg_promoted_20, attacked_avg_demoted_20, attacked_avg_changed_20, attacked_avg_unchanged_20 = aggregate_result(top_k_indice_at_20, attacked_at_num_node_dict, selected_nodes, 20)
    attacked_recall_100, attacked_retrieval_node_found_count_100, attacked_retrieval_node_avg_position_100, attacked_avg_promoted_100, attacked_avg_demoted_100, attacked_avg_changed_100, attacked_avg_unchanged_100 = aggregate_result(top_k_indice_at_100, attacked_at_num_node_dict, selected_nodes, 100)
    attacked_recall_500, attacked_retrieval_node_found_count_500, attacked_retrieval_node_avg_position_500, attacked_avg_promoted_500, attacked_avg_demoted_500, attacked_avg_changed_500, attacked_avg_unchanged_500 = aggregate_result(top_k_indice_at_500, attacked_at_num_node_dict, selected_nodes, 500)
    attacked_recall_1000, attacked_retrieval_node_found_count_1000, attacked_retrieval_node_avg_position_1000, attacked_avg_promoted_1000, attacked_avg_demoted_1000, attacked_avg_changed_1000, attacked_avg_unchanged_1000 = aggregate_result(top_k_indice_at_1000, attacked_at_num_node_dict, selected_nodes, 1000)
    attacked_recall_4000, attacked_retrieval_node_found_count_4000, attacked_retrieval_node_avg_position_4000, attacked_avg_promoted_4000, attacked_avg_demoted_4000, attacked_avg_changed_4000, attacked_avg_unchanged_4000 = aggregate_result(top_k_indice_at_4000, attacked_at_num_node_dict, selected_nodes, data_num_nodes)
    

    print(" - Attacked Result Returned!")
    print("____________________________________________")

    return (
            attacked_recall_20, attacked_retrieval_node_found_count_20, attacked_retrieval_node_avg_position_20, attacked_avg_promoted_20, attacked_avg_demoted_20, attacked_avg_changed_20, attacked_avg_unchanged_20,
            attacked_recall_100, attacked_retrieval_node_found_count_100, attacked_retrieval_node_avg_position_100, attacked_avg_promoted_100, attacked_avg_demoted_100, attacked_avg_changed_100, attacked_avg_unchanged_100,
            attacked_recall_500, attacked_retrieval_node_found_count_500, attacked_retrieval_node_avg_position_500, attacked_avg_promoted_500, attacked_avg_demoted_500, attacked_avg_changed_500, attacked_avg_unchanged_500,
            attacked_recall_1000, attacked_retrieval_node_found_count_1000, attacked_retrieval_node_avg_position_1000, attacked_avg_promoted_1000, attacked_avg_demoted_1000, attacked_avg_changed_1000, attacked_avg_unchanged_1000,
            attacked_recall_4000, attacked_retrieval_node_found_count_4000, attacked_retrieval_node_avg_position_4000, attacked_avg_promoted_4000, attacked_avg_demoted_4000, attacked_avg_changed_4000, attacked_avg_unchanged_4000

        )

#____________________________________________________________________________________________________________________________

def create_neg_list(test_data, selected_nodes, neg_pred):

    total_edges_to_remove = len(selected_nodes)
    
    selected_edges_neg_pred = []

    # Step 1: Collect all edges involving selected_nodes and their prediction scores
    for i, (src, dst) in enumerate(zip(test_data.edge_index[0], test_data.edge_index[1])):
        if src in selected_nodes or dst in selected_nodes:
            selected_edges_neg_pred.append((src.item(), dst.item(), neg_pred[i].item()))

    # Step 2: Sort edges by reverse predictions (higher values first)
    selected_edges_neg_pred.sort(key=lambda x: x[2], reverse=True)

    # Step 3: Select the top n edges, ensuring each node appears only once
    unique_edges_to_remove = []
    nodes_included = set()

    for edge in selected_edges_neg_pred:
        src, dst, _ = edge
        if src in selected_nodes and src not in nodes_included:
            unique_edges_to_remove.append((src, dst))
            nodes_included.add(src)
        elif dst in selected_nodes and dst not in nodes_included:
            unique_edges_to_remove.append((src, dst))
            nodes_included.add(dst)

        # Stop if we've reached the desired number of edges
        if len(unique_edges_to_remove) >= total_edges_to_remove:
            break

    # Step 4: Check if we have fewer than required edges
    if len(unique_edges_to_remove) < total_edges_to_remove:
        print(f"Warning: Only {len(unique_edges_to_remove)} unique edges found to remove. Required n edges.")

    # Step 5: Return only the edge pairs (source, destination)
    return [(edge[0], edge[1]) for edge in unique_edges_to_remove]


#____________________________________________________________________________________________________________________________

def predict_test_set(model, test_set):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_set = NormalizeFeatures()(test_set).to(device)
    neg_test_edge_index = negative_sampling(test_set.edge_index, num_nodes=test_set.num_nodes, 
                                            num_neg_samples=test_set.edge_index.size(1))
    
    model.eval()
    with torch.no_grad():
        z = model.encode(test_set.x, test_set.edge_index)
    auc, ap = model.test(z, test_set.edge_index, neg_test_edge_index)
    print(f'Test Set AUC: {auc:.4f}, AP: {ap:.4f}')
    return auc, ap

#____________________________________________________________________________________________________________________________

def created_edited_edge_performance(edge_performance, selected_nodes):
    # **Step 1: Remove position_change labels for selected nodes inside the dictionary**
    edited_edge_performance = {}
    for node, edges in edge_performance.items():
        edited_edges = {}
        for edge, values in edges.items():
            # Set position_change to None or NaN if node is in selected_nodes
            new_values = values[:]  # Copy list to avoid modifying original dataset
            if node in selected_nodes.tolist():
                new_values[2] = (None, new_values[2][1], new_values[2][2], new_values[2][3], new_values[2][4], new_values[2][5])  
            edited_edges[edge] = new_values
        edited_edge_performance[node] = edited_edges
    return edited_edge_performance

#____________________________________________________________________________________________________________________________

def created_edited_edge_performance_df(edited_edge_performance, selected_nodes): 
    """
    Convert edge_performance dictionary to a DataFrame and remove labels (position_change) 
    for all nodes in selected_nodes.

    Args:
        edited_edge_performance (dict): Edge performance dictionary.
        selected_nodes (torch.Tensor): Tensor of nodes whose labels should be removed.

    Returns:
        pd.DataFrame: Edited DataFrame with position_change removed for selected_nodes.
    """
    # Convert edge_performance to DataFrame
    edge_list = []
    for node, edges in edited_edge_performance.items():
        for edge, values in edges.items():
            edge_list.append([
                node, edge, values[0], values[1], values[2][0],  # top_k_indices, new_top_k_indices, position_change
                values[2][1], values[2][2], values[2][3], values[2][4], values[2][5]  # avg_promoted, avg_demoted, ...
            ])

    columns = ['node', 'edge', 'top_k', 'new_top_k', 'position_change', 'avg_promoted', 'avg_demoted', 'avg_changed', 'avg_unchanged', 'total_items']
    edge_removal_model_df = pd.DataFrame(edge_list, columns=columns)

    # **Ensure selected_nodes is a list**
    selected_nodes_list = selected_nodes.tolist() if isinstance(selected_nodes, torch.Tensor) else selected_nodes

    # *********** Remove Labels (`position_change`) for Selected Nodes *************
    edge_removal_model_df.loc[edge_removal_model_df["node"].isin(selected_nodes_list), "position_change"] = np.nan

    return edge_removal_model_df  # Return edited DataFrame
#____________________________________________________________________________________________________________________________________________


# Function to Select Edges to Remove
from torch_geometric.utils import subgraph

def select_edges_to_remove(edge_removal_model, selected_node, num_edges_to_remove, data, edge_removal_model_df):
    with torch.no_grad():
        # Find edges related to the selected node
        node_idx = edge_removal_model_df['node'] == selected_node.item()
        node_edges = edge_removal_model_df[node_idx]

        if node_edges.empty:
            print(f"No edges found for node {selected_node.item()}")
            return []

        # Extract feature subset
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        features_subset = torch.tensor(
            node_edges[['position_change', 'avg_promoted', 'avg_demoted', 'avg_changed', 'avg_unchanged', 'total_items']].values, 
            dtype=torch.float
        ).to(device)

        # Extract the subgraph containing only relevant nodes and edges
        sub_edge_index, _ = subgraph(
            selected_node, data.edge_index, relabel_nodes=True
        )

        # Forward pass through the edge_removal_model
        scores = edge_removal_model(features_subset, sub_edge_index)

        # **Fix:** Move scores to CPU and convert to NumPy
        sorted_indices = scores.argsort(descending=True)[:num_edges_to_remove].cpu().numpy()

        # Select top edges
        top_edges = node_edges.iloc[sorted_indices]

        return [tuple(e) for e in top_edges['edge'].values]  # Return edges as tuples
    
    
#____________________________________________________________________________________________________________________________________________

def update_graph(graph, edges_to_remove):
    edges_to_remove_set = set(edges_to_remove)
    mask = ~torch.tensor([tuple(e.tolist()) in edges_to_remove_set for e in graph.edge_index.T], dtype=torch.bool)
    graph.edge_index = graph.edge_index[:, mask]  # Remove edges
    return graph

#____________________________________________________________________________________________________________________________________________

def build_target_node_attack(data, selected_nodes, model_path, device='cuda'):

    if isinstance(selected_nodes, torch.Tensor) and selected_nodes.ndim == 0:
        selected_nodes = selected_nodes.unsqueeze(0)  # Convert scalar tensor to 1D tensor
    elif isinstance(selected_nodes, int):  # If it's an integer, wrap in a list
        selected_nodes = [selected_nodes]

    # Load the trained model
    class Encoder(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super(Encoder, self).__init__()
            self.linear_mu = nn.Linear(in_channels, out_channels)
            self.linear_logstd = nn.Linear(in_channels, out_channels)
            self.propagate = APPNP(K=1, alpha=0)

        def forward(self, x, edge_index):
            mu = self.linear_mu(x)
            mu = self.propagate(mu, edge_index)
            logstd = self.linear_logstd(x)
            logstd = self.propagate(logstd, edge_index)
            return mu, logstd

    # Copy the data to avoid modifying the original
    updated_data = data.clone()

    # Set device
    updated_data = updated_data.to(device)

    # Load trained VGAE model
    model = VGAE(Encoder(updated_data.num_features, 128)).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Get latent space embeddings
    with torch.no_grad():
        z = model.encode(updated_data.x, updated_data.edge_index)

    # Prepare edge removal
    edge_index = updated_data.edge_index.clone()
    edges_to_remove = []
    no_edge_count = 0
    removed_edge_count = 0
    skipped_nodes = []
    processed_nodes = []
    nodes_with_no_edge_removed = []

    # Iterate over selected nodes
    for node in selected_nodes:
        connected_edges = edge_index[:, (edge_index[0] == node) | (edge_index[1] == node)]
        if connected_edges.size(1) == 0:
            no_edge_count += 1
            skipped_nodes.append(node.item())
            continue

        # Calculate probabilities for all connected edges
        probs = torch.sigmoid((z[connected_edges[0]] * z[connected_edges[1]]).sum(dim=1))

        # Try removing an edge until successful or no edges remain
        while probs.size(0) > 0:
            max_prob_idx = torch.argmax(probs)
            edge_to_remove = connected_edges[:, max_prob_idx]

            if tuple(edge_to_remove.tolist()) not in edges_to_remove:
                edges_to_remove.append(tuple(edge_to_remove.tolist()))
                processed_nodes.append(node.item())
                removed_edge_count += 1
                break  # Exit loop after successfully removing an edge

            # Remove this edge from consideration and retry
            probs = torch.cat([probs[:max_prob_idx], probs[max_prob_idx + 1:]])
            connected_edges = torch.cat(
                [connected_edges[:, :max_prob_idx], connected_edges[:, max_prob_idx + 1:]], dim=1
            )

        # If no edge could be removed after retries
        if probs.size(0) == 0:
            nodes_with_no_edge_removed.append(node.item())

    # Deduplicate edges to remove
    unique_edges_to_remove = list(set(edges_to_remove))
    #print(f"Unique edges to remove: {len(unique_edges_to_remove)}")

    # Remove edges from the graph
    if unique_edges_to_remove:
        unique_edges_to_remove = torch.tensor(unique_edges_to_remove, device=device).T
        mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=device)
        for edge in unique_edges_to_remove.T:
            mask &= ~((edge_index[0] == edge[0]) & (edge_index[1] == edge[1]))
        edge_index = edge_index[:, mask]

    # Update edge_index
    updated_data.edge_index = edge_index

    # Print summary
    print(f"___________________________________")
    print(f"Starting to remove!")
    print(f"Number of nodes with no edges: {no_edge_count}")
    #print(f"Nodes with no edges: {skipped_nodes}")
    #print(f"Nodes with edges removed: {processed_nodes}")
    #print(f"Nodes with edges but no edge removed: {nodes_with_no_edge_removed}")
    #print(f"Total unique undirected edges removed: {len(unique_edges_to_remove)}")
    print(f"Remaining edges after removal: {updated_data.edge_index.size(1)}")
    print(f"Removing completed!")
    print(f"___________________________________")
    return updated_data

#____________________________________________________________________________________________________________________________

def compute_position_change(node, top_k_indices, new_top_k_indices):
    # Convert tensors to CPU for easier indexing
    top_k_indices_cpu = top_k_indices
    new_top_k_indices_cpu = new_top_k_indices
    
    # Find the position of the node in both lists
    old_position_tensor = (top_k_indices_cpu == node).nonzero(as_tuple=True)[1]
    new_position_tensor = (new_top_k_indices_cpu == node).nonzero(as_tuple=True)[1]

    # If node is missing in either list, return None
    if old_position_tensor.numel() == 0 or new_position_tensor.numel() == 0:
        return 1

    old_position = old_position_tensor.item()
    new_position = new_position_tensor.item()

    # Calculate the change in position
    change_position = new_position - old_position
    return change_position



########################################################################################################################

#____________________________________________________________________________________________________________________________

def gold_remove_edges(data, edges_to_remove, n_edges):
    """
    Remove the top-n edges from the data.
    """
    edge_index = data.edge_index.clone()
    
    # Ensure we only remove up to n_edges if the provided list is longer
    edges_to_remove = edges_to_remove[:n_edges]
    edges_to_remove_set = set(edges_to_remove)
    
    # Apply mask
    mask = torch.tensor([tuple(edge.tolist()) not in edges_to_remove_set for edge in data.edge_index.t()], dtype=torch.bool)
    
    updated_data = data.clone()
    updated_data.edge_index = data.edge_index[:, mask]
    
    return updated_data

#____________________________________________________________________________________________________________________________

def compute_clusters(data):
    """
    Compute clusters using the Louvain method.
    
    Args:
        data (torch_geometric.data.Data): The input graph data.
    
    Returns:
        dict: Mapping from node ID to cluster ID.
    """
    G = to_networkx(data, to_undirected=True)
    partition = community_louvain.best_partition(G)
    
    return partition


#____________________________________________________________________________________________________________________________

def create_training_set_for_trained_model(data, selected_nodes):
# Get edge index
    edge_index = data.edge_index

    # Mask to keep only edges that do not involve selected nodes
    mask = ~((edge_index[0].unsqueeze(1) == selected_nodes).any(dim=1) |
            (edge_index[1].unsqueeze(1) == selected_nodes).any(dim=1))

    # Create training set by removing edges of selected nodes
    training_edge_index = edge_index[:, mask]

    # Create a new data object for training
    training_set = data.clone()
    training_set.edge_index = training_edge_index

    return training_set

#____________________________________________________________________________________________________________________________

def create_test_set_trained_model(data, selected_nodes):

    mask = torch.isin(data.edge_index[0], selected_nodes) | torch.isin(data.edge_index[1], selected_nodes)
    filtered_edge_index = data.edge_index[:, mask]
    test_set = data.clone()
    test_set.edge_index = filtered_edge_index

    return test_set

#____________________________________________________________________________________________________________________________


def load_sort_and_create_edge_performance_dataset(graph_model, data_name, ep_save_path, embedding_version):


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    file_path = f'{ep_save_path}/change_position_{data_name}_{graph_model}_edge_performance_{embedding_version}.pkl'  #./CiteSeer_gcn_edge_performance.pkl

    if os.path.exists(file_path):
        print("_____________________________________")
        print(f"Edge Performance found in here {file_path}. Loading edge performance data...")
        print("_____________________________________")
        with open(file_path, "rb") as file:
            if torch.cuda.is_available() == True:
                edge_performance = pickle.load(file)
            else:
                edge_performance = torch.load(file, map_location=torch.device('cpu'))

        print("_____________________________________")
        print("Edge performance loaded successfully and ready to sort and create dataset!")
        print("_____________________________________")
    else:
        print("_____________________________________")
        print(f"Edge Performance not found at {file_path}. Run Edge Performance First")
        print("_____________________________________")


    #for node, edges in edge_performance.items():
    #    if promo_mode:
    #        sorted_edges = sorted(edges.items(), key=lambda x: x[1][2][0])  # Ascending order (lowest scores)
    #    else:
    #        sorted_edges = sorted(edges.items(), key=lambda x: x[1][2][0], reverse=True)  # Descending order (highest scores)
    #    
    #    top_n_edges = sorted_edges[:n]
    #    target_edges_dataset[node] = {edge[0]: edge[1] for edge in top_n_edges}  

    return edge_performance  #format of citeseer, cora and pubmed

#____________________________________________________________________________________________________________________________

def extract_edge_changes(edge_performance):
    """
    Extracts node, edge, and position change from edge_performance.
    
    Args:
        edge_performance (dict): Dictionary with node as keys and edges with their details as values.
    
    Returns:
        dict: A new dictionary with only node, edge, and position change.
    """
    simplified_performance = {}
    
    for node, edges in edge_performance.items():
        simplified_performance[node] = {}
        for edge, details in edges.items():
            position_change = details[2][0]  # Extract position change
            simplified_performance[node][edge] = position_change
    
    return simplified_performance 

 #____________________________________________________________________________________________________________________________

def load_edge_performance_change_position(graph_model, data_name, ep_save_path, embedding_version):

    edge_performance = load_sort_and_create_edge_performance_dataset(graph_model, data_name, ep_save_path, embedding_version)                                                                                                                                                                      
    simplified_performance = extract_edge_changes(edge_performance) #just edge: change_position

    return simplified_performance

 #____________________________________________________________________________________________________________________________

''' 
def create_simplified_performance(graph_model, data_name, ep_save_path):

    edge_performance = load_sort_and_create_edge_performance_dataset(graph_model, data_name, ep_save_path)                                                                                     
    simplified_performance = extract_edge_changes(edge_performance) #just edge: change_position

    return simplified_performance 
'''
 #____________________________________________________________________________________________________________________________


from torch_geometric.nn import GCNConv, VGAE

class VGAE_Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=64):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, out_channels)
        self.conv_logstd = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


#____________________________________________________________________________________________________________________________

def load_vgae_model(data, vgae_path, vgae_emb_path):

    # Recreate model architecture exactly as before
    vgae_encoder = VGAE_Encoder(data.num_features, hidden_channels=128, out_channels=64)
    vgae_model = VGAE(vgae_encoder)

    # Load weights
    vgae_model.load_state_dict(torch.load(vgae_path, map_location='cpu'))
    vgae_model.eval()
    vgae_embedding = torch.load(vgae_emb_path, map_location='cpu')


    return vgae_model, vgae_embedding

#____________________________________________________________________________________________________________________________

def node_retrieval_position(node_id, retrieval_list):
    """
    Return the position (1-indexed) of node_id in retrieval_list.
    If not found, return -1.
    """
    try:
        position = retrieval_list.index(node_id) + 1  # +1 for 1-indexed position
        return position
    except ValueError:
        return -1  # Node not found 

#____________________________________________________________________________________________________________________________

def load_data(data_name, dataset_subgraph_path):

    if data_name == 'CiteSeer': 
        dataset = Planetoid(root='data/Planetoid', name='CiteSeer')
        data = dataset[0]
    elif data_name == 'Cora': 
        dataset = Planetoid(root='data/Planetoid', name='Cora')
        data = dataset[0]
    elif data_name == 'PubMed':
        dataset = torch.load(dataset_subgraph_path, weights_only=False) #torch.load(dataset_subgraph_path)
        data = remove_isolated_nodes(dataset)

    return data

#____________________________________________________________________________________________________________________________

def show_query_position(data_name, model_name, graph_model, node_id, node_retrieval_rank, attacked_at_num_node_dict, result_path):
    
    """
    Logs retrieval vs attack positions for a node, including delta and success flag.

    Columns:
        data_name, model_name, graph_model, node_id,
        retrieval_position, attack_position, delta, success_flag
    """

    save_path = f"{result_path}/query_positions/{data_name}_{model_name}_{graph_model}_query_positions.csv"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    node_tensor = attacked_at_num_node_dict.copy()
    nodes_tensor = node_tensor[node_id].flatten()

    # --- Find attack position (after attack) ---
    positions = (nodes_tensor == node_id).nonzero(as_tuple=True)[0]
    if len(positions) > 0:
        attack_position = (positions + 1).item()  # 1-indexed
    else:
        attack_position = None

    # --- Compute delta and success flag ---
    if attack_position is not None and node_retrieval_rank is not None:
        delta = attack_position - node_retrieval_rank
        success_flag = 1 if delta > 0 else 0
    else:
        delta = None
        success_flag = None

    # --- Create DataFrame row ---
    new_row = pd.DataFrame({
        "data_name": [data_name],
        "model_name": [model_name],
        "graph_model": [graph_model],
        "node_id": [node_id],
        "retrieval_position": [node_retrieval_rank],
        "attack_position": [attack_position],
        "delta": [delta],
        "success_flag": [success_flag]
    })

    # --- Save / Append ---
    if os.path.exists(save_path):
        new_row.to_csv(save_path, mode='a', header=False, index=False)
    else:
        new_row.to_csv(save_path, mode='w', header=True, index=False)

#____________________________________________________________________________________________________________________________


def process_excel_sheets(result_path, data_name, promotion_mode, text):

    sheet_names = ["retrieval_avg_position", "attacked_retrieval_node_avg_pos", "duration_per_run"] 
    file_path = f"{result_path}/results/{data_name}_{promotion_mode}_{text}_report.xlsx"
    output_path = f"{result_path}/filtered/{data_name}_{promotion_mode}_{text}_report_filtered.xlsx"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        for sheet_name in sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # Identify Mean and Std rows
            mean_std_indices = df[df.iloc[:, 0].isin(["Mean"])].index
            
            # Initialize a list to store attack labels
            attack_labels = []
            last_attack_name = None  # Keeps track of the latest attack type
            
            for idx in df.index:
                if df.iloc[idx, 0] not in ["Mean"]:
                    last_attack_name = df.iloc[idx, 0]  # Update the last attack type
                
                if df.iloc[idx, 0] == "Mean":
                    attack_labels.append(f"MEAN {last_attack_name}")

            
            # Assign the correct labels to the DataFrame
            df.loc[mean_std_indices, "Run Identifier"] = attack_labels
            
            # Extract only Mean and Std rows
            df_filtered = df.loc[mean_std_indices, ["Run Identifier", "@4000"]]
            
            # Save the filtered data into the new Excel file
            df_filtered.to_excel(writer, sheet_name=sheet_name, index=False)

#____________________________________________________________________________________________________________________________

def create_query_position_report(result_path):

    # Path to the directory containing your CSV files
    path = f"{result_path}/query_positions"

    csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]

    # Read and concatenate all CSVs
    df_list = [pd.read_csv(os.path.join(path, file)) for file in csv_files]
    combined_df = pd.concat(df_list, ignore_index=True)

    # --- Replace model_name values ---
    model_name_mapping = {
        "per_node_highest_degree": "Degree",
        "per_node_p_page_rank": "Page Rank",
        "per_node_viking": "Viking",
        "per_node_targeted_node": "Proposed Method"
    }
    combined_df["model_name"] = combined_df["model_name"].replace(model_name_mapping)

    # --- Replace graph_model values ---
    graph_model_mapping = {
        "epagcl_gcn": "GCN",
        "epagcl_sage": "GraphSage"
    }
    combined_df["graph_model"] = combined_df["graph_model"].replace(graph_model_mapping)

    # Save the concatenated DataFrame
    output_path = os.path.join(path, "all_query_positions_combined.csv")
    combined_df.to_csv(output_path, index=False)


    print(f"✅ Combined CSV created at: {output_path}")

#____________________________________________________________________________________________________________________________


def load_embedding_model(data, graph_model, data_name, embeddings_save_dir, embedding_version): 
    

    file_prefix = f"{data_name}_{graph_model}_{embedding_version}"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)  # Move data to the appropriate device


    emb_path = os.path.join(embeddings_save_dir, f"{file_prefix}_embeddings.pt")
    model_path = os.path.join(embeddings_save_dir, f"{file_prefix}_model.pt")
    print(f">> Found cached {data_name} {graph_model} model and embeddings! Loading from disk...", flush=True)
    model = torch.load(model_path, map_location=device)
    dataset_embeddings = torch.load(emb_path, map_location="cpu")
    model.eval()
    print(f"✅ {data_name} {graph_model} model and embeddings successfully loaded.",  flush=True)
    print("____________________________________________")


    return model, dataset_embeddings

#____________________________________________________________________________________________________________________________