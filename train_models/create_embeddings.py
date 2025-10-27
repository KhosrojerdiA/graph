
import sys
import os

main_path = '/mnt/data/khosro/Graph_v2'
sys.path.append(main_path)

from utils.utils import *
from retrieval.retrieval_epaglc_v3 import * 
from attack_models.per_node_attack import *  
import torch
from torch_geometric.datasets import Planetoid
import time

main_seed = 3708

#____________________________________________________________ Inputs ____________________________________________________________


data_name_list = ['Cora', 'CiteSeer', 'PubMed']
#['Cora', 'CiteSeer', 'PubMed']


graph_model_list = ['epagcl_gcn', 'epagcl_sage']
#['epagcl_gcn', 'epagcl_sage']




model_name_list = ['per_node_random_attack']    
#['per_node_highest_degree', 'per_node_p_page_rank', 'per_node_viking', 'per_node_gold_attack', 'per_node_targeted_node']

budget_list = [1] 
#[1, 2, 3, 4, 5] 

runs = 1
#5
#Number of different random seeds 


min_number_edges_list = [10]
#Minimum number of edges connected to selected nodes (query) [5,10]


#____________________________________________________________ Folders ___________________________________________________________

embedding_save_dir = f"{main_path}/embeddings"
dataset_subgraph_path = f"{main_path}/data/pubmed_subgraph.pt"

#____________________________________________________________ Start of the loop ________________________________________________

for data_name in data_name_list:
    for graph_model in graph_model_list:
        for min_number_edges in min_number_edges_list:
            for budget in budget_list:
                for model_name in model_name_list:
            
                        print("NOTE____________________Starting Stats________________________", flush=True)
                        print(f'data_name: {data_name}', flush=True)
                        print(f'model_name: {model_name}', flush=True)
                        print(f'graph_model: {graph_model}', flush=True)
                        print(f'min_number_edges: {min_number_edges}', flush=True)
                        print("____________________Starting Stats________________________")



                        step_count = 1

                        retrieval_found_count = []
                        retrieval_recall = []
                        retrieval_avg_position = []
                        retrieval_avg_promoted = []
                        retrieval_avg_demoted = []
                        retrieval_avg_changed = []
                        retrieval_avg_unchanged = []

                        attacked_recall = []
                        attacked_retrieval_node_found_count = []
                        attacked_retrieval_node_avg_position = []
                        attacked_avg_promoted = []
                        attacked_avg_demoted = []
                        attacked_avg_changed = []
                        attacked_avg_unchanged = []
                        retrieval_position_after_attack = []

                        duration_per_run = []

                        #________________________________________________________________________________________________________________________


                        data = load_data(data_name, dataset_subgraph_path)
                        #data.is_undirected()

                        #____________________________________________________________ Loop ____________________________________________________________

                        for seed_idx in range(runs): 

                            print("***************************************************************************************************************************************", flush=True)
                            print("***************************************************************************************************************************************", flush=True)
                            print(f'Run Number {step_count:>3} for {data_name}_{model_name}_{graph_model}_{min_number_edges}')
                            print("***************************************************************************************************************************************", flush=True)
                            print("***************************************************************************************************************************************", flush=True)

                            #Retrieval
                            (
                
                            dataset_embeddings, model, selected_nodes, selected_node_embeddings, 
                            top_k_indice_at_20, top_k_indice_at_100, top_k_indice_at_500, top_k_indice_at_1000,top_k_indice_at_4000, 
                            found_count_20, found_count_100, found_count_500, found_count_1000, found_count_4000, 
                            recall_20, recall_100, recall_500, recall_1000, recall_4000, 
                            avg_position_20, avg_position_100, avg_position_500, avg_position_1000, avg_position_4000 

                            ) = retrieval_v3(data, data_name, graph_model, min_number_edges, embedding_save_dir, main_seed)
                            

#____________________________________________________________ End of the loop ________________________________________________ min_number_edges

