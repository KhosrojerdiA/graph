#With Loading grace, gcn and greet

import sys
import os

main_path = '/mnt/data/khosro/Graph_v2'
sys.path.append(main_path)

from utils.utils import *
from retrieval.retrieval_epaglc_v4 import * 
from attack_models.per_node_attack_v4 import *  
import torch
from torch_geometric.datasets import Planetoid
import time

main_seed = 3708

#____________________________________________________________ Inputs ____________________________________________________________


data_name_list = ['Cora']
#['Cora', 'CiteSeer', 'PubMed']


graph_model_list = ['epagcl_gcn', 'gca', 'greet', 'grace'] #victim
#['epagcl_gcn', 'epagcl_sage', 'grace', 'gca', 'greet']

surr_graph_model = 'grace'

model_name_list = ['per_node_highest_degree', 'per_node_gold_attack', 'per_node_targeted_node']
#['per_node_highest_degree', 'per_node_p_page_rank', 'per_node_viking', 'per_node_gold_attack', 'per_node_targeted_node']

text = "model_false"
 

result_path = f"{main_path}/outputs/run_victims"


budget_list = [1, 2, 3, 4, 5] 
#[1, 2, 3, 4, 5] 

#promotion_mode = True #True for Promotion, False for Demotion

embedding_version = 'v1'


runs = 1
#5
#Number of different random seeds 


min_number_edges_list = [10]
#Minimum number of edges connected to selected nodes (query) [5,10]


#____________________________________________________________ Folders ___________________________________________________________

os.makedirs(os.path.dirname(result_path), exist_ok=True)


ep_save_path = f"{main_path}/edge_performance_dataset"
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

                        if graph_model.lower() == "epagcl_gcn" or graph_model.lower() == "epagcl_sage":
                            embedding_save_dir = f"{main_path}/EPAGCL/embeddings"
                        else:   
                            embedding_save_dir = f"{main_path}/{graph_model}/embeddings"


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

                            ) = retrieval_v4(data, data_name, graph_model, min_number_edges, embedding_save_dir, embedding_version, main_seed)
                            

                            print("____________________###________________________", flush=True)
                            print(f'Number of Selected Nodes: {len(selected_nodes)}', flush=True) 
                            print("____________________###________________________", flush=True)
                            viz_selected_nodes_edges(data, selected_nodes)
                            print("____________________###________________________", flush=True)

                            start_time = time.time()

                            if model_name in ['per_node_highest_degree', 'per_node_p_page_rank', 'per_node_viking']:
                                promotion_mode = True 
                            else: 
                                promotion_mode = False

                            #scorer_path = f'{main_path}/trained_scorer/{data_name}_{graph_model}_scorer_model_{scorer_version}.pt'

                            #Attack
                            (
                                
                            attacked_recall_20, attacked_retrieval_node_found_count_20, attacked_retrieval_node_avg_position_20, attacked_avg_promoted_20, attacked_avg_demoted_20, attacked_avg_changed_20, attacked_avg_unchanged_20,
                            attacked_recall_100, attacked_retrieval_node_found_count_100, attacked_retrieval_node_avg_position_100, attacked_avg_promoted_100, attacked_avg_demoted_100, attacked_avg_changed_100, attacked_avg_unchanged_100,
                            attacked_recall_500, attacked_retrieval_node_found_count_500, attacked_retrieval_node_avg_position_500, attacked_avg_promoted_500, attacked_avg_demoted_500, attacked_avg_changed_500, attacked_avg_unchanged_500,
                            attacked_recall_1000, attacked_retrieval_node_found_count_1000, attacked_retrieval_node_avg_position_1000, attacked_avg_promoted_1000, attacked_avg_demoted_1000, attacked_avg_changed_1000, attacked_avg_unchanged_1000,
                            attacked_recall_4000, attacked_retrieval_node_found_count_4000, attacked_retrieval_node_avg_position_4000, attacked_avg_promoted_4000, attacked_avg_demoted_4000, attacked_avg_changed_4000, attacked_avg_unchanged_4000

                            )= per_node_attack(model_name, graph_model, data_name, data, dataset_embeddings, model, selected_nodes, selected_node_embeddings, 
                                            top_k_indice_at_20, top_k_indice_at_100, top_k_indice_at_500, top_k_indice_at_1000, top_k_indice_at_4000, 
                                            ep_save_path, budget, surr_graph_model, embedding_version, result_path, promotion_mode)

                            end_time = time.time()
                            duration = end_time - start_time 
                            duration_per_run.append(duration)

                            #Results
                            retrieval_found_count.extend([found_count_20, found_count_100, found_count_500, found_count_1000, found_count_4000])
                            retrieval_recall.extend([recall_20, recall_100, recall_500, recall_1000, recall_4000])
                            retrieval_avg_position.extend([avg_position_20, avg_position_100, avg_position_500, avg_position_1000, avg_position_4000])

                            attacked_recall.extend([attacked_recall_20, attacked_recall_100, attacked_recall_500, attacked_recall_1000, attacked_recall_4000])
                            attacked_retrieval_node_found_count.extend([attacked_retrieval_node_found_count_20, attacked_retrieval_node_found_count_100, attacked_retrieval_node_found_count_500, attacked_retrieval_node_found_count_1000, attacked_retrieval_node_found_count_4000])
                            attacked_retrieval_node_avg_position.extend([attacked_retrieval_node_avg_position_20, attacked_retrieval_node_avg_position_100, attacked_retrieval_node_avg_position_500, attacked_retrieval_node_avg_position_1000, attacked_retrieval_node_avg_position_4000])
                            attacked_avg_promoted.extend([attacked_avg_promoted_20, attacked_avg_promoted_100, attacked_avg_promoted_500, attacked_avg_promoted_1000, attacked_avg_promoted_4000])
                            attacked_avg_demoted.extend([attacked_avg_demoted_20, attacked_avg_demoted_100, attacked_avg_demoted_500, attacked_avg_demoted_1000, attacked_avg_demoted_4000])
                            attacked_avg_changed.extend([attacked_avg_changed_20, attacked_avg_changed_100, attacked_avg_changed_500, attacked_avg_changed_1000, attacked_avg_changed_4000])
                            attacked_avg_unchanged.extend([attacked_avg_unchanged_20, attacked_avg_unchanged_100, attacked_avg_unchanged_500, attacked_avg_unchanged_1000, attacked_avg_unchanged_4000])

                        
                            step_count += 1
                            
                        #____________________________________________________________ Result ____________________________________________________________

                        print("____________________###________________________", flush=True)
                        print("All Runs are Done!", flush=True)

                        #retrieval_store_to_excel(data_name, graph_model, min_number_edges, model_name, retrieval_found_count, retrieval_recall, retrieval_avg_position, result_path)
                        print("____________________###________________________", flush=True)
                        store_to_excel(data_name, graph_model, min_number_edges, model_name, budget, 
                                    retrieval_found_count, retrieval_recall, retrieval_avg_position, 
                                    attacked_recall, attacked_retrieval_node_found_count, attacked_retrieval_node_avg_position,
                                    attacked_avg_promoted, attacked_avg_demoted, 
                                    attacked_avg_changed, attacked_avg_unchanged, duration_per_run, result_path, promotion_mode, text)
                        
                        print(f'Results are ready for {data_name}_{graph_model}_{min_number_edges}_{budget}_{model_name} and appended to excel!', flush=True)
                    
process_excel_sheets(result_path, data_name, promotion_mode, text)
create_query_position_report(result_path)

#____________________________________________________________ End of the loop ________________________________________________ min_number_edges

 

