

import sys


prn_project_path = '/mnt/data/khosro/Graph_v2'
sys.path.append(prn_project_path)

from utils.utils import *
from utils.utils_scorer import *
from attack_models.random_attack import *
from attack_models.highest_degree_attack import * 
from attack_models.page_rank_attack import *
from attack_models.viking_attack import * 
from attack_models.gold_attack import *
import torch

#____________________________________________________________________________________________________________________________________


def per_node_attack(model_name, graph_model, data_name, data, dataset_embeddings, model, selected_nodes, selected_node_embeddings, 
                    top_k_indice_at_20, top_k_indice_at_100, top_k_indice_at_500, top_k_indice_at_1000, top_k_indice_at_4000, 
                    ep_save_path, budget, surr_graph_model, embedding_version, result_path, promotion_mode): 
    

    vgae_path = f"/mnt/data/khosro/Graph-Pruning/trained_scorer/{data_name}_{graph_model}_VGAE.pt"
    vgae_emb_path = f"/mnt/data/khosro/Graph-Pruning/trained_scorer/{data_name}_{graph_model}_VGAE_embeddings.pt"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #data = data.to(device)
#____________________________________________________________________________________________________________________________________

    #Loading Surrgate model

    if surr_graph_model.lower() == "epagcl_gcn" or surr_graph_model.lower() == "epagcl_sage":
        surr_embeddings_save_dir = f"{prn_project_path}/EPAGCL/embeddings"
    else:   
        surr_embeddings_save_dir = f"{prn_project_path}/{surr_graph_model}/embeddings"

    surr_embedding_version = embedding_version
    suur_scorer_path = f'{prn_project_path}/trained_scorer/{data_name}_{surr_graph_model}_scorer_model_v1.pt'
    surr_model, surr_dataset_embeddings = load_embedding_model(data, surr_graph_model, data_name, surr_embeddings_save_dir, surr_embedding_version)

#____________________________________________________________________________________________________________________________________


    data_num_nodes = data.num_nodes

    print("____________________________________________", flush=True)
    print("_")
    print(f'Model Name: {model_name:>3}')
    print("____________________________________________", flush=True)

    attacked_at_num_node_dict = {} 
      
# __________________________________________________________________________________________________________________________

    selected_node_data_structure = [{"node_id": selected_node.item(), "embedding": embedding} for selected_node, embedding in zip(selected_nodes, selected_node_embeddings)]
    step_count = 0

    if model_name == "per_node_link_prediction":
        vgae_model, vgae_embedding = load_vgae_model(data, vgae_path, vgae_emb_path)
    #elif model_name == "per_node_rl":
    #    rl_model = create_train_rl_sv2_model(data, selected_nodes)
    elif model_name == "per_node_cluster":
        clusters = compute_clusters(data)
    elif model_name == "per_node_targeted_node":
        #scorer_model = load_scorer_model(dataset_embeddings, scorer_path, data, feature_fn=edge_features_v5_targeted) 
        scorer_model = load_scorer_model(surr_dataset_embeddings, suur_scorer_path, data, feature_fn=edge_features_v5_targeted) 
    elif model_name == "per_node_gold_attack":
        simplified_performance = load_edge_performance_change_position(surr_graph_model, data_name, ep_save_path, embedding_version)
 

# ___________________________________________________________ Loop _______________________________________________________________
    
    for node_data in selected_node_data_structure:

        node_id = node_data['node_id']

        if model_name == "per_node_random_attack":
            updated_data = build_random_attack(data, selected_nodes[step_count], budget, promotion_mode) 
            #print(selected_nodes[step_count]) -> tensor(55, device='cuda:0')
        elif model_name == "per_node_highest_degree":
            updated_data = build_highest_degree_attack(data, selected_nodes[step_count], budget, promotion_mode)

        elif model_name == "per_node_page_rank":
            updated_data = build_page_rank_attack(data, selected_nodes[step_count], budget, promotion_mode)

        elif model_name == "per_node_p_page_rank":
            updated_data = build_p_page_rank_attack(data, selected_nodes[step_count], budget, promotion_mode)

        elif model_name == "per_node_viking":
            updated_data = viking_attack_per_node(data, selected_nodes[step_count], budget=budget, dim=32, window_size=5, supervised=True) 

        #elif model_name == "per_node_rl":
        #    updated_data = build_rl_attack(data, selected_nodes[step_count], rl_model, budget, promotion_mode) 

        #elif model_name == "per_node_cluster":
        #    updated_data = build_cluster_attack(data, selected_nodes[step_count], clusters, budget, promotion_mode) 

        #elif model_name == "per_node_link_prediction":
        #    updated_data = build_link_prediction_attack(data, selected_nodes[step_count], vgae_model, vgae_embedding, budget, promotion_mode) 

        elif model_name == "per_node_gold_attack":
            updated_data = build_gold_attack(data, simplified_performance, selected_nodes[step_count], budget, promotion_mode)

        elif model_name == "per_node_targeted_node":
            updated_data = build_new_target_node_attack(data, selected_nodes[step_count], budget, scorer_model, surr_dataset_embeddings, promotion_mode)

 
        compare_original_vs_updated(data, updated_data, budget)

                                                                                                                         #tensor([12])
        attacked_dataset_embeddings, attacked_one_node_selected_node_embeddings = attacked_embedding_v2(updated_data, 
                                                                                                        torch.tensor([selected_nodes[step_count]]), 
                                                                                                        graph_model, model)
        
        attacked_at_num_node_dict = per_node_attacked_return(data_num_nodes, attacked_dataset_embeddings, 
                                                             attacked_one_node_selected_node_embeddings[0], node_id, attacked_at_num_node_dict)
                                                                                                                                                

        

        node_retrieval_rank = node_retrieval_position(node_id, top_k_indice_at_4000[step_count].tolist())
        show_query_position(data_name, model_name, graph_model, node_id, node_retrieval_rank, attacked_at_num_node_dict, result_path)

        step_count += 1

# ___________________________________________________________ Loop _______________________________________________________________
    
    return(
        per_node_dictionary_return(
                                    selected_nodes, top_k_indice_at_20, 
                                      top_k_indice_at_100, 
                                      top_k_indice_at_500, 
                                      top_k_indice_at_1000, 
                                      top_k_indice_at_4000, attacked_at_num_node_dict, data_num_nodes
                                      
                                    )
                                      
            )







