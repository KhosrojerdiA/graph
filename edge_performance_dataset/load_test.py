
#with right split - classification

import sys
import os


ep_path = '/mnt/data/khosro/Graph_v2'
sys.path.append(ep_path)

# Set a seed for reproducibility
#torch.manual_seed(3708) 

from utils.utils import *
from retrieval.retrieval_epaglc_v4 import * 
import torch
from torch_geometric.datasets import Planetoid

#________________________________________________________________________________________________________________________



data_name = 'Cora'
#['Cora', 'CiteSeer', 'PubMed']

graph_model = 'epagcl'
#['gcn', 'sage', 'graphpatcher', 'gat2', 'epagcl']
       
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#____________________________________________________________ Folders ___________________________________________________________


trained_models_path = f"{ep_path}/edge_performance_dataset"
dataset_subgraph_path = f"{ep_path}/data/pubmed_subgraph.pt"

#____________________________________________________________ Seeds ____________________________________________________________

if data_name == 'CiteSeer': 
    dataset = Planetoid(root='data/Planetoid', name='CiteSeer')
    data = dataset[0]
elif data_name == 'Cora': 
    dataset = Planetoid(root='data/Planetoid', name='Cora')
    data = dataset[0]
elif data_name == 'PubMed':
    dataset = torch.load(dataset_subgraph_path)
    data = dataset
                        

#________________________________________________________________________________________________________________________



simplified_performance = load_edge_performance_change_position(graph_model, data_name, ep_path)

# Print all neighbors of node 55 for inspection
print("Node 12 edges:", simplified_performance[12]) #12, 75

# Find and print the first node whose edge label > 0
found = False
for node, edges in simplified_performance.items():
    for edge, label in edges.items():
        if label > 2:
            print(f"Node {node} has a positive edge: {edge} → label = {label}")
            found = True
            break
    if found:
        break

if not found:
    print("No edge with label > 0 found.")


#print("_______________all____________")
#print(simplified_performance)
#data
#data_with_label


#Create 22 last nodes from 122 nodes and load the the edge performnce for them

#________________________________________________________________________________________________________________________


