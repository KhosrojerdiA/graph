

import sys
ep_tr_path = '/mnt/data/khosro/Graph_v2'
sys.path.append(ep_tr_path)

from utils.utils import *
from EPAGCL.epagcl import *
import os
import torch
import random
import numpy as np

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def epagcl_embeddings(data, data_name, graph_model, save_dir, epochs, text, main_seed, n_runs=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    file_prefix = f"{data_name}_{graph_model}_{text}"
    os.makedirs(save_dir, exist_ok=True)

    best_loss = float('inf')
    best_run = -1
    best_emb = None
    best_model = None

    

    for run in range(1, n_runs + 1):
        print(f"\n========== Run {run}/{n_runs} ==========")
        setup_seed(main_seed + run)

        dataset_embeddings, model, final_loss = get_epagcl_embeddings(
            data=data,
            device=str(device),
            epochs=epochs,
            batch_compute=False,
            add_single=True,  # same as your default
            not_add_edge=False,
            not_drop_edge=False,
            add_edge_random=False,
            graph_model=graph_model
        )

        print(f"[Run {run}] Final loss: {final_loss:.6f}")

        if final_loss < best_loss:
            best_loss = final_loss
            best_emb = dataset_embeddings
            best_model = model
            best_run = run

    # Save best model and embeddings
    emb_path = os.path.join(save_dir, f"{file_prefix}_embeddings.pt")
    model_path = os.path.join(save_dir, f"{file_prefix}_model.pt")

    torch.save(best_emb, emb_path)
    torch.save(best_model, model_path)

    print("\n✅ Best run summary:")
    print(f"   • Run: {best_run}")
    print(f"   • Best final loss: {best_loss:.6f}")
    print(f"   • Saved embeddings: {emb_path}")
    print(f"   • Saved model: {model_path}")
    print("____________________________________________")


# ----------------------------- RUN CONFIG -----------------------------

data_name = 'Cora'         # 'Cora', 'CiteSeer', 'PubMed'
graph_model = 'epagcl_gcn' # 'epagcl_gcn', 'epagcl_sage'
epochs = 1950
text = "v1"

dataset_subgraph_path = f"{ep_tr_path}/data/pubmed_subgraph.pt"
main_seed = 3708
save_dir = f"{ep_tr_path}/EPAGCL/embeddings"

data = load_data(data_name, dataset_subgraph_path)

epagcl_embeddings(data, data_name, graph_model, save_dir, epochs, text, main_seed)