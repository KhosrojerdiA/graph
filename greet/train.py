
import sys
import os

prn_project_path = '/mnt/data/khosro/Graph_v2'
sys.path.append(prn_project_path)

from greet.data_loader import load_data
from greet.model import *
from greet.utils import *
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import random
import yaml

EOS = 1e-10


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)



def train_cl(cl_model, discriminator, optimizer_cl, features, str_encodings, edges):

    cl_model.train()
    discriminator.eval()

    adj_1, adj_2, weights_lp, _ = discriminator(torch.cat((features, str_encodings), 1), edges)
    features_1, adj_1, features_2, adj_2 = augmentation(features, adj_1, features, adj_2, args, cl_model.training)
    cl_loss = cl_model(features_1, adj_1, features_2, adj_2)

    optimizer_cl.zero_grad()
    cl_loss.backward()
    optimizer_cl.step()

    return cl_loss.item()


def train_discriminator(cl_model, discriminator, optimizer_disc, features, str_encodings, edges, args):

    cl_model.eval()
    discriminator.train()

    adj_1, adj_2, weights_lp, weights_hp = discriminator(torch.cat((features, str_encodings), 1), edges)
    rand_np = generate_random_node_pairs(features.shape[0], edges.shape[1])
    psu_label = torch.ones(edges.shape[1]).cuda()

    embedding = cl_model.get_embedding(features, adj_1, adj_2)
    edge_emb_sim = F.cosine_similarity(embedding[edges[0]], embedding[edges[1]])

    rnp_emb_sim_lp = F.cosine_similarity(embedding[rand_np[0]], embedding[rand_np[1]])
    loss_lp = F.margin_ranking_loss(edge_emb_sim, rnp_emb_sim_lp, psu_label, margin=args.margin_hom, reduction='none')
    loss_lp *= torch.relu(weights_lp - 0.5)

    rnp_emb_sim_hp = F.cosine_similarity(embedding[rand_np[0]], embedding[rand_np[1]])
    loss_hp = F.margin_ranking_loss(rnp_emb_sim_hp, edge_emb_sim, psu_label, margin=args.margin_het, reduction='none')
    loss_hp *= torch.relu(weights_hp - 0.5)

    rank_loss = (loss_lp.mean() + loss_hp.mean()) / 2

    optimizer_disc.zero_grad()
    rank_loss.backward()
    optimizer_disc.step()

    return rank_loss.item()



def save_model(cl_model, discriminator, features, str_encodings, edges, save_dir, dataset, best_acc_test, device="cuda"):
    """
    Save full GREET pipeline (GCL + Edge_Discriminator) and its learned embeddings.
    Returns:
        dataset_embeddings (torch.Tensor)
    """
    os.makedirs(save_dir, exist_ok=True)

    file_prefix = f"{dataset}_greet"
    model_path = os.path.join(save_dir, f"{file_prefix}_model.pt")
    emb_path = os.path.join(save_dir, f"{file_prefix}_embeddings.pt")
    metrics_path = os.path.join(save_dir, f"{file_prefix}_metrics.yaml")

    # Put models in eval mode
    cl_model.eval()
    discriminator.eval()

    # Compute embeddings
    with torch.no_grad():
        adj_1, adj_2, _, _ = discriminator(torch.cat((features, str_encodings), 1), edges)
        dataset_embeddings = cl_model.get_embedding(features, adj_1, adj_2)

    # Combine both into a single pipeline dict
    full_model = {
        "cl_model": cl_model,
        "discriminator": discriminator
    }

    # Save full model and embeddings
    torch.save(full_model, model_path)
    torch.save(dataset_embeddings.cpu(), emb_path)

    # Save metrics
    with open(metrics_path, "w") as f:
        yaml.safe_dump({"best_acc_test": float(best_acc_test)}, f)

    print(f"✅ Full GREET model saved at: {model_path}")
    print(f"✅ Embeddings saved at: {emb_path}")
    print(f"✅ Metrics saved at: {metrics_path}")

    return dataset_embeddings


def load_model(save_dir, dataset, device="cuda"):
    """
    Load GREET model (GCL + Edge_Discriminator) and embeddings.
    Returns:
        model (dict with 'cl_model' and 'discriminator')
        dataset_embeddings (torch.Tensor)
    """
    file_prefix = f"{dataset}_greet"
    model_path = os.path.join(save_dir, f"{file_prefix}_model.pt")
    emb_path = os.path.join(save_dir, f"{file_prefix}_embeddings.pt")

    if not all(os.path.exists(p) for p in [model_path, emb_path]):
        raise FileNotFoundError(f"Missing model or embeddings for {dataset} in {save_dir}")

    model = torch.load(model_path, map_location=device)
    dataset_embeddings = torch.load(emb_path, map_location="cpu")

    # Put both parts in eval mode
    model["cl_model"].eval()
    model["discriminator"].eval()

    print(f"✅ GREET model and embeddings for {dataset} loaded successfully from {save_dir}")
    print(f"   Embeddings shape: {tuple(dataset_embeddings.shape)}")

    return model, dataset_embeddings


def main(args, dataset_name, save_dir):

    setup_seed(0)
    features, edges, str_encodings, train_mask, val_mask, test_mask, labels, nnodes, nfeats = load_data(dataset_name)
    results = []

    for trial in range(args.ntrials):

        setup_seed(trial)


        cl_model = GCL(nlayers=2, nlayers_proj=args.nlayers_proj, in_dim=nfeats, emb_dim=128,
                    proj_dim=128, dropout=0.5, sparse=args.sparse, batch_size=args.cl_batch_size).cuda()
        cl_model.set_mask_knn(features.cpu(), k=args.k, dataset=dataset_name)
        discriminator = Edge_Discriminator(nnodes, nfeats + str_encodings.shape[1], args.alpha, args.sparse).cuda()

        optimizer_cl = torch.optim.Adam(cl_model.parameters(), lr=0.001, weight_decay=0.0)
        optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=args.lr_disc, weight_decay=0.0)

        features = features.cuda()
        str_encodings = str_encodings.cuda()
        edges = edges.cuda()

        best_acc_val = 0
        best_acc_test = 0

        for epoch in range(1, args.epochs + 1):

            for _ in range(args.cl_rounds):
                cl_loss = train_cl(cl_model, discriminator, optimizer_cl, features, str_encodings, edges)
            rank_loss = train_discriminator(cl_model, discriminator, optimizer_discriminator, features, str_encodings, edges, args)

            print("[TRAIN] Epoch:{:04d} | CL Loss {:.4f} | RANK loss:{:.4f} ".format(epoch, cl_loss, rank_loss))

            if epoch % args.eval_freq == 0:
                cl_model.eval()
                discriminator.eval()
                adj_1, adj_2, _, _ = discriminator(torch.cat((features, str_encodings), 1), edges)
                embedding = cl_model.get_embedding(features, adj_1, adj_2)
                cur_split = 0 if (train_mask.shape[1]==1) else (trial % train_mask.shape[1])
                acc_test, acc_val = eval_test_mode(embedding, labels, train_mask[:, cur_split],
                                                 val_mask[:, cur_split], test_mask[:, cur_split])
                print(
                    '[TEST] Epoch:{:04d} | CL loss:{:.4f} | RANK loss:{:.4f} | VAL ACC:{:.2f} | TEST ACC:{:.2f}'.format(
                        epoch, cl_loss, rank_loss, acc_val, acc_test))

                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    best_acc_test = acc_test

        results.append(best_acc_test)

    print('\n[FINAL RESULT] Dataset:{} | Run:{} | ACC:{:.2f}+-{:.2f}'.format(args.dataset, args.ntrials, np.mean(results),
                                                                           np.std(results)))


                                                                           
    # Save best model
    dataset_embeddings = save_model(cl_model, discriminator, features, str_encodings, edges, save_dir, dataset, np.mean(results))
    return dataset_embeddings



if __name__ == '__main__':


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = "Cora"  # "Cora", "CiteSeer", "PubMed", "DBLP"

    config_path = "/mnt/data/khosro/Graph_v2/greet/config.yaml"
    save_dir = "/mnt/data/khosro/Graph_v2/greet/embeddings"

    # Load full YAML and then select dataset-specific section
    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)

    # Extract only the section for the chosen dataset
    if dataset not in full_config:
        raise KeyError(f"Dataset '{dataset}' not found in {config_path}")

    config = full_config[dataset]
    print(f"Loaded config for dataset '{dataset}':")
    print(config)

    # Convert YAML config into args-like object
    class Args:
        pass
    args = Args()
    for key, value in config.items():
        setattr(args, key, value)

    # Add additional parameters not in YAML
    args.dataset = dataset
    args.config_path = config_path
    args.device = device

    # Run training
    print(args)
    dataset_embeddings = main(args, dataset, save_dir)

    model, dataset_embeddings = load_model(save_dir, dataset, device)

    # Access individual components if needed
    #cl_model = model["cl_model"]
    #discriminator = model["discriminator"]






