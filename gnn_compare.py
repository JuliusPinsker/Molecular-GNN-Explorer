#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
import random
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
# Use non-interactive backend for matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GCNConv, GATConv, GINConv, global_mean_pool, MessagePassing
)
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data

import neptune  # Remove if you don't need Neptune logging.

torch.manual_seed(69)

###############################################################################
# 1) Precompute f_i, f_j with PyTorch
###############################################################################

def precompute_fi_fj(data: Data) -> Data:
    """
    For each edge (i, j) in the graph, find:
      - f_i: the closest neighbor of node i (excluding j),
      - f_j: the closest neighbor of node j (excluding i),
    and store them as data.edge_fi, data.edge_fj.
    
    This version uses torch.cdist(...) to compute all pairwise distances
    at once, then picks the minimum among each node's neighbors.
    """
    # Ensure undirected
    edge_index = to_undirected(data.edge_index, data.num_nodes)
    data.edge_index = edge_index  # store back in data
    pos = data.pos
    num_nodes = data.num_nodes

    # Create NxN distance matrix (pure PyTorch)
    # shape: (N, N)
    dist_matrix = torch.cdist(pos, pos, p=2)

    # Create adjacency mask from edge_index
    adj_mask = torch.zeros(num_nodes, num_nodes, dtype=torch.bool)
    src, dst = edge_index
    for i, j in zip(src.tolist(), dst.tolist()):
        adj_mask[i, j] = True
        adj_mask[j, i] = True

    fi_list = []
    fj_list = []
    # For each undirected edge
    for i_, j_ in zip(src.tolist(), dst.tolist()):
        # -- neighbors of i_, excluding j_ --
        neighbors_i = adj_mask[i_].nonzero(as_tuple=True)[0]
        neighbors_i = neighbors_i[neighbors_i != j_]  # exclude j_
        if len(neighbors_i) == 0:
            fi_list.append(i_)
        else:
            # among neighbors_i, pick one with min distance
            distances_i = dist_matrix[i_, neighbors_i]
            idx_min = torch.argmin(distances_i).item()
            fi_list.append(neighbors_i[idx_min].item())

        # -- neighbors of j_, excluding i_ --
        neighbors_j = adj_mask[j_].nonzero(as_tuple=True)[0]
        neighbors_j = neighbors_j[neighbors_j != i_]
        if len(neighbors_j) == 0:
            fj_list.append(j_)
        else:
            distances_j = dist_matrix[j_, neighbors_j]
            idx_min_j = torch.argmin(distances_j).item()
            fj_list.append(neighbors_j[idx_min_j].item())

    data.edge_fi = torch.tensor(fi_list, dtype=torch.long)
    data.edge_fj = torch.tensor(fj_list, dtype=torch.long)
    return data

def maybe_compute_fi_fj(dataset, cache_file="fi_fj_cache.pkl"):
    """
    Checks if 'cache_file' exists. If so, loads it (list of Data objects).
    Otherwise, uses a ThreadPool to concurrently run precompute_fi_fj on each
    Data object in the dataset, with a tqdm progress bar. Saves results to cache.

    Returns the final list of Data objects with f_i and f_j computed.
    """
    if os.path.exists(cache_file):
        print(f"[Cache] Loading from {cache_file}")
        with open(cache_file, "rb") as f:
            data_list = pickle.load(f)
        return data_list
    else:
        print(f"[Cache] Not found: {cache_file}. Computing f_i, f_j in parallel...")

        data_list = [None]*len(dataset)  # empty placeholders
        # We'll define a helper to store results
        def compute_one(i):
            return i, precompute_fi_fj(dataset[i])

        # Use thread pool
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(compute_one, i) for i in range(len(dataset))]
            for f in tqdm(as_completed(futures), total=len(futures), desc="Precompute fi_fj"):
                i, d = f.result()
                data_list[i] = d

        # Save to cache
        with open(cache_file, "wb") as f:
            pickle.dump(data_list, f)
        print(f"[Cache] Saved precomputed data to {cache_file}")
        return data_list

###############################################################################
# 2) Neptune init (remove if not using)
###############################################################################

def init_neptune():
    api_token = os.environ.get("NEPTUNE_API_TOKEN")
    if api_token is None:
        raise ValueError("NEPTUNE_API_TOKEN environment variable not set!")
    run = neptune.init_run(
        project="happyproject235/Molecular-GNN-Explorer",
        api_token=api_token
    )
    # Log global hyperparameters
    params = {"learning_rate": 0.001, "optimizer": "Adam", "seed": 69}
    run["parameters"] = params
    return run

###############################################################################
# 3) Basic GNN Models (GCN, GAT, GIN)
###############################################################################

class GCNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = nn.Linear(hidden_channels // 2, 1)

    def forward(self, x, edge_index, batch):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = torch.relu(self.lin1(x))
        return self.lin2(x)

class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, heads=2):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1)
        self.lin1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = nn.Linear(hidden_channels // 2, 1)

    def forward(self, x, edge_index, batch):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = torch.relu(self.lin1(x))
        return self.lin2(x)

class GINModel(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GINModel, self).__init__()
        nn1 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.conv1 = GINConv(nn1)
        nn2 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.conv2 = GINConv(nn2)
        self.lin1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = nn.Linear(hidden_channels // 2, 1)

    def forward(self, x, edge_index, batch):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = torch.relu(self.lin1(x))
        return self.lin2(x)

###############################################################################
# 4) Optional More Complex Models (PAMNet, MXMNet)
###############################################################################

class PAMConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(PAMConv, self).__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels)
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.att = nn.Linear(2 * out_channels, 1)

    def forward(self, x, edge_index, pos):
        x = self.lin(x)
        return self.propagate(edge_index, x=x, pos=pos)

    def message(self, x_i, x_j, pos_i, pos_j):
        delta_pos = pos_i - pos_j
        delta_embed = self.pos_mlp(delta_pos)
        att_input = torch.cat([x_i, x_j], dim=-1)
        att_weight = torch.sigmoid(self.att(att_input))
        return att_weight * (x_j + delta_embed)

class PAMNetModel(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(PAMNetModel, self).__init__()
        self.global_conv = GCNConv(in_channels, hidden_channels)
        self.local_conv = PAMConv(in_channels, hidden_channels)
        self.fusion = nn.Linear(2 * hidden_channels, hidden_channels)
        self.lin1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = nn.Linear(hidden_channels // 2, 1)

    def forward(self, x, edge_index, batch, pos):
        global_feat = torch.relu(self.global_conv(x, edge_index))
        local_feat = torch.relu(self.local_conv(x, edge_index, pos))
        combined = torch.cat([global_feat, local_feat], dim=-1)
        fused = torch.relu(self.fusion(combined))
        pooled = global_mean_pool(fused, batch)
        x = torch.relu(self.lin1(pooled))
        return self.lin2(x)

class MXMNetModel(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(MXMNetModel, self).__init__()
        self.local_conv = GCNConv(in_channels, hidden_channels)
        self.global_conv = GCNConv(in_channels, hidden_channels)
        self.fusion = nn.Linear(2 * hidden_channels, hidden_channels)
        self.lin1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = nn.Linear(hidden_channels // 2, 1)

    def forward(self, x, edge_index, batch):
        local = torch.relu(self.local_conv(x, edge_index))
        global_ = torch.relu(self.global_conv(x, edge_index))
        combined = torch.cat([local, global_], dim=-1)
        fused = torch.relu(self.fusion(combined))
        pooled = global_mean_pool(fused, batch)
        x = torch.relu(self.lin1(pooled))
        return self.lin2(x)

###############################################################################
# 5) ComENet (Vectorized). Do geometry entirely in 'forward()'
###############################################################################

class ComENetConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels)
        self.geo_mlp = nn.Sequential(
            nn.Linear(4, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.eps = 1e-8

    def forward(self, x, edge_index, pos, edge_fi, edge_fj):
        """
        1) Transform x with a linear layer.
        2) Compute geometry (distance, angles, dihedral) for each edge
           in a vectorized manner, all inside forward().
        3) Pass geo_emb to propagate() as 'geo_emb'.
        4) 'message()' will combine x_j with geo_emb.
        """
        x = self.lin(x)

        # Edge indices
        i, j = edge_index
        # Gather positions
        pos_i = pos[i]
        pos_j = pos[j]
        pos_fi_ = pos[edge_fi]
        pos_fj_ = pos[edge_fj]

        # Compute geometry
        rel = pos_j - pos_i  # (E, 3)
        d = torch.norm(rel, dim=-1, keepdim=True)  # (E, 1)

        eps = self.eps
        z = rel[:, 2:3]
        r = d + eps
        cos_theta = torch.clamp(z / r, min=-1+eps, max=1-eps)
        theta = torch.acos(cos_theta)  # (E, 1)
        phi = torch.atan2(rel[:, 1:2], rel[:, 0:1])  # (E, 1)

        # dihedral
        v1 = pos_i - pos_fi_
        v2 = rel
        v3 = pos_j - pos_fj_
        n1 = torch.cross(v1, v2, dim=-1)
        n2 = torch.cross(v2, v3, dim=-1)
        n1_norm = torch.norm(n1, dim=-1) + eps
        n2_norm = torch.norm(n2, dim=-1) + eps
        cos_tau = torch.clamp(
            torch.sum(n1 * n2, dim=-1) / (n1_norm * n2_norm),
            min=-1+eps, max=1-eps
        )
        tau = torch.acos(cos_tau).unsqueeze(-1)

        geo_features = torch.cat([d, theta, phi, tau], dim=-1)  # (E, 4)
        geo_emb = self.geo_mlp(geo_features)                     # (E, out_channels)

        # Now call propagate, passing 'geo_emb' but NOT passing 'pos'.
        return self.propagate(edge_index, x=x, geo_emb=geo_emb)

    def message(self, x_j, geo_emb):
        """
        Combine neighbor features x_j with geometric embedding geo_emb.
        """
        return x_j + geo_emb

class ComENetModel(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(ComENetModel, self).__init__()
        self.conv1 = ComENetConv(in_channels, hidden_channels)
        self.conv2 = ComENetConv(hidden_channels, hidden_channels)
        self.self_atom = nn.Linear(hidden_channels, hidden_channels)  # self-update
        self.lin1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = nn.Linear(hidden_channels // 2, 1)

    def forward(self, x, edge_index, batch, pos, edge_fi, edge_fj):
        """
        1) Pass data into two ComENetConv layers (both do geometry).
        2) Self-atom update, global pool.
        3) Final MLP to get scalar output.
        """
        x = torch.relu(self.conv1(x, edge_index, pos, edge_fi, edge_fj))
        x = torch.relu(self.conv2(x, edge_index, pos, edge_fi, edge_fj))
        x = torch.relu(self.self_atom(x))
        x = global_mean_pool(x, batch)
        x = torch.relu(self.lin1(x))
        return self.lin2(x)

###############################################################################
# 6) Training, Testing, Evaluation
###############################################################################

def train(model, loader, criterion, optimizer, device, use_pos=False):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        target = data.y[:, 7].view(-1, 1)

        if use_pos:
            out = model(data.x, data.edge_index, data.batch,
                        data.pos, data.edge_fi, data.edge_fj)
        else:
            out = model(data.x, data.edge_index, data.batch)

        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def test(model, loader, criterion, device, use_pos=False):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            target = data.y[:, 0].view(-1, 1)

            if use_pos:
                out = model(data.x, data.edge_index, data.batch,
                            data.pos, data.edge_fi, data.edge_fj)
            else:
                out = model(data.x, data.edge_index, data.batch)

            loss = criterion(out, target)
            total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def evaluate_full(model, loader, device, use_pos=False):
    """
    Evaluate MSE, MAE, RMSE, R2 on the entire loader.
    """
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            target = data.y[:, 0].view(-1, 1)
            if use_pos:
                out = model(data.x, data.edge_index, data.batch,
                            data.pos, data.edge_fi, data.edge_fj)
            else:
                out = model(data.x, data.edge_index, data.batch)
            all_preds.append(out.cpu())
            all_targets.append(target.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    mse = nn.MSELoss()(all_preds, all_targets).item()
    mae = nn.L1Loss()(all_preds, all_targets).item()
    rmse = math.sqrt(mse)
    ss_res = torch.sum((all_targets - all_preds) ** 2)
    ss_tot = torch.sum((all_targets - torch.mean(all_targets)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return {
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2.item()
    }

###############################################################################
# 7) Main Script
###############################################################################

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # 1) Load the QM9 dataset from PyG
    dataset = QM9(root='data/QM9')

    # 2) Precompute fi_fj concurrently + cache
    cache_path = "fi_fj_cache.pkl"
    data_list = maybe_compute_fi_fj(dataset, cache_path)

    # 3) Shuffle + slice
    random.shuffle(data_list)
    data_list = data_list[:130831]
    train_list = data_list[:104665]
    test_list = data_list[104665:]
    # Use a subset for faster training if you want
    # data_list = data_list[:10000]
    # train_list = data_list[:8000]
    # test_list = data_list[8000:]

    # 4) Create DataLoaders
    train_loader = DataLoader(train_list, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_list, batch_size=32, shuffle=False)

    in_channels = dataset.num_node_features
    hidden_channels = 64
    epochs = 1000

    # Set up your models
    base_models = {
        "GCN": GCNModel(in_channels, hidden_channels),
        "GAT": GATModel(in_channels, hidden_channels),
        "GIN": GINModel(in_channels, hidden_channels)
    }
    sota_models = {
        "PAMNet": PAMNetModel(in_channels, hidden_channels),
        "MXMNet": MXMNetModel(in_channels, hidden_channels),
        "ComENet": ComENetModel(in_channels, hidden_channels)
    }
    all_models = {**base_models, **sota_models}

    combined_loss_histories = {}
    results = {}

    for model_name, model in all_models.items():
        print(f"\nTraining {model_name} model...")
        run = init_neptune()  # remove if not using Neptune

        # Log model name to Neptune
        run["parameters/model_name"] = model_name
        run["sys/tags"].add(model_name)

        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        train_losses = []
        test_losses = []

        # Some models (ComENet, PAMNet) need positional data
        use_pos = model_name in ["ComENet", "PAMNet"]

        for epoch in range(1, epochs + 1):
            t_loss = train(model, train_loader, criterion, optimizer, device, use_pos)
            te_loss = test(model, test_loader, criterion, device, use_pos)

            train_losses.append(t_loss)
            test_losses.append(te_loss)

            # Evaluate full metrics each epoch
            metrics = evaluate_full(model, test_loader, device, use_pos)

            print(f"{model_name} - Epoch {epoch:3d}: "
                  f"Train Loss = {t_loss:.4f}, Test Loss = {te_loss:.4f}")

            # Neptune logging
            run["training/loss"].append(t_loss)
            run["training/val_loss"].append(te_loss)
            run["training/metrics/mse"].append(metrics["MSE"])
            run["training/metrics/mae"].append(metrics["MAE"])
            run["training/metrics/rmse"].append(metrics["RMSE"])
            run["training/metrics/r2"].append(metrics["R2"])

        # Final evaluation
        final_metrics = evaluate_full(model, test_loader, device, use_pos)
        results[model_name] = final_metrics
        combined_loss_histories[model_name] = (train_losses, test_losses)

        # Log final metrics
        run["evaluation/mse"] = final_metrics["MSE"]
        run["evaluation/mae"] = final_metrics["MAE"]
        run["evaluation/rmse"] = final_metrics["RMSE"]
        run["evaluation/r2"] = final_metrics["R2"]
        run["evaluation/final_test_loss"] = test_losses[-1]
        run["evaluation/summary"] = (
            f"MSE: {final_metrics['MSE']:.4f}, "
            f"MAE: {final_metrics['MAE']:.4f}, "
            f"RMSE: {final_metrics['RMSE']:.4f}, "
            f"R2: {final_metrics['R2']:.4f}"
        )
        run.stop()
        print(f"Neptune run for {model_name} stopped. Metrics logged.")

    # Plot combined train/test curves
    plt.figure(figsize=(10, 6))
    for model_name, (train_losses, test_losses) in combined_loss_histories.items():
        plt.plot(range(1, epochs + 1), train_losses, 
                 label=f"{model_name} Train Loss", linestyle="--")
        plt.plot(range(1, epochs + 1), test_losses, 
                 label=f"{model_name} Test Loss", linestyle="-")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Combined Train and Test Loss for All Models")
    plt.legend()
    plt.tight_layout()
    plot_filename = "combined_loss_curves.png"
    plt.savefig(plot_filename)
    print("\nSaved combined loss curves plot as", plot_filename)

if __name__ == "__main__":
    main()
