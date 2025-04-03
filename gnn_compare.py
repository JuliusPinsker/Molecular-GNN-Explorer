import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
# Use non-interactive backend for matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool, MessagePassing

import neptune

# Set random seed for reproducibility
torch.manual_seed(69)

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

### Basic Models (GCN, GAT, GIN)

class GCNModel(nn.Module):
    """
    A basic GCN model for molecular property prediction.
    Uses two GCNConv layers followed by global mean pooling.
    """
    def __init__(self, in_channels, hidden_channels):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = nn.Linear(hidden_channels // 2, 1)  # Regress to U0 by default

    def forward(self, x, edge_index, batch):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = torch.relu(self.lin1(x))
        return self.lin2(x)

class GATModel(nn.Module):
    """
    A basic GAT model for molecular property prediction.
    Uses two GATConv layers followed by global mean pooling.
    """
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
    """
    A basic GIN model for molecular property prediction.
    Uses two GINConv layers followed by global mean pooling.
    """
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

### PAMNet Model
# PAMNet is based on a physics-aware multiplex architecture (Zhang et al., 2023).
# It separately models local (covalent) and global (non-covalent) interactions.
# Here, we implement a simplified version with a global branch (distance-based)
# and a local branch (angular/torsional).
class PAMConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(PAMConv, self).__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels)
        # MLP to encode local positional differences (3 -> out_channels)
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
        # Compute local positional difference
        delta_pos = pos_i - pos_j
        delta_embed = self.pos_mlp(delta_pos)
        # Use a simple attention mechanism to combine
        att_input = torch.cat([x_i, x_j], dim=-1)
        att_weight = torch.sigmoid(self.att(att_input))
        return att_weight * (x_j + delta_embed)

class PAMNetModel(nn.Module):
    """
    PAMNet (Zhang et al., 2023) in a simplified form.
    A global branch uses GCNConv for distance-based interactions,
    while a local branch uses PAMConv for angle/torsion-based interactions.
    """
    def __init__(self, in_channels, hidden_channels):
        super(PAMNetModel, self).__init__()
        # Global: distance-based
        self.global_conv = GCNConv(in_channels, hidden_channels)
        # Local: angle/torsion-based
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

### MXMNet Model
# MXMNet (Zhang et al., 2020) uses a molecular mechanics-inspired approach.
# It constructs a two-layer multiplex graph: a local layer (angles) and a global layer (distances).
# Here, we show a simplified parallel approach with two GCNConv streams.
class MXMNetModel(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(MXMNetModel, self).__init__()
        # Local path: angle-aware (in a simplified manner here)
        self.local_conv = GCNConv(in_channels, hidden_channels)
        # Global path: distance-based
        self.global_conv = GCNConv(in_channels, hidden_channels)
        # Fuse local/global features
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

### ComENet Model
# ComENet (Wang et al., 2022) is designed for complete and efficient 3D message passing.
# It uses a 4-tuple (d, θ, φ, τ) in the 1-hop neighborhood to achieve full completeness in O(nk).
# Below is a simplified version capturing that 4-tuple per edge.
class ComENetConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(ComENetConv, self).__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels)
        # MLP to encode the 4-tuple (d, θ, φ, τ)
        self.geo_mlp = nn.Sequential(
            nn.Linear(4, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x, edge_index, pos):
        x = self.lin(x)
        # For each edge, compute distance, angles, and rotation angle τ
        pos_i = pos[edge_index[0]]
        pos_j = pos[edge_index[1]]
        rel = pos_j - pos_i
        d = torch.norm(rel, p=2, dim=-1, keepdim=True)  # [E, 1]
        eps = 1e-8
        # Polar angle (θ) from z-axis
        theta = torch.acos(torch.clamp(rel[:, 2:3] / (d + eps), -1+eps, 1-eps))  # [E, 1]
        # Azimuth (φ) from x,y
        phi = torch.atan2(rel[:, 1:2], rel[:, 0:1])  # [E, 1]

        # Compute rotation angle τ using nearest neighbors for each endpoint
        from collections import defaultdict
        neighbors = defaultdict(list)
        src, dst = edge_index
        for i_a, j_a in zip(src.tolist(), dst.tolist()):
            neighbors[i_a].append(j_a)

        tau_list = []
        for i_a, j_a in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            neigh_i = [k for k in neighbors[i_a] if k != j_a]
            if len(neigh_i) > 0:
                dists = [torch.norm(pos[i_a] - pos[k]).item() for k in neigh_i]
                fi = neigh_i[dists.index(min(dists))]
            else:
                fi = i_a
            neigh_j = [k for k in neighbors[j_a] if k != i_a]
            if len(neigh_j) > 0:
                dists = [torch.norm(pos[j_a] - pos[k]).item() for k in neigh_j]
                fj = neigh_j[dists.index(min(dists))]
            else:
                fj = j_a

            v1 = pos[i_a] - pos[fi]
            v2 = pos[j_a] - pos[i_a]
            v3 = pos[j_a] - pos[fj]
            n1 = torch.cross(v1, v2)
            n2 = torch.cross(v2, v3)
            norm_n1 = torch.norm(n1) + eps
            norm_n2 = torch.norm(n2) + eps
            cos_tau = torch.clamp(torch.dot(n1, n2) / (norm_n1 * norm_n2), -1+eps, 1-eps)
            tau = torch.acos(cos_tau)
            tau_list.append(tau.unsqueeze(0))

        tau_tensor = torch.cat(tau_list, dim=0).to(x.device)  # shape [E], then unsqueeze to [E,1]
        tau_tensor = tau_tensor.unsqueeze(-1)
        # Combine all geometry into shape [E,4]
        geo_features = torch.cat([d, theta, phi, tau_tensor], dim=-1)
        geo_emb = self.geo_mlp(geo_features)
        return self.propagate(edge_index, x=x, geo_emb=geo_emb)

    def message(self, x_j, geo_emb):
        return x_j + geo_emb

class ComENetModel(nn.Module):
    """
    ComENet (Wang et al., 2022): 
    A complete and efficient 3D message passing approach 
    using a 4-tuple (d, θ, φ, τ) in the 1-hop neighborhood.
    """
    def __init__(self, in_channels, hidden_channels):
        super(ComENetModel, self).__init__()
        self.conv1 = ComENetConv(in_channels, hidden_channels)
        self.conv2 = ComENetConv(hidden_channels, hidden_channels)
        self.self_atom = nn.Linear(hidden_channels, hidden_channels)  # self-update
        self.lin1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = nn.Linear(hidden_channels // 2, 1)

    def forward(self, x, edge_index, batch, pos):
        x = torch.relu(self.conv1(x, edge_index, pos))
        x = torch.relu(self.conv2(x, edge_index, pos))
        x = torch.relu(self.self_atom(x))
        x = global_mean_pool(x, batch)
        x = torch.relu(self.lin1(x))
        return self.lin2(x)

### Training, Testing, and Evaluation Functions

def train(model, loader, criterion, optimizer, device, use_pos=False):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        target = data.y[:, 0].view(-1, 1)
        if use_pos:
            out = model(data.x, data.edge_index, data.batch, data.pos)
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
                out = model(data.x, data.edge_index, data.batch, data.pos)
            else:
                out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, target)
            total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def evaluate_full(model, loader, device, use_pos=False):
    """
    Compute regression metrics: MSE, MAE, RMSE, and R².
    Returns a dict with these four entries.
    """
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            target = data.y[:, 0].view(-1, 1)
            if use_pos:
                out = model(data.x, data.edge_index, data.batch, data.pos)
            else:
                out = model(data.x, data.edge_index, data.batch)
            all_preds.append(out.cpu())
            all_targets.append(target.cpu())
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    mse = nn.MSELoss()(all_preds, all_targets).item()
    mae = nn.L1Loss()(all_preds, all_targets).item()
    rmse = math.sqrt(mse)
    ss_res = torch.sum((all_targets - all_preds)**2)
    ss_tot = torch.sum((all_targets - torch.mean(all_targets))**2)
    r2 = 1 - ss_res / ss_tot
    return {"MSE": mse, "MAE": mae, "RMSE": rmse, "R2": r2.item()}

### Main Function

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Load the QM9 dataset (automatically downloaded by PyTorch Geometric)
    dataset = QM9(root='data/QM9')

    # Optionally use a subset for faster training (e.g. 10000 molecules)
    # dataset = dataset.shuffle()[:10000]
    # train_dataset = dataset[:8000]
    # test_dataset = dataset[8000:]
    
    # Otherwise use the full dataset
    dataset = dataset.shuffle()[:130831]
    train_dataset = dataset[:104665]
    test_dataset = dataset[104665:]
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    in_channels = dataset.num_node_features
    hidden_channels = 64
    epochs = 1000

    # Basic models
    base_models = {
        "GCN": GCNModel(in_channels, hidden_channels),
        "GAT": GATModel(in_channels, hidden_channels),
        "GIN": GINModel(in_channels, hidden_channels)
    }
    # PAMNet (Zhang et al., 2023), MXMNet (Zhang et al., 2020), ComENet (Wang et al., 2022)
    sota_models = {
        "PAMNet": PAMNetModel(in_channels, hidden_channels),
        "MXMNet": MXMNetModel(in_channels, hidden_channels),
        # "ComENet": ComENetModel(in_channels, hidden_channels)
    }
    all_models = {**base_models, **sota_models}
    
    combined_loss_histories = {}
    results = {}
    for model_name, model in all_models.items():
        print(f"\nTraining {model_name} model...")
        run = init_neptune()
        # Set the model name in parameters and add as a tag
        run["parameters/model_name"] = model_name
        run["sys/tags"].add(model_name)  # Use 'add' instead of 'append'
        
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        train_losses = []
        test_losses = []
        use_pos = model_name in ["PAMNet", "ComENet"]
        
        for epoch in range(1, epochs + 1):
            t_loss = train(model, train_loader, criterion, optimizer, device, use_pos)
            te_loss = test(model, test_loader, criterion, device, use_pos)
            train_losses.append(t_loss)
            test_losses.append(te_loss)
            print(f"{model_name} - Epoch {epoch}: Train Loss: {t_loss:.4f}, Test Loss: {te_loss:.4f}")
            run["training/loss"].append(t_loss)
            run["training/val_loss"].append(te_loss)
        
        # Evaluate on the test set and log additional metrics
        eval_metrics = evaluate_full(model, test_loader, device, use_pos)
        results[model_name] = eval_metrics
        combined_loss_histories[model_name] = (train_losses, test_losses)
        run["evaluation/mse"] = eval_metrics["MSE"]
        run["evaluation/mae"] = eval_metrics["MAE"]
        run["evaluation/rmse"] = eval_metrics["RMSE"]
        run["evaluation/r2"] = eval_metrics["R2"]
        run["evaluation/final_test_loss"] = test_losses[-1]
        run["evaluation/summary"] = (
            f"MSE: {eval_metrics['MSE']:.4f}, "
            f"MAE: {eval_metrics['MAE']:.4f}, "
            f"RMSE: {eval_metrics['RMSE']:.4f}, "
            f"R2: {eval_metrics['R2']:.4f}"
        )
        run.stop()
        print(f"Neptune run for {model_name} stopped. Metrics logged.")

    
    # Combined loss plots
    plt.figure(figsize=(10, 6))
    for model_name, (train_losses, test_losses) in combined_loss_histories.items():
        plt.plot(range(1, epochs + 1), train_losses, label=f"{model_name} Train Loss", linestyle="--")
        plt.plot(range(1, epochs + 1), test_losses, label=f"{model_name} Test Loss", linestyle="-")
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
