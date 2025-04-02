import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
# Use non-interactive backend for matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool

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

### Model Definitions

class GCNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = nn.Linear(hidden_channels // 2, 1)  # Regression output for U0

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
        # Define a simple MLP for the GINConv
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

### Training, Testing, and Evaluation Functions

def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        target = data.y[:, 0].view(-1, 1)
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def test(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            target = data.y[:, 0].view(-1, 1)
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, target)
            total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def evaluate_full(model, loader, device):
    """Compute regression metrics: MSE, MAE, and R² score."""
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            target = data.y[:, 0].view(-1, 1)
            out = model(data.x, data.edge_index, data.batch)
            all_preds.append(out.cpu())
            all_targets.append(target.cpu())
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    mse = nn.MSELoss()(all_preds, all_targets).item()
    mae = nn.L1Loss()(all_preds, all_targets).item()
    ss_res = torch.sum((all_targets - all_preds)**2)
    ss_tot = torch.sum((all_targets - torch.mean(all_targets))**2)
    r2 = 1 - ss_res / ss_tot
    return {"MSE": mse, "MAE": mae, "R2": r2.item()}

### Main Function

def main():
    run = init_neptune()

    # Use GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Load the QM9 dataset (automatically downloaded by PyTorch Geometric)
    dataset = QM9(root='data/QM9')
    # Use a subset for faster training (e.g., 5000 molecules)
    dataset = dataset.shuffle()[:5000]
    train_dataset = dataset[:4000]
    test_dataset = dataset[4000:]
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    in_channels = dataset.num_node_features
    hidden_channels = 64
    epochs = 50

    # Define models to compare
    models = {
        "GCN": GCNModel(in_channels, hidden_channels),
        "GAT": GATModel(in_channels, hidden_channels),
        "GIN": GINModel(in_channels, hidden_channels)
    }

    criterion = nn.MSELoss()
    results = {}
    loss_histories = {}

    for name, model in models.items():
        print(f"\nTraining {name} model...")
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train_losses = []
        test_losses = []
        for epoch in range(1, epochs + 1):
            t_loss = train(model, train_loader, criterion, optimizer, device)
            te_loss = test(model, test_loader, criterion, device)
            train_losses.append(t_loss)
            test_losses.append(te_loss)
            print(f"{name} - Epoch {epoch}: Train Loss: {t_loss:.4f}, Test Loss: {te_loss:.4f}")
            # Log per-epoch loss values to Neptune
            run[f"models/{name}/train/loss"].append(t_loss)
            run[f"models/{name}/test/loss"].append(te_loss)
        
        # Compute additional regression metrics on the test set
        eval_metrics = evaluate_full(model, test_loader, device)
        results[name] = eval_metrics
        loss_histories[name] = (train_losses, test_losses)
        run[f"models/{name}/evaluation/mse"] = eval_metrics["MSE"]
        run[f"models/{name}/evaluation/mae"] = eval_metrics["MAE"]
        run[f"models/{name}/evaluation/r2"] = eval_metrics["R2"]
        run[f"models/{name}/final_test_loss"] = test_losses[-1]
        run[f"models/{name}/summary"] = (
            f"MSE: {eval_metrics['MSE']:.4f}, "
            f"MAE: {eval_metrics['MAE']:.4f}, "
            f"R2: {eval_metrics['R2']:.4f}"
        )
    # Plot training and test loss curves for each model
    plt.figure(figsize=(10, 6))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, (name, (train_losses, test_losses)) in enumerate(loss_histories.items()):
        color = colors[i % len(colors)]
        plt.plot(range(1, epochs + 1), train_losses, label=f"{name} Train Loss", linestyle="--", color=color)
        plt.plot(range(1, epochs + 1), test_losses, label=f"{name} Test Loss", linestyle="-", color=color)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Train and Test Loss over Epochs")
    plt.legend()
    plt.tight_layout()
    plot_filename = "loss_curves.png"
    plt.savefig(plot_filename)
    print("\nSaved loss curves plot as", plot_filename)
    run["plots/loss_curves"].upload(plot_filename)

    run.stop()
    print("Neptune run stopped. Metrics logged.")

if __name__ == "__main__":
    main()
