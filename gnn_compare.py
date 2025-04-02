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
import os

# Set random seed for reproducibility (updated to 69)
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

### Define three model classes

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
        nn1 = nn.Sequential(nn.Linear(in_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
        self.conv1 = GINConv(nn1)
        nn2 = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
        self.conv2 = GINConv(nn2)
        self.lin1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = nn.Linear(hidden_channels // 2, 1)
    def forward(self, x, edge_index, batch):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = torch.relu(self.lin1(x))
        return self.lin2(x)

### Training and testing functions

def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        # Predict U0 (index 0 of data.y)
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
    epochs = 10

    # Define a dictionary of models to compare
    models = {
        "GCN": GCNModel(in_channels, hidden_channels),
        "GAT": GATModel(in_channels, hidden_channels),
        "GIN": GINModel(in_channels, hidden_channels)
    }

    criterion = nn.MSELoss()
    results = {}
    loss_histories = {}

    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name} model...")
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train_losses = []
        test_losses = []
        for epoch in range(1, epochs+1):
            t_loss = train(model, train_loader, criterion, optimizer, device)
            te_loss = test(model, test_loader, criterion, device)
            train_losses.append(t_loss)
            test_losses.append(te_loss)
            print(f"{name} - Epoch {epoch}: Train Loss: {t_loss:.4f}, Test Loss: {te_loss:.4f}")
            # Log per-epoch test loss in Neptune under each model branch
            run[f"models/{name}/train/loss"].append(t_loss)
            run[f"models/{name}/test/loss"].append(te_loss)
        results[name] = test_losses[-1]
        loss_histories[name] = (train_losses, test_losses)
        # Log final loss for this model
        run[f"models/{name}/final_test_loss"] = test_losses[-1]

    # Plot the test loss of each model over epochs
    plt.figure(figsize=(10, 6))
    for name, (train_losses, test_losses) in loss_histories.items():
        plt.plot(range(1, epochs+1), test_losses, label=f"{name} (final loss: {test_losses[-1]:.4f})")
    plt.xlabel("Epoch")
    plt.ylabel("Test MSE Loss")
    plt.title("Comparison of Test Loss over Epochs")
    plt.legend()
    plt.tight_layout()
    plot_filename = "comparison_loss.png"
    plt.savefig(plot_filename)
    print("\nSaved comparison plot as", plot_filename)
    
    # Upload the plot to Neptune
    run["plots/comparison_loss"].upload(plot_filename)
    
    # Log a dummy evaluation metric (F1 score, for demonstration)
    run["eval/f1_score"] = 0.66

    # Log final model names and performance
    for name, loss in results.items():
        run[f"models/{name}/summary"] = f"Final Test Loss: {loss:.4f}"

    run.stop()
    print("Neptune run stopped. Metrics logged.")

if __name__ == "__main__":
    main()
