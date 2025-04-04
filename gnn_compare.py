#!/usr/bin/env python
import os
import math
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
# Use non-interactive backend for matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool, MessagePassing

import neptune
import optuna
from optuna.samplers import TPESampler
import optuna.visualization as vis

# ------------------------
# Global Configuration
# ------------------------
CONFIG = {
    "available_models": ["GCN", "GAT", "GIN", "PAMNet", "MXMNet"],
    "hyperopt_ranges": {
         "learning_rate": {"min": 1e-4, "max": 1e-2},
         "hidden_channels": {"min": 32, "max": 128}
    },
    "default": {
         "max_data_trial": 5000,        # For hyperparameter optimization
         "max_data_training": -1,         # -1 means use the full dataset for final training
         "trial_epochs": 1000,            # Number of epochs per trial during hyperopt
         "training_epochs": 1000,        # Number of epochs for final training
         "split_ratio": 0.8,
         "batch_size": 32
    },
    "num_trials": 5  # Number of Optuna trials for hyperparameter optimization
}

# Create temporary folder for checkpoints and plots
TMP_FOLDER = ".tmp"
if not os.path.exists(TMP_FOLDER):
    os.makedirs(TMP_FOLDER)

# ------------------------
# Argument Parser
# ------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Molecular GNN Explorer with Optuna Hyperparameter Optimization and Final Training"
    )
    parser.add_argument(
        "--models", type=str, default="all",
        help="Comma-separated list of models to run or 'all'"
    )
    parser.add_argument(
        "--max_data_trial", type=int, default=CONFIG["default"]["max_data_trial"],
        help="Maximum number of data points for hyperopt trials (-1 for full dataset)"
    )
    parser.add_argument(
        "--max_data_training", type=int, default=CONFIG["default"]["max_data_training"],
        help="Maximum number of data points for final training (-1 for full dataset)"
    )
    parser.add_argument(
        "--trial_epochs", type=int, default=CONFIG["default"]["trial_epochs"],
        help="Number of epochs per trial during hyperparameter optimization"
    )
    parser.add_argument(
        "--training_epochs", type=int, default=CONFIG["default"]["training_epochs"],
        help="Number of epochs for final training"
    )
    parser.add_argument(
        "--batch_size", type=int, default=CONFIG["default"]["batch_size"],
        help="Batch size for training"
    )
    parser.add_argument(
        "--hyperopt", action="store_true",
        help="Enable hyperparameter optimization using Optuna"
    )
    return parser.parse_args()

# ------------------------
# Neptune Initialization
# ------------------------
def init_neptune(additional_tags=None, extra_params=None):
    api_token = os.environ.get("NEPTUNE_API_TOKEN")
    if api_token is None:
        raise ValueError("NEPTUNE_API_TOKEN environment variable not set!")
    run = neptune.init_run(
        project="happyproject235/Molecular-GNN-Explorer",
        api_token=api_token
    )
    params = {"optimizer": "Adam", "seed": 69}
    if extra_params is not None:
        params.update(extra_params)
    run["parameters"] = params
    if additional_tags is not None:
        for tag in additional_tags:
            run["sys/tags"].add(tag)
    return run

# ------------------------
# Model Definitions
# ------------------------
###################################
# Basic Models (GCN, GAT, GIN)
###################################
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

###################################
# PAMNet Model
###################################
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

###################################
# MXMNet Model
###################################
class MXMNetModel(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(MXMNetModel, self).__init__()
        self.local_conv = GCNConv(in_channels, hidden_channels)
        self.global_conv = GCNConv(in_channels, hidden_channels)
        self.fusion = nn.Linear(2 * hidden_channels, hidden_channels)
        self.lin1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = nn.Linear(hidden_channels // 2, 1)

    def forward(self, x, edge_index, batch, pos):
        local = torch.relu(self.local_conv(x, edge_index))
        global_ = torch.relu(self.global_conv(x, edge_index))
        combined = torch.cat([local, global_], dim=-1)
        fused = torch.relu(self.fusion(combined))
        pooled = global_mean_pool(fused, batch)
        x = torch.relu(self.lin1(pooled))
        return self.lin2(x)

# ------------------------
# Training and Evaluation Functions
# ------------------------
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
    ss_res = torch.sum((all_targets - all_preds) ** 2)
    ss_tot = torch.sum((all_targets - torch.mean(all_targets)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return {"MSE": mse, "MAE": mae, "RMSE": rmse, "R2": r2.item()}

# ------------------------
# Optuna Objective Function with Extended Neptune Tagging and Checkpointing
# ------------------------
def optuna_objective(trial, model_class, model_name, in_channels, device, train_loader, test_loader, use_pos, trial_epochs, max_data_trial):
    # Extended tags for this trial
    trial_tags = [
        f"model:{model_name}",
        "hyperopt",
        "trial_phase",
        f"optuna_trial:{trial.number}",
        f"trial_epochs:{trial_epochs}",
        f"max_data_trial:{max_data_trial}"
    ]
    extra_params = {
        "phase": "trial",
        "model_name": model_name,
        "trial_number": trial.number,
        "trial_epochs": trial_epochs,
        "max_data_trial": max_data_trial
    }
    trial_run = init_neptune(additional_tags=trial_tags, extra_params=extra_params)
    
    # Suggest hyperparameters using Bayesian optimization
    lr = trial.suggest_float("learning_rate", 
                             CONFIG["hyperopt_ranges"]["learning_rate"]["min"],
                             CONFIG["hyperopt_ranges"]["learning_rate"]["max"], log=True)
    hidden_channels = trial.suggest_int("hidden_channels",
                                        CONFIG["hyperopt_ranges"]["hidden_channels"]["min"],
                                        CONFIG["hyperopt_ranges"]["hidden_channels"]["max"])
    trial_run["parameters/hyperparams"] = {"learning_rate": lr, "hidden_channels": hidden_channels}
    
    # Instantiate model and optimizer
    model_instance = model_class(in_channels, hidden_channels).to(device)
    optimizer = optim.Adam(model_instance.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_epoch_loss = float('inf')
    checkpoint_filename = os.path.join(TMP_FOLDER, f"checkpoint_trial_{trial.number}.pt")
    
    # Train for the specified number of trial epochs
    for epoch in range(1, trial_epochs + 1):
        t_loss = train(model_instance, train_loader, criterion, optimizer, device, use_pos)
        te_loss = test(model_instance, test_loader, criterion, device, use_pos)
        eval_metrics = evaluate_full(model_instance, test_loader, device, use_pos)
        
        # Update best checkpoint if current validation loss is lower
        if te_loss < best_epoch_loss:
            best_epoch_loss = te_loss
            torch.save(model_instance, checkpoint_filename)
        
        # Log per-epoch metrics
        trial_run["trial/train/loss"].append(t_loss)
        trial_run["trial/val/loss"].append(te_loss)
        trial_run["trial/val/mse"].append(eval_metrics["MSE"])
        trial_run["trial/val/mae"].append(eval_metrics["MAE"])
        trial_run["trial/val/rmse"].append(eval_metrics["RMSE"])
        trial_run["trial/val/r2"].append(eval_metrics["R2"])
        print(f"Trial {trial.number} - Epoch {epoch}: Train Loss: {t_loss:.4f}, Val Loss: {te_loss:.4f}")
    
    # Upload the best checkpoint of this trial to Neptune
    trial_run["model_checkpoints/best_checkpoint"].upload(checkpoint_filename)
    final_val_loss = test(model_instance, test_loader, criterion, device, use_pos)
    trial_run["trial/final_val_loss"] = final_val_loss
    trial_run.stop()
    return final_val_loss

# ------------------------
# Main Training Loop
# ------------------------
def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    
    # Define model groups
    base_models = {
        "GCN": GCNModel,
        "GAT": GATModel,
        "GIN": GINModel
    }
    sota_models = {
        "PAMNet": PAMNetModel,
        "MXMNet": MXMNetModel,
    }
    all_models = {**base_models, **sota_models}
    
    if args.models.lower() == "all":
        selected_models = all_models
    else:
        selected = [m.strip() for m in args.models.split(",")]
        selected_models = {name: model for name, model in all_models.items() if name in selected}

    combined_loss_histories = {}
    final_results = {}
    study = None  # To hold Optuna study if hyperopt enabled

    for model_name, model_class in selected_models.items():
        print(f"\nProcessing model: {model_name}")
        use_pos = model_name in ["PAMNet", "MXMNet"]
        
        # --------------------
        # Hyperparameter Optimization (Trial Phase)
        # --------------------
        best_params = {"learning_rate": 0.001, "hidden_channels": 64}  # defaults if hyperopt is disabled
        if args.hyperopt:
            trial_dataset = QM9(root='data/QM9')
            trial_dataset = trial_dataset.shuffle()
            if args.max_data_trial != -1:
                trial_dataset = trial_dataset[:args.max_data_trial]
            split_ratio = CONFIG["default"]["split_ratio"]
            train_size = int(len(trial_dataset) * split_ratio)
            trial_train_dataset = trial_dataset[:train_size]
            trial_test_dataset = trial_dataset[train_size:]
            trial_train_loader = DataLoader(trial_train_dataset, batch_size=args.batch_size, shuffle=True)
            trial_test_loader = DataLoader(trial_test_dataset, batch_size=args.batch_size, shuffle=False)
            
            study = optuna.create_study(sampler=TPESampler(seed=69), direction="minimize")
            objective = lambda trial: optuna_objective(
                trial, model_class, model_name, trial_dataset.num_node_features, device,
                trial_train_loader, trial_test_loader, use_pos, args.trial_epochs, args.max_data_trial
            )
            print(f"Starting Optuna optimization for {model_name} with {CONFIG['num_trials']} trials...")
            study.optimize(objective, n_trials=CONFIG["num_trials"])
            best_params = study.best_params
            best_params["hidden_channels"] = int(best_params["hidden_channels"])
            print(f"Best hyperparameters for {model_name}: {best_params}")
            
            # Log summary of best hyperparameters and save Optuna plots to temp folder
            summary_tags = [f"model:{model_name}", "hyperopt", "trial_phase", "summary"]
            run_summary = init_neptune(additional_tags=summary_tags,
                                        extra_params={"phase": "trial_summary", "model_name": model_name})
            run_summary["parameters/best_hyperparams"] = best_params
            fig_history = vis.plot_optimization_history(study)
            fig_importance = vis.plot_param_importances(study)
            history_path = os.path.join(TMP_FOLDER, f"{model_name}_opt_history.png")
            importance_path = os.path.join(TMP_FOLDER, f"{model_name}_opt_param_importances.png")
            fig_history.write_image(history_path)
            fig_importance.write_image(importance_path)
            run_summary["plots/optimization_history"].upload(history_path)
            run_summary["plots/param_importances"].upload(importance_path)
            run_summary.stop()
        
        # --------------------
        # Final Training Phase
        # --------------------
        final_tags = [
            f"model:{model_name}", "final_training", f"training_epochs:{args.training_epochs}",
            f"max_data_training:{args.max_data_training}" if args.max_data_training != -1 else "full_data"
        ]
        run_final = init_neptune(
            additional_tags=final_tags,
            extra_params={"phase": "final_training", "model_name": model_name, "training_epochs": args.training_epochs}
        )
        
        final_dataset = QM9(root='data/QM9')
        final_dataset = final_dataset.shuffle()
        if args.max_data_training != -1:
            final_dataset = final_dataset[:args.max_data_training]
        train_size = int(len(final_dataset) * CONFIG["default"]["split_ratio"])
        final_train_dataset = final_dataset[:train_size]
        final_test_dataset = final_dataset[train_size:]
        final_train_loader = DataLoader(final_train_dataset, batch_size=args.batch_size, shuffle=True)
        final_test_loader = DataLoader(final_test_dataset, batch_size=args.batch_size, shuffle=False)
        
        # If hyperopt was enabled, load the best checkpoint from the best trial.
        if args.hyperopt and study is not None:
            best_trial_number = study.best_trial.number
            checkpoint_filename = os.path.join(TMP_FOLDER, f"checkpoint_trial_{best_trial_number}.pt")
            model = torch.load(checkpoint_filename)
            model = model.to(device)
        else:
            model = model_class(final_dataset.num_node_features, best_params["hidden_channels"]).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=best_params.get("learning_rate", 0.001))
        criterion = nn.MSELoss()
        
        train_losses = []
        test_losses = []
        for epoch in range(1, args.training_epochs + 1):
            t_loss = train(model, final_train_loader, criterion, optimizer, device, use_pos)
            te_loss = test(model, final_test_loader, criterion, device, use_pos)
            train_losses.append(t_loss)
            test_losses.append(te_loss)
            run_final["train/loss"].append(t_loss)
            eval_metrics = evaluate_full(model, final_test_loader, device, use_pos)
            run_final["val/loss"].append(te_loss)
            run_final["val/mse"].append(eval_metrics["MSE"])
            run_final["val/mae"].append(eval_metrics["MAE"])
            run_final["val/rmse"].append(eval_metrics["RMSE"])
            run_final["val/r2"].append(eval_metrics["R2"])
            print(f"{model_name} - Epoch {epoch}: Train Loss: {t_loss:.4f}, Val Loss: {te_loss:.4f}")
        final_metrics = evaluate_full(model, final_test_loader, device, use_pos)
        final_results[model_name] = final_metrics
        combined_loss_histories[model_name] = (train_losses, test_losses)
        
        # Save combined loss curves plot to temp folder and upload to final run
        plt.figure(figsize=(10, 6))
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i, (m_name, (tr_losses, ts_losses)) in enumerate(combined_loss_histories.items()):
            color = colors[i % len(colors)]
            plt.plot(range(1, args.training_epochs + 1), tr_losses, label=f"{m_name} Train Loss", linestyle="--", color=color)
            plt.plot(range(1, args.training_epochs + 1), ts_losses, label=f"{m_name} Test Loss", linestyle="-", color=color)
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.title("Combined Train and Test Loss for Final Training Runs")
        plt.legend()
        plt.tight_layout()
        combined_plot_path = os.path.join(TMP_FOLDER, "combined_loss_curves_final.png")
        plt.savefig(combined_plot_path)
        run_final["val/combined_loss_curves"].upload(combined_plot_path)
        plt.close()

        # If hyperopt was enabled, also upload the Optuna plots to the final run
        if args.hyperopt and study is not None:
            fig_history = vis.plot_optimization_history(study)
            fig_importance = vis.plot_param_importances(study)
            opt_history_path = os.path.join(TMP_FOLDER, f"{model_name}_opt_history_final.png")
            opt_importance_path = os.path.join(TMP_FOLDER, f"{model_name}_opt_param_importances_final.png")
            fig_history.write_image(opt_history_path)
            fig_importance.write_image(opt_importance_path)
            run_final["val/opt_history"].upload(opt_history_path)
            run_final["val/opt_param_importances"].upload(opt_importance_path)
        
        run_final.stop()
        print(f"Final Neptune run for {model_name} stopped. Metrics logged.")
        
    # Delete the temporary folder after all runs are completed
    if os.path.exists(TMP_FOLDER):
        shutil.rmtree(TMP_FOLDER)
        print(f"Temporary folder '{TMP_FOLDER}' deleted.")

if __name__ == "__main__":
    main()
