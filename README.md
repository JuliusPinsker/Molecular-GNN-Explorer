# Molecular Engineering with GNNs – Model Comparison using QM9 and Neptune

## Overview
This project demonstrates a molecular engineering application using Graph Neural Networks (GNNs) on the QM9 dataset. Several state-of-the-art GNN architectures are compared:

- **GCN (Graph Convolutional Network)**
- **GAT (Graph Attention Network)**
- **GIN (Graph Isomorphism Network)**
- **PAMNet** (Zhang et al., 2023)  
- **MXMNet** (Zhang et al., 2020)  
- **ComENet** (Wang et al., 2022)  

Each model is trained to predict the internal energy at 0K (U₀) in a regression setting. All key hyperparameters, training losses, evaluation metrics, model summaries, and plots are tracked using Neptune.

> **Note**: Only **simplified versions** of PAMNet, MXMNet, and ComENet are implemented. In particular, **ComENet** is *very heavy* because it computes detailed geometric features (distance, polar angle, azimuth, rotation angle) for each edge in the 1‐hop neighborhood. This can significantly increase the training time on large datasets.

**Check out my collection of already completed experiments on the Neptune dashboard [here](https://app.neptune.ai/happyproject235/Molecular-GNN-Explorer/runs).**

### Loss Curves Visualization
Below is an example of the training loss curves for some models, providing a quick overview of their performance during training.

![Loss Curves](loss_curves.png)

## Development Environment
This project is configured as a devcontainer so that you can open it in Visual Studio Code (or another compatible editor) with a reproducible environment. The devcontainer uses a GPU-ready PyTorch base image:
**pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime**

The NEPTUNE_API_TOKEN is copied from your local environment into the container automatically.

## How to Use
1. **Set your Neptune API Token:**
   - Ensure you have your Neptune API token set in your local environment as `NEPTUNE_API_TOKEN`.

2. **Open in a Dev Container:**
   - Install the [Remote - Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) for VS Code.
   - Reopen the project folder in a container (via the Command Palette: "Remote-Containers: Reopen in Container").

3. **Run the Model Comparison:**
   Once inside the container, run:
   ```bash
   python gnn_compare.py
   ```
   This script will download the QM9 dataset (if not already cached), train multiple models, and log all key metrics and plots to Neptune.

## Dependencies
The project relies on:
- PyTorch and PyTorch Geometric (for GNNs)
- Matplotlib (for plotting)
- Neptune (for experiment tracking)

All dependencies are listed in `requirements.txt` and are installed automatically in the devcontainer.

## Dataset Note
The QM9 dataset is automatically downloaded by PyTorch Geometric and cached under `data/QM9`. For this demo, you can optionally reduce the dataset size to speed up training (e.g., by uncommenting the shuffle[:N] lines in the script).

## References

- **PAMNet** – *A universal framework for accurate and efficient geometric deep learning of molecular systems.*  
  [Zhang et al., 2023](https://paperswithcode.com/paper/a-universal-framework-for-accurate-and-1)

- **MXMNet** – *Molecular Mechanics-Driven Graph Neural Network with Multiplex Graph for Molecular Structures.*  
  [Zhang et al., 2020](https://paperswithcode.com/paper/molecular-mechanics-driven-graph-neural)

- **ComENet** – *Towards complete and efficient message passing for 3D molecular graphs.*  
  [Wang et al., 2022](https://paperswithcode.com/paper/comenet-towards-complete-and-efficient)

- **QM9 Dataset**  
  [Link](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz)

- Fey, M., & Lenssen, J. E. (2019). *Fast Graph Representation Learning with PyTorch Geometric*.  
  [arXiv:1903.02428](https://arxiv.org/abs/1903.02428)

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
