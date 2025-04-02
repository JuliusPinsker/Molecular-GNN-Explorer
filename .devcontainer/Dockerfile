# Use the official PyTorch base image with CUDA support
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Install system dependencies required for building PyTorch Geometric packages
RUN apt-get update && apt-get install -y build-essential git && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /workspace

# The container will install Python dependencies via requirements.txt in the postCreateCommand.
