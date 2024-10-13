#!/usr/bin/env bash

# Check if wget is installed
if ! command -v wget &> /dev/null
then
    echo "Error: wget is not installed. Please install it and try again."
    exit 1
fi

# Create checkpoints directory if it doesn't exist
mkdir -p checkpoints
mkdir -p data_input
mkdir -p data_output

# Define the URLs and target file paths
MODEL1_URL="https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt"
MODEL1_PATH="checkpoints/depth_pro.pt"

MODEL2_URL="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt"
MODEL2_PATH="checkpoints/yolo11s.pt"

# Check if the first model already exists to avoid redundant download
if [[ -f "$MODEL1_PATH" ]]; then
    echo "Model already exists at $MODEL1_PATH. Skipping download."
else
    echo "Downloading model from $MODEL1_URL..."
    
    # Download the first model and handle errors
    if wget "$MODEL1_URL" -P checkpoints; then
        echo "Model downloaded successfully to $MODEL1_PATH."
    else
        echo "Error: Failed to download the model. Please check the URL or your internet connection."
        exit 1
    fi
fi

# Check if the second model already exists to avoid redundant download
if [[ -f "$MODEL2_PATH" ]]; then
    echo "Model already exists at $MODEL2_PATH. Skipping download."
else
    echo "Downloading model from $MODEL2_URL..."
    
    # Download the second model and handle errors
    if wget "$MODEL2_URL" -P checkpoints; then
        echo "Model downloaded successfully to $MODEL2_PATH."
    else
        echo "Error: Failed to download the model. Please check the URL or your internet connection."
        exit 1
    fi
fi
