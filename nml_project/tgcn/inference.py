from torch.utils.data import DataLoader
import torch
from scipy import signal
from seiz_eeg.dataset import EEGDataset
import numpy as np
import pandas as pd
from pathlib import Path
import re
import argparse
from datetime import datetime
import os
import sys

from src.utils import load_config, load_eeg_val_data, load_graph

script_dir = Path(__file__).parent.absolute()  # Folder containing this script
parent_dir = script_dir.parent                # One level up (same as your '..')
output_dir = script_dir / "outputs"
os.makedirs(output_dir, exist_ok=True)
sys.path.append(str(parent_dir)) 

from models import TGCNClassifier


def main(config, model_path):
    config["data_path"] = parent_dir.parent / "data"


    dataset_te= load_eeg_val_data(
        config["data_path"],
        "test/segments.parquet",
        config["signal_processing"]["filtering_type"],
    )

    loader_te = DataLoader(
        dataset_te, batch_size=config["training"]["batch_size"], shuffle=False
    )

    """# Model"""

    # Set up device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print("Using device:", device)

    # Load distances of nodes
    edge_index, edge_weight = load_graph(
        config["data_path"], config["distances_csv_file"]
    )
    edge_index = edge_index
    edge_weight = edge_weight

    # Create adjacency matrix
    num_nodes = edge_index.max().item() + 1  # Assuming nodes are 0-indexed
    adjacency_matrix = torch.zeros((num_nodes, num_nodes))

    # Fill the adjacency matrix
    adjacency_matrix[edge_index[0], edge_index[1]] = edge_weight

    num_nodes = config["model"]["num_eeg_channels"]
    tgcn_hidden_dim = config["model"]["tgcn_hidden_dim"]
    number_of_classes = 1

    model = TGCNClassifier(
        adj=adjacency_matrix,
        hidden_dim=tgcn_hidden_dim,
        num_classes=number_of_classes,
    ).to(device)
    print(
        "Number of trainable parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    model.load_state_dict(
        torch.load(
            model_path
        )
    )  # Load the trained model weights
    model.to(device)
    model.eval()

    # Lists to store sample IDs and predictions
    all_predictions = []
    all_ids = []

    def clean_underscores(s):
        # Step 1: Replace all double underscores (or more) with a temporary marker
        s = re.sub(r'__+', lambda m: '<<UND>>' * (len(m.group()) // 2), s)

        # Step 2: Remove remaining single underscores
        s = re.sub(r'_', '', s)

        # Step 3: Replace all temporary markers back with a single underscore each
        s = s.replace('<<UND>>', '_')

        return s

    # Disable gradient computation for inference
    with torch.no_grad():
        for batch in loader_te:
            # Assume each batch returns a tuple (x_batch, sample_id)
            # If your dataset does not provide IDs, you can generate them based on the batch index.
            x_batch, x_ids = batch
            x_ids = [clean_underscores(x_id) for x_id in x_ids]  # Clean the IDs

            # Move the input data to the device (GPU or CPU)
            x_batch = (
                x_batch.float().to(device)
            )  # [batch_size, seq_len, num_nodes]

            # Perform the forward pass to get the model's output logits
            logits = model(x_batch)

            # Convert logits to predictions.
            # For binary classification, threshold logits at 0 (adjust this if you use softmax or multi-class).
            predictions = (torch.sigmoid(logits) >= 0.5).int().cpu().numpy()

            # Append predictions and corresponding IDs to the lists
            all_predictions.extend(predictions.flatten().tolist())
            all_ids.extend(list(x_ids))

    # Create a DataFrame for Kaggle submission with the required format: "id,label"
    submission_df = pd.DataFrame({"id": all_ids, "label": all_predictions})

    # Save the DataFrame to a CSV file without an index
    submission_df.to_csv(
        f"{output_dir}/submission_stgcn_{datetime.now().strftime('%Y%m%d')}.csv", index=False
    )
    print("Kaggle submission file generated: submission.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STGCN")

    parser.add_argument("--config", required=False, help="Path to yaml config file", default=os.path.join(script_dir, "configs/tgcn_2.yaml"))
    parser.add_argument("--model", required=True, help="Path to model")

    args = parser.parse_args()

    config = load_config(args.config)

    main(config, args.model)