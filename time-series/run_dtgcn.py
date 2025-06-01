from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

from seiz_eeg.dataset import EEGDataset

import os
import random
from datetime import datetime

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.data import Data

import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse

import wandb
from sklearn.metrics import f1_score
from tqdm import tqdm

from src.dtgcn import DTGCN
from src.utils import load_config, load_eeg_data, load_graph, apply_smote_to_eeg_dataset


def seed_everything(seed: int):
    # Python random module
    random.seed(seed)
    # Numpy random module
    np.random.seed(seed)
    # Torch random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # Set PYTHONHASHSEED environment variable for hash-based operations
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Ensure deterministic behavior in cudnn (may slow down your training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(config: dict):
    seed_everything(config["seed"])

    dataset_tr, dataset_val, train_df = load_eeg_data(
        config["data_path"],
        config["train_parquet_file"],
        config["val_parquet_file"],
        config["signal_processing"]["filtering_type"],
        robust=config["val_robust"],
    )

    if config["training"]["smote"]:
        # Apply SMOTE to balance the training data
        dataset_tr = apply_smote_to_eeg_dataset(dataset_tr)

    loader_tr = DataLoader(
        dataset_tr, batch_size=config["training"]["batch_size"], shuffle=True
    )
    loader_val = DataLoader(
        dataset_val, batch_size=config["training"]["batch_size"], shuffle=False
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
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    lr = config["training"]["lr"]
    num_nodes = config["model"]["num_eeg_channels"]
    weight_decay = config["training"]["weight_decay"]
    tgcn_hidden_dim = config["model"]["tgcn_hidden_dim"]
    gsl_embed_dim = config["model"]["gsl_embed_dim"]
    window_size = config["model"]["window_size"]
    gsl_alpha = config["model"]["gsl_alpha"]
    gsl_avg_steps = config["model"]["gsl_avg_steps"]
    number_of_classes = 1
    max_norm = config["training"]["max_norm"]

    print("Training Model")

    # Initialize Weights and Biases
    wandb.init(
        project="eeg-DTGCN",
        name=f"{config['model']['name']}_{datetime.now().strftime('%Y%m%d')}",
        config={
            "data_type": config["signal_processing"]["filtering_type"],
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "model": config["model"]["name"],
            "tgcn_hidden_dim": tgcn_hidden_dim,
            "gsl_embed_dim": gsl_embed_dim,
            "window_size": window_size,
            "gsl_alpha": gsl_alpha,
            "gsl_avg_steps": gsl_avg_steps,
            "max_norm": max_norm,
            "warmup_ratio": config["training"]["warmup_ratio"]
            if config["training"]["lr_scheduler"]
            else None,
            "smote": config["training"]["smote"],
            "pos_weight": config["training"]["pos_weight"],
        },
    )

    model = DTGCN(
        num_nodes=num_nodes,
        gsl_embed_dim=gsl_embed_dim,
        tgcn_hidden_dim=tgcn_hidden_dim,
        window_size=window_size,
        gsl_alpha=gsl_alpha,
        gsl_avg_steps=gsl_avg_steps,
    ).to(device)
    print(
        "Number of trainable parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    label_counts = train_df["label"].value_counts()
    neg, pos = label_counts[0], label_counts[1]

    # Inverse frequency weighting for BCEWithLogitsLoss
    pos_weight = (
        torch.tensor([neg / pos], dtype=torch.float32).to(device)
        if config["training"]["pos_weight"]
        else None
    )
    if config["training"]["pos_weight"]:
        print(
            f"Using pos_weight={pos_weight.item():.4f} for class imbalance correction."
        )  # helps with balancing class imablance by giving more weight to the class that appears less

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Training loop
    best_acc = 0.0
    best_f1 = 0.0
    ckpt_path = os.path.join(wandb.run.dir, config["checkpoint"]["best_model_filename"])
    print(f"Model path is: {ckpt_path}")
    global_step = 0

    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        running_loss = 0.0

        for x_batch, y_batch in loader_tr:
            x_batch = (
                x_batch.float().to(device).transpose(-2, -1)
            )  # [batch_size, num_nodes, seq_len]
            y_batch = y_batch.float().unsqueeze(1).to(device)

            logits = model(x_batch, edge_index, edge_weight)
            loss = criterion(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()

            # Log total gradient norm (before clipping to really see what is happening with the model)
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

            wandb.log({"grad_norm": total_norm, "global_step": global_step})
            global_step += 1

            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(loader_tr)

        # Evaluation phase for train accuracy
        model.eval()
        correct = 0
        total = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x_batch, y_batch in loader_tr:
                x_batch = x_batch.float().to(device).transpose(-2, -1)
                y_batch = y_batch.float().unsqueeze(1).to(device)

                logits = model(x_batch, edge_index, edge_weight)
                preds = torch.sigmoid(logits) >= 0.5
                correct += (preds == y_batch.bool()).sum().item()
                total += y_batch.size(0)

                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(y_batch.cpu().numpy().flatten())

        train_acc = correct / total
        train_f1 = f1_score(all_labels, all_preds, average="macro")
        print(f"Total training points: {total}")
        print(
            f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_loss:.4f}, Train accuracy: {train_acc:.4f}, Train F1: {train_f1:.4f}"
        )
        wandb.log(
            {
                "train_loss": avg_loss,
                "train_accuracy": train_acc,
                "train_f1": train_f1,
                "epoch": epoch + 1,
            }
        )

        val_correct = 0
        val_total = 0
        val_loss = 0.0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x_batch, y_batch in loader_val:
                x_batch = x_batch.float().to(device).transpose(-2, -1)
                y_batch = y_batch.float().unsqueeze(1).to(device)

                logits = model(x_batch, edge_index, edge_weight)
                loss = criterion(logits, y_batch)
                val_loss += loss.item()

                preds = torch.sigmoid(logits) >= 0.5
                val_correct += (preds == y_batch.bool()).sum().item()
                val_total += y_batch.size(0)

                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(y_batch.cpu().numpy().flatten())

        val_acc = val_correct / val_total
        val_loss /= len(loader_val)
        val_f1 = f1_score(all_labels, all_preds, average="macro")
        print(f"Total validation points: {val_total}")

        print(
            f"Epoch [{epoch + 1}/{epochs}], Validation Loss: {val_loss:.4f}, Valid accuracy: {val_acc:.4f}, Validation F1: {val_f1:.4f}"
        )
        wandb.log(
            {
                "eval/loss": val_loss,
                "eval/accuracy": val_acc,
                "eval/f1": val_f1,
                "epoch": epoch + 1,
            }
        )

        # Should we also use here f1 instead of accuracy?
        # Save model if best accuracy so far
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), ckpt_path)
            print(
                f"✅ New best model saved with f1: {val_f1:.4f} at epoch {epoch + 1}"
            )

    # Save the model
    torch.save(
        model.state_dict(),
        os.path.join(wandb.run.dir, config["checkpoint"]["final_model_filename"]),
    )

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DTGCN")

    parser.add_argument("--config", required=True, help="Path to yaml config file")

    args = parser.parse_args()

    config = load_config(args.config)

    main(config)