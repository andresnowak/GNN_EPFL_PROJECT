from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

from seiz_eeg.dataset import EEGDataset

import os
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
from models import DTGCN

import wandb
from sklearn.model_selection import train_test_split


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


# seed_everything(1)


"""# The data

We model *segments* of brain activity, which correspond to windows of a longer *session* of EEG recording.

These segments, and their labels, are described in the `segments.parquet` files, which can be directly loaded with `pandas`.
"""

# You might need to change this according to where you store the data folder
# Inside your data folder, you should have the following structure:
# data
# ├── train
# │   ├── signals/
# │   ├── segments.parquet
# │-- test
#     ├── signals/
#     ├── segments.parquet

print("Loading Data")

data_path = "../data"  # we use relative path
DATA_ROOT = Path(data_path)
clips_tr = pd.read_parquet(DATA_ROOT / "train/segments.parquet")

# Stratified split based on labels
train_df, val_df = train_test_split(
    clips_tr, test_size=0.1, random_state=1, stratify=clips_tr["label"]
)

"""## Loading the signals

For convenience, the `EEGDataset class` provides functionality for loading each segment and its label as `numpy` arrays.

You can provide an optional `signal_transform` function to preprocess the signals. In the example below, we have two bandpass filtering functions, which extract frequencies between 0.5Hz and 30Hz which are used in seizure analysis literature:

The `EEGDataset` class also allows to load all data in memory, instead of reading it from disk at every iteration. If your compute allows it, you can use `prefetch=True`.
"""
bp_filter = signal.butter(4, (0.5, 30), btype="bandpass", output="sos", fs=250)


def time_filtering(x: np.ndarray) -> np.ndarray:
    """Filter signal in the time domain"""
    return signal.sosfiltfilt(bp_filter, x, axis=0).copy()


def fft_filtering(x: np.ndarray) -> np.ndarray:
    """Compute FFT and only keep"""
    x = np.abs(np.fft.fft(x, axis=0))
    x = np.log(np.where(x > 1e-8, x, 1e-8))
    win_len = x.shape[0]
    # Only frequencies b/w 0.5 and 30Hz
    return x[int(0.5 * win_len // 250) : 30 * win_len // 250]


# Create training and validation datasets
dataset_tr = EEGDataset(
    train_df,
    signals_root=DATA_ROOT / "train",
    signal_transform=time_filtering,
    prefetch=True,
)

dataset_val = EEGDataset(
    val_df,
    signals_root=DATA_ROOT / "train",
    signal_transform=time_filtering,
    prefetch=True,
)

"""## Compatibility with PyTorch

The `EEGDataset` class is compatible with [pytorch datasets and dataloaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html), which allow you to load batched data.
"""

batch_size = 128
loader_tr = DataLoader(dataset_tr, batch_size=batch_size, shuffle=True)
loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)


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
distances_df = pd.read_csv(DATA_ROOT / "distances_3d.csv")
distances_df.head()

src, dst, weights = [], [], []

nodes_names = distances_df["from"].unique()
N = len(distances_df["from"].unique())

for i in range(N):
    for j in range(N):
        src.append(i)
        dst.append(j)

        if i != j:
            nodes_distance = distances_df[
                (distances_df["from"] == nodes_names[i])
                & (distances_df["to"] == nodes_names[j])
            ]["distance"]
            weights.append(nodes_distance.item())
        else:
            weights.append(
                1
            )  # how much distance do we want for the self loop, do we use the original 0 or?


edge_index = torch.tensor([src, dst], dtype=torch.long).to(device)
edge_weight = torch.tensor(weights, dtype=torch.float32).to(device)


epochs = 100
lr = 5e-5
weight_decay = 1e-5
number_of_nodes = len(edge_index)
gsl_embed_dim = 100
tgcn_hidden_dim = 200
window_size = 50
gsl_alpha = 1
gsl_average_steps = 1
number_of_classes = 1

print("Training Model")

# Initialize Weights and Biases
wandb.init(
    project="eeg-DTGCN",
    name="DTGCN",
    config={
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "model": "DTGCN",
        "gsl_embed_dim": gsl_embed_dim,
        "tgcn_hidden_dim": tgcn_hidden_dim,
        "window_size": window_size,
        "gsl_alpha": gsl_alpha,
        "gsl_average_steps": gsl_average_steps,
    },
)

model = DTGCN(
    number_of_nodes,
    gsl_embed_dim,
    tgcn_hidden_dim,
    window_size,
    gsl_alpha,
    gsl_average_steps,
    number_of_classes,
    device,
)  # binary classification
print(
    "Number of trainable parameters:",
    sum(p.numel() for p in model.parameters() if p.requires_grad),
)

label_counts = train_df["label"].value_counts()
neg, pos = label_counts[0], label_counts[1]

# # Inverse frequency weighting for BCEWithLogitsLoss
pos_weight = torch.tensor([neg / pos], dtype=torch.float32).to(device)
print(f"Using pos_weight={pos_weight.item():.4f} for class imbalance correction.")

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)

# Training loop
best_acc = 0.0
ckpt_path = os.path.join(wandb.run.dir, "best_dtgcn_model.pth")
print(f"Model path is: {ckpt_path}")
global_step = 0

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for x_batch, y_batch in loader_tr:
        x_batch = (
            x_batch.float().to(device).transpose(-2, -1)
        )  # [batch_size, seq_len, input_dim]
        y_batch = y_batch.float().unsqueeze(1).to(device)

        logits = model(x_batch, edge_index, edge_weight)
        loss = criterion(logits, y_batch)

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Log total gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5
        wandb.log({"grad_norm": total_norm, "global_step": global_step})
        global_step += 1

        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(loader_tr)

    # Evaluation phase for train accuracy
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x_batch, y_batch in loader_tr:
            x_batch = x_batch.float().to(device)
            y_batch = y_batch.float().unsqueeze(1).to(device)

            logits = model(x_batch, edge_index)
            preds = torch.sigmoid(logits) >= 0.5
            correct += (preds == y_batch.bool()).sum().item()
            total += y_batch.size(0)

    train_acc = correct / total
    print(f"Total training points: {total}")
    print(
        f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_loss:.4f}, Train accuracy: {train_acc:.4f}"
    )
    wandb.log({"train_loss": avg_loss, "train_accuracy": train_acc, "epoch": epoch + 1})

    val_correct = 0
    val_total = 0
    val_loss = 0.0

    with torch.no_grad():
        for x_batch, y_batch in loader_val:
            x_batch = x_batch.float().to(device)
            y_batch = y_batch.float().unsqueeze(1).to(device)

            logits = model(x_batch, edge_index)
            loss = criterion(logits, y_batch)
            val_loss += loss.item()

            preds = torch.sigmoid(logits) >= 0.5
            val_correct += (preds == y_batch.bool()).sum().item()
            val_total += y_batch.size(0)

    val_acc = val_correct / val_total
    val_loss /= len(loader_val)
    print(f"Total validation points: {val_total}")

    print(
        f"Epoch [{epoch + 1}/{epochs}], Validation Loss: {val_loss:.4f}, Valid accuracy: {val_acc:.4f}"
    )
    wandb.log({"val_loss": val_loss, "valid_accuracy": val_acc, "epoch": epoch + 1})

    # Save model if best accuracy so far
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), ckpt_path)
        print(
            f"✅ New best model saved with accuracy: {val_acc:.4f} at epoch {epoch + 1}"
        )

wandb.finish()

# Save the model
torch.save(model.state_dict(), os.path.join(wandb.run.dir, "final_lstm_gcn_model.pth"))
