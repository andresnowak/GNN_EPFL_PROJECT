from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

from seiz_eeg.dataset import EEGDataset

import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

import wandb

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from models import GAT

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


seed_everything(1)


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

data_path = "/work/cvlab/students/bhagavan/GNN_EPFL_PROJECT/nml_project/data/train"
DATA_ROOT = Path(data_path)
clips_tr = pd.read_parquet(DATA_ROOT / "train/segments.parquet")

# Stratified split based on labels
train_df, val_df = train_test_split(clips_tr, test_size=0.2, random_state=1, stratify=clips_tr['label'])

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
    signal_transform=fft_filtering,
    prefetch=True,
)

dataset_val = EEGDataset(
    val_df,
    signals_root=DATA_ROOT / "train",
    signal_transform=fft_filtering,
    prefetch=True,
)

"""## Compatibility with PyTorch

The `EEGDataset` class is compatible with [pytorch datasets and dataloaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html), which allow you to load batched data.
"""

batch_size = 64
loader_tr = DataLoader(dataset_tr, batch_size=batch_size, shuffle=True)
loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

"""## Baseline: LSTM model for sequential data

In this section, we provide a simple baseline for the project using an LSTM model without any special optimization.
"""

"""# LSTM + GNN"""

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define node list (in order, matching your image)
nodes = [
    'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
    'T3', 'C3', 'CZ', 'C4', 'T4',
    'T5', 'P3', 'PZ', 'P4', 'T6',
    'O1', 'O2'
]

# Define edge list
edges = [
    ('FP1', 'F7'), ('FP1', 'F3'), ('FP1', 'FP2'), ('FP1', 'FZ'),
    ('FP2', 'F4'), ('FP2', 'F8'), ('FP2', 'FZ'),
    ('F7', 'F3'), ('F3', 'FZ'), ('FZ', 'F4'), ('F4', 'F8'),
    ('F7', 'T3'), ('F3', 'C3'), ('FZ', 'CZ'), ('F4', 'C4'), ('F8', 'T4'),
    ('T3', 'C3'), ('C3', 'CZ'), ('CZ', 'C4'), ('C4', 'T4'),
    ('T3', 'T5'), ('C3', 'P3'), ('CZ', 'PZ'), ('C4', 'P4'), ('T4', 'T6'),
    ('T5', 'P3'), ('P3', 'PZ'), ('PZ', 'P4'), ('P4', 'T6'),
    ('T5', 'O1'), ('P3', 'O1'), ('PZ', 'O1'), ('PZ', 'O2'), ('P4', 'O2'), ('T6', 'O2'),
    ('O1', 'O2')
]

# Create a mapping from node names to indices
node_idx = {node: i for i, node in enumerate(nodes)}

# Load distances
dist_df = pd.read_csv('/work/cvlab/students/bhagavan/GNN_EPFL_PROJECT/nml_project/distances_3d.csv')
distance_dict = {}

for _, row in dist_df.iterrows():
    key1 = (row['from'], row['to'])
    key2 = (row['to'], row['from'])  # ensure symmetry
    distance_dict[key1] = row['distance']
    distance_dict[key2] = row['distance']
    
epsilon = 1e-6
edge_weights = []
edge_index = []

for u, v in edges:
    dist = distance_dict.get((u, v), 1.0)  # fallback if missing
    weight = 1.0 / (dist + epsilon)
    edge_weights.append(weight)  # u→v
    edge_weights.append(weight)  # v→u
    edge_index.append((node_idx[u], node_idx[v]))  # u→v
    edge_index.append((node_idx[v], node_idx[u]))  # v→u
    
edge_weight = torch.tensor(edge_weights, dtype=torch.float32).to(device)
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)

epochs = 500
lr = 7.5e-5
weight_decay = 1e-5
lstm_hidden_dim = 128
lstm_num_layers = 3
gtf_hidden = 256
gtf_out = 256
num_heads = 16
dropout = 0.2
pos_enc_dim = 8

# Initialize Weights and Biases
wandb.init(project="eeg-lstm-gat", config={
    "epochs": epochs,
    "batch_size": batch_size,
    "lr": lr,
    "weight_decay": weight_decay,
    "model": "LSTM + gtf",
    "lstm_hidden_dim": lstm_hidden_dim,
    "lstm_num_layers": lstm_num_layers,
    "dropout": dropout,
    "num_heads": num_heads,
    "gtf_hidden": gtf_hidden,
    "gtf_out": gtf_out,
    "pos_enc_dim": pos_enc_dim
})

model = LSTM_GraphTransformer(lstm_hidden_dim, lstm_num_layers, gtf_hidden, gtf_out, num_heads, dropout, pos_enc_dim).to(device)  # binary classification
print('Number of trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
print('Number of parameters in LSTM:', sum(p.numel() for p in model.lstm.parameters() if p.requires_grad))
print('Number of parameters in GraphTransformers:', sum(p.numel() for p in model.gtf1.parameters() if p.requires_grad) + sum(p.numel() for p in model.gtf2.parameters() if p.requires_grad) + sum(p.numel() for p in model.gtf3.parameters() if p.requires_grad))

# label_counts = train_df['label'].value_counts()
# neg, pos = label_counts[0], label_counts[1]

# # # Inverse frequency weighting for BCEWithLogitsLoss
# pos_weight = torch.tensor([neg / pos], dtype=torch.float32).to(device)
# print(f"Using pos_weight={pos_weight.item():.4f} for class imbalance correction.")

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# Training loop
best_f1 = 0.0
ckpt_path = os.path.join(wandb.run.dir, "best_lstm_gtf_model.pth")
print(f'Model path is: {ckpt_path}')
global_step = 0

global_min = float('inf')
global_max = float('-inf')

for x, _ in loader_tr:
    batch_min = x.min().item()
    batch_max = x.max().item()
    global_min = min(global_min, batch_min)
    global_max = max(global_max, batch_max)

print(f"Global min: {global_min}, max: {global_max}")
normalization_stats = {
    "global_min": global_min,
    "global_max": global_max
}
torch.save(normalization_stats, os.path.join(wandb.run.dir, "normalization_stats.pth"))

epsilon = 1e-8

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for x_batch, y_batch in loader_tr:
        x_batch = x_batch.float().to(device)
        y_batch = y_batch.float().unsqueeze(1).to(device)
        x_batch = 2 * (x_batch - global_min) / (global_max - global_min + epsilon) - 1

        logits = model(x_batch, edge_index, edge_weight)
        loss = criterion(logits, y_batch)

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Log total gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        wandb.log({"grad_norm": total_norm, "global_step": global_step})
        global_step += 1

        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(loader_tr)

    # Evaluation phase for train accuracy
    model.eval()
    correct = 0
    total = 0
    train_preds = []
    train_labels = []
    with torch.no_grad():
        for x_batch, y_batch in loader_tr:
            x_batch = x_batch.float().to(device)
            x_batch = 2 * (x_batch - global_min) / (global_max - global_min + epsilon) - 1
            y_batch = y_batch.float().unsqueeze(1).to(device)

            logits = model(x_batch, edge_index, edge_weight)
            preds = torch.sigmoid(logits) >= 0.5
            correct += (preds == y_batch.bool()).sum().item()
            total += y_batch.size(0)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(y_batch.cpu().numpy())

    train_acc = correct / total
    train_f1 = f1_score(train_labels, train_preds, zero_division=0)
    print(f'Total training points: {total}')
    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.4f}, Train accuracy: {train_acc:.4f}, Train F1: {train_f1:.4f}")
    wandb.log({"train_loss": avg_loss, "train_accuracy": train_acc, "train_f1": train_f1, "epoch": epoch + 1})
    
    val_correct = 0
    val_total = 0
    val_loss = 0.0
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for x_batch, y_batch in loader_val:
            x_batch = x_batch.float().to(device)
            x_batch = 2 * (x_batch - global_min) / (global_max - global_min + epsilon) - 1
            y_batch = y_batch.float().unsqueeze(1).to(device)

            logits = model(x_batch, edge_index, edge_weight)
            loss = criterion(logits, y_batch)
            val_loss += loss.item()

            preds = torch.sigmoid(logits) >= 0.5
            val_correct += (preds == y_batch.bool()).sum().item()
            val_total += y_batch.size(0)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(y_batch.cpu().numpy())

    val_acc = val_correct / val_total
    val_f1 = f1_score(val_labels, val_preds, zero_division=0)
    val_loss /= len(loader_val)
    print(f'Total validation points: {val_total}')
    print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss:.4f}, Valid accuracy: {val_acc:.4f}, Valid F1: {val_f1:.4f}")
    wandb.log({"val_loss": val_loss, "val_accuracy": val_acc, "val_f1": val_f1, "epoch": epoch + 1})
    
    # Save model if best accuracy so far
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), ckpt_path)
        print(f"✅ New best model saved with F1: {val_f1:.4f} at epoch {epoch+1}")

wandb.finish()

# Save the model
torch.save(model.state_dict(), os.path.join(wandb.run.dir, "final_lstm_gat_model.pth"))
