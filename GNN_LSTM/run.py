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

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import f1_score

import wandb
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse
from model import GCN_LSTM_Model
from utils import get_adjacency_matrix

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

data_path = "/content/drive/MyDrive/EPFL_NetworkML/epfl-network-machine-learning-2025/"

DATA_ROOT = Path(data_path)

clips_tr = pd.read_parquet(DATA_ROOT / "train/segments_train.parquet")
clips_va = pd.read_parquet(DATA_ROOT / "train/segments_val.parquet")


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

# You can change the signal_transform, or remove it completely
print('Loading Train Data Start')
dataset_tr = EEGDataset(
    clips_tr,
    signals_root=DATA_ROOT / "train",
    signal_transform=fft_filtering,
    prefetch=False,  # If your compute does not allow it, you can use `prefetch=False`
)
print("Loading Train Data End")

print('Loading Val Data Start')
dataset_va = EEGDataset(
    clips_va,
    signals_root=DATA_ROOT / "train",
    signal_transform=fft_filtering,
    prefetch=False,  # If your compute does not allow it, you can use `prefetch=False`
)
print("Loading Val Data End")

"""## Compatibility with PyTorch

The `EEGDataset` class is compatible with [pytorch datasets and dataloaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html), which allow you to load batched data.
"""
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
batch_size = 128
loader_tr = DataLoader(dataset_tr, batch_size=batch_size, shuffle=True)
loader_va = DataLoader(dataset_va, batch_size=batch_size, shuffle=False)

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

INCLUDED_CHANNELS = [
    'FP1',
    'FP2',
    'F3',
    'F4',
    'C3',
    'C4',
    'P3',
    'P4',
    'O1',
    'O2',
    'F7',
    'F8',
    'T3',
    'T4',
    'T5',
    'T6',
    'FZ',
    'CZ',
    'PZ'
]

thresh = 0.9
dist_df = pd.read_csv(DATA_ROOT /'distances_3d.csv')
A, sensor_id_to_ind = get_adjacency_matrix(dist_df, INCLUDED_CHANNELS, dist_k=thresh)

# --------------------------
# Model, Loss, Optimizer
# --------------------------

num_nodes = 19
in_channels = 1
gcn_hidden = 16
lstm_hidden = 32
lstm_layers = 2
output_dim = 1
adj = torch.from_numpy(A)
epochs = 100
lr = 1e-4

# Initialize Weights and Biases
wandb.init(project="gnn-lstm", config={
    "num_nodes": num_nodes,
    "in_channels": in_channels,
    "gcn_hidden": gcn_hidden,
    "lstm_hidden": lstm_hidden,
    "lstm_layers": lstm_layers,
    "epochs": epochs,
    "lr": lr,
    "batch_size": batch_size
})

model = GCN_LSTM_Model(
        num_nodes,
        in_channels,
        gcn_hidden,
        lstm_hidden,
        output_dim,
        lstm_layers,
        adj
        ).to(device)
print('Number of trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

criterion = nn.BCEWithLogitsLoss()

# --------------------------
# Training Loop
# --------------------------
train_losses = []

ckpt_path = os.path.join(wandb.run.dir, "best_gnn_lstm_model.pth")

global_step = 0
best_macrof1 = 0

for epoch in tqdm(range(epochs), desc="Training"):
    model.train()
    running_loss = 0.0

    for x_batch, y_batch in loader_tr:
        x_batch = x_batch.float().unsqueeze(-1).to(device)
        y_batch = y_batch.float().unsqueeze(1).to(device)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        wandb.log({"loss": loss.item(), "global_step": global_step})
        print(f"Step {global_step}, Loss: {loss.item():.4f}")
        global_step += 1

    avg_loss = running_loss / len(loader_tr)
    train_losses.append(avg_loss)

    # Evaluation phase for train accuracy
    model.eval()

    with torch.no_grad():
        y_pred_all = []
        y_true_all = []
        for x_batch, y_batch in loader_va:
            x_batch = x_batch.float().unsqueeze(-1).to(device)
            y_batch = y_batch.float().unsqueeze(1).to(device)
            logits = model(x_batch)
            preds = torch.sigmoid(logits) >= 0.5
            y_pred_all.append(preds)
            y_true_all.append(y_batch.bool())

    y_pred_all = torch.flatten(torch.concatenate(y_pred_all, axis = 0))
    y_true_all = torch.flatten(torch.concatenate(y_true_all, axis = 0))
    macrof1 = f1_score(y_true_all.cpu(), y_pred_all.cpu(), average='macro')
    print(y_pred_all.shape, y_true_all.shape)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Macrof1: {macrof1:.4f}")
    wandb.log({"loss": avg_loss, "Macrof1": macrof1, "epoch": epoch + 1})
    
    # Save model if best accuracy so far
    if macrof1 > best_macrof1:
        best_macrof1 = macrof1
        torch.save(model.state_dict(), ckpt_path)
        print(f"✅ New best model saved with macrof1: {macrof1:.4f}")

    scheduler.step()

    torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                }, os.path.join("/home/chaurasi/networkml/gnn-lstm/wandb/Continue/best_diffusion_gnn_model.pth"))

wandb.finish()