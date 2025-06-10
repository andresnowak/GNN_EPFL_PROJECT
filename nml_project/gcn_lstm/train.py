import numpy as np
import torch.optim as optim
from pathlib import Path
import pandas as pd
from seiz_eeg.dataset import EEGDataset
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score
import wandb

from constants import INCLUDED_CHANNELS
import utils
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '..'))
from models import GCN_LSTM_Model, GCN2_LSTM_Model

########### CONFIG ################
wandblog = False # If you want to log the results of wandb
graph_type = 'gcnlstm' # Change it to 'gcnlstm' or 'gcn2lstm'
########### CONFIG ################

def train_model(model, clips_tr, loader_tr, loader_va, num_epochs, lr, device):

    # Calculate the class imbalance to be used in weighted loss.
    label_counts = clips_tr['label'].value_counts()
    neg, pos = label_counts[0], label_counts[1]
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32).to(device)
    if wandblog:
        wandb.config.pos_weight = pos_weight
    print(f"Using pos_weight={pos_weight.item():.4f} for class imbalance correction.")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    epochs = num_epochs # Total epochs.
    optimizer = optim.Adam(model.parameters(), lr=lr) # Adam Optimizer used.
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs) # Learning rate decay.

    train_losses = []
    best_ckpt_path = os.path.join(script_dir, "best_gnn_lstm_model.pth")

    global_step = 0
    best_macrof1 = 0 # Store the best Macro F1 score.

    for epoch in tqdm(range(epochs), desc="Training"):

        # Turn on the model's training mode.
        model.train()
        running_loss = 0.0

        # Per epoch training loop
        for x_batch, y_batch in loader_tr:
            x_batch = x_batch.float().unsqueeze(-1).to(device)
            y_batch = y_batch.float().unsqueeze(1).to(device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            running_loss += loss.item()

            # Log the loss to wandb.
            if wandblog:
                wandb.log({"loss": loss.item(), "global_step": global_step})
            print(f"Step {global_step}, Loss: {loss.item():.4f}")
            global_step += 1

        avg_loss = running_loss / len(loader_tr)
        train_losses.append(avg_loss)

        # Evaluation phase on validation data.

        # Turn on the model's evaluation mode.
        model.eval()

        with torch.no_grad():
            y_pred_all = []
            y_true_all = []
            va_loss = 0.0
            for x_batch, y_batch in loader_va:
                x_batch = x_batch.float().unsqueeze(-1).to(device)
                y_batch = y_batch.float().unsqueeze(1).to(device)
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                va_loss += loss.item()

                # 0.5 threshold used for binary classification.
                preds = torch.sigmoid(logits) >= 0.5
                y_pred_all.append(preds)
                y_true_all.append(y_batch.bool())

        # Track the validation loss to know overfitting or underfitting.
        va_loss /= len(loader_va)
        y_pred_all = torch.flatten(torch.concatenate(y_pred_all, axis = 0))
        y_true_all = torch.flatten(torch.concatenate(y_true_all, axis = 0))

        # Calculate the MacroF1 score on the validation set.
        macrof1 = f1_score(y_true_all.cpu(), y_pred_all.cpu(), average='macro')
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Macrof1: {macrof1:.4f}")
        if wandblog:
            wandb.log({"val_loss": va_loss, "train_loss": avg_loss, "val_Macrof1": macrof1, "epoch": epoch + 1})
        
        # Save model if best MacroF1 so far
        if macrof1 > best_macrof1:
            best_macrof1 = macrof1
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"✅ New best model saved with macrof1: {macrof1:.4f}")

        # Learning Rate Decay Step.
        scheduler.step()

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

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Seed everything for reproduction of results
seed_everything(1)

# You might need to change this according to where you store the data folder
# Inside your data folder, you should have the following structure:
# data
# ├── train
# │   ├── signals/
# │   ├── segments.parquet
# │-- test
#     ├── signals/
#     ├── segments.parquet

data_path = Path(os.path.join(script_dir, ".."))
DATA_ROOT = Path(data_path)

# Load the train, validation split.
clips_tr = pd.read_parquet(DATA_ROOT / "segments_train.parquet")
clips_va = pd.read_parquet(DATA_ROOT / "segments_val.parquet")

dataset_tr = EEGDataset(
clips_tr,
signals_root=DATA_ROOT /"data"/ "train",
signal_transform=utils.fft_filtering, # Frequency domain transformation
prefetch=True,  # If your compute does not allow it, you can use `prefetch=False`
)

dataset_va = EEGDataset(
    clips_va,
    signals_root=DATA_ROOT /"data"/ "train",
    signal_transform=utils.fft_filtering, # Frequency domain transformation
    prefetch=True,  # If your compute does not allow it, you can use `prefetch=False`
)  

# Initialize the data loaders
batch_size = 128
loader_tr = DataLoader(dataset_tr, batch_size=batch_size, shuffle=True)
loader_va = DataLoader(dataset_va, batch_size=batch_size, shuffle=False)

# Compute the Distance Adjacency Matrix and make it sparse using threshold 0.9
thresh = 0.9
dist_df = pd.read_csv('distances_3d.csv')
A = utils.get_adjacency_matrix(dist_df, INCLUDED_CHANNELS, dist_k=thresh)

# Initialize the model parameters.
num_epochs = 200 # Total epochs.
num_nodes = 19 # Number of electrodes.
in_channels = 1 # Each electrode has one dimension.
output_dim = 1 # Seizure or not.
weighted_loss = True # We use weighted loss to balance class imbalance.
seq_len = 354 # Length of the frequency series.
gradient_clip = True # Clip very large gradients.

# For initializing the model with GCN_LSTM.
# gcnlstm = one layer of gcn is used per time step.
# gcn2lstm = two layers of gcn are used per time step.
if graph_type == 'gcnlstm' or graph_type == 'gcn2lstm':

    # Best Configuration of gcnlstm
    gcn_hidden = 16 # The output dimension of the gcn.
    lstm_hidden = 128 # The hidden variable dimension of lstm.
    lstm_layers = 3 # Number of lstm layers.
    dropout = 0 # Amount of dropout.
    lr = 2e-4 # Learning rate.

    if graph_type == 'gcnlstm':
        model = GCN_LSTM_Model(
                num_nodes,
                in_channels,
                gcn_hidden,
                lstm_hidden,
                output_dim,
                lstm_layers,
                torch.from_numpy(A),
                dropout,
                seq_len
                ).to(device)
        
    elif graph_type == 'gcn2lstm':
        # Best Configuration of gcn2lstm
        lstm_hidden = 64 # The hidden variable dimension of lstm.
        model = GCN2_LSTM_Model(
                num_nodes,
                in_channels,
                gcn_hidden,
                lstm_hidden,
                output_dim,
                lstm_layers,
                torch.from_numpy(A),
                dropout,
                seq_len
                ).to(device)
        
    # Whether you want to log your plots on wandb.
    if wandblog:
        wandb.init(project="gcn-lstm", config={
            "num_nodes": num_nodes,
            "in_channels": in_channels,
            "gcn_hidden": gcn_hidden,
            "lstm_hidden": lstm_hidden,
            "lstm_layers": lstm_layers,
            "epochs": num_epochs,
            "lr": lr,
            "dropout": dropout,
            "batch_size": batch_size,
            "weighted_loss": weighted_loss,
            "seq_len": seq_len,
            "gradient_clip": gradient_clip
        })

print('Number of trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

# Start the training of the model.
train_model(model, clips_tr, loader_tr, loader_va, num_epochs, lr, device)