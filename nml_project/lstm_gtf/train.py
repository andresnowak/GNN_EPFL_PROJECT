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
from sklearn.metrics import f1_score
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '..'))
from models import LSTM_GraphTransformer

log_wandb = False

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(1)

data_path = Path(os.path.join(script_dir, ".."))
DATA_ROOT = Path(data_path)
clips_tr = pd.read_parquet(DATA_ROOT / "segments_train.parquet")
clips_va = pd.read_parquet(DATA_ROOT / "segments_val.parquet")

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
    clips_tr,
    signals_root= DATA_ROOT/"data"/"train",
    signal_transform=fft_filtering,
    prefetch=True,
)

dataset_val = EEGDataset(
    clips_va,
    signals_root=DATA_ROOT/"data"/"train",
    signal_transform=fft_filtering,
    prefetch=True,
)

batch_size = 64
loader_tr = DataLoader(dataset_tr, batch_size=batch_size, shuffle=True)
loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

nodes = [
    'FP1', 'FP2', 'F3', 'F4', 
    'C3', 'C4', 'P3', 'P4', 
    'O1', 'O2', 'F7', 'F8', 
    'T3', 'T4', 'T5','T6', 
    'FZ', 'CZ', 'PZ', 
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
dist_df = pd.read_csv(os.path.join(script_dir, '..', 'distances_3d.csv'))
distance_dict = {}

for _, row in dist_df.iterrows():
    key1 = (row['from'], row['to'])
    key2 = (row['to'], row['from'])
    distance_dict[key1] = row['distance']
    distance_dict[key2] = row['distance']
    
epsilon = 1e-6
edge_weights = []
edge_index = []

for u, v in edges:
    dist = distance_dict.get((u, v), 1.0)
    weight = 1.0 / (dist + epsilon)
    edge_weights.append(weight)  # u→v
    edge_weights.append(weight)  # v→u
    edge_index.append((node_idx[u], node_idx[v]))  # u→v
    edge_index.append((node_idx[v], node_idx[u]))  # v→u
    
edge_weight = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(-1).to(device)
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)

epochs = 300
lr = 3e-5
weight_decay = 1e-5
lstm_hidden_dim = 128
lstm_num_layers = 3
gtf_hidden = 272
gtf_out = 272
num_heads = 4
dropout = 0.2
pos_enc_dim = 16
weighted_loss = True
gradient_clip = True
normalization = False
learn_rate_scheduler = False

# Initialize Weights and Biases
if log_wandb:
    wandb.init(project="eeg-lstm-gtf", config={
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "model": "lstm + gtf",
        "lstm_hidden_dim": lstm_hidden_dim,
        "lstm_num_layers": lstm_num_layers,
        "gtf_hidden": gtf_hidden,
        "gtf_out": gtf_out,
        "num_heads": num_heads,
        "pos_enc_dim": pos_enc_dim,
        "dropout": dropout,
        "weighted_loss": weighted_loss,
        "gradient_clip": gradient_clip,
        "normalization": normalization,
        "learn_rate_scheduler": learn_rate_scheduler,
        "time_or_fft": "fft_filtering",
    })
    print("WandB initialized successfully.")

model = LSTM_GraphTransformer(lstm_hidden_dim, lstm_num_layers, gtf_hidden, gtf_out, num_heads, dropout, pos_enc_dim).to(device)
print('Number of trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
print('Number of parameters in LSTM:', sum(p.numel() for p in model.lstm.parameters() if p.requires_grad))
print('Number of parameters in GTF:', sum(p.numel() for p in model.gtf1.parameters() if p.requires_grad) + sum(p.numel() for p in model.gtf2.parameters() if p.requires_grad) + sum(p.numel() for p in model.gtf3.parameters() if p.requires_grad))
print("Model initialized. Starting training...")

if weighted_loss:
    label_counts = clips_tr['label'].value_counts()
    neg, pos = label_counts[0], label_counts[1]
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32).to(device)
    if log_wandb:
        wandb.config.pos_weight = pos_weight
    print(f"Using pos_weight={pos_weight.item():.4f} for class imbalance correction.")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
else:
    criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
if learn_rate_scheduler:
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=epochs)

# Training loop
best_f1 = 0.0
if log_wandb:
    ckpt_path = os.path.join(wandb.run.dir, "best_lstm_gtf_model.pth")
else:
    ckpt_path = os.path.join(script_dir, "best_lstm_gtf_model.pth")
global_step = 0
global_min = float('inf')
global_max = float('-inf')

if normalization:
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
    if log_wandb:
        torch.save(normalization_stats, os.path.join(wandb.run.dir, "normalization_stats.pth"))
    else:
        torch.save(normalization_stats, os.path.join(script_dir, "normalization_stats.pth"))
    epsilon = 1e-8

print("Starting training loop...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for i, (x_batch, y_batch) in enumerate(loader_tr):
        if (i + 1) % 20 == 0:
            print(f"Training batch {i + 1}/{len(loader_tr)}")
        x_batch = x_batch.float().to(device)
        y_batch = y_batch.float().unsqueeze(1).to(device)
        if normalization:
            x_batch = 2 * (x_batch - global_min) / (global_max - global_min + epsilon) - 1

        logits = model(x_batch, edge_index, edge_weight)
        loss = criterion(logits, y_batch)

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if gradient_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Log total gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        if log_wandb:
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
            if normalization:
                x_batch = 2 * (x_batch - global_min) / (global_max - global_min + epsilon) - 1
            y_batch = y_batch.float().unsqueeze(1).to(device)

            logits = model(x_batch, edge_index, edge_weight)
            preds = torch.sigmoid(logits) >= 0.5
            correct += (preds == y_batch.bool()).sum().item()
            total += y_batch.size(0)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(y_batch.cpu().numpy())

    train_acc = correct / total
    train_f1 = f1_score(train_labels, train_preds, average='macro')
    print(f'Total training points: {total}')
    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.4f}, Train accuracy: {train_acc:.4f}, Train F1: {train_f1:.4f}")
    if log_wandb:
        wandb.log({"train_loss": avg_loss, "train_accuracy": train_acc, "train_f1": train_f1, "epoch": epoch + 1})
    
    val_correct = 0
    val_total = 0
    val_loss = 0.0
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(loader_val):
            if (i + 1) % 20 == 0:
                print(f"Validation batch {i + 1}/{len(loader_val)}")
            x_batch = x_batch.float().to(device)
            if normalization:
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
    val_f1 = f1_score(val_labels, val_preds, average='macro')
    val_loss /= len(loader_val)
    print(f'Total validation points: {val_total}')
    print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss:.4f}, Valid accuracy: {val_acc:.4f}, Valid F1: {val_f1:.4f}")
    if log_wandb:
        wandb.log({"val_loss": val_loss, "val_accuracy": val_acc, "val_f1": val_f1, "epoch": epoch + 1})
    
    # Save model if best F1 so far
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), ckpt_path)
        print(f"✅ New best model saved with F1: {val_f1:.4f} at epoch {epoch+1}")
    
    if learn_rate_scheduler:
        scheduler.step()
        if log_wandb:
            wandb.log({"learning_rate": optimizer.param_groups[0]['lr'], "epoch": epoch + 1})

# Save the model
if log_wandb:
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, "final_lstm_gtf_model.pth"))
else:
    torch.save(model.state_dict(), os.path.join(script_dir, "final_lstm_gtf_model.pth"))
print("Training complete. Final model saved.")

if log_wandb:
    wandb.finish()
    print("WandB run finished.")
