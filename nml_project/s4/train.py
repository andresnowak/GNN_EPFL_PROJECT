from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal
from seiz_eeg.dataset import EEGDataset
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import OneCycleLR
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '..'))
from models import S4_

log_wandb = True

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
    return signal.sosfiltfilt(bp_filter, x, axis=0).copy()

def fft_filtering(x: np.ndarray) -> np.ndarray:
    x = np.abs(np.fft.fft(x, axis=0))
    x = np.log(np.where(x > 1e-8, x, 1e-8))
    win_len = x.shape[0]
    return x[int(0.5 * win_len // 250) : 30 * win_len // 250]

# Create training and validation datasets
dataset_tr = EEGDataset(
    clips_tr,
    signals_root=DATA_ROOT/"data"/"train"/"train",
    signal_transform=fft_filtering,
    prefetch=True,
)
dataset_val = EEGDataset(
    clips_va,
    signals_root=DATA_ROOT/"data"/"train"/"train",
    signal_transform=fft_filtering,
    prefetch=True,
)

batch_size = 64
loader_tr = DataLoader(dataset_tr, batch_size=batch_size, shuffle=True)
loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

epochs = 300
lr = 1e-3
weight_decay = 1e-5
s4_hidden_dim = 256
s4_dropout = 0.1

if log_wandb:
    wandb.init(project="eeg-s4", config={
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "model": "S4",
        "s4_hidden_dim": s4_hidden_dim,
        "s4_dropout": s4_dropout,
    })

model = S4_(s4_hidden_dim, s4_dropout).to(device)
print("Total params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

label_counts = clips_tr["label"].value_counts()
neg, pos = label_counts[0], label_counts[1]
pos_weight = torch.tensor([neg / pos], device=device, dtype=torch.float32)
print(f"Using pos_weight={pos_weight.item():.4f} for class imbalance correction.")
if log_wandb:
    wandb.config.update({"pos_weight": pos_weight.item()})

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

scheduler = OneCycleLR(
    optimizer,
    max_lr=lr,
    total_steps=epochs * len(loader_tr),
    pct_start=0.1,
    anneal_strategy="cos"
)

# Training loop
best_f1 = 0.0
if log_wandb:
    ckpt_path = os.path.join(wandb.run.dir, "best_s4_model.pth")
else:
    ckpt_path = os.path.join(script_dir, "best_s4_model.pth")
global_step = 0

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for x_batch, y_batch in loader_tr:
        # x_batch: [B, T, N], y_batch: [B]
        x_batch = x_batch.float().to(device)
        y_batch = y_batch.float().unsqueeze(1).to(device)

        logits = model(x_batch)
        loss = criterion(logits, y_batch)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += (p.grad.data.norm(2).item() ** 2)
        total_norm = total_norm ** 0.5
        if log_wandb:
            wandb.log({"grad_norm": total_norm, "global_step": global_step})
        global_step += 1

        optimizer.step()
        scheduler.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(loader_tr)

    model.eval()
    correct = 0
    total = 0
    train_preds = []
    train_labels = []
    with torch.no_grad():
        for x_batch, y_batch in loader_tr:
            x_batch = x_batch.float().to(device)
            y_batch = y_batch.float().unsqueeze(1).to(device)
            logits = model(x_batch)
            preds = torch.sigmoid(logits) >= 0.5
            correct += (preds == y_batch.bool()).sum().item()
            total += y_batch.size(0)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(y_batch.cpu().numpy())

    train_acc = correct / total
    train_f1 = f1_score(train_labels, train_preds, zero_division=0, average="macro")
    print(f"[Epoch {epoch+1}/{epochs}] Train Loss: {avg_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
    if log_wandb:
        wandb.log({
            "train_loss": avg_loss,
            "train_accuracy": train_acc,
            "train_f1": train_f1,
            "epoch": epoch + 1
        })

    # Validation
    val_correct = 0
    val_total = 0
    val_loss = 0.0
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for x_batch, y_batch in loader_val:
            x_batch = x_batch.float().to(device)
            y_batch = y_batch.float().unsqueeze(1).to(device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            val_loss += loss.item()
            preds = torch.sigmoid(logits) >= 0.5
            val_correct += (preds == y_batch.bool()).sum().item()
            val_total += y_batch.size(0)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(y_batch.cpu().numpy())

    val_acc = val_correct / val_total
    val_f1 = f1_score(val_labels, val_preds, zero_division=0, average="macro")
    val_loss /= len(loader_val)
    print(f"        Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
    if log_wandb:
        wandb.log({
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_f1": val_f1,
            "epoch": epoch + 1
        })

    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), ckpt_path)
        print(f"✅ New best S4‐only model saved with F1: {val_f1:.4f} at epoch {epoch+1}")

# Save the model
if log_wandb:
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, "final_s4_model.pth"))
else:
    torch.save(model.state_dict(), os.path.join(script_dir, "final_s4_model.pth"))

if log_wandb:
    wandb.finish()