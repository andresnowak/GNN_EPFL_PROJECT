from torch.utils.data import DataLoader
import torch
from scipy import signal
from seiz_eeg.dataset import EEGDataset
import numpy as np
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '..'))
from models import LSTM_GraphTransformer
import pandas as pd
from pathlib import Path
import re
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Run inference on the LSTM-GTF model.")
parser.add_argument('--model', type=str, required=True, help='Model path')
args = parser.parse_args()
model_path = args.model

DATA_ROOT_TEST = Path(os.path.join(script_dir, "..", "data", "test"))
clips_te = pd.read_parquet(DATA_ROOT_TEST / "test/segments.parquet")

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

# Create test dataset
dataset_te = EEGDataset(
    clips_te,
    signals_root=DATA_ROOT_TEST / "test", 
    signal_transform=fft_filtering,
    prefetch=True,
    return_id=True,
)

# Create DataLoader for the test dataset
loader_te = DataLoader(dataset_te, batch_size=64, shuffle=False)

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

lstm_hidden_dim = 128
lstm_num_layers = 3
gtf_hidden = 272
gtf_out = 272
num_heads = 4
dropout = 0.2
pos_enc_dim = 16

model = LSTM_GraphTransformer(lstm_hidden_dim, lstm_num_layers, gtf_hidden, gtf_out, num_heads, dropout, pos_enc_dim).to(device)
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()
print("Model loaded successfully and set to evaluation mode.")

all_predictions = []
all_ids = []

print(f"Starting inference on {len(dataset_te)} samples...")

def clean_underscores(s):
    s = re.sub(r'__+', lambda m: '<<UND>>' * (len(m.group()) // 2), s)
    s = re.sub(r'_', '', s)
    s = s.replace('<<UND>>', '_')
    return s

with torch.no_grad():
    for i, batch in enumerate(loader_te):
        if (i + 1) % 10 == 0:
            print(f"Processing batch {i + 1}/{len(loader_te)}")
        x_batch, x_ids = batch
        x_ids = [clean_underscores(x_id) for x_id in x_ids]
        x_batch = x_batch.float().to(device)
        logits = model(x_batch, edge_index, edge_weight)
        predictions = (torch.sigmoid(logits) >= 0.5).int().cpu().numpy()
        all_predictions.extend(predictions.flatten().tolist())
        all_ids.extend(list(x_ids))

print("Inference complete. Generating submission file...")
submission_df = pd.DataFrame({"id": all_ids, "label": all_predictions})
submission_df.to_csv(os.path.join(script_dir, "submission.csv"), index=False)
print(f"Submission file saved as {os.path.join(script_dir, 'submission.csv')}")
