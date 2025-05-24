from torch.utils.data import DataLoader
import torch
from scipy import signal
from seiz_eeg.dataset import EEGDataset
import numpy as np
from model import LSTM_GCN, LSTM_GAT
import pandas as pd
from pathlib import Path
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT_TEST = Path("/work/cvlab/students/bhagavan/GNN_EPFL_PROJECT/nml_project/data/test")
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
    clips_te,  # Your test clips variable
    signals_root=DATA_ROOT_TEST / "test",  # Update this path if your test signals are stored elsewhere
    signal_transform=fft_filtering,  # You can change or remove the signal_transform as needed
    prefetch=True,  # Set to False if prefetching causes memory issues on your compute environment
    return_id=True,  # Return the id of each sample instead of the label
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

lstm_hidden_dim = 128
lstm_num_layers = 3
gat_hidden = 256
gat_out = 256
num_heads = 4
dropout = 0.2

dir = '/work/cvlab/students/bhagavan/GNN_EPFL_PROJECT/nml_project/wandb/run-20250512_224508-c3obaacy/files'
model = LSTM_GAT(lstm_hidden_dim, lstm_num_layers, gat_hidden, gat_out, num_heads, dropout).to(device) 
model.load_state_dict(torch.load(f"{dir}/best_lstm_gat_model.pth"))  # Load the trained model weights
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

normalization_stats = torch.load(f'{dir}/normalization_stats.pth')
global_min = normalization_stats["global_min"]
global_max = normalization_stats["global_max"]
epsilon = 1e-8

# Disable gradient computation for inference
with torch.no_grad():
    for batch in loader_te:
        # Assume each batch returns a tuple (x_batch, sample_id)
        # If your dataset does not provide IDs, you can generate them based on the batch index.
        x_batch, x_ids = batch
        x_ids = [clean_underscores(x_id) for x_id in x_ids]  # Clean the IDs

        # Move the input data to the device (GPU or CPU)
        x_batch = x_batch.float().to(device)
        x_batch = 2 * (x_batch - global_min) / (global_max - global_min + epsilon) - 1

        # Perform the forward pass to get the model's output logits
        logits = model(x_batch, edge_index, edge_weight)

        # Convert logits to predictions.
        # For binary classification, threshold logits at 0 (adjust this if you use softmax or multi-class).
        predictions = (torch.sigmoid(logits) >= 0.5).int().cpu().numpy()

        # Append predictions and corresponding IDs to the lists
        all_predictions.extend(predictions.flatten().tolist())
        all_ids.extend(list(x_ids))

# Create a DataFrame for Kaggle submission with the required format: "id,label"
submission_df = pd.DataFrame({"id": all_ids, "label": all_predictions})

# Save the DataFrame to a CSV file without an index
submission_df.to_csv(f"{dir}/submission.csv", index=False)
print("Kaggle submission file generated: submission.csv")