from torch.utils.data import DataLoader
import torch
from scipy import signal
from seiz_eeg.dataset import EEGDataset
import numpy as np
import pandas as pd
from pathlib import Path
import re
from model import GCN_LSTM_Model
from utils import get_adjacency_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT_TEST = Path("/content/drive/MyDrive/EPFL_NetworkML/epfl-network-machine-learning-2025/")
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
print('Started Loading Data')
dataset_te = EEGDataset(
    clips_te,  # Your test clips variable
    signals_root=DATA_ROOT_TEST / "test",  # Update this path if your test signals are stored elsewhere
    signal_transform=fft_filtering,  # You can change or remove the signal_transform as needed
    prefetch=False,  # Set to False if prefetching causes memory issues on your compute environment
    return_id=True,  # Return the id of each sample instead of the label
)
print('Ended Loading Data')

# Create DataLoader for the test dataset
loader_te = DataLoader(dataset_te, batch_size=128, shuffle=False)

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
dist_df = pd.read_csv(DATA_ROOT_TEST /'distances_3d.csv')
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

model.load_state_dict(torch.load("/content/wandb/run-20250523_193840-xjhqicnd/files/best_gnn_lstm_model.pth", map_location=device))  # Load the trained model weights
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

        actual_x_ids = x_ids
        x_ids = [clean_underscores(x_id) for x_id in x_ids]  # Clean the IDs

        # Move the input data to the device (GPU or CPU)
        x_batch = x_batch.float().unsqueeze(-1).to(device)

        # Perform the forward pass to get the model's output logits
        logits = model(x_batch)

        # Convert logits to predictions.
        # For binary classification, threshold logits at 0 (adjust this if you use softmax or multi-class).
        predictions = (torch.sigmoid(logits) >= 0.5).int().cpu().numpy()

        # Append predictions and corresponding IDs to the lists
        all_predictions.extend(predictions.flatten().tolist())
        all_ids.extend(list(actual_x_ids))

# Create a DataFrame for Kaggle submission with the required format: "id,label"
submission_df = pd.DataFrame({"id": all_ids, "label": all_predictions})

# Save the DataFrame to a CSV file without an index
submission_df.to_csv("submission_Sixteenth_HypTuning.csv", index=False)
print("Kaggle submission file generated: submission.csv")