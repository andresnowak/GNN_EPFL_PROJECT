from torch.utils.data import DataLoader
import torch
from scipy import signal
from seiz_eeg.dataset import EEGDataset
import numpy as np
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '..'))
from models import LSTM_
import pandas as pd
from pathlib import Path
import re
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Run inference on the LSTM model.")
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

lstm_hidden_dim = 128
lstm_num_layers = 3
dropout = 0.2

model = LSTM_(lstm_hidden_dim, lstm_num_layers, dropout).to(device) 
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()
print("Model loaded successfully and set to evaluation mode.")

# Lists to store sample IDs and predictions
all_predictions = []
all_ids = []

print(f"Starting inference on {len(dataset_te)} samples...")

def clean_underscores(s):
    s = re.sub(r'__+', lambda m: '<<UND>>' * (len(m.group()) // 2), s)
    s = re.sub(r'_', '', s)
    s = s.replace('<<UND>>', '_')
    return s

# Disable gradient computation for inference
with torch.no_grad():
    for i, batch in enumerate(loader_te):
        if (i + 1) % 10 == 0:
            print(f"Processing batch {i + 1}/{len(loader_te)}")
        x_batch, x_ids = batch
        x_ids = [clean_underscores(x_id) for x_id in x_ids]
        x_batch = x_batch.float().to(device)
        logits = model(x_batch)
        predictions = (torch.sigmoid(logits) >= 0.5).int().cpu().numpy()
        all_predictions.extend(predictions.flatten().tolist())
        all_ids.extend(list(x_ids))

print("Inference complete. Generating submission file...")
submission_df = pd.DataFrame({"id": all_ids, "label": all_predictions})
submission_df.to_csv(os.path.join(script_dir, "submission.csv"), index=False)
print(f"Submission file saved as {os.path.join(script_dir, 'submission.csv')}")