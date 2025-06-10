from torch.utils.data import DataLoader
import torch
from scipy import signal
from seiz_eeg.dataset import EEGDataset
import numpy as np
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '..'))
from models import S4_
import pandas as pd
from pathlib import Path
import re
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Run inference on the S4 model.")
parser.add_argument('--model', type=str, required=True, help='Model path')
args = parser.parse_args()
model_path = args.model

DATA_ROOT_TEST = Path(os.path.join(script_dir, "..", "data", "test"))
clips_te = pd.read_parquet(DATA_ROOT_TEST / "segments.parquet")

bp_filter = signal.butter(4, (0.5, 30), btype="bandpass", output="sos", fs=250)

def time_filtering(x: np.ndarray) -> np.ndarray:
    """Filter signal in the time domain"""
    return signal.sosfiltfilt(bp_filter, x, axis=0).copy()

def fft_filtering(x: np.ndarray) -> np.ndarray:
    """Compute FFT and only keep frequencies between 0.5 and 30 Hz"""
    x = np.abs(np.fft.fft(x, axis=0))
    x = np.log(np.where(x > 1e-8, x, 1e-8))
    win_len = x.shape[0]
    return x[int(0.5 * win_len // 250) : 30 * win_len // 250]

# Create test dataset
dataset_te = EEGDataset(
    clips_te,
    signals_root=DATA_ROOT_TEST,
    signal_transform=fft_filtering,
    prefetch=True,
    return_id=True
)

loader_te = DataLoader(dataset_te, batch_size=64, shuffle=False)

s4_hidden_dim = 256
s4_dropout = 0.1

model = S4_(s4_hidden_dim, s4_dropout).to(device)
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

all_predictions = []
all_ids = []

def clean_underscores(s):
    s = re.sub(r'__+', lambda m: '<<UND>>' * (len(m.group()) // 2), s)
    s = re.sub(r'_', '', s)
    s = s.replace('<<UND>>', '_')
    return s

with torch.no_grad():
    for x_batch, x_ids in loader_te:
        x_batch = x_batch.float().to(device)
        x_ids = [clean_underscores(x_id) for x_id in x_ids]
        logits = model(x_batch)
        predictions = (torch.sigmoid(logits) >= 0.5).int().cpu().numpy()
        all_predictions.extend(predictions.flatten().tolist())
        all_ids.extend(list(x_ids))

submission_df = pd.DataFrame({"id": all_ids, "label": all_predictions})
submission_df.to_csv(os.path.join(script_dir, "submission.csv"), index=False)

print(f"Submission file saved as {os.path.join(script_dir, 'submission.csv')}")
