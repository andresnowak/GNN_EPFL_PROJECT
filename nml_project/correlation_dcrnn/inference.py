from pathlib import Path
import numpy as np
import pandas as pd
from seiz_eeg.dataset import EEGDataset
import os
import random
import torch
from torch.utils.data import DataLoader
import utils
import argparse
from constants import INCLUDED_CHANNELS
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '..'))
from models import DCRNNModel_classification

########### CONFIG ################
graph_type = 'correlation'
########### CONFIG ################

parser = argparse.ArgumentParser(description="Run inference on the DCRNN model.")
parser.add_argument('--model', type=str, required=True, help='Model path')
args = parser.parse_args()
model_ckpt_path = args.model

def inference_correlation_graph(model, loader_te, device):
    
    # Turn on the model's evaluation mode.
    model.eval()
    
    filter_type = 'dual_random_walk'

    # Lists to store sample IDs and predictions
    all_predictions = []
    all_ids = []

    # Disable gradient computation for inference
    with torch.no_grad():
        for batch in loader_te:
            # Assume each batch returns a tuple (x_batch, sample_id)
            # If your dataset does not provide IDs, you can generate them based on the batch index.
            x_batch, x_ids = batch

            # Compute on the fly Correlation Adjacency matrix and supports (Bidirectional random walk)
            A = utils.get_indiv_graphs(torch.moveaxis(x_batch, 0, 2))
            supports = utils.compute_supports(A, filter_type)
            supports = [support.to(device) for support in supports]

            # Move the input data to the device (GPU or CPU)
            x_batch = x_batch.float().to(device)

            # Perform the forward pass to get the model's output logits
            seq_lengths = torch.ones(x_batch.shape[0], dtype=torch.long).to(device)*354
            logits = model(x_batch, seq_lengths, supports)

            # Convert logits to predictions.
            # 0.5 threshold used for binary classification.
            predictions = (torch.sigmoid(logits) >= 0.5).int().cpu().numpy()

            # Append predictions and corresponding IDs to the lists
            all_predictions.extend(predictions.flatten().tolist())
            all_ids.extend(list(x_ids))

    # Create a DataFrame for Kaggle submission with the required format: "id,label"
    submission_df = pd.DataFrame({"id": all_ids, "label": all_predictions})

    # Save the DataFrame to a CSV file without an index
    submission_df.to_csv(os.path.join(script_dir, "submission.csv"), index=False)
    print(f"Submission file saved as {os.path.join(script_dir, 'submission.csv')}")

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

DATA_ROOT_TEST = Path(os.path.join(script_dir, "..", "data", "test"))

# Load the test split.
clips_te = pd.read_parquet(DATA_ROOT_TEST / "segments.parquet")

dataset_te = EEGDataset(
    clips_te,  # Your test clips variable
    signals_root=DATA_ROOT_TEST,  # Update this path if your test signals are stored elsewhere
    signal_transform=utils.fft_filtering,  # Frequency domain transformation
    prefetch=True,  # Set to False if prefetching causes memory issues on your compute environment
    return_id=True,  # Return the id of each sample instead of the label
)    

# Initialize the data loaders
batch_size = 128
loader_te = DataLoader(dataset_te, batch_size=batch_size, shuffle=False)

# Best Configuration of Correlation Graph defined below.
# Initialize the model parameters.
num_nodes = 19 # Number of electrodes.
rnn_units = 64 # The hidden variable dimension of gru.
num_rnn_layers = 2 # Number of gru layers.
input_dim = 1 # Each electrode has one dimension.
num_classes = 1 # Seizure or not.
max_diffusion_step = 2 # How deep the diffusion walk should be.
dcgru_activation = 'tanh'
dropout = 0 # Amount of dropout.
lr = 1e-3 # Learning rate.


# Correlation Graphs are directed, hence we chose 'bidirectional random walk' for diffusion steps.
filter_type = 'dual_random_walk'

model = DCRNNModel_classification(
    input_dim=input_dim,
    num_nodes=num_nodes,
    num_classes=num_classes,
    num_rnn_layers=num_rnn_layers,
    rnn_units=rnn_units,
    max_diffusion_step=max_diffusion_step,
    dcgru_activation=dcgru_activation,
    filter_type=filter_type,
    dropout=dropout,
    device=device
).to(device)
print('Number of trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

# Load the best model from model_ckpt_path
model.load_state_dict(torch.load(model_ckpt_path, map_location=device))  # Load the trained model weights
model.to(device)

# Generate the model's predictions on test set.
inference_correlation_graph(model, loader_te, device)
