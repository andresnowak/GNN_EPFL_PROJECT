import numpy as np
import argparse
from pathlib import Path
import pandas as pd
from seiz_eeg.dataset import EEGDataset
import os
import random
import torch
from torch.utils.data import DataLoader

from model.model import GCN_LSTM_Model, GCN2_LSTM_Model, GAT_LSTM_Model, GAT2_LSTM_Model
from constants import INCLUDED_CHANNELS
import utils

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

def main(args):

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

    DATA_ROOT = Path(args.data_path)

    # Load test split.
    clips_te = pd.read_parquet(DATA_ROOT / "test/segments.parquet")

    dataset_te = EEGDataset(
        clips_te,
        signals_root=DATA_ROOT / "test",  # Update this path if your test signals are stored elsewhere
        signal_transform=utils.fft_filtering,  # Frequency domain transformation
        prefetch=True,  # Set to False if prefetching causes memory issues on your compute environment
        return_id=True,  # Return the id of each sample instead of the label
    )    

    # Initialize the data loaders
    batch_size = args.batch_size
    loader_te = DataLoader(dataset_te, batch_size=batch_size, shuffle=False)

    # Compute the Distance Adjacency Matrix and make it sparse using threshold 0.9
    thresh = 0.9
    dist_df = pd.read_csv('distances_3d.csv')
    A = utils.get_adjacency_matrix(dist_df, INCLUDED_CHANNELS, dist_k=thresh)

    # Initialize the model parameters.
    num_nodes = 19 # Number of electrodes.
    in_channels = 1 # Each electrode has one dimension.
    output_dim = 1 # Seizure or not.
    seq_len = 354 # Length of the frequency series.

    # For initializing the model with GCN_LSTM.
    # gcnlstm = one layer of gcn is used per time step.
    # gcn2lstm = two layers of gcn are used per time step.
    if args.graph_type == 'gcnlstm' or args.graph_type == 'gcn2lstm':

        # Best Configuration of gcnlstm
        gcn_hidden = 16 # The output dimension of the gcn.
        lstm_hidden = 128 # The hidden variable dimension of lstm.
        lstm_layers = 3 # Number of lstm layers.
        dropout = 0 # Amount of dropout.

        if args.graph_type == 'gcnlstm':
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
            
        elif args.graph_type == 'gcn2lstm':
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

    # For initializing the model with GAT_LSTM.
    # gatlstm = one layer of gat is used per time step.
    # gat2lstm = two layers of gat are used per time step.
    elif args.graph_type == 'gatlstm' or args.graph_type == 'gat2lstm':

        # Best Configuration of gatlstm
        gat_hidden = 16 # The output dimension of one head of gat.
        lstm_hidden = 64 # The hidden variable dimension of lstm.
        lstm_layers = 3 # Number of lstm layers.
        dropout = 0 # Amount of dropout.
        num_heads = 3 # Number of heads in gat.

        if args.graph_type == 'gatlstm':
            model = GAT_LSTM_Model(
                    num_nodes,
                    in_channels,
                    gat_hidden,
                    lstm_hidden,
                    output_dim,
                    lstm_layers,
                    num_heads,
                    dropout,
                    seq_len,
                    torch.from_numpy(A)
                    ).to(device)
            
        elif args.graph_type == 'gat2lstm':
            # Best Configuration of gat2lstm
            gat_hidden = 8 # The output dimension of one head of gat.
            lstm_hidden = 128 # The hidden variable dimension of lstm.
            dropout = 0.2 # Amount of dropout.
            num_heads = 2 # Number of heads in gat to make it divisible by gat_hidden
            model = GAT2_LSTM_Model(
                    num_nodes,
                    in_channels,
                    gat_hidden,
                    lstm_hidden,
                    output_dim,
                    lstm_layers,
                    num_heads,
                    dropout,
                    seq_len,
                    torch.from_numpy(A)
                    ).to(device)
    
    print('Number of trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Load the pretrained model from args.model_ckpt_path
    model.load_state_dict(torch.load(args.model_ckpt_path, map_location=device))  # Load the trained model weights
    model.to(device)

    # Generate the model's predictions on test set.
    inference_model(model, loader_te, device)

def inference_model(model, loader_te, device):

    # Turn on the model's evaluation mode.
    model.eval()

    # Lists to store sample IDs and predictions
    all_predictions = []
    all_ids = []

    # Disable gradient computation for inference
    with torch.no_grad():
        for batch in loader_te:
            # Assume each batch returns a tuple (x_batch, sample_id)
            # If your dataset does not provide IDs, you can generate them based on the batch index.
            x_batch, x_ids = batch

            # Move the input data to the device (GPU or CPU)
            x_batch = x_batch.float().unsqueeze(-1).to(device)

            # Perform the forward pass to get the model's output logits
            logits = model(x_batch)

            # Convert logits to predictions.
            # 0.5 threshold used for binary classification.
            predictions = (torch.sigmoid(logits) >= 0.5).int().cpu().numpy()

            # Append predictions and corresponding IDs to the lists
            all_predictions.extend(predictions.flatten().tolist())
            all_ids.extend(list(x_ids))

    # Create a DataFrame for Kaggle submission with the required format: "id,label"
    submission_df = pd.DataFrame({"id": all_ids, "label": all_predictions})

    # Save the DataFrame to a CSV file without an index
    submission_df.to_csv("submission.csv", index=False)
    print("Kaggle submission file generated")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='') # directory where the eeg test and train data is present.
    parser.add_argument('--batch_size', type=int, default=128) # Batch size to be used.
    parser.add_argument('--graph_type', type=str, default='gcnlstm') # What type of model you are using.
    parser.add_argument('--model_ckpt_path', type=str, default='') # Directory where the pretrained model will be loaded.

    args = parser.parse_args()
    main(args)