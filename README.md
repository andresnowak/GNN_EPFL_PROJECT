# GNN_EPFL_PROJECT

This is the code for the course project in EE-452 (Network Machine Learning).

## Data
First make sure you have the data inside nml_project folder

data 
├── train 
│   ├── signals/
│   ├── segments.parquet
│-- test
     ├── signals/
     ├── segments.parquet


## Installation instructions

First create a conda environment with python 3.11.12.

```bash
bash install.sh
```

## Structure of the repo

Each model has a separate subdirectory which has a train and an inference script. Model definitions are in `models.py`

This repo contains these models:

*   LSTM + GAT
*   LSTM + GCN
*   LSTM + GraphTransformer
*   LSTM
*   GAT
*   CNN + GAT
*   STGCN Avg Pooling
*   STGCN Attention Pooling
*   Resnet LSTM
*   DTGCN
*   TGCN
*   S4 
*   S4 + GCN
*   S4 + GAT
*   DCRNN (Distance DCRNN and Correlation DCRNN)
*   GNN + LSTM (GCN + LSTM, GAT + LSTM)

## Inference

To run inference, first download the trained model weights from the provided Google Drive links given below.

Then, navigate to the `nml_project` directory and execute the corresponding `inference.py` script, providing the path to the downloaded model weights using the `--model` argument.

Example:

```bash
cd GNN_EPFL_PROJECT/nml_project
python <model_name>/inference.py --model /path/to/your/downloaded_model.pth
```

Please replace `<model_name>` with the actual model directory (e.g., `lstm_gat`, `lstm`) and `/path/to/your/downloaded_model.pth` with the actual path to your downloaded model weights.

## Train

To train, update the batch script `run.batch` to point the correct environment and which script to run and do `sbatch run.batch`. To run it locally,

```bash
cd GNN_EPFL_PROJECT/nml_project
python <model_name>/train.py
```

Please replace `<model_name>` with the actual model directory (e.g., `lstm_gat`, `lstm`).

Note: Some training scripts are configured to log metrics to Weights & Biases (wandb). If `log_wandb` is set to `True` in the `train.py` script, ensure you are logged into your wandb account before running the training:

```bash
wandb login
```

## Download Links

For LSTM, GAT, LSTM + GCN, LSTM + GAT, LSTM + GraphTransformer, CNN + GAT - https://drive.google.com/drive/folders/1qUHm-iL-1SH-mw-PkawRmklAVrRn1pH9?usp=sharing

For S4, S4 + GCN, S4 + GAT - https://drive.google.com/drive/folders/10tZ6OtLPOFxnhR_BfsiJ_PhQkXdrSoT0?usp=sharing

For STGCN and Resnet LSTM - https://drive.google.com/drive/u/0/folders/15rc7KJ-a0dDGIbac6jVJmRPPm03uIrrH

For DCRNN (Distance DCRNN and Correlation DCRNN), GNN_LSTM (GCN + LSTM and GAT + LSTM) - https://drive.google.com/drive/folders/1-Pe-KoBPkV4kKo4ti0O4lKSI2tYhxs97?usp=sharing
