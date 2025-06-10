# GNN_EPFL_PROJECT

This is the code for the course project in EE-452 (Network Machine Learning).

## Installation instructions

First create an environment with python 3.11.12

```bash
bash install.sh
```

## Structure of the repo

Each model has a separate subdirectory which has a train and an inference script. Model definitions are in `models.py`

This repo contains these models:

*   LSTM_GAT
*   LSTM_GCN
*   LSTM_GTF
*   LSTM
*   GAT
*   CNN_GAT
*   STGCN avg pooling
*   STGCN attn pooling
*   Resnet lstm
*   DTGCN
*   TGCN

## Inference

To run inference, first download the trained model weights from the provided Google Drive links (you will provide these links separately).

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
