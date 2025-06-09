The requirements.txt file lists all the Python libraries that the code depends on. On terminal, run the following to install them:

```
pip install -r requirements.txt
```

## GCN(1 Layer) + LSTM:


For Training: on terminal, run the following:
```
python3 train.py --data_path <data_path> --batch_size 128 --wandblog 0 --graph_type 'gcnlstm' --num_epochs 1 --best_ckpt_path <best_ckpt_path>
```
where `<best_ckpt_path>` is the directory where you want to save your best model checkpoint, and `<data_path>` is the directory where the eeg test and train data is present.

For Inference: on terminal, run the following:
```
python3 inference.py --data_path <data_path> --batch_size 128 --graph_type 'gcnlstm' --model_ckpt_path <model_ckpt_path>
```
where `<model_ckpt_path>` is the directory from where you want to load your pretrained model, and `<data_path>` is the directory where the eeg test and train data is present.

---

## GCN(2 Layers) + LSTM:

For Training: on terminal, run the following:
```
python3 train.py --data_path <data_path> --batch_size 128 --wandblog 0 --graph_type 'gcn2lstm' --num_epochs 1 --best_ckpt_path <best_ckpt_path>
```
where `<best_ckpt_path>` is the directory where you want to save your best model checkpoint, and `<data_path>` is the directory where the eeg test and train data is present.

For Inference: on terminal, run the following:
```
python3 inference.py --data_path <data_path> --batch_size 128 --graph_type 'gcn2lstm' --model_ckpt_path <model_ckpt_path>
```
where `<model_ckpt_path>` is the directory from where you want to load your pretrained model, and `<data_path>` is the directory where the eeg test and train data is present.

---

## GAT(1 Layer) + LSTM:

For Training: on terminal, run the following:
```
python3 train.py --data_path <data_path> --batch_size 128 --wandblog 0 --graph_type 'gatlstm' --num_epochs 1 --best_ckpt_path <best_ckpt_path>
```
where `<best_ckpt_path>` is the directory where you want to save your best model checkpoint, and `<data_path>` is the directory where the eeg test and train data is present.

For Inference: on terminal, run the following:
```
python3 inference.py --data_path <data_path> --batch_size 128 --graph_type 'gatlstm' --model_ckpt_path <model_ckpt_path>
```
where `<model_ckpt_path>` is the directory from where you want to load your pretrained model, and `<data_path>` is the directory where the eeg test and train data is present.

---

## GAT(2 Layers) + LSTM:

For Training: on terminal, run the following:
```
python3 train.py --data_path <data_path> --batch_size 128 --wandblog 0 --graph_type 'gat2lstm' --num_epochs 1 --best_ckpt_path <best_ckpt_path>
```
where `<best_ckpt_path>` is the directory where you want to save your best model checkpoint, and `<data_path>` is the directory where the eeg test and train data is present.

For Inference: on terminal, run the following:
```
python3 inference.py --data_path <data_path> --batch_size 128 --graph_type 'gat2lstm' --model_ckpt_path <model_ckpt_path>
```
where `<model_ckpt_path>` is the directory from where you want to load your pretrained model, and `<data_path>` is the directory where the eeg test and train data is present.

---

## Hyperparameter Tuning Results 

| Configuration  | Max Macro-F1 on Val Set|
| ------------- | ------------- |
| GCN+LSTM, lr=2e-4, lstm=(64,3), gcn=(16,1), dropout=0.2   | 0.6741 |
| GCN+LSTM, lr=2e-4, lstm=(128,3), gcn=(16,1), dropout=0  | 0.7356 | 
| GCN+LSTM, lr=2e-4, lstm=(64,3), gcn=(16,1), dropout=0  | 0.7069 |
| GCN+LSTM, lr=1e-4, lstm=(64,3), gcn=(16,1), dropout=0 | 0.6500 |
| GCN+LSTM, lr=2e-4, lstm=(64,3), gcn=(8,1), dropout=0  | 0.7168 |
| GCN+LSTM, lr=1e-4, lstm=(64,3), gcn=(8,1), dropout=0  | 0.6514 |
| GCN2+LSTM, lr=2e-4, lstm=(64,3), gcn=(16,1), dropout=0  | 0.6712 |
| GCN2+LSTM, lr=1e-4, lstm=(64,3), gcn=(16,1), dropout=0.2  | 0.6700 |
| GCN2+LSTM, lr=1e-4, lstm=(64,3), gcn=(8,1), dropout=0  | 0.5427 |
| GCN2+LSTM, lr=2e-4, lstm=(64,3), gcn=(8,1), dropout=0  | 0.6211 |
| GCN2+LSTM, lr=2e-4, lstm=(64,3), gcn=(16,1), dropout=0, norm_layers = False  | 0.6633 |
| GAT+LSTM, lr=1e-4, lstm=(64,3), gat=(8,1), dropout=0  | 0.6430 |
| GAT+LSTM, lr=1e-4, lstm=(64,3), gat=(16,1), dropout=0  | 0.6556 | 
| GAT+LSTM, lr=2e-4, lstm=(64,3), gat=(16,1), dropout=0  | 0.6215 | 
| GAT+LSTM, lr=2e-4, lstm=(64,3), gat=(16,2), dropout=0  | 0.7017 |
| GAT+LSTM, lr=2e-4, lstm=(64,3), gat=(16,3), dropout=0  | 0.7205 |
| GAT2+LSTM, lr=2e-4, lstm=(64,3), gat=(16,2), dropout=0  | 0.6109 |
| GAT2+LSTM, lr=2e-4, lstm=(128,3), gat=(16,2), dropout=0  | 0.6206 |
| GAT2+LSTM, lr=2e-4, lstm=(128,3), gat=(16,2), dropout=0.2  | 0.6264 |
| GAT2+LSTM, lr=2e-4, lstm=(128,3), gat=(8,2), dropout=0.2  | 0.6299 |

