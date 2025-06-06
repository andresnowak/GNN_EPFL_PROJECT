| Configuration  | Max F1 on Val Set | Wandb Config Name |
| ------------- | ------------- | ------------- | 
| GCN+LSTM, lr=2e-4, lstm=(64,3), gcn=(16,1), dropout=0.2   | 0.6741 |gcn-lstm: breezy-fire-43  | 
| GCN+LSTM, lr=2e-4, lstm=(128,3), gcn=(16,1), dropout=0  | 0.7356 |gcn-lstm: efficient-cherry-42 | 
| GCN+LSTM, lr=2e-4, lstm=(64,3), gcn=(16,1), dropout=0  | 0.7069 |gcn-lstm: hopeful-pond-41 | 
| GCN+LSTM, lr=1e-4, lstm=(64,3), gcn=(16,1), dropout=0 | 0.6500 |gcn-lstm: leafy-sky-40  | 
| GCN+LSTM, lr=2e-4, lstm=(64,3), gcn=(8,1), dropout=0  | 0.7168 |gcn-lstm: cool-eon-39 | 
| GCN+LSTM, lr=1e-4, lstm=(64,3), gcn=(8,1), dropout=0  | 0.6514 |gcn-lstm: hearty-salad-38 | 
| GAT+LSTM, lr=1e-4, lstm=(64,3), gat=(8,1), dropout=0  | 0.6430 |gat-lstm: gentle-gorge-6 | 
| GAT+LSTM, lr=1e-4, lstm=(64,3), gat=(16,1), dropout=0  | 0.6556 |gat-lstm: bumbling-vortex-7 | 
| GAT+LSTM, lr=2e-4, lstm=(64,3), gat=(16,1), dropout=0  | 0.6215 |gat-lstm: whole-snowball-8 | 
| GAT+LSTM, lr=2e-4, lstm=(64,3), gat=(16,2), dropout=0  | 0.7017 |gat-lstm: happy-moon-9 | 
| GAT+LSTM, lr=2e-4, lstm=(64,3), gat=(16,3), dropout=0  | 0.7205 |gat-lstm: deep-violet-10 | 

# GNN_EPFL_PROJECT

### Train and Validation Set
segments_train.parquet: Contains 88 Unique Patients, 19.37% Samples are Seizures, Total% Samples are 9.12%

segments_val.parquet: Contains 9 Unique Patients, 19.39% Samples are Seizures, Total% of Samples are 90.87%

You can use it directly as:

```
clips_tr = pd.read_parquet("/content/segments_train.parquet")
clips_va = pd.read_parquet("/content/segments_val.parquet")

dataset_tr = EEGDataset(
    clips_tr,
    signals_root=DATA_ROOT / "train", # signals_root directory remains same for val and tr set.
    signal_transform=fft_filtering,
    prefetch=True,  # If your compute does not allow it, you can use `prefetch=False`
)

dataset_va = EEGDataset(
    clips_va,
    signals_root=DATA_ROOT / "train", # signals_root directory remains same for val and tr set.
    signal_transform=fft_filtering,
    prefetch=True,  # If your compute does not allow it, you can use `prefetch=False`
)
```
