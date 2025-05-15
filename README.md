# GNN_EPFL_PROJECT

### Train and Validation Set
segments_train.parquet: Contains 88 Unique Patients and 19.37% Samples are Seizures.

segments_val.parquet: Contains 9 Unique Patients and 19.39% Samples are Seizures.

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
