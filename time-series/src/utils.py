from pathlib import Path
import yaml
from typing import Dict, Any
import torch
from seiz_eeg.dataset import EEGDataset
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from time import time
from scipy import signal
from imblearn.over_sampling import SMOTE
from collections import Counter
import os
import random


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML config with validation"""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file {path} not found")
    
    with open(config_path, 'r') as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML: {e}")
        
def save_config(config, path: str, name: str):
    # Define config file path
    config_path = os.path.join(path, name)

    # Save config as YAML
    with open(config_path, "w") as f:
        yaml.dump(
            config, f, default_flow_style=False, sort_keys=False
        )  # Preserve structure


def load_eeg_data(
    data_path: str, train_path: str, val_path: str, filtering_type: str, robust: bool = False
) -> tuple[EEGDataset, EEGDataset, pd.DataFrame]:
    """## Compatibility with PyTorch

    The `EEGDataset` class is compatible with [pytorch datasets and dataloaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html), which allow you to load batched data.
    """

    """# The data

    We model *segments* of brain activity, which correspond to windows of a longer *session* of EEG recording.

    These segments, and their labels, are described in the `segments.parquet` files, which can be directly loaded with `pandas`.
    """

    # You might need to change this according to where you store the data folder
    # Inside your data folder, you should have the following structure:
    # data
    # ├── train
    # │   ├── signals/
    # │   ├── segments.parquet
    # │-- test
    #     ├── signals/
    #     ├── segments.parquet

    print("Loading Data")

    data_path = "../data"  # we use relative path
    DATA_ROOT = Path(data_path)
    clips_tr = pd.read_parquet(
        DATA_ROOT / train_path, engine="pyarrow", use_threads=True, memory_map=True
    )
    clips_val = pd.read_parquet(
        DATA_ROOT / val_path,
        engine="pyarrow",
        use_threads=True,
        memory_map=True,
    )

    # Stratified split based on labels
    if robust:
        train_df = clips_tr
        val_df = clips_val
    else:
        train_df, val_df = train_test_split(
        clips_tr, test_size=0.1, random_state=1, stratify=clips_tr["label"]
    )

    """## Loading the signals

    For convenience, the `EEGDataset class` provides functionality for loading each segment and its label as `numpy` arrays.

    You can provide an optional `signal_transform` function to preprocess the signals. In the example below, we have two bandpass filtering functions, which extract frequencies between 0.5Hz and 30Hz which are used in seizure analysis literature:

    The `EEGDataset` class also allows to load all data in memory, instead of reading it from disk at every iteration. If your compute allows it, you can use `prefetch=True`.
    """
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

    # Create training and validation datasets
    start_temp = time()
    dataset_tr = EEGDataset(
        train_df,
        signals_root=DATA_ROOT / "train",  # type: ignore
        signal_transform=time_filtering
        if filtering_type == "time_domain"
        else fft_filtering,
        prefetch=True,
    )

    dataset_val = EEGDataset(
        val_df,
        signals_root=DATA_ROOT / "train",
        signal_transform=time_filtering
        if filtering_type == "time_domain"
        else fft_filtering,
        prefetch=True,
    )

    print(f"Time taken for filtering: {time() - start_temp}")

    return dataset_tr, dataset_val, train_df


def load_eeg_val_data(
    data_path: str, val_path: str, filtering_type: str
) -> EEGDataset:
    """## Compatibility with PyTorch

    The `EEGDataset` class is compatible with [pytorch datasets and dataloaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html), which allow you to load batched data.
    """

    """# The data

    We model *segments* of brain activity, which correspond to windows of a longer *session* of EEG recording.

    These segments, and their labels, are described in the `segments.parquet` files, which can be directly loaded with `pandas`.
    """

    # You might need to change this according to where you store the data folder
    # Inside your data folder, you should have the following structure:
    # data
    # ├── train
    # │   ├── signals/
    # │   ├── segments.parquet
    # │-- test
    #     ├── signals/
    #     ├── segments.parquet

    print("Loading Data")

    data_path = "../data"  # we use relative path
    DATA_ROOT = Path(data_path)

    clips_val = pd.read_parquet(
        DATA_ROOT / val_path,
        engine="pyarrow",
        use_threads=True,
        memory_map=True,
    )

    """## Loading the signals

    For convenience, the `EEGDataset class` provides functionality for loading each segment and its label as `numpy` arrays.

    You can provide an optional `signal_transform` function to preprocess the signals. In the example below, we have two bandpass filtering functions, which extract frequencies between 0.5Hz and 30Hz which are used in seizure analysis literature:

    The `EEGDataset` class also allows to load all data in memory, instead of reading it from disk at every iteration. If your compute allows it, you can use `prefetch=True`.
    """
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

    # Create training and validation datasets
    start_temp = time()

    dataset_val = EEGDataset(
        clips_val,
        signals_root=DATA_ROOT / "test",
        signal_transform=time_filtering
        if filtering_type == "time_domain"
        else fft_filtering,
        prefetch=True,
        return_id=True,
    )

    print(f"Time taken for filtering: {time() - start_temp}")

    return dataset_val


def load_graph(data_path: str, graph_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    data_path = "../data"  # we use relative path
    DATA_ROOT = Path(data_path)

    # Load distances of nodes
    distances_df = pd.read_csv(DATA_ROOT / graph_path)
    distances_df.head()

    src, dst, weights = [], [], []

    nodes_names = distances_df["from"].unique()
    N = len(distances_df["from"].unique())

    epsilon = 1e-6
    constant = 4 # to make the weights more pronounced

    for i in range(N):
        for j in range(N):
            src.append(i)
            dst.append(j)

            if i != j:
                nodes_distance = distances_df[
                    (distances_df["from"] == nodes_names[i])
                    & (distances_df["to"] == nodes_names[j])
                ]["distance"]
                weights.append(1.0 / (nodes_distance.item() + epsilon) * constant) # we want smaller distances to have a higher weight
            else:
                weights.append(
                    1
                )  # how much distance do we want for the self loop, do we use the original 0 or?

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)

    return edge_index, edge_weight

## SMOTE
def apply_smote_to_eeg_dataset(eeg_dataset):
    # Extract and flatten data
    X = []
    y = []
    original_shape = None

    for i in range(len(eeg_dataset)):
        data, label = eeg_dataset[i]
        if original_shape is None:
            original_shape = data.shape
        X.append(data.flatten())
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Reshape X back to original EEG format
    X_resampled = X_resampled.reshape(-1, *original_shape)

    print("Resampled dataset shape %s" % Counter(y_resampled))

    # Combine X and y back together as list of tuples
    combined_dataset = list(
        zip(torch.from_numpy(X_resampled), torch.from_numpy(y_resampled))
    )

    return combined_dataset

# ------ Augmentations --------

def add_gaussian_noise(x, std=0.01):
    noise = np.random.randn(*x.shape) * std
    return x + noise


def frequency_scaling(x, scale_range=(0.9, 1.1)):
    scale = np.random.uniform(
        *scale_range, size=(x.shape[0],)
    )  # One scale per frequency
    return x * scale[:, np.newaxis]  # Shape: (F, 1)

def time_mask(x, max_mask_length=250):  # Adjust max_mask_length based on sampling rate
    x = x.copy()
    max_start = max(1, x.shape[0] - max_mask_length)
    t0 = np.random.randint(0, max_start)
    width = np.random.randint(1, min(max_mask_length, x.shape[0] - t0))
    x[t0 : t0 + width, :] = 0
    return x

def frequency_mask(x, max_width=20):
    x = x.copy()
    max_start = max(1, x.shape[0] - max_width)
    f0 = np.random.randint(0, max_start)
    width = np.random.randint(1, min(max_width, x.shape[0] - f0))
    x[f0 : f0 + width, :] = 0
    return x


def channel_amplitude_scaling(x, scale_range=(0.9, 1.1)):
    scale = np.random.uniform(
        *scale_range, size=(x.shape[1],)
    )  # One scale per channel/node
    return x * scale[np.newaxis, :]  # Shape: (1, N)


def apply_augmentations(x, p_dict):
    x_aug = x.copy()

    if random.random() < p_dict.get("gaussian", 0.0):
        x_aug = add_gaussian_noise(x_aug)

    if random.random() < p_dict.get("freq_mask", 0.0):
        x_aug = frequency_mask(x_aug)

    if random.random() < p_dict.get("time_mask", 0.0):
        x_aug = time_mask(x_aug)

    if random.random() < p_dict.get("freq_scale", 0.0):
        x_aug = frequency_scaling(x_aug)

    if random.random() < p_dict.get("amp_scale", 0.0):
        x_aug = channel_amplitude_scaling(x_aug)

    return x_aug


class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, augment=True, p_dict=None):
        self.base_dataset = base_dataset
        self.augment = augment
        self.p_dict = p_dict or {
            "gaussian": 0.5,
            "freq_mask": 0.5,
            "time_mask": 0.0,
            "freq_scale": 0.5,
            "amp_scale": 0.5,
        }

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        signal, label = self.base_dataset[idx]
        if self.augment:
            signal = apply_augmentations(signal, self.p_dict)
        return signal, label
