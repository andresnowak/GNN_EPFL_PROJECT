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