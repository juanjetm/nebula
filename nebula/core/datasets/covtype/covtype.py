# nebula/core/datasets/covtype/covtype.py

import os
from typing import Tuple, Any

import numpy as np
import torch
from torch.utils.data import Dataset

from nebula.core.datasets.nebuladataset import NebulaDataset, NebulaPartitionHandler


class CovtypeTorchDataset(Dataset):
    """
    Simple torch Dataset wrapper for tabular Covtype data.

    Returns:
        x: torch.float32 tensor of shape (n_features,)
        y: torch.long scalar in [0, num_classes-1]
    """
    def __init__(self, x: np.ndarray, y: np.ndarray):
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("x and y must be numpy arrays")

        if x.ndim != 2:
            raise ValueError(f"x must be 2D (n_samples, n_features). Got shape={x.shape}")
        if y.ndim != 1:
            y = y.reshape(-1)

        if x.shape[0] != y.shape[0]:
            raise ValueError(f"x and y must have same number of samples. Got {x.shape[0]} != {y.shape[0]}")

        self.x = x.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)

        self.data = self.x
        self.targets = self.y

        n_classes = int(np.max(self.targets)) + 1
        self.classes = [str(i) for i in range(n_classes)]

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_i = torch.from_numpy(self.x[idx])
        y_i = torch.tensor(self.y[idx], dtype=torch.long)
        return x_i, y_i


class CovtypePartitionHandler(NebulaPartitionHandler):
    """
    Partition handler for tabular datasets.

    NebulaPartitionHandler provides (data, target) from the partition storage.
    For images, we usually convert to PIL and apply torchvision transforms.
    Here we convert features to float32 torch tensors and targets to long.
    """
    def __init__(self, file_path: str, prefix: str, config: Any, empty: bool = False):
        super().__init__(file_path, prefix, config, empty)

        # For tabular data we typically don't apply torchvision transforms.
        # If you later want normalization here, do it explicitly and carefully
        # (train stats vs test stats, per-partition stats, etc.).
        self.transform = None

    def __getitem__(self, idx: int):
        data, target = super().__getitem__(idx)

        # Defensive: depending on how NebulaPartitionHandler stores/returns,
        # "data" might be list/tuple/np.ndarray. Ensure we end up with 1D float32 tensor.
        if isinstance(data, tuple):
            # Some vision datasets store (img, meta). For tabular we ignore extras.
            data = data[0]

        if isinstance(data, torch.Tensor):
            x = data.to(dtype=torch.float32)
        else:
            x = torch.tensor(np.asarray(data), dtype=torch.float32)

        # Ensure target in [0..num_classes-1] and torch.long
        if isinstance(target, torch.Tensor):
            y = target.to(dtype=torch.long)
        else:
            y = torch.tensor(int(target), dtype=torch.long)

        if self.target_transform is not None:
            y = self.target_transform(y)

        return x, y


class CovtypeDataset(NebulaDataset):
    """
    Covtype (Forest CoverType) dataset integration for Nebula.

    Notes:
    - Covtype has 7 classes.
    - Features are tabular (54 features in the classic version).
    - We provide a simple train/test split with fixed seed.

    Requirements:
    - scikit-learn must be installed (for fetch_covtype + train_test_split).
    """
    def __init__(
        self,
        num_classes: int = 7,
        partitions_number: int = 1,
        batch_size: int = 32,
        num_workers: int = 4,
        iid: bool = True,
        partition: str = "dirichlet",
        partition_parameter: float = 0.5,
        seed: int = 42,
        config_dir: str | None = None,
        test_size: float = 0.2,
        train_limit: int | None = None,
        test_limit: int | None = None,
    ):
        super().__init__(
            num_classes=num_classes,
            partitions_number=partitions_number,
            batch_size=batch_size,
            num_workers=num_workers,
            iid=iid,
            partition=partition,
            partition_parameter=partition_parameter,
            seed=seed,
            config_dir=config_dir,
        )
        self.test_size = float(test_size)
        self.train_limit = train_limit
        self.test_limit = test_limit

    def initialize_dataset(self):
        if self.train_set is None or self.test_set is None:
            self.train_set, self.test_set = self.load_covtype_dataset()

        self.data_partitioning(plot=True)

    def load_covtype_dataset(self):
        """
        Loads Covtype via sklearn, performs a deterministic train/test split,
        and wraps into torch Datasets.
        """
        # Local cache directory for sklearn dataset downloads
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        os.makedirs(data_dir, exist_ok=True)

        try:
            from sklearn.datasets import fetch_covtype
            from sklearn.model_selection import train_test_split
        except Exception as e:
            raise ImportError(
                "CovtypeDataset requires scikit-learn. Install it (e.g., pip install scikit-learn)."
            ) from e

        cov = fetch_covtype(data_home=data_dir, download_if_missing=True)

        x = cov.data
        y = cov.target  # commonly 1..7 in sklearn

        # Map labels to 0..6 (CrossEntropyLoss convention)
        # If already 0..6, this is harmless for 1..7 only if we detect min.
        y = np.asarray(y).reshape(-1)
        if y.min() == 1:
            y = y - 1

        # Split "grande"
        x_train, x_test, y_train, y_test = train_test_split(
            x, y,
            test_size=self.test_size,
            random_state=self.seed,
            shuffle=True,
            stratify=y,
        )

        # Submuestreo estratificado (corto y determinista)
        if self.train_limit is not None and len(y_train) > self.train_limit:
            x_train, _, y_train, _ = train_test_split(
                x_train, y_train,
                train_size=self.train_limit,
                random_state=self.seed,
                shuffle=True,
                stratify=y_train,
            )

        if self.test_limit is not None and len(y_test) > self.test_limit:
            x_test, _, y_test, _ = train_test_split(
                x_test, y_test,
                train_size=self.test_limit,
                random_state=self.seed,
                shuffle=True,
                stratify=y_test,
            )

        train_ds = CovtypeTorchDataset(x_train, y_train)
        test_ds = CovtypeTorchDataset(x_test, y_test)

        return train_ds, test_ds

    def generate_non_iid_map(self, dataset, partition: str = "dirichlet", partition_parameter: float = 0.5):
        if partition == "dirichlet":
            partitions_map = self.dirichlet_partition(dataset, alpha=partition_parameter)
        elif partition == "percent":
            partitions_map = self.percentage_partition(dataset, percentage=partition_parameter)
        else:
            raise ValueError(f"Partition {partition} is not supported for Non-IID map")

        return partitions_map

    def generate_iid_map(self, dataset, partition: str = "balancediid", partition_parameter: float = 2):
        if partition == "balancediid":
            partitions_map = self.balanced_iid_partition(dataset)
        elif partition == "unbalancediid":
            partitions_map = self.unbalanced_iid_partition(dataset, imbalance_factor=partition_parameter)
        else:
            raise ValueError(f"Partition {partition} is not supported for IID map")

        return partitions_map
