import os
from typing import Tuple, Any

import numpy as np
import torch
from torch.utils.data import Dataset

from nebula.core.datasets.nebuladataset import NebulaDataset, NebulaPartitionHandler


class KDDCUP99TorchDataset(Dataset):
    """
    Simple torch Dataset wrapper for tabular KDDCUP99 data.

    Returns:
        x: torch.float32 tensor of shape (n_features,)
        y: torch.long scalar in [0, num_classes-1]
    """
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        continuous_features: list[int] | None = None,
        binary_features: list[int] | None = None,
    ):
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
        self.feature_names = feature_names or [f"feature_{i}" for i in range(self.x.shape[1])]
        self.continuous_features = continuous_features or []
        self.binary_features = binary_features or []
        self.input_dim = int(self.x.shape[1])

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_i = torch.from_numpy(self.x[idx])
        y_i = torch.tensor(self.y[idx], dtype=torch.long)
        return x_i, y_i


class KDDCUP99PartitionHandler(NebulaPartitionHandler):
    """
    Partition handler for tabular datasets.

    NebulaPartitionHandler provides (data, target) from the partition storage.
    For images, we usually convert to PIL and apply torchvision transforms.
    Here we convert features to float32 torch tensors and targets to long.
    """
    def __init__(self, file_path: str, prefix: str, config: Any, empty: bool = False):
        super().__init__(file_path, prefix, config, empty)

        # For tabular data we typically don't apply torchvision transforms.
        self.transform = None

    def __getitem__(self, idx: int):
        data, target = super().__getitem__(idx)

        # Defensive: depending on how NebulaPartitionHandler stores/returns,
        # "data" might be list/tuple/np.ndarray. Ensure we end up with 1D float32 tensor.
        if isinstance(data, tuple):
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


class KDDCUP99Dataset(NebulaDataset):
    """
    KDDCUP99 dataset integration for Nebula.

    Notes:
    - KDDCUP99 is a tabular intrusion-detection dataset.
    - sklearn fetch_kddcup99 exposes 41 features and 23 classes.
    - Some columns are categorical/string-like, so we one-hot encode them.
    - Targets may come as bytes/strings, so we map them to 0..num_classes-1.

    Requirements:
    - scikit-learn must be installed
    - pandas must be installed
    """
    def __init__(
        self,
        num_classes: int = 23,
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
        subset: str | None = None,
        percent10: bool = True,
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
        self.subset = subset
        self.percent10 = percent10

    def initialize_dataset(self):
        if self.train_set is None or self.test_set is None:
            self.train_set, self.test_set = self.load_kddcup99_dataset()

        self.data_partitioning(plot=True)

    def load_kddcup99_dataset(self):
        """
        Loads KDDCUP99 via sklearn, performs deterministic preprocessing
        and train/test split, and wraps into torch Datasets.
        """
        # Local cache directory for sklearn dataset downloads
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        os.makedirs(data_dir, exist_ok=True)

        try:
            import pandas as pd
            from sklearn.datasets import fetch_kddcup99
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
        except Exception as e:
            raise ImportError(
                "KDDCUP99Dataset requires scikit-learn and pandas. "
                "Install them (e.g., pip install scikit-learn pandas)."
            ) from e

        kdd = fetch_kddcup99(
            subset=self.subset,
            data_home=data_dir,
            shuffle=True,
            random_state=self.seed,
            percent10=self.percent10,
            download_if_missing=True,
            as_frame=True,
        )

        x = kdd.data
        y = kdd.target

        # Defensive conversion to pandas objects
        if not hasattr(x, "columns"):
            x = pd.DataFrame(x)
        if not hasattr(y, "astype"):
            y = pd.Series(y)

        # Decode bytes -> str where needed
        def _decode_if_bytes(v):
            if isinstance(v, (bytes, bytearray)):
                return v.decode("utf-8", errors="ignore")
            return v

        # Some KDDCUP99 columns are categorical (e.g. protocol/service/flag).
        # We decode bytes and one-hot encode object/category columns.
        for col in x.columns:
            if x[col].dtype == object:
                x[col] = x[col].map(_decode_if_bytes)

        y = y.map(_decode_if_bytes)
        numeric_columns = x.select_dtypes(exclude=["object", "category"]).columns.tolist()

        # One-hot encode categorical columns, keep numeric ones as-is.
        x = pd.get_dummies(x, drop_first=False)
        feature_names = [str(col) for col in x.columns]
        numeric_columns = [col for col in numeric_columns if col in x.columns]
        continuous_features = [x.columns.get_loc(col) for col in numeric_columns]
        binary_features = [i for i in range(len(feature_names)) if i not in continuous_features]

        # Map labels to 0..num_classes-1 deterministically
        y = pd.Series(y).astype(str)
        classes = sorted(y.unique().tolist())
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        y = y.map(class_to_idx).to_numpy(dtype=np.int64, copy=False)

        # Keep self.num_classes aligned with actual loaded subset
        self.num_classes = len(classes)

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

        x_train_np = x_train.astype(np.float32).to_numpy(copy=True)
        x_test_np = x_test.astype(np.float32).to_numpy(copy=True)

        # Scale the original numeric columns after splitting. One-hot columns stay binary.
        if continuous_features:
            scaler = StandardScaler()
            x_train_np[:, continuous_features] = scaler.fit_transform(x_train_np[:, continuous_features])
            x_test_np[:, continuous_features] = scaler.transform(x_test_np[:, continuous_features])

        train_ds = KDDCUP99TorchDataset(
            x_train_np,
            y_train,
            feature_names=feature_names,
            continuous_features=continuous_features,
            binary_features=binary_features,
        )
        test_ds = KDDCUP99TorchDataset(
            x_test_np,
            y_test,
            feature_names=feature_names,
            continuous_features=continuous_features,
            binary_features=binary_features,
        )

        # Optional: preserve original class names for inspection/debugging
        train_ds.classes = classes
        test_ds.classes = classes

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
