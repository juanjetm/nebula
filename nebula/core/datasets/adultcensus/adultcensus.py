# nebula/core/datasets/adultcensus/adultcensus.py

import os
from typing import Any, ClassVar

import numpy as np
import torch
from torch.utils.data import Dataset

from nebula.core.datasets.nebuladataset import NebulaDataset, NebulaPartitionHandler
from nebula.core.datasets.tabular_metadata import CATEGORICAL, CONTINUOUS, INTEGER, TabularAdversarialMetadata


class AdultCensusTorchDataset(Dataset):
    """
    Torch Dataset wrapper for Adult Census Income dataset (tabular, already numeric).
    x: float32 tensor (n_features,)
    y: long scalar {0,1} where 1 means >50K
    """
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        continuous_features: list[int] | None = None,
        integer_features: list[int] | None = None,
        categorical_features: list[int] | None = None,
        categorical_groups: list[list[int]] | None = None,
        tabular_metadata: dict | None = None,
    ):
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("x and y must be numpy arrays")

        if x.ndim != 2:
            raise ValueError(f"x must be 2D (n_samples, n_features). Got shape={x.shape}")

        y_arr: np.ndarray = np.asarray(y).reshape(-1)
        if x.shape[0] != y_arr.shape[0]:
            raise ValueError(f"x and y must have same number of samples. Got {x.shape[0]} != {y_arr.shape[0]}")

        self.x: np.ndarray = x.astype(np.float32, copy=False)
        self.y: np.ndarray = y_arr.astype(np.int64, copy=False)

        # Nebula conventions
        self.data: np.ndarray = self.x
        self.targets: np.ndarray = self.y
        self.classes: list[str] = ["<=50K", ">50K"]
        self.feature_names = feature_names or [f"feature_{i}" for i in range(self.x.shape[1])]
        self.continuous_features = continuous_features or []
        self.integer_features = integer_features or []
        self.categorical_features = categorical_features or []
        self.categorical_groups = categorical_groups or []
        self.tabular_metadata = tabular_metadata
        self.input_dim = int(self.x.shape[1])

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_i: torch.Tensor = torch.from_numpy(self.x[idx])
        y_i: torch.Tensor = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return x_i, y_i


class AdultCensusPartitionHandler(NebulaPartitionHandler):
    """
    Partition handler for tabular data.
    """
    def __init__(self, file_path: str, prefix: str, config: Any, empty: bool = False):
        super().__init__(file_path, prefix, config, empty)
        self.transform = None  # no torchvision transforms for tabular

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        data, target = super().__getitem__(idx)

        # Some Nebula handlers may wrap data in tuples
        if isinstance(data, tuple):
            data = data[0]

        if isinstance(data, torch.Tensor):
            x: torch.Tensor = data.to(dtype=torch.float32)
        else:
            x = torch.tensor(np.asarray(data), dtype=torch.float32)

        if isinstance(target, torch.Tensor):
            y: torch.Tensor = target.to(dtype=torch.long)
        else:
            y = torch.tensor(int(target), dtype=torch.long)

        if self.target_transform is not None:
            y = self.target_transform(y)

        return x, y


class AdultCensusDataset(NebulaDataset):
    """
    Adult Census Income dataset integration for Nebula.

    - 2 classes: <=50K vs >50K
    - mixed tabular data -> numeric model input via preprocessing
    - deterministic stratified train/test split
    """
    CONTINUOUS_COLUMNS: ClassVar[list[str]] = []
    INTEGER_COLUMNS: ClassVar[list[str]] = [
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]
    CATEGORICAL_COLUMNS: ClassVar[list[str]] = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    def __init__(
        self,
        num_classes: int = 2,
        partitions_number: int = 1,
        batch_size: int = 32,
        num_workers: int = 4,
        iid: bool = True,
        partition: str = "dirichlet",
        partition_parameter: float = 0.5,
        seed: int = 42,
        config_dir: str | None = None,
        test_size: float = 0.2,
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
        self.test_size: float = float(test_size)

    def initialize_dataset(self) -> None:
        if self.train_set is None or self.test_set is None:
            self.train_set, self.test_set = self.load_adult_census_dataset()

        self.data_partitioning(plot=True)

    @staticmethod
    def _make_ohe_dense():
        """
        scikit-learn compatibility:
        - older: OneHotEncoder(..., sparse=False)
        - newer: OneHotEncoder(..., sparse_output=False)
        """
        from sklearn.preprocessing import OneHotEncoder

        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=False)

    @classmethod
    def _validate_manual_schema(cls, columns) -> None:
        continuous_columns = set(cls.CONTINUOUS_COLUMNS)
        integer_columns = set(cls.INTEGER_COLUMNS)
        categorical_columns = set(cls.CATEGORICAL_COLUMNS)
        overlapping_columns = sorted(
            (continuous_columns & integer_columns)
            | (continuous_columns & categorical_columns)
            | (integer_columns & categorical_columns)
        )
        if overlapping_columns:
            raise ValueError(f"AdultCensusDataset columns configured twice: {overlapping_columns}")

        configured_columns = continuous_columns | integer_columns | categorical_columns
        dataset_columns = set(columns)
        missing_columns = sorted(configured_columns - dataset_columns)
        if missing_columns:
            raise ValueError(f"AdultCensusDataset is missing configured columns: {missing_columns}")
        unconfigured_columns = sorted(dataset_columns - configured_columns)
        if unconfigured_columns:
            raise ValueError(f"AdultCensusDataset has unconfigured columns: {unconfigured_columns}")

    def load_adult_census_dataset(self) -> tuple[AdultCensusTorchDataset, AdultCensusTorchDataset]:
        """
        Loads Adult dataset from OpenML and preprocesses to all-numeric features.

        Steps:
          1) fetch_openml(data_id=1590, as_frame=True)
          2) y = (target == '>50K').astype(int)
          3) replace '?' with NA for missing values
          4) ColumnTransformer:
              - continuous: median impute + StandardScaler
              - integer: median impute + StandardScaler
              - categorical: most_frequent impute + OneHotEncoder(dense)
          5) train/test split (stratified), fit preprocessing only on train (avoid leakage)
        """
        data_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        os.makedirs(data_dir, exist_ok=True)

        try:
            import pandas as pd
            from sklearn.compose import ColumnTransformer
            from sklearn.datasets import fetch_openml
            from sklearn.impute import SimpleImputer
            from sklearn.model_selection import train_test_split
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
        except Exception as e:
            raise ImportError(
                "AdultCensusDataset requires pandas + scikit-learn. Install them (e.g., pip install pandas scikit-learn)."
            ) from e

        # 1) Load from OpenML
        bunch = fetch_openml(data_id=1590, as_frame=True, data_home=data_dir)
        X_df = bunch.data.copy()
        y_raw = bunch.target

        # 2) Target -> {0,1}
        # Normalize spaces to avoid variants like ' >50K'
        y_str = y_raw.astype(str).str.strip()
        y: np.ndarray = (y_str == ">50K").astype(np.int64).to_numpy()

        # 3) Replace '?' markers with np.nan and drop rows with missing configured features.
        X_df = X_df.replace(r"^\s*\?\s*$", np.nan, regex=True)
        self._validate_manual_schema(X_df.columns)

        numeric_columns = self.CONTINUOUS_COLUMNS + self.INTEGER_COLUMNS
        for column in numeric_columns:
            X_df[column] = pd.to_numeric(X_df[column], errors="coerce")
        for column in self.CATEGORICAL_COLUMNS:
            X_df[column] = X_df[column].astype(object)

        configured_columns = numeric_columns + self.CATEGORICAL_COLUMNS
        valid_rows = ~X_df[configured_columns].isna().any(axis=1)
        removed_rows = int((~valid_rows).sum())
        if removed_rows:
            import logging
            logging.getLogger().info("[AdultCensus] Dropping %s rows with NA values", removed_rows)
        X_df = X_df.loc[valid_rows].copy()
        y = y[valid_rows.to_numpy()]

        # 4) Preprocess
        numeric_transformer = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("ohe", self._make_ohe_dense()),
            ]
        )

        transformers = []
        if self.CONTINUOUS_COLUMNS:
            transformers.append(("continuous", numeric_transformer, self.CONTINUOUS_COLUMNS))
        if self.INTEGER_COLUMNS:
            transformers.append(("integer", numeric_transformer, self.INTEGER_COLUMNS))
        if self.CATEGORICAL_COLUMNS:
            transformers.append(("categorical", categorical_transformer, self.CATEGORICAL_COLUMNS))

        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

        # 5) Split then fit on train
        X_train_df, X_test_df, y_train, y_test = train_test_split(
            X_df,
            y,
            test_size=self.test_size,
            random_state=self.seed,
            shuffle=True,
            stratify=y,
        )

        X_train = preprocessor.fit_transform(X_train_df)
        X_test = preprocessor.transform(X_test_df)
        try:
            feature_names = [str(name) for name in preprocessor.get_feature_names_out()]
        except Exception:
            feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

        # In case some sklearn path returns sparse matrices, densify safely
        if hasattr(X_train, "toarray"):
            X_train = X_train.toarray()
        if hasattr(X_test, "toarray"):
            X_test = X_test.toarray()

        X_train_np: np.ndarray = np.asarray(X_train, dtype=np.float32)
        import logging
        logging.getLogger().info(f"[AdultCensus] X_train shape = {X_train_np.shape}")
        logging.getLogger().info(f"[AdultCensus] INPUT_DIM (post-OHE) = {int(X_train_np.shape[1])}")
        X_test_np: np.ndarray = np.asarray(X_test, dtype=np.float32)
        continuous_features = [
            idx for idx, name in enumerate(feature_names)
            if name.startswith("continuous__")
        ]
        integer_features = [
            idx for idx, name in enumerate(feature_names)
            if name.startswith("integer__")
        ]
        categorical_features = [
            idx for idx, name in enumerate(feature_names)
            if name.startswith("categorical__")
        ]
        continuous_feature_set = set(continuous_features)
        integer_feature_set = set(integer_features)
        categorical_feature_set = set(categorical_features)
        assigned_feature_set = continuous_feature_set | integer_feature_set | categorical_feature_set
        unknown_features = [
            feature_names[idx]
            for idx in range(len(feature_names))
            if idx not in assigned_feature_set
        ]
        if unknown_features:
            raise ValueError(f"AdultCensusDataset generated untyped features: {unknown_features}")
        feature_type_by_idx = {
            **{idx: CONTINUOUS for idx in continuous_feature_set},
            **{idx: INTEGER for idx in integer_feature_set},
            **{idx: CATEGORICAL for idx in categorical_feature_set},
        }

        categorical_groups = self._build_categorical_groups(feature_names)
        integer_step_norm = {}
        if integer_features:
            integer_scaler = preprocessor.named_transformers_["integer"].named_steps["scaler"]
            integer_step_norm = {
                idx: float(1.0 / scale)
                for idx, scale in zip(integer_features, integer_scaler.scale_, strict=False)
            }
        tabular_metadata = TabularAdversarialMetadata(
            feature_names=feature_names,
            feature_types=[feature_type_by_idx[idx] for idx in range(len(feature_names))],
            feature_min_norm=np.min(X_train_np, axis=0).astype(float).tolist(),
            feature_max_norm=np.max(X_train_np, axis=0).astype(float).tolist(),
            integer_step_norm=integer_step_norm,
            categorical_groups=categorical_groups,
        ).to_dict()
        logging.getLogger().info(
            "[AdultCensus] Tabular adversarial feature mask | continuous=%s | integer=%s | "
            "categorical=%s | categorical_groups=%s | continuous_features=%s | integer_features=%s | "
            "integer_step_norm=%s",
            len(continuous_features),
            len(integer_features),
            len(categorical_features),
            len(categorical_groups),
            [feature_names[idx] for idx in continuous_features],
            [feature_names[idx] for idx in integer_features],
            integer_step_norm,
        )

        train_ds = AdultCensusTorchDataset(
            X_train_np,
            np.asarray(y_train, dtype=np.int64),
            feature_names=feature_names,
            continuous_features=continuous_features,
            integer_features=integer_features,
            categorical_features=categorical_features,
            categorical_groups=categorical_groups,
            tabular_metadata=tabular_metadata,
        )
        test_ds = AdultCensusTorchDataset(
            X_test_np,
            np.asarray(y_test, dtype=np.int64),
            feature_names=feature_names,
            continuous_features=continuous_features,
            integer_features=integer_features,
            categorical_features=categorical_features,
            categorical_groups=categorical_groups,
            tabular_metadata=tabular_metadata,
        )

        return train_ds, test_ds

    @classmethod
    def _build_categorical_groups(cls, feature_names: list[str]) -> list[list[int]]:
        groups = []
        for column in cls.CATEGORICAL_COLUMNS:
            prefix = f"categorical__{column}_"
            group = [idx for idx, name in enumerate(feature_names) if name.startswith(prefix)]
            if group:
                groups.append(group)
        return groups

    def generate_non_iid_map(self, dataset, partition: str = "dirichlet", partition_parameter: float = 0.5):
        if partition == "dirichlet":
            return self.dirichlet_partition(dataset, alpha=partition_parameter)
        if partition == "percent":
            return self.percentage_partition(dataset, percentage=partition_parameter)
        raise ValueError(f"Partition {partition} is not supported for Non-IID map")

    def generate_iid_map(self, dataset, partition: str = "balancediid", partition_parameter: float = 2):
        if partition == "balancediid":
            return self.balanced_iid_partition(dataset)
        if partition == "unbalancediid":
            return self.unbalanced_iid_partition(dataset, imbalance_factor=partition_parameter)
        raise ValueError(f"Partition {partition} is not supported for IID map")
