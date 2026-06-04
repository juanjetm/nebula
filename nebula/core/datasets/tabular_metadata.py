from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CONTINUOUS = "continuous"
INTEGER = "integer"
CATEGORICAL = "categorical"
NON_PERTURBABLE = "non_perturbable"

ERR_FEATURE_TYPES_LENGTH = "feature_types length must match feature_names length"
ERR_FEATURE_MIN_LENGTH = "feature_min_norm length must match feature_names length"
ERR_FEATURE_MAX_LENGTH = "feature_max_norm length must match feature_names length"
ERR_UNSUPPORTED_FEATURE_TYPES = "Unsupported tabular feature types: {feature_types}"
ERR_FEATURE_BOUNDS = "feature_min_norm must be <= feature_max_norm for every feature"
ERR_INTEGER_STEP_INDEX = "integer_step_norm contains invalid feature indices: {indices}"
ERR_INTEGER_STEP_VALUE = "integer_step_norm values must be > 0"
ERR_INTEGER_STEP_TYPE = "integer_step_norm contains non-integer feature indices: {indices}"
ERR_CATEGORICAL_GROUP_SIZE = "categorical_groups entries must contain at least two feature indices"
ERR_CATEGORICAL_GROUP_INDEX = "categorical_groups contains invalid feature indices: {indices}"
ERR_CATEGORICAL_GROUP_TYPE = "categorical_groups contains non-categorical feature indices: {indices}"
ERR_CATEGORICAL_GROUP_OVERLAP = "categorical_groups contains duplicated feature indices: {indices}"
ERR_CATEGORICAL_GROUP_COVERAGE = "categorical feature indices missing from categorical_groups: {indices}"


@dataclass(frozen=True)
class TabularAdversarialMetadata:
    """Minimal metadata for tabular adversarial training."""

    # These fields describe the exact vector received by the model after preprocessing.
    # Bounds and steps must use the same normalized space as the training tensors.
    feature_names: list[str]
    feature_types: list[str]
    feature_min_norm: list[float]
    feature_max_norm: list[float]
    integer_step_norm: dict[int, float] | None = None
    categorical_groups: list[list[int]] | None = None

    def __post_init__(self):
        # Fail early if a dataset exposes incomplete metadata. The attack relies on
        # these arrays lining up feature-by-feature.
        n_features = len(self.feature_names)
        if len(self.feature_types) != n_features:
            raise ValueError(ERR_FEATURE_TYPES_LENGTH)
        if len(self.feature_min_norm) != n_features:
            raise ValueError(ERR_FEATURE_MIN_LENGTH)
        if len(self.feature_max_norm) != n_features:
            raise ValueError(ERR_FEATURE_MAX_LENGTH)

        # Every feature needs a valid normalized interval so projection can clamp safely.
        invalid_bounds = [
            idx
            for idx, (min_value, max_value) in enumerate(
                zip(self.feature_min_norm, self.feature_max_norm, strict=True)
            )
            if min_value > max_value
        ]
        if invalid_bounds:
            raise ValueError(ERR_FEATURE_BOUNDS)
        invalid_types = set(self.feature_types) - {CONTINUOUS, INTEGER, CATEGORICAL, NON_PERTURBABLE}
        if invalid_types:
            raise ValueError(ERR_UNSUPPORTED_FEATURE_TYPES.format(feature_types=sorted(invalid_types)))

        # Integer steps represent the normalized distance between consecutive integer values.
        # They only make sense for features marked as INTEGER.
        invalid_step_indices = [
            idx
            for idx in (self.integer_step_norm or {})
            if int(idx) < 0 or int(idx) >= n_features
        ]
        if invalid_step_indices:
            raise ValueError(ERR_INTEGER_STEP_INDEX.format(indices=invalid_step_indices))
        non_integer_step_indices = [
            idx
            for idx in (self.integer_step_norm or {})
            if self.feature_types[int(idx)] != INTEGER
        ]
        if non_integer_step_indices:
            raise ValueError(ERR_INTEGER_STEP_TYPE.format(indices=non_integer_step_indices))
        if any(step <= 0 for step in (self.integer_step_norm or {}).values()):
            raise ValueError(ERR_INTEGER_STEP_VALUE)

        # Categorical groups represent one original categorical column after one-hot encoding.
        # Each group must be disjoint so projection can activate exactly one value per group.
        grouped_counts: dict[int, int] = {}
        for group in self.categorical_groups or []:
            if len(group) < 2:
                raise ValueError(ERR_CATEGORICAL_GROUP_SIZE)
            invalid_indices = [idx for idx in group if idx < 0 or idx >= n_features]
            if invalid_indices:
                raise ValueError(ERR_CATEGORICAL_GROUP_INDEX.format(indices=invalid_indices))
            non_categorical_indices = [idx for idx in group if self.feature_types[idx] != CATEGORICAL]
            if non_categorical_indices:
                raise ValueError(ERR_CATEGORICAL_GROUP_TYPE.format(indices=non_categorical_indices))
            for idx in group:
                grouped_counts[idx] = grouped_counts.get(idx, 0) + 1

        duplicated_group_indices = sorted(idx for idx, count in grouped_counts.items() if count > 1)
        if duplicated_group_indices:
            raise ValueError(ERR_CATEGORICAL_GROUP_OVERLAP.format(indices=duplicated_group_indices))

        # A categorical feature without a group cannot be projected back to a valid one-hot state.
        grouped_categorical_indices = {
            idx
            for group in self.categorical_groups or []
            for idx in group
        }
        categorical_indices = {
            idx
            for idx, feature_type in enumerate(self.feature_types)
            if feature_type == CATEGORICAL
        }
        missing_categorical_indices = sorted(categorical_indices - grouped_categorical_indices)
        if missing_categorical_indices:
            raise ValueError(ERR_CATEGORICAL_GROUP_COVERAGE.format(indices=missing_categorical_indices))

    def to_dict(self) -> dict[str, Any]:
        # Partitions persist metadata as JSON-like dictionaries in HDF5 attributes.
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TabularAdversarialMetadata:
        # HDF5/JSON round-trips can turn integer keys into strings; normalize them here.
        return cls(
            feature_names=[str(value) for value in data["feature_names"]],
            feature_types=[str(value) for value in data["feature_types"]],
            feature_min_norm=[float(value) for value in data["feature_min_norm"]],
            feature_max_norm=[float(value) for value in data["feature_max_norm"]],
            integer_step_norm={int(k): float(v) for k, v in (data.get("integer_step_norm") or {}).items()},
            categorical_groups=[
                [int(idx) for idx in group]
                for group in data.get("categorical_groups") or []
            ],
        )
