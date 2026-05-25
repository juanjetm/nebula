from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


CONTINUOUS = "continuous"
INTEGER = "integer"
NON_PERTURBABLE = "non_perturbable"


@dataclass(frozen=True)
class TabularAdversarialMetadata:
    """Minimal metadata for tabular adversarial training."""

    feature_names: list[str]
    feature_types: list[str]
    feature_min_norm: list[float]
    feature_max_norm: list[float]
    integer_step_norm: dict[int, float] | None = None

    def __post_init__(self):
        n_features = len(self.feature_names)
        if len(self.feature_types) != n_features:
            raise ValueError("feature_types length must match feature_names length")
        if len(self.feature_min_norm) != n_features:
            raise ValueError("feature_min_norm length must match feature_names length")
        if len(self.feature_max_norm) != n_features:
            raise ValueError("feature_max_norm length must match feature_names length")
        invalid_types = set(self.feature_types) - {CONTINUOUS, INTEGER, NON_PERTURBABLE}
        if invalid_types:
            raise ValueError(f"Unsupported tabular feature types: {sorted(invalid_types)}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TabularAdversarialMetadata":
        return cls(
            feature_names=[str(value) for value in data["feature_names"]],
            feature_types=[str(value) for value in data["feature_types"]],
            feature_min_norm=[float(value) for value in data["feature_min_norm"]],
            feature_max_norm=[float(value) for value in data["feature_max_norm"]],
            integer_step_norm={int(k): float(v) for k, v in data.get("integer_step_norm", {}).items()},
        )
