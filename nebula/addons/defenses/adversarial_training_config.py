from dataclasses import dataclass
from typing import Any

IMAGE_ADVERSARIAL_ATTACKS = {"fgsm", "pgd"}
CAA_TABULAR_DATASETS = {"AdultCensus"}

ERR_IMAGE_ATTACK = "image adversarial_training.attack must be one of: fgsm, pgd"
ERR_MODE = "adversarial_training.mode must be one of: clean, adversarial, mixed"
ERR_EPSILON = "adversarial_training.epsilon must be >= 0"
ERR_ALPHA = "adversarial_training.alpha must be >= 0"
ERR_STEPS = "adversarial_training.steps must be >= 1"
ERR_APPLY_PROBABILITY = "adversarial_training.apply_probability must be in [0, 1]"
ERR_LOSS_WEIGHTS = "adversarial_training loss weights must be >= 0"
ERR_MIXED_WEIGHTS = "adversarial_training mixed mode requires at least one positive loss weight"
ERR_CLIP_BOUNDS = "adversarial_training.clip_min must be smaller than clip_max"
ERR_TABULAR_METADATA = "Tabular adversarial training requires tabular_metadata"
ERR_UNSUPPORTED_ATTACK = "Unsupported adversarial training attack: {attack}"

IMAGE_DATASET_NORMALIZATION = {
    "MNIST": ((0.5,), (0.5,)),
    "FashionMNIST": ((0.5,), (0.5,)),
    "EMNIST": ((0.5,), (0.5,)),
    "CIFAR10": ((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    "CIFAR100": ((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
}


@dataclass(frozen=True)
class AdversarialTrainingConfig:
    enabled: bool = False
    dataset_name: str | None = None
    domain: str = "image"
    attack: str = "fgsm"
    epsilon: float = 8.0 / 255.0
    alpha: float | None = None
    steps: int = 1
    clean_weight: float = 0.5
    adversarial_weight: float = 0.5
    mode: str = "mixed"
    apply_probability: float = 1.0
    clip_min: float = 0.0
    clip_max: float = 1.0
    log_adversarial_metrics: bool = True


def config_from_participant(participant_config: dict[str, Any]) -> AdversarialTrainingConfig | None:
    # Read the raw participant config and normalize it into a typed defense config.
    raw = participant_config.get("defense_args", {}).get("adversarial_training", {})
    if not raw or not raw.get("enabled", False):
        return None

    dataset_name = participant_config.get("data_args", {}).get("dataset")
    domain = str(raw.get("domain", "image")).lower()
    # Tabular adversarial training exposes a single attack: CAA.
    attack = "caa" if domain == "tabular" else str(raw.get("attack", "fgsm")).lower()

    return AdversarialTrainingConfig(
        enabled=True,
        dataset_name=dataset_name,
        domain=domain,
        attack=attack,
        epsilon=float(raw.get("epsilon", 8.0 / 255.0)),
        alpha=float(raw["alpha"]) if raw.get("alpha") is not None else None,
        steps=int(raw.get("steps", 1)),
        clean_weight=float(raw.get("clean_weight", 0.5)),
        adversarial_weight=float(raw.get("adversarial_weight", 0.5)),
        mode=str(raw.get("mode", "mixed")).lower(),
        apply_probability=float(raw.get("apply_probability", 1.0)),
        clip_min=float(raw.get("clip_min", 0.0)),
        clip_max=float(raw.get("clip_max", 1.0)),
        log_adversarial_metrics=bool(raw.get("log_adversarial_metrics", True)),
    )


def validate_config(config: AdversarialTrainingConfig) -> None:
    # Fail early when a frontend/backend config value cannot produce a valid attack.
    if config.mode not in {"clean", "adversarial", "mixed"}:
        raise ValueError(ERR_MODE)
    if config.domain == "image" and config.attack not in IMAGE_ADVERSARIAL_ATTACKS:
        raise ValueError(ERR_IMAGE_ATTACK)
    if config.epsilon < 0:
        raise ValueError(ERR_EPSILON)
    if config.alpha is not None and config.alpha < 0:
        raise ValueError(ERR_ALPHA)
    if config.steps < 1:
        raise ValueError(ERR_STEPS)
    if not 0.0 <= config.apply_probability <= 1.0:
        raise ValueError(ERR_APPLY_PROBABILITY)
    if config.clean_weight < 0 or config.adversarial_weight < 0:
        raise ValueError(ERR_LOSS_WEIGHTS)
    if config.mode == "mixed" and config.clean_weight + config.adversarial_weight == 0:
        raise ValueError(ERR_MIXED_WEIGHTS)
    if config.clip_min >= config.clip_max:
        raise ValueError(ERR_CLIP_BOUNDS)
