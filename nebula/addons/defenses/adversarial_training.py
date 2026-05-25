import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

from nebula.config.config import TRAINING_LOGGER
from nebula.core.datasets.tabular_metadata import CONTINUOUS, INTEGER, TabularAdversarialMetadata

logging_training = logging.getLogger(TRAINING_LOGGER)

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


class AdversarialExampleGenerator(ABC):
    """Base interface for domain-specific adversarial example generators."""

    last_epsilon: float | None = None

    @abstractmethod
    def generate(self, model, x, y, criterion):
        raise NotImplementedError

    def _sample_epsilon(self, device: torch.device) -> float:
        epsilon_max = float(self.config.epsilon)
        if epsilon_max <= 0.0:
            self.last_epsilon = 0.0
            return 0.0

        epsilon_min = epsilon_max / 4.0
        epsilon_step = epsilon_max / 8.0
        num_values = max(int(round((epsilon_max - epsilon_min) / epsilon_step)) + 1, 1)
        index = int(torch.randint(num_values, (), device=device).item())
        epsilon = min(epsilon_min + index * epsilon_step, epsilon_max)
        self.last_epsilon = epsilon
        return epsilon


class ImageAdversarialExampleGenerator(AdversarialExampleGenerator):
    def __init__(self, config: AdversarialTrainingConfig, mean: tuple[float, ...], std: tuple[float, ...]):
        self.config = config
        self.mean = mean
        self.std = std

    def _channel_tensor(self, values: tuple[float, ...], x: torch.Tensor) -> torch.Tensor:
        shape = [1, len(values)] + [1] * max(x.dim() - 2, 0)
        return torch.tensor(values, dtype=x.dtype, device=x.device).view(*shape)

    def _epsilon(self, x: torch.Tensor, epsilon: float) -> torch.Tensor:
        std = self._channel_tensor(self.std, x)
        return float(epsilon) / std

    def _alpha(self, x: torch.Tensor, epsilon: float) -> torch.Tensor:
        alpha = self.config.alpha
        if alpha is None:
            alpha = epsilon / max(int(self.config.steps), 1)
        std = self._channel_tensor(self.std, x)
        return float(alpha) / std

    def _bounds(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self._channel_tensor(self.mean, x)
        std = self._channel_tensor(self.std, x)
        lower = (float(self.config.clip_min) - mean) / std
        upper = (float(self.config.clip_max) - mean) / std
        return lower, upper

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = self._channel_tensor(self.mean, x)
        std = self._channel_tensor(self.std, x)
        return (x * std + mean).clamp(float(self.config.clip_min), float(self.config.clip_max))

    def _project(self, x_adv: torch.Tensor, x_clean: torch.Tensor, epsilon: float) -> torch.Tensor:
        epsilon = self._epsilon(x_clean, epsilon)
        lower, upper = self._bounds(x_clean)
        x_adv = torch.max(torch.min(x_adv, x_clean + epsilon), x_clean - epsilon)
        return torch.max(torch.min(x_adv, upper), lower)


class ImageFGSMGenerator(ImageAdversarialExampleGenerator):
    def generate(self, model, x, y, criterion):
        epsilon = self._sample_epsilon(x.device)
        x_adv = x.detach().clone().requires_grad_(True)
        logits = model(x_adv)
        loss = criterion(logits, y)
        grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
        x_adv = x_adv + self._epsilon(x_adv, epsilon) * grad.sign()
        return self._project(x_adv.detach(), x.detach(), epsilon)


class ImagePGDGenerator(ImageAdversarialExampleGenerator):
    def generate(self, model, x, y, criterion):
        epsilon = self._sample_epsilon(x.device)
        x_clean = x.detach()
        x_adv = x_clean.clone()
        steps = max(int(self.config.steps), 1)

        for _ in range(steps):
            x_adv = x_adv.detach().requires_grad_(True)
            logits = model(x_adv)
            loss = criterion(logits, y)
            grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
            x_adv = x_adv + self._alpha(x_adv, epsilon) * grad.sign()
            x_adv = self._project(x_adv.detach(), x_clean, epsilon)

        return x_adv.detach()


class TabularAdversarialExampleGenerator(AdversarialExampleGenerator):
    """Adversarial generator for perturbable continuous and integer tabular features."""

    def __init__(self, config: AdversarialTrainingConfig, metadata: TabularAdversarialMetadata):
        self.config = config
        self.metadata = metadata
        self._tensor_cache: dict[tuple[torch.device, torch.dtype], dict[str, torch.Tensor]] = {}

    def _alpha(self, epsilon: float) -> float:
        if self.config.alpha is not None:
            return float(self.config.alpha)
        return float(epsilon) / max(int(self.config.steps), 1)

    def _tensors(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        key = (x.device, x.dtype)
        cached = self._tensor_cache.get(key)
        if cached is not None:
            return cached

        cached = {
            "continuous": torch.tensor(
                [feature_type == CONTINUOUS for feature_type in self.metadata.feature_types],
                dtype=torch.bool,
                device=x.device,
            ).view(1, -1),
            "integer": torch.tensor(
                [feature_type == INTEGER for feature_type in self.metadata.feature_types],
                dtype=torch.bool,
                device=x.device,
            ).view(1, -1),
            "min": torch.tensor(self.metadata.feature_min_norm, dtype=x.dtype, device=x.device).view(1, -1),
            "max": torch.tensor(self.metadata.feature_max_norm, dtype=x.dtype, device=x.device).view(1, -1),
        }
        cached["perturbable"] = cached["continuous"] | cached["integer"]
        integer_steps = torch.ones_like(cached["min"])
        for idx, step in (self.metadata.integer_step_norm or {}).items():
            integer_steps[0, int(idx)] = float(step)
        cached["integer_step"] = integer_steps
        self._tensor_cache[key] = cached
        return cached

    def _gradient(self, model, x, y, criterion):
        x_grad = x.detach().clone().requires_grad_(True)
        logits = model(x_grad)
        loss = criterion(logits, y)
        return torch.autograd.grad(loss, x_grad, only_inputs=True)[0]

    def _project(self, x_adv: torch.Tensor, x_clean: torch.Tensor, epsilon: float) -> torch.Tensor:
        tensors = self._tensors(x_clean)
        lower = torch.maximum(tensors["min"], x_clean - float(epsilon))
        upper = torch.minimum(tensors["max"], x_clean + float(epsilon))
        x_adv = torch.max(torch.min(x_adv, upper), lower)

        integer_mask = tensors["integer"]
        if integer_mask.any():
            step = torch.clamp(tensors["integer_step"], min=torch.finfo(x_adv.dtype).eps)
            projected_integer = torch.round((x_adv - tensors["min"]) / step) * step + tensors["min"]
            grid_lower = torch.ceil((lower - tensors["min"]) / step) * step + tensors["min"]
            grid_upper = torch.floor((upper - tensors["min"]) / step) * step + tensors["min"]
            projected_integer = torch.max(torch.min(projected_integer, grid_upper), grid_lower)
            has_valid_grid = grid_lower <= grid_upper
            projected_integer = torch.where(has_valid_grid, projected_integer, x_clean)
            x_adv = torch.where(integer_mask, projected_integer, x_adv)

        return torch.where(tensors["perturbable"], x_adv, x_clean)


class TabularFGSMGenerator(TabularAdversarialExampleGenerator):
    def generate(self, model, x, y, criterion):
        epsilon = self._sample_epsilon(x.device)
        grad = self._gradient(model, x, y, criterion)
        x_clean = x.detach()
        perturbable_mask = self._tensors(x_clean)["perturbable"]
        x_adv = x_clean + float(epsilon) * grad.sign() * perturbable_mask
        return self._project(x_adv.detach(), x_clean, epsilon)


class TabularPGDGenerator(TabularAdversarialExampleGenerator):
    def generate(self, model, x, y, criterion):
        epsilon = self._sample_epsilon(x.device)
        x_clean = x.detach()
        x_adv = x_clean.clone()
        steps = max(int(self.config.steps), 1)

        for _ in range(steps):
            grad = self._gradient(model, x_adv, y, criterion)
            perturbable_mask = self._tensors(x_clean)["perturbable"]
            x_adv = x_adv.detach() + self._alpha(epsilon) * grad.sign() * perturbable_mask
            x_adv = self._project(x_adv.detach(), x_clean, epsilon)

        return x_adv.detach()


class AdversarialTrainingDefense:
    """Batch-level adversarial training defense for Nebula models."""

    LOGGED_SAMPLES_PER_ROUND = 3

    def __init__(self, config: AdversarialTrainingConfig, generator: AdversarialExampleGenerator):
        self.config = config
        self.generator = generator
        self._logged_adversarial_samples_by_round: dict[int, int] = {}

    @classmethod
    def from_participant_config(
        cls,
        participant_config: dict[str, Any],
        partition=None,
    ) -> "AdversarialTrainingDefense | None":
        raw = participant_config.get("defense_args", {}).get("adversarial_training", {})
        if not raw or not raw.get("enabled", False):
            return None

        dataset_name = participant_config.get("data_args", {}).get("dataset")
        config = AdversarialTrainingConfig(
            enabled=True,
            dataset_name=dataset_name,
            domain=str(raw.get("domain", "image")).lower(),
            attack=str(raw.get("attack", "fgsm")).lower(),
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
        cls._validate_config(config)

        if config.domain == "tabular":
            supported_tabular_datasets = {"AdultCensus", "BreastCancer", "Covtype", "KDDCUP99"}
            if dataset_name not in supported_tabular_datasets:
                logging.warning(
                    "[AdversarialTrainingDefense] Skipping tabular adversarial training: dataset '%s' is not supported yet",
                    dataset_name,
                )
                return None
            metadata = cls._get_tabular_metadata(partition)
            generator = cls._build_tabular_generator(config, metadata)
            return cls(config=config, generator=generator)

        if config.domain != "image":
            logging.warning(
                "[AdversarialTrainingDefense] Skipping adversarial training: domain '%s' is not implemented yet",
                config.domain,
            )
            return None

        normalization = IMAGE_DATASET_NORMALIZATION.get(dataset_name)
        if normalization is None:
            logging.warning(
                "[AdversarialTrainingDefense] Skipping adversarial training: dataset '%s' has no image bounds",
                dataset_name,
            )
            return None

        generator = cls._build_generator(config, normalization)
        return cls(config=config, generator=generator)

    @staticmethod
    def _validate_config(config: AdversarialTrainingConfig) -> None:
        if config.mode not in {"clean", "adversarial", "mixed"}:
            raise ValueError("adversarial_training.mode must be one of: clean, adversarial, mixed")
        if config.attack not in {"fgsm", "pgd"}:
            raise ValueError("adversarial_training.attack must be one of: fgsm, pgd")
        if config.epsilon < 0:
            raise ValueError("adversarial_training.epsilon must be >= 0")
        if config.alpha is not None and config.alpha < 0:
            raise ValueError("adversarial_training.alpha must be >= 0")
        if config.steps < 1:
            raise ValueError("adversarial_training.steps must be >= 1")
        if not 0.0 <= config.apply_probability <= 1.0:
            raise ValueError("adversarial_training.apply_probability must be in [0, 1]")
        if config.clean_weight < 0 or config.adversarial_weight < 0:
            raise ValueError("adversarial_training loss weights must be >= 0")
        if config.mode == "mixed" and config.clean_weight + config.adversarial_weight == 0:
            raise ValueError("adversarial_training mixed mode requires at least one positive loss weight")
        if config.clip_min >= config.clip_max:
            raise ValueError("adversarial_training.clip_min must be smaller than clip_max")

    @staticmethod
    def _build_generator(config, normalization):
        mean, std = normalization
        if config.attack == "fgsm":
            return ImageFGSMGenerator(config, mean, std)
        if config.attack == "pgd":
            return ImagePGDGenerator(config, mean, std)
        raise ValueError(f"Unsupported adversarial training attack: {config.attack}")

    @staticmethod
    def _build_tabular_generator(config, metadata: TabularAdversarialMetadata):
        if config.attack == "fgsm":
            return TabularFGSMGenerator(config, metadata)
        if config.attack == "pgd":
            return TabularPGDGenerator(config, metadata)
        raise ValueError(f"Unsupported adversarial training attack: {config.attack}")

    @staticmethod
    def _get_tabular_metadata(partition) -> TabularAdversarialMetadata:
        train_set = getattr(partition, "train_set", None) if partition is not None else None
        metadata = getattr(train_set, "tabular_metadata", None)
        if metadata is None:
            raise ValueError("Tabular adversarial training requires tabular_metadata")
        if isinstance(metadata, TabularAdversarialMetadata):
            tabular_metadata = metadata
        else:
            tabular_metadata = TabularAdversarialMetadata.from_dict(metadata)

        integer_features = [
            name
            for name, feature_type in zip(tabular_metadata.feature_names, tabular_metadata.feature_types)
            if feature_type == INTEGER
        ]
        continuous_features = [
            name
            for name, feature_type in zip(tabular_metadata.feature_names, tabular_metadata.feature_types)
            if feature_type == CONTINUOUS
        ]
        non_perturbable_features = [
            name
            for name, feature_type in zip(tabular_metadata.feature_names, tabular_metadata.feature_types)
            if feature_type not in {CONTINUOUS, INTEGER}
        ]
        logging.info(
            "[AdversarialTrainingDefense] Tabular feature mask loaded | integer=%s | continuous=%s | "
            "non_perturbable=%s | integer_features=%s | continuous_features=%s | non_perturbable_preview=%s",
            len(integer_features),
            len(continuous_features),
            len(non_perturbable_features),
            integer_features,
            continuous_features,
            non_perturbable_features[:20],
        )
        return tabular_metadata

    def should_apply(self, x: torch.Tensor) -> bool:
        if self.config.apply_probability >= 1.0:
            return True
        if self.config.apply_probability <= 0.0:
            return False
        return bool(torch.rand((), device=x.device).item() < self.config.apply_probability)

    def compute_training_step(self, model, x, y, criterion):
        if not self.should_apply(x):
            logits = model(x)
            loss = criterion(logits, y)
            return loss, logits, {}

        if self.config.mode == "clean":
            logits = model(x)
            loss = criterion(logits, y)
            return loss, logits, {}

        x_adv = self.generator.generate(model, x, y, criterion)
        self._log_adversarial_samples(model, x, x_adv, y)
        adv_logits = model(x_adv)
        adv_loss = criterion(adv_logits, y)

        if self.config.mode == "adversarial":
            return adv_loss, adv_logits, self._extra_metrics({
                "Adversarial Loss": adv_loss,
                "Adversarial Accuracy": self._accuracy(adv_logits, y),
            })

        clean_logits = model(x)
        clean_loss = criterion(clean_logits, y)
        total_weight = self.config.clean_weight + self.config.adversarial_weight
        loss = (
            self.config.clean_weight * clean_loss + self.config.adversarial_weight * adv_loss
        ) / total_weight

        return loss, clean_logits, self._extra_metrics({
            "Clean Loss": clean_loss,
            "Adversarial Loss": adv_loss,
            "Adversarial Accuracy": self._accuracy(adv_logits, y),
        })

    def _accuracy(self, logits, y):
        predictions = torch.argmax(logits, dim=1)
        return torch.mean((predictions == y).float())

    def _extra_metrics(self, metrics):
        if not self.config.log_adversarial_metrics:
            return {}
        return metrics

    def _log_adversarial_samples(self, model, x_clean: torch.Tensor, x_adv: torch.Tensor, y: torch.Tensor) -> None:
        if not self.config.log_adversarial_metrics:
            return

        current_round = int(getattr(model, "round", 0))
        already_logged = self._logged_adversarial_samples_by_round.get(current_round, 0)
        remaining = self.LOGGED_SAMPLES_PER_ROUND - already_logged
        if remaining <= 0:
            return

        with torch.no_grad():
            clean_view = x_clean.detach()
            adv_view = x_adv.detach()
            if hasattr(self.generator, "denormalize"):
                clean_view = self.generator.denormalize(clean_view)
                adv_view = self.generator.denormalize(adv_view)

            delta = adv_view - clean_view
            samples_to_log = min(remaining, int(clean_view.size(0)))

            for sample_idx in range(samples_to_log):
                sample_clean = clean_view[sample_idx].detach().float().cpu()
                sample_adv = adv_view[sample_idx].detach().float().cpu()
                sample_delta = delta[sample_idx].detach().float().cpu()

                logging_training.info(
                    "[AdversarialTrainingDefense] Round %s | Sample %s/%s before/after distortion | "
                    "dataset=%s | attack=%s | epsilon_effective=%.6f | label=%s | "
                    "clean[min=%.6f max=%.6f mean=%.6f] | "
                    "adv[min=%.6f max=%.6f mean=%.6f] | delta_linf=%.6f | delta_l2=%.6f",
                    current_round,
                    already_logged + sample_idx + 1,
                    self.LOGGED_SAMPLES_PER_ROUND,
                    self.config.dataset_name,
                    self.config.attack,
                    float(getattr(self.generator, "last_epsilon", self.config.epsilon) or 0.0),
                    int(y[sample_idx].detach().cpu().item()) if y.numel() > sample_idx else None,
                    sample_clean.min().item(),
                    sample_clean.max().item(),
                    sample_clean.mean().item(),
                    sample_adv.min().item(),
                    sample_adv.max().item(),
                    sample_adv.mean().item(),
                    sample_delta.abs().max().item(),
                    sample_delta.reshape(-1).norm(p=2).item(),
                )
                logging_training.info(
                    "[AdversarialTrainingDefense] Round %s | Clean sample %s channel0 4x4:\n%s",
                    current_round,
                    already_logged + sample_idx + 1,
                    self._format_patch(sample_clean),
                )
                logging_training.info(
                    "[AdversarialTrainingDefense] Round %s | Adversarial sample %s channel0 4x4:\n%s",
                    current_round,
                    already_logged + sample_idx + 1,
                    self._format_patch(sample_adv),
                )
                logging_training.info(
                    "[AdversarialTrainingDefense] Round %s | Delta sample %s channel0 4x4:\n%s",
                    current_round,
                    already_logged + sample_idx + 1,
                    self._format_patch(sample_delta),
                )

            self._logged_adversarial_samples_by_round[current_round] = already_logged + samples_to_log

    @staticmethod
    def _format_patch(sample: torch.Tensor, patch_size: int = 4) -> str:
        if sample.dim() >= 3:
            patch = sample[0, :patch_size, :patch_size]
        elif sample.dim() == 2:
            patch = sample[:patch_size, :patch_size]
        else:
            patch = sample[:patch_size]
        values = patch.tolist()
        if sample.dim() < 2:
            return str([round(float(value), 6) for value in values])
        return str([[round(float(value), 6) for value in row] for row in values])


def apply_adversarial_training_if_enabled(model, participant_config: dict[str, Any], partition=None) -> None:
    defense = AdversarialTrainingDefense.from_participant_config(participant_config, partition=partition)
    if defense is not None:
        model.set_adversarial_training(defense)
        logging.info(
            "[AdversarialTrainingDefense] Enabled | dataset=%s | attack=%s | epsilon_max=%s | "
            "epsilon_range=[%.6f, %.6f] | epsilon_step=%.6f | mode=%s",
            defense.config.dataset_name,
            defense.config.attack,
            defense.config.epsilon,
            defense.config.epsilon / 4.0,
            defense.config.epsilon,
            defense.config.epsilon / 8.0,
            defense.config.mode,
        )
