import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

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

    @abstractmethod
    def generate(self, model, x, y, criterion):
        raise NotImplementedError


class ImageAdversarialExampleGenerator(AdversarialExampleGenerator):
    def __init__(self, config: AdversarialTrainingConfig, mean: tuple[float, ...], std: tuple[float, ...]):
        self.config = config
        self.mean = mean
        self.std = std

    def _channel_tensor(self, values: tuple[float, ...], x: torch.Tensor) -> torch.Tensor:
        shape = [1, len(values)] + [1] * max(x.dim() - 2, 0)
        return torch.tensor(values, dtype=x.dtype, device=x.device).view(*shape)

    def _epsilon(self, x: torch.Tensor) -> torch.Tensor:
        std = self._channel_tensor(self.std, x)
        return float(self.config.epsilon) / std

    def _alpha(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.config.alpha
        if alpha is None:
            alpha = self.config.epsilon / max(int(self.config.steps), 1)
        std = self._channel_tensor(self.std, x)
        return float(alpha) / std

    def _bounds(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self._channel_tensor(self.mean, x)
        std = self._channel_tensor(self.std, x)
        lower = (float(self.config.clip_min) - mean) / std
        upper = (float(self.config.clip_max) - mean) / std
        return lower, upper

    def _project(self, x_adv: torch.Tensor, x_clean: torch.Tensor) -> torch.Tensor:
        epsilon = self._epsilon(x_clean)
        lower, upper = self._bounds(x_clean)
        x_adv = torch.max(torch.min(x_adv, x_clean + epsilon), x_clean - epsilon)
        return torch.max(torch.min(x_adv, upper), lower)


class ImageFGSMGenerator(ImageAdversarialExampleGenerator):
    def generate(self, model, x, y, criterion):
        x_adv = x.detach().clone().requires_grad_(True)
        logits = model(x_adv)
        loss = criterion(logits, y)
        grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
        x_adv = x_adv + self._epsilon(x_adv) * grad.sign()
        return self._project(x_adv.detach(), x.detach())


class ImagePGDGenerator(ImageAdversarialExampleGenerator):
    def generate(self, model, x, y, criterion):
        x_clean = x.detach()
        x_adv = x_clean.clone()
        steps = max(int(self.config.steps), 1)

        for _ in range(steps):
            x_adv = x_adv.detach().requires_grad_(True)
            logits = model(x_adv)
            loss = criterion(logits, y)
            grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
            x_adv = x_adv + self._alpha(x_adv) * grad.sign()
            x_adv = self._project(x_adv.detach(), x_clean)

        return x_adv.detach()


class AdversarialTrainingDefense:
    """Batch-level adversarial training defense for Nebula models."""

    def __init__(self, config: AdversarialTrainingConfig, generator: AdversarialExampleGenerator):
        self.config = config
        self.generator = generator

    @classmethod
    def from_participant_config(cls, participant_config: dict[str, Any]) -> "AdversarialTrainingDefense | None":
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


def apply_adversarial_training_if_enabled(model, participant_config: dict[str, Any]) -> None:
    defense = AdversarialTrainingDefense.from_participant_config(participant_config)
    if defense is not None:
        model.set_adversarial_training(defense)
        logging.info(
            "[AdversarialTrainingDefense] Enabled | dataset=%s | attack=%s | epsilon=%s | mode=%s",
            defense.config.dataset_name,
            defense.config.attack,
            defense.config.epsilon,
            defense.config.mode,
        )
