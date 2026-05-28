import torch
import torch.nn.functional as F

from nebula.addons.defenses.adversarial_training_base import AdversarialExampleGenerator
from nebula.addons.defenses.adversarial_training_config import AdversarialTrainingConfig
from nebula.core.datasets.tabular_metadata import CATEGORICAL, CONTINUOUS, INTEGER, TabularAdversarialMetadata


class TabularConstraintSet:
    """Projection and mutation rules derived from tabular metadata."""

    def __init__(self, metadata: TabularAdversarialMetadata):
        # Store metadata and cache derived tensors by device/dtype for speed.
        self.metadata = metadata
        self._tensor_cache: dict[tuple[torch.device, torch.dtype], dict[str, torch.Tensor]] = {}

    def tensors(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Return reusable masks, bounds and integer steps for a batch tensor.
        key = (x.device, x.dtype)
        cached = self._tensor_cache.get(key)
        if cached is not None:
            return cached

        # Convert metadata lists to tensors once per device/dtype; CAA uses them in every step.
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
            "categorical": torch.tensor(
                [feature_type == CATEGORICAL for feature_type in self.metadata.feature_types],
                dtype=torch.bool,
                device=x.device,
            ).view(1, -1),
            "min": torch.tensor(self.metadata.feature_min_norm, dtype=x.dtype, device=x.device).view(1, -1),
            "max": torch.tensor(self.metadata.feature_max_norm, dtype=x.dtype, device=x.device).view(1, -1),
        }
        cached["numeric"] = cached["continuous"] | cached["integer"]
        cached["perturbable"] = cached["numeric"] | cached["categorical"]
        cached["integer_step"] = self._integer_steps(cached["min"])
        self._tensor_cache[key] = cached
        return cached

    def perturbable_mask(self, x: torch.Tensor) -> torch.Tensor:
        # Expose the final boolean mask used to block immutable features.
        return self.tensors(x)["perturbable"]

    def project(self, x_adv: torch.Tensor, x_clean: torch.Tensor, epsilon: float) -> torch.Tensor:
        # Project a candidate back to valid tabular values around the clean sample.
        tensors = self.tensors(x_clean)
        # Numeric features are bounded by epsilon; categorical one-hot features use dataset bounds.
        numeric_lower = torch.maximum(tensors["min"], x_clean - float(epsilon))
        numeric_upper = torch.minimum(tensors["max"], x_clean + float(epsilon))
        lower = torch.where(tensors["categorical"], tensors["min"], numeric_lower)
        upper = torch.where(tensors["categorical"], tensors["max"], numeric_upper)
        x_adv = torch.max(torch.min(x_adv, upper), lower)

        x_adv = self._project_integer_features(x_adv, x_clean, lower, upper, tensors)
        x_adv = self.project_categorical_groups(x_adv)
        return torch.where(tensors["perturbable"], x_adv, x_clean)

    def project_categorical_groups(self, x_adv: torch.Tensor) -> torch.Tensor:
        # Enforce one-hot validity after gradient or evolutionary changes.
        if not self.metadata.categorical_groups:
            return x_adv

        # Each one-hot group must end with exactly one active category.
        x_projected = x_adv.clone()
        for group in self.metadata.categorical_groups:
            group_tensor = torch.tensor(group, dtype=torch.long, device=x_adv.device)
            group_values = x_adv.index_select(1, group_tensor)
            selected = group_values.argmax(dim=1)
            one_hot = F.one_hot(selected, num_classes=len(group)).to(dtype=x_adv.dtype)
            x_projected[:, group_tensor] = one_hot
        return x_projected

    def categorical_gradient_step(self, x_candidate: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        # Apply a discrete gradient step to categorical one-hot groups.
        if not self.metadata.categorical_groups:
            return x_candidate

        # For one-hot features, choose the category with the largest adversarial gradient.
        x_stepped = x_candidate.clone()
        for group in self.metadata.categorical_groups:
            group_tensor = torch.tensor(group, dtype=torch.long, device=x_candidate.device)
            selected = grad.index_select(1, group_tensor).argmax(dim=1)
            one_hot = F.one_hot(selected, num_classes=len(group)).to(dtype=x_candidate.dtype)
            x_stepped[:, group_tensor] = one_hot
        return x_stepped

    def randomize_categorical_groups(
        self,
        candidates: torch.Tensor,
        mutation_probability: float,
    ) -> torch.Tensor:
        # Randomly switch categories for evolutionary exploration.
        if not self.metadata.categorical_groups:
            return candidates

        original_shape = candidates.shape
        flat_candidates = candidates.reshape(-1, original_shape[-1]).clone()
        for group in self.metadata.categorical_groups:
            # Mutation explores alternative categories when the gradient phase is not enough.
            group_tensor = torch.tensor(group, dtype=torch.long, device=candidates.device)
            current = flat_candidates.index_select(1, group_tensor).argmax(dim=1)
            random_choice = torch.randint(len(group), current.shape, device=candidates.device)
            mutate = torch.rand(current.shape, device=candidates.device) < float(mutation_probability)
            selected = torch.where(mutate, random_choice, current)
            one_hot = F.one_hot(selected, num_classes=len(group)).to(dtype=candidates.dtype)
            flat_candidates[:, group_tensor] = one_hot
        return flat_candidates.reshape(original_shape)

    def _integer_steps(self, minimum: torch.Tensor) -> torch.Tensor:
        # Build the normalized integer grid spacing tensor from metadata.
        integer_steps = torch.ones_like(minimum)
        for idx, step in (self.metadata.integer_step_norm or {}).items():
            integer_steps[0, int(idx)] = float(step)
        return integer_steps

    def _project_integer_features(
        self,
        x_adv: torch.Tensor,
        x_clean: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        tensors: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # Round integer columns while keeping them inside the allowed epsilon interval.
        integer_mask = tensors["integer"]
        if not integer_mask.any():
            return x_adv

        # Integer features live on a normalized grid, so round to the closest valid grid value.
        step = torch.clamp(tensors["integer_step"], min=torch.finfo(x_adv.dtype).eps)
        projected_integer = torch.round((x_adv - tensors["min"]) / step) * step + tensors["min"]
        grid_lower = torch.ceil((lower - tensors["min"]) / step) * step + tensors["min"]
        grid_upper = torch.floor((upper - tensors["min"]) / step) * step + tensors["min"]
        projected_integer = torch.max(torch.min(projected_integer, grid_upper), grid_lower)
        has_valid_grid = grid_lower <= grid_upper
        projected_integer = torch.where(has_valid_grid, projected_integer, x_clean)
        return torch.where(integer_mask, projected_integer, x_adv)


class TabularAdversarialExampleGenerator(AdversarialExampleGenerator):
    """Base generator for constrained tabular adversarial examples."""

    def __init__(self, config: AdversarialTrainingConfig, metadata: TabularAdversarialMetadata):
        # Share config, metadata and constraints across CAA phases.
        self.config = config
        self.metadata = metadata
        self.constraints = TabularConstraintSet(metadata)

    def _alpha(self, epsilon: float) -> float:
        # Use an explicit alpha when provided; otherwise distribute epsilon across steps.
        if self.config.alpha is not None:
            return float(self.config.alpha)
        return float(epsilon) / max(int(self.config.steps), 1)

    def _margin(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Score how close each sample is to being misclassified.
        # Positive margin means some wrong class beats the true class.
        true_logits = logits.gather(1, y.view(-1, 1)).squeeze(1)
        other_logits = logits.masked_fill(F.one_hot(y, num_classes=logits.size(1)).bool(), float("-inf"))
        return other_logits.max(dim=1).values - true_logits

    def _success_mask(self, model, x_adv: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Mark samples whose adversarial version changes the model prediction.
        with torch.no_grad():
            return torch.argmax(model(x_adv), dim=1) != y

    def _better_mask(
        self,
        candidate_success: torch.Tensor,
        candidate_score: torch.Tensor,
        best_success: torch.Tensor,
        best_score: torch.Tensor,
    ) -> torch.Tensor:
        # Prefer successful attacks, then candidates with a better adversarial margin.
        return (candidate_success & ~best_success) | (
            (candidate_success == best_success) & (candidate_score > best_score)
        )


class TabularCAAGenerator(TabularAdversarialExampleGenerator):
    """CAA-style generator for constrained tabular adversarial training."""

    def generate(self, model, x, y, criterion):
        # Generate a constrained tabular adversarial batch with CAA.
        epsilon = self._sample_epsilon(x.device)
        x_clean = x.detach()
        if epsilon <= 0.0:
            return x_clean

        # First try a gradient-guided CAA search; then mutate only samples that still resist.
        x_adv = self._capgd_phase(model, x_clean, y, criterion, epsilon)
        failed = ~self._success_mask(model, x_adv, y)
        if failed.any():
            x_fallback = self._evolutionary_phase(model, x_clean[failed], y[failed], x_adv[failed], epsilon)
            x_adv = x_adv.clone()
            x_adv[failed] = x_fallback
        return x_adv.detach()

    def _capgd_phase(self, model, x_clean: torch.Tensor, y: torch.Tensor, criterion, epsilon: float) -> torch.Tensor:
        # Run the gradient-based part of CAA with projection after every candidate step.
        steps = max(int(self.config.steps), 1)
        step_size = self._alpha(epsilon)
        perturbable_mask = self.constraints.perturbable_mask(x_clean)
        x_adv = x_clean.clone()
        best_adv = x_adv.clone()
        best_score = torch.full((x_clean.size(0),), float("-inf"), dtype=x_clean.dtype, device=x_clean.device)
        best_success = torch.zeros(x_clean.size(0), dtype=torch.bool, device=x_clean.device)
        previous_loss = None

        for _ in range(steps):
            x_grad = x_adv.detach().requires_grad_(True)
            logits = model(x_grad)
            loss = criterion(logits, y)
            grad = torch.autograd.grad(loss, x_grad, only_inputs=True)[0]

            candidate = x_adv.detach() + float(step_size) * grad.sign() * perturbable_mask
            candidate = self.constraints.categorical_gradient_step(candidate, grad)
            candidate = self.constraints.project(candidate, x_clean, epsilon)

            with torch.no_grad():
                candidate_logits = model(candidate)
                candidate_score = self._margin(candidate_logits, y)
                candidate_success = torch.argmax(candidate_logits, dim=1) != y
                # Keep successful adversarial samples first; otherwise keep the highest margin.
                better = self._better_mask(candidate_success, candidate_score, best_success, best_score)
                best_adv = torch.where(better.view(-1, 1), candidate, best_adv)
                best_score = torch.where(better, candidate_score, best_score)
                best_success = best_success | candidate_success

                candidate_loss = F.cross_entropy(candidate_logits, y)
                if previous_loss is not None and candidate_loss <= previous_loss:
                    step_size *= 0.75
                previous_loss = candidate_loss

            x_adv = candidate

        return best_adv.detach()

    def _evolutionary_phase(
        self,
        model,
        x_clean: torch.Tensor,
        y: torch.Tensor,
        x_seed: torch.Tensor,
        epsilon: float,
    ) -> torch.Tensor:
        # Use random mutations as a fallback for samples not solved by the gradient phase.
        if x_clean.numel() == 0:
            return x_clean

        tensors = self.constraints.tensors(x_clean)
        perturbable_mask = tensors["perturbable"].to(dtype=x_clean.dtype)
        batch_size = x_clean.size(0)
        population_size = min(max(int(self.config.steps) * 4, 8), 32)
        generations = min(max(int(self.config.steps), 3), 20)
        mutation_scale = max(float(epsilon) / 2.0, torch.finfo(x_clean.dtype).eps)

        best_adv = self.constraints.project(x_seed.detach(), x_clean, epsilon)
        with torch.no_grad():
            best_logits = model(best_adv)
            best_score = self._margin(best_logits, y)
            best_success = torch.argmax(best_logits, dim=1) != y

        for _ in range(generations):
            random_noise = torch.empty(
                population_size,
                *x_clean.shape,
                dtype=x_clean.dtype,
                device=x_clean.device,
            ).uniform_(-float(epsilon), float(epsilon))
            mutations = torch.randn(
                population_size,
                *x_clean.shape,
                dtype=x_clean.dtype,
                device=x_clean.device,
            ) * mutation_scale
            candidates = x_clean.unsqueeze(0) + random_noise * perturbable_mask
            candidates[0] = best_adv + mutations[0] * perturbable_mask
            if population_size > 1:
                candidates[1:] = candidates[1:] + mutations[1:] * perturbable_mask
            candidates = self.constraints.randomize_categorical_groups(candidates, mutation_probability=0.35)

            flat_candidates = candidates.reshape(population_size * batch_size, -1)
            flat_clean = x_clean.repeat(population_size, 1)
            # Every random candidate is projected back to the valid tabular domain before scoring.
            flat_candidates = self.constraints.project(flat_candidates, flat_clean, epsilon)
            repeated_y = y.repeat(population_size)

            with torch.no_grad():
                logits = model(flat_candidates)
                scores = self._margin(logits, repeated_y).view(population_size, batch_size)
                successes = (torch.argmax(logits, dim=1) != repeated_y).view(population_size, batch_size)
                candidate_rank = scores + successes.to(dtype=scores.dtype) * 1_000.0
                best_population_idx = candidate_rank.argmax(dim=0)

                selected = flat_candidates.view(population_size, batch_size, -1)[
                    best_population_idx,
                    torch.arange(batch_size, device=x_clean.device),
                ]
                selected_score = scores[
                    best_population_idx,
                    torch.arange(batch_size, device=x_clean.device),
                ]
                selected_success = successes[
                    best_population_idx,
                    torch.arange(batch_size, device=x_clean.device),
                ]
                better = self._better_mask(selected_success, selected_score, best_success, best_score)
                best_adv = torch.where(better.view(-1, 1), selected, best_adv)
                best_score = torch.where(better, selected_score, best_score)
                best_success = best_success | selected_success

        return best_adv.detach()
