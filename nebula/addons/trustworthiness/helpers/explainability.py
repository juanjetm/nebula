import copy
import gc
import logging
import math

import numpy as np
import shap
import torch
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, variation

logger = logging.getLogger(__name__)

def get_feature_importance_cv(model, test_sample):
    """
    Calculates the coefficient of variation of the feature importance.

    Args:
        model (object): The model.
        test_sample (object): One test sample to calculate the feature importance.

    Returns:
        float: The coefficient of variation of the feature importance.
    """

    try:
        vals = np.asarray(_get_feature_importances(model, test_sample), dtype=float).reshape(-1)
        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        vals = vals[vals > 0]

        if len(vals) <= 1:
            return 0.0

        cv = float(variation(vals))
        if math.isnan(cv) or math.isinf(cv):
            return 1.0
        return max(0.0, cv)
    except Exception as exc:
        logger.warning("Could not compute feature importance CV with shap")
        logger.warning(exc)
        return 1.0


def _get_feature_importances(model, test_sample):
    """
    Computes global feature importances from SHAP values.

    Args:
        model (object): The model.
        test_sample (object): One test sample batch.

    Returns:
        np.ndarray: Global importances per feature.
    """
    if not isinstance(model, torch.nn.Module):
        logger.warning("Model is not a torch.nn.Module")
        return np.array([])

    def _clone_model(model_ref, device):
        optimizer_attrs = ("_optimizer", "_optimizer_override")
        optimizer_state = {}
        try:
            for attr in optimizer_attrs:
                if hasattr(model_ref, attr):
                    optimizer_state[attr] = getattr(model_ref, attr)
                    setattr(model_ref, attr, None)

            model_clone = copy.deepcopy(model_ref)
            for attr in optimizer_attrs:
                if hasattr(model_clone, attr):
                    setattr(model_clone, attr, None)

            model_clone.to(device)
            model_clone.eval()
            return model_clone
        except Exception as exc:
            logger.warning("Could not clone model for SHAP, using original model")
            logger.warning(exc)
            model_ref.eval()
            return model_ref
        finally:
            for attr, value in optimizer_state.items():
                setattr(model_ref, attr, value)

    def _prepare_shap_inputs(sample):
        if not (isinstance(sample, (tuple, list)) and len(sample) >= 1):
            return None, None, None

        batched_data = sample[0]
        if not torch.is_tensor(batched_data) or batched_data.ndim == 0 or batched_data.size(0) == 0:
            return None, None, None

        if not torch.is_floating_point(batched_data):
            batched_data = batched_data.float()

        batch_size = int(batched_data.size(0))
        input_shape = tuple(int(dim) for dim in batched_data.shape[1:])

        if batch_size == 1:
            return batched_data[:1], batched_data[:1], input_shape

        background_size = min(max(8, batch_size // 4), 32, batch_size - 1)
        explainable = batch_size - background_size
        explain_size = min(max(4, explainable), 32, explainable)

        background = batched_data[:background_size]
        test_data = batched_data[background_size:background_size + explain_size]

        if test_data.size(0) == 0:
            test_data = batched_data[: min(batch_size, 32)]

        return background, test_data, input_shape

    def _compute_shap_values(model_ref, background, test_data):
        explainer_errors = []

        for explainer_name in ("DeepExplainer", "GradientExplainer"):
            explainer = None
            try:
                if explainer_name == "DeepExplainer":
                    explainer = shap.DeepExplainer(model_ref, background)
                    return explainer.shap_values(test_data, check_additivity=False)

                explainer = shap.GradientExplainer(model_ref, background)
                return explainer.shap_values(test_data)
            except Exception as exc:
                explainer_errors.append(f"{explainer_name}: {exc}")
            finally:
                # SHAP explainers may register autograd hooks. If we explain on the
                # original model, those hooks can leak into later ART metrics.
                del explainer
                gc.collect()

        raise RuntimeError("; ".join(explainer_errors))

    def _compute_gradient_importances(model_ref, test_data):
        was_training = bool(getattr(model_ref, "training", False))
        model_ref.eval()

        try:
            inputs = test_data.detach().clone().requires_grad_(True)
            model_ref.zero_grad(set_to_none=True)

            outputs = model_ref(inputs)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]

            if outputs.ndim == 1:
                score = outputs.sum()
            else:
                score = outputs.reshape(outputs.shape[0], -1).max(dim=1).values.sum()

            score.backward()
            if inputs.grad is None:
                return np.array([])

            importances = torch.abs(inputs.grad * inputs).mean(dim=0)
            importances = importances.detach().cpu().numpy().reshape(-1)
            importances = np.nan_to_num(importances, nan=0.0, posinf=0.0, neginf=0.0)
            return np.maximum(importances, 0.0)
        finally:
            if was_training:
                model_ref.train()

    def _feature_axes_from_shape(arr_shape, input_shape, n_samples):
        input_shape = tuple(input_shape)
        input_rank = len(input_shape)

        if input_rank == 0 or len(arr_shape) < input_rank:
            return None

        if len(arr_shape) >= input_rank + 1 and tuple(arr_shape[1:1 + input_rank]) == input_shape:
            return tuple(range(1, 1 + input_rank))

        if len(arr_shape) >= input_rank + 2 and arr_shape[1] == n_samples and tuple(arr_shape[2:2 + input_rank]) == input_shape:
            return tuple(range(2, 2 + input_rank))

        candidates = []
        for start in range(len(arr_shape) - input_rank + 1):
            if tuple(arr_shape[start:start + input_rank]) == input_shape:
                candidates.append(start)

        if not candidates:
            return None

        # Prefer matches that do not consume the leading sample/output axes.
        non_leading = [start for start in candidates if start > 0]
        if non_leading:
            candidates = non_leading

        if len(arr_shape) > 1 and arr_shape[1] == n_samples:
            non_output_sample = [start for start in candidates if start > 1]
            if non_output_sample:
                candidates = non_output_sample

        start = candidates[0]
        return tuple(range(start, start + input_rank))

    try:
        try:
            device = next(model.parameters()).device
        except Exception:
            device = torch.device("cpu")

        background, test_data, input_shape = _prepare_shap_inputs(test_sample)
        if background is None or test_data is None or input_shape is None:
            return np.array([])

        background = background.to(device)
        test_data = test_data.to(device)

        shap_model = _clone_model(model, device)
        try:
            shap_values = _compute_shap_values(shap_model, background, test_data)
        except Exception as exc:
            logger.debug("Could not compute feature importances with SHAP, using gradient fallback: %s", exc)
            shap_model = None
            gc.collect()

            gradient_model = _clone_model(model, device)
            try:
                return _compute_gradient_importances(gradient_model, test_data)
            except Exception as fallback_exc:
                logger.debug("Could not compute feature importances with gradient fallback: %s", fallback_exc)
                return np.array([])
            finally:
                del gradient_model
                gc.collect()
        finally:
            if shap_model is not None:
                del shap_model
            gc.collect()

        if shap_values is None:
            return np.array([])

        if isinstance(shap_values, (list, tuple)):
            arrays = [np.asarray(val, dtype=float) for val in shap_values if val is not None]
            if not arrays:
                return np.array([])
            shap_arr = np.stack(arrays, axis=0)
        else:
            shap_arr = np.asarray(shap_values, dtype=float)

        if shap_arr.size == 0:
            return np.array([])

        shap_arr = np.nan_to_num(shap_arr, nan=0.0, posinf=0.0, neginf=0.0)
        feature_axes = _feature_axes_from_shape(tuple(shap_arr.shape), input_shape, int(test_data.size(0)))

        if feature_axes is None:
            # Conservative fallback: treat the first axis as samples when possible and
            # flatten the remaining dimensions into features.
            if shap_arr.ndim == 1:
                importances = np.abs(shap_arr)
            else:
                aggregate_axes = (0,)
                importances = np.mean(np.abs(shap_arr), axis=aggregate_axes)
        else:
            aggregate_axes = tuple(idx for idx in range(shap_arr.ndim) if idx not in feature_axes)
            if aggregate_axes:
                importances = np.mean(np.abs(shap_arr), axis=aggregate_axes)
            else:
                importances = np.abs(shap_arr)

        importances = np.asarray(importances, dtype=float).reshape(-1)
        importances = np.nan_to_num(importances, nan=0.0, posinf=0.0, neginf=0.0)
        return np.maximum(importances, 0.0)
    except Exception as exc:
        logger.debug("Could not compute feature importances")
        logger.debug(exc)
        return np.array([])


def get_alpha_score(model, test_sample, alpha=0.8):
    """
    Computes alpha score from global feature importances.
    """
    try:
        vals = np.asarray(_get_feature_importances(model, test_sample), dtype=float).reshape(-1)
        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        vals = np.maximum(vals, 0.0)
        total_features = len(vals)
        if total_features == 0 or np.sum(vals) <= 1e-12:
            return 1.0

        try:
            alpha = float(alpha)
        except Exception:
            alpha = 0.8
        alpha = min(max(alpha, 0.0), 1.0)

        vals_sorted = np.sort(vals)[::-1]
        cum_sum = np.cumsum(vals_sorted)
        threshold = float(alpha) * np.sum(vals_sorted)
        idx = np.searchsorted(cum_sum, threshold)
        return float(min(total_features, idx + 1) / total_features)
    except Exception as exc:
        logger.warning("Could not compute alpha score")
        logger.warning(exc)
        return 1.0


def _get_spread_base(model, test_sample, divergence=True):
    vals = _get_feature_importances(model, test_sample)
    tol = 1e-8

    if len(vals) == 0 or np.sum(vals) < tol:
        return 0.0 if divergence else 1.0
    if len(vals) == 1:
        return 0.0 if divergence else 1.0

    weights = vals / np.sum(vals)
    equal_weights = np.ones(len(vals)) / len(vals)

    if divergence:
        metric = jensenshannon(weights, equal_weights, base=2)
    else:
        denom = entropy(equal_weights)
        metric = 0.0 if denom <= tol else entropy(weights) / denom

    if math.isnan(metric) or math.isinf(metric):
        return 0.0 if divergence else 1.0
    return float(np.clip(metric, 0.0, 1.0))


def get_spread_ratio(model, test_sample):
    """
    Computes spread ratio from global feature importances.
    """
    try:
        return _get_spread_base(model, test_sample, divergence=False)
    except Exception as exc:
        logger.warning("Could not compute spread ratio")
        logger.warning(exc)
        return 1.0


def get_spread_divergence(model, test_sample):
    """
    Computes spread divergence from global feature importances.
    """
    try:
        return _get_spread_base(model, test_sample, divergence=True)
    except Exception as exc:
        logger.warning("Could not compute spread divergence")
        logger.warning(exc)
        return 0.0


def get_explainability_metrics_summary(model, test_dataloader, max_batches=4):
    """
    Computes explainability metrics over multiple test batches and returns
    their mean values.

    Args:
        model (object): The model.
        test_dataloader: Test dataloader providing batches.
        max_batches (int): Maximum number of batches to use.

    Returns:
        dict: Mean values for feature_importance_cv, alpha_score,
        spread_ratio and spread_divergence.
    """
    summary = {
        "feature_importance_cv": 1.0,
        "alpha_score": 1.0,
        "spread_ratio": 1.0,
        "spread_divergence": 0.0,
    }

    if test_dataloader is None:
        return summary

    try:
        max_batches = max(1, int(max_batches))
    except Exception:
        max_batches = 4

    fi_values = []
    alpha_values = []
    spread_ratio_values = []
    spread_divergence_values = []

    try:
        for batch_idx, test_sample in enumerate(test_dataloader):
            if batch_idx >= max_batches:
                break

            fi_values.append(float(get_feature_importance_cv(model, test_sample)))
            alpha_values.append(float(get_alpha_score(model, test_sample)))
            spread_ratio_values.append(float(get_spread_ratio(model, test_sample)))
            spread_divergence_values.append(float(get_spread_divergence(model, test_sample)))
    except Exception as exc:
        logger.warning("Could not compute explainability metrics summary")
        logger.warning(exc)

    if fi_values:
        summary["feature_importance_cv"] = float(np.mean(fi_values))
    if alpha_values:
        summary["alpha_score"] = float(np.mean(alpha_values))
    if spread_ratio_values:
        summary["spread_ratio"] = float(np.mean(spread_ratio_values))
    if spread_divergence_values:
        summary["spread_divergence"] = float(np.mean(spread_divergence_values))

    return summary
