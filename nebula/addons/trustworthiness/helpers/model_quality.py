import logging
import math

import numpy as np
import torch
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)

def _get_model_accuracy(model, dataloader):
    """
    Calculates model accuracy over a dataloader.

    Args:
        model (torch.nn.Module): Model to evaluate.
        dataloader (DataLoader): Dataloader with (x, y) batches.

    Returns:
        float: Accuracy in [0, 1].
    """
    if not isinstance(model, torch.nn.Module):
        logger.warning("Model is not a torch.nn.Module")
        return 0.0

    try:
        device = next(model.parameters()).device
    except Exception:
        device = torch.device("cpu")

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            out = model(x)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            preds = logits.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total if total > 0 else 0.0


def get_macro_f1_score(model, dataloader):
    """
    Calculates macro F1 score over a dataloader.

    Args:
        model (torch.nn.Module): Model to evaluate.
        dataloader (DataLoader): Dataloader with (x, y) batches.

    Returns:
        float: Macro F1 score in [0, 1].
    """
    if not isinstance(model, torch.nn.Module):
        logger.warning("Model is not a torch.nn.Module")
        return 0.0

    try:
        device = next(model.parameters()).device
    except Exception:
        device = torch.device("cpu")

    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            out = model(x)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            preds = logits.argmax(dim=1)

            y_true.extend(y.detach().cpu().numpy().tolist())
            y_pred.extend(preds.detach().cpu().numpy().tolist())

    if not y_true:
        return 0.0

    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _extract_model_logits(model_output):
    """
    Normalize the output returned by a model forward pass into a logits tensor.

    Some models may return tuples/lists; for trust metrics we always consume the
    first element as the classification output.
    """
    return model_output[0] if isinstance(model_output, (tuple, list)) else model_output


def _prepare_class_targets(y):
    """
    Convert different target representations into a flat class-index tensor.
    """
    if not torch.is_tensor(y):
        y = torch.as_tensor(y)

    if y.ndim > 1:
        if y.size(-1) > 1:
            y = y.argmax(dim=-1)
        else:
            y = y.view(-1)

    return y.long().view(-1)


def _logits_to_probabilities(logits):
    """
    Convert model outputs into a probability matrix of shape (N, C).

    Supports:
    - multiclass logits/log-probabilities with shape (N, C)
    - binary logits with shape (N,) or (N, 1)
    - already-normalized probability matrices
    """
    if not torch.is_tensor(logits):
        logits = torch.as_tensor(logits)

    if logits.ndim == 0:
        logits = logits.view(1, 1)
    elif logits.ndim == 1:
        logits = logits.view(-1, 1)
    elif logits.ndim > 2:
        logits = logits.reshape(logits.shape[0], -1)

    if logits.size(1) == 1:
        pos_prob = torch.sigmoid(logits[:, 0])
        probs = torch.stack([1.0 - pos_prob, pos_prob], dim=1)
    else:
        row_sums = logits.sum(dim=1)
        looks_like_probs = (
            torch.all(logits >= 0)
            and torch.all(logits <= 1.0 + 1e-6)
            and torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4, rtol=1e-4)
        )
        probs = logits if looks_like_probs else torch.softmax(logits, dim=1)

    probs = torch.clamp(probs, min=0.0, max=1.0)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return probs


def _collect_classification_statistics(model, dataloader):
    """
    Collect prediction statistics required by calibration and inequality metrics.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        predicted labels, true labels, prediction confidences, correctness flags,
        and probability assigned to the true class.
    """
    if not isinstance(model, torch.nn.Module):
        logger.warning("Model is not a torch.nn.Module")
        empty = np.array([], dtype=float)
        return empty, empty, empty, empty, empty

    try:
        device = next(model.parameters()).device
    except Exception:
        device = torch.device("cpu")

    preds_all = []
    targets_all = []
    confidences_all = []
    correct_all = []
    true_probs_all = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                continue

            x, y = batch[0], batch[1]
            if not (torch.is_tensor(x) and torch.is_tensor(y)):
                continue

            x = x.to(device)
            y = _prepare_class_targets(y).to(device)

            out = model(x)
            logits = _extract_model_logits(out)
            probs = _logits_to_probabilities(logits)

            if probs.ndim != 2 or probs.size(0) == 0:
                continue

            if y.numel() != probs.size(0):
                n = min(int(y.numel()), int(probs.size(0)))
                if n == 0:
                    continue
                y = y[:n]
                probs = probs[:n]

            valid_mask = (y >= 0) & (y < probs.size(1))
            if not torch.any(valid_mask):
                continue

            y = y[valid_mask]
            probs = probs[valid_mask]

            conf, preds = probs.max(dim=1)
            true_probs = probs.gather(1, y.view(-1, 1)).squeeze(1)
            correct = preds.eq(y).float()

            preds_all.extend(preds.detach().cpu().numpy().tolist())
            targets_all.extend(y.detach().cpu().numpy().tolist())
            confidences_all.extend(conf.detach().cpu().numpy().tolist())
            correct_all.extend(correct.detach().cpu().numpy().tolist())
            true_probs_all.extend(true_probs.detach().cpu().numpy().tolist())

    return (
        np.asarray(preds_all, dtype=int),
        np.asarray(targets_all, dtype=int),
        np.asarray(confidences_all, dtype=float),
        np.asarray(correct_all, dtype=float),
        np.asarray(true_probs_all, dtype=float),
    )


def get_overfitting_score(model, train_dataloader, test_accuracy):
    """
    Calculates overfitting as the positive train-test accuracy gap.

    Args:
        model (torch.nn.Module): Model to evaluate on training data.
        train_dataloader (DataLoader): Training dataloader.
        test_accuracy (float): Test accuracy in [0, 1].

    Returns:
        float: Positive train-test accuracy gap.
    """
    try:
        train_accuracy = _get_model_accuracy(model, train_dataloader)
        return max(0.0, float(train_accuracy) - float(test_accuracy))
    except Exception as exc:
        logger.warning("Could not compute overfitting score")
        logger.warning(exc)
        return 0.0


def get_well_calibration_error(model, test_dataloader, n_bins=10):
    """
    Calculates a well-calibration error style metric using prediction confidence.

    For multiclass models, confidence is taken as the max softmax probability and
    the observed outcome is whether the prediction is correct.

    Args:
        model (torch.nn.Module): Model to evaluate.
        test_dataloader (DataLoader): Test dataloader.
        n_bins (int): Number of quantile bins.

    Returns:
        float: Calibration error in [0, 1] when computation succeeds.
    """
    if not isinstance(model, torch.nn.Module):
        logger.warning("Model is not a torch.nn.Module")
        return 0.0

    try:
        n_bins = max(2, int(n_bins))
    except Exception:
        n_bins = 10

    _, _, confidences, correct, _ = _collect_classification_statistics(model, test_dataloader)

    if len(confidences) == 0 or len(correct) == 0:
        return 0.0

    confidences = np.clip(np.asarray(confidences, dtype=float), 0.0, 1.0)
    correct = np.clip(np.asarray(correct, dtype=float), 0.0, 1.0)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = float(len(confidences))

    for idx in range(n_bins):
        left = bin_edges[idx]
        right = bin_edges[idx + 1]
        if idx == n_bins - 1:
            mask = (confidences >= left) & (confidences <= right)
        else:
            mask = (confidences >= left) & (confidences < right)

        if not np.any(mask):
            continue

        bin_weight = float(mask.sum()) / total
        bin_accuracy = float(correct[mask].mean())
        bin_confidence = float(confidences[mask].mean())
        ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return float(np.clip(ece, 0.0, 1.0))


def get_generalized_entropy_index(model, test_dataloader, alpha=2):
    """
    Calculates generalized entropy index from model predictions.

    Args:
        model (torch.nn.Module): Model to evaluate.
        test_dataloader (DataLoader): Test dataloader.
        alpha (float): GEI alpha parameter.

    Returns:
        float: Generalized entropy index value.
    """
    try:
        _, _, _, _, true_class_probs = _collect_classification_statistics(model, test_dataloader)
        if len(true_class_probs) == 0:
            return 0.0

        # Use the probability assigned to the true class as a continuous, positive
        # benefit. This works consistently for multiclass neural models on both
        # images and tabular data, and avoids collapsing the metric to a coarse
        # correct/incorrect indicator.
        eps = 1e-12
        b = np.clip(np.asarray(true_class_probs, dtype=float), eps, 1.0)
        mu = float(np.mean(b))
        if mu <= 0:
            return 0.0

        ratio = np.clip(b / mu, eps, None)

        if alpha == 0:
            val = float(np.mean(-np.log(ratio)))
        elif alpha == 1:
            val = float(np.mean(ratio * np.log(ratio)))
        elif alpha == 2:
            val = float(np.mean((ratio - 1.0) ** 2) / 2.0)
        else:
            val = float(np.mean(ratio**alpha - 1.0) / (alpha * (alpha - 1.0)))

        if math.isnan(val) or math.isinf(val):
            return 0.0
        return max(0.0, val)
    except Exception as exc:
        logger.warning("Could not compute generalized entropy index")
        logger.warning(exc)
        return 0.0


def get_theil_index(model, test_dataloader):
    """
    Convenience wrapper for generalized entropy index with alpha=1.
    """
    return get_generalized_entropy_index(model, test_dataloader, alpha=1)


def get_coefficient_of_variation(model, test_dataloader):
    """
    Calculates coefficient of variation from GEI(alpha=2).
    """
    try:
        gei = get_generalized_entropy_index(model, test_dataloader, alpha=2)
        return float(np.sqrt(2 * gei))
    except Exception as exc:
        logger.warning("Could not compute coefficient of variation")
        logger.warning(exc)
        return 0.0
