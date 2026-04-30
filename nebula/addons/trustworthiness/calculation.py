import logging
import math
import numbers
import os.path
import statistics
import copy
import gc
from datetime import datetime
from math import e
from os.path import exists
import json

import numpy as np
import pandas as pd
import shap
import torch
import torch.nn
from art.estimators.classification import PyTorchClassifier
from art.metrics import clever_u, loss_sensitivity, empirical_robustness
from codecarbon import EmissionsTracker
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, variation
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from torch import nn, optim
import torch.nn.functional as F
import io


from nebula.addons.trustworthiness.utils import read_csv

dirname = os.path.dirname(__file__)
logger = logging.getLogger(__name__)

R_L1 = 40
R_L2 = 2
R_LI = 0.1


# ---------------------------------------------------------------------------
# Generic score mapping helpers used by eval_metrics*.json
# ---------------------------------------------------------------------------

def get_mapped_score(score_key, score_map):
    """
    Finds the score by the score_key in the score_map.

    Args:
        score_key (string): The key to look up in the score_map.
        score_map (dict): The score map defined in the eval_metrics.json file.

    Returns:
        float: The normalized score of [0, 1].
    """
    score = 0
    if score_map is None:
        logger.warning("Score map is missing")
    else:
        keys = [key for key, value in score_map.items()]
        scores = [value for key, value in score_map.items()]
        normalized_scores = get_normalized_scores(scores)
        normalized_score_map = dict(zip(keys, normalized_scores, strict=False))
        score = normalized_score_map.get(score_key, np.nan)

    return score


def get_normalized_scores(scores):
    """
    Calculates the normalized scores of a list.

    Args:
        scores (list): The values that will be normalized.

    Returns:
        list: The normalized list.
    """
    if scores is None or len(scores) == 0:
        return []

    min_score = np.min(scores)
    max_score = np.max(scores)
    if max_score == min_score:
        return [1.0 for _ in scores]

    normalized = [(x - min_score) / (max_score - min_score) for x in scores]
    return normalized


def get_range_score(value, ranges, direction="asc"):
    """
    Maps the value to a range and gets the score by the range and direction.

    Args:
        value (int): The input score.
        ranges (list): The ranges defined.
        direction (string): Asc means the higher the range the higher the score, desc means otherwise.

    Returns:
        float: The normalized score of [0, 1].
    """

    if not (type(value) == int or type(value) == float):
        logger.warning("Input value is not a number")
        logger.warning(f"{value}")
        return 0
    else:
        score = 0
        if ranges is None:
            logger.warning("Score ranges are missing")
        else:
            total_bins = len(ranges) + 1
            bin = np.digitize(value, ranges, right=True)
            score = 1 - (bin / total_bins) if direction == "desc" else bin / total_bins
        return score


def get_map_value_score(score_key, score_map):
    """
    Finds the score by the score_key in the score_map and returns the value.

    Args:
        score_key (string): The key to look up in the score_map.
        score_map (dict): The score map defined in the eval_metrics.json file.

    Returns:
        float: The score obtained in the score_map.
    """
    score = 0
    if score_map is None:
        logger.warning("Score map is missing")
    else:
        score = score_map[score_key]
    return score


def get_true_score(value, direction):
    """
    Returns the negative of the value if direction is 'desc', otherwise returns value.

    Args:
        value (int): The input score.
        direction (string): Asc means the higher the range the higher the score, desc means otherwise.

    Returns:
        float: The score obtained.
    """

    if value is True:
        return 1
    elif value is False:
        return 0
    else:
        if not (type(value) == int or type(value) == float):
            logger.warning("Input value is not a number")
            logger.warning(f"{value}.")
            return 0
        else:
            if direction == "desc":
                return 1 - value
            else:
                return value


def get_scaled_score(value, scale: list, direction: str):
    """
    Maps a score of a specific scale into the scale between zero and one.

    Args:
        value (int or float): The raw value of the metric.
        scale (list): List containing the minimum and maximum value the value can fall in between.

    Returns:
        float: The normalized score of [0, 1].
    """

    score = 0
    try:
        value_min, value_max = scale[0], scale[1]
    except Exception:
        logger.warning("Score minimum or score maximum is missing. The minimum has been set to 0 and the maximum to 1")
        value_min, value_max = 0, 1
    if value is None or value == "":
        logger.warning("Score value is missing. Set value to zero")
    else:
        low, high = 0, 1
        if value >= value_max:
            score = 1
        elif value <= value_min:
            score = 0
        else:
            diff = value_max - value_min
            diffScale = high - low
            score = (float(value) - value_min) * (float(diffScale) / diff) + low
        if direction == "desc":
            score = high - score

    return score


def get_value(value):
    """
    Get the value of a metric.

    Args:
        value (float): The value of the metric.

    Returns:
        float: The value of the metric.
    """

    return value


def check_properties(*args):
    """
    Check if all the arguments have values.

    Args:
        args (list): All the arguments.

    Returns:
        float: The mean of arguments that have values.
    """

    result = map(lambda x: x is not None and x != "", args)
    return np.mean(list(result))


# ---------------------------------------------------------------------------
# Local/global data distribution and participation metrics
# ---------------------------------------------------------------------------

def get_class_imbalance_local(participant_id, experiment_name):
    data_class_count_file = os.path.join(os.environ.get('NEBULA_LOGS_DIR'), experiment_name, "trustworthiness", f"{str(participant_id)}_class_count.json")

    with open(data_class_count_file, "r") as file:
        class_distribution = json.load(file)

    class_samples_sizes = [x for x in class_distribution.values()]
    class_imbalance = get_cv(list=class_samples_sizes)

    return class_imbalance


def get_cv(list=None, std=None, mean=None):
    """
    Get the coefficient of variation.

    Args:
        list (list): List in which the coefficient of variation will be calculated.
        std (float): Standard deviation of a list.
        mean (float): Mean of a list.

    Returns:
        float: The coefficient of variation calculated.
    """
    if std is not None and mean is not None:
        if mean == 0:
            return 0
        return std / mean

    if list is not None:
        mean_value = np.mean(list)
        if mean_value == 0:
            return 0
        return np.std(list) / mean_value

    return 0


def get_participation_variation_score(participation_counts):
    """
    Convert participation-count dispersion into a trust-oriented score.

    Args:
        participation_counts (list[float | int]): Number of participations per client.

    Returns:
        float: Score in [0, 1] where 1 means equal participation.
    """
    if not participation_counts:
        return 1.0

    counts = np.asarray(participation_counts, dtype=float)
    mean_count = float(np.mean(counts))

    if mean_count <= 0:
        return 0.0

    cv = get_cv(list=counts)
    if not np.isfinite(cv):
        return 0.0

    return float(1 / (1 + cv))


# ---------------------------------------------------------------------------
# Privacy metrics
# ---------------------------------------------------------------------------


def get_global_privacy_risk(dp, epsilon, n):
    """
    Calculates the global privacy risk by epsilon and the number of clients.

    Args:
        dp (bool): Indicates if differential privacy is used or not.
        epsilon (int): The epsilon value.
        n (int): The number of clients in the scenario.

    Returns:
        float: The global privacy risk.
    """

    if dp is True and isinstance(epsilon, numbers.Number):
        return 1 / (1 + (n - 1) * math.pow(e, -epsilon))
    else:
        return 1

def get_global_privacy_risk_dfl(dp, epsilon, n):
    """
    Calculates the global privacy risk by epsilon and the number of clients.

    Args:
        dp (bool): Indicates if differential privacy is used or not.
        epsilon (int): The epsilon value.
        n (int): The number of neighbours.

    Returns:
        float: The global privacy risk.
    """

    if dp is True and isinstance(epsilon, numbers.Number):
        return 1 / (1 + (n + 1) * math.pow(e, -epsilon))
    else:
        return 1


def _collect_per_sample_losses(model, dataloader, max_samples=5000):
    """
    Compute per-sample cross-entropy losses for a dataloader.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader: DataLoader providing (samples, labels).
        max_samples (int): Maximum number of samples to process.

    Returns:
        np.ndarray: Losses per sample.
    """
    if not isinstance(model, torch.nn.Module) or dataloader is None:
        return np.array([])

    try:
        device = next(model.parameters()).device
    except Exception:
        device = torch.device("cpu")

    criterion = nn.CrossEntropyLoss(reduction="none")
    losses = []
    collected = 0

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                continue

            samples, labels = batch[0], batch[1]
            if not torch.is_tensor(samples) or not torch.is_tensor(labels):
                continue

            remaining = max_samples - collected
            if remaining <= 0:
                break

            samples = samples[:remaining].to(device)
            labels = labels[:remaining]

            if labels.ndim > 1:
                labels = torch.argmax(labels, dim=1)

            labels = labels.long().to(device)

            outputs = model(samples)
            logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            batch_losses = criterion(logits, labels)

            losses.append(batch_losses.detach().cpu().numpy())
            collected += int(batch_losses.shape[0])

    if not losses:
        return np.array([])

    return np.concatenate(losses, axis=0)


def get_epsilon_star(model, train_dataloader, test_dataloader, max_samples=5000):
    """
    Compute empirical epsilon* from train/test loss distributions.

    This follows the same core structure as privacy_metrics_core.epsilon_star,
    adapted to PyTorch models and DataLoaders used in Nebula.

    Args:
        model (torch.nn.Module): Model to evaluate.
        train_dataloader: Training DataLoader.
        test_dataloader: Test DataLoader.
        max_samples (int): Maximum samples to evaluate per split.

    Returns:
        float: Empirical epsilon* value. Returns 0.0 on failure.
    """
    try:
        loss_train = _collect_per_sample_losses(model, train_dataloader, max_samples=max_samples)
        loss_test = _collect_per_sample_losses(model, test_dataloader, max_samples=max_samples)

        if loss_train.size == 0 or loss_test.size == 0:
            return 0.0

        scores = np.concatenate([-loss_train, -loss_test])
        y_true = np.concatenate([np.ones(len(loss_train)), np.zeros(len(loss_test))])

        fpr, tpr, _ = roc_curve(y_true, scores)

        fpr = np.clip(fpr, 1e-10, 1 - 1e-10)
        tpr = np.clip(tpr, 1e-10, 1 - 1e-10)
        fnr = 1 - tpr

        delta = 1.0 / len(loss_train) if len(loss_train) > 0 else 1e-5

        m1 = (1 - delta - fnr) / fpr
        m2 = (1 - delta - fpr) / fnr
        m3 = (fnr - delta) / (1 - fpr)
        m4 = (fpr - delta) / (1 - fnr)

        epsilon_star_val = np.log(
            np.nanmax(np.maximum.reduce([m1, m2, m3, m4, np.ones_like(m1)]))
        )

        if np.isnan(epsilon_star_val) or np.isinf(epsilon_star_val):
            return 0.0

        return float(max(0.0, epsilon_star_val))
    except Exception as exc:
        logger.warning("Could not compute epsilon_star")
        logger.warning(exc)
        return 0.0


def get_mia_auc(model, train_dataloader, test_dataloader, max_samples=5000):
    """
    Compute membership inference attack AUC using per-sample loss as the attack score.

    Lower loss suggests a sample is more likely to be a training member, so the
    attack score is defined as negative loss.

    Args:
        model (torch.nn.Module): Model to evaluate.
        train_dataloader: Training DataLoader.
        test_dataloader: Test DataLoader.
        max_samples (int): Maximum samples to evaluate per split.

    Returns:
        float: ROC-AUC of the loss-threshold membership attack. Returns 0.5 on failure.
    """
    try:
        loss_train = _collect_per_sample_losses(model, train_dataloader, max_samples=max_samples)
        loss_test = _collect_per_sample_losses(model, test_dataloader, max_samples=max_samples)

        if loss_train.size == 0 or loss_test.size == 0:
            return 0.5

        scores = np.concatenate([-loss_train, -loss_test])
        y_true = np.concatenate([np.ones(len(loss_train)), np.zeros(len(loss_test))])
        mia_auc = roc_auc_score(y_true, scores)

        if np.isnan(mia_auc) or np.isinf(mia_auc):
            return 0.5

        return float(np.clip(mia_auc, 0.0, 1.0))
    except Exception as exc:
        logger.warning("Could not compute mia_auc")
        logger.warning(exc)
        return 0.5


# ---------------------------------------------------------------------------
# Scenario report readers and aggregate system metrics
# ---------------------------------------------------------------------------

def get_elapsed_time(start_time, end_time):
    """
    Calculates the elapsed time during the execution of the scenario.

    Args:
        start_time (datetime): Start datetime.
        end_time (datetime): End datetime.

    Returns:
        float: The elapsed time.
    """
    start_date = datetime.strptime(start_time, "%d/%m/%Y %H:%M:%S")
    end_date = datetime.strptime(end_time, "%d/%m/%Y %H:%M:%S")

    elapsed_time = (end_date - start_date).total_seconds() / 60

    return elapsed_time


def _trustworthiness_dir(scenario_name):
    return os.path.join(os.environ.get('NEBULA_LOGS_DIR'), scenario_name, "trustworthiness")


def _global_data_results_path(scenario_name):
    return os.path.join(_trustworthiness_dir(scenario_name), "data_results.csv")


def _participant_data_results_path(scenario_name, participant_id):
    return os.path.join(_trustworthiness_dir(scenario_name), f"data_results_{participant_id}.csv")


def _read_global_results(scenario_name):
    return read_csv(_global_data_results_path(scenario_name))


def _read_participant_results(scenario_name, participant_id):
    return read_csv(_participant_data_results_path(scenario_name, participant_id))


def _find_participant_row(data, participant_id, source_name):
    row = data[data["id"] == participant_id]

    if row.empty:
        try:
            row = data[data["id"] == int(participant_id)]
        except (TypeError, ValueError):
            row = data.iloc[0:0]

    if row.empty:
        raise ValueError(f"Participant {participant_id} not found in {source_name}")

    return row.iloc[0]


def get_bytes_model(model):
    """
    Calculates the serialized size in bytes of a PyTorch model state_dict.

    Args:
        model (nn.Module): PyTorch model.

    Returns:
        int: Model size in bytes.
    """
    buffer: io.BytesIO = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    model_size: int = buffer.tell()

    return model_size


def get_bytes_sent_recv(scenario_name):
    """
    Calculates the mean bytes sent and received of the nodes.

    Args:
        bytes_sent_files (list): Files that contain the bytes sent of the nodes.
        bytes_recv_files (list): Files that contain the bytes received of the nodes.

    Returns:
        4-tupla: The total bytes sent, the total bytes received, the mean bytes sent and the mean bytes received of the nodes.
    """
    data = _read_global_results(scenario_name)

    number_files = len(data)

    total_upload_bytes = int(data["bytes_sent"].sum())
    total_download_bytes = int(data["bytes_recv"].sum())

    avg_upload_bytes = total_upload_bytes / number_files
    avg_download_bytes = total_download_bytes / number_files

    return total_upload_bytes, total_download_bytes, avg_upload_bytes, avg_download_bytes


def get_avg_loss_accuracy(scenario_name):
    """
    Calculates the mean accuracy and loss models of the nodes.

    Args:
        loss_files (list): Files that contain the loss of the models of the nodes.
        accuracy_files (list): Files that contain the acurracies of the models of the nodes.

    Returns:
        3-tupla: The mean loss of the models, the mean accuracies of the models, the standard deviation of the accuracies of the models.
    """
    data = _read_global_results(scenario_name)

    number_files = len(data)

    total_loss = data["loss"].sum()
    total_accuracy = data["accuracy"].sum()

    denominator = max(1, number_files - 1)
    avg_loss = total_loss / denominator
    avg_accuracy = total_accuracy / denominator
    std_accuracy = statistics.stdev(data["accuracy"]) if number_files > 1 else 0.0

    return avg_loss, avg_accuracy, std_accuracy

def get_underfitting_score(scenario_name, id):
    """
    Calculates the mean val accuracy of the nodes.
    """
    data = _read_global_results(scenario_name)

    number_files = len(data)

    total_val_accuracy = data["val_accuracy"].sum()

    avg_val_accuracy = total_val_accuracy / max(1, number_files - 1)

    return avg_val_accuracy


def get_participant_loss_accuracy(scenario_name, participant_id):
    """
    Gets loss and accuracy for a specific participant from CFL aggregated results.

    Args:
        scenario_name (str): Scenario name.
        participant_id (int | str): Participant identifier.

    Returns:
        tuple[float, float]: (loss, accuracy)
    """
    data_file = _global_data_results_path(scenario_name)
    row = _find_participant_row(read_csv(data_file), participant_id, data_file)

    loss = float(row["loss"])
    accuracy = float(row["accuracy"])
    return loss, accuracy


# ---------------------------------------------------------------------------
# Model performance metrics
# ---------------------------------------------------------------------------


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

def get_underfitting_score_local(scenario_name, id):
    """
    Gets the local validation accuracy for a specific DFL/SDFL participant.

    Args:
        scenario_name (str): Scenario name.
        participant_id (int | str): Participant identifier.

    Returns:
        float: Validation accuracy.
    """
    data = _read_participant_results(scenario_name, id)
    return float(data["val_accuracy"].iloc[0])

def get_dp_local(scenario_name, id):
    """
    Gets the dp metrics for a specific DFL/SDFL participant.

    Args:
        scenario_name (str): Scenario name.
        participant_id (int | str): Participant identifier.

    Returns:
        float: DP Enabled, Epsilon.
    """
    data = _read_participant_results(scenario_name, id)
    return data["dp_enabled"].iloc[0], float(data["dp_epsilon"].iloc[0])


def get_dp_global(scenario_name):
    """
    Gets the aggregated DP metrics for a CFL scenario, excluding the server node.

    Args:
        scenario_name (str): Scenario name.

    Returns:
        tuple[bool, float | str]: Whether DP is enabled, and the
        average epsilon across client nodes.
    """
    data = _read_global_results(scenario_name)

    if data["dp_enabled"].iloc[0] == False:
        return False, 0.0

    number_files = len(data)

    avg_epsilon = data["dp_epsilon"].sum() / max(1, number_files - 1)

    return True, avg_epsilon


# ---------------------------------------------------------------------------
# Fairness and calibration metrics
# ---------------------------------------------------------------------------

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


def get_avg_class_imbalance_model_size(scenario_name):
    """
    Calculates the mean class imbalance and model size of the nodes.

    Args:
        data_results_files (list): Files that contain the class imbalance and model size of the nodes

    Returns:
        2-tupla: The mean class imbalance mean and model size mean of the nodes.
    """
    data = _read_global_results(scenario_name)

    number_files = len(data)

    total_class_imbalance = data["class_imbalance"].sum()
    total_model_size = data["model_size"].sum()

    avg_class_imbalance = total_class_imbalance / number_files
    avg_model_size = total_model_size / number_files

    return avg_class_imbalance, avg_model_size


def get_entropy_list(scenario_name):
    """
    Obtiene una lista con los valores de entropy de todos los nodos.

    Args:
        scenario_name (str): Nombre del escenario.

    Returns:
        list: Lista con los valores de entropy
    """
    data = _read_global_results(scenario_name)

    entropy_list = data["local_entropy"].tolist()

    return entropy_list


# ---------------------------------------------------------------------------
# Explainability metrics
# ---------------------------------------------------------------------------

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
        shap_values = _compute_shap_values(shap_model, background, test_data)
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
        logger.warning("Could not compute feature importances with shap")
        logger.warning(exc)
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


# ---------------------------------------------------------------------------
# Robustness metrics based on ART estimators
# ---------------------------------------------------------------------------

def _build_art_classifier(model, input_shape, nb_classes, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), learning_rate)

    return PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=tuple(input_shape),
        nb_classes=nb_classes,
    )


def _validate_test_sample_tensors(test_sample):
    if not (isinstance(test_sample, (tuple, list)) and len(test_sample) >= 2):
        raise ValueError("`test_sample` must contain samples and labels.")

    samples, labels = test_sample[0], test_sample[1]
    if not (torch.is_tensor(samples) and torch.is_tensor(labels) and samples.shape[0] > 0):
        raise ValueError("`test_sample` must contain non-empty tensors for samples and labels.")

    return samples, labels


def _coerce_max_samples(max_samples, default=8):
    try:
        return max(1, int(max_samples))
    except Exception:
        return default


def get_clever_score(model, test_sample, nb_classes, learning_rate, max_samples=8):
    """
    Calculates the CLEVER score as the mean score over multiple samples.

    Args:
        model (object): The model.
        test_sample (object): A batch from the test dataloader.
        nb_classes (int): The nb_classes of the model.
        learning_rate (float): The learning rate of the model.
        max_samples (int): Maximum number of samples from the batch to evaluate.

    Returns:
        float: Mean CLEVER score across the selected samples.
    """
    samples, _ = _validate_test_sample_tensors(test_sample)

    input_shape = tuple(samples.shape[1:]) if samples.dim() >= 2 else tuple(samples.shape)

    max_samples = _coerce_max_samples(max_samples)
    n_samples = min(int(samples.shape[0]), max_samples)

    # Create the ART classifier once and reuse it for all selected samples.
    classifier = _build_art_classifier(model, input_shape, nb_classes, learning_rate)

    clever_scores = []
    for idx in range(n_samples):
        background = samples[idx].detach().cpu()
        sample_np = background.numpy()

        try:
            score_untargeted = clever_u(
                classifier,
                sample_np,
                10,
                5,
                R_L2,
                norm=2,
                pool_factor=3,
                verbose=False,
            )
            if score_untargeted is not None and not math.isnan(float(score_untargeted)):
                clever_scores.append(float(score_untargeted))
        except Exception as exc:
            logger.warning("Could not compute CLEVER score for sample index %s", idx)
            logger.warning(exc)

    if not clever_scores:
        return 0.0

    return float(np.mean(clever_scores))


# ---------------------------------------------------------------------------
# Sustainability and communication metrics
# ---------------------------------------------------------------------------

def stop_emissions_tracking_and_save(
    tracker: EmissionsTracker,
    outdir: str,
    emissions_file: str,
    role: str,
    workload: str,
    sample_size: int = 0,
    participant_idx=None,
):
    """
    Stops emissions tracking object from CodeCarbon and saves relevant information to emissions.csv file.

    Args:
        tracker (object): The emissions tracker object holding information.
        outdir (str): The path of the output directory of the experiment.
        emissions_file (str): The path to the emissions file.
        role (str): Either client or server depending on the role.
        workload (str): Either aggregation or training depending on the workload.
        sample_size (int): The number of samples used for training, if aggregation 0.
    """

    tracker.stop()

    emissions_file = os.path.join(outdir, emissions_file)

    if exists(emissions_file):
        df = pd.read_csv(emissions_file)
    else:
        df = pd.DataFrame(
            columns=[
                "id",
                "role",
                "energy_grid",
                "emissions",
                "workload",
                "CPU_model",
                "GPU_model",
            ]
        )
    try:
        energy_grid = (tracker.final_emissions_data.emissions / tracker.final_emissions_data.energy_consumed) * 1000
        df = pd.concat(
            [
                df,
                pd.DataFrame({
                    "id": participant_idx,
                    "role": role,
                    "energy_grid": [energy_grid],
                    "emissions": [tracker.final_emissions_data.emissions],
                    "workload": workload,
                    "CPU_model": tracker.final_emissions_data.cpu_model
                    if tracker.final_emissions_data.cpu_model
                    else "None",
                    "GPU_model": tracker.final_emissions_data.gpu_model
                    if tracker.final_emissions_data.gpu_model
                    else "None",
                    "CPU_used": True if tracker.final_emissions_data.cpu_energy else False,
                    "GPU_used": True if tracker.final_emissions_data.gpu_energy else False,
                    "energy_consumed": tracker.final_emissions_data.energy_consumed,
                    "sample_size": sample_size,
                }),
            ],
            ignore_index=True,
        )
        df.to_csv(emissions_file, encoding="utf-8", index=False)
    except Exception as e:
        logger.warning(e)

def comm_efficiency(bytes_up: int, bytes_down: int, test_acc_avg: float, eps: float = 1e-12) -> float:
    """
    Communication efficiency = total_bytes / final_accuracy.
    Lower is better.

    Args:
        bytes_up: total uploaded bytes
        bytes_down: total downloaded bytes
        final_accuracy: final test accuracy in [0,1]
        eps: small constant to avoid division by zero

    Returns:
        float
    """
    total_bytes = float(bytes_up) + float(bytes_down)
    acc = float(test_acc_avg)

    if acc < eps:
        acc = eps

    return total_bytes / acc


# ---------------------------------------------------------------------------
# Additional robustness and adversarial metrics
# ---------------------------------------------------------------------------

def get_loss_sensitivity_score(model, test_sample, nb_classes, learning_rate, max_samples=8):

    """
    Calculates the loss sensitivity score as the mean score over multiple samples.

    Args:
        model (object): The model.
        test_sample (object): A batch from the test dataloader.
        nb_classes (int): The nb_classes of the model.
        learning_rate (float): The learning rate of the model.
        max_samples (int): Maximum number of samples from the batch to evaluate.

    Returns:
        float: Mean loss sensitivity score across the selected samples.
    """
    samples, labels = _validate_test_sample_tensors(test_sample)

    max_samples = _coerce_max_samples(max_samples)
    n_samples = min(int(samples.shape[0]), max_samples)

    # Create the ART classifier once and reuse it for all selected samples.
    classifier = _build_art_classifier(model, samples.shape[1:], nb_classes, learning_rate)

    sensitivity_scores = []
    for idx in range(n_samples):
        sample = samples[idx].detach().cpu().unsqueeze(0)
        label = labels[idx].detach().cpu().unsqueeze(0)
        label = F.one_hot(label, num_classes=nb_classes).float()

        try:
            score = loss_sensitivity(
                classifier,
                sample.numpy(),
                label.numpy(),
            )
            if score is not None and not math.isnan(float(score)):
                sensitivity_scores.append(float(score))
        except Exception as exc:
            logger.warning("Could not compute loss sensitivity for sample index %s", idx)
            logger.warning(exc)

    if not sensitivity_scores:
        return 0.0

    return float(np.mean(sensitivity_scores))

def compute_adversarial_accuracy_art(
    model,
    test_loader,
    nb_classes,
    learning_rate,
    epsilon=0.03
):
    """
    Computes adversarial accuracy using FGSM attack.

    Args:
        model (object): The model.
        test_loader (DataLoader): DataLoader providing test samples.
        nb_classes (int): The nb_classes of the model.
        learning_rate (float): The learning rate of the model.
        epsilon (float): Maximum perturbation magnitude for the attacks.

    Returns:
        float: The adversarial accuracy score.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    correct = 0
    total = 0

    for samples, labels in test_loader:
        samples = samples.to(device)
        labels = labels.to(device)

        x_adv = fgsm_attack(model, samples, labels, epsilon=epsilon)

        with torch.no_grad():
            outputs = model(x_adv)
            logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total if total > 0 else 0.0

def get_empirical_robustness_score(
    model,
    test_sample,
    nb_classes,
    learning_rate,
    attack_name = "fgsm",
    attack_params = None,
    max_samples = 128,
):
    """
    Calculates the Empirical Robustness score using Adversarial Robustness Toolbox (ART).

    Args:
        model (object): The model.
        test_sample (object): A batch from the test dataloader (samples, labels).
        nb_classes (int): The nb_classes of the model.
        learning_rate (float): The learning rate of the model.
        attack_name (str): Attack key supported by ART empirical_robustness.
        attack_params (dict | None): Optional attack parameters.
        max_samples (int): Max number of samples from the batch to use.

    Returns:
        float: Empirical robustness score (>= 0.0). If it cannot be computed, returns 0.0.
    """
    try:
        samples, _ = _validate_test_sample_tensors(test_sample)

        batch_size: int = int(samples.shape[0])
        n: int = int(min(max_samples, batch_size))
        x = samples[:n].detach().cpu().numpy()

        classifier = _build_art_classifier(model, samples.shape[1:], nb_classes, learning_rate)

        score = empirical_robustness(
            classifier=classifier,
            x=x,
            attack_name=attack_name,
            attack_params=attack_params,
        )

        if isinstance(score, np.ndarray):
            score = float(np.mean(score))

        if score is None or (isinstance(score, float) and math.isnan(score)):
            return 0.0

        return float(score)

    except Exception as exc:
        logger.warning("Could not compute empirical robustness (ART). Returning 0.0")
        logger.warning(exc)
        return 0.0



def fgsm_attack(model, samples, labels, epsilon=0.03):
    """
        Performs an FGSM (Fast Gradient Sign Method) adversarial attack on a batch of samples.

        Args:
            model (torch.nn.Module): The PyTorch model to attack.
            samples (torch.Tensor): Input samples to perturb, shape (B, ...).
            labels (torch.Tensor): True labels corresponding to the samples.
            epsilon (float, optional): Maximum perturbation magnitude for the attack. Defaults to 0.03.

        Returns:
            torch.Tensor: Adversarially perturbed samples with the same shape as `samples`.
    """
    try:
        device = next(model.parameters()).device
    except Exception:
        device = samples.device

    samples = samples.clone().detach().to(device)
    labels = labels.to(device)
    samples.requires_grad = True

    outputs = model(samples)
    logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
    loss = nn.CrossEntropyLoss()(logits, labels)
    model.zero_grad()
    loss.backward()

    perturbation = epsilon * samples.grad.sign()
    x_adv = samples + perturbation

    return x_adv.detach()

def get_confidence_score(
    model,
    test_sample,
    max_samples = 128,
    use_true_label = True,
):
    """
    Calculates the confidence score.

    Args:
        model (object): The model.
        test_sample (object): A batch from the test dataloader (samples, labels).
        max_samples (int): Max number of samples from the batch to use.
        use_true_label (bool): Whether to compute confidence with respect to the true labels. Defaults to True.

    Returns:
        float: Confidence score.
    """
    try:
        if not isinstance(model, torch.nn.Module):
            logger.warning("Model is not a torch.nn.Module")
            return 0.0

        x, y = test_sample

        if isinstance(x, torch.Tensor):
            x = x[:max_samples]
        if isinstance(y, torch.Tensor):
            y = y[:max_samples]

        try:
            device = next(model.parameters()).device
        except Exception:
            device = torch.device("cpu")

        model.eval()
        with torch.no_grad():
            x = x.to(device) if isinstance(x, torch.Tensor) else x
            out = model(x)

            logits = out[0] if isinstance(out, (tuple, list)) else out
            probs = torch.softmax(logits, dim=1)

            if use_true_label and isinstance(y, torch.Tensor):
                if y.ndim > 1:
                    y_idx = torch.argmax(y, dim=1)
                else:
                    y_idx = y
                y_idx = y_idx.to(device)

                true_probs = probs.gather(1, y_idx.view(-1, 1)).squeeze(1)
                return float(true_probs.mean().detach().cpu().item())

            msp = probs.max(dim=1).values
            return float(msp.mean().detach().cpu().item())

    except Exception as e:
        logger.warning("Could not compute confidence score")
        logger.warning(e)
        return 0.0

def attack_success_rate(model, test_sample,epsilon=0.03):
    """
    Calculates the ASR.

    Args:
        model (object): The model.
        test_sample (object): A batch from the test dataloader (samples, labels).
        epsilon (float): Maximum perturbation magnitude for the attacks.

    Returns:
        float: The ASR.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    images, labels = test_sample
    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        preds = logits.argmax(dim=1)

    correct_mask = preds.eq(labels)
    num_correct = correct_mask.sum().item()

    if num_correct == 0:
        return 0.0

    x_adv = fgsm_attack(model, images, labels, epsilon=epsilon)

    with torch.no_grad():
        outputs_adv = model(x_adv)
        logits_adv = outputs_adv[0] if isinstance(outputs_adv, (tuple, list)) else outputs_adv
        preds_adv = logits_adv.argmax(dim=1)

    successful_attacks = (correct_mask & preds_adv.ne(labels)).sum().item()

    asr = successful_attacks / num_correct

    return asr
