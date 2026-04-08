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
from torch import nn, optim
import torch.nn.functional as F
import time

from nebula.addons.trustworthiness.utils import read_csv

dirname = os.path.dirname(__file__)
logger = logging.getLogger(__name__)

R_L1 = 40
R_L2 = 2
R_LI = 0.1


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
    normalized = [(x - np.min(scores)) / (np.max(scores) - np.min(scores)) for x in scores]
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
    if not value:
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
        return std / mean

    if list is not None:
        return np.std(list) / np.mean(list)

    return 0


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


def get_bytes_models(models_files):
    """
    Calculates the mean bytes of the final models of the nodes.

    Args:
        models_files (list): List of final models.

    Returns:
        float: The mean bytes of the models.
    """

    total_models_size = 0
    number_models = len(models_files)

    for file in models_files:
        model_size = os.path.getsize(file)
        total_models_size += model_size

    avg_model_size = total_models_size / number_models

    return avg_model_size

def get_bytes_model(model_file):
    """
    Calculates the bytes of the final model of a node.

    Args:
        model_file: Final model.

    Returns:
        float: The bytes of the model.
    """

    model_size = os.path.getsize(model_file)

    return model_size

def get_bytes_final_model_id(id, scenario_name):
    """
    Calculates the bytes of the final model of a node by id.

    Args:
        id: Participant ID.

    Returns:
        float: The bytes of the model.
    """


    model_file = os.path.join(os.environ.get('NEBULA_LOGS_DIR'), scenario_name, "trustworthiness", f"participant_{id}_final_model.pk")

    model_size = os.path.getsize(model_file)

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
    total_upload_bytes = 0
    total_download_bytes = 0

    data_file = os.path.join(os.environ.get('NEBULA_LOGS_DIR'), scenario_name, "trustworthiness", "data_results.csv")

    data = read_csv(data_file)

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
    total_accuracy = 0
    total_loss = 0

    data_file = os.path.join(os.environ.get('NEBULA_LOGS_DIR'), scenario_name, "trustworthiness", "data_results.csv")

    data = read_csv(data_file)

    number_files = len(data)

    total_loss = data["loss"].sum()
    total_accuracy = data["accuracy"].sum()

    avg_loss = total_loss / number_files
    avg_accuracy = total_accuracy / number_files
    std_accuracy = statistics.stdev(data["accuracy"])

    return avg_loss, avg_accuracy, std_accuracy


def get_participant_loss_accuracy(scenario_name, participant_id):
    """
    Gets loss and accuracy for a specific participant from CFL aggregated results.

    Args:
        scenario_name (str): Scenario name.
        participant_id (int | str): Participant identifier.

    Returns:
        tuple[float, float]: (loss, accuracy)
    """
    data_file = os.path.join(os.environ.get('NEBULA_LOGS_DIR'), scenario_name, "trustworthiness", "data_results.csv")
    data = read_csv(data_file)
    row = data[data["id"] == participant_id]

    if row.empty:
        row = data[data["id"] == int(participant_id)]

    loss = float(row["loss"].iloc[0])
    accuracy = float(row["accuracy"].iloc[0])
    return loss, accuracy


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


def get_underfitting_score(test_accuracy):
    """
    Uses test accuracy as a proxy for underfitting.

    Args:
        test_accuracy (float): Test accuracy in [0, 1].

    Returns:
        float: Underfitting proxy value.
    """
    try:
        return float(test_accuracy)
    except Exception:
        logger.warning("Could not compute underfitting score")
        return 0.0


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


def get_avg_class_imbalance_model_size(scenario_name):
    """
    Calculates the mean class imbalance and model size of the nodes.

    Args:
        data_results_files (list): Files that contain the class imbalance and model size of the nodes

    Returns:
        2-tupla: The mean class imbalance mean and model size mean of the nodes.
    """
    total_class_imbalance = 0
    total_model_size = 0

    data_file = os.path.join(os.environ.get('NEBULA_LOGS_DIR'), scenario_name, "trustworthiness", "data_results.csv")

    data = read_csv(data_file)

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
    data_file = os.path.join(
        os.environ.get('NEBULA_LOGS_DIR'),
        scenario_name,
        "trustworthiness",
        "data_results.csv"
    )

    data = read_csv(data_file)

    entropy_list = data["local_entropy"].tolist()

    return entropy_list

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
        try:
            model_clone = copy.deepcopy(model_ref)
            model_clone.to(device)
            model_clone.eval()
            return model_clone
        except Exception as exc:
            logger.warning("Could not clone model for SHAP, using original model")
            logger.warning(exc)
            model_ref.eval()
            return model_ref

    def _prepare_shap_inputs(sample):
        if not (isinstance(sample, (tuple, list)) and len(sample) >= 1):
            return None, None, None

        batched_data = sample[0]
        if not torch.is_tensor(batched_data) or batched_data.ndim == 0 or batched_data.size(0) == 0:
            return None, None, None

        if not torch.is_floating_point(batched_data):
            batched_data = batched_data.float()

        batch_size = int(batched_data.size(0))
        if batched_data.ndim == 4:
            # SHAP image explainers operate more naturally on channel-last images.
            input_shape = (
                int(batched_data.shape[2]),
                int(batched_data.shape[3]),
                int(batched_data.shape[1]),
            )
        else:
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

        if test_data.ndim == 4:
            def predict_fn(images):
                if isinstance(images, list):
                    images = np.asarray(images)

                image_tensor = torch.as_tensor(images, dtype=test_data.dtype, device=background.device)
                if image_tensor.ndim == 3:
                    image_tensor = image_tensor.unsqueeze(0)

                if image_tensor.ndim != 4:
                    raise ValueError(f"Expected 4D image batch for SHAP, got shape {tuple(image_tensor.shape)}")

                # SHAP image maskers provide NHWC arrays; convert back to NCHW for the model.
                image_tensor = image_tensor.permute(0, 3, 1, 2).contiguous()

                with torch.no_grad():
                    logits = _extract_model_logits(model_ref(image_tensor))
                    probs = _logits_to_probabilities(logits)
                return probs.detach().cpu().numpy()

            try:
                test_images = test_data.detach().cpu().numpy().transpose(0, 2, 3, 1)
                masker = shap.maskers.Image("blur(8,8)", test_images[0].shape)
                explainer = shap.Explainer(predict_fn, masker)
                explanation = explainer(
                    test_images[: min(int(test_images.shape[0]), 4)],
                    max_evals=128,
                    batch_size=8,
                )
                return explanation.values
            except Exception as exc:
                explainer_errors.append(f"ImageExplainer: {exc}")

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


def get_clever_score(model, test_sample, nb_classes, learning_rate):
    """
    Calculates the CLEVER score.

    Args:
        model (object): The model.
        test_sample (object): One test sample to calculate the CLEVER score.
        nb_classes (int): The nb_classes of the model.
        learning_rate (float): The learning rate of the model.

    Returns:
        float: The CLEVER score.
    """


    samples, _ = test_sample
    input_shape = None

    if torch.is_tensor(samples) and samples.dim() >= 1 and samples.shape[0] != 0:
        pass
    else:
        raise ValueError("`test_sample[0]` must be a non-empty torch.Tensor.")

    if input_shape is None:
        if samples.dim() >= 2:
            # (B, ...) -> input_shape = (...)
            input_shape = tuple(samples.shape[1:])
        else:
            # (...) without batch
            input_shape = tuple(samples.shape)

    background = samples[-1] if samples.dim() >= 2 else samples

    x = background.detach().cpu().numpy()

    if tuple(x.shape) == tuple(input_shape):
        x = x.reshape((1,) + tuple(input_shape))


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), learning_rate)

    # Create the ART classifier
    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=input_shape,
        nb_classes=nb_classes,
    )

    score_untargeted = clever_u(
        classifier,
        background.numpy(),
        10,
        5,
        R_L2,
        norm=2,
        pool_factor=3,
        verbose=False,
    )
    return score_untargeted


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

def get_loss_sensitivity_score(model, test_sample, nb_classes, learning_rate):

    """
    Calculates the loss sensitivity score.

    Args:
        model (object): The model.
        test_sample (object): One test sample to calculate the loss sensitivity score.
        nb_classes (int): The nb_classes of the model.
        learning_rate (float): The learning rate of the model.

    Returns:
        float: The loss sensitivity score.
    """

    samples, labels = test_sample
    sample = samples[-1].unsqueeze(0)
    label = labels[-1].unsqueeze(0)

    label = F.one_hot(label, num_classes=nb_classes).float()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), learning_rate)

    # Create the ART classifier
    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=sample.shape[1:],
        nb_classes=nb_classes,
    )

    score = loss_sensitivity(
        classifier,
        sample.numpy(),
        label.numpy(),
    )
    return float(score)

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

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    sample_batch = next(iter(test_loader))
    samples, _ = sample_batch
    input_shape = samples.shape[1:]

    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=input_shape,
        nb_classes=nb_classes,
    )

    correct = 0
    total = 0

    for samples, labels in test_loader:
        samples = samples.to(device)
        labels = labels.to(device)

        x_adv = fgsm_attack(model, samples, labels, epsilon=epsilon)

        with torch.no_grad():
            outputs = model(x_adv)
            preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total

def get_empirical_robustness_score(
    model,
    test_sample,
    nb_classes,
    learning_rate,
    attack_name = "fgsm",
    attack_params = None,
    max_samples = 32,
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
        samples, _ = test_sample

        batch_size: int = int(samples.shape[0])
        n: int = int(min(max_samples, batch_size))
        x = samples[:n].detach().cpu().numpy()

        input_shape = tuple(samples.shape[1:])

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), learning_rate)

        classifier = PyTorchClassifier(
            model=model,
            loss=criterion,
            optimizer=optimizer,
            input_shape=input_shape,
            nb_classes=nb_classes,
        )

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
    samples = samples.clone().detach().to(samples.device)
    labels = labels.to(samples.device)
    samples.requires_grad = True

    outputs = model(samples)
    loss = nn.CrossEntropyLoss()(outputs, labels)
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

    images, labels = test_sample
    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        preds = outputs.argmax(dim=1)

    correct_mask = preds.eq(labels)
    num_correct = correct_mask.sum().item()

    if num_correct == 0:
        return 0.0

    x_adv = fgsm_attack(model, images, labels, epsilon=epsilon)

    with torch.no_grad():
        outputs_adv = model(x_adv)
        preds_adv = outputs_adv.argmax(dim=1)

    successful_attacks = (correct_mask & preds_adv.ne(labels)).sum().item()

    asr = successful_attacks / num_correct

    return asr
