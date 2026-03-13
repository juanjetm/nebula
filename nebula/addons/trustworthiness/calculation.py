import logging
import math
import numbers
import os.path
import statistics
from datetime import datetime
from math import e
from os.path import exists
import json

import numpy as np
import pandas as pd
import shap
import torch.nn
from art.estimators.classification import PyTorchClassifier
from art.metrics import clever_u, loss_sensitivity, empirical_robustness
from codecarbon import EmissionsTracker
from scipy.stats import variation
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
        cv = 0
        batch_size = 10
        device = "cpu"

        if isinstance(model, torch.nn.Module):
            batched_data, _ = test_sample

            n = batch_size
            m = math.floor(0.8 * n)

            background = batched_data[:m].to(device)
            test_data = batched_data[m:n].to(device)

            e = shap.DeepExplainer(model, background)
            shap_values = e.shap_values(test_data)
            if shap_values is not None and len(shap_values) > 0:
                sums = np.array([shap_values[i].sum() for i in range(len(shap_values))])
                abs_sums = np.absolute(sums)
                cv = variation(abs_sums)
    except Exception as e:
        logger.warning("Could not compute feature importance CV with shap")
        cv = 1
    if math.isnan(cv):
        cv = 1
    return cv


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
