import io
import logging
import os
import statistics
from datetime import datetime
from os.path import exists

import pandas as pd
import torch
from codecarbon import EmissionsTracker

from nebula.addons.trustworthiness.helpers.csv_io import read_csv

logger = logging.getLogger(__name__)

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
