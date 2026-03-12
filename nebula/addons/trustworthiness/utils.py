import json
import csv
import logging
import math
import os
import pickle
from os.path import exists

import pandas as pd
from hashids import Hashids
from scipy.stats import entropy

from nebula.addons.trustworthiness import calculation
from collections import Counter

hashids = Hashids()
logger = logging.getLogger(__name__)
dirname = os.path.dirname(__file__)


def save_class_count_per_participant(experiment_name, class_counter: Counter, idx):
    class_count = os.path.join(os.environ.get('NEBULA_LOGS_DIR'), experiment_name, "trustworthiness", f"{str(idx)}_class_count.json")
    result = {hashids.encode(int(class_id)): count for class_id, count in class_counter.items()}
    with open(class_count, "w") as f:
        json.dump(result, f)

def count_all_class_samples(experiment_name):
    participant_id = 0
    global_class_count = {}

    while True:
        data_class_count_file = os.path.join(os.environ.get('NEBULA_LOGS_DIR'), experiment_name, "trustworthiness", f"{str(participant_id)}_class_count.json")

        if not os.path.exists(data_class_count_file):
            break

        with open(data_class_count_file, "r") as f:
            class_count = json.load(f)

        for class_hash, count in class_count.items():
            global_class_count[class_hash] = global_class_count.get(class_hash, 0) + count

        participant_id += 1

    # Guardar conteo total en class_count.json
    output_file = os.path.join(os.environ.get('NEBULA_LOGS_DIR'),experiment_name, "trustworthiness", "count_class.json")

    with open(output_file, "w") as f:
        json.dump(global_class_count, f, indent=2)

def count_class_samples(scenario_name, dataloaders_files, class_counter: Counter = None):
    """
    Counts the number of samples by class.

    Args:
        scenario_name (string): Name of the scenario.
        dataloaders_files (list): Files that contain the dataloaders.

    """

    result = {}
    dataloaders = []

    if class_counter:
        result = {hashids.encode(int(class_id)): count for class_id, count in class_counter.items()}
    else:
        for file in dataloaders_files:
            with open(file, "rb") as f:
                dataloader = pickle.load(f)
                dataloaders.append(dataloader)

        for dataloader in dataloaders:
            for batch, labels in dataloader:
                for b, label in zip(batch, labels):
                    l = hashids.encode(label.item())
                    if l in result:
                        result[l] += 1
                    else:
                        result[l] = 1

    try:
        name_file = os.path.join(os.environ.get('NEBULA_LOGS_DIR'), scenario_name, "trustworthiness", "count_class.json")
    except:
        name_file = os.path.join("nebula", "app", "logs", scenario_name, "trustworthiness", "count_class.json")

    with open(name_file, "w") as f:
        json.dump(result, f)


def get_all_data_entropy(experiment_name):
    participant_id = 0
    data_class_count_file = os.path.join(os.environ.get('NEBULA_LOGS_DIR'), experiment_name, "trustworthiness", f"{str(participant_id)}_class_count.json")
    entropy_per_participant = {}

    while True:
        data_class_count_file = os.path.join(os.environ.get('NEBULA_LOGS_DIR'), experiment_name, "trustworthiness", f"{str(participant_id)}_class_count.json")

        if not os.path.exists(data_class_count_file):
            break

        with open(data_class_count_file, "r") as f:
            class_count = json.load(f)

        total = sum(class_count.values())
        if total == 0:
            entropy_value = 0.0
        else:
            probabilities = [count / total for count in class_count.values()]
            entropy_value = entropy(probabilities, base=2)

        entropy_per_participant[str(participant_id)] = round(entropy_value, 6)
        participant_id += 1

    name_file = os.path.join(os.environ.get('NEBULA_LOGS_DIR'),experiment_name, "trustworthiness", "entropy.json")

    with open(name_file, "w") as f:
        json.dump(entropy_per_participant, f, indent=2)

def get_entropy(client_id, scenario_name, dataloader):
    """
    Get the entropy of each client in the scenario.

    Args:
        client_id (int): The client id.
        scenario_name (string): Name of the scenario.
        dataloaders_files (list): Files that contain the dataloaders.

    """
    result = {}
    client_entropy = {}

    name_file = os.path.join(os.environ.get('NEBULA_LOGS_DIR'), scenario_name, "trustworthiness", "entropy.json")

    if os.path.exists(name_file):
        logging.info(f"entropy fiel already exists.. loading.")
        with open(name_file, "r") as f:
            client_entropy = json.load(f)

    client_id_hash = hashids.encode(client_id)

    for batch, labels in dataloader:
        for b, label in zip(batch, labels):
            l = hashids.encode(label.item())
            if l in result:
                result[l] += 1
            else:
                result[l] = 1

    n = len(dataloader)
    entropy_value = entropy([x / n for x in result.values()], base=2)
    client_entropy[client_id_hash] = entropy_value
    with open(name_file, "w") as f:
        json.dump(client_entropy, f)


def read_csv(filename):
    """
    Read a CSV file.

    Args:
        filename (string): Name of the file.

    Returns:
        object: The CSV readed.

    """
    if exists(filename):
        return pd.read_csv(filename)


def check_field_filled(factsheet_dict, factsheet_path, value, empty=""):
    """
    Check if the field in the factsheet file is filled or not.

    Args:
        factsheet_dict (dict): The factshett dict.
        factsheet_path (list): The factsheet field to check.
        value (float): The value to add in the field.
        empty (string): If the value could not be appended, the empty string is returned.

    Returns:
        float: The value added in the factsheet or empty if the value could not be appened

    """
    if factsheet_dict[factsheet_path[0]][factsheet_path[1]]:
        return factsheet_dict[factsheet_path[0]][factsheet_path[1]]
    elif value != "" and value != "nan":
        if type(value) != str and type(value) != list:
            if math.isnan(value):
                return 0
            else:
                return value
        else:
            return value
    else:
        return empty


def get_input_value(input_docs, inputs, operation):
    """
    Gets the input value from input document and apply the metric operation on the value.

    Args:
        inputs_docs (map): The input document map.
        inputs (list): All the inputs.
        operation (string): The metric operation.

    Returns:
        float: The metric value

    """

    input_value = None
    args = []
    for i in inputs:
        source = i.get("source", "")
        field = i.get("field_path", "")
        input_doc = input_docs.get(source, None)
        if input_doc is None:
            logger.warning(f"{source} is null")
        else:
            input = get_value_from_path(input_doc, field)
            args.append(input)
    try:
        operationFn = getattr(calculation, operation)
        input_value = operationFn(*args)
    except TypeError:
        logger.warning(f"{operation} is not valid")

    return input_value


def get_value_from_path(input_doc, path):
    """
    Gets the input value from input document by path.

    Args:
        inputs_doc (map): The input document map.
        path (string): The field name of the input value of interest.

    Returns:
        float: The input value from the input document

    """

    d = input_doc
    for nested_key in path.split("/"):
        temp = d.get(nested_key)
        if isinstance(temp, dict):
            d = d.get(nested_key)
        else:
            return temp
    return None


def write_results_json(out_file, dict):
    """
    Writes the result to JSON.

    Args:
        out_file (string): The output file.
        dict (dict): The object to be witten into JSON.

    Returns:
        float: The input value from the input document

    """

    with open(out_file, "a") as f:
        json.dump(dict, f, indent=4)

def load_data_results_participant(experiment_name: str, participant_id: int | str):
    data_results_path = os.path.join(os.environ.get('NEBULA_LOGS_DIR'), experiment_name, "trustworthiness", f"data_results_{participant_id}.csv")

    if not os.path.exists(data_results_path):
        raise FileNotFoundError(f"File not found: {data_results_path}")

    with open(data_results_path, "r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    if len(rows) == 0:
        raise ValueError(f"No rows found in {data_results_path}")

    row = rows[0]

    bytes_sent = int(float(row["bytes_sent"]))
    bytes_recv = int(float(row["bytes_recv"]))
    accuracy = float(row["accuracy"])
    loss = float(row["loss"])

    return bytes_sent, bytes_recv, accuracy, loss


def load_emissions_participant(experiment_name: str, participant_id: int | str):
    emissions_path = os.path.join(os.environ.get('NEBULA_LOGS_DIR'), experiment_name, "trustworthiness", f"emissions_{participant_id}.csv")

    if not os.path.exists(emissions_path):
        raise FileNotFoundError(f"File not found: {emissions_path}")

    with open(emissions_path, "r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    if len(rows) == 0:
        raise ValueError(f"No rows found in {emissions_path}")

    row = rows[0]

    role = str(row["role"])
    energy_grid = float(row["energy_grid"])
    emissions = float(row["emissions"])
    workload = str(row["workload"])
    cpu_model = str(row["CPU_model"])
    gpu_model = str(row["GPU_model"])
    cpu_used = str(row["CPU_used"]).strip().lower() == "true"
    gpu_used = str(row["GPU_used"]).strip().lower() == "true"
    energy_consumed = float(row["energy_consumed"])
    sample_size = int(float(row["sample_size"]))

    return role, energy_grid, emissions, workload, cpu_model, gpu_model, cpu_used, gpu_used, energy_consumed, sample_size

def save_trustworthiness_reports_csv(
    reports: dict,
    experiment_name: str,
) -> None:

    data_results_path = os.path.join("nebula", "app", "logs", experiment_name, "trustworthiness", "data_results.csv")
    emissions_path = os.path.join("nebula", "app", "logs", experiment_name, "trustworthiness", "emissions.csv")

    sorted_reports = sorted(
        reports.values(),
        key=lambda report: int(report["node_id"])
    )

    with open(data_results_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["id", "bytes_sent", "bytes_recv", "accuracy", "loss"],
        )
        writer.writeheader()

        for report in sorted_reports:
            writer.writerow({
                "id": report["node_id"],
                "bytes_sent": report["bytes_sent"],
                "bytes_recv": report["bytes_recv"],
                "accuracy": report["accuracy"],
                "loss": report["loss"],
            })

    with open(emissions_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["id", "role", "energy_grid", "emissions", "workload", "CPU_model", "GPU_model", "CPU_used", "GPU_used", "energy_consumed", "sample_size"],
        )
        writer.writeheader()

        for report in sorted_reports:
            writer.writerow({
                "id": report["node_id"],
                "role": report["role"],
                "energy_grid": report["energy_grid"],
                "emissions": report["emissions"],
                "workload": report["workload"],
                "CPU_model": report["cpu_model"],
                "GPU_model": report["gpu_model"],
                "CPU_used": report["cpu_used"],
                "GPU_used": report["gpu_used"],
                "energy_consumed": report["energy_consumed"],
                "sample_size": report["sample_size"],
            })

    logging.info(
        "[TW SERVER] CSV files written correctly: %s, %s",
        data_results_path,
        emissions_path,
    )

def save_results_csv_cfl(scenario_name: str, id: int, bytes_sent: int, bytes_recv: int, accuracy: float, loss: float):
    try:
        data_results_file = os.path.join(os.environ.get('NEBULA_LOGS_DIR'), scenario_name, "trustworthiness", "data_results.csv")
    except:
        data_results_file = os.path.join("nebula", "app", "logs", scenario_name, "trustworthiness", "data_results.csv")

    if exists(data_results_file):
        df = pd.read_csv(data_results_file)
    else:
        df = pd.DataFrame(columns=["id", "bytes_sent", "bytes_recv", "accuracy", "loss"])

    try:
        # Add new entry to DataFrame
        new_data = pd.DataFrame({'id': [id], 'bytes_sent': [bytes_sent],
                                    'bytes_recv': [bytes_recv], 'accuracy': [accuracy],
                                    'loss': [loss]})
        df = pd.concat([df, new_data], ignore_index=True)
        logger.info(f"new_data={new_data}")

        df.to_csv(data_results_file, encoding='utf-8', index=False)

    except Exception as e:
        logger.warning(e)

def save_emissions_csv_cfl(scenario_name: str, id: int, role: str, energy_grid: float, emissions: float, workload: str, cpu_model: str, gpu_model: str, cpu_used: bool, gpu_used: bool, energy_consumed: float, sample_size: int):
    try:
        data_results_file = os.path.join(os.environ.get('NEBULA_LOGS_DIR'), scenario_name, "trustworthiness", "emissions.csv")
    except:
        data_results_file = os.path.join("nebula", "app", "logs", scenario_name, "trustworthiness", "emissions.csv")

    if exists(data_results_file):
        df = pd.read_csv(data_results_file)
    else:
        df = pd.DataFrame(columns=["id", "role", "energy_grid", "emissions", "workload", "CPU_model", "GPU_model", "CPU_used", "GPU_used", "energy_consumed", "sample_size"])

    try:
        # Add new entry to DataFrame
        new_data = pd.DataFrame({'id': [id], 'role': [role], 'energy_grid': [energy_grid],
                                    'emissions': [emissions], 'workload': [workload], 'CPU_model': [cpu_model], 'GPU_model': [gpu_model], 'CPU_used': [cpu_used], 'GPU_used': [gpu_used], 'energy_consumed': [energy_consumed],
                                    'sample_size': [sample_size]})
        df = pd.concat([df, new_data], ignore_index=True)
        logger.info(f"new_data={new_data}")

        df.to_csv(data_results_file, encoding='utf-8', index=False)

    except Exception as e:
        logger.warning(e)


def save_results_csv(scenario_name: str, id: int, bytes_sent: int, bytes_recv: int, accuracy: float, loss: float):
    """
    try:
        data_results_file = os.path.join(os.environ.get('NEBULA_LOGS_DIR'), scenario_name, "trustworthiness", "data_results.csv")
    except:
        data_results_file = os.path.join("nebula", "app", "logs", scenario_name, "trustworthiness", "data_results.csv")

    if exists(data_results_file):
        df = pd.read_csv(data_results_file)
    else:
        df = pd.DataFrame(columns=["id", "bytes_sent", "bytes_recv", "accuracy", "loss"])

    try:
        # Add new entry to DataFrame
        new_data = pd.DataFrame({'id': [id], 'bytes_sent': [bytes_sent],
                                    'bytes_recv': [bytes_recv], 'accuracy': [accuracy],
                                    'loss': [loss]})
        df = pd.concat([df, new_data], ignore_index=True)
        logger.info(f"new_data={new_data}")

        df.to_csv(data_results_file, encoding='utf-8', index=False)

    except Exception as e:
        logger.warning(e)
    """

    try:
        data_results_id_file = os.path.join(os.environ.get('NEBULA_LOGS_DIR'), scenario_name, "trustworthiness", f"data_results_{id}.csv")
    except:
        data_results_id_file = os.path.join("nebula", "app", "logs", scenario_name, "trustworthiness", f"data_results_{id}.csv")

    if exists(data_results_id_file):
        df = pd.read_csv(data_results_id_file)
    else:
        df = pd.DataFrame(columns=["id", "bytes_sent", "bytes_recv", "accuracy", "loss"])

    try:
        # Add new entry to DataFrame
        new_data = pd.DataFrame({'id': [id], 'bytes_sent': [bytes_sent],
                                    'bytes_recv': [bytes_recv], 'accuracy': [accuracy],
                                    'loss': [loss]})
        df = pd.concat([df, new_data], ignore_index=True)
        logger.info(f"new_data={new_data}")

        df.to_csv(data_results_id_file, encoding='utf-8', index=False)

    except Exception as e:
        logger.warning(e)

def save_confirmation_csv(scenario_name: str, id: int):
    try:
        data_results_file = os.path.join(os.environ.get('NEBULA_LOGS_DIR'), scenario_name, "trustworthiness", "confirmation.csv")
    except:
        data_results_file = os.path.join("nebula", "app", "logs", scenario_name, "trustworthiness", "confirmation.csv")

    if exists(data_results_file):
        df = pd.read_csv(data_results_file)
    else:
        df = pd.DataFrame(columns=["id", "OK"])

    try:
        # Add new entry to DataFrame
        new_data = pd.DataFrame({'id': [id], 'OK': ["OK"]})
        df = pd.concat([df, new_data], ignore_index=True)
        logger.info(f"new_data={new_data}")

        df.to_csv(data_results_file, encoding='utf-8', index=False)

    except Exception as e:
        logger.warning(e)
