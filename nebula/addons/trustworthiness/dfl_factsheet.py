import logging
import json
import os
import shutil
import numpy as np
import pandas as pd

from nebula.addons.trustworthiness.calculation import (
    get_bytes_model,
    get_cv,
    get_dp_local,
    get_elapsed_time,
    get_underfitting_score_local,
)
from nebula.addons.trustworthiness.factsheet_common import (
    cap_score,
    populate_common_pre_train_sections,
    populate_model_quality_metrics,
    populate_participation,
    populate_reliability,
    populate_reputation,
    set_dp_configuration,
)
from nebula.addons.trustworthiness.utils import read_csv, get_all_data_entropy

dirname = os.path.dirname(__file__)
logger = logging.getLogger(__name__)


def populate_factsheet(
    experiment_name,
    participant_idx,
    data,
    start_time,
    end_time,
    model,
    train_loader,
    test_loader,
    reputation_summary=None,
    participation_summary=None,
    reliability_summary=None,
):
    trust_dir = os.path.join(os.environ.get("NEBULA_LOGS_DIR"), experiment_name, "trustworthiness")
    os.makedirs(trust_dir, exist_ok=True)

    factsheet_name = f"factsheet_participant_{participant_idx}.json"
    factsheet_path = os.path.join(trust_dir, factsheet_name)

    template_path = os.path.join(dirname, "configs", "factsheet_template_dfl.json")
    if not os.path.exists(factsheet_path):
        shutil.copyfile(template_path, factsheet_path)

    with open(factsheet_path, "r+", encoding="utf-8") as f:
        factsheet = {}
        factsheet = json.load(f)

        logging.info("DFL FactSheet: Populating factsheet")

        populate_common_pre_train_sections(factsheet, data, model)

        dp_enabled, dp_epsilon = get_dp_local(experiment_name, participant_idx)
        set_dp_configuration(factsheet, dp_enabled, dp_epsilon)

        files_dir = os.path.join(os.environ.get("NEBULA_LOGS_DIR"), experiment_name, "trustworthiness")

        emissions_file = os.path.join(files_dir, f"emissions_{participant_idx}.csv")

        get_all_data_entropy(experiment_name)

        data_class_count_file = os.path.join(
            os.environ.get('NEBULA_LOGS_DIR'),
            experiment_name,
            "trustworthiness",
            f"{str(participant_idx)}_class_count.json",
        )

        entropy_local = normalized_entropy_from_class_counts(data_class_count_file)

        factsheet["data"]["entropy_local"] = entropy_local

        df = load_round_metrics(experiment_name, participant_idx)
        acc = df["accuracy"].astype(float).to_numpy()
        loss = df["loss"].astype(float).to_numpy()

        final_acc = float(acc[-1])
        final_loss = float(loss[-1])

        factsheet["performance"]["test_loss"] = float(final_loss)
        factsheet["performance"]["test_acc"] = float(final_acc)

        bytes_sent, bytes_recv = get_bytes(experiment_name, participant_idx)

        factsheet["system"]["model_size"] = get_bytes_model(model)

        factsheet["system"]["upload_bytes"] = int(bytes_sent)
        factsheet["system"]["download_bytes"] = int(bytes_recv)

        populate_reliability(factsheet, reliability_summary)

        factsheet["system"]["time_minutes"] = get_elapsed_time(start_time, end_time)

        count_class_file = os.path.join(files_dir, f"{participant_idx}_class_count.json")
        if os.path.exists(count_class_file):
            with open(count_class_file, "r") as fs:
                class_distribution = json.load(fs)
            class_samples_sizes = list(class_distribution.values())
            class_imbalance = get_cv(list=class_samples_sizes)
            factsheet["fairness"]["class_imbalance"] = cap_score(class_imbalance)
        else:
            factsheet["fairness"]["class_imbalance"] = factsheet["fairness"].get("class_imbalance", 0.0)

        populate_participation(factsheet, participation_summary)

        carbon_intensity_local, emissions_training_local, energy_consumed_local, sample_size = get_emissions(
            emissions_file,
            participant_idx,
        )

        factsheet["sustainability"]["carbon_intensity_local"] = carbon_intensity_local
        factsheet["sustainability"]["emissions_training_local"] = emissions_training_local
        factsheet["sustainability"]["energy_consumed_local"] = energy_consumed_local
        factsheet["participants"]["local_dataset_size"] = sample_size

        populate_reputation(factsheet, reputation_summary, include_neighbor_num=True)

        factsheet["sustainability"]["emissions_communication_local"] = (
            (bytes_sent * 2.24e-10 * carbon_intensity_local)
            + (bytes_recv * 2.24e-10 * carbon_intensity_local)
        )

        factsheet["fairness"]["underfitting"] = get_underfitting_score_local(experiment_name, participant_idx)
        populate_model_quality_metrics(
            factsheet,
            model,
            train_loader,
            test_loader,
            factsheet["performance"]["test_acc"],
        )

        f.seek(0)
        f.truncate()
        json.dump(factsheet, f, indent=4)

def load_round_metrics(experiment_name, participant_idx):
    files_dir = os.path.join(os.environ.get("NEBULA_LOGS_DIR"), experiment_name, "trustworthiness")
    path = os.path.join(files_dir, f"round_metrics_participant_{participant_idx}.csv")
    df = pd.read_csv(path)

    if "round" in df.columns:
        df = df.sort_values("round")

    df = df.dropna(subset=["loss", "accuracy"])
    return df

def get_bytes(experiment_name, participant_idx):
    data_file = os.path.join(
        os.environ.get('NEBULA_LOGS_DIR'),
        experiment_name,
        "trustworthiness",
        f"data_results_{participant_idx}.csv",
    )

    data = read_csv(data_file)

    row = data[data["id"] == participant_idx]

    bytes_sent = row["bytes_sent"].iloc[0]
    bytes_recv = row["bytes_recv"].iloc[0]

    return bytes_sent, bytes_recv

def get_emissions(emissions_file, participant_idx):
    data = read_csv(emissions_file)

    row = data[data["id"] == participant_idx]

    avg_carbon_intensity_clients = row["energy_grid"].iloc[0]
    emissions_training = row["emissions"].iloc[0]
    energy_consumed = row["energy_consumed"].iloc[0]
    sample_size = row["sample_size"].iloc[0]

    return avg_carbon_intensity_clients, emissions_training, energy_consumed, sample_size

def normalized_entropy_from_class_counts(count_class_file):
    with open(count_class_file, "r") as f:
        dist = json.load(f)

    counts = np.array(list(dist.values()), dtype=float)
    total = counts.sum()
    if total <= 0:
        return 0.0

    p = counts / total

    eps = 1e-12
    H = -float(np.sum(p * np.log(p + eps)))

    K = len(p)
    if K <= 1:
        return 0.0

    H_norm = H / float(np.log(K))

    return max(0.0, min(1.0, H_norm))
