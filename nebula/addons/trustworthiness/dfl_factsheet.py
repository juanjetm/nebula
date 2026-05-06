import logging
import os
import pandas as pd

from nebula.addons.trustworthiness.calculation import (
    get_bytes_model,
    get_dp_local,
    get_elapsed_time,
    get_local_class_imbalance_score,
    get_local_normalized_entropy,
    get_underfitting_score_local,
)
from nebula.addons.trustworthiness.factsheet_common import (
    get_factsheet_path,
    get_trustworthiness_dir,
    load_or_create_factsheet,
    populate_common_pre_train_sections,
    populate_model_quality_metrics,
    populate_participation,
    populate_reliability,
    populate_reputation,
    set_dp_configuration,
    write_factsheet,
)
from nebula.addons.trustworthiness.utils import read_csv, get_all_data_entropy

logger = logging.getLogger(__name__)

class DflFactsheet:
    def __init__(self):
        """
        Manager class to populate the FactSheet
        """
        self.factsheet_template_file_nm = "factsheet_template_dfl.json"

    def populate_factsheet_dfl(
        self,
        scenario_name,
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

        self.factsheet_file_nm = f"factsheet_participant_{participant_idx}.json"

        factsheet_file = get_factsheet_path(scenario_name, self.factsheet_file_nm)

        factsheet_file, factsheet = load_or_create_factsheet(
            scenario_name,
            self.factsheet_file_nm,
            self.factsheet_template_file_nm,
        )

        logging.info("DFL FactSheet: Populating factsheet")

        populate_common_pre_train_sections(factsheet, data, model)

        dp_enabled, dp_epsilon = get_dp_local(scenario_name, participant_idx)
        set_dp_configuration(factsheet, dp_enabled, dp_epsilon)

        files_dir = get_trustworthiness_dir(scenario_name)

        emissions_file = os.path.join(files_dir, f"emissions_{participant_idx}.csv")

        get_all_data_entropy(scenario_name)

        factsheet["data"]["entropy_local"] = get_local_normalized_entropy(scenario_name, participant_idx)

        df = load_round_metrics(scenario_name, participant_idx)
        acc = df["accuracy"].astype(float).to_numpy()
        loss = df["loss"].astype(float).to_numpy()

        final_acc = float(acc[-1])
        final_loss = float(loss[-1])

        factsheet["performance"]["test_loss"] = float(final_loss)
        factsheet["performance"]["test_acc"] = float(final_acc)

        bytes_sent, bytes_recv = get_bytes(scenario_name, participant_idx)

        factsheet["system"]["model_size"] = get_bytes_model(model)

        factsheet["system"]["upload_bytes"] = int(bytes_sent)
        factsheet["system"]["download_bytes"] = int(bytes_recv)

        populate_reliability(factsheet, reliability_summary)

        factsheet["system"]["time_minutes"] = get_elapsed_time(start_time, end_time)

        count_class_file = os.path.join(files_dir, f"{participant_idx}_class_count.json")
        factsheet["fairness"]["class_imbalance"] = (
            get_local_class_imbalance_score(scenario_name, participant_idx)
            if os.path.exists(count_class_file)
            else factsheet["fairness"].get("class_imbalance", 0.0)
        )

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

        factsheet["fairness"]["underfitting"] = get_underfitting_score_local(scenario_name, participant_idx)
        populate_model_quality_metrics(
            factsheet,
            model,
            train_loader,
            test_loader,
            factsheet["performance"]["test_acc"],
        )

        write_factsheet(factsheet_file, factsheet)


def load_round_metrics(scenario_name, participant_idx):
    files_dir = get_trustworthiness_dir(scenario_name)
    path = os.path.join(files_dir, f"round_metrics_participant_{participant_idx}.csv")
    df = pd.read_csv(path)

    if "round" in df.columns:
        df = df.sort_values("round")

    df = df.dropna(subset=["loss", "accuracy"])
    return df


def get_bytes(scenario_name, participant_idx):
    data_file = os.path.join(
        get_trustworthiness_dir(scenario_name),
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
