import logging
import os
from json import JSONDecodeError
import numpy as np
import pandas as pd

from nebula.addons.trustworthiness.calculation import (
    get_avg_class_imbalance_model_size,
    get_avg_loss_accuracy,
    get_bytes_sent_recv,
    get_class_imbalance_score,
    get_cv,
    get_dp_global,
    get_elapsed_time,
    get_entropy_list,
    get_participant_loss_accuracy,
    get_underfitting_score,
)
from nebula.addons.trustworthiness.factsheet_common import (
    cap_score,
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
from nebula.addons.trustworthiness.utils import read_csv, check_field_filled
# from nebula.core.models.syscall.mlp import SyscallModelMLP

logger = logging.getLogger(__name__)

class CflFactsheet:
    def __init__(self):
        """
        Manager class to populate the FactSheet
        """
        self.factsheet_file_nm = "factsheet.json"
        self.factsheet_template_file_nm = "factsheet_template.json"

    def populate_factsheet_cfl(
        self,
        scenario_name,
        data,
        start_time,
        end_time,
        participant_idx,
        model,
        train_loader,
        test_loader,
        reputation_summary=None,
        participation_summary=None,
        reliability_summary=None,
    ):

        factsheet_file = get_factsheet_path(scenario_name, self.factsheet_file_nm)

        try:
            factsheet_file, factsheet = load_or_create_factsheet(
                scenario_name,
                self.factsheet_file_nm,
                self.factsheet_template_file_nm,
            )

            logging.info("FactSheet: Populating factsheet with pre training metrics")

            populate_common_pre_train_sections(factsheet, data, model)

            files_dir = get_trustworthiness_dir(scenario_name)

            emissions_file = os.path.join(files_dir, "emissions.csv")

            avg_class_imbalance, avg_model_size = get_avg_class_imbalance_model_size(scenario_name)
            entropy_distribution = get_entropy_list (scenario_name)

            values = np.array(entropy_distribution)

            normalized_values = (values - np.min(values)) / (np.max(values) - np.min(values))

            avg_entropy = np.mean(normalized_values)

            factsheet["data"]["avg_entropy"] = avg_entropy

            # Set performance data
            result_avg_loss_accuracy = get_avg_loss_accuracy(scenario_name)
            factsheet["performance"]["test_loss_avg"] = result_avg_loss_accuracy[0]
            factsheet["performance"]["test_acc_avg"] = result_avg_loss_accuracy[1]
            test_acc_cv = get_cv(std=result_avg_loss_accuracy[2], mean=result_avg_loss_accuracy[1])
            factsheet["fairness"]["test_acc_cv"] = 1 if test_acc_cv > 1 else test_acc_cv
            _, participant_test_acc = get_participant_loss_accuracy(scenario_name, participant_idx)

            dp_enabled, dp_epsilon = get_dp_global(scenario_name)
            set_dp_configuration(factsheet, dp_enabled, dp_epsilon)

            factsheet["system"]["avg_time_minutes"] = get_elapsed_time(start_time, end_time)
            factsheet["system"]["avg_model_size"] = avg_model_size

            result_bytes_sent_recv = get_bytes_sent_recv(scenario_name)
            factsheet["system"]["total_upload_bytes"] = result_bytes_sent_recv[0]
            factsheet["system"]["total_download_bytes"] = result_bytes_sent_recv[1]
            factsheet["system"]["avg_upload_bytes"] = result_bytes_sent_recv[2]
            factsheet["system"]["avg_download_bytes"] = result_bytes_sent_recv[3]
            populate_reliability(factsheet, reliability_summary)
            populate_participation(factsheet, participation_summary)

            class_imbalance_score = get_class_imbalance_score(avg_class_imbalance)
            factsheet["fairness"]["class_imbalance"] = cap_score(class_imbalance_score)
            populate_reputation(factsheet, reputation_summary)

            underfitting_score = get_underfitting_score(scenario_name, participant_idx)

            factsheet["fairness"]["underfitting"] = underfitting_score
            populate_model_quality_metrics(
                factsheet,
                model,
                train_loader,
                test_loader,
                participant_test_acc,
            )

            # Set emissions metrics
            emissions = None if emissions_file is None else read_csv(emissions_file)
            if emissions is not None:
                logging.info("FactSheet: Populating emissions")
                cpu_spez_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "benchmarks", "CPU_benchmarks_v4.csv"), header=0)
                emissions["CPU_model"] = emissions["CPU_model"].astype(str).str.replace(r"\([^)]*\)", "", regex=True)
                emissions["CPU_model"] = emissions["CPU_model"].astype(str).str.replace(r" CPU", "", regex=True)
                emissions["GPU_model"] = emissions["GPU_model"].astype(str).str.replace(r"[0-9] x ", "", regex=True)
                emissions = pd.merge(emissions, cpu_spez_df[["cpuName", "powerPerf"]], left_on="CPU_model", right_on="cpuName", how="left")
                gpu_spez_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "benchmarks", "GPU_benchmarks_v7.csv"), header=0)
                emissions = pd.merge(emissions, gpu_spez_df[["gpuName", "powerPerformance"]], left_on="GPU_model", right_on="gpuName", how="left")

                emissions.drop("cpuName", axis=1, inplace=True)
                emissions.drop("gpuName", axis=1, inplace=True)
                emissions["powerPerf"] = emissions["powerPerf"].astype(float)
                emissions["powerPerformance"] = emissions["powerPerformance"].astype(float)
                client_emissions = emissions.loc[emissions["role"] == "trainer"]
                client_avg_carbon_intensity = round(client_emissions["energy_grid"].mean(), 2)
                factsheet["sustainability"]["avg_carbon_intensity_clients"] = check_field_filled(factsheet, ["sustainability", "avg_carbon_intensity_clients"], client_avg_carbon_intensity, "")
                factsheet["sustainability"]["emissions_training"] = check_field_filled(factsheet, ["sustainability", "emissions_training"], client_emissions["emissions"].sum(), "")
                factsheet["participants"]["avg_dataset_size"] = check_field_filled(factsheet, ["participants", "avg_dataset_size"], client_emissions["sample_size"].mean(), "")
                GPU_powerperf = (client_emissions.loc[client_emissions["GPU_used"] == True])["powerPerformance"]
                CPU_powerperf = (client_emissions.loc[client_emissions["CPU_used"] == True])["powerPerf"]
                clients_power_performance = round(pd.concat([GPU_powerperf, CPU_powerperf]).mean(), 2)
                factsheet["sustainability"]["avg_power_performance_clients"] = check_field_filled(factsheet, ["sustainability", "avg_power_performance_clients"], clients_power_performance, "")

                server_emissions = emissions.loc[emissions["role"] == "server"]
                server_avg_carbon_intensity = round(server_emissions["energy_grid"].mean(), 2)
                factsheet["sustainability"]["avg_carbon_intensity_server"] = check_field_filled(factsheet, ["sustainability", "avg_carbon_intensity_server"], server_avg_carbon_intensity, "")
                factsheet["sustainability"]["emissions_aggregation"] = check_field_filled(factsheet, ["sustainability", "emissions_aggregation"], server_emissions["emissions"].sum(), "")
                GPU_powerperf = (server_emissions.loc[server_emissions["GPU_used"] == True])["powerPerformance"]
                CPU_powerperf = (server_emissions.loc[server_emissions["CPU_used"] == True])["powerPerf"]
                server_power_performance = round(pd.concat([GPU_powerperf, CPU_powerperf]).mean(), 2)
                factsheet["sustainability"]["avg_power_performance_server"] = check_field_filled(factsheet, ["sustainability", "avg_power_performance_server"], server_power_performance, "")

                factsheet["sustainability"]["emissions_communication_uplink"] = check_field_filled(factsheet, ["sustainability", "emissions_communication_uplink"], factsheet["system"]["total_upload_bytes"] * 2.24e-10 * factsheet["sustainability"]["avg_carbon_intensity_clients"], "")
                factsheet["sustainability"]["emissions_communication_downlink"] = check_field_filled(factsheet, ["sustainability", "emissions_communication_downlink"], factsheet["system"]["total_download_bytes"] * 2.24e-10 * factsheet["sustainability"]["avg_carbon_intensity_server"], "")

            write_factsheet(factsheet_file, factsheet)

        except JSONDecodeError as e:
            logging.info(f"{factsheet_file} is invalid")
            logging.error(e)
