import json
import logging
import os
import glob
import shutil
from json import JSONDecodeError
import pickle
import numpy as np
import pandas as pd
import time

# from nebula.core.models.cifar10.cnn import CIFAR10ModelCNN
from nebula.core.models.mnist.mlp import MNISTModelMLP
from nebula.core.models.mnist.cnn import MNISTModelCNN
from nebula.core.models.covtype.mlp import CovtypeModelMLP
from nebula.core.models.kddcup99.mlp import KDDCUP99ModelMLP
from nebula.core.models.adultcensus.mlp import AdultCensusModelMLP
from nebula.core.models.breast_cancer.mlp import BreastCancerModelMLP
from nebula.core.models.fashionmnist.mlp import FashionMNISTModelMLP
from nebula.core.models.fashionmnist.cnn import FashionMNISTModelCNN
from nebula.core.models.emnist.mlp import EMNISTModelMLP
from nebula.core.models.emnist.cnn import EMNISTModelCNN
from nebula.core.models.cifar10.cnn import CIFAR10ModelCNN
from nebula.core.models.cifar10.cnnV2 import CIFAR10ModelCNN_V2
from nebula.core.models.cifar10.cnnV3 import CIFAR10ModelCNN_V3
from nebula.core.models.cifar10.fastermobilenet import FasterMobileNet
from nebula.core.models.cifar10.resnet import CIFAR10ModelResNet
from nebula.core.models.cifar10.simplemobilenet import SimpleMobileNetV1
from nebula.core.models.cifar100.cnn import CIFAR100ModelCNN
from nebula.addons.trustworthiness.calculation import get_elapsed_time, get_bytes_models, get_bytes_sent_recv, get_avg_loss_accuracy, get_cv, get_clever_score, get_feature_importance_cv, get_loss_sensitivity_score, compute_adversarial_accuracy_art,get_empirical_robustness_score,get_confidence_score,attack_success_rate, get_entropy_list, get_avg_class_imbalance_model_size, get_underfitting_score, get_overfitting_score, get_participant_loss_accuracy, get_well_calibration_error, get_generalized_entropy_index, get_theil_index, get_coefficient_of_variation, get_alpha_score, get_spread_ratio, get_spread_divergence, get_epsilon_star, get_mia_auc, get_explainability_metrics_summary, get_macro_f1_score, get_dp_global
from nebula.addons.trustworthiness.utils import count_all_class_samples, read_csv, check_field_filled, get_all_data_entropy
# from nebula.core.models.syscall.mlp import SyscallModelMLP

dirname = os.path.dirname(__file__)
logger = logging.getLogger(__name__)

class Factsheet:
    def __init__(self):
        """
        Manager class to populate the FactSheet
        """
        self.factsheet_file_nm = "factsheet.json"
        self.factsheet_template_file_nm = "factsheet_template.json"

    def populate_factsheet_pre_train(self, data, scenario_name):
        """
        Populates the factsheet with values before the training.

        Args:
            data (dict): Contains the data from the scenario.
            scenario_name (string): The name of the scenario.
        """

        factsheet_file = os.path.join(os.environ.get('NEBULA_LOGS_DIR'), scenario_name, "trustworthiness", self.factsheet_file_nm)

        factsheet_template = os.path.join(dirname, "configs", self.factsheet_template_file_nm)

        if not os.path.exists(factsheet_file):
            shutil.copyfile(factsheet_template, factsheet_file)

        with open(factsheet_file, "r+") as f:
            factsheet = {}

            try:
                factsheet = json.load(f)

                if data is not None:
                    logging.info("FactSheet: Populating factsheet with pre training metrics")

                    federation = data["federation"]
                    n_nodes = int(data["n_nodes"])
                    dataset = data["dataset"]
                    algorithm = data["model"]
                    aggregation_algorithm = data["agg_algorithm"]
                    n_rounds = int(data["rounds"])
                    attack = data["attack_params"]["attacks"]

                    attack_params = data.get("attack_params", {})

                    poisoned_node_percent = int(attack_params.get("poisoned_node_percent", 0) or 0)
                    poisoned_sample_percent = int(attack_params.get("poisoned_sample_percent", 0) or 0)
                    poisoned_noise_percent = float(attack_params.get("poisoned_noise_percent", 0) or 0)

                    with_reputation = data["reputation"]["enabled"]
                    topology = data["topology"]

                    if attack != "No Attack" and with_reputation == True:
                        background = f"For the project setup, the most important aspects are the following: The federation architecture is {federation}, involving {n_nodes} clients, the dataset used is {dataset}, the learning algorithm is {algorithm}, the aggregation algorithm is {aggregation_algorithm} and the number of rounds is {n_rounds}. In addition, the type of attack used is {attack}. A reputation-based defence is used, and the trustworthiness of the project is desired."

                    elif attack != "No Attack" and with_reputation == False:
                        background = f"For the project setup, the most important aspects are the following: The federation architecture is {federation}, involving {n_nodes} clients, the dataset used is {dataset}, the learning algorithm is {algorithm}, the aggregation algorithm is {aggregation_algorithm} and the number of rounds is {n_rounds}. In addition, the type of attack used is {attack}. No defence mechanism is used, and the trustworthiness of the project is desired."

                    elif attack == "No Attack" and with_reputation == True:
                        background = f"For the project setup, the most important aspects are the following: The federation architecture is {federation}, involving {n_nodes} clients, the dataset used is {dataset}, the learning algorithm is {algorithm}, the aggregation algorithm is {aggregation_algorithm} and the number of rounds is {n_rounds}. No attacks are used. A reputation-based defence is used, and the trustworthiness of the project is desired."

                    elif attack == "No Attack" and with_reputation == False:
                        background = f"For the project setup, the most important aspects are the following: The federation architecture is {federation}, involving {n_nodes} clients, the dataset used is {dataset}, the learning algorithm is {algorithm}, the aggregation algorithm is {aggregation_algorithm} and the number of rounds is {n_rounds}. No attacks are used. No defence mechanism is used, and the trustworthiness of the project is desired."

                    # Set project specifications
                    factsheet["project"]["overview"] = data["scenario_title"]
                    factsheet["project"]["purpose"] = data["scenario_description"]
                    factsheet["project"]["background"] = background

                    # Set data specifications
                    factsheet["data"]["provenance"] = data["dataset"]
                    factsheet["data"]["preprocessing"] = data["topology"]

                    # Set participants
                    factsheet["participants"]["client_num"] = data["n_nodes"] or ""
                    factsheet["participants"]["sample_client_rate"] = 1
                    if with_reputation == True:
                        factsheet["participants"]["client_selector"] = "Reputation Based"
                    else:
                        factsheet["participants"]["client_selector"] = "Full Participation"

                    # Set configuration
                    factsheet["configuration"]["aggregation_algorithm"] = data["agg_algorithm"] or ""
                    factsheet["configuration"]["training_model"] = data["model"] or ""
                    factsheet["configuration"]["personalization"] = False
                    factsheet["configuration"]["reputation_enabled"] = bool(data.get("reputation", {}).get("enabled", False))
                    factsheet["configuration"]["visualization"] = True
                    factsheet["configuration"]["monitoring"] = True
                    factsheet["configuration"]["total_round_num"] = n_rounds

                    """
                    if poisoned_noise_percent != 0:
                        factsheet["configuration"]["differential_privacy"] = True
                        factsheet["configuration"]["dp_epsilon"] = poisoned_noise_percent
                    else:
                        factsheet["configuration"]["differential_privacy"] = False
                        factsheet["configuration"]["dp_epsilon"] = ""
                    """

                    factsheet["configuration"]["differential_privacy"] = False
                    factsheet["configuration"]["dp_epsilon"] = ""

                    if dataset == "MNIST" and algorithm == "MLP":
                        model = MNISTModelMLP()
                        num_classes_temp = 10
                    elif dataset == "MNIST" and algorithm == "CNN":
                        model = MNISTModelCNN()
                        num_classes_temp = 10
                    elif dataset == "FashionMNIST" and algorithm == "MLP":
                        model = FashionMNISTModelMLP()
                        num_classes_temp = 10
                    elif dataset == "FashionMNIST" and algorithm == "CNN":
                        model = FashionMNISTModelCNN()
                        num_classes_temp = 10
                    elif dataset == "Covtype" and algorithm == "MLP":
                        model = CovtypeModelMLP()
                        num_classes_temp = 7
                    elif dataset == "KDDCUP99" and algorithm == "MLP":
                        model = KDDCUP99ModelMLP()
                        num_classes_temp = 23
                    elif dataset == "AdultCensus" and algorithm == "MLP":
                        model = AdultCensusModelMLP()
                        num_classes_temp = 2
                    elif dataset == "BreastCancer" and algorithm == "MLP":
                        model = BreastCancerModelMLP()
                        num_classes_temp = 2
                    elif dataset == "EMNIST" and algorithm == "MLP":
                        model = EMNISTModelMLP()
                        num_classes_temp = 47
                    elif dataset == "EMNIST" and algorithm == "CNN":
                        model = EMNISTModelCNN()
                        num_classes_temp = 47
                    elif dataset == "CIFAR10" and algorithm == "ResNet9":
                        model = CIFAR10ModelResNet(classifier="resnet9")
                        num_classes_temp = 10
                    elif dataset == "CIFAR10" and algorithm == "fastermobilenet":
                        model = FasterMobileNet()
                        num_classes_temp = 10
                    elif dataset == "CIFAR10" and algorithm == "simplemobilenet":
                        model = SimpleMobileNetV1()
                        num_classes_temp = 10
                    elif dataset == "CIFAR10" and algorithm == "CNN":
                        model = CIFAR10ModelCNN()
                        num_classes_temp = 10
                    elif dataset == "CIFAR10" and algorithm == "CNNv2":
                        model = CIFAR10ModelCNN_V2()
                        num_classes_temp = 10
                    elif dataset == "CIFAR10" and algorithm == "CNNv3":
                        model = CIFAR10ModelCNN_V3()
                        num_classes_temp = 10
                    elif dataset == "CIFAR100" and algorithm == "CNN":
                        model = CIFAR100ModelCNN()
                        num_classes_temp = 100

                    factsheet["configuration"]["learning_rate"] = model.get_learning_rate()
                    factsheet["configuration"]["trainable_param_num"] = model.count_parameters()
                    factsheet["configuration"]["local_update_steps"] = 1

                    f.seek(0)
                    f.truncate()
                    json.dump(factsheet, f, indent=4)

            except JSONDecodeError as e:
                logging.warning(f"{factsheet_file} is invalid")
                logging.error(e)

    def populate_factsheet_post_train(self, scenario_name, start_time, end_time, participant_idx, reputation_summary=None, participation_summary=None, reliability_summary=None):
        """
        Populates the factsheet with values after the training.

        Args:
            scenario (object): The scenario object.
        """
        factsheet_file = os.path.join(f"{os.environ.get('NEBULA_LOGS_DIR')}{scenario_name}/trustworthiness/{self.factsheet_file_nm}")

        logging.info("FactSheet: Populating factsheet with post training metrics")

        with open(factsheet_file, "r+") as f:
            factsheet = {}
            try:
                factsheet = json.load(f)

                dataset = factsheet["data"]["provenance"]
                model = factsheet["configuration"]["training_model"]

                files_dir = f"{os.environ.get('NEBULA_LOGS_DIR')}/{scenario_name}/trustworthiness"

                train_dataloader_file = f"{files_dir}/participant_{participant_idx}_train_loader.pk"
                test_dataloader_file = f"{files_dir}/participant_{participant_idx}_test_loader.pk"
                final_model_file = f"{files_dir}/participant_{participant_idx}_final_model.pk"
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
                if dp_enabled:
                    factsheet["configuration"]["differential_privacy"] = True
                    factsheet["configuration"]["dp_epsilon"] = dp_epsilon
                else:
                    factsheet["configuration"]["differential_privacy"] = False
                    factsheet["configuration"]["dp_epsilon"] = ""

                factsheet["system"]["avg_time_minutes"] = get_elapsed_time(start_time, end_time)
                factsheet["system"]["avg_model_size"] = avg_model_size

                result_bytes_sent_recv = get_bytes_sent_recv(scenario_name)
                factsheet["system"]["total_upload_bytes"] = result_bytes_sent_recv[0]
                factsheet["system"]["total_download_bytes"] = result_bytes_sent_recv[1]
                factsheet["system"]["avg_upload_bytes"] = result_bytes_sent_recv[2]
                factsheet["system"]["avg_download_bytes"] = result_bytes_sent_recv[3]
                if reliability_summary is not None:
                    factsheet["system"]["dropout_rate"] = reliability_summary.get("dropout_rate", 0.0)
                    factsheet["system"]["timeout_rate"] = reliability_summary.get("timeout_rate", 0.0)
                else:
                    factsheet["system"]["dropout_rate"] = 0.0
                    factsheet["system"]["timeout_rate"] = 0.0

                if participation_summary is not None:
                    factsheet["fairness"]["selection_cv"] = participation_summary.get("selection_cv", 1)
                else:
                    factsheet["fairness"]["selection_cv"] = 1

                class_imbalance_score = 1 / (1+avg_class_imbalance)
                factsheet["fairness"]["class_imbalance"] = 1 if class_imbalance_score > 1 else class_imbalance_score
                if reputation_summary is not None:
                    factsheet["participants"]["avg_neighbor_reputation"] = reputation_summary.get("avg_neighbor_reputation", "")
                else:
                    factsheet["participants"]["avg_neighbor_reputation"] = 0

                with open(final_model_file, "rb") as file:
                    lightning_model = pickle.load(file)

                if dataset == "MNIST" and model == "MLP":
                    model = MNISTModelMLP()
                    num_classes_temp = 10
                elif dataset == "MNIST" and model == "CNN":
                    model = MNISTModelCNN()
                    num_classes_temp = 10
                elif dataset == "FashionMNIST" and model == "MLP":
                    model = FashionMNISTModelMLP()
                    num_classes_temp = 10
                elif dataset == "FashionMNIST" and model == "CNN":
                    model = FashionMNISTModelCNN()
                    num_classes_temp = 10
                elif dataset == "Covtype" and model == "MLP":
                    model = CovtypeModelMLP()
                    num_classes_temp = 7
                elif dataset == "KDDCUP99" and model == "MLP":
                    model = KDDCUP99ModelMLP()
                    num_classes_temp = 23
                elif dataset == "AdultCensus" and model == "MLP":
                    model = AdultCensusModelMLP()
                    num_classes_temp = 2
                elif dataset == "BreastCancer" and model == "MLP":
                    model = BreastCancerModelMLP()
                    num_classes_temp = 2
                elif dataset == "EMNIST" and model == "MLP":
                    model = EMNISTModelMLP()
                    num_classes_temp = 47
                elif dataset == "EMNIST" and model == "CNN":
                    model = EMNISTModelCNN()
                    num_classes_temp = 47
                elif dataset == "CIFAR10" and model == "ResNet9":
                    model = CIFAR10ModelResNet(classifier="resnet9")
                    num_classes_temp = 10
                elif dataset == "CIFAR10" and model == "fastermobilenet":
                    model = FasterMobileNet()
                    num_classes_temp = 10
                elif dataset == "CIFAR10" and model == "simplemobilenet":
                    model = SimpleMobileNetV1()
                    num_classes_temp = 10
                elif dataset == "CIFAR10" and model == "CNN":
                    model = CIFAR10ModelCNN()
                    num_classes_temp = 10
                elif dataset == "CIFAR10" and model == "CNNv2":
                    model = CIFAR10ModelCNN_V2()
                    num_classes_temp = 10
                elif dataset == "CIFAR10" and model == "CNNv3":
                    model = CIFAR10ModelCNN_V3()
                    num_classes_temp = 10
                elif dataset == "CIFAR100" and model == "CNN":
                    model = CIFAR100ModelCNN()
                    num_classes_temp = 100

                model.load_state_dict(lightning_model.state_dict())

                with open(train_dataloader_file, "rb") as file:
                    train_dataloader = pickle.load(file)

                with open(test_dataloader_file, "rb") as file:
                    test_dataloader = pickle.load(file)

                test_sample = next(iter(test_dataloader))
                explainability_metrics = get_explainability_metrics_summary(model, test_dataloader)
                factsheet["performance"]["test_macro_f1"] = get_macro_f1_score(model, test_dataloader)
                factsheet["privacy"]["epsilon_star"] = get_epsilon_star(
                    model,
                    train_dataloader,
                    test_dataloader,
                )
                factsheet["privacy"]["epsilon_star_score"] = 1/(1 + factsheet["privacy"]["epsilon_star"])
                factsheet["privacy"]["mia_auc"] = get_mia_auc(
                    model,
                    train_dataloader,
                    test_dataloader,
                )
                factsheet["privacy"]["mia_auc_score"] = 1 - 2 * abs(factsheet["privacy"]["mia_auc"] - 0.5)

                underfitting_score = get_underfitting_score(scenario_name, participant_idx)

                factsheet["fairness"]["underfitting"] = underfitting_score
                overfitting_value = get_overfitting_score(
                    model,
                    train_dataloader,
                    participant_test_acc,
                )
                factsheet["fairness"]["overfitting"] = 1/(1 + overfitting_value)
                well_calibration_error_value = get_well_calibration_error(
                    model,
                    test_dataloader,
                )

                factsheet["fairness"]["well_calibration_error"] = 1/(1 + well_calibration_error_value)
                generalized_entropy_index_value = get_generalized_entropy_index(
                    model,
                    test_dataloader,
                )
                factsheet["fairness"]["generalized_entropy_index"] = 1/(1 + generalized_entropy_index_value)
                theil_index_value = get_theil_index(
                    model,
                    test_dataloader,
                )
                factsheet["fairness"]["theil_index"] = 1/(1 + theil_index_value)
                coefficient_of_variation_value = get_coefficient_of_variation(
                    model,
                    test_dataloader,
                )
                factsheet["fairness"]["coefficient_of_variation"] = 1/(1 + coefficient_of_variation_value)
                factsheet["explainability"]["alpha_score"] = explainability_metrics["alpha_score"]
                factsheet["explainability"]["spread_ratio"] = explainability_metrics["spread_ratio"]
                factsheet["explainability"]["spread_divergence"] = explainability_metrics["spread_divergence"]

                lr = factsheet["configuration"]["learning_rate"]

                value_clever = get_clever_score(model, test_sample, num_classes_temp, lr)
                factsheet["performance"]["test_clever"] = 1 if value_clever > 1 else value_clever

                value_loss_sensitivity = get_loss_sensitivity_score(model, test_sample, num_classes_temp, lr)
                factsheet["performance"]["test_loss_sensitivity"] = 1 / (1 + value_loss_sensitivity)

                value_adv_accuracy = compute_adversarial_accuracy_art(model, test_dataloader, num_classes_temp, lr)
                factsheet["performance"]["test_adv_accuracy"] = 1 if value_adv_accuracy > 1 else value_adv_accuracy

                value_empirical_robustness = get_empirical_robustness_score(model, test_sample, num_classes_temp, lr)
                factsheet["performance"]["test_empirical_robustness"] = 1 if value_empirical_robustness > 1 else value_empirical_robustness

                value_confidence_score = get_confidence_score(model, test_sample)
                factsheet["performance"]["test_confidence_score"] = 1 if value_confidence_score > 1 else value_confidence_score

                value_attack_success_rate = attack_success_rate(model, test_sample)
                factsheet["performance"]["test_attack_success_rate"] = 1 - value_attack_success_rate

                feature_importance = explainability_metrics["feature_importance_cv"]
                factsheet["performance"]["test_feature_importance_cv"] = 1 if feature_importance > 1 else feature_importance

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

                f.seek(0)
                f.truncate()
                json.dump(factsheet, f, indent=4)

            except JSONDecodeError as e:
                logging.info(f"{factsheet_file} is invalid")
                logging.error(e)
