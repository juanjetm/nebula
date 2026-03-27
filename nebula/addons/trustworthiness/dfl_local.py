# nebula/addons/trustworthiness/dfl_local.py
import json, os, shutil
from datetime import datetime
from nebula.addons.trustworthiness.metric import TrustMetricManager
import logging
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
from nebula.addons.trustworthiness.calculation import get_elapsed_time, get_bytes_models, get_bytes_sent_recv, get_avg_loss_accuracy, get_cv, get_clever_score, get_feature_importance_cv, get_loss_sensitivity_score, compute_adversarial_accuracy_art,get_empirical_robustness_score,get_confidence_score,attack_success_rate, get_bytes_model
from nebula.addons.trustworthiness.utils import count_all_class_samples, read_csv, check_field_filled, get_all_data_entropy

dirname = os.path.dirname(__file__)
logger = logging.getLogger(__name__)

def compute_trust_local_dfl(experiment_name, participant_idx, data, start_time, end_time):
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

        logging.info("DFL FactSheet: Populating factsheet with pre training metrics")

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

        else:
            background = f"This shouldn't be here xd"

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
        factsheet["participants"]["client_selector"] = ""

        # Set configuration
        factsheet["configuration"]["aggregation_algorithm"] = data["agg_algorithm"] or ""
        factsheet["configuration"]["training_model"] = data["model"] or ""
        factsheet["configuration"]["personalization"] = False
        factsheet["configuration"]["visualization"] = True
        factsheet["configuration"]["total_round_num"] = n_rounds

        if poisoned_noise_percent != 0:
            factsheet["configuration"]["differential_privacy"] = True
            factsheet["configuration"]["dp_epsilon"] = poisoned_noise_percent
        else:
            factsheet["configuration"]["differential_privacy"] = False
            factsheet["configuration"]["dp_epsilon"] = ""

        if dataset == "MNIST" and algorithm == "MLP":
            model = MNISTModelMLP()
            num_classes_temp = 10
        elif dataset == "MNIST" and algorithm == "CNN":
            model = MNISTModelCNN()
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

        factsheet["configuration"]["learning_rate"] = model.get_learning_rate()
        factsheet["configuration"]["trainable_param_num"] = model.count_parameters()
        factsheet["configuration"]["local_update_steps"] = 1

        files_dir = os.path.join(os.environ.get("NEBULA_LOGS_DIR"), experiment_name, "trustworthiness")

        train_model_file = os.path.join(files_dir, f"participant_{participant_idx}_final_model.pk")
        #train_model_file = os.path.join(files_dir, f"participant_{participant_idx}_train_model.pk")
        test_dataloader_file = os.path.join(files_dir, f"participant_{participant_idx}_test_loader.pk")
        emissions_file = os.path.join(files_dir, f"emissions_{participant_idx}.csv")

        with open(train_model_file, "rb") as t_file:
            lightning_model = pickle.load(t_file)

        get_all_data_entropy(experiment_name)

        data_class_count_file = os.path.join(os.environ.get('NEBULA_LOGS_DIR'), experiment_name, "trustworthiness", f"{str(participant_idx)}_class_count.json")

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

        model_file = os.path.join(files_dir, f"participant_{participant_idx}_final_model.pk")
        factsheet["system"]["model_size"] = get_bytes_model(model_file)

        factsheet["system"]["upload_bytes"] = int(bytes_sent)
        factsheet["system"]["download_bytes"] = int(bytes_recv)

        factsheet["system"]["time_minutes"] = get_elapsed_time(start_time, end_time)

        count_class_file = os.path.join(files_dir, f"{participant_idx}_class_count.json")
        if os.path.exists(count_class_file):
            with open(count_class_file, "r") as fs:
                class_distribution = json.load(fs)
            class_samples_sizes = list(class_distribution.values())
            class_imbalance = get_cv(list=class_samples_sizes)
            factsheet["fairness"]["class_imbalance"] = 1 if class_imbalance > 1 else class_imbalance
        else:
            factsheet["fairness"]["class_imbalance"] = factsheet["fairness"].get("class_imbalance", 0.0)

        carbon_intensity_local, emissions_training_local, energy_consumed_local, sample_size = get_emissions(emissions_file, participant_idx)

        factsheet["sustainability"]["carbon_intensity_local"] = carbon_intensity_local
        factsheet["sustainability"]["emissions_training_local"] = emissions_training_local
        factsheet["sustainability"]["energy_consumed_local"] = energy_consumed_local
        factsheet["participants"]["local_dataset_size"] = sample_size

        factsheet["sustainability"]["emissions_communication_local"] = (bytes_sent * 2.24e-10 * carbon_intensity_local)+(bytes_recv * 2.24e-10 * carbon_intensity_local)

        model.load_state_dict(lightning_model.state_dict())

        with open(test_dataloader_file, "rb") as d_file:
            test_dataloader = pickle.load(d_file)

        test_sample = next(iter(test_dataloader))

        lr = factsheet["configuration"]["learning_rate"]
        value_clever = get_clever_score(model, test_sample, num_classes_temp, lr)

        factsheet["performance"]["test_clever"] = 1 if value_clever > 1 else value_clever

        value_loss_sensitivity = get_loss_sensitivity_score(model, test_sample, num_classes_temp, lr)

        factsheet["performance"]["test_loss_sensitivity"] = 1 if value_loss_sensitivity > 1 else value_loss_sensitivity

        value_adv_accuracy = compute_adversarial_accuracy_art(model, test_dataloader, num_classes_temp, lr)

        factsheet["performance"]["test_adv_accuracy"] = 1 if value_adv_accuracy > 1 else value_adv_accuracy

        value_empirical_robustness = get_empirical_robustness_score(model, test_sample, num_classes_temp, lr)

        factsheet["performance"]["test_empirical_robustness"] = 1 if value_empirical_robustness > 1 else value_empirical_robustness

        value_confidence_score = get_confidence_score(model, test_sample)

        factsheet["performance"]["test_confidence_score"] = 1 if value_confidence_score > 1 else value_confidence_score
        attack_success_rate

        value_attack_success_rate = attack_success_rate(model, test_sample)

        factsheet["performance"]["test_attack_success_rate"] = 1 if value_attack_success_rate > 1 else value_attack_success_rate

        feature_importance = get_feature_importance_cv(model, test_sample)

        factsheet["performance"]["test_feature_importance_cv"] = 1 if feature_importance > 1 else feature_importance

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
    data_file = os.path.join(os.environ.get('NEBULA_LOGS_DIR'), experiment_name, "trustworthiness", f"data_results_{participant_idx}.csv")

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
