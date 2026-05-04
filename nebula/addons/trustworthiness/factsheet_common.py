"""Shared helpers for trustworthiness factsheet generation."""

from nebula.addons.trustworthiness.calculation import (
    attack_success_rate,
    compute_adversarial_accuracy_art,
    get_clever_score,
    get_coefficient_of_variation,
    get_confidence_score,
    get_empirical_robustness_score,
    get_epsilon_star,
    get_explainability_metrics_summary,
    get_generalized_entropy_index,
    get_loss_sensitivity_score,
    get_macro_f1_score,
    get_mia_auc,
    get_overfitting_score,
    get_theil_index,
    get_well_calibration_error,
)


def cap_score(value, maximum=1):
    """Caps a score to the maximum value expected by the factsheet."""
    return maximum if value > maximum else value


def inverse_score(value):
    """Converts an error or risk value into a bounded inverse score."""
    return 1 / (1 + value)


def build_project_background(data):
    """Builds the natural-language scenario description used in factsheets."""
    federation = data["federation"]
    n_nodes = int(data["n_nodes"])
    dataset = data["dataset"]
    algorithm = data["model"]
    aggregation_algorithm = data["agg_algorithm"]
    n_rounds = int(data["rounds"])
    attack = data["attack_params"]["attacks"]
    with_reputation = data["reputation"]["enabled"]

    base = (
        "For the project setup, the most important aspects are the following: "
        f"The federation architecture is {federation}, involving {n_nodes} clients, "
        f"the dataset used is {dataset}, the learning algorithm is {algorithm}, "
        f"the aggregation algorithm is {aggregation_algorithm} and the number of rounds is {n_rounds}. "
    )

    if attack != "No Attack":
        attack_text = f"In addition, the type of attack used is {attack}. "
    else:
        attack_text = "No attacks are used. "

    if with_reputation:
        defence_text = "A reputation-based defence is used, and the trustworthiness of the project is desired."
    else:
        defence_text = "No defence mechanism is used, and the trustworthiness of the project is desired."

    return base + attack_text + defence_text


def populate_common_pre_train_sections(factsheet, data, model):
    """Populates project, data, participant and training configuration fields."""
    with_reputation = data["reputation"]["enabled"]

    factsheet["project"]["overview"] = data["scenario_title"]
    factsheet["project"]["purpose"] = data["scenario_description"]
    factsheet["project"]["background"] = build_project_background(data)

    factsheet["data"]["provenance"] = data["dataset"]
    factsheet["data"]["preprocessing"] = data["topology"]

    factsheet["participants"]["client_num"] = data["n_nodes"] or ""
    factsheet["participants"]["sample_client_rate"] = 1
    factsheet["participants"]["client_selector"] = (
        "Reputation Based" if with_reputation else "Full Participation"
    )

    factsheet["configuration"]["aggregation_algorithm"] = data["agg_algorithm"] or ""
    factsheet["configuration"]["training_model"] = data["model"] or ""
    factsheet["configuration"]["personalization"] = False
    factsheet["configuration"]["reputation_enabled"] = bool(
        data.get("reputation", {}).get("enabled", False)
    )
    factsheet["configuration"]["visualization"] = True
    factsheet["configuration"]["monitoring"] = True
    factsheet["configuration"]["total_round_num"] = int(data["rounds"])
    factsheet["configuration"]["learning_rate"] = model.get_learning_rate()
    factsheet["configuration"]["trainable_param_num"] = model.count_parameters()
    factsheet["configuration"]["local_update_steps"] = data["epochs"]


def set_dp_configuration(factsheet, dp_enabled, dp_epsilon):
    """Writes differential privacy configuration using the factsheet schema."""
    factsheet["configuration"]["differential_privacy"] = bool(dp_enabled)
    factsheet["configuration"]["dp_epsilon"] = dp_epsilon if dp_enabled else ""


def populate_reliability(factsheet, reliability_summary):
    """Writes dropout and timeout rates, defaulting to a fully reliable run."""
    factsheet["system"]["dropout_rate"] = (
        reliability_summary.get("dropout_rate", 0.0)
        if reliability_summary is not None
        else 0.0
    )
    factsheet["system"]["timeout_rate"] = (
        reliability_summary.get("timeout_rate", 0.0)
        if reliability_summary is not None
        else 0.0
    )


def populate_participation(factsheet, participation_summary):
    """Writes participant selection dispersion, defaulting to full participation."""
    factsheet["fairness"]["selection_cv"] = (
        participation_summary.get("selection_cv", 1)
        if participation_summary is not None
        else 1
    )


def populate_reputation(factsheet, reputation_summary, include_neighbor_num=False):
    """Writes reputation information for centralized or decentralized factsheets."""
    if reputation_summary is not None:
        factsheet["participants"]["avg_neighbor_reputation"] = reputation_summary.get(
            "avg_neighbor_reputation",
            "",
        )
        if include_neighbor_num:
            factsheet["participants"]["neighbor_num"] = reputation_summary.get(
                "neighbor_num",
                0,
            )
        return

    factsheet["participants"]["avg_neighbor_reputation"] = 0
    if include_neighbor_num:
        factsheet["participants"]["neighbor_num"] = 0


def populate_model_quality_metrics(factsheet, model, train_loader, test_loader, test_accuracy):
    """Calculates common privacy, fairness, explainability and robustness metrics."""
    test_sample = next(iter(test_loader))
    explainability_metrics = get_explainability_metrics_summary(model, test_loader)

    factsheet["performance"]["test_macro_f1"] = get_macro_f1_score(model, test_loader)

    factsheet["privacy"]["epsilon_star"] = get_epsilon_star(model, train_loader, test_loader)
    factsheet["privacy"]["epsilon_star_score"] = inverse_score(factsheet["privacy"]["epsilon_star"])
    factsheet["privacy"]["mia_auc"] = get_mia_auc(model, train_loader, test_loader)
    factsheet["privacy"]["mia_auc_score"] = 1 - 2 * abs(factsheet["privacy"]["mia_auc"] - 0.5)

    overfitting_value = get_overfitting_score(model, train_loader, test_accuracy)
    factsheet["fairness"]["overfitting"] = inverse_score(overfitting_value)

    well_calibration_error_value = get_well_calibration_error(model, test_loader)
    factsheet["fairness"]["well_calibration_error"] = inverse_score(well_calibration_error_value)

    generalized_entropy_index_value = get_generalized_entropy_index(model, test_loader)
    factsheet["fairness"]["generalized_entropy_index"] = inverse_score(generalized_entropy_index_value)

    theil_index_value = get_theil_index(model, test_loader)
    factsheet["fairness"]["theil_index"] = inverse_score(theil_index_value)

    coefficient_of_variation_value = get_coefficient_of_variation(model, test_loader)
    factsheet["fairness"]["coefficient_of_variation"] = inverse_score(coefficient_of_variation_value)

    factsheet["explainability"]["alpha_score"] = explainability_metrics["alpha_score"]
    factsheet["explainability"]["spread_ratio"] = explainability_metrics["spread_ratio"]
    factsheet["explainability"]["spread_divergence"] = explainability_metrics["spread_divergence"]

    lr = factsheet["configuration"]["learning_rate"]
    num_classes = model.get_num_classes()

    value_clever = get_clever_score(model, test_sample, num_classes, lr)
    factsheet["performance"]["test_clever"] = cap_score(value_clever)

    value_loss_sensitivity = get_loss_sensitivity_score(model, test_sample, num_classes, lr)
    factsheet["performance"]["test_loss_sensitivity"] = inverse_score(value_loss_sensitivity)

    value_adv_accuracy = compute_adversarial_accuracy_art(model, test_loader, num_classes, lr)
    factsheet["performance"]["test_adv_accuracy"] = cap_score(value_adv_accuracy)

    value_empirical_robustness = get_empirical_robustness_score(model, test_sample, num_classes, lr)
    factsheet["performance"]["test_empirical_robustness"] = cap_score(value_empirical_robustness)

    value_confidence_score = get_confidence_score(model, test_sample)
    factsheet["performance"]["test_confidence_score"] = cap_score(value_confidence_score)

    value_attack_success_rate = attack_success_rate(model, test_sample)
    factsheet["performance"]["test_attack_success_rate"] = 1 - value_attack_success_rate

    feature_importance = explainability_metrics["feature_importance_cv"]
    factsheet["performance"]["test_feature_importance_cv"] = cap_score(feature_importance)
