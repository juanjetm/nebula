from __future__ import annotations

import csv
import os
from typing import Any


def ensure_trust_dir(base_logs_dir: str, experiment_name: str) -> str:
    trust_dir = os.path.join(base_logs_dir, experiment_name, "trustworthiness")
    os.makedirs(trust_dir, exist_ok=True)
    return trust_dir


def append_trust_report_to_csv(
    base_logs_dir: str,
    experiment_name: str,
    report: dict[str, Any],
) -> None:
    """
    Escribe la información del reporte en:
      - data_results.csv
      - emissions.csv
    """

    trust_dir = ensure_trust_dir(base_logs_dir, experiment_name)

    data_results_path = os.path.join(trust_dir, "data_results.csv")
    emissions_path = os.path.join(trust_dir, "emissions.csv")

    _append_data_results(data_results_path, report)
    _append_emissions(emissions_path, report)


def _append_data_results(path: str, report: dict[str, Any]) -> None:
    exists = os.path.exists(path)

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "node_id",
                "round",
                "bytes_sent",
                "bytes_recv",
                "loss",
                "accuracy",
            ],
        )

        if not exists:
            writer.writeheader()

        writer.writerow(
            {
                "node_id": report["node_id"],
                "round": report["round"],
                "bytes_sent": report["bytes_sent"],
                "bytes_recv": report["bytes_recv"],
                "loss": report["loss"],
                "accuracy": report["accuracy"],
            }
        )


def _append_emissions(path: str, report: dict[str, Any]) -> None:
    exists = os.path.exists(path)

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "node_role",
                "node_id",
                "round",
                "workload",
                "sample_size",
                "emissions",
            ],
        )

        if not exists:
            writer.writeheader()

        writer.writerow(
            {
                "node_role": report["node_role"],
                "node_id": report["node_id"],
                "round": report["round"],
                "workload": report["workload"],
                "sample_size": report["sample_size"],
                "emissions": report["emissions"],
            }
        )
