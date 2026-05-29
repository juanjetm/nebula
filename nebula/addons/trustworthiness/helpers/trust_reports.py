import copy
import json
import os

def load_trust_report_json_dumped(scenario_name: str, participant_id: int) -> str:
    """
    Read a participant trustworthiness JSON file and return it
    serialized as a string with json.dumps(...).

    Args:
        scenario_name (str): Scenario/experiment name.
        participant_id (int): Participant ID.

    Returns:
        str: JSON content serialized as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file content is not valid JSON.
    """
    logs_dir = os.environ.get("NEBULA_LOGS_DIR")
    if not logs_dir:
        raise ValueError("The NEBULA_LOGS_DIR environment variable is not defined.")

    file_name = f"nebula_trust_results_{participant_id}.json"
    file_path = os.path.join(
        logs_dir,
        scenario_name,
        "trustworthiness",
        file_name,
    )

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file does not exist: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            trust_report = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"The file does not contain valid JSON: {file_path}") from e

    return json.dumps(trust_report)


def load_trust_report_json(scenario_name: str, participant_id: int | str) -> dict:
    trust_report_json = load_trust_report_json_dumped(scenario_name, participant_id)
    return json.loads(trust_report_json)


def create_local_trust_report_copy(scenario_name: str, participant_id: int | str, suffix: str = "global") -> tuple[dict, str]:
    trust_report = load_trust_report_json(scenario_name, participant_id)
    logs_dir = os.environ.get("NEBULA_LOGS_DIR")
    if not logs_dir:
        raise ValueError("The NEBULA_LOGS_DIR environment variable is not defined.")

    trust_dir = os.path.join(logs_dir, scenario_name, "trustworthiness")
    os.makedirs(trust_dir, exist_ok=True)

    file_path = os.path.join(trust_dir, f"nebula_trust_results_{participant_id}_{suffix}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(trust_report, f, indent=4)

    return trust_report, file_path


def save_trust_report_json(file_path: str, trust_report: dict) -> str:
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(trust_report, f, indent=4)

    return file_path


def accumulate_weighted_trustscores(report: dict, weight: float, score_accumulator: dict, weight_accumulator: dict):
    if weight <= 0:
        raise ValueError("The aggregation weight must be greater than 0.")

    _accumulate_weighted_trustscores_recursive(
        obj=report,
        weight=float(weight),
        path=(),
        score_accumulator=score_accumulator,
        weight_accumulator=weight_accumulator,
    )


def build_weighted_trustscores_report(template_report: dict, score_accumulator: dict, weight_accumulator: dict) -> dict:
    aggregated_report = copy.deepcopy(template_report)
    _apply_weighted_trustscores_recursive(
        obj=aggregated_report,
        path=(),
        score_accumulator=score_accumulator,
        weight_accumulator=weight_accumulator,
    )
    return aggregated_report


def _accumulate_weighted_trustscores_recursive(obj, weight: float, path: tuple, score_accumulator: dict, weight_accumulator: dict):
    if isinstance(obj, dict):
        structural_named_entry = _get_structural_named_entry(obj)
        if structural_named_entry is not None:
            _, nested_value = structural_named_entry
            _accumulate_weighted_trustscores_recursive(
                obj=nested_value,
                weight=weight,
                path=path + ("__named_entry__",),
                score_accumulator=score_accumulator,
                weight_accumulator=weight_accumulator,
            )
            return

        for key, value in obj.items():
            if key in {"trust_score", "score"} and _is_numeric_score(value):
                score_path = path + (key,)
                score_accumulator[score_path] = score_accumulator.get(score_path, 0.0) + (float(value) * weight)
                weight_accumulator[score_path] = weight_accumulator.get(score_path, 0.0) + weight
                continue

            _accumulate_weighted_trustscores_recursive(
                obj=value,
                weight=weight,
                path=path + (key,),
                score_accumulator=score_accumulator,
                weight_accumulator=weight_accumulator,
            )
        return

    if isinstance(obj, list):
        for index, item in enumerate(obj):
            _accumulate_weighted_trustscores_recursive(
                obj=item,
                weight=weight,
                path=path + (index,),
                score_accumulator=score_accumulator,
                weight_accumulator=weight_accumulator,
            )


def _apply_weighted_trustscores_recursive(obj, path: tuple, score_accumulator: dict, weight_accumulator: dict):
    if isinstance(obj, dict):
        structural_named_entry = _get_structural_named_entry(obj)
        if structural_named_entry is not None:
            entry_key, nested_value = structural_named_entry
            obj[entry_key] = _apply_weighted_trustscores_recursive(
                obj=nested_value,
                path=path + ("__named_entry__",),
                score_accumulator=score_accumulator,
                weight_accumulator=weight_accumulator,
            )
            return obj

        for key, value in obj.items():
            if key in {"trust_score", "score"} and _is_numeric_score(value):
                score_path = path + (key,)
                total_weight = weight_accumulator.get(score_path)
                if total_weight:
                    obj[key] = round(score_accumulator[score_path] / total_weight, 6)
                continue

            obj[key] = _apply_weighted_trustscores_recursive(
                obj=value,
                path=path + (key,),
                score_accumulator=score_accumulator,
                weight_accumulator=weight_accumulator,
            )
        return obj

    if isinstance(obj, list):
        for index, item in enumerate(obj):
            obj[index] = _apply_weighted_trustscores_recursive(
                obj=item,
                path=path + (index,),
                score_accumulator=score_accumulator,
                weight_accumulator=weight_accumulator,
            )
    return obj


def _get_structural_named_entry(obj: dict):
    if len(obj) != 1:
        return None

    entry_key, nested_value = next(iter(obj.items()))
    if not isinstance(nested_value, dict):
        return None

    if any(key in nested_value for key in ("score", "metrics", "notions", "pillars")):
        return entry_key, nested_value

    return None


def _is_numeric_score(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)
