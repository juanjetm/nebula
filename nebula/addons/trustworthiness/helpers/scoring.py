import logging

import numpy as np

logger = logging.getLogger(__name__)

def get_mapped_score(score_key, score_map):
    """
    Finds the score by the score_key in the score_map.

    Args:
        score_key (string): The key to look up in the score_map.
        score_map (dict): The score map defined in the eval_metrics.json file.

    Returns:
        float: The normalized score of [0, 1].
    """
    score = 0
    if score_map is None:
        logger.warning("Score map is missing")
    else:
        keys = [key for key, value in score_map.items()]
        scores = [value for key, value in score_map.items()]
        normalized_scores = get_normalized_scores(scores)
        normalized_score_map = dict(zip(keys, normalized_scores, strict=False))
        score = normalized_score_map.get(score_key, np.nan)

    return score


def get_normalized_scores(scores):
    """
    Calculates the normalized scores of a list.

    Args:
        scores (list): The values that will be normalized.

    Returns:
        list: The normalized list.
    """
    if scores is None or len(scores) == 0:
        return []

    min_score = np.min(scores)
    max_score = np.max(scores)
    if max_score == min_score:
        return [1.0 for _ in scores]

    normalized = [(x - min_score) / (max_score - min_score) for x in scores]
    return normalized


def get_range_score(value, ranges, direction="asc"):
    """
    Maps the value to a range and gets the score by the range and direction.

    Args:
        value (int): The input score.
        ranges (list): The ranges defined.
        direction (string): Asc means the higher the range the higher the score, desc means otherwise.

    Returns:
        float: The normalized score of [0, 1].
    """

    if not (type(value) == int or type(value) == float):
        logger.warning("Input value is not a number")
        logger.warning(f"{value}")
        return 0
    else:
        score = 0
        if ranges is None:
            logger.warning("Score ranges are missing")
        else:
            total_bins = len(ranges) + 1
            bin = np.digitize(value, ranges, right=True)
            score = 1 - (bin / total_bins) if direction == "desc" else bin / total_bins
        return score


def get_map_value_score(score_key, score_map):
    """
    Finds the score by the score_key in the score_map and returns the value.

    Args:
        score_key (string): The key to look up in the score_map.
        score_map (dict): The score map defined in the eval_metrics.json file.

    Returns:
        float: The score obtained in the score_map.
    """
    score = 0
    if score_map is None:
        logger.warning("Score map is missing")
    else:
        score = score_map[score_key]
    return score


def get_true_score(value, direction):
    """
    Returns the negative of the value if direction is 'desc', otherwise returns value.

    Args:
        value (int): The input score.
        direction (string): Asc means the higher the range the higher the score, desc means otherwise.

    Returns:
        float: The score obtained.
    """

    if value is True:
        return 1
    elif value is False:
        return 0
    else:
        if not (type(value) == int or type(value) == float):
            logger.warning("Input value is not a number")
            logger.warning(f"{value}.")
            return 0
        else:
            if direction == "desc":
                return 1 - value
            else:
                return value


def get_scaled_score(value, scale: list, direction: str):
    """
    Maps a score of a specific scale into the scale between zero and one.

    Args:
        value (int or float): The raw value of the metric.
        scale (list): List containing the minimum and maximum value the value can fall in between.

    Returns:
        float: The normalized score of [0, 1].
    """

    score = 0
    try:
        value_min, value_max = scale[0], scale[1]
    except Exception:
        logger.warning("Score minimum or score maximum is missing. The minimum has been set to 0 and the maximum to 1")
        value_min, value_max = 0, 1
    if value is None or value == "":
        logger.warning("Score value is missing. Set value to zero")
    else:
        low, high = 0, 1
        if value >= value_max:
            score = 1
        elif value <= value_min:
            score = 0
        else:
            diff = value_max - value_min
            diffScale = high - low
            score = (float(value) - value_min) * (float(diffScale) / diff) + low
        if direction == "desc":
            score = high - score

    return score


def get_value(value):
    """
    Get the value of a metric.

    Args:
        value (float): The value of the metric.

    Returns:
        float: The value of the metric.
    """

    return value


def check_properties(*args):
    """
    Check if all the arguments have values.

    Args:
        args (list): All the arguments.

    Returns:
        float: The mean of arguments that have values.
    """

    result = map(lambda x: x is not None and x != "", args)
    return np.mean(list(result))
