import logging
import math

from nebula.addons.trustworthiness.helpers.privacy import (
    get_global_privacy_risk,
    get_global_privacy_risk_dfl,
)
from nebula.addons.trustworthiness.helpers.scenario_metrics import comm_efficiency
from nebula.addons.trustworthiness.helpers.scoring import (
    check_properties,
    get_value,
)

logger = logging.getLogger(__name__)

OPERATIONS = {
    "check_properties": check_properties,
    "comm_efficiency": comm_efficiency,
    "get_global_privacy_risk": get_global_privacy_risk,
    "get_global_privacy_risk_dfl": get_global_privacy_risk_dfl,
    "get_value": get_value,
}

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
        operationFn = OPERATIONS[operation]
        input_value = operationFn(*args)
    except KeyError:
        logger.warning(f"{operation} is not valid")
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
