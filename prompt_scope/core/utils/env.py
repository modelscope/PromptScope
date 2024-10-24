import os
from typing import Any, Dict, Optional


def get_from_dict_or_env(
    data: Dict[str, Any],
    key: str,
    default: Optional[str] = None,
) -> str:
    """Get a value from a dictionary or an environment variable.

    Args:
        data: The dictionary to look up the key in.
        key: The key to look up in the dictionary or environment. This can be a list of keys to try
            in order.
        default: The default value to return if the key is not in the dictionary
            or the environment. Defaults to None.
    """
    if key in data and data[key]:
        return data[key]
    elif key.upper() in os.environ and os.environ[key.upper()]:
        return os.environ[key.upper()]
    elif default is not None:
        return default
    else:
        raise ValueError(
            f"Did not find {key}, please add an environment variable"
            f" `{key.upper()}` which contains it, or pass"
            f" `{key}` as a named parameter."
        )
