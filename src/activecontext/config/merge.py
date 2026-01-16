"""Deep merge algorithm for configuration cascading.

Supports merging multiple config dicts where later values override earlier ones,
with special handling for nested dicts and None values.
"""

from __future__ import annotations

from typing import Any


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries.

    Override values take precedence over base values, with these rules:
    - Nested dicts are recursively merged
    - Lists are replaced entirely (not concatenated)
    - None values in override do NOT override base (enables partial configs)
    - Other values are replaced

    Args:
        base: The base dictionary.
        override: The dictionary with overriding values.

    Returns:
        A new dictionary with merged values.
    """
    result = base.copy()

    for key, override_value in override.items():
        if override_value is None:
            # Skip None - allows partial configs to leave values unset
            continue

        if key in result:
            base_value = result[key]
            if isinstance(base_value, dict) and isinstance(override_value, dict):
                # Recursive merge for nested dicts
                result[key] = deep_merge(base_value, override_value)
            else:
                # Replace non-dict values (including lists)
                result[key] = override_value
        else:
            result[key] = override_value

    return result


def merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
    """Merge multiple configs in order (later overrides earlier).

    Args:
        *configs: Variable number of config dicts to merge.

    Returns:
        A single merged config dict.
    """
    result: dict[str, Any] = {}
    for config in configs:
        if config:
            result = deep_merge(result, config)
    return result
