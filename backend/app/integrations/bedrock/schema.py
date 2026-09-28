"""Build the JSON Schema subset accepted by Bedrock structured outputs."""

import json
from typing import Any

from pydantic import BaseModel

_UNSUPPORTED_KEYS = {
    "default",
    "exclusiveMaximum",
    "exclusiveMinimum",
    "maxItems",
    "maxLength",
    "maximum",
    "minLength",
    "minimum",
    "multipleOf",
}


def structured_output_config(model: type[BaseModel], *, name: str) -> dict[str, Any]:
    """Return a Converse outputConfig using Bedrock's supported schema subset."""

    schema = _sanitize(model.model_json_schema(mode="serialization"))
    return {
        "textFormat": {
            "type": "json_schema",
            "structure": {
                "jsonSchema": {
                    "schema": json.dumps(schema, separators=(",", ":")),
                    "name": name,
                    "description": f"Validated {name.replace('_', ' ')} output",
                }
            },
        }
    }


def _sanitize(value: Any) -> Any:
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if not isinstance(value, dict):
        return value
    sanitized = {
        key: _sanitize(item) for key, item in value.items() if key not in _UNSUPPORTED_KEYS
    }
    if sanitized.get("type") == "object":
        sanitized["additionalProperties"] = False
    min_items = sanitized.get("minItems")
    if isinstance(min_items, int) and min_items > 1:
        sanitized["minItems"] = 1
    return sanitized
