"""
Semantic urban layers.

H3 resolution and semantic layer are independent concepts.
The same resolution can store several semantic layers, and different
resolutions can carry different kinds of information.
"""

from __future__ import annotations

from collections import OrderedDict


SEMANTIC_LAYERS = OrderedDict(
    [
        ("D3",  "Retail Store"),
        ("D4",  "Commercial"),
        ("D5",  "Education"),
        ("D6",  "Housing"),
        ("D7",  "Religious"),
        ("D8",  "Healthcare"),
        ("D9",  "Government Operations"),
        ("D10", "Mid-Career Housing"),
        ("D11", "Senior Housing"),
        ("D12", "Affordable Housing"),
        ("D13", "Executive Housing"),
        ("D14", "Headquarter Office"),
        ("D15", "Grocery-Market"),
        ("D16", "Recreation & Fitness"),
        ("D17", "Pharmacy"),
        ("D18", "Career Training"),
        ("D19", "Daycare Center"),
        ("D20", "Coworking Office"),
        ("D21", "Restaurant"),
        ("D22", "Cafe"),
        ("D23", "Bar"),
        ("D24", "Pub"),
        ("D25", "Park"),
    ]
)

NAME_TO_CODE = {
    name: code
    for code, name in SEMANTIC_LAYERS.items()
}

LEGACY_CATEGORY_TO_CODE = {
    "retail": "D3",
    "education": "D5",
    "health": "D8",
    "food": "D21",
    "leisure": "D25",
}


def layer_name(code: str) -> str:
    return SEMANTIC_LAYERS.get(code, code)


def all_layer_codes():
    return tuple(SEMANTIC_LAYERS.keys())


def code_to_id(code: str | None) -> int:
    """
    Compact uint8-compatible representation.
    0 is reserved for HOME / no spatial query.
    D3..D25 map to 3..25.
    """
    if code is None or code == "":
        return 0

    code = str(code)
    if not code.startswith("D"):
        raise ValueError(f"Invalid semantic layer code: {code}")

    value = int(code[1:])

    if value < 3 or value > 25:
        raise ValueError(f"Semantic layer id out of range: {value}")

    return value


def id_to_code(layer_id: int) -> str | None:
    layer_id = int(layer_id)

    if layer_id == 0:
        return None

    code = f"D{layer_id}"

    if code not in SEMANTIC_LAYERS:
        return code

    return code
