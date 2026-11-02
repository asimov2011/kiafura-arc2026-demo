"""A-only semantic-IR factorization for CT-2*.

The compiler turns complete audit derivations into reusable, alpha-normalized
subtree templates. Runtime memory receives only templates supported by at
least two distinct experiences; occurrence bindings remain audit-only. This
module proves structural reconstruction, not executable lowering.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


@dataclass(frozen=True)
class DerivationNode:
    production: str
    parameters: Mapping[str, Any]
    children: tuple["DerivationNode", ...]
    preorder_index: int

    @property
    def node_count(self) -> int:
        return 1 + sum(child.node_count for child in self.children)


def decode_derivation(
    codes: str,
    arities: Sequence[int],
    parameters: Mapping[str, Mapping[str, Any]],
    legend: Mapping[str, str],
) -> DerivationNode:
    if not codes or len(codes) != len(arities):
        raise ValueError("invalid prefix derivation")
    cursor = 0

    def decode() -> DerivationNode:
        nonlocal cursor
        if cursor >= len(codes):
            raise ValueError("truncated prefix derivation")
        index = cursor
        cursor += 1
        children = tuple(decode() for _ in range(int(arities[index])))
        return DerivationNode(
            production=legend[codes[index]],
            parameters=dict(parameters.get(str(index), {})),
            children=children,
            preorder_index=index,
        )

    root = decode()
    if cursor != len(codes):
        raise ValueError("prefix derivation has multiple roots")
    return root


def walk(root: DerivationNode) -> Iterable[DerivationNode]:
    yield root
    for child in root.children:
        yield from walk(child)


def _literal_type(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, str):
        return "string"
    return type(value).__name__


def normalize_subtree(root: DerivationNode) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return an alpha-normalized template and its audit-only substitution.

    Lexical identity and repeated literal equality are preserved.  Concrete
    names and literal values are replaced by ports, so neither can identify an
    experience in operational memory.
    """

    bound_names = {
        str(node.parameters["lexical_name"])
        for node in walk(root)
        if "lexical_name" in node.parameters
        and node.parameters.get("context") in {"Param", "Store"}
    }
    local_slots: dict[str, str] = {}
    free_slots: dict[str, str] = {}
    literal_slots: dict[tuple[str, str], str] = {}
    substitution: dict[str, Any] = {}

    def lexical_slot(name: str) -> str:
        slots = local_slots if name in bound_names else free_slots
        prefix = "local" if name in bound_names else "free"
        if name not in slots:
            slot = f"{prefix}_{len(slots)}"
            slots[name] = slot
            substitution[slot] = name
        return slots[name]

    def literal_slot(value: Any) -> str:
        kind = _literal_type(value)
        key = kind, json.dumps(value, sort_keys=True)
        if key not in literal_slots:
            slot = f"literal_{kind}_{len(literal_slots)}"
            literal_slots[key] = slot
            substitution[slot] = value
        return literal_slots[key]

    def normalize(node: DerivationNode) -> dict[str, Any]:
        parameters: dict[str, Any] = {}
        for key, value in sorted(node.parameters.items()):
            if key == "lexical_name":
                parameters[key] = {"port": lexical_slot(str(value))}
            elif key == "literal":
                parameters[key] = {
                    "domain": _literal_type(value),
                    "port": literal_slot(value),
                }
            else:
                parameters[key] = value
        return {
            "children": [normalize(child) for child in node.children],
            "parameters": parameters,
            "production": node.production,
        }

    return normalize(root), substitution


def instantiate_template(
    template: Mapping[str, Any],
    substitution: Mapping[str, Any],
) -> dict[str, Any]:
    """Materialize the normalized semantic tree for round-trip auditing."""

    parameters: dict[str, Any] = {}
    for key, value in template["parameters"].items():
        if isinstance(value, Mapping) and "port" in value:
            parameters[key] = substitution[value["port"]]
        else:
            parameters[key] = value
    return {
        "children": [
            instantiate_template(child, substitution)
            for child in template["children"]
        ],
        "parameters": parameters,
        "production": template["production"],
    }


def concrete_tree(root: DerivationNode) -> dict[str, Any]:
    return {
        "children": [concrete_tree(child) for child in root.children],
        "parameters": dict(root.parameters),
        "production": root.production,
    }


def template_id(template: Mapping[str, Any]) -> str:
    return digest(template)[:24]
