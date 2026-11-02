"""Strict Hodel-SSA tokens -> Typed Graph IR parser.

M0A fixes:
- lossless multi-digit SSA references (``REF D1 D0 END_REF`` -> ``D10``);
- support for OUT INVOKE as well as LET INVOKE;
- unique constant slots per literal occurrence;
- direct contextual typing of numeric literals where the operator port is known;
- fail-closed handling of unknown CALL primitives and malformed streams;
- post-parse SSA/DAG validation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from schema.typed_graph_ir import (
    Constant,
    GraphNode,
    OperatorSchema,
    OPERATOR_REGISTRY,
    ParamSlot,
    ParametricGraph,
    ValueType,
)


class HodelParseError(ValueError):
    """Raised when the token stream cannot be lowered losslessly."""


class HodelValidationError(ValueError):
    """Raised when a parsed graph violates SSA or registry invariants."""


SSA_REF_RE = re.compile(r"^D([0-9]+)$")
NUMERIC_NAMES: dict[str, int] = {
    "ZERO": 0,
    "ONE": 1,
    "TWO": 2,
    "THREE": 3,
    "FOUR": 4,
    "FIVE": 5,
    "SIX": 6,
    "SEVEN": 7,
    "EIGHT": 8,
    "NINE": 9,
    "TEN": 10,
}

# Constants whose semantic class is unambiguous independently of call context.
FIXED_NAME_TABLE: dict[str, Constant] = {
    "T": Constant(ValueType.BOOL, True),
    "F": Constant(ValueType.BOOL, False),
    "ORIGIN": Constant(ValueType.POINT, (0, 0)),
    "UNITY": Constant(ValueType.VECTOR, (1, 1)),
    "NEG_UNITY": Constant(ValueType.VECTOR, (-1, -1)),
    "RIGHT": Constant(ValueType.VECTOR, (0, 1)),
    "LEFT": Constant(ValueType.VECTOR, (0, -1)),
    "UP": Constant(ValueType.VECTOR, (-1, 0)),
    "DOWN": Constant(ValueType.VECTOR, (1, 0)),
    "UP_RIGHT": Constant(ValueType.VECTOR, (-1, 1)),
    "DOWN_LEFT": Constant(ValueType.VECTOR, (1, -1)),
    "THREE_BY_THREE": Constant(ValueType.TUPLE, (3, 3)),
    "TWO_BY_TWO": Constant(ValueType.TUPLE, (2, 2)),
    "TWO_BY_ZERO": Constant(ValueType.VECTOR, (2, 0)),
    "ZERO_BY_TWO": Constant(ValueType.VECTOR, (0, 2)),
    "NEG_TWO": Constant(ValueType.INT, -2),
    "NEG_ONE": Constant(ValueType.INT, -1),
}


@dataclass
class TokenStream:
    tokens: Sequence[str]
    pos: int = 0

    def peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def read(self) -> str:
        if self.pos >= len(self.tokens):
            raise HodelParseError("Unexpected end of token stream")
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def expect(self, expected: str) -> None:
        token = self.read()
        if token != expected:
            raise HodelParseError(
                f"Expected {expected!r} at token {self.pos - 1}, got {token!r}"
            )

    def remaining(self) -> tuple[str, ...]:
        return tuple(self.tokens[self.pos :])


def decode_ssa_ref(parts: Sequence[str]) -> str:
    """Decode Hodel's digit-tokenized SSA reference.

    Examples:
        ["D9"]       -> "D9"
        ["D1", "D0"] -> "D10"
        ["D5", "D9"] -> "D59"
    """

    if not parts:
        raise HodelParseError("Empty REF ... END_REF")

    digits: list[str] = []
    for token in parts:
        match = SSA_REF_RE.fullmatch(token)
        if match is None:
            raise HodelParseError(f"Invalid SSA reference token {token!r}")
        digits.append(match.group(1))

    decoded = "D" + "".join(digits)
    if decoded == "D0":
        raise HodelParseError("D0 is not a valid Hodel SSA definition")
    return decoded


class HodelParser:
    def __init__(
        self,
        task_id: str,
        tokens: Sequence[str],
        *,
        strict_registry: bool = True,
    ) -> None:
        self.task_id = task_id
        self.ts = TokenStream(tuple(tokens))
        self.strict_registry = strict_registry
        self.definitions: list[GraphNode] = []
        self.constants: dict[str, Constant] = {}
        self.output_node = ""
        self._constant_counter = 0

    def _new_constant(self, source_name: str, constant: Constant) -> str:
        self._constant_counter += 1
        safe = re.sub(r"[^a-z0-9_]+", "_", source_name.lower()).strip("_")
        slot_name = f"c_{self._constant_counter:03d}_{safe or 'literal'}"
        self.constants[slot_name] = constant
        return slot_name

    def _schema(self, operator: str, arity: int) -> OperatorSchema:
        schema = OPERATOR_REGISTRY.get(operator)
        if schema is None:
            if self.strict_registry:
                raise HodelParseError(
                    f"Unknown CALL primitive {operator!r} in task {self.task_id}"
                )
            return OperatorSchema(operator, (ValueType.ANY,) * arity, ValueType.ANY)
        if schema.arity() != arity:
            raise HodelParseError(
                f"Arity mismatch for {operator!r}: token stream={arity}, "
                f"registry={schema.arity()}"
            )
        return schema

    def parse_ref(self) -> str:
        self.ts.expect("REF")
        parts: list[str] = []
        while True:
            token = self.ts.peek()
            if token is None:
                raise HodelParseError("Unterminated REF: missing END_REF")
            if token == "END_REF":
                self.ts.read()
                break
            parts.append(self.ts.read())
        return decode_ssa_ref(parts)

    def _constant_for_name(
        self,
        name: str,
        expected_type: ValueType,
    ) -> Constant:
        if name in FIXED_NAME_TABLE:
            return FIXED_NAME_TABLE[name]

        if name in NUMERIC_NAMES:
            # ZERO..TEN are overloaded in Hodel.  Resolve only when the call
            # port says COLOR or INT; otherwise retain explicit uncertainty.
            resolved_type = (
                expected_type
                if expected_type in (ValueType.COLOR, ValueType.INT)
                else ValueType.ANY
            )
            return Constant(resolved_type, NUMERIC_NAMES[name])

        # Lower-case and otherwise non-literal NAME tokens denote primitive or
        # composed functions in Hodel programs.  Expected FUNCTION is the
        # strongest direct evidence; unknown names are still represented as
        # function references rather than executable strings.
        if expected_type == ValueType.FUNCTION or name[:1].islower():
            return Constant(ValueType.FUNCTION, name)

        return Constant(expected_type if expected_type != ValueType.ANY else ValueType.STR, name)

    def parse_arg(self, expected_type: ValueType = ValueType.ANY) -> str:
        token = self.ts.peek()
        if token is None:
            raise HodelParseError("Missing call argument")

        if token == "INPUT":
            self.ts.read()
            return "INPUT"

        if token == "REF":
            return self.parse_ref()

        if SSA_REF_RE.fullmatch(token):
            self.ts.read()
            return token

        if token.startswith("NAME:"):
            self.ts.read()
            name = token.split(":", 1)[1]
            constant = self._constant_for_name(name, expected_type)
            return self._new_constant(name, constant)

        # Preserve unrecognized literal tokens losslessly in a typed slot.
        self.ts.read()
        literal_type = expected_type if expected_type != ValueType.ANY else ValueType.STR
        return self._new_constant(token, Constant(literal_type, token))

    def _parse_arity(self) -> int:
        token = self.ts.read()
        if not token.startswith("ARITY:"):
            raise HodelParseError(f"Expected ARITY:n, got {token!r}")
        try:
            arity = int(token.split(":", 1)[1])
        except ValueError as exc:
            raise HodelParseError(f"Invalid arity token {token!r}") from exc
        if arity < 0:
            raise HodelParseError(f"Negative arity {arity}")
        return arity

    def _bound_literal_type(self, function_ref: str, *, bind_left: bool) -> ValueType:
        constant = self.constants.get(function_ref)
        if constant is None or constant.type != ValueType.FUNCTION:
            return ValueType.ANY
        schema = OPERATOR_REGISTRY.get(str(constant.value))
        if schema is None or not schema.input_types:
            return ValueType.ANY
        return schema.input_types[0] if bind_left else schema.input_types[-1]

    def _parse_call_args(
        self,
        operator: str,
        schema: OperatorSchema,
    ) -> tuple[str, ...]:
        args: list[str] = []
        for index, expected_type in enumerate(schema.input_types):
            # Direct propagation through the common higher-order binders.
            if index == 1 and args and operator in {"lbind", "rbind"}:
                expected_type = self._bound_literal_type(
                    args[0], bind_left=(operator == "lbind")
                )
            elif index == 1 and args and operator == "matcher":
                function_constant = self.constants.get(args[0])
                if (
                    function_constant is not None
                    and function_constant.type == ValueType.FUNCTION
                    and str(function_constant.value) in OPERATOR_REGISTRY
                ):
                    expected_type = OPERATOR_REGISTRY[
                        str(function_constant.value)
                    ].output_type
            args.append(self.parse_arg(expected_type))
        return tuple(args)

    def parse_call(self, *, output_name: str) -> GraphNode:
        call_token = self.ts.read()
        if not call_token.startswith("CALL:"):
            raise HodelParseError(f"Expected CALL:name, got {call_token!r}")
        operator = call_token.split(":", 1)[1]
        arity = self._parse_arity()
        schema = self._schema(operator, arity)
        args = self._parse_call_args(operator, schema)
        return GraphNode(operator, args, schema.output_type, output_name)

    def parse_invoke(self, *, output_name: str) -> GraphNode:
        self.ts.expect("INVOKE")
        function_ref = self.parse_ref()
        arity = self._parse_arity()
        args = tuple(self.parse_arg(ValueType.ANY) for _ in range(arity))
        return GraphNode(
            operator="invoke",
            inputs=(function_ref,) + args,
            output_type=ValueType.ANY,
            output_name=output_name,
        )

    def parse_definition(self) -> GraphNode:
        self.ts.expect("LET")
        output_name = f"D{len(self.definitions) + 1}"
        token = self.ts.peek()
        if token == "INVOKE":
            node = self.parse_invoke(output_name=output_name)
        elif token is not None and token.startswith("CALL:"):
            node = self.parse_call(output_name=output_name)
        else:
            raise HodelParseError(
                f"Expected CALL:name or INVOKE after LET, got {token!r}"
            )
        self.definitions.append(node)
        return node

    def parse_output(self) -> None:
        self.ts.expect("OUT")
        token = self.ts.peek()
        if token == "INVOKE":
            node = self.parse_invoke(output_name="OUTPUT")
            self.definitions.append(node)
            self.output_node = "OUTPUT"
            return
        if token is not None and token.startswith("CALL:"):
            node = self.parse_call(output_name="OUTPUT")
            self.definitions.append(node)
            self.output_node = "OUTPUT"
            return
        self.output_node = self.parse_arg(ValueType.GRID)

    def parse(self) -> ParametricGraph:
        if self.ts.peek() == "BOS":
            self.ts.read()

        while self.ts.peek() == "LET":
            self.parse_definition()

        if self.ts.peek() != "OUT":
            raise HodelParseError(
                f"Task {self.task_id} has no OUT expression; next={self.ts.peek()!r}"
            )
        self.parse_output()

        if self.ts.peek() == "EOS":
            self.ts.read()
        if self.ts.peek() is not None:
            raise HodelParseError(
                f"Trailing tokens after EOS/output: {self.ts.remaining()!r}"
            )

        nodes = tuple(self.definitions)
        params = tuple(
            ParamSlot(
                name=name,
                value_type=constant.type,
                default=constant,
                description="Hodel literal/function reference",
            )
            for name, constant in self.constants.items()
        )
        cost = sum(
            OPERATOR_REGISTRY[node.operator].cost
            if node.operator in OPERATOR_REGISTRY
            else 1
            for node in nodes
        )
        graph = ParametricGraph(
            graph_id=self.task_id,
            family="compiled",
            version="v1-m0a",
            inputs=(ParamSlot("input", ValueType.GRID),),
            parameters=params,
            output_type=ValueType.GRID,
            nodes=nodes,
            output_node=self.output_node,
            support=1,
            cost=cost,
            source_tasks=(self.task_id,),
            is_singleton=True,
        )
        validate_graph_ssa(graph)
        return graph


def validate_graph_ssa(graph: ParametricGraph) -> None:
    """Validate the structural invariants required by the DAG miner."""

    available = {"INPUT"}
    available.update(slot.name for slot in graph.parameters)
    output_names: set[str] = set()

    for index, node in enumerate(graph.nodes, start=1):
        if node.operator != "invoke" and node.operator not in OPERATOR_REGISTRY:
            raise HodelValidationError(
                f"{graph.graph_id}: unregistered operator {node.operator!r}"
            )
        if node.output_name in available or node.output_name in output_names:
            raise HodelValidationError(
                f"{graph.graph_id}: duplicate output name {node.output_name!r}"
            )
        for ref in node.inputs:
            if ref not in available:
                raise HodelValidationError(
                    f"{graph.graph_id}: {node.output_name} input {ref!r} "
                    f"does not refer to INPUT, a parameter, or a prior SSA node"
                )
        available.add(node.output_name)
        output_names.add(node.output_name)

    if not graph.output_node:
        raise HodelValidationError(f"{graph.graph_id}: empty output node")
    if graph.output_node not in available:
        raise HodelValidationError(
            f"{graph.graph_id}: unresolved graph output {graph.output_node!r}"
        )


def parse_hodel_tokens(
    task_id: str,
    tokens: Sequence[str],
    *,
    strict_registry: bool = True,
) -> ParametricGraph:
    return HodelParser(
        task_id,
        tokens,
        strict_registry=strict_registry,
    ).parse()


def load_hodel_program(
    jsonl_path: str,
    task_id: str,
    *,
    strict_registry: bool = True,
) -> ParametricGraph | None:
    with open(jsonl_path, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("backend") == "hodel-ssa" and record.get("task_id") == task_id:
                return parse_hodel_tokens(
                    task_id,
                    record["tokens"],
                    strict_registry=strict_registry,
                )
    return None
