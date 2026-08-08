import math
from collections import UserDict, defaultdict
from collections.abc import Iterator, Mapping
from copy import deepcopy
from types import MappingProxyType

import pytest

from knowledge_aug_lab.augmentation import ContextCache, MemoryStore, TableStore, ToolRegistry, ToolResult
from knowledge_aug_lab.models import Document, FrozenMetadata


class StatefulToolMapping(Mapping[str, int]):
    def __init__(self) -> None:
        self.reads = 0

    def __getitem__(self, key: str) -> int:
        if key != "amount":
            raise KeyError(key)
        self.reads += 1
        return 1 if self.reads == 1 else 999

    def __iter__(self) -> Iterator[str]:
        return iter(("amount",))

    def __len__(self) -> int:
        return 1


class CustomToolDict(dict[str, object]):
    pass


class CustomToolList(list[object]):
    pass


class CustomToolTuple(tuple[object, ...]):
    pass


class CustomToolString(str):
    pass


class CustomToolInt(int):
    pass


class CustomToolFloat(float):
    pass


class CustomFrozenMetadata(FrozenMetadata):
    pass


def test_cag_preloads_corpus_once_and_reuses_the_same_context() -> None:
    cache = ContextCache(
        [
            Document("rag", "RAG performs retrieval for each query."),
            Document("cag", "CAG reuses a preloaded bounded context for low latency."),
        ],
        max_characters=500,
    )

    first = cache.context_for("What does CAG reuse?")
    second = cache.context_for("Why can CAG be fast?")

    assert "preloaded bounded context" in first
    assert first == second
    assert cache.build_count == 1
    assert cache.hit_count == 2


def test_table_augmented_generation_executes_typed_aggregation() -> None:
    table = TableStore(
        [
            {"strategy": "RAG", "latency_ms": 120},
            {"strategy": "RAG", "latency_ms": 180},
            {"strategy": "CAG", "latency_ms": 35},
        ]
    )

    result = table.aggregate("latency_ms", operation="mean", where={"strategy": "RAG"})

    assert result.value == 150
    assert result.rows_used == 2
    assert result.provenance == (0, 1)


def test_memory_augmented_generation_recalls_relevant_session_facts() -> None:
    memory = MemoryStore()
    memory.remember("user-7", "The user prefers local models and no cloud APIs.")
    memory.remember("user-7", "The current project demonstrates knowledge augmentation.")
    memory.remember("other-user", "The user prefers hosted models.")

    recalled = memory.recall("user-7", "Which model deployment preference should I follow?", top_k=1)

    assert recalled == ["The user prefers local models and no cloud APIs."]


def test_memory_recall_rejects_invalid_limits_and_excludes_zero_overlap() -> None:
    memory = MemoryStore()
    memory.remember("user-7", "The user prefers local models.")

    assert memory.recall("user-7", "unrelated weather forecast", top_k=1) == []

    try:
        memory.recall("user-7", "models", top_k=0)
    except ValueError as error:
        assert str(error) == "top_k must be positive"
    else:
        raise AssertionError("top_k=0 must be rejected")


def test_tool_augmented_generation_calls_allowlisted_tool_with_structured_arguments() -> None:
    tools = ToolRegistry()
    tools.register("estimate_cost", lambda tokens, rate: tokens / 1_000_000 * rate)

    result = tools.call("estimate_cost", tokens=2_500_000, rate=0.40)

    assert result.name == "estimate_cost"
    assert result.output == 1.0
    assert result.arguments == {"tokens": 2_500_000, "rate": 0.40}


def test_tool_result_recursively_snapshots_arguments() -> None:
    tools = ToolRegistry()
    tools.register("inspect_config", lambda config: config["limits"][0])
    config = {"limits": [10]}

    result = tools.call("inspect_config", config=config)
    config["limits"][0] = 0

    assert result.output == 10
    assert result.arguments["config"]["limits"] == (10,)
    with pytest.raises(TypeError, match="metadata is immutable"):
        result.arguments["config"]["limits"] = (0,)  # type: ignore[index]

    copied = deepcopy(result)
    assert isinstance(copied.arguments, FrozenMetadata)
    with pytest.raises(TypeError, match="metadata is immutable"):
        copied.arguments["config"]["limits"] = (0,)  # type: ignore[index]


def test_tool_registry_snapshots_arguments_before_tool_execution() -> None:
    tools = ToolRegistry()

    def mutate_config(config: dict[str, list[int]]) -> int:
        config["limits"][0] = 0
        return config["limits"][0]

    tools.register("mutate_config", mutate_config)
    config = {"limits": [10]}

    result = tools.call("mutate_config", config=config)

    assert result.output == 0
    assert config == {"limits": [0]}
    assert result.arguments["config"]["limits"] == (10,)


def test_tool_registry_rejects_stateful_mapping_before_execution() -> None:
    called = False

    def inspect_amount(payload: Mapping[str, int]) -> int:
        nonlocal called
        called = True
        return payload["amount"]

    tools = ToolRegistry()
    tools.register("inspect_amount", inspect_amount)

    with pytest.raises(TypeError, match="unsupported tool value type: StatefulToolMapping"):
        tools.call("inspect_amount", payload=StatefulToolMapping())
    assert called is False


def test_tool_registry_rejects_custom_mapping_nested_in_plain_dictionary() -> None:
    called = False

    def inspect_payload(payload: object) -> None:
        nonlocal called
        called = True

    tools = ToolRegistry()
    tools.register("inspect_payload", inspect_payload)

    with pytest.raises(TypeError, match="unsupported tool value type: StatefulToolMapping"):
        tools.call("inspect_payload", payload={"nested": StatefulToolMapping()})
    assert called is False


@pytest.mark.parametrize(
    "value",
    [
        UserDict({"amount": 1}),
        defaultdict(int, {"amount": 1}),
        MappingProxyType({"amount": 1}),
        CustomToolDict({"amount": 1}),
    ],
)
def test_tool_registry_rejects_non_builtin_mapping_arguments_before_execution(value: object) -> None:
    called = False

    def inspect_payload(payload: object) -> None:
        nonlocal called
        called = True

    tools = ToolRegistry()
    tools.register("inspect_payload", inspect_payload)

    with pytest.raises(TypeError, match=f"unsupported tool value type: {type(value).__name__}"):
        tools.call("inspect_payload", payload=value)
    assert called is False


@pytest.mark.parametrize(
    "value",
    [
        CustomToolList([1]),
        CustomToolTuple((1,)),
        CustomToolString("value"),
        CustomToolInt(1),
        CustomToolFloat(1.0),
        CustomFrozenMetadata({"amount": 1}),
    ],
)
def test_tool_registry_rejects_behavior_bearing_value_subclasses(value: object) -> None:
    called = False

    def inspect_payload(payload: object) -> None:
        nonlocal called
        called = True

    tools = ToolRegistry()
    tools.register("inspect_payload", inspect_payload)

    with pytest.raises(TypeError, match=f"unsupported tool value type: {type(value).__name__}"):
        tools.call("inspect_payload", payload=value)
    assert called is False


def test_tool_registry_rejects_string_subclass_dictionary_keys() -> None:
    called = False

    def inspect_payload(payload: object) -> None:
        nonlocal called
        called = True

    tools = ToolRegistry()
    tools.register("inspect_payload", inspect_payload)

    with pytest.raises(TypeError, match="tool dictionary keys must be exact strings"):
        tools.call("inspect_payload", payload={CustomToolString("amount"): 1})
    assert called is False


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_tool_registry_rejects_nonfinite_float_arguments_before_execution(value: float) -> None:
    called = False

    def inspect_payload(payload: float) -> None:
        nonlocal called
        called = True

    tools = ToolRegistry()
    tools.register("inspect_payload", inspect_payload)

    with pytest.raises(ValueError, match="tool float values must be finite"):
        tools.call("inspect_payload", payload=value)
    assert called is False


def test_tool_result_recursively_snapshots_output() -> None:
    tools = ToolRegistry()
    output = {"values": [1, 2]}
    tools.register("mutable_output", lambda: output)

    result = tools.call("mutable_output")
    output["values"].append(3)

    assert result.output == {"values": (1, 2)}
    with pytest.raises(TypeError, match="metadata is immutable"):
        result.output["values"] = (0,)  # type: ignore[index]


def test_tool_registry_rejects_frozen_metadata_returned_by_tool() -> None:
    tools = ToolRegistry()
    tools.register("frozen_output", lambda: FrozenMetadata({"amount": 1}))

    with pytest.raises(TypeError, match="unsupported tool value type: FrozenMetadata"):
        tools.call("frozen_output")


def test_tool_registry_rejects_custom_mapping_output() -> None:
    tools = ToolRegistry()
    tools.register("custom_output", lambda: StatefulToolMapping())

    with pytest.raises(TypeError, match="unsupported tool value type: StatefulToolMapping"):
        tools.call("custom_output")


@pytest.mark.parametrize("value", [bytearray(b"AB"), memoryview(b"AB"), range(3)])
def test_tool_registry_rejects_unsupported_argument_snapshots_before_execution(value: object) -> None:
    called = False

    def inspect_payload(payload: object) -> None:
        nonlocal called
        called = True

    tools = ToolRegistry()
    tools.register("inspect_payload", inspect_payload)

    with pytest.raises(TypeError, match=f"unsupported tool value type: {type(value).__name__}"):
        tools.call("inspect_payload", payload=value)
    assert called is False


def test_tool_registry_rejects_nested_unsupported_output_snapshots() -> None:
    tools = ToolRegistry()
    tools.register("unsupported_output", lambda: {"nested": [memoryview(b"AB")]})

    with pytest.raises(TypeError, match="unsupported tool value type: memoryview"):
        tools.call("unsupported_output")


def test_tool_audit_snapshot_detaches_nested_lists_and_mappings() -> None:
    arguments = {"nested": {"values": [1, 2]}}
    output = {"nested": {"values": [3, 4]}}
    tools = ToolRegistry()
    tools.register("snapshot", lambda payload: output)

    result = tools.call("snapshot", payload=arguments)
    arguments["nested"]["values"].append(99)
    arguments["nested"]["forged"] = True
    output["nested"]["values"].append(99)
    output["nested"]["forged"] = True

    assert result.arguments == {"payload": {"nested": {"values": (1, 2)}}}
    assert result.output == {"nested": {"values": (3, 4)}}


def test_tool_registry_accepts_exact_finite_json_values_and_intentional_tuples() -> None:
    payload = {
        "string": "value",
        "boolean": True,
        "integer": 7,
        "float": 1.25,
        "none": None,
        "list": [1, "two"],
        "tuple": (False, 3.5),
        "dictionary": {"nested": [None]},
    }
    tools = ToolRegistry()
    tools.register("echo", lambda value: value)

    result = tools.call("echo", value=payload)

    expected = {
        "string": "value",
        "boolean": True,
        "integer": 7,
        "float": 1.25,
        "none": None,
        "list": (1, "two"),
        "tuple": (False, 3.5),
        "dictionary": {"nested": (None,)},
    }
    assert result.arguments == {"value": expected}
    assert result.output == expected


def test_tool_result_allows_frozen_arguments_only_for_idempotent_reconstruction() -> None:
    arguments = FrozenMetadata({"payload": {"amount": 1}})

    result = ToolResult("inspect", arguments, {"ok": True})

    assert result.arguments is arguments
    assert result.output == {"ok": True}


def test_tool_result_rejects_nested_frozen_metadata_arguments() -> None:
    with pytest.raises(TypeError, match="unsupported tool value type: FrozenMetadata"):
        ToolResult("inspect", {"payload": FrozenMetadata({"amount": 1})}, None)


def test_tool_registry_rejects_nested_frozen_metadata_before_execution() -> None:
    called = False

    def inspect_payload(payload: object) -> None:
        nonlocal called
        called = True

    tools = ToolRegistry()
    tools.register("inspect_payload", inspect_payload)

    with pytest.raises(TypeError, match="unsupported tool value type: FrozenMetadata"):
        tools.call("inspect_payload", payload={"nested": FrozenMetadata({"amount": 1})})
    assert called is False
