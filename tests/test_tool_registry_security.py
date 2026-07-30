import pytest

from knowledge_aug_lab.augmentation import ToolRegistry, ToolSpec


def test_duplicate_tool_registration_fails_closed() -> None:
    registry = ToolRegistry()
    registry.register("calculate", lambda value: value)

    with pytest.raises(ValueError, match="tool is already registered"):
        registry.register("calculate", lambda value: value * 2)


@pytest.mark.parametrize("name", ["", "not valid", "a-b", None, 1])
def test_tool_names_must_be_identifiers(name: object) -> None:
    with pytest.raises(ValueError, match="valid identifiers"):
        ToolRegistry().register(name, lambda: None)  # type: ignore[arg-type]


def test_non_callable_registration_and_replacement_are_rejected() -> None:
    registry = ToolRegistry()
    with pytest.raises(TypeError, match="tool must be callable"):
        registry.register("bad", 1)  # type: ignore[arg-type]

    registry.register("good", lambda: "ok")
    with pytest.raises(TypeError, match="tool must be callable"):
        registry.replace("good", None)  # type: ignore[arg-type]


def test_replacement_must_be_explicit_and_target_an_existing_tool() -> None:
    registry = ToolRegistry()
    registry.register("value", lambda: 1)
    registry.replace("value", lambda: 2)

    assert registry.call("value").output == 2
    with pytest.raises(KeyError, match="tool is not registered"):
        registry.replace("missing", lambda: 3)


def test_typed_tool_spec_rejects_unexpected_arguments_before_invocation() -> None:
    calls = 0

    def calculate(amount: float) -> float:
        nonlocal calls
        calls += 1
        return amount * 2

    registry = ToolRegistry()
    registry.register_spec(
        ToolSpec(
            name="calculate",
            description="Double an approved numeric amount.",
            function=calculate,
            allowed_arguments=frozenset({"amount"}),
        )
    )

    with pytest.raises(ValueError, match="unexpected tool arguments: secret"):
        registry.call("calculate", amount=2, secret="exfiltrate")
    assert calls == 0
    assert registry.call("calculate", amount=2).output == 4


def test_tool_exceptions_are_surfaced_without_being_hidden() -> None:
    registry = ToolRegistry()

    def fail() -> None:
        raise RuntimeError("tool failed predictably")

    registry.register("fail", fail)

    with pytest.raises(RuntimeError, match="tool failed predictably"):
        registry.call("fail")


def test_uninspectable_tool_signatures_fail_closed() -> None:
    class Uninspectable:
        __signature__ = "invalid"

        def __call__(self) -> None:
            return None

    with pytest.raises(ValueError, match="tool signature cannot be inspected"):
        ToolRegistry().register("opaque", Uninspectable())


def test_tool_specs_must_allow_every_required_keyword_argument() -> None:
    with pytest.raises(ValueError, match="required tool arguments must be allowlisted: required"):
        ToolSpec(
            name="incomplete",
            description="An intentionally incomplete tool specification.",
            function=lambda required: required,
            allowed_arguments=frozenset(),
        )


def test_required_positional_only_tools_cannot_be_registered() -> None:
    def positional_only(value: int, /) -> int:
        return value

    with pytest.raises(ValueError, match="required positional-only arguments"):
        ToolRegistry().register("positional_only", positional_only)
