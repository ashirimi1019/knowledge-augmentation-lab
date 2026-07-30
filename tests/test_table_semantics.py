import math

import pytest

from knowledge_aug_lab.augmentation import TableStore


def test_count_does_not_require_numeric_column_values() -> None:
    table = TableStore([{"team": "red"}, {"team": "blue"}, {"team": "red"}])

    result = table.aggregate("team", "count", where={"team": "red"})

    assert result.value == 2.0
    assert result.rows_used == 2
    assert result.provenance == [0, 2]


@pytest.mark.parametrize("operation", ["sum", "mean", "min", "max"])
def test_numeric_aggregations_reject_boolean_values(operation: str) -> None:
    table = TableStore([{"value": True}])

    with pytest.raises(TypeError, match="must contain numeric values"):
        table.aggregate("value", operation)


@pytest.mark.parametrize("rows", [None, {}, ["not-a-row"], [1]])
def test_table_store_validates_rows(rows: object) -> None:
    expected = "rows must be a list" if not isinstance(rows, list) else "every row must be a dictionary"
    with pytest.raises(TypeError, match=expected):
        TableStore(rows)  # type: ignore[arg-type]


def test_table_store_copies_input_rows() -> None:
    rows = [{"value": 2}]
    table = TableStore(rows)
    rows[0]["value"] = 100

    assert table.aggregate("value", "sum").value == 2.0


@pytest.mark.parametrize("column", ["", " ", None, 1])
def test_aggregation_rejects_invalid_column(column: object) -> None:
    with pytest.raises(ValueError, match="column cannot be empty"):
        TableStore([{"value": 1}]).aggregate(column, "sum")  # type: ignore[arg-type]


def test_aggregation_validates_operation_filters_and_empty_selection() -> None:
    table = TableStore([{"value": 1}])
    with pytest.raises(ValueError, match="unsupported aggregation"):
        table.aggregate("value", "median")
    with pytest.raises(TypeError, match="where must be a dictionary"):
        table.aggregate("value", "sum", where=[])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="no rows match"):
        table.aggregate("value", "sum", where={"value": 2})


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, 10**400])
def test_aggregation_rejects_non_finite_or_overflowing_values(value: object) -> None:
    with pytest.raises(ValueError, match="finite numeric values"):
        TableStore([{"value": value}]).aggregate("value", "sum")


def test_count_requires_the_named_column_but_accepts_non_numeric_values() -> None:
    table = TableStore([{"label": "alpha"}, {"label": "beta"}])

    assert table.aggregate("label", "count").value == 2.0
    with pytest.raises(ValueError, match="column 'missing' is not present in every selected row"):
        table.aggregate("missing", "count")


def test_mean_preserves_representable_subnormal_values() -> None:
    value = math.ulp(0.0)

    assert TableStore([{"value": value}, {"value": value}]).aggregate("value", "mean").value == value
