"""Tests for provider-independent data transformations."""

from ai_platform.data.contracts import DataTransformer
from ai_platform.data.transformations import (
    ColumnRenameTransformer,
    FilterTransformer,
    SelectColumnsTransformer,
)


def test_column_rename_transformer_implements_contract() -> None:
    transformer = ColumnRenameTransformer(
        {
            "customer_id": "id",
            "customer_name": "name",
        }
    )

    assert isinstance(transformer, DataTransformer)


def test_column_rename_transformer() -> None:
    transformer = ColumnRenameTransformer(
        {
            "customer_id": "id",
            "customer_name": "name",
        }
    )

    data = [
        {
            "customer_id": "1",
            "customer_name": "Alice",
            "country": "US",
        },
        {
            "customer_id": "2",
            "customer_name": "Bob",
            "country": "UK",
        },
    ]

    result = transformer.transform(data)

    assert result == [
        {
            "id": "1",
            "name": "Alice",
            "country": "US",
        },
        {
            "id": "2",
            "name": "Bob",
            "country": "UK",
        },
    ]


def test_column_rename_transformer_preserves_unmapped_columns() -> None:
    transformer = ColumnRenameTransformer(
        {
            "customer_id": "id",
        }
    )

    result = transformer.transform(
        [
            {
                "customer_id": 10,
                "name": "Alice",
            }
        ]
    )

    assert result == [
        {
            "id": 10,
            "name": "Alice",
        }
    ]


def test_select_columns_transformer_implements_contract() -> None:
    transformer = SelectColumnsTransformer(["id", "name"])

    assert isinstance(transformer, DataTransformer)


def test_select_columns_transformer() -> None:
    transformer = SelectColumnsTransformer(["id", "name"])

    data = [
        {
            "id": 1,
            "name": "Alice",
            "country": "US",
        },
        {
            "id": 2,
            "name": "Bob",
            "country": "UK",
        },
    ]

    result = transformer.transform(data)

    assert result == [
        {
            "id": 1,
            "name": "Alice",
        },
        {
            "id": 2,
            "name": "Bob",
        },
    ]


def test_select_columns_transformer_ignores_missing_columns() -> None:
    transformer = SelectColumnsTransformer(["id", "name", "email"])

    result = transformer.transform(
        [
            {
                "id": 1,
                "name": "Alice",
            }
        ]
    )

    assert result == [
        {
            "id": 1,
            "name": "Alice",
        }
    ]


def test_filter_transformer_implements_contract() -> None:
    transformer = FilterTransformer(lambda record: record["active"] is True)

    assert isinstance(transformer, DataTransformer)


def test_filter_transformer() -> None:
    transformer = FilterTransformer(lambda record: record["active"] is True)

    data = [
        {"id": 1, "active": True},
        {"id": 2, "active": False},
        {"id": 3, "active": True},
    ]

    result = transformer.transform(data)

    assert result == [
        {"id": 1, "active": True},
        {"id": 3, "active": True},
    ]


def test_filter_transformer_does_not_modify_input() -> None:
    data = [
        {"id": 1, "active": True},
        {"id": 2, "active": False},
    ]

    transformer = FilterTransformer(lambda record: record["active"])

    result = transformer.transform(data)

    assert data == [
        {"id": 1, "active": True},
        {"id": 2, "active": False},
    ]

    assert result == [
        {"id": 1, "active": True},
    ]
