import inspect

from common.readers.base_reader import BaseReader
from common.readers.fabric_reader import FabricReader
from common.readers.postgres_reader import PostgresReader
from common.readers.snowflake_reader import SnowflakeReader

EXPECTED_PARAMETERS = [
    "self",
    "spark",
]


def test_base_reader_contract():
    signature = inspect.signature(BaseReader.read)

    assert list(signature.parameters) == EXPECTED_PARAMETERS


def test_table_readers_participate_in_base_reader_contract():
    readers = [
        FabricReader,
        PostgresReader,
        SnowflakeReader,
    ]

    for reader_cls in readers:
        assert issubclass(reader_cls, BaseReader)

        signature = inspect.signature(reader_cls.read)

        assert list(signature.parameters) == EXPECTED_PARAMETERS
