"""Shared fixtures. PROCESSED_DIR selects the output folder under test."""
import os
import pathlib

import duckdb
import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]


def processed_dir():
    path = pathlib.Path(os.environ.get("PROCESSED_DIR", "data/processed"))
    return path if path.is_absolute() else ROOT / path


def pytest_collection_modifyitems(config, items):
    """Skip `data` tests when the text table has not been built (e.g. a fresh clone)."""
    if (processed_dir() / "features_text.parquet").exists():
        return
    skip = pytest.mark.skip(reason="data/processed/features_text.parquet is absent")
    for item in items:
        if "data" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def processed():
    path = pathlib.Path(os.environ.get("PROCESSED_DIR", "data/processed"))
    return path if path.is_absolute() else ROOT / path


@pytest.fixture(scope="session")
def is_full_data(processed):
    return processed == ROOT / "data/processed"


@pytest.fixture(scope="session")
def cfg():
    return yaml.safe_load((ROOT / "configs/features.yaml").read_text())


@pytest.fixture(scope="session")
def seed():
    return yaml.safe_load((ROOT / "configs/global.yaml").read_text())["seed"]


@pytest.fixture(scope="session")
def con(tmp_path_factory):
    # Single-threaded: multi-threaded writes corrupted Parquet output (see decisions_log.md).
    connection = duckdb.connect(config={"threads": 1})
    connection.execute(f"SET temp_directory = '{tmp_path_factory.mktemp('duckdb')}'")
    return connection


@pytest.fixture(scope="session")
def table(processed):
    """SQL for reading one processed table."""
    return lambda name: f"read_parquet('{processed}/{name}.parquet')"
