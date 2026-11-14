"""Versioned SQLite storage; numeric arrays use NPZ without pickle."""

from io import BytesIO
import json
from pathlib import Path
import sqlite3

import numpy as np

SCHEMA_VERSION = 1
SCHEMA = """
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE sources (
    source_path TEXT PRIMARY KEY,
    sha256 TEXT NOT NULL,
    cif_text TEXT NOT NULL,
    parser_warnings TEXT NOT NULL
);
CREATE TABLE samples (
    id INTEGER PRIMARY KEY,
    sample_id TEXT UNIQUE NOT NULL,
    source_path TEXT NOT NULL REFERENCES sources(source_path),
    structure_index INTEGER NOT NULL,
    formula TEXT NOT NULL,
    num_atoms INTEGER NOT NULL CHECK (num_atoms > 0),
    num_edges INTEGER NOT NULL CHECK (num_edges >= 0),
    split TEXT NOT NULL CHECK (split IN ('train', 'val', 'test')),
    arrays BLOB NOT NULL
);
CREATE INDEX samples_split ON samples(split);
CREATE INDEX samples_source ON samples(source_path);
CREATE INDEX sources_hash ON sources(sha256);
"""


def pack_arrays(arrays: dict[str, np.ndarray]) -> bytes:
    buffer = BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def unpack_arrays(payload: bytes) -> dict[str, np.ndarray]:
    with np.load(BytesIO(payload), allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def connect_readonly(path: str | Path) -> sqlite3.Connection:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported dataset schema {version}; expected {SCHEMA_VERSION}")
    except Exception:
        connection.close()
        raise
    return connection


def inspect_dataset(path: str | Path) -> dict:
    """Read build configuration and counts without decoding structure arrays."""
    connection = connect_readonly(path)
    try:
        row = connection.execute("SELECT value FROM metadata WHERE key = 'build'").fetchone()
        if row is None:
            raise ValueError("Dataset has no build metadata")
        return json.loads(row[0])
    finally:
        connection.close()
