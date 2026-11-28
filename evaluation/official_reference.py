"""Read MatterGen's Alex-MP reference archive using a single read-only LMDB handle."""

import gzip
import pickle
import shutil
from contextlib import ExitStack, contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

from pymatgen.core import Structure


@contextmanager
def open_official_reference(archive: Path | None):
    try:
        import lmdb
        if archive is None:
            from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise SystemExit("Install the official reference dependencies with: "
                         "python -m pip install -r requirements-official.txt") from error

    if archive is None:
        archive = Path(hf_hub_download(
            repo_id="microsoft/mattergen",
            filename="data-release/alex-mp/reference_MP2020correction.gz",
            revision="5244495dd9a979ff71abc7548a0b14b9deb0069a",
        ))

    with ExitStack() as cleanup:
        with gzip.open(archive, "rb") as source, NamedTemporaryFile(
            prefix="mattergen-ref-", suffix=".lmdb", delete=False
        ) as target:
            database = Path(target.name)
            cleanup.callback(database.unlink, missing_ok=True)
            shutil.copyfileobj(source, target)

        with lmdb.open(str(database), subdir=False, readonly=True, lock=False,
                       readahead=False) as env, env.begin() as txn:
            def read(key):
                value = txn.get(key.encode("ascii"))
                if value is None:
                    raise ValueError(f"Missing key in reference archive: {key}")
                return pickle.loads(value)

            groups = {}
            for system in read("chemical_systems"):
                for formula in read(f"{system}.reduced_formulas"):
                    prefix = f"{system}.{formula}"
                    groups[formula] = (prefix, read(f"{prefix}.length"))

            def entries_for_formula(formula):
                if formula in groups:
                    prefix, count = groups[formula]
                    for index in range(count):
                        entry = read(f"{prefix}.{index}")
                        material_id = str(entry["data"].get("material_id", "reference"))
                        yield material_id, Structure.from_dict(entry["structure"])

            yield read("name"), sum(count for _, count in groups.values()), entries_for_formula
