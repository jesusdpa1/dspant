"""
/mp_cache.py
Author: jpenalozaa
Description: Parquet cache for expensive intermediate results of the figure scripts.

    out = cached("fig06_tf", params, lambda: {"cwt": W, "scales": s})

Each cache entry is a folder <CACHE_DIR>/<name>/ with one parquet file per array.
The parameters (JSON), array shape and dtype are stored in the parquet schema metadata;
if `params` differ from the stored ones the entry is recomputed, so a changed filter,
wavelet or window can never silently reuse stale results.
CACHE_DIR defaults to <DATA_DIR>/../cache (i.e. .data/cache, gitignored).
"""

import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CACHE_VERSION = 1


def cache_dir() -> Path:
    env = os.getenv("CACHE_DIR")
    path = Path(env) if env else Path(os.getenv("DATA_DIR")).parent / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _key(params: dict) -> str:
    return json.dumps({"_cache_version": CACHE_VERSION, **params}, sort_keys=True, default=str)


def _write(arr: np.ndarray, path: Path, key: str) -> None:
    arr = np.asarray(arr)
    transposed = arr.ndim == 2 and arr.shape[1] > arr.shape[0]  # keep the long axis as rows
    data = arr.T if transposed else arr
    data = data.reshape(len(data), -1) if data.ndim > 1 else data[:, None]
    columns = {}
    for j in range(data.shape[1]):
        if np.iscomplexobj(data):
            columns[f"re_{j}"] = data[:, j].real
            columns[f"im_{j}"] = data[:, j].imag
        else:
            columns[f"c_{j}"] = data[:, j]
    meta = {
        "params": key,
        "shape": json.dumps(list(arr.shape)),
        "dtype": str(arr.dtype),
        "transposed": json.dumps(transposed),
    }
    table = pa.table(columns).replace_schema_metadata(meta)
    pq.write_table(table, path)


def _read(path: Path):
    table = pq.read_table(path)
    meta = {k.decode(): v.decode() for k, v in table.schema.metadata.items()}
    shape = tuple(json.loads(meta["shape"]))
    dtype = np.dtype(meta["dtype"])
    names = table.column_names
    if names[0].startswith("re_"):
        n = len(names) // 2
        data = np.stack(
            [table[f"re_{j}"].to_numpy() + 1j * table[f"im_{j}"].to_numpy() for j in range(n)], axis=1
        )
    else:
        data = np.stack([table[f"c_{j}"].to_numpy() for j in range(len(names))], axis=1)
    if json.loads(meta["transposed"]):
        data = data.T
    return data.reshape(shape).astype(dtype, copy=False), meta["params"]


def cached(name: str, params: dict, compute) -> dict:
    """Return dict of arrays from cache if `params` match, else compute, store and return."""
    folder = cache_dir() / name
    key = _key(params)
    if folder.exists():
        files = sorted(folder.glob("*.parquet"))
        if files:
            out, stored_keys = {}, set()
            for f in files:
                out[f.stem], k = _read(f)
                stored_keys.add(k)
            if stored_keys == {key}:
                print(f"[cache] loaded {name} from {folder}")
                return out
            print(f"[cache] parameters changed for {name}; recomputing")
        for f in folder.glob("*.parquet"):
            f.unlink()
    folder.mkdir(parents=True, exist_ok=True)
    out = {k: np.asarray(v) for k, v in compute().items()}
    for k, v in out.items():
        _write(v, folder / f"{k}.parquet", key)
    print(f"[cache] saved {name} to {folder}")
    return out
