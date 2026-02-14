#!/usr/bin/env python3
"""
Generate MatrixMarket input for AMGX MPI examples from binary CSR files.

Expected files in input directory:
  - row_ptr.bin (int32 CSR row pointer)
  - col_ind.bin (int32 CSR column indices)
  - val.bin     (float64 CSR values)

Output:
  - matrix.mtx
Optionally:
  - partition_vector.bin (int32, global-row -> rank mapping)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


def _fail(msg: str) -> None:
    raise RuntimeError(msg)


def _normalize_index_base(row_ptr: np.ndarray, col_ind: np.ndarray, n: int, nnz: int) -> tuple[np.ndarray, np.ndarray]:
    """Return zero-based CSR arrays, auto-detecting one-based input."""
    col_min = int(col_ind.min()) if nnz > 0 else 0
    col_max = int(col_ind.max()) if nnz > 0 else -1

    zero_based = (
        int(row_ptr[0]) == 0
        and int(row_ptr[-1]) == nnz
        and col_min >= 0
        and col_max < n
    )
    one_based = (
        int(row_ptr[0]) == 1
        and int(row_ptr[-1]) == nnz + 1
        and col_min >= 1
        and col_max <= n
    )

    if zero_based:
        return row_ptr.copy(), col_ind.copy()
    if one_based:
        return row_ptr - 1, col_ind - 1

    # Heuristic fallback for mixed data: treat as one-based if row_ptr starts at 1.
    if int(row_ptr[0]) == 1:
        return row_ptr - 1, col_ind - 1
    return row_ptr.copy(), col_ind.copy()


def _write_matrix_market(path: Path, n: int, row_ptr0: np.ndarray, col_ind0: np.ndarray, values: np.ndarray) -> None:
    nnz = int(values.size)
    with path.open("w", encoding="ascii") as f:
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write(f"{n} {n} {nnz}\n")
        for i in range(n):
            start = int(row_ptr0[i])
            end = int(row_ptr0[i + 1])
            for jj in range(start, end):
                # MatrixMarket uses 1-based row/col indices.
                f.write(f"{i + 1} {int(col_ind0[jj]) + 1} {values[jj]:.16e}\n")


def _write_partition_vector(path: Path, n: int, nranks: int) -> None:
    base = n // nranks
    rem = n % nranks
    part = np.empty(n, dtype=np.int32)
    cursor = 0
    for r in range(nranks):
        sz = base + (1 if r < rem else 0)
        part[cursor:cursor + sz] = r
        cursor += sz
    part.tofile(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create MatrixMarket file from binary CSR data.")
    parser.add_argument("data_dir", help="Directory containing row_ptr.bin, col_ind.bin, val.bin")
    parser.add_argument("--output", default="matrix.mtx", help="Output MatrixMarket filename (default: matrix.mtx)")
    parser.add_argument(
        "--write-partvec",
        type=int,
        default=0,
        metavar="NRANKS",
        help="If > 0, also write partition_vector.bin for NRANKS contiguous partitions",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    row_ptr_path = data_dir / "row_ptr.bin"
    col_ind_path = data_dir / "col_ind.bin"
    val_path = data_dir / "val.bin"

    if not row_ptr_path.exists() or not col_ind_path.exists() or not val_path.exists():
        _fail(f"Missing one or more files in {data_dir}: row_ptr.bin, col_ind.bin, val.bin")

    row_ptr = np.fromfile(row_ptr_path, dtype=np.int32)
    col_ind = np.fromfile(col_ind_path, dtype=np.int32)
    values = np.fromfile(val_path, dtype=np.float64)

    if row_ptr.size < 2:
        _fail("row_ptr.bin is too small")
    if col_ind.size != values.size:
        _fail("col_ind.bin and val.bin size mismatch")

    n = int(row_ptr.size - 1)
    nnz = int(values.size)

    row_ptr0, col_ind0 = _normalize_index_base(row_ptr, col_ind, n, nnz)

    if int(row_ptr0[0]) != 0 or int(row_ptr0[-1]) != nnz:
        _fail("CSR row_ptr appears invalid after normalization")
    if nnz > 0:
        cmin = int(col_ind0.min())
        cmax = int(col_ind0.max())
        if cmin < 0 or cmax >= n:
            _fail(f"Column indices out of range after normalization: min={cmin}, max={cmax}, n={n}")

    out_path = data_dir / args.output
    _write_matrix_market(out_path, n, row_ptr0, col_ind0, values)

    print(f"Wrote {out_path}")
    print(f"n={n}, nnz={nnz}")

    if args.write_partvec and args.write_partvec > 0:
        partvec_path = data_dir / "partition_vector.bin"
        _write_partition_vector(partvec_path, n, args.write_partvec)
        print(f"Wrote {partvec_path} for nranks={args.write_partvec}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

