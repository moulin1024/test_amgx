#!/usr/bin/env python3
"""
Generate partition_vector.bin (global row -> rank) from binary CSR input.

Input directory must contain:
  - row_ptr.bin (int32)
  - col_ind.bin (int32)

By default, this script tries to use pymetis (k-way graph partitioning).
If pymetis is unavailable, pass --fallback contiguous.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


def _fail(msg: str) -> None:
    raise RuntimeError(msg)


def _normalize_csr_index_base(
    row_ptr: np.ndarray, col_ind: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return zero-based CSR arrays, auto-detecting one-based input."""
    nnz = int(col_ind.size)
    if row_ptr.size != n + 1:
        _fail(f"row_ptr length mismatch: got {row_ptr.size}, expected {n + 1}")

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

    # Heuristic fallback for mixed data.
    if int(row_ptr[0]) == 1:
        return row_ptr - 1, col_ind - 1
    return row_ptr.copy(), col_ind.copy()


def _contiguous_partition(n: int, nparts: int) -> np.ndarray:
    base = n // nparts
    rem = n % nparts
    out = np.empty(n, dtype=np.int32)
    cursor = 0
    for r in range(nparts):
        size = base + (1 if r < rem else 0)
        out[cursor : cursor + size] = r
        cursor += size
    return out


def _drop_invalid_columns(
    row_ptr: np.ndarray, col_ind: np.ndarray, n: int, drop_diagonal: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Filter invalid column indices; optionally drop diagonal entries."""
    new_row_ptr = np.empty_like(row_ptr)
    kept_cols: list[np.ndarray] = []
    cursor = 0
    new_row_ptr[0] = 0

    for i in range(n):
        b = int(row_ptr[i])
        e = int(row_ptr[i + 1])
        cols = col_ind[b:e]
        mask = (cols >= 0) & (cols < n)
        if drop_diagonal:
            mask &= cols != i
        cols_kept = cols[mask]
        kept_cols.append(cols_kept.astype(np.int32, copy=False))
        cursor += int(cols_kept.size)
        new_row_ptr[i + 1] = cursor

    if cursor == 0:
        new_col_ind = np.empty(0, dtype=np.int32)
    else:
        new_col_ind = np.concatenate(kept_cols).astype(np.int32, copy=False)
    return new_row_ptr.astype(np.int32, copy=False), new_col_ind


def _metis_partition(row_ptr: np.ndarray, col_ind: np.ndarray, nparts: int) -> np.ndarray:
    try:
        import pymetis  # type: ignore
    except Exception as exc:
        _fail(
            "pymetis is not available. Install it or use --fallback contiguous. "
            f"Import error: {exc}"
        )

    # pymetis expects Python sequences.
    xadj = row_ptr.tolist()
    adjncy = col_ind.tolist()
    _, membership = pymetis.part_graph(nparts, xadj=xadj, adjncy=adjncy)
    return np.asarray(membership, dtype=np.int32)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate partition_vector.bin from binary CSR using METIS (pymetis)."
    )
    parser.add_argument("data_dir", help="Directory containing row_ptr.bin and col_ind.bin")
    parser.add_argument(
        "--nparts",
        type=int,
        required=True,
        help="Number of partitions (typically MPI ranks)",
    )
    parser.add_argument(
        "--output",
        default="partition_vector.bin",
        help="Output filename inside data_dir (default: partition_vector.bin)",
    )
    parser.add_argument(
        "--fallback",
        choices=("error", "contiguous"),
        default="error",
        help="Behavior if METIS backend unavailable (default: error)",
    )
    parser.add_argument(
        "--drop-diagonal",
        action="store_true",
        help="Drop diagonal entries from graph before partitioning",
    )
    args = parser.parse_args()

    if args.nparts <= 0:
        _fail("--nparts must be > 0")

    data_dir = Path(args.data_dir)
    row_ptr_path = data_dir / "row_ptr.bin"
    col_ind_path = data_dir / "col_ind.bin"
    if not row_ptr_path.exists() or not col_ind_path.exists():
        _fail(f"Missing row_ptr.bin or col_ind.bin in {data_dir}")

    row_ptr = np.fromfile(row_ptr_path, dtype=np.int32)
    col_ind = np.fromfile(col_ind_path, dtype=np.int32)
    if row_ptr.size < 2:
        _fail("row_ptr.bin is too small")

    n = int(row_ptr.size - 1)
    row_ptr0, col_ind0 = _normalize_csr_index_base(row_ptr, col_ind, n)

    if int(row_ptr0[0]) != 0 or int(row_ptr0[-1]) != int(col_ind0.size):
        _fail("CSR row_ptr appears invalid after normalization")

    row_ptr_f, col_ind_f = _drop_invalid_columns(
        row_ptr0, col_ind0, n, drop_diagonal=args.drop_diagonal
    )
    if int(row_ptr_f[-1]) != int(col_ind_f.size):
        _fail("Filtered CSR is inconsistent")

    try:
        part = _metis_partition(row_ptr_f, col_ind_f, args.nparts)
        backend = "pymetis"
    except Exception as exc:
        if args.fallback == "contiguous":
            part = _contiguous_partition(n, args.nparts)
            backend = "contiguous-fallback"
            print(f"WARNING: METIS failed, using contiguous partition. reason: {exc}")
        else:
            raise

    if part.size != n:
        _fail(f"Invalid partition size: got {part.size}, expected {n}")
    if part.min(initial=0) < 0 or part.max(initial=0) >= args.nparts:
        _fail(
            f"Invalid partition labels: min={int(part.min(initial=0))} "
            f"max={int(part.max(initial=0))} nparts={args.nparts}"
        )

    out_path = data_dir / args.output
    part.astype(np.int32, copy=False).tofile(out_path)

    counts = np.bincount(part, minlength=args.nparts)
    imbalance = float(counts.max(initial=0)) / max(1.0, float(counts.mean()))
    print(f"Wrote {out_path}")
    print(f"backend={backend}, n={n}, nparts={args.nparts}, nnz_graph={int(col_ind_f.size)}")
    print(
        f"partition sizes: min={int(counts.min(initial=0))} "
        f"max={int(counts.max(initial=0))} imbalance(max/avg)={imbalance:.4f}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
