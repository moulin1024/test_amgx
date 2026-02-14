#!/usr/bin/env python3
"""
Permute binary CSR/RHS data so partition ownership is contiguous.

Input directory must contain:
  - row_ptr.bin (int32)
  - col_ind.bin (int32)
  - val.bin     (float64)
  - rhs.bin     (float64)
  - partition_vector.bin (int32, global row -> partition id)

Output directory will contain permuted binaries:
  - row_ptr.bin, col_ind.bin, val.bin, rhs.bin
  - partition_vector.bin (contiguous by construction)
  - old_to_new.bin (int32)
  - new_to_old.bin (int32)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


def _fail(msg: str) -> None:
    raise RuntimeError(msg)


def _normalize_csr_index_base(
    row_ptr: np.ndarray, col_ind: np.ndarray, n: int, nnz: int
) -> tuple[np.ndarray, np.ndarray]:
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
    if int(row_ptr[0]) == 1:
        return row_ptr - 1, col_ind - 1
    return row_ptr.copy(), col_ind.copy()


def _build_permutation(part: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = int(part.size)
    old_idx = np.arange(n, dtype=np.int32)
    # Stable grouping by partition id; preserve relative order inside each partition.
    order = np.argsort(part, kind="stable").astype(np.int32)
    new_to_old = old_idx[order]
    old_to_new = np.empty(n, dtype=np.int32)
    old_to_new[new_to_old] = np.arange(n, dtype=np.int32)
    return old_to_new, new_to_old


def _permute_csr(
    row_ptr: np.ndarray,
    col_ind: np.ndarray,
    val: np.ndarray,
    rhs: np.ndarray,
    old_to_new: np.ndarray,
    new_to_old: np.ndarray,
    sort_cols: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = int(rhs.size)
    new_row_ptr = np.empty(n + 1, dtype=np.int32)
    new_row_ptr[0] = 0

    nnz_per_new_row = np.empty(n, dtype=np.int32)
    for new_r in range(n):
        old_r = int(new_to_old[new_r])
        nnz_per_new_row[new_r] = int(row_ptr[old_r + 1] - row_ptr[old_r])
    np.cumsum(nnz_per_new_row, out=new_row_ptr[1:])

    nnz = int(new_row_ptr[-1])
    new_col_ind = np.empty(nnz, dtype=np.int32)
    new_val = np.empty(nnz, dtype=np.float64)
    new_rhs = rhs[new_to_old].astype(np.float64, copy=False)

    for new_r in range(n):
        old_r = int(new_to_old[new_r])
        src_b = int(row_ptr[old_r])
        src_e = int(row_ptr[old_r + 1])
        dst_b = int(new_row_ptr[new_r])
        dst_e = int(new_row_ptr[new_r + 1])
        cols_old = col_ind[src_b:src_e]
        vals_old = val[src_b:src_e]
        cols_new = old_to_new[cols_old]

        if sort_cols and cols_new.size > 1:
            idx = np.argsort(cols_new, kind="stable")
            cols_new = cols_new[idx]
            vals_old = vals_old[idx]

        new_col_ind[dst_b:dst_e] = cols_new
        new_val[dst_b:dst_e] = vals_old

    return new_row_ptr, new_col_ind, new_val, new_rhs


def _contiguous_partvec_from_sorted_ids(sorted_part: np.ndarray) -> np.ndarray:
    return sorted_part.astype(np.int32, copy=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Permute CSR/RHS binaries so rows become contiguous by partition."
    )
    parser.add_argument("input_dir", help="Input directory with binary CSR/rhs/partition files")
    parser.add_argument("output_dir", help="Output directory for permuted binaries")
    parser.add_argument(
        "--partition",
        default="partition_vector.bin",
        help="Partition vector filename in input_dir (default: partition_vector.bin)",
    )
    parser.add_argument(
        "--sort-cols",
        action="store_true",
        help="Sort columns within each row after renumbering",
    )
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    row_ptr = np.fromfile(in_dir / "row_ptr.bin", dtype=np.int32)
    col_ind = np.fromfile(in_dir / "col_ind.bin", dtype=np.int32)
    val = np.fromfile(in_dir / "val.bin", dtype=np.float64)
    rhs = np.fromfile(in_dir / "rhs.bin", dtype=np.float64)
    part = np.fromfile(in_dir / args.partition, dtype=np.int32)

    if row_ptr.size < 2:
        _fail("row_ptr.bin is too small")
    n = int(row_ptr.size - 1)
    nnz = int(val.size)

    if rhs.size != n:
        _fail(f"rhs size mismatch: rhs={rhs.size}, n={n}")
    if col_ind.size != nnz:
        _fail(f"col_ind size mismatch: col_ind={col_ind.size}, val={nnz}")
    if part.size != n:
        _fail(f"partition size mismatch: partition={part.size}, n={n}")

    row_ptr0, col_ind0 = _normalize_csr_index_base(row_ptr, col_ind, n, nnz)
    if int(row_ptr0[0]) != 0 or int(row_ptr0[-1]) != nnz:
        _fail("CSR row_ptr invalid after normalization")

    pmin = int(part.min(initial=0))
    pmax = int(part.max(initial=0))
    if pmin < 0:
        _fail(f"partition ids must be non-negative, min={pmin}")

    old_to_new, new_to_old = _build_permutation(part)
    sorted_part = part[new_to_old]
    new_row_ptr, new_col_ind, new_val, new_rhs = _permute_csr(
        row_ptr0, col_ind0, val, rhs, old_to_new, new_to_old, sort_cols=args.sort_cols
    )
    new_part = _contiguous_partvec_from_sorted_ids(sorted_part)

    # Write permuted binaries.
    new_row_ptr.astype(np.int32, copy=False).tofile(out_dir / "row_ptr.bin")
    new_col_ind.astype(np.int32, copy=False).tofile(out_dir / "col_ind.bin")
    new_val.astype(np.float64, copy=False).tofile(out_dir / "val.bin")
    new_rhs.astype(np.float64, copy=False).tofile(out_dir / "rhs.bin")
    new_part.astype(np.int32, copy=False).tofile(out_dir / "partition_vector.bin")
    old_to_new.astype(np.int32, copy=False).tofile(out_dir / "old_to_new.bin")
    new_to_old.astype(np.int32, copy=False).tofile(out_dir / "new_to_old.bin")

    # Report block boundaries per partition id.
    uniq, counts = np.unique(new_part, return_counts=True)
    print(f"Wrote permuted data to {out_dir}")
    print(f"n={n}, nnz={nnz}, partition_ids=[{pmin}, {pmax}], nparts_present={uniq.size}")
    print(
        f"per-part rows: min={int(counts.min(initial=0))}, "
        f"max={int(counts.max(initial=0))}, avg={float(counts.mean()):.1f}"
    )
    if uniq.size > 0:
        offsets = np.zeros(int(uniq.size) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(counts, dtype=np.int64)
        print("partition contiguous ranges (new row indices):")
        for i, pid in enumerate(uniq.tolist()):
            b = int(offsets[i])
            e = int(offsets[i + 1])
            print(f"  part {pid}: [{b}, {e})")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
