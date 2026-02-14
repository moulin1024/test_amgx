#!/usr/bin/env python3
"""
Plot sparsity pattern from binary CSR matrix data.

Input directory must contain:
  - row_ptr.bin (int32 CSR row pointer)
  - col_ind.bin (int32 CSR column indices)
  - val.bin     (float64 CSR values) [optional for plotting, but validated if present]

Usage:
  python3 plot_sparsity_from_bin.py /path/to/data
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
    if int(row_ptr[0]) == 1:
        return row_ptr - 1, col_ind - 1
    return row_ptr.copy(), col_ind.copy()


def _build_row_indices(row_ptr: np.ndarray) -> np.ndarray:
    n = int(row_ptr.size - 1)
    counts = np.diff(row_ptr).astype(np.int64, copy=False)
    rows = np.repeat(np.arange(n, dtype=np.int32), counts)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot matrix sparsity pattern from binary CSR files.")
    parser.add_argument("data_dir", help="Directory containing row_ptr.bin and col_ind.bin")
    parser.add_argument(
        "--output",
        default="sparsity.png",
        help="Output image filename (placed in data_dir if relative). Default: sparsity.png",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=2_000_000,
        help="Maximum nonzeros to draw (random sample if nnz is larger). Default: 2000000",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=0.1,
        help="Scatter marker size. Default: 0.1",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Scatter alpha in [0,1]. Default: 0.6",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for point sampling. Default: 0",
    )
    args = parser.parse_args()

    if args.max_points <= 0:
        _fail("--max-points must be > 0")
    if not (0.0 <= args.alpha <= 1.0):
        _fail("--alpha must be in [0,1]")
    if args.point_size <= 0.0:
        _fail("--point-size must be > 0")

    data_dir = Path(args.data_dir)
    row_ptr_path = data_dir / "row_ptr.bin"
    col_ind_path = data_dir / "col_ind.bin"
    val_path = data_dir / "val.bin"

    if not row_ptr_path.exists() or not col_ind_path.exists():
        _fail(f"Missing row_ptr.bin or col_ind.bin in {data_dir}")

    row_ptr = np.fromfile(row_ptr_path, dtype=np.int32)
    col_ind = np.fromfile(col_ind_path, dtype=np.int32)
    vals = np.fromfile(val_path, dtype=np.float64) if val_path.exists() else None

    if row_ptr.size < 2:
        _fail("row_ptr.bin is too small")

    n = int(row_ptr.size - 1)
    nnz = int(col_ind.size)
    if vals is not None and vals.size != nnz:
        _fail(f"val.bin size mismatch: val={vals.size}, col_ind={nnz}")

    row_ptr0, col_ind0 = _normalize_index_base(row_ptr, col_ind, n, nnz)
    if int(row_ptr0[0]) != 0 or int(row_ptr0[-1]) != nnz:
        _fail("CSR row_ptr appears invalid after normalization")
    if nnz > 0:
        cmin = int(col_ind0.min())
        cmax = int(col_ind0.max())
        if cmin < 0 or cmax >= n:
            _fail(f"Column index out of range after normalization: min={cmin}, max={cmax}, n={n}")

    rows = _build_row_indices(row_ptr0)
    cols = col_ind0.astype(np.int32, copy=False)
    if rows.size != cols.size:
        _fail("Internal error: row/col point arrays size mismatch")

    used = rows.size
    sampled = False
    if used > args.max_points:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(used, size=args.max_points, replace=False)
        rows = rows[idx]
        cols = cols[idx]
        used = int(args.max_points)
        sampled = True

    # Delay import so users can still validate input even without matplotlib installed.
    import matplotlib.pyplot as plt

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = data_dir / out_path

    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
    ax.scatter(cols, rows, s=args.point_size, c="black", alpha=args.alpha, linewidths=0)
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    title = f"Sparsity pattern (n={n}, nnz={nnz}"
    if sampled:
        title += f", plotted={used} sampled"
    else:
        title += f", plotted={used}"
    title += ")"
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    print(f"Wrote {out_path}")
    print(f"n={n}, nnz={nnz}, plotted_points={used}, sampled={sampled}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
