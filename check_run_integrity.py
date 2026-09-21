"""Sanity-check a benchopt result file for cross-run leakage of the fold
bookkeeping (see Objective._reset_run_state).

Every (dataset, solver, objective-params) run must start at
``objective_split_index == 1`` with ``split_test_overlap_first_frac == 1.0``
and its split_index must equal ``idx_rep + 1`` on every row.  Exit code 1 if
any run violates this.

Usage: python check_run_integrity.py outputs/<name>.parquet [more.parquet ...]
"""
import sys
import pandas as pd
import pyarrow.parquet as pq

def check(path):
    names = pq.read_schema(path).names
    si = [c for c in names if c.endswith("split_index")]
    ov = [c for c in names if c.endswith("split_test_overlap_first_frac")]
    if not si:
        print(f"{path}: no split_index column, nothing to check")
        return True
    pcols = [c for c in names if c.startswith("p_obj_") or c.startswith("p_dataset_")]
    cols = ["solver_name", "dataset_name", "idx_rep"] + pcols + si + ov
    d = pd.read_parquet(path, columns=cols)
    key = ["dataset_name", "solver_name"] + pcols
    c = si[0]
    d["_expected"] = d["idx_rep"] + 1
    bad = d[d[c].notna() & (d[c] != d["_expected"])]
    n_runs = d.groupby(key, dropna=False).ngroups
    n_bad = bad.groupby(key, dropna=False).ngroups if len(bad) else 0
    n_null = int(d[c].isna().sum())
    msg = (f"{path}: runs={n_runs}, contaminated runs={n_bad}, "
           f"null split_index rows={n_null}")
    ok = n_bad == 0 and n_null == 0
    if ov:
        f0 = d[d.idx_rep == 0].groupby(key, dropna=False)[ov[0]].first()
        n_first_bad = int((f0 != 1.0).sum())
        msg += f", runs whose first split overlap != 1.0: {n_first_bad}"
        ok = ok and n_first_bad == 0
    print(("OK   " if ok else "FAIL ") + msg)
    if len(bad):
        print(bad.head(5)[key + ["idx_rep", c]].to_string(index=False))
    return ok

if __name__ == "__main__":
    sys.exit(0 if all([check(p) for p in sys.argv[1:]]) else 1)
