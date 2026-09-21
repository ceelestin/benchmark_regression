#!/usr/bin/env python3
"""Recombine per-seed chunk parquets into one parquet per base config.

After the SLURM array finishes, ``$SCRATCH/ranking_outputs`` holds many small
``<base>__cNN.parquet`` files. This streams them (one chunk at a time, so memory
stays low) into a single ``<base>.parquet`` per base config, recreating the
familiar few-files layout the analysis (`ranking.py`) expects.

Requires only ``pyarrow`` (already a benchopt dependency).

Example
-------
    python merge_chunks.py \\
        --in-dir  "$SCRATCH/ranking_outputs" \\
        --out-dir "$SCRATCH/ranking_merged"
"""
import argparse
import glob
import os
import re

import pyarrow.parquet as pq

CHUNK_SUFFIX = re.compile(r"__c\d+$")


def base_of(path):
    name = os.path.splitext(os.path.basename(path))[0]
    return CHUNK_SUFFIX.sub("", name)


def main():
    scratch = os.environ.get("SCRATCH", ".")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", default=os.path.join(scratch, "ranking_outputs"))
    ap.add_argument("--out-dir", default=os.path.join(scratch, "ranking_merged"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.in_dir, "*.parquet")))
    if not files:
        raise SystemExit(f"No parquet files in {args.in_dir}")

    groups = {}
    for f in files:
        groups.setdefault(base_of(f), []).append(f)

    for base, fs in sorted(groups.items()):
        out = os.path.join(args.out_dir, base + ".parquet")
        writer = None
        n_rows = 0
        for f in sorted(fs):
            table = pq.read_table(f)
            if writer is None:
                writer = pq.ParquetWriter(out, table.schema)
            writer.write_table(table)
            n_rows += table.num_rows
        writer.close()
        print(f"{base}: merged {len(fs)} chunks, {n_rows:,} rows -> {out}")


if __name__ == "__main__":
    main()
