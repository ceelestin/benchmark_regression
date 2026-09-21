#!/usr/bin/env python3
"""Split benchopt ranking configs into per-seed chunks for a SLURM job array.

Each ranking config has a single-line ``seed: [0, 1, ..., 999]`` list. This
script slices that list into ``--chunks`` contiguous pieces and writes one
config per piece, leaving every other line byte-for-byte identical. A manifest
listing every chunk config (one path per line) is written so the SLURM array
can pick its config with ``sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" manifest``.

Stdlib only (no PyYAML) — the seed list is rewritten with a regex so this runs
on a bare Python on the cluster login node.

Example
-------
    python make_chunks.py --chunks 40 \\
        configs/ranking_1000seeds_part1.yml \\
        configs/ranking_1000seeds_part2.yml \\
        configs/ranking_1000seeds_part3.yml
"""
import argparse
import os
import re

# Matches a single-line ``seed: [ ... ]`` list (the only multi-hundred-element
# list in these configs).
SEED_RE = re.compile(r"seed:\s*\[[^\]]*\]")


def split_even(items, k):
    """Split *items* into *k* contiguous, near-equal pieces (drops empties)."""
    n = len(items)
    pieces = [items[i * n // k:(i + 1) * n // k] for i in range(k)]
    return [p for p in pieces if p]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="+", help="Base config YAMLs to chunk.")
    ap.add_argument("--chunks", type=int, default=40,
                    help="Number of seed-chunks per config (default: 40).")
    ap.add_argument("--out-dir", default="configs/chunks",
                    help="Where to write chunk configs (default: configs/chunks).")
    ap.add_argument("--manifest", default="configs/chunks/manifest.txt",
                    help="Manifest path listing every chunk config.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = []
    for cfg in args.configs:
        with open(cfg) as fh:
            text = fh.read()
        m = SEED_RE.search(text)
        if m is None:
            raise SystemExit(f"No 'seed: [...]' list found in {cfg}")
        seeds = [s.strip() for s in m.group(0)[m.group(0).index("[") + 1:-1].split(",")
                 if s.strip()]
        base = os.path.splitext(os.path.basename(cfg))[0]
        pieces = split_even(seeds, args.chunks)
        for i, piece in enumerate(pieces):
            new_seed = "seed: [" + ", ".join(piece) + "]"
            chunk_text = text[:m.start()] + new_seed + text[m.end():]
            name = f"{base}__c{i:02d}"
            path = os.path.join(args.out_dir, name + ".yml")
            with open(path, "w") as fh:
                fh.write(chunk_text)
            manifest.append(path)
        print(f"{cfg}: {len(seeds)} seeds -> {len(pieces)} chunks "
              f"(~{len(seeds) // len(pieces)} seeds each)")

    with open(args.manifest, "w") as fh:
        fh.write("\n".join(manifest) + "\n")
    n = len(manifest)
    print(f"\nWrote {n} chunk configs to {args.out_dir}/")
    print(f"Manifest: {args.manifest}")
    print(f"SLURM array range: --array=0-{n - 1}")


if __name__ == "__main__":
    main()
