# Running the 1000-seed ranking benchmark on Jean-Zay

The ranking benchmark is **CPU-only** (scikit-learn `ExtraTrees`, single-threaded
fits) and embarrassingly parallel over seeds. On a 256-core node each config
takes ~60 h; the 4 configs are ~240 h total. This kit splits each config's 1000
seeds into chunks and runs them as a **SLURM job array** across many `cpu_p1`
nodes (40 cores each), cutting wall time to ~10–40 h depending on how many nodes
run at once.

Files in this kit:
- `make_chunks.py`  — split configs into per-seed chunk configs + a manifest
- `submit_ranking.slurm` — the cpu_p1 job-array script (edit 3 marked lines)
- `merge_chunks.py` — recombine chunk parquets into one file per base config

---

## 0. Drop the redundant config (saves ~25%)

`configs/ranking_1000seeds.yml`'s 7 train sizes (`10,30,100,300,1000,3000,10000`)
are **all already covered** by `part1` + `part2` + `part3`, which are mutually
disjoint and together cover the full 25-size sweep exactly once. So **run only
parts 1–3** — the main config is pure duplicate compute. (To include it anyway,
just add `configs/ranking_1000seeds.yml` to the `make_chunks.py` call below and
bump the array range.)

> Heads-up for the analysis side: because `main` overlaps the parts, mixing it
> in would put some `(train_size, seed)` rows in two parquets. Running parts 1–3
> only keeps every row unique.

---

## 1. One-time setup (login node — has internet)

```bash
# Find your absolute paths and your CPU allocation
echo $WORK ; echo $SCRATCH
idracctinfo                      # shows your project(s); the account is <proj>@cpu

# Python env (venv is lightest on Jean-Zay; conda also fine)
module avail python              # pick an available version
module load python/3.11.5        # (adjust to what `module avail` shows)
python -m venv $WORK/venvs/benchcv
source $WORK/venvs/benchcv/bin/activate
pip install --upgrade pip
pip install benchopt scikit-learn numpy scipy pandas pyarrow
```

Copy this benchmark repo to `$WORK` **from your laptop** (rsync preserves your
local configs + any solver edits — better than a fresh git clone):

```bash
JZ=<login>@jean-zay.idris.fr
WORKDIR=$(ssh "$JZ" 'echo $WORK')
rsync -avz --exclude .git --exclude __pycache__ --exclude outputs \
  "/home/ceve/Documents/benchmark_regression/" \
  "$JZ:$WORKDIR/benchmark_regression/"
```

Back on the **login node**, create the output/log dirs and point benchopt's
`outputs/` at `$SCRATCH` (fast, big; copy results off it within ~30 days):

```bash
cd $WORK/benchmark_regression
mkdir -p logs "$SCRATCH/ranking_outputs"
ln -sfn "$SCRATCH/ranking_outputs" outputs     # benchopt writes here via -o
```

---

## 2. Generate the seed-chunks

```bash
cd $WORK/benchmark_regression
python make_chunks.py --chunks 40 \
  configs/ranking_1000seeds_part1.yml \
  configs/ranking_1000seeds_part2.yml \
  configs/ranking_1000seeds_part3.yml
```

This prints the array range, e.g. `--array=0-119` (3 configs × 40 chunks = 120
tasks, 25 seeds each ≈ ~10 h/task given your 60 h/256-core baseline).

---

## 3. Edit `submit_ranking.slurm` (3 lines)

1. `#SBATCH --account=XXX@cpu`   → your `<proj>@cpu` from `idracctinfo`
2. `#SBATCH --array=0-119%48`    → `0-(N-1)` from step 2; `%48` = max nodes at once
3. `source "$WORK/venvs/benchcv/bin/activate"` → your env

`%` controls concurrency vs. queue pressure: `%48` ≈ 3 waves ≈ ~30 h; raise it
(up to your QoS node limit) to finish faster — `%120` runs all at once (~10 h).

---

## 4. Smoke-test one task first (cheap, fast queue)

```bash
sbatch --array=0 --qos=qos_cpu-dev --time=02:00:00 submit_ranking.slurm
# watch it:
squeue -u $USER
tail -f logs/rank_*_0.out
```

When it finishes, check the parquet looks right and note how long it took — if a
task is approaching 19 h, re-run `make_chunks.py` with more `--chunks`.

```bash
ls -la outputs/ranking_1000seeds_part1__c00.parquet
python -c "import pyarrow.parquet as pq; \
  print(pq.ParquetFile('outputs/ranking_1000seeds_part1__c00.parquet').metadata.num_rows, 'rows')"
```

---

## 5. Launch the full array

```bash
sbatch submit_ranking.slurm
squeue -u $USER                  # monitor
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS   # after/while running
```

Failed/timed-out tasks can be re-run individually:
`sbatch --array=<id1>,<id2> submit_ranking.slurm`.

---

## 6. Merge and bring results home

```bash
# on the login node
cd $WORK/benchmark_regression
python merge_chunks.py \
  --in-dir  "$SCRATCH/ranking_outputs" \
  --out-dir "$SCRATCH/ranking_merged"
# -> ranking_1000seeds_part1.parquet, _part2.parquet, _part3.parquet
```

Then from your laptop:

```bash
rsync -avz "$JZ:$SCRATCH/ranking_merged/*.parquet" \
  "/home/ceve/Documents/Run outputs/Regression/data/ranking_1000seeds/"
```

Point the analysis at them — either edit `DEFAULT_DATA_PATHS` in `ranking.py`, or
pass a glob:

```bash
python ranking.py --data-paths \
  "/home/ceve/Documents/Run outputs/Regression/data/ranking_1000seeds/"*.parquet
```

---

## Tuning cheatsheet

| Goal | Lever |
|------|-------|
| Finish sooner | raise `%NN` in `--array` (more concurrent nodes) |
| Task hits 19 h walltime | more `--chunks` (smaller tasks); resubmit |
| Need >20 h tasks | `--qos=qos_cpu-t4 --time=99:00:00` (fewer nodes allowed) |
| Add the redundant `main` config | add it to `make_chunks.py` + widen `--array` |
| Fewer, larger tasks | fewer `--chunks` (watch the 20 h cap) |

Notes: solvers run single-threaded (`OMP_NUM_THREADS=1`) with 40 benchopt
workers per node; `sim_linear` data is generated in-memory so **no internet is
needed on compute nodes**; `--no-cache` avoids shared-FS cache contention.
