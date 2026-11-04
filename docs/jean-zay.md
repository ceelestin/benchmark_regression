# Running the simulated benchmark as SLURM arrays (Jean-Zay)

The chunked runs of the paper (see [../configs/README.md](../configs/README.md)) ran on the CPU partition of the
Jean-Zay cluster (IDRIS), `cpu_p1`, whose nodes have 40 cores. The benchmark is CPU-only and embarrassingly parallel
over seeds, so each configuration is split into seed chunks and each chunk runs on one node as one task of a SLURM job
array. The launchers `submit_<design>.slurm` at the repository root are written for Jean-Zay; on another cluster,
adapt the `#SBATCH` lines, the module and the environment.

The kit:

- `make_chunks.py`: splits configs into per-seed chunk configs and writes a manifest (one chunk path per line);
- `submit_<design>.slurm`: the job array of one design; task `i` runs line `i + 1` of its manifest;
- `merge_chunks.py`: merges the chunk parquets into one parquet per base config;
- `check_run_integrity.py`: checks that every run's `split_index` starts at 1 and follows `idx_rep` (guards against
  a cross-solver leak of the fold bookkeeping with benchopt <= 1.9.0 and `-j 1`, fixed in
  `Objective._reset_run_state`).

Placeholders below: `<project>` is your IDRIS project, `<login>` your login.

## 1. One-time setup (login node)

```bash
idracctinfo                                  # lists your projects; the CPU account is <project>@cpu
module load python/3.11.5                    # the version the paper's runs used
python -m venv $WORK/venvs/benchcv
source $WORK/venvs/benchcv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt              # pins benchopt==1.9.0, see the file for why
```

The launchers `cd "$WORK/benchmark_regression"` and activate `$WORK/venvs/benchcv`: clone the repository there, or edit
these two lines.

```bash
git clone https://github.com/ceelestin/benchmark_regression $WORK/benchmark_regression
cd $WORK/benchmark_regression
mkdir -p logs                                # SLURM writes logs/rank_<job>_<task>.{out,err} and does not create logs/
mkdir -p "$SCRATCH/ranking_outputs"
ln -sfn "$SCRATCH/ranking_outputs" outputs # benchopt writes outputs/<name>.parquet; keep the large files on $SCRATCH
```

`$SCRATCH` is purged after 30 days without access: copy the outputs elsewhere when the runs are done.

## 2. Seed chunks

The chunks of the paper's runs are tracked in `configs/chunks_<design>/` with their manifests, so the launchers can be
submitted as they are. For a new configuration:

```bash
python make_chunks.py --chunks 10 \
    --out-dir configs/chunks_mydesign --manifest configs/chunks_mydesign/manifest.txt \
    configs/my_config.yml
```

It prints the array range (`--array=0-9` here). Chunks are contiguous slices of the `seed: [...]` list; every other line
of the config is copied unchanged, so a chunk is a valid benchopt config on its own.

## 3. Edit the launcher

In `submit_<design>.slurm`, the lines marked `<<< EDIT`:

1. `#SBATCH --account=<project>@cpu`;
2. `#SBATCH --array=0-(N-1)%M`: N tasks from step 2, at most M nodes at once;
3. the `source` line that activates the environment.

Each task runs, on one full node (40 cores, one benchopt worker per core and single-threaded BLAS):

```bash
benchopt run . --config "$CFG" --no-plot --no-cache --no-timeout -j 40 --output "$NAME"
```

with `CFG` the manifest line of the task and `NAME` its base name, so the task writes `outputs/<config>__cNN.parquet`.

## 4. Test one task, then launch

```bash
sbatch --array=0 --qos=qos_cpu-dev --time=02:00:00 submit_<design>.slurm    # one task on the fast queue
squeue -u $USER
python check_run_integrity.py outputs/<config>__c00.parquet                  # must print OK
sbatch submit_<design>.slurm                                                 # the whole array
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS
```

Failed or timed-out tasks can be re-run alone: `sbatch --array=<i>,<j> submit_<design>.slurm`. A task close to the
20 h limit of `qos_cpu-t3` calls for more chunks.

## 5. Merge, check and read the outputs

The figure and table scripts read the chunk parquets directly (`<data root>/outputs/<config>__cNN.parquet`). To get one
file per config:

```bash
python merge_chunks.py --in-dir outputs --out-dir "$SCRATCH/ranking_merged"
python check_run_integrity.py "$SCRATCH"/ranking_merged/*.parquet
```

To point the analysis scripts at the outputs, make a data root whose sub-folders are links (layout in the main
README), for example:

```bash
mkdir -p $WORK/benchmark_data
ln -sfn "$SCRATCH/ranking_outputs" $WORK/benchmark_data/outputs
ln -sfn "$SCRATCH/ranking_merged"  $WORK/benchmark_data/merged
export DATA_ROOT=$WORK/benchmark_data
```

and copy the results to your machine with, for example,
`rsync -avz <login>@jean-zay.idris.fr:<path of the data root>/ <local data root>/`.

## Tuning

| Goal | Lever |
|---|---|
| Finish sooner | raise `%M` in `--array` (more nodes at once, within your QoS limit) |
| Task near the 20 h limit | more chunks (`make_chunks.py --chunks`), resubmit |
| Tasks longer than 20 h | `--qos=qos_cpu-t4 --time=99:00:00` (fewer nodes allowed) |
| Fewer, larger tasks | fewer chunks (watch the 20 h limit) |

The simulated datasets are generated in memory, so the compute nodes need no internet access; `--no-cache` avoids
contention on the shared file system.
