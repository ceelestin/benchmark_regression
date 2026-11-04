"""Where the analysis, figure and table scripts find their inputs.

Four locations, each set by an environment variable; a script's --data-root (or --figs-dir) argument, when it has
one, takes precedence over the variable. Relative values are resolved against the current directory.

DATA_ROOT       the run outputs (default: data/ at the root of this repository):
                  outputs/      benchopt outputs of the chunked simulated runs, <config>__cNN.parquet
                  merged/       the same runs merged per config by merge_chunks.py, <config>.parquet
                  paper_runs/   single-file simulated runs (May and July 2026), benchopt_run_<date>.parquet
                  ranking/      the 12 simulated runs of Figure O.1, benchopt_run_<date>.parquet
                  pcam/         PatchCamelyon run outputs (results_output_pcam/, stratandnewmetrics/, ...)
                  nlp/          Yelp run outputs (yelp_runs_a/, yelp_runs_b/, yelp_qwen3_0.6b/, yelp_aligned/)
                The README of the repository lists every sub-folder and the runs it holds.
FIGS_DIR        what the figure scripts write and the table scripts read back (default: figs/ at the root):
                  sim/          paper_style_figs.py --out-dir (simulated gain figures and their tables/)
                  real/         regen_real_data_gain_figures.py --out-dir (PCam and Yelp gain figures, tables/)
BENCHMARK_PCAM  a checkout of github.com/ceelestin/benchmark_pcam (default: benchmark_pcam/ next to this
                repository): its analysis outputs and its results_output_pcam/ folder.
BENCHMARK_NLP   a checkout of github.com/ceelestin/benchmark_nlp (default: benchmark_nlp/ next to this repository).
"""
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))


def _location(env, default, value=None):
    return os.path.abspath(os.path.expanduser(value or os.environ.get(env) or default))


def data_root(root=None):
    """The data root: ``root`` if given, else $DATA_ROOT, else <repository>/data."""
    return _location("DATA_ROOT", os.path.join(REPO, "data"), root)


def data_path(*parts, root=None):
    """A path under the data root, e.g. data_path("pcam", "results_output_pcam")."""
    return os.path.join(data_root(root), *parts)


def figs_dir(root=None):
    """The figure-output root: ``root`` if given, else $FIGS_DIR, else <repository>/figs."""
    return _location("FIGS_DIR", os.path.join(REPO, "figs"), root)


def figs_path(*parts, root=None):
    """A path under the figure-output root, e.g. figs_path("sim", "tables")."""
    return os.path.join(figs_dir(root), *parts)


def repo_path(*parts):
    """A path inside this repository, e.g. repo_path("analysis", "out_omega")."""
    return os.path.join(REPO, *parts)


def pcam_repo_path(*parts):
    """A path inside the benchmark_pcam checkout ($BENCHMARK_PCAM, else ../benchmark_pcam)."""
    return os.path.join(_location("BENCHMARK_PCAM", os.path.join(os.path.dirname(REPO), "benchmark_pcam")), *parts)


def nlp_repo_path(*parts):
    """A path inside the benchmark_nlp checkout ($BENCHMARK_NLP, else ../benchmark_nlp)."""
    return os.path.join(_location("BENCHMARK_NLP", os.path.join(os.path.dirname(REPO), "benchmark_nlp")), *parts)
