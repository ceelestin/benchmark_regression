"""Unit checks for the incremental cross-fold statistics in objective.py.

Run with ``python tests/test_incremental_stats.py`` (or pytest) from the
benchmark root, inside an environment where benchopt is importable.

Checks, on random fold stacks:
1. ``_icc_from_stack_incremental`` reproduces ``_icc_from_stack(np.vstack)``
   (mean off-diagonal covariance, mean variance, rho) at every fold count.
2. ``_oof_anova_summary`` reproduces a direct per-sample computation of the
   sample-driven and fold-driven variance components.
3. The noisy estimators are deterministic per sample and per fit, and
   ``noise = 0`` reproduces the base learner exactly.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import importlib.util  # noqa: E402

from benchmark_utils.noisy_estimators import (  # noqa: E402
    NoisyCoefRegressor, NoisyPredictionRegressor,
)
from sklearn.linear_model import Ridge  # noqa: E402


def _objective():
    # Bare Objective instance: the helpers under test only rely on the
    # per-run caches they create themselves and on ``n_splits``.
    spec = importlib.util.spec_from_file_location(
        "bench_objective", os.path.join(ROOT, "objective.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    obj = module.Objective.__new__(module.Objective)
    obj.n_splits = 6
    return obj


def test_incremental_icc_matches_full(K=7, n=500, seed=0):
    rng = np.random.default_rng(seed)
    obj = _objective()
    base = rng.normal(size=n)
    rows = [base + 0.5 * rng.normal(size=n) for _ in range(K)]
    worst = 0.0
    for k in range(1, K + 1):
        inc = obj._icc_from_stack_incremental("t", [rows[k - 1]])
        full = obj._icc_from_stack(np.vstack(rows[:k]))
        if k == 1:
            assert inc == (None, None, None) and full == (None, None, None)
            continue
        for a, b in zip(inc, full):
            worst = max(worst, abs(a - b) / max(abs(b), 1e-12))
    assert worst < 1e-10, worst
    return worst


def test_oof_anova_matches_direct(K=8, n=300, seed=1):
    rng = np.random.default_rng(seed)
    obj = _objective()
    a = rng.normal(size=n)                     # sample-driven component
    rows, masks = [], []
    for _ in range(K):
        mask = rng.random(n) < 0.35
        row = a + 0.7 * rng.normal(size=n)     # + fold-driven component
        rows.append(row)
        masks.append(mask)
        obj._oof_anova_update({"q": row}, mask)
    summary = obj._oof_anova_summary()
    # direct computation
    R = np.vstack(rows)
    M = np.vstack(masks)
    count = M.sum(axis=0)
    seen = count >= 2
    means = np.array([R[M[:, i], i].mean() for i in range(n)])[seen]
    within = np.array([R[M[:, i], i].var(ddof=1) for i in range(n)])[seen]
    fold_var = within.mean()
    sample_var = max(np.var(means, ddof=1) - np.mean(within / count[seen]), 0)
    assert abs(summary["q"]["fold_var"] - fold_var) < 1e-10
    assert abs(summary["q"]["sample_var"] - sample_var) < 1e-10
    assert summary["n_samples_seen_twice"] == int(seen.sum())
    # the ICC should sit near the true share a-var / (a-var + 0.49)
    true_icc = 1.0 / (1.0 + 0.49)
    assert abs(summary["q"]["icc"] - true_icc) < 0.15, summary["q"]
    return summary["q"]


def test_noisy_estimators(seed=2):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(200, 5))
    y = X @ rng.normal(size=5) + 0.3 * rng.normal(size=200)
    base = Ridge(alpha=1.0).fit(X, y)
    # noise 0 == base
    for est in (NoisyPredictionRegressor(Ridge(alpha=1.0), 0.0),
                NoisyCoefRegressor(Ridge(alpha=1.0), 0.0)):
        est.fit(X, y)
        assert np.allclose(est.predict(X), base.predict(X))
    # per-sample determinism: same rows -> same noise, across calls/arrays
    est = NoisyPredictionRegressor(Ridge(alpha=1.0), 1.0).fit(X, y)
    p_all = est.predict(X)
    p_sub = est.predict(X[50:120])
    assert np.allclose(p_all[50:120], p_sub)
    noise = p_all - base.predict(X)
    assert abs(noise.std() / np.std(y) - 1.0) < 0.25, noise.std() / np.std(y)
    assert abs(np.corrcoef(noise, base.predict(X))[0, 1]) < 0.2
    # different training sets -> different noise
    est2 = NoisyPredictionRegressor(Ridge(alpha=1.0), 1.0).fit(X[:150], y[:150])
    noise2 = est2.predict(X) - est2.model_.predict(X)
    assert abs(np.corrcoef(noise, noise2)[0, 1]) < 0.2
    # coef noise: prediction differs, intercept/coef shape preserved
    estc = NoisyCoefRegressor(Ridge(alpha=1.0), 0.5).fit(X, y)
    assert estc.model_.coef_.shape == base.coef_.shape
    assert not np.allclose(estc.predict(X), base.predict(X))
    return noise.std() / np.std(y)


if __name__ == "__main__":
    print("incremental ICC vs full: max rel err", test_incremental_icc_matches_full())
    print("oof anova vs direct:", test_oof_anova_matches_direct())
    print("noisy estimators OK; pred-noise std ratio", test_noisy_estimators())
    print("ALL OK")
