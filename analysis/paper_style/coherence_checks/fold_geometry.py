"""Appendix L: probability that the redundancy is defined after k splits, i.e. that some pair of the first
k random test folds shares at least two held-out observations (exact fold geometry, Monte Carlo).
Reproduces Appendix L's 60% / 95% at n_tr = 20 (study 25, folds of 5) and gives 73% / 99% for PCam at n_tr = 24
(study 30, folds of 6)."""
import random


def p_defined(n_study, n_test, k, reps=100000, seed=0):
    rng = random.Random(seed)
    ok = 0
    for _ in range(reps):
        folds = [set(rng.sample(range(n_study), n_test)) for _ in range(k)]
        ok += any(len(folds[a] & folds[b]) >= 2 for a in range(k) for b in range(a + 1, k))
    return ok / reps


if __name__ == "__main__":
    for lab, n_study, n_te in (("simulated n_tr=20", 25, 5), ("PCam n_tr=24", 30, 6)):
        print(lab, {k: round(p_defined(n_study, n_te, k), 3) for k in (3, 5)})
