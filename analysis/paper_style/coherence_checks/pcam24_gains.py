"""Item 9: PCam n_tr = 24 gains (paper estimator, plug-in on all seeds + 200 bootstrap draws) vs App M's claims."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))  # code/datapaths.py
import datapaths
import numpy as np, pandas as pd
T = datapaths.figs_path("pcam24", "tables")
OM = {"accuracy": {"densenet121": .065, "mobilenet_v2": .052, "vit_b_16": .073, "wide_resnet101_2": .033},
      "nll": {"densenet121": .061, "mobilenet_v2": .043, "vit_b_16": .085, "wide_resnet101_2": .031},
      "brier": {"densenet121": .077, "mobilenet_v2": .054, "vit_b_16": .085, "wide_resnet101_2": .039}}
print(sorted(os.listdir(T)))
rows = []
for loss in ("accuracy", "nll", "brier"):
    f = f"{T}/pcam_{loss}.parquet"
    if not os.path.exists(f): print("missing", f); continue
    r = pd.read_parquet(f)
    est = r[r.bootstrap_iter < 0]; dr = r[r.bootstrap_iter >= 0]
    for m in sorted(r.model.unique()):
        e = est[est.model == m].set_index("K").sample_gain
        q = dr[dr.model == m].groupby("K").sample_gain.quantile([0.025, 0.975]).unstack()
        g5, g20 = e.get(5), e.get(20)
        rd = (20 / g20 - 1) / 19
        om = OM[loss][m]
        rows.append({"loss": loss, "model": m, "N": int(r.N.iloc[0]), "n_tr": int(r.train_size.iloc[0]) if "train_size" in r else None,
                     "G5": round(g5, 2), "G20": round(g20, 2), "G20 95%": f"[{q.loc[20, 0.025]:.1f}, {q.loc[20, 0.975]:.1f}]",
                     "G20/G5": round(g20 / g5, 2), "omega": om, "pred K/(1+(K-1)w)": round(20 / (1 + 19 * om), 1),
                     "rho_delta": round(rd, 3), "kappa": round(rd - om, 3)})
print(pd.DataFrame(rows).to_string(index=False))
