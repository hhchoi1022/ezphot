#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260926_adelta_all.py — direct colour-term sensitivity for all 20 filters: a(delta) on -15..+15 nm (0.5 nm) for the
images of the tiles with the widest filter coverage (same stars / curves / solver inputs as 260926_m500_adelta.py,
beam + vacuum curve, atmosphere column fixed).  Replaces the rough median(-a0/delta*) estimate of the report table.

  per filter: n images, slope of the median a(delta) over |delta| <= 6 nm, local slope at delta = 0
  ((a(+1) - a(-1)) / 2), median per-image error of a at delta = 0, fraction of images whose a(delta) crosses zero more
  than once within +-15 nm, zero crossings of the median curve
Output (~/ezphot/log/20260926_delta_star_v2/): adelta_all.csv, adelta_all_summary.csv
"""
import importlib
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
MA = importlib.import_module("260926_m500_adelta")
DS, OUT, DG, VAR = MA.DS, MA.OUT, MA.DG, MA.VAR
N_TILES, PER_FILTER = 24, 3


def crossings(v):
    s = np.sign(v)
    return int((s[1:] != s[:-1]).sum())


if __name__ == "__main__":
    F = pd.read_csv(OUT / "delta_star_v2_frames.csv")
    F = F[(F.variant == VAR) & (F.n >= 30)]
    cov = F.groupby("tile")["filter"].nunique().sort_values(ascending=False)
    tiles = list(cov.index[:N_TILES])
    items = []
    for t in tiles:
        g = F[F.tile == t].sort_values("n", ascending=False).groupby("filter").head(PER_FILTER)
        items.append((t, list(g[["file", "filter"]].itertuples(index=False, name=None))))
    print(f"{len(tiles)} tiles, filters per tile {cov.iloc[:N_TILES].min()}-{cov.iloc[:N_TILES].max()}, "
          f"{sum(len(i[1]) for i in items)} images", flush=True)
    with Pool(len(items), initializer=MA.init) as pool:
        res = pool.map(MA.tile_frames, items)
    A = pd.DataFrame([r for rows, _ in res for r in rows])
    A.to_csv(OUT / "adelta_all.csv", index=False)
    rows = []
    for filt, g in A.groupby("filter"):
        W = g.pivot_table(index="file", columns="delta", values="a") * 1e3
        med = W.median(axis=0)
        inner = np.abs(DG) <= 6
        nc = np.array([crossings(r.to_numpy()) for _, r in W.iterrows()])
        mc = [float(DG[i] + (DG[i + 1] - DG[i]) * med.iloc[i] / (med.iloc[i] - med.iloc[i + 1]))
              for i in range(len(DG) - 1) if np.sign(med.iloc[i]) != np.sign(med.iloc[i + 1])]
        rows.append(dict(filter=filt, pivot=int(filt[1:]), n_images=len(W), n_tiles=g.tile.nunique(),
                         slope6=np.polyfit(DG[inner], med.to_numpy()[inner], 1)[0],
                         slope0=(med[1.0] - med[-1.0]) / 2, a0=med[0.0], a_err0=1e3 * g[g.delta == 0].a_err.median(),
                         frac_multi=(nc > 1).mean(), median_crossings=";".join(f"{c:+.1f}" for c in mc)))
    S = pd.DataFrame(rows).sort_values("pivot")
    S["dstar_err_1img"] = S.a_err0 / np.abs(S.slope6)
    S.to_csv(OUT / "adelta_all_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(S.round(2).to_string(index=False))
