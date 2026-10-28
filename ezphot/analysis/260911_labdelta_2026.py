#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260911_labdelta_2026.py

Compare the on-sky optimal offset delta* restricted to 2026+ (mjd >= 61041, i.e. the currently-installed filter
copies — the epoch most comparable to the lab measurements) with the lab-measured delta_f (LAB_DELTA).
Uses the per-frame delta* already computed by 260911_offset_timeseries.py — no recomputation.

x = lab delta_f, y = median delta*(2026+); error bar = robust scatter of nightly medians / sqrt(N nights).
Red filters (pivot >= 675 nm) are leverage-less (err 7-16 nm) -> greyed out and excluded from corr/rms.

  MPLCONFIGDIR=/tmp/ezphot-matplotlib /home/hhchoi1022/miniconda3/envs/too/bin/python -u \\
      ezphot/analysis/260911_labdelta_2026.py
"""
from __future__ import annotations

import importlib
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402
import pandas as pd                 # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
L = importlib.import_module("260911_colorterm_labdelta")

FOLDER = Path("/home/hhchoi1022/ezphot/log/20260911_offset_timeseries")
MJD_2026 = 61041.0                                  # 2026-01-01


def main():
    r = pd.read_csv(FOLDER / "offset_timeseries.csv")
    r = r[r.mjd >= MJD_2026]
    rows = []
    for f, g in r.groupby("filter"):
        ng = g.groupby("night").dstar.median()      # nightly medians
        med = float(ng.median())
        rob = 1.4826 * float(np.median(np.abs(ng - ng.median())))
        err = rob / np.sqrt(max(len(ng), 1))
        rows.append(dict(filter=f, pivot=int(f[1:4]), lab=L.LAB_DELTA.get(f, np.nan),
                         dstar=round(med, 3), err=round(err, 3), n_frames=len(g), n_nights=len(ng),
                         units=";".join(sorted(g.unit.unique()))))
    T = pd.DataFrame(rows).sort_values("pivot").reset_index(drop=True)
    T.to_csv(FOLDER / "labdelta_2026_compare.csv", index=False)
    print("=== delta*(2026+) vs lab delta_f ===")
    print(T.to_string(index=False))

    good = T[(T["pivot"] < 675) & np.isfinite(T["lab"])]
    dif = good.dstar - good.lab
    corr = float(np.corrcoef(good.lab, good.dstar)[0, 1])
    rms = float(np.sqrt(np.mean(dif ** 2)))
    g2 = good[good["filter"] != "m500"]                        # known shape-anomaly outlier
    corr2 = float(np.corrcoef(g2.lab, g2.dstar)[0, 1])
    rms2 = float(np.sqrt(np.mean((g2.dstar - g2.lab) ** 2)))
    print(f"\npivot<675 ({len(good)} filters): corr={corr:.2f}, rms(dstar-lab)={rms:.2f} nm, "
          f"median offset={np.median(dif):+.2f} nm | excl m500: corr={corr2:.2f}, rms={rms2:.2f}")

    fig, ax = plt.subplots(figsize=(7.5, 7))
    lim = (-6.5, 6.5)
    ax.plot(lim, lim, "k--", lw=0.8, alpha=0.6, label="y = x")
    ax.plot(lim, [lim[0] + np.median(dif), lim[1] + np.median(dif)], ":", color="#9467bd", lw=1.2,
            label=f"y = x {np.median(dif):+.2f} nm (median)")
    red = T[T["pivot"] >= 675]
    ax.errorbar(red.lab, red.dstar, yerr=red.err, fmt="o", ms=6, color="#bbbbbb", capsize=2,
                label="m675+ (no leverage, excluded)")
    ax.errorbar(good.lab, good.dstar, yerr=good.err, fmt="o", ms=7, color="#d62728", capsize=2,
                label=f"m400-m650: corr {corr:.2f}, rms {rms:.2f} nm\n(excl m500: corr {corr2:.2f}, rms {rms2:.2f} nm)")
    for _, s in T.iterrows():
        if np.isfinite(s.lab):
            ax.annotate(s["filter"], (s.lab, s.dstar), textcoords="offset points", xytext=(6, 4), fontsize=8,
                        color="#666" if s["pivot"] >= 675 else "#000")
    ax.set(xlim=lim, ylim=lim, xlabel="lab δ_f [nm]", ylabel="on-sky δ* (2026+, colour-term zero) [nm]",
           title="Lab-measured δ_f vs on-sky δ* — 2026+ epoch (currently installed copies)")
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout(); fig.savefig(FOLDER / "F29_labdelta_2026.png", dpi=150)
    print(f"\nsaved -> F29_labdelta_2026.png, labdelta_2026_compare.csv in {FOLDER}")


if __name__ == "__main__":
    main()
