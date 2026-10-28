#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260911_offset_replot_ez.py

Final F28ez: the delta* trend 2024-05 -> 2026-04 from ONE uniform source — ezphot direct photometry (10",
select_stars-aligned cuts, corrected-XP reference, physical curve).  Per-filter panels coloured by unit
(filter-copy rotations visible), nightly medians, old global delta_f dashed red, 2026-01 marker.
Also prints a per-filter epoch table (2024 / 2025 / 2026+).

  MPLCONFIGDIR=/tmp/ezphot-matplotlib /home/hhchoi1022/miniconda3/envs/too/bin/python -u \\
      ezphot/analysis/260911_offset_replot_ez.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402
import pandas as pd                 # noqa: E402

warnings.filterwarnings("ignore")
FOLDER = Path("/home/hhchoi1022/ezphot/log/20260911_offset_timeseries")
FIT = pd.read_csv("/home/hhchoi1022/ezphot/log/20260910_effective_curves/offset_plus_linear_summary.csv").set_index("filter")
MJD_2025, MJD_2026 = 60676.0, 61041.0


def main():
    r = pd.read_csv(FOLDER / "offset_timeseries_ezall.csv")
    r = r[np.isfinite(r.dstar)]
    print(f"{len(r)} frames, MJD {r.mjd.min():.0f}..{r.mjd.max():.0f}, filters {r['filter'].nunique()}")
    order = sorted(r["filter"].unique(), key=lambda f: int(f[1:4]))
    units_all = sorted(r["unit"].unique())
    cmap = plt.get_cmap("tab20"); ucol = {un: cmap(i % 20) for i, un in enumerate(units_all)}

    ncol = 4; nrow = int(np.ceil(len(order) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 2.8 * nrow), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    rows = []
    for ax, filt in zip(axes, order):
        s = r[r["filter"] == filt].sort_values("mjd")
        for un in sorted(s["unit"].unique()):
            su = s[s["unit"] == un]
            ax.scatter(su["mjd"], su["dstar"], s=5, color=ucol[un], alpha=0.35, label=un, rasterized=True)
        ng = s.groupby("night").agg(mjd=("mjd", "median"), d=("dstar", "median"))
        ax.plot(ng["mjd"], ng["d"], "k-", lw=0.7, alpha=0.75)
        go = float(FIT.loc[filt, "offset"]) if filt in FIT.index else np.nan
        if np.isfinite(go):
            ax.axhline(go, color="red", lw=1.2, ls="--", alpha=0.85)
        ax.axvline(MJD_2026, color="k", lw=0.6, ls=":", alpha=0.6)
        ax.axhline(0, color="grey", lw=0.5)
        med = {"e24": s[s.mjd < MJD_2025].dstar.median(), "e25": s[(s.mjd >= MJD_2025) & (s.mjd < MJD_2026)].dstar.median(),
               "e26": s[s.mjd >= MJD_2026].dstar.median()}
        rows.append(dict(filter=filt, pivot=int(filt[1:4]), global_off=round(go, 2),
                         d2024=round(med["e24"], 2), d2025=round(med["e25"], 2), d2026=round(med["e26"], 2),
                         n=len(s)))
        ax.set_title(f"{filt}  δ_f={go:+.2f} | 24:{med['e24']:+.1f} 25:{med['e25']:+.1f} 26:{med['e26']:+.1f}", fontsize=8)
        ax.set_ylabel("δ* [nm]", fontsize=8); ax.grid(alpha=0.3)
        lo, hi = np.nanpercentile(s["dstar"], [2, 98])
        ax.set_ylim(lo - 1, hi + 1)
    for ax in axes[len(order):]:
        ax.axis("off")
    h, l = [], []
    for un in units_all:                     # build one legend across all panels
        h.append(plt.Line2D([0], [0], marker="o", ls="", color=ucol[un])); l.append(un)
    fig.legend(h, l, loc="lower center", ncol=min(len(units_all), 14), fontsize=8, frameon=False)
    fig.suptitle("δ* 2024-05 → 2026-04 — UNIFORM ezphot direct photometry (10\"), colour = unit; "
                 "red dashed = old global δ_f (2024-08 std fields); dotted = 2026-01", y=0.998)
    fig.tight_layout(rect=[0, 0.03, 1, 0.985])
    fig.savefig(FOLDER / "F28ez_offset_2024_2026.png", dpi=130)
    T = pd.DataFrame(rows)
    T.to_csv(FOLDER / "offset_ez_2024_2026_summary.csv", index=False)
    print(T.to_string(index=False))
    print(f"\nsaved -> F28ez_offset_2024_2026.png, offset_ez_2024_2026_summary.csv in {FOLDER}")


if __name__ == "__main__":
    main()
