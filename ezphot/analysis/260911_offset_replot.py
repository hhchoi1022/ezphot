#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260911_offset_replot.py

Merge the delta* frame CSVs from all three sources — 2024-08 standard campaign (the old fit's own data),
2024 pipeline survey, 2025-26 pipeline survey — and redraw the F28 per-filter time series over the full
2024-2026 span, so the old global fitted delta_f (dashed red) can be compared against delta* AT ITS OWN EPOCH.

  MPLCONFIGDIR=/tmp/ezphot-matplotlib /home/hhchoi1022/miniconda3/envs/too/bin/python -u \\
      ezphot/analysis/260911_offset_replot.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402
import pandas as pd                 # noqa: E402

warnings.filterwarnings("ignore")
FOLDER = Path("/home/hhchoi1022/ezphot/log/20260911_offset_timeseries")
FITDF = pd.read_csv("/home/hhchoi1022/ezphot/log/20260910_effective_curves/offset_plus_linear_summary.csv").set_index("filter")
SOURCES = [("offset_timeseries_2024early.csv", "pipe2024"),      # 2024-05..06 (all frames with cats)
           ("offset_timeseries_std2024.csv", "std2024"),
           ("offset_timeseries_2024pipe.csv", "pipe2024"),
           ("offset_timeseries.csv", "survey")]
MJD_SWAP = 61041.0                                  # 2026-01-01 (m400 swap epoch marker)


def main():
    parts = []
    for fn, src in SOURCES:
        p = FOLDER / fn
        if p.exists():
            d = pd.read_csv(p); d["src"] = src; parts.append(d)
            print(f"  {fn}: {len(d)} frames, MJD {d.mjd.min():.0f}..{d.mjd.max():.0f}")
        else:
            print(f"  {fn}: MISSING (skipped)")
    r = pd.concat(parts, ignore_index=True)
    order = sorted(r["filter"].unique(), key=lambda f: int(f[1:4]))

    ncol = 4; nrow = int(np.ceil(len(order) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 2.8 * nrow), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    scol = {"std2024": "#2ca02c", "pipe2024": "#ff7f0e", "survey": "#1f77b4"}
    rows = []
    for ax, filt in zip(axes, order):
        s = r[r["filter"] == filt].sort_values("mjd")
        for src, g in s.groupby("src"):
            ax.scatter(g["mjd"], g["dstar"], s=6, color=scol[src], alpha=0.25, label=src, rasterized=True)
            ng = g.groupby("night").agg(mjd=("mjd", "median"), d=("dstar", "median"))
            ax.plot(ng["mjd"], ng["d"], ".", ms=4, color=scol[src], alpha=0.9)
        go = float(FITDF.loc[filt, "offset"]) if filt in FITDF.index else np.nan
        if np.isfinite(go):
            ax.axhline(go, color="red", lw=1.2, ls="--", alpha=0.85)
        ax.axvline(MJD_SWAP, color="k", lw=0.6, ls=":", alpha=0.6)
        ax.axhline(0, color="grey", lw=0.5)
        med_by = {src: float(g.dstar.median()) for src, g in s.groupby("src")}
        rows.append(dict(filter=filt, pivot=int(filt[1:4]), global_off=round(go, 2),
                         **{f"dstar_{k}": round(v, 2) for k, v in med_by.items()}))
        txt = "  ".join(f"{k[:1]}24:{v:+.1f}" if "2024" in k else f"25+:{v:+.1f}" for k, v in med_by.items())
        ax.set_title(f"{filt}  δ_f={go:+.2f} | {txt}", fontsize=8)
        ax.set_ylabel("δ* [nm]", fontsize=8); ax.grid(alpha=0.3)
        lo, hi = np.nanpercentile(s["dstar"], [2, 98])
        ax.set_ylim(lo - 1, hi + 1)
    for ax in axes[len(order):]:
        ax.axis("off")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=10, frameon=False, markerscale=2.5)
    fig.suptitle("δ* (colour-term zero on physical curve) 2024→2026 — green=2024-08 std campaign (old-fit data), "
                 "orange=2024 pipeline, blue=2025-26 survey; red dashed=old global δ_f; dotted=2026-01", y=0.998)
    fig.tight_layout(rect=[0, 0.02, 1, 0.985])
    fig.savefig(FOLDER / "F28full_offset_2024_2026.png", dpi=130)
    T = pd.DataFrame(rows)
    T.to_csv(FOLDER / "offset_2024_2026_summary.csv", index=False)
    print("\n=== per-filter medians by source vs old global δ_f ===")
    print(T.to_string(index=False))
    print(f"\nsaved -> F28full_offset_2024_2026.png, offset_2024_2026_summary.csv in {FOLDER}")


if __name__ == "__main__":
    main()
