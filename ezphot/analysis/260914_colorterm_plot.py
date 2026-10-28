#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260914_colorterm_plot.py

F37v3: colour-term a(t) 2024-05→2026-04 (uniform ezphot photometry) with BROKEN y-axes —
outlier clusters (e.g. early-2024 m475 at −0.85) get their own thin strip separated by axis-break
slashes, so the mainstream trend stays readable.  Nightly-median line uses nights with >=5 frames.
Quality cut: dstar_err < 3 x per-filter median.

  MPLCONFIGDIR=/tmp/ezphot-matplotlib /home/hhchoi1022/miniconda3/envs/too/bin/python -u \\
      ezphot/analysis/260914_colorterm_plot.py
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
MJD_2025, MJD_2026 = 60676.0, 61041.0
XLIM = (60465, 61150)
MIN_NIGHT = 5


def breakmarks(ax_top, ax_bot):
    """diagonal slash marks at the junction of a broken axis."""
    d = 0.018
    kw = dict(color="k", clip_on=False, lw=0.9)
    ax_top.plot((-d, +d), (-d, +d), transform=ax_top.transAxes, **kw)
    ax_top.plot((1 - d, 1 + d), (-d, +d), transform=ax_top.transAxes, **kw)
    ax_bot.plot((-d, +d), (1 - d, 1 + d), transform=ax_bot.transAxes, **kw)
    ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_bot.transAxes, **kw)


def main():
    r = pd.read_csv(FOLDER / "offset_timeseries_ezall.csv")
    r = r[np.isfinite(r["a_phys"]) & np.isfinite(r["dstar_err"])]
    thr = r.groupby("filter")["dstar_err"].transform("median") * 3.0
    r = r[r["dstar_err"] < thr]

    order = sorted(r["filter"].unique(), key=lambda f: int(f[1:4]))
    units_all = sorted(r["unit"].unique())
    cmap = plt.get_cmap("tab20"); ucol = {un: cmap(i % 20) for i, un in enumerate(units_all)}

    fig = plt.figure(figsize=(22, 15))
    outer = fig.add_gridspec(5, 4, hspace=0.55, wspace=0.27, top=0.955, bottom=0.06, left=0.045, right=0.99)
    for k, filt in enumerate(order):
        s = r[r["filter"] == filt].sort_values("mjd")
        med = s["a_phys"].median()
        mad = 1.4826 * np.median(np.abs(s["a_phys"] - med))
        lo_b, hi_b = med - 7 * max(mad, 1e-3), med + 7 * max(mad, 1e-3)
        bulk = s[(s["a_phys"] >= lo_b) & (s["a_phys"] <= hi_b)]
        up = s[s["a_phys"] > hi_b]; dn = s[s["a_phys"] < lo_b]
        need_up = len(up) >= 15; need_dn = len(dn) >= 15
        heights = ([0.30] if need_up else []) + [1.0] + ([0.30] if need_dn else [])
        inner = outer[k // 4, k % 4].subgridspec(len(heights), 1, height_ratios=heights, hspace=0.09)
        axs = [fig.add_subplot(inner[i]) for i in range(len(heights))]
        i_main = 1 if need_up else 0
        axm = axs[i_main]

        # main panel
        blo, bhi = np.nanpercentile(bulk["a_phys"], [0.5, 99.5])
        pad = 0.10 * (bhi - blo)
        for un in sorted(bulk["unit"].unique()):
            su = bulk[bulk["unit"] == un]
            axm.scatter(su["mjd"], su["a_phys"], s=5, color=ucol[un], alpha=0.3, rasterized=True)
        ng = bulk.groupby("night").agg(mjd=("mjd", "median"), a=("a_phys", "median"), n=("a_phys", "size"))
        ng = ng[ng["n"] >= MIN_NIGHT].sort_values("mjd")
        axm.plot(ng["mjd"], ng["a"], "k-", lw=0.9, alpha=0.85)
        axm.axhline(0, color="red", lw=0.8, ls="--", alpha=0.7)
        axm.axvline(MJD_2026, color="k", lw=0.6, ls=":", alpha=0.6)
        axm.set_ylim(blo - pad, bhi + pad)
        axm.set_ylabel("a", fontsize=8)
        axm.grid(alpha=0.3)

        # outlier strips
        if need_up:
            axu = axs[0]
            ulo, uhi = np.nanpercentile(up["a_phys"], [2, 98]); upad = 0.2 * max(uhi - ulo, 1e-3)
            for un in sorted(up["unit"].unique()):
                su = up[up["unit"] == un]
                axu.scatter(su["mjd"], su["a_phys"], s=6, color=ucol[un], alpha=0.5, rasterized=True)
            axu.set_ylim(ulo - upad, uhi + upad)
            axu.set_yticks([round(float(np.median(up["a_phys"])), 2)])
            axu.text(0.99, 0.8, f"n={len(up)}", transform=axu.transAxes, ha="right", fontsize=6.5, color="#666")
            axu.grid(alpha=0.2)
            breakmarks(axu, axm)
            axu.spines["bottom"].set_visible(False); axm.spines["top"].set_visible(False)
            axu.tick_params(labelbottom=False, bottom=False, labelsize=7)
        if need_dn:
            axd = axs[-1]
            dlo, dhi = np.nanpercentile(dn["a_phys"], [2, 98]); dpad = 0.2 * max(dhi - dlo, 1e-3)
            for un in sorted(dn["unit"].unique()):
                su = dn[dn["unit"] == un]
                axd.scatter(su["mjd"], su["a_phys"], s=6, color=ucol[un], alpha=0.5, rasterized=True)
            axd.set_ylim(dlo - dpad, dhi + dpad)
            axd.set_yticks([round(float(np.median(dn["a_phys"])), 2)])
            axd.text(0.99, 0.8, f"n={len(dn)}", transform=axd.transAxes, ha="right", fontsize=6.5, color="#666")
            axd.grid(alpha=0.2)
            breakmarks(axm, axd)
            axm.spines["bottom"].set_visible(False); axd.spines["top"].set_visible(False)
            axm.tick_params(labelbottom=False, bottom=False)
        # x cosmetics: labels only on the lowest axis of each panel
        for a_ in axs:
            a_.set_xlim(*XLIM); a_.tick_params(labelsize=7)
        for a_ in axs[:-1]:
            a_.tick_params(labelbottom=False)
        e24 = s[s.mjd < MJD_2025]["a_phys"].median(); e25 = s[(s.mjd >= MJD_2025) & (s.mjd < MJD_2026)]["a_phys"].median()
        e26 = s[s.mjd >= MJD_2026]["a_phys"].median()
        axs[0].set_title(f"{filt}  a: 24:{e24:+.3f} 25:{e25:+.3f} 26:{e26:+.3f}", fontsize=8.5)

    h = [plt.Line2D([0], [0], marker="o", ls="", color=ucol[un]) for un in units_all]
    fig.legend(h, units_all, loc="lower center", ncol=min(len(units_all), 16), fontsize=8, frameon=False)
    fig.suptitle("Colour term a(t) 2024-05→2026-04 — broken y-axis for outlier clusters (slash marks), "
                 "nightly median needs ≥5 frames, err<3×med", y=0.985)
    fig.savefig(FOLDER / "F37v3_colorterm_2024_2026.png", dpi=130)
    print(f"saved -> F37v3_colorterm_2024_2026.png ({len(r)} frames)")


if __name__ == "__main__":
    main()
