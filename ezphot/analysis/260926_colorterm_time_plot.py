#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260926_colorterm_time_plot.py — colour term vs time, one figure per transmission-curve version, one panel per filter.

  dots   : per-image colour term a(delta = 0) [mmag / mag], coloured by unit (same unit colours in every panel/figure)
  line   : median per (unit, night) with >= 2 images
  band   : +-5 mmag/mag; y range per filter is the same in every version so offsets between versions are visible
           (range from the 2-98 % of unit-night medians; images outside it are drawn as triangles on the edge)
  suspect: unit-nights with >= 2 images whose median is > max(100, 10 MAD) mmag/mag from the filter median
           (likely wrong filter label / wheel position) -> CT_<tag>_suspect.csv, drawn as crosses, excluded from medians
Usage:
  python 260926_colorterm_time_plot.py <frames.csv> <outdir> <tag> csv_variant=label [csv_variant=label ...]
e.g. colorterm_time_frames.csv (variants manufacturer_nobeam ...) or delta_star_v2_frames.csv (manufacturer_beam_vac ...)
Output: <outdir>/CT_<tag>_<label>.png, <outdir>/CT_<tag>_unit_night.csv, <outdir>/CT_<tag>_suspect.csv
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time

GRID = ["m400", "m425", "m450", "m475", "m500", "m525", "m550", "m575", "m600", "m625",
        "m650", "m675", "m700", "m725", "m750", "m775", "m800", "m825", "m850", "m875"]
TITLE = dict(manufacturer_nobeam="manufacturer curve, no beam", manufacturer_beam="manufacturer curve + f/3 beam (n_eff 1.5)",
             mhj_nobeam="manufacturer + lab offset (mhj), no beam", mhj_beam="manufacturer + lab offset (mhj) + f/3 beam",
             hyp_nobeam="lab-measured curve (version_hyp), no beam", hyp_beam="lab-measured curve (version_hyp) + f/3 beam",
             ky_nobeam="lab-measured curve (version_ky), no beam", ky_beam="lab-measured curve (version_ky) + f/3 beam")

if __name__ == "__main__":
    csv, outdir, tag = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    vmap = dict(a.split("=") for a in sys.argv[4:])
    outdir.mkdir(parents=True, exist_ok=True)
    A = pd.read_csv(csv, usecols=["filter", "file", "unit", "night", "mjd", "variant", "a0"])
    A = A[A.variant.isin(vmap) & np.isfinite(A.a0)].copy()
    A["label"] = A.variant.map(vmap)
    A["a"] = 1e3 * A.a0
    A["date"] = Time(A.mjd.to_numpy(), format="mjd").datetime
    units = sorted(A.unit.unique())
    cmap = plt.get_cmap("tab20")
    ucol = {u: cmap(i % 20) for i, u in enumerate(units)}
    UN = (A.groupby(["label", "filter", "unit", "night"])
            .agg(n=("a", "size"), a=("a", "median"), mjd=("mjd", "median")).reset_index())
    # suspect unit-nights: judged on the first version, applied to all
    ref = UN[UN.label == list(vmap.values())[0]]
    fm = ref.groupby("filter").a.median()
    fmad = ref.groupby("filter").a.apply(lambda v: 1.4826 * np.median(np.abs(v - v.median())))
    ref = ref.assign(dev=ref.a - ref["filter"].map(fm))
    SUS = ref[(ref.n >= 2) & (np.abs(ref.dev) > np.maximum(100, 10 * ref["filter"].map(fmad)))]
    SUS.drop(columns="label").to_csv(outdir / f"CT_{tag}_suspect.csv", index=False)
    skey = set(zip(SUS["filter"], SUS.unit, SUS.night))
    A["suspect"] = [k in skey for k in zip(A["filter"], A.unit, A.night)]
    UN = UN[(UN.n >= 2) & ~pd.Series([k in skey for k in zip(UN["filter"], UN.unit, UN.night)], index=UN.index)]
    # common y range per filter over all versions (from unit-night medians)
    yr = {}
    for f, g in UN.groupby("filter"):
        lo, hi = np.nanpercentile(g.a, [2, 98])
        pad = 0.35 * (hi - lo) + 5
        yr[f] = (min(lo - pad, -15), max(hi + pad, 15))
    UN["date"] = Time(UN.mjd.to_numpy(), format="mjd").datetime
    UN.drop(columns="date").to_csv(outdir / f"CT_{tag}_unit_night.csv", index=False)
    t0, t1 = A.date.min(), A.date.max()
    present = sorted(A["filter"].unique(), key=lambda f: int(f[1:4]))
    panels = GRID if set(present) <= set(GRID) else present
    nrow = int(np.ceil(len(panels) / 4))
    for label in vmap.values():
        fig, axs = plt.subplots(nrow, 4, figsize=(22, 4.2 * nrow), sharex=True, squeeze=False)
        for ax in axs.flat[len(panels):]:
            ax.set_visible(False)
        for ax, f in zip(axs.flat, panels):
            ga = A[(A.label == label) & (A["filter"] == f)]
            g = ga[~ga.suspect]
            u = UN[(UN.label == label) & (UN["filter"] == f)]
            y0, y1 = yr.get(f, (-50, 50))
            ax.axhspan(-5, 5, color="tab:green", alpha=0.10, zorder=0)
            ax.axhline(0, color="k", lw=0.6)
            for un, gu in g.groupby("unit"):
                ax.plot(gu.date, gu.a.clip(y0, y1), ".", ms=1.5, alpha=0.25, color=ucol[un], rasterized=True)
                off = gu[(gu.a < y0) | (gu.a > y1)]
                ax.plot(off.date, np.where(off.a > y1, y1, y0), "^", ms=3, alpha=0.5, color=ucol[un])
            gs = ga[ga.suspect]
            if len(gs):
                ax.plot(gs.date, np.clip(gs.a, y0, y1), "x", ms=5, color="k", mew=1.0,
                        label=f"suspect label: {len(gs)} images")
            for un, uu in u.sort_values("mjd").groupby("unit"):
                ax.plot(uu.date, uu.a, "o", ms=3, color=ucol[un], mec="k", mew=0.3,
                        label=f"{un} ({len(g[g.unit == un])}, med {g[g.unit == un].a.median():+.1f})")
            ax.set_ylim(y0, y1)
            ax.set_xlim(t0, t1)
            med = g.a.median()
            nout = int(((g.a < y0) | (g.a > y1)).sum())
            ax.set_title(f"{f}   median {med:+.1f} mmag/mag   ({len(g)} images, {nout} off scale)", fontsize=10)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=6.5, loc="upper left", ncol=2, markerscale=1.2, handletextpad=0.2, columnspacing=0.8)
            short = (t1 - t0).days < 400
            ax.xaxis.set_major_locator(mdates.MonthLocator() if short else mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
        for ax in axs[:, 0]:
            ax.set_ylabel("colour term a [mmag / mag BP−RP]")
        fig.suptitle(f"Colour term vs time — {TITLE.get(label, label)}  (vacuum; SNR ≥ 100 stars, corrected Gaia XP; "
                     f"dots = images, circles = unit-night medians, ▲ = off scale, × = suspect wrong-filter nights; legend: unit (images, median))", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.1 / nrow))
        fig.savefig(outdir / f"CT_{tag}_{label}.png", dpi=90)
        plt.close(fig)
        print("saved", outdir / f"CT_{tag}_{label}.png")
