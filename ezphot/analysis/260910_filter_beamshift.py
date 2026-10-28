#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260910_filter_beamshift.py

Apply the DR500 f/3.03 converging-beam blueshift (dr500_transmission.py in the log dir, n_eff = 1.5,
obscuration 0.59) to the vendor filter curves in Original_sources — the beam shift applies to the
interference filters ONLY (mirror/AR/QE are smooth in angle; the atmosphere is upstream of the optics).
Per-filter intrinsic offsets delta_f are NOT included here; they belong to the later assembly step.

Outputs in ~/ezphot/log/20260910_transmission_sources/:
  filters_beamshift/<name>.csv        beam-shifted curves (lam, trans), original wavelength grid
  filter_beamshift_summary.csv        per-filter centroid/edge shifts, area and peak ratios
  figures/F10_filter_beamshift.png    comparison plots
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402
import pandas as pd                 # noqa: E402

SRC = Path("~/ezphot/config/common/transmission/Original_sources").expanduser()
OUT = Path("~/ezphot/log/20260910_transmission_sources").expanduser()
sys.path.insert(0, str(OUT))
from dr500_transmission import dr500_transmission   # noqa: E402

N_EFF, FNUM, EPS = 1.5, 3.03, 0.59


def centroid(l, t):
    return np.trapezoid(l * t, l) / np.trapezoid(t, l)


def half_edges(l, t):
    h = t.max() / 2.0
    above = np.where(t > h)[0]
    i, j = above[0], above[-1]
    lo = np.interp(h, [t[i - 1], t[i]], [l[i - 1], l[i]]) if i > 0 else l[0]
    hi = np.interp(h, [t[j + 1], t[j]], [l[j + 1], l[j]]) if j < len(l) - 1 else l[-1]
    return lo, hi


def main():
    out_curves = OUT / "filters_beamshift"; out_curves.mkdir(exist_ok=True)
    files = sorted(p for p in SRC.glob("*.csv") if re.fullmatch(r"(m\d{3}w?|[ugriz])", p.stem))
    rows, curves = [], {}
    for p in files:
        d = pd.read_csv(p)
        lam, tr = d.iloc[:, 0].to_numpy(float), d.iloc[:, 1].to_numpy(float)
        o = np.argsort(lam); lam, tr = lam[o], tr[o]          # m466w is stored descending
        te = dr500_transmission(lam, tr, n_eff=N_EFF, f_number=FNUM, obscuration=EPS)
        pd.DataFrame(dict(lam=lam, trans=te)).to_csv(out_curves / f"{p.stem}.csv", index=False)
        curves[p.stem] = (lam, tr, te)
        b0, r0 = half_edges(lam, tr); b1, r1 = half_edges(lam, te)
        piv = centroid(lam, tr)
        rows.append(dict(filter=p.stem, pivot_nm=round(piv, 1),
                         dcentroid_nm=round(centroid(lam, te) - piv, 3),
                         dblue_edge_nm=round(b1 - b0, 3), dred_edge_nm=round(r1 - r0, 3),
                         peak_ratio=round(te.max() / tr.max(), 5),
                         area_ratio=round(np.trapezoid(te, lam) / np.trapezoid(tr, lam), 5)))
    summ = pd.DataFrame(rows).sort_values("pivot_nm")
    summ.to_csv(OUT / "filter_beamshift_summary.csv", index=False)
    med = summ[summ["filter"].str.match(r"m\d{3}w?")]
    print(summ.to_string(index=False))

    # ---- figure ----
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1, 1])
    cmap = plt.get_cmap("turbo")

    ax = fig.add_subplot(gs[0, :])
    for name, (lam, tr, te) in curves.items():
        if not re.fullmatch(r"m\d{3}w?", name):
            continue
        piv = float(med.loc[med["filter"] == name, "pivot_nm"].iloc[0])
        ax.plot(lam, te - tr, lw=0.8, color=cmap((piv - 370) / (890 - 370)))
    ax.set(xlim=(340, 960), xlabel="wavelength [nm]", ylabel="shifted - original",
           title=f"F10  beam-shifted (n_eff={N_EFF}, f/{FNUM}, eps={EPS}) minus vendor curve - 37 medium bands")
    ax.grid(alpha=0.3)

    for k, name in enumerate(["m400", "m875"]):
        ax = fig.add_subplot(gs[1, k])
        lam, tr, te = curves[name]
        ax.plot(lam, tr, "-", color="#888", lw=1.5, label="vendor (normal incidence)")
        ax.plot(lam, te, "-", color="#d62728", lw=1.5, label=f"beam-averaged (n_eff={N_EFF})")
        piv = float(med.loc[med["filter"] == name, "pivot_nm"].iloc[0])
        dc = float(med.loc[med["filter"] == name, "dcentroid_nm"].iloc[0])
        ax.set(xlim=(piv - 26, piv + 26), ylim=(0, 1.05), xlabel="wavelength [nm]", ylabel="transmission",
               title=f"{name}: centroid {dc:+.2f} nm")
        ax.legend(loc="lower center", fontsize=8); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[2, :])
    lgrid = np.linspace(360, 900, 50)
    ax.plot(lgrid, -0.402e-2 * lgrid, "-", color="#888", lw=1.5,
            label="-0.402% x lambda (n_eff=1.5 geometry)")
    ax.plot(lgrid, -0.386e-2 * lgrid, ":", color="#2ca02c", lw=1.5,
            label="-0.386% x lambda (measured k, effective-curve fit)")
    ax.plot(med.pivot_nm, med.dcentroid_nm, "o", ms=6, color="#d62728", label="applied centroid shift (medium bands)")
    ax.plot(med.pivot_nm, med.dblue_edge_nm, "v", ms=4, mfc="none", color="#1f77b4", label="blue-edge shift")
    ax.plot(med.pivot_nm, med.dred_edge_nm, "^", ms=4, mfc="none", color="#9467bd", label="red-edge shift")
    ax.set(xlim=(350, 910), xlabel="pivot wavelength [nm]", ylabel="shift [nm]",
           title="applied shift vs wavelength")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "figures" / "F10_filter_beamshift.png", dpi=150)
    print("saved ->", OUT / "figures" / "F10_filter_beamshift.png", "| curves:", out_curves)


if __name__ == "__main__":
    main()
