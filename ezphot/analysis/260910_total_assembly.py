#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260910_total_assembly.py

Assemble the purely physical prediction of the 7DT medium-band total transmission — NO per-filter intrinsic
offsets (delta_f), by decision (2026-09-10):

    T_phys(lambda_air) = F_beam x QE x optics x atm_ElSauce

  F_beam : vendor filter curve with the f/3.03 converging-beam average (n_eff = 1.5, eps = 0.59)
           -> <log>/filters_beamshift/<f>.csv  (260910_filter_beamshift.py)
  QE     : Original_sources/QE.csv (IMX455, Gill+2022-consistent)
  optics : Original_sources/optics.efficiency.dr350.dr500.csv 'total' (7 nodes, linear interp;
           linearly extrapolated below 400 nm — the weakest assumption, flagged)
  atm    : <log>/atmosphere_elsauce.csv 'trans' (El Sauce 1580 m, X = 1.23, PWV = 7.5), on lam_air

Comparisons per band against (a) the nominal curves in ~/ezphot/config/common/transmission/<f>.csv
(Angstrom grid, Final_transmission generation) and (b) the fitted effective curves in
ezphot/analysis/effective_curves/<f>.csv. Metrics: throughput-weighted centroid difference and
photon-integral throughput ratio in mag.

Outputs in ~/ezphot/log/20260910_transmission_sources/:
  total_curves_physical/<f>.csv    assembled curves (lam [nm, air], trans), trimmed to band support
  total_assembly_summary.csv       per-band metrics
  figures/F11_total_assembly.png
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402
import pandas as pd                 # noqa: E402

SRC = Path("~/ezphot/config/common/transmission/Original_sources").expanduser()
NOM = Path("~/ezphot/config/common/transmission").expanduser()
EFF = Path("/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis/effective_curves")
OUT = Path("~/ezphot/log/20260910_transmission_sources").expanduser()


def log(msg):
    print(time.strftime("[%Y-%m-%d %H:%M:%S] ") + msg, flush=True)


def centroid(l, t):
    return np.trapezoid(l * t, l) / np.trapezoid(t, l)


def photon_integral(l, t):
    return np.trapezoid(t / l, l)


def main():
    out_curves = OUT / "total_curves_physical"; out_curves.mkdir(exist_ok=True)

    atm = pd.read_csv(OUT / "atmosphere_elsauce.csv")
    grid = atm.lam_air.to_numpy()
    sel = (grid >= 330) & (grid <= 1010)
    grid, t_atm = grid[sel], atm.trans.to_numpy()[sel]

    qe = pd.read_csv(SRC / "QE.csv")
    t_qe = np.interp(grid, qe.iloc[:, 0], qe.iloc[:, 1])

    opt = pd.read_csv(SRC / "optics.efficiency.dr350.dr500.csv", sep=r"\s+", comment="#", skiprows=11)
    opt.columns = ["nm", "K9LA", "LaK4L", "FK95N", "VIS", "Al", "Total_pct", "total"]
    t_opt = np.interp(grid, opt.nm, opt.total)
    slope = (opt.total.iloc[1] - opt.total.iloc[0]) / (opt.nm.iloc[1] - opt.nm.iloc[0])
    below = grid < opt.nm.iloc[0]
    t_opt[below] = np.maximum(opt.total.iloc[0] + slope * (grid[below] - opt.nm.iloc[0]), 0.0)

    s_new = t_qe * t_opt * t_atm

    rows = []
    for p in sorted((OUT / "filters_beamshift").glob("m*.csv")):
        name = p.stem
        fb = pd.read_csv(p)
        t_f = np.interp(grid, fb.lam, fb.trans, left=0, right=0)
        t_phys = t_f * s_new
        m = t_f > 1e-4
        i0 = max(np.argmax(m) - 50, 0); i1 = min(len(m) - np.argmax(m[::-1]) + 50, len(grid))
        pd.DataFrame(dict(lam=grid[i0:i1], trans=t_phys[i0:i1])).to_csv(out_curves / f"{name}.csv", index=False)

        row = dict(filter=name, pivot_nm=round(centroid(grid, t_phys), 2))
        for tag, path, ang in [("nom", NOM / f"{name}.csv", True), ("eff", EFF / f"{name}.csv", False)]:
            if not path.exists():
                row[f"dcen_{tag}_nm"] = np.nan; row[f"dmag_{tag}"] = np.nan
                continue
            d = pd.read_csv(path)
            l_ref = d.iloc[:, 0].to_numpy(float) / (10.0 if ang else 1.0)
            t_ref = np.interp(grid, l_ref, d.iloc[:, 1].to_numpy(float), left=0, right=0)
            row[f"dcen_{tag}_nm"] = round(centroid(grid, t_phys) - centroid(grid, t_ref), 3)
            row[f"dmag_{tag}"] = round(-2.5 * np.log10(photon_integral(grid, t_phys)
                                                       / photon_integral(grid, t_ref)), 4)
            if tag == "eff":
                t_eff_plot = t_ref
        rows.append(row)
        if name in ("m475", "m725"):
            globals()[f"_zoom_{name}"] = (t_f, t_phys, t_eff_plot)
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "total_assembly_summary.csv", index=False)
    print(summ.to_string(index=False))

    # ---- figure ----
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])

    for k, name in enumerate(["m475", "m725"]):
        ax = fig.add_subplot(gs[0, k])
        t_f, t_phys, t_eff = globals()[f"_zoom_{name}"]
        nomd = pd.read_csv(NOM / f"{name}.csv")
        piv = float(summ.loc[summ["filter"] == name, "pivot_nm"].iloc[0])
        ax.plot(nomd.iloc[:, 0] / 10.0, nomd.iloc[:, 1], color="#888", lw=1.0, label="nominal (config)")
        ax.plot(grid, t_eff, color="#2ca02c", lw=1.0, label="effective (fitted)")
        ax.plot(grid, t_phys, color="#d62728", lw=0.7, alpha=0.9, label="physical (this work)")
        ax.set(xlim=(piv - 25, piv + 25), ylim=(0, 0.65), xlabel="wavelength [nm]", ylabel="total transmission",
               title=name)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, :])
    lg = np.linspace(360, 900, 40)
    ax.axhline(0, color="k", lw=0.5)
    ax.plot(lg, -0.402e-2 * lg, "-", color="#888", lw=1, label="-0.402% x lambda (beam line)")
    ax.plot(summ.pivot_nm, summ.dcen_nom_nm, "o", color="#555", ms=6, label="physical - nominal")
    ax.plot(summ.pivot_nm, summ.dcen_eff_nm, "s", color="#d62728", ms=6, label="physical - effective (= -applied delta_f)")
    for _, rr in summ.iterrows():
        ax.annotate(rr["filter"], (rr["pivot_nm"], rr["dcen_nom_nm"]), fontsize=6, color="#555",
                    rotation=90, ha="center", va="top", xytext=(0, -6), textcoords="offset points")
        ax.annotate(rr["filter"], (rr["pivot_nm"], rr["dcen_eff_nm"]), fontsize=6, color="#d62728",
                    rotation=90, ha="center", va="bottom", xytext=(0, 6), textcoords="offset points")
    ax.set(xlim=(350, 910), ylim=(-4.7, 2.4), xlabel="pivot [nm]", ylabel="centroid difference [nm]",
           title="F11  centroid: the physical curve vs the two references")
    ax.legend(fontsize=9, loc="lower left"); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[2, :])
    ax.axhline(0, color="k", lw=0.5)
    ax.plot(summ.pivot_nm, summ.dmag_nom * 1000, "o", color="#555", ms=6, label="physical vs nominal")
    ax.plot(summ.pivot_nm, summ.dmag_eff * 1000, "s", color="#d62728", ms=6, label="physical vs effective")
    ax.set(xlim=(350, 910), xlabel="pivot [nm]", ylabel="throughput difference [mmag]",
           title="photon-integral throughput (positive = physical curve is fainter); grey scale is absorbed by ZP")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "figures" / "F11_total_assembly.png", dpi=150)
    log(f"saved -> {out_curves} (37 curves), total_assembly_summary.csv, figures/F11_total_assembly.png")


if __name__ == "__main__":
    main()
