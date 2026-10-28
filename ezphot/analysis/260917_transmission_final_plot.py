#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260917_transmission_final_plot.py

Two presentation plots of the 7DT total transmission curves, assembled purely from
~/ezphot/config/common/transmission/Original_sources (vendor filter curves, IMX455 QE,
optics efficiency, SkyCalc atmosphere):

  T_total(lambda) = Filter x QE x Optics x Sky

Figure 1: the 30 narrow (~25 nm) medium-band filters.
Figure 2: the 7 wide (w-series) filters.

Style follows ~/ezphot/config/common/transmission/7DT_transmission_new_filters.png:
the final total curves are bold, every individual component (raw filter, QE, optics,
sky, QE x optics x sky envelope) is drawn faint in the background.

  MPLCONFIGDIR=/tmp/ezphot-matplotlib /home/hhchoi1022/miniconda3/envs/too/bin/python \\
      ezphot/analysis/260917_transmission_final_plot.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402
import pandas as pd                 # noqa: E402
from astropy.io import fits         # noqa: E402

SRC = Path("~/ezphot/config/common/transmission/Original_sources").expanduser()
OUT = Path("~/ezphot/config/common/transmission").expanduser()

NARROW = ["m386", "m400", "m425", "m438", "m450", "m475", "m483", "m500", "m512", "m525",
          "m534", "m550", "m561", "m575", "m586", "m600", "m615", "m625", "m640", "m650",
          "m661", "m675", "m700", "m725", "m750", "m775", "m800", "m825", "m850", "m875"]
WIDE = ["m375w", "m425w", "m466w", "m692w", "m710w", "m769w", "m832w"]


def pivot_wavelength(lam, tr):
    return np.sqrt(np.trapezoid(tr * lam, lam) / np.trapezoid(tr / lam, lam))


def main():
    # ---- common wavelength grid: SkyCalc sampling, 300-1010 nm ----------
    with fits.open(SRC / "skytable.fits") as hdul:
        lam_sky = np.asarray(hdul[1].data["lam"], dtype="f8")
        t_sky = np.asarray(hdul[1].data["trans"], dtype="f8")
    sel = (lam_sky >= 300) & (lam_sky <= 1010)
    lam_sky, t_sky = lam_sky[sel], t_sky[sel]
    # bin the atmosphere to 0.5 nm to tame the dense telluric lines
    edges = np.arange(300.0, 1010.5, 0.5)
    idx = np.digitize(lam_sky, edges) - 1
    grid = 0.5 * (edges[:-1] + edges[1:])
    t_sky = np.array([t_sky[idx == i].mean() if np.any(idx == i) else np.nan
                      for i in range(len(grid))])
    ok = np.isfinite(t_sky)
    grid, t_sky = grid[ok], t_sky[ok]

    qe = pd.read_csv(SRC / "QE.csv")
    t_qe = np.interp(grid, qe.wave, qe.QE)

    opt = pd.read_csv(SRC / "optics.efficiency.dr350.dr500.csv", sep=r"\s+", comment="#", skiprows=11)
    opt.columns = ["nm", "K9LA", "LaK4L", "FK95N", "VIS", "Al", "Total_pct", "total"]
    t_opt = np.interp(grid, opt.nm, opt.total)
    slope = (opt.total.iloc[1] - opt.total.iloc[0]) / (opt.nm.iloc[1] - opt.nm.iloc[0])
    below = grid < opt.nm.iloc[0]
    t_opt[below] = np.maximum(opt.total.iloc[0] + slope * (grid[below] - opt.nm.iloc[0]), 0.0)

    envelope = t_qe * t_opt * t_sky

    def make_figure(filters, png_name, label_mode="two-row"):
        # label_mode: "two-row" (all names on two fixed rows), "one-row", "none", "alternate" (every other filter)
        # sort by pivot wavelength and assign rainbow colors
        curves = {}
        for name in filters:
            d = pd.read_csv(SRC / f"{name}.csv")
            lam, tr = d.iloc[:, 0].to_numpy(float), d.iloc[:, 1].to_numpy(float)
            order = np.argsort(lam)                      # m466w is stored descending
            curves[name] = np.interp(grid, lam[order], tr[order], left=0, right=0)
        piv = {n: pivot_wavelength(grid, t) for n, t in curves.items()}
        ordered = sorted(filters, key=lambda n: piv[n])
        cmap = plt.get_cmap("turbo")
        colors = {n: cmap(0.05 + 0.9 * i / max(len(ordered) - 1, 1)) for i, n in enumerate(ordered)}

        # sized for a single column of a two-column manuscript (~3.4 in wide in print)
        fig, ax = plt.subplots(figsize=(7.0, 3.2))
        # ---- faint components ----
        ax.plot(grid, t_sky, color="gray", lw=1.2, alpha=0.45)
        ax.plot(grid, t_qe, color="green", lw=1.8, alpha=0.35)
        ax.plot(grid, t_opt, color="royalblue", lw=1.8, alpha=0.35)
        for name in ordered:
            ax.plot(grid, curves[name], color=colors[name], lw=1.0, alpha=0.18)
        # ---- final total curves: solid edge + translucent fill ----
        for k, name in enumerate(ordered):
            total = curves[name] * envelope
            ax.fill_between(grid, total, color=colors[name], alpha=0.18, lw=0)
            ax.plot(grid, total, color=colors[name], lw=1.0, alpha=1.0)
            if label_mode == "none":
                continue
            if label_mode == "alternate" and k % 2 == 1:
                continue
            if label_mode == "two-row":
                y = 0.70 if k % 2 == 0 else 0.84
            elif label_mode == "one-row":
                y = 0.70
            else:
                y = total[np.argmax(total)] + 0.02
            ax.text(piv[name], y, name, color=colors[name], rotation=90,
                    ha="center", va="bottom", fontsize=7, fontweight="bold")
        ax.set(xlim=(300, 1000), ylim=(0, 1.05))
        ax.set_xticks(np.arange(400, 1001, 100))
        ax.set_xlabel("Wavelength (nm)", fontsize=21)
        ax.set_ylabel("Transmission", fontsize=21)
        ax.tick_params(labelsize=18)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / png_name, dpi=300)
        fig.savefig((OUT / png_name).with_suffix(".pdf"))
        plt.close(fig)
        print(f"saved {OUT / png_name} (+ .pdf)")

    make_figure(NARROW, "7DT_transmission_medium30.png", label_mode="two-row")
    make_figure(WIDE, "7DT_transmission_wide7.png", label_mode="one-row")


if __name__ == "__main__":
    main()
