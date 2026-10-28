#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""260925_bkg_report_figs.py — A14 injection–recovery and A15 background-route comparison figures for the report
(from bkg_injection.csv and bkg_direct_mean.csv)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

LOG = Path("/home/hhchoi1022/ezphot/log/20260924_stepA")
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "lines.linewidth": 1.1})
COL = {"sepmode": "tab:orange", "ezmean": "0.5", "pumean5": "tab:blue"}
LAB = {"sepmode": "SExtractor mode (current)", "ezmean": "mode + global offset (ezphot)",
       "pumean5": "per-mesh mean 5σ (new)"}

# ---- A14 injection–recovery --------------------------------------------------------------------------------------
I = pd.read_csv(LOG / "bkg_injection.csv")
I = I[I["error"].isna()] if "error" in I.columns else I
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
for r, g in I.groupby("route"):
    y = [g[f"inj{s}_mmag"] for s in (22, 50, 120)]
    ax[0].errorbar([22, 50, 120], [v.median() for v in y], [1.2533 * v.std() / np.sqrt(len(v)) for v in y],
                   color=COL[r], marker="o", capsize=3, label=f"{LAB[r]} ({g.name.nunique()} frames)")
    ax[1].scatter(g.blank10, g.inj22_adupix, s=16, color=COL[r], alpha=0.8, label=LAB[r])
ax[0].axhline(0, color="k", lw=0.7); ax[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax[0].set(xscale="log", xticks=[22, 50, 120], xticklabels=["22", "50", "120"], xlabel="injected star SNR (10″ aperture)",
          ylabel="recovered − injected [mmag]  (>0: too bright)",
          title="(a) artificial stars: recovery bias (median ± SEM over frames)")
ax[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
lim = [-0.4, 1.7]
ax[1].plot(lim, lim, "k-", lw=0.7)
ax[1].set(xlim=lim, ylim=lim, xlabel="blank-sky residual, 10″ apertures [ADU/pix]",
          ylabel="injected SNR-22 flux error / aperture area [ADU/pix]",
          title="(b) per frame: injected-star error = blank-sky residual")
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
fig.suptitle("A14  Injection–recovery on the pilot frames (empirical PSF, photon noise, measured at the true positions)",
             fontsize=10)
fig.tight_layout(); fig.savefig(LOG / "A14_injection.png", dpi=130)

# ---- A15 background routes: level and field uniformity ----------------------------------------------------------
D = pd.read_csv(LOG / "bkg_direct_mean.csv")
D = D[D["error"].isna()] if "error" in D.columns else D
order = ["sepmode", "ez_goff", "pu_mean3", "pu_mean5"]
names = {"sepmode": "SExtractor mode\n(current)", "ez_goff": "mode + global\noffset (ezphot)",
         "pu_mean3": "per-mesh mean\n3σ clip", "pu_mean5": "per-mesh mean\n5σ clip (new)"}
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
rng = np.random.default_rng(0)
for k, r in enumerate(order):
    g = D[D.route == r]
    for a_, col in [(ax[0], "blank10"), (ax[1], "cell_sd")]:
        a_.scatter(k + rng.uniform(-0.18, 0.18, len(g)), g[col], s=9, alpha=0.5, color="tab:blue" if r == "pu_mean5" else "0.45")
        a_.plot([k - 0.28, k + 0.28], [g[col].median()] * 2, color="k", lw=2)
        a_.text(k + 0.3, g[col].median(), f"{g[col].median():.3f}", va="center", fontsize=8)
for a_, t, yl in [(ax[0], "(a) level: blank 10″ apertures (want 0)", "mean residual [ADU/pix]"),
                  (ax[1], "(b) field uniformity: sd of blank residual over 3×3 cells", "sd over cells [ADU/pix]")]:
    a_.axhline(0, color="k", lw=0.6); a_.set_xticks(range(len(order))); a_.set_xticklabels([names[o] for o in order], fontsize=8)
    a_.set(title=t, ylabel=yl); a_.grid(alpha=0.3, axis="y")
ax[0].set_ylim(-0.3, 1.8); ax[1].set_ylim(0, 0.45)
fig.suptitle("A15  Background routes on 65 pilot frames (same ezphot source mask; 10000 blank apertures per frame)",
             fontsize=10)
fig.tight_layout(); fig.savefig(LOG / "A15_bkg_routes.png", dpi=130)
print("saved A14, A15")

# ---- A12 re-plot (labels: radii; linear profile axis) from star_profiles.csv ---------------------------------------
P = pd.read_csv(LOG / "star_profiles.csv")
P = P[P["error"].isna()] if "error" in P.columns else P
fr = pd.read_csv(LOG / "pilot" / "pilot_frames.csv"); fr["name"] = [Path(p).name for p in fr.path]
if "exptime" not in P.columns:
    P = P.merge(fr[["name", "exptime"]], on="name", how="left")
P["grp"] = P.dataset + " " + P.exptime.astype(int).astype(str) + "s"
EDGES = np.array([0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8.5, 10, 12.5, 15, 20, 25, 30])
rmid = 0.5 * (EDGES[:-1] + EDGES[1:])
nk = len(EDGES) - 1
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
for grp, g in P.groupby("grp"):
    res = g[[f"res_{k}" for k in range(nk)]].median().values
    ax[0].plot(rmid, res, "o-", ms=3, label=f"{grp} ({len(g)})")
    fa = g[[f"fa_{k}" for k in range(nk)]].median().values; brr = g[[f"br_{k}" for k in range(nk)]].median().values
    l, = ax[1].plot(rmid, fa / fa[0], "-", label=f"{grp} faint")
    ax[1].plot(rmid, brr / brr[0], "--", color=l.get_color())
ax[0].axhline(0, color="k", lw=0.6); ax[0].axvspan(2.5, 5.0, color="0.88", zorder=0)
ax[0].set(xscale="log", ylim=(-1.5, 1.0), xlabel="radius [arcsec]", ylabel="faint − s·bright [ADU/pix]",
          title="(a) residual profile of selected faint stars\n(grey: r = 2.5–5″, between the 5″ and 10″ diameter apertures)")
ax[0].legend(fontsize=7.5, title="group (frames)", title_fontsize=7.5); ax[0].grid(alpha=0.3)
ax[1].set(xscale="log", ylim=(-0.02, 1.02), xlabel="radius [arcsec]", ylabel="I(r) / I(centre)",
          title="(b) normalised stacked profiles\n(solid: faint SNR 20–40, dashed: bright SNR 100–150)")
ax[1].legend(fontsize=7.5); ax[1].grid(alpha=0.3)
fig.suptitle("A12  Real faint vs bright star profiles after a mean background (select_stars_like set, blank sky ≈ 0)",
             fontsize=10)
fig.tight_layout(); fig.savefig(LOG / "A12_star_profiles.png", dpi=130)
print("saved A12")
