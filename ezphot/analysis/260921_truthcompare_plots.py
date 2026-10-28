#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Report figures from truthcompare_summary.csv (pipeline: 260921_truthcompare.py).

T1a: CALSPEC images / T1b: SPSS images — rows = curve version, columns = beam variant
(nobeam, beam); three truths per panel. hyp/ky cover only new-generation filters and
never appear in the CALSPEC dataset.
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shutil
from pathlib import Path

OUT = Path("/home/hhchoi1022/ezphot/log/20260918_spss")
FIGD = Path("/home/hhchoi1022/ezphot/log/20260921_curve_versions")
TRU = [("gnd", "#1f4e8c"), ("cor", "#C25B4E"), ("raw", "#E09A3E")]
VARS = ["nobeam", "beam"]

T = pd.read_csv(OUT / "truthcompare_summary.csv")


def grid_figure(ds, gnd_label, fname, code, subtitle, ylim):
    vers = [v for v in ["manufacturer", "version_mhj", "version_hyp", "version_ky"]
            if ((T.dataset == ds) & (T.version == v)).any()]
    fig, axes = plt.subplots(len(vers), 2, figsize=(16.5, 4.4 * len(vers)),
                             sharex=True, sharey=True)
    axes = axes.reshape(len(vers), 2)
    for i, ver in enumerate(vers):
        for j, var in enumerate(VARS):
            ax = axes[i, j]
            S = T[(T.dataset == ds) & (T.version == ver) & (T.variant == var)]
            for tag, col in TRU:
                lab = {"gnd": gnd_label, "cor": "corrected XP", "raw": "raw XP"}[tag]
                g = S.dropna(subset=[f"{tag}_mmag"]).sort_values("pivot")
                ax.errorbar(g["pivot"], g[f"{tag}_mmag"], yerr=g[f"{tag}_err"], ls="-",
                            marker="o", ms=4, lw=1.4, color=col, label=lab)
            ax.axhline(0, color="k", lw=0.7); ax.grid(alpha=0.3)
            ax.set_title(f"{ver} · {var}", fontsize=11.5)
            if j == 0:
                ax.set_ylabel("m_cal - syn_truth [mmag]")
            if i == 0 and j == 0:
                ax.legend(fontsize=9)
    axes[0, 0].set_ylim(*ylim)
    for j in range(2):
        axes[-1, j].set_xlabel("pivot wavelength [nm]")
    fig.suptitle(f"{code}  {ds} standards vs three truths — curve version x f/3 beam variant\n"
                 f"(ZP from corrected-XP field stars; {subtitle})", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=140)
    shutil.copy2(OUT / fname, FIGD / fname)
    print(f"saved {code}:", vers, "x", VARS)
    return vers


grid_figure("CALSPEC", "CALSPEC spectrum", "T1a_truthcompare_calspec.png", "T1a",
            "hyp/ky absent: new-generation filters only", (-40, 60))
grid_figure("SPSS", "SPSS flux table", "T1b_truthcompare_spss.png", "T1b",
            "m375w off scale in manufacturer rows (+118 nobeam)", (-50, 50))
