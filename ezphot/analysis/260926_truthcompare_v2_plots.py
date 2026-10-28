#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Report figures T1a_v2 / T1b_v2 from truthcompare_v2_summary.csv (pipeline: 260926_truthcompare_v2.py).

Same layout as 260921_truthcompare_plots.py (rows = curve version, columns = beam variant, three truths per panel).
Points resting on fewer than 5 standards are drawn open; the previous T1 (old star selection, air curves) ground-truth
curve is drawn as a grey dashed line for reference.
"""
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("/home/hhchoi1022/ezphot/log/20260918_spss")
FIGD = Path("/home/hhchoi1022/ezphot/log/20260921_curve_versions")
TRU = [("gnd", "#1f4e8c"), ("cor", "#C25B4E"), ("raw", "#E09A3E")]
VARS = ["nobeam", "beam"]
NMIN = 5

T = pd.read_csv(OUT / "truthcompare_v2_summary.csv")
T0 = pd.read_csv(OUT / "truthcompare_summary.csv")


def grid_figure(ds, gnd_label, fname, code, subtitle, ylim):
    vers = [v for v in ["manufacturer", "version_mhj", "version_hyp", "version_ky"] if ((T.dataset == ds) & (T.version == v)).any()]
    fig, axes = plt.subplots(len(vers), 2, figsize=(16.5, 4.4 * len(vers)), sharex=True, sharey=True)
    axes = axes.reshape(len(vers), 2)
    for i, ver in enumerate(vers):
        for j, var in enumerate(VARS):
            ax = axes[i, j]
            S = T[(T.dataset == ds) & (T.version == ver) & (T.variant == var)]
            S0 = T0[(T0.dataset == ds) & (T0.version == ver) & (T0.variant == var)].dropna(subset=["gnd_mmag"]).sort_values("pivot")
            ax.plot(S0["pivot"], S0.gnd_mmag, "--", color="0.55", lw=1.1, label=f"{gnd_label} — previous T1")
            for tag, col in TRU:
                lab = {"gnd": gnd_label, "cor": "corrected XP", "raw": "raw XP"}[tag]
                g = S.dropna(subset=[f"{tag}_mmag"]).sort_values("pivot")
                ax.plot(g["pivot"], g[f"{tag}_mmag"], "-", color=col, lw=1.3, alpha=0.8)
                ok = g[f"{tag}_n"] >= NMIN
                ax.errorbar(g["pivot"][ok], g[f"{tag}_mmag"][ok], yerr=g[f"{tag}_err"][ok], ls="none", marker="o", ms=4.5,
                            color=col, label=lab)
                ax.plot(g["pivot"][~ok], g[f"{tag}_mmag"][~ok], "o", ms=5, mfc="none", color=col, alpha=0.8)
            ax.axhline(0, color="k", lw=0.7); ax.grid(alpha=0.3)
            ax.set_title(f"{ver} · {var}", fontsize=11.5)
            if j == 0:
                ax.set_ylabel("m_cal − syn_truth [mmag]")
            if i == 0 and j == 0:
                ax.legend(fontsize=9)
    axes[0, 0].set_ylim(*ylim)
    for j in range(2):
        axes[-1, j].set_xlabel("pivot wavelength [nm]")
    fig.suptitle(f"{code}  {ds} standards vs three truths — curve version × f/3 beam variant (new star selection, vacuum)\n"
                 f"(ZP from corrected-XP field stars SNR ≥ 100; standards SNR ≥ 100; open = fewer than {NMIN} standards; {subtitle})",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=140)
    shutil.copy2(OUT / fname, FIGD / fname)
    print(f"saved {code}:", vers, "x", VARS)


grid_figure("CALSPEC", "CALSPEC spectrum", "T1a_v2_truthcompare_calspec.png", "T1a",
            "hyp/ky absent: new-generation filters only", (-50, 60))
grid_figure("SPSS", "SPSS flux table", "T1b_v2_truthcompare_spss.png", "T1b",
            "m375w off scale in manufacturer rows", (-50, 50))
