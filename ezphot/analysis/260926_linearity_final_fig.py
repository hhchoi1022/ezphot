#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""260926_linearity_final_fig.py — L4: real g-band linearity series vs perfectly linear synthetic series (method bias)
and background residuals.  Reads the outputs of 260926_linearity_test/pairs/summary."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

D = Path("/home/hhchoi1022/ezphot/log/20260926_linearity")
R = pd.read_csv(D / "g_20260925" / "lin_pairs.csv")
SL = pd.read_csv(D / "synth_g5" / "lin_pairs.csv")
SB = pd.read_csv(D / "synth_bright" / "lin_pairs.csv")
F = pd.read_csv(D / "g_20260925" / "lin_frames.csv")
Dt = pd.read_csv(D / "g_20260925" / "lin_summary_shutter.csv")
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9})
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
cols = {"3/10": "tab:blue", "3/20": "tab:green", "5/20": "tab:purple", "1/10": "tab:orange"}
for pair, c in cols.items():
    r = R[(R.pair == pair) & (R.peak_s_e > 4) & (R.peak_s_e < 400)]
    ax[0].errorbar(r.peak_s_e, r.dmag, r.err, marker="o", ms=3.5, capsize=0, color=c, label=f"real {pair} s")
    s = SL[(SL.pair == pair) & (SL.peak_s_e > 4) & (SL.peak_s_e < 400)]
    if len(s) and s.dmag.notna().any():
        ax[0].errorbar(s.peak_s_e * 1.04, s.dmag, s.err, marker="o", ms=3.5, mfc="white", ls="--", lw=0.9, capsize=0, color=c,
                       label=f"linear synthetic {pair} s")
ax[0].plot([5.5, 11, 23, 45, 140], [51, 37, 23, 10, 0], "k:", lw=1.3, label="2024 medium band (XP-based)")
ax[0].axhline(0, color="k", lw=0.6); ax[0].set_xscale("log"); ax[0].set_ylim(-20, 60); ax[0].grid(alpha=0.3)
ax[0].set(xlabel="expected star peak above sky in the short exposure [e⁻]", ylabel="short-exposure flux deficit [mmag]",
          title="(a) low signal (filled: real, open: perfectly linear synthetic)")
ax[0].legend(fontsize=6.5, ncol=2)
for pair, c in {"5/40": "tab:red", "10/100": "tab:brown", "10/40": "tab:pink"}.items():
    r = R[(R.pair == pair) & (R.tot_l > 3000) & (R.tot_l < 61000)]
    ax[1].errorbar(r.tot_l, r.dmag, r.err, marker="s", ms=3.5, capsize=0, color=c, label=f"real {pair} s")
    s = SB[(SB.pair == pair) & (SB.tot_l > 3000)]
    if len(s):
        ax[1].errorbar(s.tot_l * 1.03, s.dmag, s.err, marker="s", ms=3.5, mfc="white", ls="--", lw=0.9, capsize=0, color=c,
                       label=f"linear synthetic {pair} s")
ax[1].axhline(0, color="k", lw=0.6); ax[1].set_xscale("log"); ax[1].grid(alpha=0.3); ax[1].set_ylim(-10, 15)
ax[1].axvspan(1000, 6000, color="0.92", zorder=0)
ax[1].set(xlabel="total level at the star peak in the long exposure [ADU]", ylabel="dR [mmag] (>0: long exposure brighter)",
          title="(b) high level vs linear short exposure (grey = reference)")
ax[1].legend(fontsize=6.5)
F["sky_e"] = F.sky_med * F.egain
B = F.groupby("exptime")[["sky_e"]].median().join((F.assign(o=F.blank_old * F.egain, n=F.blank_new * F.egain)
                                                    .groupby("exptime")[["o", "n"]].median()))
ax[2].plot(B.sky_e, B.o, "o-", color="tab:orange", label="SExtractor mode (current)")
ax[2].plot(B.sky_e, B.n, "o-", color="tab:blue", label="per-mesh mean 5σ (new)")
ax[2].axhline(0.42, color="0.5", ls=":", label="Poisson mode bias ≈ 0.42 e⁻")
ax[2].axhline(0, color="k", lw=0.6); ax[2].set_xscale("log"); ax[2].grid(alpha=0.3); ax[2].legend(fontsize=7)
ax[2].set(xlabel="sky [e⁻/pix] (1–200 s)", ylabel="blank 10″ residual [e⁻/pix]", title="(c) background residual, 15 units")
# shutter offset omitted: its estimate depends on the reference definition (0.01-0.16 s), i.e. not robust
fig.suptitle("L4  g-band linearity series, WD2039-682, 15 units × 9 exposures × 5 (654 of 675 frames photometered; 7DT11 partial), 2026-09-26, full moon", fontsize=10)
fig.tight_layout(); fig.savefig(D / "g_20260925" / "L4_final.png", dpi=125)
print("saved")
