#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C1: per-filter field colour term (corrected XP) for each curve combo, from colorterm_versions_frames.csv.
Panel A: colour-term medians vs wavelength per combo.  Panel B: change relative to the previous analysis
curve (phys_orig).  Errors = MAD/sqrt(n) of the per-frame values.

  /home/hhchoi1022/miniconda3/envs/too/bin/python ezphot/analysis/260924_colorterm_versions_plot.py [--tag _smoke]
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("/home/hhchoi1022/ezphot/log/20260924_colorterm_versions")
STY = {"phys_orig": ("0.25", "o", "-", "phys_orig (2024-26 analysis: manufacturer·beam, original build)"),
       "manufacturer_beam": ("#7A5EA8", "s", "--", "manufacturer · beam"),
       "manufacturer_nobeam": ("#2F5D9E", "o", "-", "manufacturer · nobeam"),
       "version_mhj_beam": ("#C25B4E", "^", "--", "version_mhj · beam"),
       "version_mhj_nobeam": ("#2E8B6E", "D", "-", "version_mhj · nobeam")}

ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="")
a = ap.parse_args()
fr = pd.read_csv(OUT / f"colorterm_versions_frames{a.tag}.csv")
g = fr.groupby(["filter", "combo"]).a0
S = pd.DataFrame({"a_med": g.median(),
                  "a_err": g.apply(lambda v: 1.4826 * np.median(np.abs(v - v.median())) / np.sqrt(len(v))),
                  "n": g.size()}).reset_index()
S["pivot_nm"] = S["filter"].str[1:4].astype(int)
S = S.sort_values(["pivot_nm", "combo"])
S.to_csv(OUT / f"colorterm_versions_table{a.tag}.csv", index=False)
ref = S[S.combo == "phys_orig"].set_index("filter").a_med

fig, ax = plt.subplots(2, 1, figsize=(13.5, 9.5), sharex=True,
                       gridspec_kw=dict(height_ratios=[1.35, 1.0], hspace=0.12))
for combo, (col, mk, ls, lab) in STY.items():
    s = S[S.combo == combo]
    if not len(s):
        continue
    ax[0].errorbar(s.pivot_nm, s.a_med, yerr=s.a_err, color=col, marker=mk, ls=ls, ms=5, lw=1.4, label=lab)
    if combo != "phys_orig":
        d = s.set_index("filter").a_med - ref.reindex(s["filter"]).values
        ax[1].plot(s.pivot_nm, d.values, color=col, marker=mk, ls=ls, ms=5, lw=1.4, label=lab)
for axx in ax:
    axx.axhline(0, color="k", lw=0.7); axx.grid(alpha=0.3)
ax[0].set_ylabel("field colour term a  [mag / mag(BP-RP)]")
ax[0].set_title("C1  Survey field colour term vs corrected XP (2024-06 .. 2026-04, %d frames) — per curve combo"
                % fr.file.nunique(), fontsize=11.5)
ax[0].legend(fontsize=8.5)
ax[1].set_ylabel("a(combo) − a(phys_orig)")
ax[1].set_xlabel("pivot wavelength [nm]")
ax[1].set_title("B. change of the colour term relative to the previous analysis curve", fontsize=10.5)
ax[1].legend(fontsize=8.5)
fig.savefig(OUT / f"C1_colorterm_versions{a.tag}.png", dpi=140, bbox_inches="tight")
print("saved C1; frames:", fr.file.nunique())
print(S.pivot_table(index="pivot_nm", columns="combo", values="a_med").round(4).to_string())
