#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260929_std_fgk_spectrum_fig.py — per-star SPSS offsets vs wavelength after the 2026-09-27 red stars: white dwarfs vs
FGK/M standards, ground truth vs corrected-XP truth (figure O7).

Selection = variant B of 260929_std_colorterm_variants.py (local ZP, or image ZP when < 5 local ZP stars; std SNR >= 50
with expected_bias subtracted), manufacturer·beam, screened frames.  Per star: median over frames.
Panels: (a) Delta_gnd, (b) Delta_cor, (c) Delta_cor - Delta_gnd = syn_gnd - syn_cor (reference-spectrum difference, no
photometry).  White dwarfs/sdOB are drawn as a median line with the 16-84 % band; each FGK/M star as its own line.
Output: ~/ezphot/log/20260927_std_newbkg/O7_std_fgk_vs_wd.png, std_fgk_vs_wd.csv
"""
import importlib
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
SS = importlib.import_module("260927_std_offset_summary")
CT = importlib.import_module("260927_std_colorterm")
D, AN = CT.D, CT.AN
NEW = {"LP885-23", "LTT1020", "LTT2415", "LTT7379", "HD271783"}


def main():
    S = pd.read_csv(D / "std_colours.csv")
    S = S[S.dataset == "SPSS"]
    info = {k: (b, c, t) for k, b, c, t in zip(S.key, S.bprp, S.cls, S.sptype)}
    F = SS.screened(pd.read_csv(D / "std_offset_frames.csv"))
    F = F[(F.dataset == "SPSS") & (F.version == "manufacturer") & (F.variant == "beam")].copy()
    F["k"] = F.object.map(AN.norm)
    F = F[F.k.isin(info)]
    F["cls"] = F.k.map(lambda k: info[k][1])
    bias = SS.expected_bias(F.std_peak_e.to_numpy(), "SPSS") / 1e3
    for t in ("gnd", "cor"):
        zp = np.where(F.n_loc >= 5, F[f"dl_{t}"], F[f"d_{t}"])
        F[t] = np.where(F.std_snr >= 50, zp - bias, np.nan) * 1e3
    F["dif"] = F["cor"] - F["gnd"]
    F["pivot"] = F["filter"].str[1:4].astype(int)
    P = F.groupby(["object", "k", "cls", "pivot"])[["gnd", "cor", "dif"]].median().reset_index()
    P = P[(P.gnd.abs() <= 500) | P.gnd.isna()]
    P.to_csv(D / "std_fgk_vs_wd.csv", index=False)

    fig, axs = plt.subplots(3, 1, figsize=(12, 13), sharex=True)
    red = P[P.cls.isin(["FGK", "M"])]
    hot = P[P.cls.isin(["WD", "sdOB"])]
    cmap = plt.get_cmap("plasma")
    order = sorted(red.k.unique(), key=lambda k: info[k][0])
    for ax, col, lab in zip(axs, ("gnd", "cor", "dif"),
                            ("(a) Δ, ground truth (SPSS V2 spectrum)", "(b) Δ, corrected-XP truth",
                             "(c) Δ_cor − Δ_gnd = syn_gnd − syn_cor  (reference-spectrum difference, no photometry)")):
        h = hot.groupby("pivot")[col]
        med, lo, hi = h.median(), h.quantile(0.16), h.quantile(0.84)
        ax.fill_between(med.index, lo, hi, color="tab:blue", alpha=0.18, lw=0)
        ax.plot(med.index, med, "-", color="tab:blue", lw=2.2, label=f"white dwarfs / sdOB (median, 16–84 %; {hot.k.nunique()} stars)")
        for i, k in enumerate(order):
            g = red[red.k == k].sort_values("pivot")
            b, c, t = info[k]
            obj = g.object.iloc[0]
            ax.plot(g["pivot"], g[col], "o-" if obj in NEW else "s--", ms=3.5, lw=1.1,
                    color=cmap(0.05 + 0.85 * i / max(1, len(order) - 1)),
                    label=f"{obj} ({t}, BP−RP {b:.2f}){' new' if obj in NEW else ''}")
        ax.axhline(0, color="k", lw=0.6)
        for x0, x1 in ((686, 690), (716, 732), (759, 770), (810, 840)):
            ax.axvspan(x0, x1, color="0.85", lw=0, zorder=0)
        ax.set_ylim(-80, 80)
        ax.set_ylabel("mmag")
        ax.set_title(lab, fontsize=10, loc="left")
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=7, ncol=3, loc="lower left")
    axs[2].set_xlabel("filter pivot [nm]  (grey = telluric bands)")
    fig.suptitle("O7  SPSS standards per star: white dwarfs vs FGK/M (manufacturer·beam; local ZP or image ZP when < 5 local "
                 "stars; std SNR ≥ 50, faint-star bias removed)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(D / "O7_std_fgk_vs_wd.png", dpi=130)
    band = P[(P["pivot"] >= 550) & (P["pivot"] <= 700)]
    s = band.groupby(["cls", "object"])[["gnd", "cor", "dif"]].median().round(1)
    pd.set_option("display.width", 200)
    print(s.to_string())
    print(band.groupby(band.cls.isin(["FGK", "M"]))[["gnd", "cor", "dif"]].median().round(1))


if __name__ == "__main__":
    main()
