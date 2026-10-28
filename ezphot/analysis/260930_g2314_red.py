#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260930_g2314_red.py — why the SPSS standard G23-14 (G5, 2026) reads -30..-58 mmag in m800-m850 with its SPSS V2 spectrum
as truth but ~0 with its Gaia XP spectrum.

For G23-14 and the other SPSS FGK standards: ratio of the SPSS V2 flux table to the star's own corrected / raw Gaia XP
spectrum (both on the XP 2-nm grid, normalised over 700-760 nm), the SPSS error column, and synthetic
syn_gnd - syn_cor per medium-band filter (manufacturer beam).  Telluric bands marked (O2 B 687, H2O 720-730,
O2 A 760-770, H2O 810-840, H2O 890-990 nm).
Output (~/ezphot/log/20260930_g2314/): R1_g2314_red.png, ratio_bands.csv
"""
import importlib
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
AN = importlib.import_module("260919_spss_analysis")
CT = importlib.import_module("260927_std_colorterm")
R, V2 = CT.R, CT.V2
OUT = Path("/home/hhchoi1022/ezphot/log/20260930_g2314")
XPW = np.arange(336.0, 1020.01, 2.0)
STARS = ["G23-14", "HD271783", "LTT7379", "G24-25", "LTT1020", "LTT2415", "HD270477", "LTT1788", "LTT9239"]
TELL = [(686, 691), (716, 734), (759, 771), (810, 840), (890, 990)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    AN._init()
    sp = pd.read_csv(AN.SPSS_CSV).set_index("Name")
    tot = V2.build_totals()[("manufacturer", "beam")]
    filt = ["m725", "m750", "m769w", "m775", "m800", "m825", "m850", "m875"]
    rows, curves = [], {}
    for name in STARS:
        s = sp.loc[name]
        p = AN.FLUXD / f"V2.SPSS{int(s.SPSS):03d}.ascii"
        d = np.genfromtxt(p, comments="#", missing_values="NA", filling_values=np.nan)
        w, f, e = d[:, 0], d[:, 1], d[:, 2]
        ok = np.isfinite(f) & (f > 0)
        ref = AN.gather_reference(s.RA_ICRS, s.DE_ICRS, radius=0.05)
        if ref is None:
            continue
        j = np.where(ref["sid"] == str(int(s.GaiaEDR3)))[0]
        if not len(j):
            continue
        cor = ref["flux"][j[0]]
        # raw XP from the GAIA_BEST tile if available through the truthcompare helper
        TC = importlib.import_module("260921_truthcompare")
        xp = TC.OZ.xp_for(s.GaiaEDR3) if hasattr(TC.OZ, "xp_for") else None
        fs = np.interp(XPW, w[ok], f[ok])
        es = np.interp(XPW, w[ok], np.where(np.isfinite(e[ok]), e[ok], np.nan))
        norm = (XPW >= 700) & (XPW <= 760)
        ratio = fs / cor
        ratio /= np.nanmedian(ratio[norm])
        curves[name] = (ratio, es / fs, s.Flag, s.SpType)
        r = dict(star=name, sptype=s.SpType, flag=s.Flag)
        for fl in filt:
            fw, ft = tot[fl]
            r[f"{fl}_gnd_minus_cor"] = round(1e3 * (float(R.syn_mag(w[ok], f[ok], fw, ft)) - float(R.syn_mag(XPW, cor, fw, ft))), 1)
        for lo, hi in [(780, 808), (812, 838), (842, 870), (872, 890)]:
            m = (XPW >= lo) & (XPW <= hi)
            r[f"ratio_{lo}_{hi}"] = round(float(np.nanmedian(ratio[m])), 4)
        rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "ratio_bands.csv", index=False)
    pd.set_option("display.width", 240)
    print(T.to_string(index=False))

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
    cmap = plt.get_cmap("tab10")
    for k, (name, (ratio, rel_err, flag, spt)) in enumerate(curves.items()):
        lw = 2.2 if name == "G23-14" else 0.9
        axs[0].plot(XPW, ratio, "-", color="k" if name == "G23-14" else cmap(k % 10), lw=lw, label=f"{name} ({spt}, Flag {flag:.0f})")
        axs[1].plot(XPW, 100 * rel_err, "-", color="k" if name == "G23-14" else cmap(k % 10), lw=lw)
    for ax in axs:
        for lo, hi in TELL:
            ax.axvspan(lo, hi, color="0.9", zorder=0)
        ax.grid(alpha=0.3)
    for fl in ["m750", "m775", "m800", "m825", "m850", "m875"]:
        fw, ft = tot[fl]
        axs[0].fill_between(fw, 0.9, 0.9 + 0.04 * ft / ft.max(), color="tab:blue", alpha=0.3, lw=0)
        axs[0].text(fw[np.argmax(ft)], 0.945, fl, fontsize=7, ha="center", color="tab:blue")
    axs[0].axhline(1, color="k", lw=0.5)
    axs[0].set_xlim(640, 1000); axs[0].set_ylim(0.9, 1.06)
    axs[0].set_ylabel("SPSS V2 / own corrected Gaia XP\n(normalised at 700-760 nm)")
    axs[0].legend(fontsize=7, ncol=3, loc="lower left")
    axs[1].set_ylabel("SPSS error / flux [%]"); axs[1].set_ylim(0, 6)
    axs[1].set_xlabel("wavelength [nm] (grey: telluric bands; blue: 7DT filters)")
    fig.suptitle("R1  SPSS FGK standards: SPSS V2 spectrum relative to the star's own corrected Gaia XP in the red", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(OUT / "R1_g2314_red.png", dpi=120)


if __name__ == "__main__":
    main()
