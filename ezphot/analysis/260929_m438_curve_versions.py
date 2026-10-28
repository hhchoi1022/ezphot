#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260929_m438_curve_versions.py — m438 (7DT04) field and standard-star colour terms for the six curves
(manufacturer / version_ky / version_hyp x nobeam / beam), against the effective centroid of each total curve
(filter x QE x optics x atmosphere, vacuum).  Figure O9.

Field colour term: median a_field of the SPSS frames used for the standard colour term.  Standard colour term: variant A
(local ZP, std SNR >= 100, ground truth; 260929_std_colorterm_variants), star bootstrap error.  Both are fitted linearly
against the effective centroid; the zero crossings say where each indicator wants the band to sit.
Output (~/ezphot/log/20260927_std_newbkg/): m438_curve_versions.csv, O9_m438_curve_versions.png
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
LAB = {"manufacturer": "manufacturer", "version_ky": "ky", "version_hyp": "hyp"}


def main():
    tot = CT.V2.build_totals()
    S = pd.read_csv(D / "std_colours.csv")
    S = S[S.dataset == "SPSS"]
    key = dict(zip(S.key, S.bprp))
    F = SS.screened(pd.read_csv(D / "std_offset_frames.csv"))
    F = F[(F.dataset == "SPSS") & (F["filter"] == "m438")].copy()
    F["bprp"] = F.object.map(AN.norm).map(key)
    F = F[F.bprp.notna()]
    F["A"] = np.where(F.std_snr >= 100, F.dl_gnd, np.nan)
    rows = []
    for (ver, var), g in F.groupby(["version", "variant"]):
        lam, tr = tot[(ver, var)]["m438"]
        cen = float(np.trapz(lam * tr, lam) / np.trapz(tr, lam))
        p = g.dropna(subset=["A"]).groupby("object").agg(d=("A", lambda v: 1e3 * np.median(v)), bprp=("bprp", "first"))
        p = p[p.d.abs() <= 1e3 * SS.D_MAX]
        s, e = CT.fit(p)
        rows.append(dict(curve=f"{LAB[ver]}·{var}", centroid=round(cen, 2), a_field=round(1e3 * g.loc[g.A.notna(), "a_field"].median(), 1),
                         ct=round(s, 1), ct_err=round(e, 1), n=len(p), delta_level=round(p.d.median(), 1)))
    R = pd.DataFrame(rows).sort_values("centroid")
    R.to_csv(D / "m438_curve_versions.csv", index=False)
    cf = np.polyfit(R.centroid, R.a_field, 1)
    cs = np.polyfit(R.centroid, R.ct, 1, w=1 / R.ct_err)
    z_f, z_s = -cf[1] / cf[0], -cs[1] / cs[0]
    print(R.to_string(index=False))
    print(f"field: slope {cf[0]:+.1f} per nm, zero at {z_f:.2f} nm;  standards: slope {cs[0]:+.1f} per nm, zero at {z_s:.2f} nm;"
          f"  slope ratio {cs[0] / cf[0]:.2f}")

    fig, ax = plt.subplots(figsize=(9, 5.2))
    xx = np.linspace(R.centroid.min() - 0.2, R.centroid.max() + 0.2, 50)
    ax.plot(xx, np.polyval(cf, xx), "--", color="tab:purple", lw=1)
    ax.plot(xx, np.polyval(cs, xx), "--", color="k", lw=1)
    ax.scatter(R.centroid, R.a_field, s=60, color="tab:purple", zorder=3, label="field colour term (corrected XP)")
    ax.errorbar(R.centroid, R.ct, R.ct_err, fmt="o", color="k", capsize=3, zorder=3, label="standard colour term (ground truth, local ZP, SNR ≥ 100)")
    for _, r in R.iterrows():
        ax.annotate(r["curve"], (r.centroid, max(r.a_field, r.ct) + 3), ha="center", fontsize=8)
    ax.axhline(0, color="0.3", lw=0.8)
    ax.axvline(z_f, color="tab:purple", lw=0.8, alpha=0.6)
    ax.axvline(z_s, color="k", lw=0.8, alpha=0.6)
    ax.text(z_f, ax.get_ylim()[0] + 2, f" field = 0\n {z_f:.2f} nm", color="tab:purple", fontsize=8)
    ax.text(z_s, ax.get_ylim()[0] + 2, f"standards = 0 \n{z_s:.2f} nm ", color="k", fontsize=8, ha="right")
    ax.set_xlabel("effective centroid of the total curve [nm, vacuum]  (beam shifts it by −1.73 nm)")
    ax.set_ylabel("colour term [mmag/mag]")
    ax.set_title("O9  m438 (7DT04): field and standard colour terms for six curves (SPSS frames)", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(D / "O9_m438_curve_versions.png", dpi=130)


if __name__ == "__main__":
    main()
