#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260929_ky_curve_versions.py — field and standard-star colour terms of the three filters with a ky lab curve (m438, m483, m512) for the six curves
(manufacturer / version_ky / version_hyp x nobeam / beam), against the effective centroid of each total curve
(filter x QE x optics x atmosphere, vacuum).  Figure O9 (one panel per filter).

Field colour term: median a_field of the SPSS frames used for the standard colour term.  Standard colour term: variant A
(local ZP, std SNR >= 100, ground truth; 260929_std_colorterm_variants), star bootstrap error.  Both are fitted linearly
against the effective centroid; the zero crossings say where each indicator wants the band to sit.
Output (~/ezphot/log/20260927_std_newbkg/): ky_curve_versions.csv, O9_ky_curve_versions.png
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
FILTERS = ["m438", "m483", "m512"]


def main():
    tot = CT.V2.build_totals()
    S = pd.read_csv(D / "std_colours.csv")
    S = S[S.dataset == "SPSS"]
    key = dict(zip(S.key, S.bprp))
    F = SS.screened(pd.read_csv(D / "std_offset_frames.csv"))
    F = F[(F.dataset == "SPSS") & F["filter"].isin(FILTERS)].copy()
    F["bprp"] = F.object.map(AN.norm).map(key)
    F = F[F.bprp.notna()]
    F["A"] = np.where(F.std_snr >= 100, F.dl_gnd, np.nan)
    rows = []
    for (filt, ver, var), g in F.groupby(["filter", "version", "variant"]):
        if (ver, var) not in tot or filt not in tot[(ver, var)]:
            continue
        lam, tr = tot[(ver, var)][filt]
        cen = float(np.trapz(lam * tr, lam) / np.trapz(tr, lam))
        p = g.dropna(subset=["A"]).groupby("object").agg(d=("A", lambda v: 1e3 * np.median(v)), bprp=("bprp", "first"))
        p = p[p.d.abs() <= 1e3 * SS.D_MAX]
        s, e = CT.fit(p)
        rows.append(dict(filter=filt, units="/".join(sorted(g.unit.unique())), curve=f"{LAB[ver]}·{var}", centroid=round(cen, 2),
                         a_field=round(1e3 * g.loc[g.A.notna(), "a_field"].median(), 1),
                         ct=round(s, 1), ct_err=round(e, 1), n=len(p), delta_level=round(p.d.median(), 1)))
    R = pd.DataFrame(rows).sort_values(["filter", "centroid"])
    fits = {}
    for filt, r in R.groupby("filter"):
        cf = np.polyfit(r.centroid, r.a_field, 1)
        cs = np.polyfit(r.centroid, r.ct, 1, w=1 / r.ct_err)
        fits[filt] = (cf, cs, -cf[1] / cf[0], -cs[1] / cs[0])
        R.loc[r.index, "field_zero_nm"] = round(fits[filt][2], 2)
        R.loc[r.index, "std_zero_nm"] = round(fits[filt][3], 2)
    R.to_csv(D / "ky_curve_versions.csv", index=False)
    pd.set_option("display.width", 220)
    print(R.to_string(index=False))
    for filt, (cf, cs, zf, zs) in fits.items():
        print(f"{filt}: field slope {cf[0]:+.1f}/nm zero {zf:.2f} nm; standards slope {cs[0]:+.1f}/nm zero {zs:.2f} nm; ratio {cs[0] / cf[0]:.2f}")

    fig, axs = plt.subplots(1, len(fits), figsize=(6 * len(fits), 5.2))
    for ax, (filt, r) in zip(np.atleast_1d(axs), R.groupby("filter")):
        cf, cs, zf, zs = fits[filt]
        xx = np.linspace(r.centroid.min() - 0.2, r.centroid.max() + 0.2, 50)
        ax.plot(xx, np.polyval(cf, xx), "--", color="tab:purple", lw=1)
        ax.plot(xx, np.polyval(cs, xx), "--", color="k", lw=1)
        ax.scatter(r.centroid, r.a_field, s=60, color="tab:purple", zorder=3, label="field colour term (corrected XP)")
        ax.errorbar(r.centroid, r.ct, r.ct_err, fmt="o", color="k", capsize=3, zorder=3, label="standard colour term (ground truth, local ZP, SNR ≥ 100)")
        for _, q in r.iterrows():
            ax.annotate(q["curve"], (q.centroid, max(q.a_field, q.ct) + 2), ha="center", fontsize=7)
        ax.axhline(0, color="0.3", lw=0.8)
        ax.axvline(zf, color="tab:purple", lw=0.8, alpha=0.6)
        ax.axvline(zs, color="k", lw=0.8, alpha=0.6)
        ax.set_title(f"{filt} ({r.units.iloc[0]}): field = 0 at {zf:.2f} nm, standards = 0 at {zs:.2f} nm", fontsize=9)
        ax.set_xlabel("effective centroid of the total curve [nm, vacuum]")
        ax.grid(alpha=0.3)
    np.atleast_1d(axs)[0].set_ylabel("colour term [mmag/mag]")
    np.atleast_1d(axs)[0].legend(fontsize=7, loc="upper right")
    fig.suptitle("O9  Filters with a ky lab curve: field and standard colour terms for six curves (manufacturer/ky/hyp × nobeam/beam; SPSS frames)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(D / "O9_ky_curve_versions.png", dpi=130)


if __name__ == "__main__":
    main()
