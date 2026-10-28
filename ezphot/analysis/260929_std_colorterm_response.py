#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260929_std_colorterm_response.py — how much of a curve-induced field colour term the standard-star colour term should
show, i.e. the response of the standards' colour term to a bandpass change relative to the field colour term.

The field colour term a_field is a slope over field FGK stars (BP-RP ~0.5-1.5); the standards' colour term ct is a slope
dominated by white dwarfs (BP-RP -0.5..0.2) against FGK.  A bandpass error changes the two by different amounts when the
colour dependence of the synthetic magnitudes is not linear in BP-RP (Balmer lines in hot DA stars in the blue).
Measured empirically on the same frames for three real curve changes:
  manufacturer beam - manufacturer nobeam, mhj beam - manufacturer beam, hyp beam - manufacturer beam
Per filter (SPSS, ground truth, image ZP d_gnd — the ZP choice cancels in the difference between curves, and the image
ZP keeps every frame): response r = d(ct_std)/d(a_field), ct_std = -slope of per-star Delta vs BP-RP with d(a_field) =
median over frames.  r depends on the star set (white-dwarf/FGK mix), so it is computed twice: r_all over all SPSS
standards and r_A over the standards of the measured colour term (variant A: std SNR >= 100 with a local ZP, manufacturer
beam).  r_A is the one to compare with; r_all gives the range.  Then: if the whole field colour term were a bandpass error
of that kind, the standards should show ct_expected = r x a_field; compare with the measured ct (variant A of
260929_std_colorterm_variants, per unit where the field term differs between units).
Output (~/ezphot/log/20260927_std_newbkg/): std_colorterm_response.csv, std_colorterm_expected.csv
"""
import importlib
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
SS = importlib.import_module("260927_std_offset_summary")
CT = importlib.import_module("260927_std_colorterm")
AN, D = CT.AN, CT.D
PAIRS = ((("manufacturer", "beam"), ("manufacturer", "nobeam")),
         (("version_mhj", "beam"), ("manufacturer", "beam")),
         (("version_hyp", "beam"), ("manufacturer", "beam")))


def load():
    S = pd.read_csv(D / "std_colours.csv")
    S = S[S.dataset == "SPSS"]
    key = dict(zip(S.key, zip(S.bprp, S.cls)))
    F = SS.screened(pd.read_csv(D / "std_offset_frames.csv"))
    F = F[F.dataset == "SPSS"].copy()
    F["k"] = F.object.map(AN.norm)
    F["bprp"] = [key.get(k, (np.nan,))[0] for k in F.k]
    F["cls"] = [key.get(k, (np.nan, "?"))[1] for k in F.k]
    return F[F.bprp.notna()]


def slope(p, col):
    if len(p) < 4 or np.ptp(p.bprp) < 0.5:
        return np.nan
    return -np.polyfit(p.bprp, p[col], 1)[0]


def main():
    F = load()
    G0 = F[(F.version == "manufacturer") & (F.variant == "beam") & (F.std_snr >= 100) & F.dl_gnd.notna()]
    aset = G0.groupby("filter").object.apply(set).to_dict()
    rows = []
    for v1, v2 in PAIRS:
        a = F[(F.version == v1[0]) & (F.variant == v1[1])].set_index("file")
        b = F[(F.version == v2[0]) & (F.variant == v2[1])].set_index("file")
        J = a[["object", "filter", "unit", "bprp", "cls", "d_gnd", "a_field"]].join(b[["d_gnd", "a_field"]], rsuffix="_2", how="inner")
        J["dd"] = 1e3 * (J.d_gnd - J.d_gnd_2)
        J["da"] = 1e3 * (J.a_field - J.a_field_2)
        for f, g in J.groupby("filter"):
            p = g.groupby("object").agg(dd=("dd", "median"), bprp=("bprp", "first"))
            pa = p[p.index.isin(aset.get(f, set()))]
            da, dct, dcta = g.da.median(), slope(p, "dd"), slope(pa, "dd")
            ok = abs(da) >= 5
            rows.append(dict(change=f"{v1[0]}/{v1[1]} - {v2[0]}/{v2[1]}", filter=f, pivot=int(f[1:4]), d_afield=round(da, 1),
                             d_ct_std=round(dct, 1), r=round(dct / da, 2) if ok else np.nan, n=len(p),
                             d_ct_std_A=round(dcta, 1), r_A=round(dcta / da, 2) if ok else np.nan, n_A=len(pa)))
    R = pd.DataFrame(rows)
    R.to_csv(D / "std_colorterm_response.csv", index=False)
    r = R.groupby("filter").agg(pivot=("pivot", "first"), r_med=("r", "median"), r_min=("r", "min"), r_max=("r", "max"),
                                r_A=("r_A", "median"), n_changes=("r", "count"))

    # measured standard colour term (variant A: local ZP, SNR >= 100) per unit and pooled, manufacturer beam
    G = F[(F.version == "manufacturer") & (F.variant == "beam")].copy()
    G["A"] = np.where(G.std_snr >= 100, G.dl_gnd, np.nan)
    out = []
    for f, g in G.groupby("filter"):
        if f not in r.index or not np.isfinite(r.loc[f, "r_med"]):
            continue
        for unit, gg in [("all", g)] + list(g.groupby("unit")):
            p = gg.dropna(subset=["A"]).groupby("object").agg(d=("A", lambda v: 1e3 * np.median(v)), bprp=("bprp", "first"))
            s, e = CT.fit(p)
            if not np.isfinite(s):
                continue
            af = 1e3 * gg.loc[gg.A.notna(), "a_field"].median()
            rr = r.loc[f]
            ra = rr.r_A if np.isfinite(rr.r_A) else rr.r_med
            out.append(dict(filter=f, pivot=int(rr["pivot"]), unit=unit, n=len(p), a_field=round(af, 1), ct=round(s, 1), ct_err=round(e, 1),
                            r_A=ra, r_all=rr.r_med, ct_expected=round(ra * af, 1), ct_expected_all=round(rr.r_med * af, 1),
                            z=round((s - ra * af) / e, 1) if e > 0 else np.nan,
                            z_all=round((s - rr.r_med * af) / e, 1) if e > 0 else np.nan))
    E = pd.DataFrame(out).sort_values(["pivot", "unit"])
    E.to_csv(D / "std_colorterm_expected.csv", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 300)
    print(r.sort_values("pivot").to_string())
    print(E[E["pivot"] < 560].to_string(index=False))




def figure():
    """O8: field colour term, the standards' expected colour term if the field term were a bandpass error (r x a_field),
    and the measured standard colour term, blue-green filters, per unit (manufacturer beam)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    E = pd.read_csv(D / "std_colorterm_expected.csv")
    E = E[(E["pivot"] <= 534) & ((E.unit != "all") | (E.groupby("filter").unit.transform("size") == 2))]
    E = E[E.unit != "all"].reset_index(drop=True) if (E.unit != "all").any() else E
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(E))
    ax.bar(x - 0.25, E.a_field, 0.25, color="tab:purple", alpha=0.55, label="field colour term (corrected XP)")
    ax.bar(x, E.ct_expected, 0.25, color="0.6", label="expected for the standards if the field term were a bandpass error (r × field; r of the same stars)")
    lo, hi = np.minimum(E.ct_expected, E.ct_expected_all), np.maximum(E.ct_expected, E.ct_expected_all)
    ax.vlines(x, lo, hi, color="0.25", lw=2, label="range with r of all SPSS standards")
    ax.errorbar(x + 0.25, E.ct, E.ct_err, fmt="o", color="k", capsize=3, label="measured standard colour term (ground truth, local ZP, SNR ≥ 100)")
    ax.set_xticks(x); ax.set_xticklabels([f"{f}\n{u}\nr={r:.2f}" for f, u, r in zip(E["filter"], E.unit, E.r_A)], fontsize=8)
    ax.axhline(0, color="k", lw=0.6); ax.grid(axis="y", alpha=0.3)
    ax.set_ylabel("colour term [mmag/mag]")
    ax.set_title("O8  Standard vs field colour term with the nonlinear response r = d(ct_std)/d(a_field) measured from three curve changes\n"
                 "(SPSS, manufacturer·beam; r from beam−nobeam, mhj−manufacturer, hyp−manufacturer on the same frames)", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(D / "O8_std_colorterm_expected.png", dpi=130)


if __name__ == "__main__":
    main()
    figure()
