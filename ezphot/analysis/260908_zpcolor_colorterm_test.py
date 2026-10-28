#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260908_zpcolor_colorterm_test.py

Does applying the pipeline's colour term fix the CALSPEC residuals of line-dominated stars
(DA white dwarfs in m475 / m675)?  Four calibrations of the same measurements are compared:

  none   : pipeline magnitude MAGSKY_APER_3 (colour-independent ZP), as in the 2026-09-08 analysis
  ct_gr  : + header colour term K_COLOR_APER_3_g-r * (g-r) + C_COLOR_APER_3_g-r
  ct_nbr : + header colour term of the (filter - neighbouring medium band) pair used by the refit
  shift  : re-calibrated with the shifted-curve reference photometry (closed loop, case 'rec')

The colour term is applied exactly as PhotometricCalibration.apply_color_terms would
(MAGSKY' = MAGSKY + a*colour + b), with the standard's colour taken from its CALSPEC spectrum
synthesised through the same curves (calspec_synphot.csv), i.e. the analogue of the catalogue
colours the terms were fitted against.  Residuals are against the synthetic magnitude
(nominal curve for none/ct_*, shifted curve for shift).

Inputs (--out-dir): calspec_frames.csv, frames.csv, colorterms_header.csv, calspec_synphot.csv,
                    refit_frames.csv, shift_recal/recal_calspec_stars.csv
Outputs: colorterm_test/colorterm_stars.csv, colorterm_summary.csv,
         figures/C7_colorterm_vs_shift.png, figures/C8_colorterm_DA_slopegraph.png

  MPLCONFIGDIR=/tmp/ezphot-matplotlib /home/hhchoi1022/anaconda3/envs/py311/bin/python \\
      ezphot/analysis/260908_zpcolor_colorterm_test.py
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

warnings.filterwarnings("ignore")

DEFAULT_OUT_DIR = Path("/home/hhchoi1022/ezphot/log/20260908_zp_color_check")
GRID = [f"m{w}" for w in range(400, 876, 25)]
STANDARDS = ["BD+02_3375", "BD-11_3759", "FEIGE110", "GD71", "GJ754.1A", "HZ4", "LDS749B", "WD0227+050",
             "WD0308-565", "WD1544-377", "WD1919+145", "WD2039-682", "WD2149+021"]
ZPERR_ANOMALY, CAL_MAX_MAGERR, CAL_MAX_FLUX, MIN_FRAMES = 0.08, 0.05, 45000.0, 10
ERR_FLOOR = 0.005
CASES = {"none": ("0.55", "o", "no colour term (pipeline)"),
         "ct_gr": ("tab:green", "^", "colour term applied, g−r (header K/C)"),
         "ct_nbr": ("tab:purple", "v", "colour term applied, filter−neighbour band (header K/C)"),
         "shift": ("tab:orange", "s", "shifted curve (closed loop, recommended shift)")}
PIVOT = {"m400": 401.3, "m425": 425.5, "m450": 450.8, "m475": 475.2, "m500": 500.3, "m525": 524.8, "m550": 550.1, "m575": 574.9,
         "m600": 600.1, "m625": 624.8, "m650": 650.1, "m675": 674.5, "m700": 700.0, "m725": 724.7, "m750": 749.0, "m775": 775.3,
         "m800": 799.4, "m825": 824.4, "m850": 849.4, "m875": 873.8,
         "u": 371.4, "g": 479.4, "r": 616.2, "i": 758.7, "z": 876.6}


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def mad(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan


def main():
    global MIN_FRAMES, GRID
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--exclude-filters", default="", help="comma-separated filters to drop (e.g. m800,m825 for coadds)")
    ap.add_argument("--min-frames", type=int, default=MIN_FRAMES, help="frames per star and filter required (use 1 for coadds)")
    ap.add_argument("--filters", default="", help="comma-separated filters (default: the 25 nm grid)")
    args = ap.parse_args()
    MIN_FRAMES = args.min_frames
    if args.filters: GRID = [f.strip() for f in args.filters.split(",") if f.strip()]
    excl = {f.strip() for f in args.exclude_filters.split(",") if f.strip()}
    out = Path(args.out_dir); d = out / "colorterm_test"; d.mkdir(exist_ok=True); fig_dir = out / "figures"

    fr = pd.read_csv(out / "frames.csv", low_memory=False)
    bad = set(fr[fr["ZPERR_APER_3"] > ZPERR_ANOMALY]["file"])
    c = pd.read_csv(out / "calspec_frames.csv")
    c = c[(c["status"] == "ok") & ~c["file"].isin(bad)]
    c = c[(c["flags"].fillna(0) <= 1) & (~c["saturated"].astype(bool)) & (c["sep_arcsec"] < 1.5)
          & (c["magerr_APER_3"] < CAL_MAX_MAGERR) & (c["flux_max"] < CAL_MAX_FLUX) & c["filter"].isin([g for g in GRID if g not in excl])].copy()
    files = set(c["file"])
    log(f"{len(c)} standard-field frames after cleaning")

    # header colour terms (APER_3) of those frames
    ct = pd.read_csv(out / "colorterms_header.csv", usecols=["file", "aper", "f1", "f2", "K", "C"])
    ct = ct[(ct["aper"] == "APER_3") & ct["file"].isin(files)]
    log(f"{len(ct)} header colour-term entries for those frames")
    gr = ct[(ct.f1 == "g") & (ct.f2 == "r")].set_index("file")
    # neighbour pair used by the refit (same rule as the closed loop)
    rf = pd.read_csv(out / "refit_frames.csv", low_memory=False)
    rf = rf[(rf.catalog == "GAIAXP_CORR_LAMOST") & (rf.aper == "APER_3")][["file", "nbr_filter"]].drop_duplicates("file").set_index("file")
    c["nbr"] = c["file"].map(rf["nbr_filter"])
    ct_key = ct.set_index(["file", "f1", "f2"])
    def lookup(row):
        try:
            r = ct_key.loc[(row["file"], row["filter"], row["nbr"])]
            return float(r["K"]), float(r["C"])
        except KeyError:
            return np.nan, np.nan
    kc = np.array([lookup(r) for _, r in c.iterrows()])
    c["K_nbr"], c["C_nbr"] = kc[:, 0], kc[:, 1]
    c["K_gr"] = c["file"].map(gr["K"]); c["C_gr"] = c["file"].map(gr["C"])

    # synthetic colours of the standards through the same curves
    syn = pd.read_csv(out / "calspec_synphot.csv")
    smag = {(o, f): m for o, f, m in zip(syn["object"], syn["filter"], syn["mag_syn"])}
    c["gr_syn"] = [smag.get((o, "g"), np.nan) - smag.get((o, "r"), np.nan) for o in c["object"]]
    c["nbr_syn"] = [smag.get((o, f), np.nan) - smag.get((o, n), np.nan) if isinstance(n, str) else np.nan
                    for o, f, n in zip(c["object"], c["filter"], c["nbr"])]
    c["d_none"] = c["delta_APER_3"]
    c["d_ct_gr"] = c["d_none"] + c["K_gr"] * c["gr_syn"] + c["C_gr"]
    c["d_ct_nbr"] = c["d_none"] + c["K_nbr"] * c["nbr_syn"] + c["C_nbr"]
    c["is_da"] = c["spectral_type"].fillna("").str.startswith("DA")

    # per-star medians
    rows = []
    for (f, o), g in c.groupby(["filter", "object"]):
        if len(g) < MIN_FRAMES: continue
        rec = dict(filter=f, object=o, is_da=bool(g["is_da"].iloc[0]), n=len(g), gr_syn=float(g["gr_syn"].iloc[0]),
                   nbr=g["nbr"].dropna().iloc[0] if g["nbr"].notna().any() else "",
                   K_gr_median=float(g["K_gr"].median()), K_nbr_median=float(g["K_nbr"].median()))
        magerr_med = float(g["magerr_APER_3"].median())
        for case in ("none", "ct_gr", "ct_nbr"):
            v = g[f"d_{case}"].dropna()
            rec[case] = float(v.median()) if len(v) else np.nan
            # error of the median: frame scatter and photometric error in quadrature, both / sqrt(n)
            rec[f"{case}_err"] = max(float(np.sqrt((mad(v) / np.sqrt(len(v))) ** 2 + (magerr_med / np.sqrt(len(v))) ** 2)), 0.001) if len(v) else np.nan
        rows.append(rec)
    S = pd.DataFrame(rows)
    # shifted-curve case from the closed loop (per-star medians, case 'rec')
    rc = pd.read_csv(out / "shift_recal" / "recal_calspec_stars.csv")
    rc = rc[rc["case"] == "rec"].set_index(["filter", "object"])
    S["shift"] = [rc["delta"].get((f, o), np.nan) for f, o in zip(S["filter"], S["object"])]
    S["shift_err"] = [rc["err"].get((f, o), np.nan) for f, o in zip(S["filter"], S["object"])]
    S["shift_nm"] = [rc["shift_nm"].get((f, o), np.nan) for f, o in zip(S["filter"], S["object"])]
    S.to_csv(d / "colorterm_stars.csv", index=False)

    # per-filter summary
    summ = []
    for f in GRID:
        g = S[S["filter"] == f]
        if not len(g): continue
        rec = dict(filter=f, pivot=PIVOT[f], n_stars=len(g), n_da=int(g.is_da.sum()),
                   K_gr_median=float(g.K_gr_median.median()), K_nbr_median=float(g.K_nbr_median.median()), nbr=g["nbr"].mode().iloc[0] if g["nbr"].notna().any() else "")
        for case in CASES:
            v = g[case]; da = g[g.is_da][case]; other = g[~g.is_da][case]
            rec[f"{case}_da_minus_other"] = float(da.median() - other.median()) if da.notna().any() and other.notna().any() else np.nan
            rec[f"{case}_between_star_mad"] = mad(v) if v.notna().sum() >= 3 else np.nan
            rec[f"{case}_median_all"] = float(v.median()) if v.notna().any() else np.nan
        summ.append(rec)
    T = pd.DataFrame(summ); T.to_csv(d / "colorterm_summary.csv", index=False)

    print("\nfilter | K(g-r) K(nbr) | DA-other: none / ct_gr / ct_nbr / shift | between-star MAD: none / ct_gr / ct_nbr / shift")
    for _, r in T.iterrows():
        print(f"{r['filter']:5s} | {r.K_gr_median:+.3f} {r.K_nbr_median:+.3f} | {r.none_da_minus_other:+.3f} / {r.ct_gr_da_minus_other:+.3f} / {r.ct_nbr_da_minus_other:+.3f} / {r.shift_da_minus_other:+.3f} | "
              f"{r.none_between_star_mad:.3f} / {r.ct_gr_between_star_mad:.3f} / {r.ct_nbr_between_star_mad:.3f} / {r.shift_between_star_mad:.3f}")

    # ---- figure C7: summary + per-filter per-star panels
    fig = plt.figure(figsize=(18, 19)); gs = fig.add_gridspec(5, 5, height_ratios=[1.2, 1, 1, 1, 1])
    for k, (col, lab, yl) in enumerate([("da_minus_other", "DA − other stars [mag]", (-0.07, 0.05)), ("between_star_mad", "between-star MAD [mag]", (0, 0.05)),
                                         ("median_all", "median residual, all stars [mag]", (-0.04, 0.04))]):
        ax = fig.add_subplot(gs[0, k])
        for case, (cc, mk, lb) in CASES.items():
            ax.plot(T["pivot"], T[f"{case}_{col}"], mk, color=cc, ms=5, label=lb, alpha=.9)
        ax.axhline(0, color="k", lw=.5); ax.set_ylim(*yl); ax.grid(alpha=.3); ax.set_title(lab, fontsize=9); ax.set_xlabel("pivot [nm]", fontsize=8)
        if k == 0: ax.legend(fontsize=7, loc="lower left")
    ax = fig.add_subplot(gs[0, 3:5])
    ax.plot(T["pivot"], T["K_gr_median"], "^", color="tab:green", label="header K(g−r), median over frames")
    ax.plot(T["pivot"], T["K_nbr_median"], "v", color="tab:purple", label="header K(filter − neighbour)")
    ax.axhline(0, color="k", lw=.5); ax.set_ylim(-0.06, 0.06); ax.grid(alpha=.3); ax.legend(fontsize=7); ax.set_title("colour terms that were applied [mag/mag]", fontsize=9); ax.set_xlabel("pivot [nm]", fontsize=8)
    for k, f in enumerate(GRID):
        a = fig.add_subplot(gs[1 + k // 5, k % 5])
        if not (T["filter"] == f).any():
            a.set_title(f"{f} (excluded)", fontsize=7); a.axis("off"); continue
        g = S[S["filter"] == f].set_index("object")
        objs = [o for o in STANDARDS if o in g.index]; x = np.arange(len(objs))
        for i, o in enumerate(objs):
            if g.loc[o, "is_da"]: a.axvspan(i - 0.5, i + 0.5, color="tab:blue", alpha=.06, lw=0)
            for j, (case, (cc, mk, lb)) in enumerate(CASES.items()):
                v = g.loc[o, case]; e = g.loc[o, f"{case}_err"]
                if np.isfinite(v): a.errorbar(i + (j - 1.5) * 0.18, v, yerr=e if np.isfinite(e) else 0, fmt=mk, color=cc, ms=4.5, capsize=1.5, zorder=3 + j)
        t = T[T["filter"] == f].iloc[0]
        a.axhline(0, color="k", lw=.5); a.set_ylim(-0.1, 0.1); a.grid(alpha=.3)
        a.set_xticks(x); a.set_xticklabels(objs, rotation=90, fontsize=6)
        a.set_title(f"{f} | DA−other: none {t.none_da_minus_other:+.3f}, ct(g−r) {t.ct_gr_da_minus_other:+.3f}, shift {t.shift_da_minus_other:+.3f}", fontsize=7)
        if k % 5 == 0: a.set_ylabel("m_obs − m_syn [mag]", fontsize=8)
    fig.suptitle("Colour-term correction vs curve shift on the CALSPEC standards (shaded = DA white dwarfs)\n"
                 "grey = pipeline, green = + header colour term (g−r), purple = + header colour term (filter − neighbour band), orange = shifted curves", fontsize=11)
    fig.tight_layout(); fig.savefig(fig_dir / "C7_colorterm_vs_shift.png", dpi=120); plt.close(fig)

    # ---- figure C8: slope graph for the Balmer-edge filters
    focus = [f for f in ["m450", "m475", "m600", "m675"] if (T["filter"] == f).any()]
    if not focus:  # e.g. broad bands: show every filter present (at most 5)
        focus = [f for f in T.sort_values("pivot")["filter"].tolist() if f in set(S["filter"])][:5]
    fig, axes = plt.subplots(1, len(focus), figsize=(4.3 * len(focus), 5.2), sharey=True)
    order = list(CASES)
    for ax, f in zip(axes, focus):
        g = S[S["filter"] == f].set_index("object")
        for o in [o for o in STANDARDS if o in g.index]:
            y = [g.loc[o, case] for case in order]
            col = "tab:blue" if g.loc[o, "is_da"] else "0.5"
            ax.plot(range(len(order)), y, "-o", color=col, ms=4, lw=1.2 if g.loc[o, "is_da"] else 0.8, alpha=.9 if g.loc[o, "is_da"] else .6)
            ax.annotate(o, (len(order) - 1 + 0.08, y[-1]), fontsize=6, color=col, va="center")
        ax.axhspan(-0.02, 0.02, color="grey", alpha=.12); ax.axhline(0, color="k", lw=.6)
        ax.set_xticks(range(len(order))); ax.set_xticklabels(["pipeline", "+ct (g−r)", "+ct (nbr)", "shifted\ncurve"], fontsize=8)
        ax.set_xlim(-0.3, len(order) + 0.9); ax.set_ylim(-0.09, 0.09); ax.grid(alpha=.3)
        t = T[T["filter"] == f].iloc[0]
        ax.set_title(f"{f}   K(g−r) = {t.K_gr_median:+.3f}\nDA−other: {t.none_da_minus_other:+.3f} → {t.ct_gr_da_minus_other:+.3f} → {t.ct_nbr_da_minus_other:+.3f} → {t.shift_da_minus_other:+.3f}", fontsize=8.5)
    axes[0].set_ylabel("m_obs − m_syn [mag]")
    axes[0].plot([], [], "-o", color="tab:blue", label="DA white dwarf"); axes[0].plot([], [], "-o", color="0.5", label="continuum-dominated star"); axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("The same measurements under four calibrations: pipeline, + header colour term (g−r), + header colour term (filter − neighbour band), shifted curves (recommended shift)", fontsize=10)
    fig.tight_layout(); fig.savefig(fig_dir / "C8_colorterm_DA_slopegraph.png", dpi=130); plt.close(fig)
    log("colorterm_test/ and figures C7, C8 written")


if __name__ == "__main__":
    main()
