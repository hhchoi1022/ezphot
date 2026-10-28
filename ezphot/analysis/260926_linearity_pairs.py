#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260926_linearity_pairs.py — pairwise exposure comparison for the linearity series (reads lin_stars.csv / lin_frames.csv
of 260926_linearity_test.py; no re-photometry).

For each unit and exposure pair (t_s < t_l):
  F_t(star)  : median over the 5 frames of that exposure of the forced 10" flux, frames scaled to a common transparency
               with stars that are linear in that exposure (total peak level 1000-6000 ADU)
  R          = (F_s / t_s) / (F_l / t_l);  dR = -2.5 log10(R / R_ref)  [mmag, >0: short exposure measured fainter]
  R_ref      = median R of stars in the Gaia-G range whose median long-frame total peak level is 1000-6000 ADU and
               whose expected short-exposure peak above sky is > 200 ADU (selection on G only: no Eddington bias)
  binning    : Gaia G (independent of both measurements); per bin the median expected short-exposure peak above sky
               [e-] and long-exposure total level [ADU] are reported
Low-signal pairs (1,5), (1,10), (3,10), (3,20), (5,20): faint end = short-exposure peak of a few e-.
Bright pairs (5,40), (5,80), (10,100), (10,200), (20,200): bright end = long-exposure total level up to saturation,
compared with the linear short exposure.
Usage: python 260926_linearity_pairs.py <tag>   -> lin_pairs.csv, L3_pairs.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTROOT = Path("/home/hhchoi1022/ezphot/log/20260926_linearity")
LOW = [(1, 5), (1, 10), (3, 10), (3, 20), (5, 20)]
HIGH = [(5, 40), (5, 80), (10, 100), (10, 200), (20, 200), (10, 40)]
TOT_LO, TOT_HI = 1000.0, 6000.0


def per_exposure_flux(g):
    """g: one unit, one exposure time -> per-star median flux, frames scaled with linear stars."""
    g = g.copy()
    g["tot"] = g.peak_adu + g.sky_adu
    P = g.pivot_table(index="sid", columns="name", values="f10_new", aggfunc="first")
    T = g.pivot_table(index="sid", columns="name", values="tot", aggfunc="first")
    lin = (T > TOT_LO) & (T < TOT_HI)
    ref = P.where(lin).median(axis=1)
    scale = P.where(lin).div(ref, axis=0).median(axis=0)
    F = P.div(scale, axis=1).median(axis=1)
    out = pd.DataFrame(dict(F=F, tot=T.median(axis=1), peak=g.pivot_table(index="sid", columns="name", values="peak_adu",
                                                                           aggfunc="first").median(axis=1)))
    pk_frac = (g[(g.peak_adu + g.sky_adu > TOT_LO) & (g.peak_adu + g.sky_adu < TOT_HI)].eval("peak_adu / f10_new")).median()
    return out, pk_frac


def boot(v, n=300):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 15:
        return np.nan, np.nan
    rng = np.random.default_rng(0)
    b = np.array([np.median(rng.choice(v, len(v))) for _ in range(n)])
    med = np.median(v)
    if med <= 0:
        return np.nan, np.nan
    return float(-2.5e3 * np.log10(med)), float(np.std(-2.5e3 * np.log10(np.clip(b, 1e-6, None))))


if __name__ == "__main__":
    tag = sys.argv[1]
    d = OUTROOT / tag
    cols = ["sid", "g", "iso", "peak_adu", "sky_adu", "f10_new", "name", "unit", "exptime", "egain"]
    A = pd.read_csv(d / "lin_stars.csv", usecols=lambda c: c in cols)
    A = A[A.iso.astype(bool) & np.isfinite(A.f10_new)]
    gmag = A.groupby("sid").g.first()
    rows, stars = [], []
    for unit, U in A.groupby("unit"):
        egain = U.egain.iloc[0]
        E = {t: per_exposure_flux(g) for t, g in U.groupby("exptime")}
        for ts, tl in LOW + HIGH:
            if ts not in E or tl not in E:
                continue
            (Fs, pks), (Fl, pkl) = E[ts], E[tl]
            J = (Fs[["F", "peak"]].rename(columns={"F": "F_s", "peak": "peak_s"})
                 .join(Fl[["F", "tot"]].rename(columns={"F": "F_l"}), how="inner").join(gmag))
            J["R"] = (J.F_s / ts) / (J.F_l / tl)
            J["peak_s_exp_adu"] = J.F_l * pkl * ts / tl
            # reference = Gaia-G range (independent of both measurements) whose median long-exposure total level is in
            # the linear window and whose expected short-exposure peak is > 200 ADU.  Selecting on a measured,
            # noisy quantity (short peak or long flux) biases R_ref by ~1 % (Eddington).
            gb = np.floor(J.g / 0.25) * 0.25
            st = J.groupby(gb).agg(tot=("tot", "median"), pk=("peak_s_exp_adu", "median"), n=("R", "size"))
            okb = st[(st.tot > TOT_LO) & (st.tot < TOT_HI) & (st.pk > 200) & (st.n >= 5)].index
            ref = J[gb.isin(okb)]
            if len(ref) < 20:
                continue
            J["Rn"] = J.R / ref.R.median()
            J["peak_s_exp_e"] = J.F_l * pkl * ts / tl * egain           # expected short-exposure peak from the long flux
            J = J.assign(unit=unit, pair=f"{ts:g}/{tl:g}", ts=ts, tl=tl)
            stars.append(J.reset_index())
    S = pd.concat(stars, ignore_index=True)
    S.to_csv(d / "lin_pairs_stars.csv", index=False)
    # pooled over units, binned on Gaia G per pair
    for pair, P in S.groupby("pair"):
        for lo in np.arange(8.0, 19.0, 0.5):
            m = (P.g >= lo) & (P.g < lo + 0.5)
            if m.sum() < 15:
                continue
            med, err = boot(P.loc[m, "Rn"])
            rows.append(dict(pair=pair, ts=P.ts.iloc[0], tl=P.tl.iloc[0], g_lo=lo, n=int(m.sum()), dmag=med, err=err,
                             peak_s_e=P.loc[m, "peak_s_exp_e"].median(), tot_l=P.loc[m, "tot"].median()))
    R = pd.DataFrame(rows); R.to_csv(d / "lin_pairs.csv", index=False)
    pd.set_option("display.width", 220)
    print("LOW-SIGNAL pairs: dR [mmag] (>0: short exposure fainter) vs expected short-exposure peak [e-], pooled over units")
    for pair in [f"{a:g}/{b:g}" for a, b in LOW]:
        r = R[R.pair == pair]
        print(f"  {pair:6s}", "  ".join(f"{p:6.1f}e:{v:+6.1f}±{e:4.1f}" for p, v, e in zip(r.peak_s_e, r.dmag, r.err) if p < 400))
    print("\nBRIGHT pairs: dR [mmag] (<0: long exposure fainter = sub-linear at high level) vs long-exposure total level [ADU]")
    for pair in [f"{a:g}/{b:g}" for a, b in HIGH]:
        r = R[(R.pair == pair) & (R.tot_l > 3000)]
        print(f"  {pair:7s}", "  ".join(f"{t:7.0f}:{v:+6.1f}±{e:4.1f}" for t, v, e in zip(r.tot_l, r.dmag, r.err)))
    # figure
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    cm = plt.get_cmap("viridis")
    for i, (a, b) in enumerate(LOW):
        r = R[(R.pair == f"{a:g}/{b:g}") & (R.peak_s_e < 1000)]
        ax[0].errorbar(r.peak_s_e, r.dmag, r.err, marker="o", ms=3, capsize=0, color=cm(i / (len(LOW) - 1)), label=f"{a:g} s vs {b:g} s")
    ax[0].plot([5.5, 11, 23, 45, 140], [51, 37, 23, 10, 0], "k--", lw=1.2, label="2024 medium band (XP-based)")
    ax[0].axhline(0, color="k", lw=0.6); ax[0].set_xscale("log"); ax[0].set_ylim(-40, 80); ax[0].grid(alpha=0.3)
    ax[0].set(xlabel="expected star peak above sky in the short exposure [e⁻]", ylabel="dR [mmag] (>0: short exposure fainter)",
              title="(a) low signal: short vs longer exposure (Gaia-G bins, linear reference)")
    ax[0].legend(fontsize=7)
    for i, (a, b) in enumerate(HIGH):
        r = R[(R.pair == f"{a:g}/{b:g}")]
        ax[1].errorbar(r.tot_l, -r.dmag, r.err, marker="s", ms=3, capsize=0, color=cm(i / (len(HIGH) - 1)), label=f"{b:g} s vs {a:g} s")
    ax[1].axhline(0, color="k", lw=0.6); ax[1].set_xscale("log"); ax[1].grid(alpha=0.3); ax[1].set_ylim(-10, 60)
    ax[1].axvspan(TOT_LO, TOT_HI, color="0.92", zorder=0)
    ax[1].set(xlabel="total level at the star peak in the long exposure (sky + star) [ADU]",
              ylabel="flux deficit of the long exposure [mmag]", title="(b) high signal: long vs linear short exposure")
    ax[1].legend(fontsize=7)
    fig.suptitle(f"L3  Pairwise exposure comparison ({tag}; all units pooled; grey = reference range)", fontsize=10)
    fig.tight_layout(); fig.savefig(d / "L3_pairs.png", dpi=125)
    print("saved", d / "L3_pairs.png")
