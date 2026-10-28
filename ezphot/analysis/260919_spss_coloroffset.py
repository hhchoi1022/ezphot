#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260919_spss_coloroffset.py — colour-magnitude-offset relation of SPSS standards vs field colour term.

Physics: if the field colour term a (XP-reference residual slope) is a REAL instrument-curve effect,
the standards' Delta_spss (ground-truth spectra) must slope as -a with star colour; if a is an
XP-reference colour systematic, Delta_spss stays flat. Delta_xp (truth = star's own corrected XP)
must slope as -a BY CONSTRUCTION -> machinery null test.

S1: per-filter Delta vs BP-RP panels (star x unit medians), fitted slope + (-a_field) line
S2: summary -slope(Delta_spss) and -slope(Delta_xp) vs a_field per filter
Also writes spss_star_medians.csv (per star/filter/unit medians — input for the delta_f map).

  MPLCONFIGDIR=/tmp/ezphot-matplotlib /home/hhchoi1022/miniconda3/envs/too/bin/python -u \\
      ezphot/analysis/260919_spss_coloroffset.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402
import pandas as pd                 # noqa: E402

warnings.filterwarnings("ignore")
plt.rcParams["axes.unicode_minus"] = False
OUT = Path("/home/hhchoi1022/ezphot/log/20260918_spss")
MIN_FRAMES = 4


def wslope(x, y, w):
    """weighted straight-line slope with error."""
    x, y, w = np.asarray(x, float), np.asarray(y, float), np.asarray(w, float)
    fin = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    if fin.sum() < 4 or np.ptp(x[fin]) < 0.3:
        return np.nan, np.nan
    x, y, w = x[fin], y[fin], w[fin]
    xm = np.average(x, weights=w); ym = np.average(y, weights=w)
    sxx = np.sum(w * (x - xm) ** 2)
    b = np.sum(w * (x - xm) * (y - ym)) / sxx
    resid = y - ym - b * (x - xm)
    dof = max(len(x) - 2, 1)
    berr = np.sqrt(np.sum(w * resid ** 2) / dof / sxx)
    return float(b), float(berr)


EXCLUDE_UNITS = {"7DT11"}          # filter wheel stuck on r for the whole campaign (report 20.2)


def main():
    fr = pd.read_csv(OUT / "spss_frames.csv")
    ok = fr[(fr.status == "ok") & ~fr.unit.isin(EXCLUDE_UNITS)].copy()
    # per (filter, unit, object) medians
    med = (ok.dropna(subset=["c_star"])
             .groupby(["filter", "unit", "object"])
             .agg(n=("file", "size"), c=("c_star", "first"),
                  d_spss=("delta_spss", "median"), d_xp=("delta_xp", "median"),
                  scat=("delta_spss", lambda v: 1.4826 * np.nanmedian(np.abs(v - np.nanmedian(v)))),
                  serr=("star_err", "median"))
             .reset_index())
    med = med[med.n >= MIN_FRAMES]
    med.to_csv(OUT / "spss_star_medians.csv", index=False)
    afield = (ok.groupby("filter")
                .agg(a_all=("a_all", "median"), a_frame=("a_frame", "median"),
                     nfr=("file", "size"), c_ref=("c_ref", "median")))

    filts = [f for f in sorted(med["filter"].unique(), key=lambda s: (s[0] != "m", s))
             if f.startswith("m") and f[1:4].isdigit() and "w" not in f]
    rows = []
    ncol = 5
    nrow = int(np.ceil(len(filts) / ncol))
    fig = plt.figure(figsize=(4.2 * ncol, 3.4 * nrow))
    for k, f in enumerate(filts):
        g = med[med["filter"] == f].copy()
        ax = fig.add_subplot(nrow, ncol, k + 1)
        a_f = float(afield.loc[f, "a_all"]) if f in afield.index else np.nan
        c0 = float(afield.loc[f, "c_ref"]) if f in afield.index else 0.9
        units = sorted(g.unit.unique())
        for un in units:
            gu = g[g.unit == un].copy()
            gu["dr"] = gu.d_spss - gu.d_spss.median()
            ax.errorbar(gu.c, gu.dr, yerr=np.hypot(gu.scat / np.sqrt(gu.n), gu.serr.fillna(0.01)),
                        fmt="o", ms=4, alpha=0.75, label=un)
        # pooled slope after per-unit centering
        y, x, w = [], [], []
        for un in units:
            gu = g[g.unit == un]
            y += list(gu.d_spss - gu.d_spss.median()); x += list(gu.c)
            w += list(1.0 / np.maximum(gu.scat / np.sqrt(gu.n), 0.004) ** 2)
        s_spss, e_spss = wslope(x, y, w)
        yx = []
        for un in units:
            gu = g[g.unit == un]
            yx += list(gu.d_xp - gu.d_xp.median())
        s_xp, e_xp = wslope(x, yx, w)
        rows.append(dict(filter=f, pivot=int(f[1:4]), n_stars=len(g),
                         slope_spss=s_spss, slope_spss_err=e_spss,
                         slope_xp=s_xp, slope_xp_err=e_xp,
                         a_field=a_f, nfr=int(afield.loc[f, "nfr"]) if f in afield.index else 0))
        cc = np.linspace(-0.7, max(1.6, np.nanmax(g.c) + 0.1), 10)
        if np.isfinite(s_spss):
            ax.plot(cc, s_spss * (cc - np.nanmedian(g.c)), "-", color="k", lw=1.4,
                    label=f"fit {s_spss:+.3f}")
        if np.isfinite(a_f):
            ax.plot(cc, -a_f * (cc - np.nanmedian(g.c)), "--", color="#d62728", lw=1.4,
                    label=f"$-a_{{field}}$ {-a_f:+.3f}")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title(f, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6)
        if k % ncol == 0:
            ax.set_ylabel("$\\Delta_{SPSS}$ − unit median [mag]")
        if k >= (nrow - 1) * ncol:
            ax.set_xlabel("BP−RP")
    fig.suptitle("S1  SPSS standards: colour vs magnitude offset (truth = SPSS ground spectra), "
                 "black = fit, red dashed = $-a_{field}$ (XP colour term) — agreement ⇒ real curve effect",
                 y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(OUT / "S1_coloroffset.png", dpi=130)

    T = pd.DataFrame(rows).sort_values("pivot")
    T.to_csv(OUT / "spss_colour_slopes.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(14.5, 6))
    ax[0].errorbar(T["pivot"], -T.slope_spss, yerr=T.slope_spss_err, fmt="o", ms=7, capsize=3,
                   color="#2ca02c", label="−slope($\\Delta_{SPSS}$ vs c)  [ground truth]")
    ax[0].errorbar(T["pivot"], -T.slope_xp, yerr=T.slope_xp_err, fmt="s", ms=5, capsize=2,
                   color="#888", alpha=0.8, label="−slope($\\Delta_{XP}$ vs c)  [null test]")
    ax[0].plot(T["pivot"], T.a_field, "d--", color="#d62728", ms=6, label="field colour term a (XP)")
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set(xlabel="pivot [nm]", ylabel="colour slope [mag/mag]",
              title="(A) standards' colour slope vs field colour term")
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=9)
    lim = np.nanmax(np.abs(np.r_[T.a_field, -T.slope_spss])) * 1.2
    ax[1].errorbar(T.a_field, -T.slope_spss, yerr=T.slope_spss_err, fmt="o", ms=7, color="#2ca02c")
    for _, s in T.iterrows():
        ax[1].annotate(s["filter"], (s.a_field, -s.slope_spss), textcoords="offset points",
                       xytext=(4, 4), fontsize=7)
    ax[1].plot([-lim, lim], [-lim, lim], "k--", lw=0.8, label="1:1 (real curve effect)")
    ax[1].axhline(0, color="grey", lw=0.6)
    ax[1].set(xlabel="field colour term a (XP reference)", ylabel="−slope($\\Delta_{SPSS}$)",
              xlim=(-lim, lim), ylim=(-lim, lim), title="(B) agreement test")
    ax[1].grid(alpha=0.3); ax[1].legend(fontsize=9)
    fig.suptitle("S2  is the field colour term a real instrument effect? "
                 "(on 1:1 ⇒ real curve; at 0 ⇒ XP-reference artifact)")
    fig.tight_layout()
    fig.savefig(OUT / "S2_slope_compare.png", dpi=145)
    print(T.round(4).to_string(index=False))
    print(f"saved -> S1_coloroffset.png, S2_slope_compare.png, spss_colour_slopes.csv, "
          f"spss_star_medians.csv in {OUT}")


if __name__ == "__main__":
    main()
