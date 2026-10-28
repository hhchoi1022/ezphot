#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260924_stepA_analyze.py — Step A triage of offsets shared by all references (from stepA_frames.csv).

Per (dataset, filter):
  offset        : Delta_gnd (star medians -> median), and with corrected/raw XP truths
  ZP reference  : zp_red - zp_blue, zp_raw - zp_cor, zp_faint - zp_bright (field-star dependence of the ZP)
  colour trend  : field residual profile r(c) in BP-RP bins; linear fit over 0.4-1.6 extrapolated to the
                  standard's colour -> Delta_trend; anomaly = Delta_gnd - Delta_trend
  conditions    : within-star slopes of Delta_gnd vs airmass, log10(FLUX_MAX), humidity, seeing;
                  medians per exposure time, per unit, per night
  water proxy   : CALSPEC only — (ZP_f - 2.5 log t) minus the mean of the two neighbour bands observed on the
                  same object within 10 min, per frame; within-unit variation tracks PWV
Outputs: stepA_summary.csv, A1_m825.png, A2_m375w.png, A3_overview.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("/home/hhchoi1022/ezphot/log/20260924_stepA")
import sys as _sys
TAG = _sys.argv[1] if len(_sys.argv) > 1 else ""
BINS = np.array([-0.6, 0.4, 0.8, 1.2, 1.6, 2.0, 3.6])
BC = 0.5 * (BINS[1:] + BINS[:-1]); BC[0] = 0.2; BC[-1] = 2.4
NEIGH = {"m400": ("m425",), "m825": ("m800", "m850"), "m800": ("m775", "m825"), "m850": ("m825", "m875"),
         "m725": ("m700", "m750"), "m775": ("m750", "m800")}

F = pd.read_csv(OUT / f"stepA_frames{TAG}.csv")
F["logfmax"] = np.log10(F.star_fmax.clip(lower=1))
F["logexp"] = np.log10(F.exptime)
F["zp_t"] = F.zp_cor - 2.5 * np.log10(F.exptime)
F["night"] = np.floor(F.mjd - 0.5).astype("Int64")


def within_slope(g, x, y="d_gnd", min_range=None):
    """pooled within-star slope of y vs x (star means removed); returns slope, err, n.
    min_range: keep only stars whose x spans at least this much (airmass needs a real lever)."""
    g = g.dropna(subset=[x, y])
    if min_range is not None:
        rng = g.groupby("object")[x].transform(lambda v: v.max() - v.min())
        g = g[rng >= min_range]
    if len(g) < 8:
        return np.nan, np.nan, len(g)
    xm = g[x] - g.groupby("object")[x].transform("mean")
    ym = g[y] - g.groupby("object")[y].transform("median")
    if np.nanstd(xm) < 1e-6:
        return np.nan, np.nan, len(g)
    keep = np.abs(ym - np.median(ym)) < 5 * 1.4826 * np.median(np.abs(ym - np.median(ym))) + 1e-4
    xm, ym = xm[keep], ym[keep]
    A = np.vstack([xm, np.ones(len(xm))]).T
    sol, res, *_ = np.linalg.lstsq(A, ym, rcond=None)
    dof = max(len(xm) - 2, 1)
    s2 = float(np.sum((ym - A @ sol) ** 2) / dof)
    cov = s2 * np.linalg.inv(A.T @ A)
    return float(sol[0]), float(np.sqrt(cov[0, 0])), int(len(xm))


def profile(g):
    """pooled field residual profile (median over frames of per-frame bin medians)."""
    return np.array([np.nanmedian(g[f"rb{k}"]) for k in range(len(BINS) - 1)])


# ---- water proxy (CALSPEC: neighbours observed on the same object within 10 min)
cal = F[F.dataset == "CALSPEC"]
wp = {}
for f, nb in NEIGH.items():
    a = cal[cal["filter"] == f][["object", "mjd", "zp_t", "file"]]
    parts = []
    for n in nb:
        b = cal[cal["filter"] == n][["object", "mjd", "zp_t"]].rename(columns={"mjd": "mjd_n", "zp_t": f"zp_{n}"})
        m = pd.merge_asof(a.sort_values("mjd"), b.sort_values("mjd_n"), left_on="mjd", right_on="mjd_n",
                          by="object", tolerance=0.007, direction="nearest")
        parts.append(m[["file", f"zp_{n}"]])
    m = parts[0]
    for p_ in parts[1:]:
        m = m.merge(p_, on="file")
    cols = [c for c in m.columns if c.startswith("zp_")]
    m = m.merge(a[["file", "zp_t"]], on="file")
    m["wproxy"] = m.zp_t - m[cols].mean(axis=1)
    wp.update(dict(zip(m.file, m.wproxy)))
F["wproxy"] = F.file.map(wp)

# ---- per (dataset, filter) summary
rows = []
for (ds, f), g in F.groupby(["dataset", "filter"]):
    gs = g.dropna(subset=["d_gnd"])
    if gs.object.nunique() < 2:
        continue
    per = gs.groupby("object").agg(d=("d_gnd", "median"), dc=("d_cor", "median"), dr=("d_raw", "median"),
                                    c=("c_star", "first"), n=("d_gnd", "size"))
    prof = profile(g)
    fit = np.isfinite(prof[1:5])
    if fit.sum() >= 3:
        k, b = np.polyfit(BC[1:5][fit], prof[1:5][fit], 1)
        trend = per.c.apply(lambda c: -(k * c + b))   # Delta_std = -r(c_std)
    else:
        k, trend = np.nan, per.c * np.nan
    row = dict(dataset=ds, filter=f, pivot_nm=int(f[1:4]), n_stars=len(per), n_frames=len(gs),
               d_gnd=1e3 * per.d.median(), d_cor=1e3 * per.dc.median(), d_raw=1e3 * per.dr.median(),
               d_err=1e3 * 1.4826 * np.median(np.abs(per.d - per.d.median())) / np.sqrt(len(per)),
               zp_red_blue=1e3 * (g.zp_red - g.zp_blue).median(), zp_raw_cor=1e3 * (g.zp_raw - g.zp_cor).median(),
               zp_faint_bright=1e3 * (g.zp_gfaint - g.zp_gbright).median(),
               field_slope=1e3 * k, prof_red_excess=1e3 * (prof[5] - (k * BC[5] + b)) if np.isfinite(k) else np.nan,
               d_trend=1e3 * trend.median(), anomaly=1e3 * (per.d - trend).median(),
               c_std=per.c.median(), c_field=g.field_c_med.median())
    for x, lab, mr in [("airmass", "s_airmass", 0.15), ("logfmax", "s_logfmax", 0.3), ("humidity", "s_humid", 5.0),
                       ("seeing", "s_seeing", 0.5), ("wproxy", "s_wproxy", 0.02), ("logexp", "s_logexp", 0.3)]:
        s, e, n = within_slope(gs, x, min_range=mr)
        row[lab] = 1e3 * s if np.isfinite(s) else np.nan
        row[lab + "_err"] = 1e3 * e if np.isfinite(e) else np.nan
    rows.append(row)
S = pd.DataFrame(rows).sort_values(["dataset", "pivot_nm"]).round(2)
S.to_csv(OUT / f"stepA_summary{TAG}.csv", index=False)
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40)
print(S[["dataset", "filter", "n_stars", "d_gnd", "d_err", "d_cor", "d_raw", "zp_red_blue", "zp_raw_cor", "zp_faint_bright",
         "field_slope", "prof_red_excess", "d_trend", "anomaly", "s_airmass", "s_airmass_err", "s_logfmax", "s_logfmax_err",
         "s_logexp", "s_logexp_err", "s_seeing", "s_seeing_err", "s_wproxy", "s_wproxy_err"]].to_string(index=False))


# ---- A1: m825 focus
def focus_fig(filt, fname, title):
    g = F[(F["filter"] == filt)].dropna(subset=["d_gnd"])
    if not len(g):
        return
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    ax = ax.ravel()
    cmap = plt.get_cmap("tab20")
    objs = sorted(g.object.unique()); oc = {o: cmap(i % 20) for i, o in enumerate(objs)}
    for ds, mk in [("CALSPEC", "o"), ("SPSS", "^")]:
        s = g[g.dataset == ds]
        for o in sorted(s.object.unique()):
            so = s[s.object == o]
            lab = f"{ds[:3]} {o}" if len(objs) <= 14 else None
            ax[0].scatter(so.mjd, 1e3 * so.d_gnd, s=10, marker=mk, color=oc[o], alpha=0.7, label=lab)
            ax[1].scatter(so.airmass, 1e3 * so.d_gnd, s=10, marker=mk, color=oc[o], alpha=0.7)
            ax[2].scatter(so.star_fmax, 1e3 * so.d_gnd, s=10, marker=mk, color=oc[o], alpha=0.7)
            ax[3].scatter(so.humidity, 1e3 * so.d_gnd, s=10, marker=mk, color=oc[o], alpha=0.7)
            ax[4].scatter(so.wproxy, 1e3 * so.d_gnd, s=10, marker=mk, color=oc[o], alpha=0.7)
    ax[2].set_xscale("log")
    for a_, xl in zip(ax[:5], ["MJD", "airmass", "standard FLUX_MAX [ADU]", "humidity [%]",
                              "water proxy: ZP_t(f) - mean ZP_t(neighbours) [mag]"]):
        a_.axhline(0, color="k", lw=0.6); a_.grid(alpha=0.3); a_.set_xlabel(xl); a_.set_ylabel("Delta vs own spectrum [mmag]")
    ax[0].legend(fontsize=6.5, ncol=2)
    # field residual profiles by dataset
    for ds, col in [("CALSPEC", "#1f4e8c"), ("SPSS", "#C25B4E")]:
        s = F[(F["filter"] == filt) & (F.dataset == ds)]
        if len(s):
            p = profile(s)
            ax[5].plot(BC, -1e3 * p, "o-", color=col, label=f"{ds} field stars (-residual = Delta a star of that colour would show)")
            sd = s.dropna(subset=["d_gnd"])
            per = sd.groupby("object").agg(d=("d_gnd", "median"), c=("c_star", "first"))
            ax[5].scatter(per.c, 1e3 * per.d, marker="*", s=90, color=col, edgecolor="k", label=f"{ds} standards")
    ax[5].axhline(0, color="k", lw=0.6); ax[5].grid(alpha=0.3); ax[5].legend(fontsize=8)
    ax[5].set(xlabel="BP-RP", ylabel="Delta-equivalent [mmag]",
              title="field-star residual profile vs standards")
    fig.suptitle(title, fontsize=12.5)
    fig.tight_layout(); fig.savefig(OUT / fname, dpi=130)


focus_fig("m825", f"A1_m825{TAG}.png", "A1  m825 triage: Delta vs time / airmass / peak counts / humidity / water proxy; colour profile")
focus_fig("m375w", f"A2_m375w{TAG}.png", "A2  m375w triage: Delta vs time / airmass / peak counts / humidity; colour profile (red-leak test)")

# ---- A3 overview
fig, ax = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
for ds, col, mk in [("CALSPEC", "#1f4e8c", "o"), ("SPSS", "#C25B4E", "^")]:
    s = S[S.dataset == ds].sort_values("pivot_nm")
    ax[0].errorbar(s.pivot_nm, s.d_gnd, yerr=s.d_err, color=col, marker=mk, lw=1.3, label=f"{ds}: Delta (own spectrum)")
    ax[0].plot(s.pivot_nm, s.d_trend, color=col, ls=":", marker=mk, ms=3, label=f"{ds}: field colour trend extrapolated")
    ax[1].plot(s.pivot_nm, s.zp_red_blue, color=col, marker=mk, lw=1.3, label=f"{ds}: ZP(red field) - ZP(blue field)")
    ax[1].plot(s.pivot_nm, s.zp_raw_cor, color=col, marker=mk, ls="--", lw=1.0, label=f"{ds}: ZP(raw XP) - ZP(corr XP)")
    ax[2].errorbar(s.pivot_nm, s.s_airmass, yerr=s.s_airmass_err, color=col, marker=mk, lw=1.2, label=f"{ds}: dDelta/dX")
    ax[2].errorbar(s.pivot_nm + 2, s.s_logfmax, yerr=s.s_logfmax_err, color=col, marker="s", ls="--", lw=1.0,
                   label=f"{ds}: dDelta/dlog(peak)")
for a_, yl in zip(ax, ["[mmag]", "[mmag]", "[mmag per unit]"]):
    a_.axhline(0, color="k", lw=0.6); a_.grid(alpha=0.3); a_.set_ylabel(yl); a_.legend(fontsize=8, ncol=2)
ax[0].set_ylim(-60, 160)
ax[0].set_title("A. standard offset and what the field-star colour trend predicts at the standard's colour")
ax[1].set_title("B. ZP-reference sensitivity (which field stars / which XP define the ZP)")
ax[2].set_title("C. within-star dependence on airmass and on peak counts (nonlinearity)")
ax[2].set_xlabel("pivot wavelength [nm]"); ax[2].set_ylim(-80, 80)
fig.tight_layout(); fig.savefig(OUT / f"A3_overview{TAG}.png", dpi=130)
print("saved A1/A2/A3 + stepA_summary.csv")
