#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260908_zpcolor_plot.py

Figures and a Markdown summary for the 7DT medium-band ZP colour-dependency check.
Reads only the CSV products written by ``260908_zpcolor_collect.py`` (no FITS access,
no ezphot import) so it can be re-run quickly while iterating on the plots.

Inputs (in --out-dir): frames.csv, colorterms_header.csv, refit_frames.csv,
refit_stars.csv.gz, calspec_frames.csv, calspec_synphot.csv, curves_comparison.csv
Outputs: figures/H*.png R*.png C*.png, summary_by_filter.csv, summary.md

  MPLCONFIGDIR=/tmp/ezphot-matplotlib /home/hhchoi1022/anaconda3/envs/py311/bin/python \\
      ezphot/analysis/260908_zpcolor_plot.py --out-dir /home/hhchoi1022/ezphot/log/20260908_zp_color_check
"""
from __future__ import annotations

import argparse
import re
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
CATALOGS = ("GAIAXP", "GAIAXP_CORR_LAMOST")
CAT_COLOR = {"GAIAXP": "tab:orange", "GAIAXP_CORR_LAMOST": "tab:blue"}
CAT_LABEL = {"GAIAXP": "GAIAXP v1 (original XP)", "GAIAXP_CORR_LAMOST": "GAIAXP_CORR_LAMOST (LAMOST-corrected XP)"}
UNITS = [f"7DT{i:02d}" for i in range(1, 17)]
STANDARD_GRID = {f"m{w}" for w in range(400, 876, 25)}
MAGERR_FLOOR = 0.01
BPRP_EDGES = np.array([-0.5, 0.5, 0.8, 1.1, 1.5, 3.0])
AIRMASS_EDGES = np.array([1.0, 1.2, 1.5, 2.0, 3.5])
RADIUS_EDGES = np.arange(0, 50, 5.0)
FLAG_SLOPE = 0.02        # mag per mag of colour
FLAG_CALSPEC = 0.03      # mag
ZPERR_ANOMALY = 0.08     # header ZPERR_APER_3 above this = calibration itself failed (normal is 0.01-0.03)
CAL_MAX_MAGERR = 0.05    # CALSPEC detections fainter than this are noise-dominated
CAL_MAX_FLUX = 45000.0   # ADU; brighter peaks are in the non-linear/saturated regime
MAGERR_EDGES = np.array([0.0, 0.005, 0.01, 0.02, 0.03, 0.05])


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def mad(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan


def robust(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = len(x)
    return (np.median(x) if n else np.nan, mad(x), n, (mad(x) / np.sqrt(n) if n else np.nan))


def weighted_linfit(x, y, sigma, nsig=3.0, maxiter=5):
    x = np.asarray(x, float); y = np.asarray(y, float); sigma = np.asarray(sigma, float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(sigma) & (sigma > 0)
    a = b = a_err = np.nan
    for _ in range(maxiter):
        if m.sum() < 5 or np.ptp(x[m]) == 0:
            return np.nan, np.nan, np.nan, int(m.sum())
        (a, b), cov = np.polyfit(x[m], y[m], 1, w=1.0 / sigma[m], cov=True)
        a_err = float(np.sqrt(cov[0, 0]))
        res = y - (a * x + b); s = max(np.std(res[m]), 1e-4)
        new = m & (np.abs(res) < nsig * s)
        if new.sum() == m.sum(): break
        m = new
    return float(a), a_err, float(b), int(m.sum())


def filter_nm(name):
    m = re.findall(r"\d+", str(name)); return int(m[0]) if m else None


def binned_median(x, y, edges, min_n=20):
    x = np.asarray(x, float); y = np.asarray(y, float)
    cen, med, err = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi) & np.isfinite(y)
        if m.sum() >= min_n:
            cen.append(0.5 * (lo + hi)); med.append(np.median(y[m])); err.append(mad(y[m]) / np.sqrt(m.sum()))
    return np.array(cen), np.array(med), np.array(err)


class Data:
    def __init__(self, out_dir: Path):
        self.out = out_dir; self.fig = out_dir / "figures"; self.fig.mkdir(exist_ok=True)
        rd = lambda n, **k: pd.read_csv(out_dir / n, low_memory=False, **k) if (out_dir / n).exists() else None
        self.frames = rd("frames.csv")
        self.ct = rd("colorterms_header.csv")
        self.refit = rd("refit_frames.csv")
        self.stars = rd("refit_stars.csv.gz")
        self.cal = rd("calspec_frames.csv")
        self.syn = rd("calspec_synphot.csv")
        curves = rd("curves_comparison.csv")
        self.pivot = {}
        if curves is not None:
            c = curves[curves.source == "config"]
            self.pivot = dict(zip(c["filter"], c["pivot_AA"] / 10.0))
        if self.refit is not None:
            self.refit["proxy"] = self.refit["proxy"].fillna("")
        if self.frames is not None:
            self.frames["unit"] = self.frames["TELESCOP"]
        # Frames whose calibration itself failed (ZP scatter far above normal) are excluded from every
        # statistic and listed separately in summary.md.
        bad = self.frames[self.frames["ZPERR_APER_3"] > ZPERR_ANOMALY]
        self.anomalies = bad.groupby(["TELESCOP", "FILTER"]).agg(
            n=("file", "size"), mjd_min=("MJD", "min"), mjd_max=("MJD", "max"), zperr_median=("ZPERR_APER_3", "median"),
            objects=("OBJECT", lambda s: ",".join(sorted(set(s))[:8]))).reset_index()
        self.bad_files = set(bad["file"])
        self.frames = self.frames[~self.frames["file"].isin(self.bad_files)]
        for attr in ("ct", "refit", "stars", "cal"):
            d = getattr(self, attr)
            if d is not None:
                setattr(self, attr, d[~d["file"].isin(self.bad_files)])
        # filters present, ordered by pivot wavelength
        fl = set()
        for d, col in ((self.frames, "FILTER"), (self.refit, "filter"), (self.cal, "filter")):
            if d is not None: fl |= set(d[col].dropna().unique())
        self.filters = sorted(fl, key=lambda f: self.pivot.get(f, filter_nm(f) or 0))
        self.xpiv = {f: self.pivot.get(f, filter_nm(f)) for f in self.filters}

    def piv(self, f): return self.xpiv.get(f, np.nan)


def style_filter(f):
    return dict(marker="o", mfc="none") if f not in STANDARD_GRID else dict(marker="o")


# --------------------------------------------------------------------------------------
# H: header colour terms
# --------------------------------------------------------------------------------------
def header_pair_table(D: Data, aper="APER_3"):
    """Per-frame header slopes for g-r and for the (filter - neighbour) pair used by refit."""
    ct = D.ct[D.ct.aper == aper]
    gr = ct[(ct.f1 == "g") & (ct.f2 == "r")][["file", "K"]].rename(columns={"K": "K_gr"})
    nbr = None
    if D.refit is not None:
        nb = D.refit[(D.refit.catalog == "GAIAXP_CORR_LAMOST") & (D.refit.aper == aper)][["file", "filter", "nbr_filter"]].drop_duplicates("file")
        m = ct.merge(nb, on="file", suffixes=("", "_r"))
        m = m[(m.f1 == m["filter_r"]) & (m.f2 == m["nbr_filter"])]
        nbr = m[["file", "K", "nbr_filter"]].rename(columns={"K": "K_nbr"})
    fr = D.frames[["file", "FILTER", "unit", "MJD", "AIRMASS", "SEEING", "obj_type", "n_refstar", f"ZPERR_{aper}", f"K_MAG_{aper}", "refcat_class"]]
    t = fr.merge(gr, on="file", how="left")
    if nbr is not None: t = t.merge(nbr, on="file", how="left")
    return t


def fig_H1(D: Data, T: pd.DataFrame):
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    for ax, col, lab in ((axes[0], "K_gr", "header K_COLOR_APER_3 (g-r)"),
                         (axes[1], "K_nbr", "header K_COLOR_APER_3 (filter - neighbour band)"),
                         (axes[2], "K_MAG_APER_3", "header K_MAG_APER_3 (mag per mag)")):
        if col not in T: continue
        for f in D.filters:
            d = T[T.FILTER == f]
            for ot, c, dx in (("standard", "tab:red", -2), ("tile", "tab:green", 2)):
                v = d[d.obj_type == ot][col].dropna()
                if len(v) == 0: continue
                ax.plot(np.full(len(v), D.piv(f) + dx), v, ".", color=c, alpha=.08, ms=3)
                med, md, n, sem = robust(v)
                ax.errorbar(D.piv(f) + dx, med, yerr=md, fmt="o", color=c, ms=4, capsize=2,
                            mfc="none" if f not in STANDARD_GRID else c)
        ax.axhline(0, color="k", lw=.6); ax.set_ylabel(lab); ax.set_ylim(-0.3, 0.3)
        ax.grid(alpha=.3)
    axes[0].plot([], [], "o", color="tab:red", label="standard-star fields (2024-08)")
    axes[0].plot([], [], "o", color="tab:green", label="survey tiles")
    axes[0].plot([], [], "o", color="k", mfc="none", label="hollow = non-25nm-grid filter")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title("Colour terms already stored in the headers (median ± MAD per filter, APER_3)")
    axes[2].set_xlabel("pivot wavelength [nm]")
    fig.tight_layout(); fig.savefig(D.fig / "H1_header_colorterms_vs_pivot.png", dpi=130); plt.close(fig)


def heatmap(ax, mat, cnt, filters, cols, title, vlim, fmt="{:.2f}"):
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vlim, vmax=vlim)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=90, fontsize=7)
    ax.set_yticks(range(len(filters))); ax.set_yticklabels(filters, fontsize=7)
    for i in range(len(filters)):
        for j in range(len(cols)):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{fmt.format(mat[i, j])}\n{int(cnt[i, j])}", ha="center", va="center", fontsize=5)
    ax.set_title(title, fontsize=9); return im


def fig_H2(D: Data, T: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(16, 11))
    for ax, col, lab in ((axes[0], "K_gr", "header K(g-r)"), (axes[1], "K_nbr", "header K(filter-neighbour)")):
        if col not in T: continue
        mat = np.full((len(D.filters), len(UNITS)), np.nan); cnt = np.zeros_like(mat)
        for i, f in enumerate(D.filters):
            for j, u in enumerate(UNITS):
                v = T[(T.FILTER == f) & (T.unit == u)][col].dropna()
                if len(v) >= 3: mat[i, j] = np.median(v); cnt[i, j] = len(v)
        im = heatmap(ax, mat, cnt, D.filters, UNITS, f"{lab}: median per filter x unit (n below)", 0.15)
    fig.colorbar(im, ax=axes, shrink=.5, label="slope [mag/mag]")
    fig.savefig(D.fig / "H2_header_colorterm_filter_x_unit.png", dpi=130); plt.close(fig)


def fig_H3(D: Data, T: pd.DataFrame):
    if "K_gr" not in T: return
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    cmap = plt.get_cmap("viridis"); pv = np.array([D.piv(f) for f in T.FILTER]); norm = plt.Normalize(380, 900)
    for ax, x, lab in zip(axes.ravel(), ("MJD", "AIRMASS", "ZPERR_APER_3", "n_refstar", "SEEING", "K_MAG_APER_3"),
                          ("MJD", "airmass", "ZPERR_APER_3", "N reference stars", "seeing [arcsec]", "K_MAG_APER_3")):
        ax.scatter(T[x], T["K_gr"], c=pv, cmap=cmap, norm=norm, s=4, alpha=.4)
        ax.axhline(0, color="k", lw=.5); ax.set_xlabel(lab); ax.set_ylabel("header K(g-r)"); ax.set_ylim(-0.3, 0.3); ax.grid(alpha=.3)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, shrink=.6, label="pivot wavelength [nm]")
    fig.suptitle("Header colour term (g-r, APER_3) against frame properties")
    fig.savefig(D.fig / "H3_header_colorterm_vs_frame_properties.png", dpi=130); plt.close(fig)


# --------------------------------------------------------------------------------------
# R: refit
# --------------------------------------------------------------------------------------
def pooled_slopes(D: Data, group_cols=("filter", "catalog"), min_n=100):
    """Weighted robust slope of per-star ZP residual vs bp-rp, pooled over frames."""
    rows = []
    if D.stars is None: return pd.DataFrame(rows)
    st = D.stars
    sig = np.sqrt(st["magerr"].to_numpy(float) ** 2 + MAGERR_FLOOR ** 2)
    st = st.assign(sig=sig)
    for keys, g in st.groupby(list(group_cols)):
        if len(g) < min_n: continue
        a, ae, b, n = weighted_linfit(g["bprp"], g["resid"], g["sig"])
        rec = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        rec.update(a_pooled=a, a_pooled_err=ae, n_stars=n, n_frames=g["file"].nunique())
        if len(g) >= 3 * min_n and np.isfinite(a):
            good = np.isfinite(g["bprp"]) & np.isfinite(g["resid"])
            rec["a2_pooled"] = float(np.polyfit(g["bprp"][good], g["resid"][good], 2)[0])
        rows.append(rec)
    return pd.DataFrame(rows)


def fig_R1(D: Data, P: pd.DataFrame):
    R = D.refit[(D.refit.status == "ok") & (D.refit.aper == "APER_3")]
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for cat in CATALOGS:
        dx = -3 if cat == "GAIAXP" else 3
        for f in D.filters:
            d = R[(R.catalog == cat) & (R["filter"] == f)]
            if len(d) == 0: continue
            proxy = (d["proxy"] != "").any()
            med, md, n, sem = robust(d["a_bprp"])
            axes[0].errorbar(D.piv(f) + dx, med, yerr=md, fmt="o" if not proxy else "s", color=CAT_COLOR[cat], ms=4, capsize=2,
                             mfc=CAT_COLOR[cat] if f in STANDARD_GRID else "none", alpha=.9)
            pp = P[(P.catalog == cat) & (P["filter"] == f)] if len(P) else P
            if len(pp):
                axes[1].errorbar(D.piv(f) + dx, pp["a_pooled"].iloc[0], yerr=pp["a_pooled_err"].iloc[0], fmt="o" if not proxy else "s",
                                 color=CAT_COLOR[cat], ms=4, capsize=2, mfc=CAT_COLOR[cat] if f in STANDARD_GRID else "none")
        axes[0].plot([], [], "o", color=CAT_COLOR[cat], label=CAT_LABEL[cat])
    axes[0].plot([], [], "s", color="k", mfc="none", label="square = proxy band in GAIAXP v1")
    for ax in axes:
        ax.axhline(0, color="k", lw=.6); ax.axhspan(-FLAG_SLOPE, FLAG_SLOPE, color="grey", alpha=.12); ax.grid(alpha=.3)
    axes[0].set_ylabel("per-frame slope dZP/d(BP-RP)\nmedian ± MAD over frames"); axes[0].set_ylim(-0.2, 0.2)
    axes[1].set_ylabel("pooled slope dZP/d(BP-RP)\nall stars of all frames (± fit error)"); axes[1].set_ylim(-0.2, 0.2)
    axes[1].set_xlabel("pivot wavelength [nm]"); axes[0].legend(fontsize=8, loc="upper right")
    axes[0].set_title("Refit colour term of the zero point against Gaia BP-RP (APER_3, refcat stars, weighted robust fit)")
    fig.tight_layout(); fig.savefig(D.fig / "R1_refit_colorterm_bprp_vs_pivot.png", dpi=130); plt.close(fig)


def fig_R2(D: Data):
    if D.stars is None: return
    st = D.stars
    n = len(D.filters); ncol = 6; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.6 * nrow), sharex=True, sharey=True)
    for ax, f in zip(axes.ravel(), D.filters):
        d = st[(st["filter"] == f) & (st.catalog == "GAIAXP_CORR_LAMOST")]
        if len(d) > 50:
            ax.hexbin(d["bprp"], d["resid"], gridsize=40, extent=(-0.5, 3.0, -0.15, 0.15), cmap="Blues", mincnt=1, linewidths=0)
        for cat in CATALOGS:
            d = st[(st["filter"] == f) & (st.catalog == cat)]
            c, m, e = binned_median(d["bprp"], d["resid"], BPRP_EDGES)
            if len(c): ax.errorbar(c, m, yerr=e, fmt="o-", color=CAT_COLOR[cat], ms=3, lw=1, capsize=2)
        ax.axhline(0, color="k", lw=.5); ax.set_title(f"{f}  ({D.piv(f):.0f} nm)", fontsize=8)
        ax.set_xlim(-0.5, 3.0); ax.set_ylim(-0.15, 0.15); ax.tick_params(labelsize=7)
    for ax in axes.ravel()[n:]: ax.axis("off")
    fig.supxlabel("Gaia BP-RP"); fig.supylabel("ZP residual (star - frame median) [mag]")
    fig.suptitle("Pooled per-star ZP residual vs colour: hexbin = CORR_LAMOST; binned medians orange = GAIAXP v1, blue = CORR_LAMOST", fontsize=10)
    fig.tight_layout(); fig.savefig(D.fig / "R2_pooled_residual_vs_bprp_per_filter.png", dpi=120); plt.close(fig)


def fig_R3(D: Data):
    R = D.refit[(D.refit.status == "ok") & (D.refit.aper == "APER_3")]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for cat in CATALOGS:
        dx = -3 if cat == "GAIAXP" else 3
        for f in D.filters:
            d = R[(R.catalog == cat) & (R["filter"] == f)]
            if len(d) == 0: continue
            ax.plot(D.piv(f) + dx, d["sigma_before"].median(), "o", color=CAT_COLOR[cat], mfc="none", ms=5)
            ax.plot(D.piv(f) + dx, d["sigma_after_bprp"].median(), "v", color=CAT_COLOR[cat], ms=5)
        ax.plot([], [], "o", color=CAT_COLOR[cat], mfc="none", label=f"{cat}: ZP scatter before colour term")
        ax.plot([], [], "v", color=CAT_COLOR[cat], label=f"{cat}: after removing BP-RP term")
    ax.set_xlabel("pivot wavelength [nm]"); ax.set_ylabel("median per-frame ZP scatter [mag]"); ax.set_ylim(0, 0.08)
    ax.grid(alpha=.3); ax.legend(fontsize=8); ax.set_title("How much of the ZP scatter is explained by a linear BP-RP term (APER_3)")
    fig.tight_layout(); fig.savefig(D.fig / "R3_zp_scatter_before_after.png", dpi=130); plt.close(fig)


def fig_R4(D: Data, T: pd.DataFrame):
    R = D.refit[(D.refit.status == "ok") & (D.refit.aper == "APER_3") & (D.refit.catalog == "GAIAXP_CORR_LAMOST")]
    m = R.merge(T[["file", "K_gr"] + (["K_nbr"] if "K_nbr" in T else [])], on="file")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, hk, rk, lab in ((axes[0], "K_gr", "a_gr_unw", "g-r"), (axes[1], "K_nbr", "a_nbr_unw", "filter - neighbour")):
        if hk not in m or rk not in m: ax.axis("off"); continue
        d = m[[hk, rk]].dropna()
        ax.plot(d[hk], d[rk], ".", ms=3, alpha=.4); lim = (-0.3, 0.3)
        ax.plot(lim, lim, "k-", lw=.7); ax.set_xlim(lim); ax.set_ylim(lim); ax.grid(alpha=.3)
        dd = d[rk] - d[hk]
        ax.set_title(f"{lab}: N={len(d)}, refit-header median {dd.median():+.4f}, MAD {mad(dd):.4f}", fontsize=9)
        ax.set_xlabel(f"header K_COLOR_APER_3 ({lab})"); ax.set_ylabel("refit unweighted slope, CORR_LAMOST, same stars")
    fig.suptitle("Consistency gate: header colour terms reproduced by the refit")
    fig.tight_layout(); fig.savefig(D.fig / "R4_header_vs_refit_consistency.png", dpi=130); plt.close(fig)


def fig_R5(D: Data):
    if D.stars is None: return
    PU = pooled_slopes(D, ("filter", "catalog", "telescop"), min_n=200)
    fig, axes = plt.subplots(1, 2, figsize=(16, 11))
    for ax, cat in zip(axes, CATALOGS):
        mat = np.full((len(D.filters), len(UNITS)), np.nan); cnt = np.zeros_like(mat)
        for i, f in enumerate(D.filters):
            for j, u in enumerate(UNITS):
                d = PU[(PU["filter"] == f) & (PU.telescop == u) & (PU.catalog == cat)]
                if len(d): mat[i, j] = d["a_pooled"].iloc[0]; cnt[i, j] = d["n_frames"].iloc[0]
        im = heatmap(ax, mat, cnt, D.filters, UNITS, f"{cat}: pooled dZP/d(BP-RP) per filter x unit (n frames below)", 0.15)
    fig.colorbar(im, ax=axes, shrink=.5, label="slope [mag/mag]")
    fig.savefig(D.fig / "R5_pooled_colorterm_filter_x_unit.png", dpi=130); plt.close(fig)
    return PU


def fig_R6(D: Data):
    if D.stars is None: return pd.DataFrame()
    st = D.stars[D.stars.catalog == "GAIAXP_CORR_LAMOST"].merge(
        D.frames[["file", "AIRMASS"]], on="file", how="left")
    st = st.assign(sig=np.sqrt(st["magerr"] ** 2 + MAGERR_FLOOR ** 2))
    cmap = plt.get_cmap("viridis"); norm = plt.Normalize(380, 900)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    rows = []
    # (a) pooled slope in airmass bins
    for f in D.filters:
        d = st[st["filter"] == f]; xs, ys, es = [], [], []
        for lo, hi in zip(AIRMASS_EDGES[:-1], AIRMASS_EDGES[1:]):
            g = d[(d.AIRMASS >= lo) & (d.AIRMASS < hi)]
            if len(g) < 200: continue
            a, ae, _, n = weighted_linfit(g["bprp"], g["resid"], g["sig"])
            if np.isfinite(a): xs.append(0.5 * (lo + hi)); ys.append(a); es.append(ae)
            rows.append(dict(filter=f, airmass_lo=lo, airmass_hi=hi, a_bprp=a, a_bprp_err=ae, n_stars=n))
        if xs: axes[0].errorbar(xs, ys, yerr=es, fmt="o-", color=cmap(norm(D.piv(f))), ms=3, lw=1, alpha=.8, capsize=2)
    axes[0].axhline(0, color="k", lw=.5); axes[0].set_xlabel("airmass (bin centre)"); axes[0].set_ylabel("pooled dZP/d(BP-RP)")
    axes[0].set_title("colour term vs airmass (CORR_LAMOST)", fontsize=9); axes[0].set_ylim(-0.2, 0.2); axes[0].grid(alpha=.3)
    # (b) residual vs field radius
    for f in D.filters:
        d = st[st["filter"] == f]
        c, m, e = binned_median(d["r_arcmin"], d["resid"], RADIUS_EDGES, min_n=50)
        if len(c): axes[1].plot(c, m, "-", color=cmap(norm(D.piv(f))), lw=1, alpha=.8)
    axes[1].axhline(0, color="k", lw=.5); axes[1].set_xlabel("distance from image centre [arcmin]"); axes[1].set_ylabel("median ZP residual [mag]")
    axes[1].set_title("ZP residual vs field position (CORR_LAMOST)", fontsize=9); axes[1].set_ylim(-0.05, 0.05); axes[1].grid(alpha=.3)
    # (c) colour term inner vs outer half of the field
    R = D.refit[(D.refit.status == "ok") & (D.refit.aper == "APER_3") & (D.refit.catalog == "GAIAXP_CORR_LAMOST")]
    for f in D.filters:
        d = R[R["filter"] == f]
        if len(d) < 3: continue
        axes[2].errorbar(D.piv(f) - 2, d["a_bprp_inner"].median(), yerr=mad(d["a_bprp_inner"]) / np.sqrt(len(d)), fmt="o", color="tab:purple", ms=4, capsize=2)
        axes[2].errorbar(D.piv(f) + 2, d["a_bprp_outer"].median(), yerr=mad(d["a_bprp_outer"]) / np.sqrt(len(d)), fmt="^", color="tab:brown", ms=4, capsize=2)
    axes[2].plot([], [], "o", color="tab:purple", label="inner half (r < median)"); axes[2].plot([], [], "^", color="tab:brown", label="outer half")
    axes[2].axhline(0, color="k", lw=.5); axes[2].set_xlabel("pivot wavelength [nm]"); axes[2].set_ylabel("median per-frame dZP/d(BP-RP)")
    axes[2].set_title("colour term: inner vs outer field (CORR_LAMOST)", fontsize=9); axes[2].set_ylim(-0.2, 0.2); axes[2].legend(fontsize=8); axes[2].grid(alpha=.3)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes[:2], shrink=.7, label="pivot wavelength [nm]")
    fig.savefig(D.fig / "R6_airmass_and_field_position.png", dpi=130); plt.close(fig)
    return pd.DataFrame(rows)


def fig_R7(D: Data):
    """Catalogue-only comparison on the same stars: CORR_LAMOST - GAIAXP v1 vs BP-RP (direct bands only)."""
    if D.stars is None: return pd.DataFrame()
    R = D.refit[(D.refit.aper == "APER_3") & (D.refit.status == "ok")]
    direct = set(R[(R.catalog == "GAIAXP") & (R["proxy"] == "")]["filter"])
    a = D.stars[D.stars.catalog == "GAIAXP"][["file", "ref_id", "filter", "bprp", "ref_mag"]]
    b = D.stars[D.stars.catalog == "GAIAXP_CORR_LAMOST"][["file", "ref_id", "ref_mag"]]
    m = a.merge(b, on=["file", "ref_id"], suffixes=("_orig", "_corr"))
    m = m[m["filter"].isin(direct)].drop_duplicates(["ref_id", "filter"])
    m["dmag"] = m["ref_mag_corr"] - m["ref_mag_orig"]
    rows = []
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for f in D.filters:
        d = m[m["filter"] == f]
        if len(d) < 30: continue
        med, md, n, sem = robust(d["dmag"])
        a_, ae, _, _ = weighted_linfit(d["bprp"], d["dmag"], np.full(len(d), 0.01))
        rows.append(dict(filter=f, n_stars=n, dmag_median=med, dmag_mad=md, dmag_slope_bprp=a_, dmag_slope_err=ae))
        axes[0].errorbar(D.piv(f), med, yerr=md, fmt="o", color="tab:blue", ms=4, capsize=2, mfc="tab:blue" if f in STANDARD_GRID else "none")
        axes[1].errorbar(D.piv(f), a_, yerr=ae, fmt="o", color="tab:blue", ms=4, capsize=2, mfc="tab:blue" if f in STANDARD_GRID else "none")
    for ax in axes: ax.axhline(0, color="k", lw=.6); ax.grid(alpha=.3)
    axes[0].set_ylabel("median (CORR_LAMOST - GAIAXP v1) [mag]\n± MAD over stars"); axes[1].set_ylabel("d(CORR - GAIAXP)/d(BP-RP) [mag/mag]")
    axes[1].set_xlabel("pivot wavelength [nm]"); axes[0].set_title("Catalogue-only difference on the same stars (no image involved)")
    fig.tight_layout(); fig.savefig(D.fig / "R7_catalog_difference_vs_bprp.png", dpi=130); plt.close(fig)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# C: CALSPEC
# --------------------------------------------------------------------------------------
def calspec_clean(D: Data):
    c = D.cal[(D.cal.status == "ok")].copy()
    c = c[(c["flags"].fillna(0) <= 1) & (~c["saturated"].astype(bool)) & (c["sep_arcsec"] < 1.5)
          & (c["magerr_APER_3"] < CAL_MAX_MAGERR) & (c["flux_max"] < CAL_MAX_FLUX)]
    if D.syn is not None:
        gr = D.syn.pivot(index="object", columns="filter", values="mag_syn")
        c["star_gr_syn"] = c["object"].map((gr["g"] - gr["r"]).to_dict())
    return c


def fig_C1(D: Data, C: pd.DataFrame):
    stars = sorted(C["object"].unique(), key=lambda o: C[C.object == o]["star_gr_syn"].iloc[0])
    cmap = plt.get_cmap("coolwarm"); grs = {o: C[C.object == o]["star_gr_syn"].iloc[0] for o in stars}
    norm = plt.Normalize(min(grs.values()), max(grs.values()))
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for ax, ap in zip(axes, ("APER_3", "AUTO")):
        col = f"delta_{ap}"
        for f in D.filters:
            d = C[C["filter"] == f]
            allmed = robust(d[col])[0]
            ax.plot(D.piv(f), allmed, "_", color="k", ms=14, mew=2)
            for k, o in enumerate(stars):
                v = d[d.object == o][col]
                if len(v) < 2: continue
                med, md, n, sem = robust(v)
                ax.errorbar(D.piv(f) + (k - len(stars) / 2) * 0.9, med, yerr=sem, fmt="o", color=cmap(norm(grs[o])), ms=3.5, capsize=1.5, alpha=.9)
        ax.axhline(0, color="k", lw=.6); ax.axhspan(-FLAG_CALSPEC, FLAG_CALSPEC, color="grey", alpha=.12); ax.grid(alpha=.3)
        ax.set_ylabel(f"m_obs({ap}) - m_syn(CALSPEC) [mag]"); ax.set_ylim(-0.15, 0.15)
    for o in stars: axes[0].plot([], [], "o", color=cmap(norm(grs[o])), label=f"{o} (g-r={grs[o]:+.2f})")
    axes[0].legend(fontsize=7, ncol=3, loc="upper right"); axes[1].set_xlabel("pivot wavelength [nm]")
    axes[0].set_title("CALSPEC standards: observed calibrated magnitude minus synthetic magnitude (median ± MAD/√n per star; black = all stars)")
    fig.tight_layout(); fig.savefig(D.fig / "C1_calspec_residual_vs_pivot.png", dpi=130); plt.close(fig)


def fig_C2(D: Data, C: pd.DataFrame):
    """Slope of the CALSPEC residual across stars vs their synthetic g-r, compared with the refit a_gr."""
    R = D.refit[(D.refit.status == "ok") & (D.refit.aper == "APER_3") & (D.refit.catalog == "GAIAXP_CORR_LAMOST")]
    rows = []
    fig, ax = plt.subplots(figsize=(12, 5))
    for f in D.filters:
        d = C[C["filter"] == f]
        per_star = d.groupby("object").agg(delta=("delta_APER_3", "median"), n=("delta_APER_3", "size"), gr=("star_gr_syn", "first"))
        per_star = per_star[per_star.n >= 2]
        if len(per_star) >= 4 and np.ptp(per_star.gr) > 0.3:
            a, ae, b, n = weighted_linfit(per_star.gr, per_star.delta, np.full(len(per_star), 0.01))
            rows.append(dict(filter=f, n_stars=len(per_star), calspec_slope_gr=a, calspec_slope_gr_err=ae,
                             calspec_between_star_mad=mad(per_star.delta), calspec_delta_median=np.median(per_star.delta)))
            ax.errorbar(D.piv(f) - 2, a, yerr=ae, fmt="D", color="tab:red", ms=4, capsize=2, mfc="tab:red" if f in STANDARD_GRID else "none")
        r = R[R["filter"] == f]
        if len(r): ax.errorbar(D.piv(f) + 2, r["a_gr"].median(), yerr=mad(r["a_gr"]) / np.sqrt(len(r)), fmt="o", color="tab:blue", ms=4, capsize=2,
                               mfc="tab:blue" if f in STANDARD_GRID else "none")
    ax.plot([], [], "D", color="tab:red", label="CALSPEC: d(m_obs - m_syn)/d(g-r) across the 13 standards")
    ax.plot([], [], "o", color="tab:blue", label="refit vs CORR_LAMOST: dZP/d(g-r) of field stars (median over frames)")
    ax.axhline(0, color="k", lw=.6); ax.axhspan(-FLAG_SLOPE, FLAG_SLOPE, color="grey", alpha=.12); ax.grid(alpha=.3)
    ax.set_ylim(-0.2, 0.2); ax.set_xlabel("pivot wavelength [nm]"); ax.set_ylabel("slope [mag per mag of (g-r)]"); ax.legend(fontsize=8)
    ax.set_title("Colour dependence from two independent routes: CALSPEC standards vs GaiaXP field stars")
    fig.tight_layout(); fig.savefig(D.fig / "C2_calspec_slope_vs_refit.png", dpi=130); plt.close(fig)
    return pd.DataFrame(rows)


def fig_C3(D: Data, C: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    mat = np.full((len(D.filters), len(UNITS)), np.nan); cnt = np.zeros_like(mat)
    for i, f in enumerate(D.filters):
        for j, u in enumerate(UNITS):
            v = C[(C["filter"] == f) & (C.telescop == u)]["delta_APER_3"]
            if len(v) >= 3: mat[i, j] = np.median(v); cnt[i, j] = len(v)
    im = heatmap(axes[0, 0], mat, cnt, D.filters, UNITS, "CALSPEC residual (APER_3): median per filter x unit", 0.08, "{:+.3f}")
    fig.colorbar(im, ax=axes[0, 0], shrink=.7)
    cmap = plt.get_cmap("viridis"); norm = plt.Normalize(380, 900)
    for ax, x, edges, lab in ((axes[0, 1], "airmass", AIRMASS_EDGES, "airmass"),
                              (axes[1, 0], "mjd", None, "MJD"),
                              (axes[1, 1], "magerr_APER_3", MAGERR_EDGES, "MAGERR_APER_3 of the standard (S/N proxy)")):
        for f in D.filters:
            d = C[C["filter"] == f]
            if edges is None:
                g = d.groupby(np.floor(d["mjd"]))["delta_APER_3"].agg(["median", "size"]); g = g[g["size"] >= 3]
                ax.plot(g.index + 0.5, g["median"], "o-", color=cmap(norm(D.piv(f))), ms=3, lw=.8, alpha=.8)
            else:
                c, m, e = binned_median(d[x], d["delta_APER_3"], edges, min_n=5)
                if len(c): ax.plot(c, m, "o-", color=cmap(norm(D.piv(f))), ms=3, lw=.8, alpha=.8)
        ax.axhline(0, color="k", lw=.5); ax.set_xlabel(lab); ax.set_ylabel("median CALSPEC residual [mag]"); ax.set_ylim(-0.1, 0.1); ax.grid(alpha=.3)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=[axes[0, 1], axes[1, 0], axes[1, 1]], shrink=.5, label="pivot wavelength [nm]")
    fig.suptitle("CALSPEC residual against unit telescope, airmass, night and photometric error")
    fig.savefig(D.fig / "C3_calspec_residual_dependencies.png", dpi=130); plt.close(fig)


# --------------------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------------------
def build_summary(D: Data, T, P, PA, R7, C, C2):
    rows = []
    R = D.refit[(D.refit.status == "ok") & (D.refit.aper == "APER_3")] if D.refit is not None else None
    for f in D.filters:
        rec = dict(filter=f, pivot_nm=D.piv(f), grid="25nm" if f in STANDARD_GRID else "new")
        fr = D.frames[D.frames.FILTER == f]
        rec.update(n_frames=len(fr), n_frames_zp=int(fr["ZP_APER_3"].notna().sum()),
                   n_standard=int((fr.obj_type == "standard").sum()), n_tile=int((fr.obj_type == "tile").sum()))
        t = T[T.FILTER == f]
        for col, name in (("K_gr", "hdr_K_gr"), ("K_nbr", "hdr_K_nbr"), ("K_MAG_APER_3", "hdr_K_mag")):
            if col in t:
                med, md, n, sem = robust(t[col]); rec[f"{name}_median"] = med; rec[f"{name}_mad"] = md
        if "nbr_filter" in t: rec["nbr_filter"] = t["nbr_filter"].dropna().iloc[0] if t["nbr_filter"].notna().any() else None
        if R is not None:
            for cat, tag in (("GAIAXP", "orig"), ("GAIAXP_CORR_LAMOST", "corr")):
                d = R[(R.catalog == cat) & (R["filter"] == f)]
                rec[f"{tag}_n_frames"] = len(d)
                if len(d) == 0: continue
                rec[f"{tag}_proxy"] = d["proxy"].iloc[0]
                for k in ("a_bprp", "a_gr", "a_nbr", "a2_bprp", "sigma_before", "sigma_after_bprp", "a_radius", "a_bprp_inner", "a_bprp_outer"):
                    rec[f"{tag}_{k}_median"] = robust(d[k])[0]
                rec[f"{tag}_a_bprp_mad"] = mad(d["a_bprp"])
                rec[f"{tag}_a_bprp_sem"] = robust(d["a_bprp"])[3]
                rec[f"{tag}_zp_minus_header_median"] = robust(d["zp_median"] - d["zp_header"])[0]
                pp = P[(P.catalog == cat) & (P["filter"] == f)] if len(P) else P
                if len(pp):
                    rec[f"{tag}_a_bprp_pooled"] = pp["a_pooled"].iloc[0]; rec[f"{tag}_a_bprp_pooled_err"] = pp["a_pooled_err"].iloc[0]
                    rec[f"{tag}_n_stars"] = pp["n_stars"].iloc[0]
                    if "a2_pooled" in pp: rec[f"{tag}_a2_bprp_pooled"] = pp["a2_pooled"].iloc[0]
        if len(PA):
            pa = PA[PA["filter"] == f].dropna(subset=["a_bprp"])
            if len(pa) >= 2:
                rec["airmass_slope_range"] = float(pa["a_bprp"].max() - pa["a_bprp"].min())
        if len(R7):
            r7 = R7[R7["filter"] == f]
            if len(r7): rec["cat_diff_median"] = r7["dmag_median"].iloc[0]; rec["cat_diff_slope_bprp"] = r7["dmag_slope_bprp"].iloc[0]
        if C is not None:
            c = C[C["filter"] == f]
            rec["calspec_n_frames"] = len(c); rec["calspec_n_stars"] = c["object"].nunique()
            if len(c):
                rec["calspec_delta_median"] = robust(c["delta_APER_3"])[0]
                per_star = c.groupby("object")["delta_APER_3"].median()
                rec["calspec_between_star_mad"] = mad(per_star) if len(per_star) >= 3 else np.nan
                rec["calspec_between_star_range"] = float(per_star.max() - per_star.min()) if len(per_star) >= 2 else np.nan
            if len(C2):
                c2 = C2[C2["filter"] == f]
                if len(c2): rec["calspec_slope_gr"] = c2["calspec_slope_gr"].iloc[0]; rec["calspec_slope_gr_err"] = c2["calspec_slope_gr_err"].iloc[0]
            # what the CALSPEC offset would have been with the other catalogue (ZP shifts by the catalogue difference)
            if np.isfinite(rec.get("calspec_delta_median", np.nan)) and np.isfinite(rec.get("cat_diff_median", np.nan)):
                rec["calspec_delta_if_orig"] = rec["calspec_delta_median"] - rec["cat_diff_median"]
        # verdict flags
        a = rec.get("corr_a_bprp_pooled", np.nan); ae = rec.get("corr_a_bprp_pooled_err", np.nan); am = rec.get("corr_a_bprp_median", np.nan)
        rec["flag_color_corr"] = bool(np.isfinite(a) and abs(a) > FLAG_SLOPE and abs(a) > 3 * ae and np.sign(a) == np.sign(am))
        a = rec.get("orig_a_bprp_pooled", np.nan); ae = rec.get("orig_a_bprp_pooled_err", np.nan); am = rec.get("orig_a_bprp_median", np.nan)
        rec["flag_color_orig"] = bool(np.isfinite(a) and abs(a) > FLAG_SLOPE and abs(a) > 3 * ae and np.sign(a) == np.sign(am))
        rec["flag_calspec"] = bool((rec.get("calspec_between_star_mad", 0) or 0) > FLAG_CALSPEC
                                   or abs(rec.get("calspec_slope_gr", 0) or 0) > FLAG_SLOPE)
        rows.append(rec)
    return pd.DataFrame(rows)


def write_markdown(D: Data, S: pd.DataFrame, T: pd.DataFrame, C):
    L = []
    L.append("# 7DT 중대역 영점 색 의존성 검사 — 자동 요약\n")
    L.append(f"생성: {time.strftime('%Y-%m-%d %H:%M')}  |  산출물: `{D.out}`\n")
    fr = D.frames
    L.append("## 표본\n")
    L.append(f"- 단일 프레임 {len(fr)}장 (ZP_APER_3 있음 {int(fr['ZP_APER_3'].notna().sum())}, .refcat 있음 {int(fr['has_refcat'].sum())}); "
             f"표준성 필드 {int((fr.obj_type == 'standard').sum())}, 서베이 타일 {int((fr.obj_type == 'tile').sum())}, 기타 {int((fr.obj_type == 'other').sum())}")
    L.append(f"- 헤더 색 쌍으로 추정한 참조 카탈로그: {fr['refcat_class'].value_counts().to_dict()}")
    L.append(f"- 헤더 ZPERR_APER_3 > {ZPERR_ANOMALY} 인 이상 프레임 {len(D.bad_files)}장은 모든 통계에서 제외 (아래 표)")
    if len(D.anomalies):
        L.append(f"\n## 제외된 이상 프레임 (유닛 × 필터)\n")
        L.append("| unit | filter | N | MJD 범위 | ZPERR 중앙값 | 대상 |"); L.append("|---|---|---|---|---|---|")
        for _, r in D.anomalies.iterrows():
            L.append(f"| {r.TELESCOP} | {r.FILTER} | {int(r.n)} | {r.mjd_min:.1f}–{r.mjd_max:.1f} | {r.zperr_median:.3f} | {r.objects} |")
    if D.refit is not None:
        R = D.refit[(D.refit.status == "ok") & (D.refit.aper == "APER_3")]
        L.append("\n## 게이트: 재적합이 헤더를 재현하는가\n")
        for cat in CATALOGS:
            d = R[R.catalog == cat]
            if len(d) == 0: continue
            dz = (d["zp_median"] - d["zp_header"]).abs()
            line = f"- {cat}: {len(d)} 프레임, |ZP_refit − ZP_header| 중앙값 {dz.median():.4f} mag, 0.002 이내 비율 {(dz < 0.002).mean():.2f}"
            if "K_gr" in T:
                m = d.merge(T[["file", "K_gr"]], on="file"); dk = m["a_gr_unw"] - m["K_gr"]
                line += f"; g−r 기울기 차이 중앙값 {dk.median():+.4f}, MAD {mad(dk):.4f}"
            L.append(line)
        L.append(f"- 상태 집계: {D.refit['status'].value_counts().to_dict()}")
    L.append("\n## 필터별 요약 (APER_3)\n")
    cols = [("filter", "filter"), ("pivot_nm", "pivot"), ("n_frames_zp", "N"), ("hdr_K_gr_median", "hdr K(g-r)"), ("hdr_K_nbr_median", "hdr K(nbr)"),
            ("orig_a_bprp_pooled", "a_bprp GAIAXP"), ("orig_a_bprp_pooled_err", "±"), ("corr_a_bprp_pooled", "a_bprp CORR"), ("corr_a_bprp_pooled_err", "±"),
            ("corr_a_bprp_median", "a_bprp CORR (frame med)"), ("corr_a2_bprp_pooled", "a2 CORR"), ("corr_sigma_before_median", "σ before"), ("corr_sigma_after_bprp_median", "σ after"),
            ("cat_diff_median", "CORR−orig ZP"), ("cat_diff_slope_bprp", "d(CORR−orig)/d(bprp)"), ("calspec_delta_median", "CALSPEC Δ"), ("calspec_delta_if_orig", "CALSPEC Δ if GAIAXP v1"),
            ("calspec_between_star_mad", "CALSPEC star MAD"), ("calspec_slope_gr", "CALSPEC d/d(g-r)"),
            ("airmass_slope_range", "a_bprp airmass range"), ("flag_color_corr", "flag colour"), ("flag_calspec", "flag CALSPEC")]
    cols = [(c, l) for c, l in cols if c in S]
    L.append("| " + " | ".join(l for _, l in cols) + " |"); L.append("|" + "---|" * len(cols))
    for _, r in S.iterrows():
        cells = []
        for c, _ in cols:
            v = r[c]
            if isinstance(v, (bool, np.bool_)): cells.append("**Y**" if v else "")
            elif isinstance(v, str): cells.append(v)
            elif pd.isna(v): cells.append("")
            elif c in ("pivot_nm",): cells.append(f"{v:.0f}")
            elif c in ("n_frames_zp",): cells.append(f"{int(v)}")
            elif c.endswith("_err") or "sigma" in c or "mad" in c or "range" in c: cells.append(f"{v:.3f}")
            else: cells.append(f"{v:+.3f}")
        L.append("| " + " | ".join(cells) + " |")
    L.append("\n주: `a_bprp` = dZP/d(BP−RP) [mag/mag], 가중·3σ 반복 적합. 'pooled'는 모든 프레임의 별을 합친 적합(± 적합 오차), "
             "'frame med'는 프레임별 기울기의 중앙값. GAIAXP v1에 없는 필터는 인접 옛 밴드 proxy 사용(요약 CSV의 `orig_proxy`). "
             f"플래그 기준: |a| > {FLAG_SLOPE} 이고 3σ 이상, CALSPEC 별 간 MAD > {FLAG_CALSPEC} 또는 |d/d(g−r)| > {FLAG_SLOPE}.\n")
    flagged = S[S.flag_color_corr | S.flag_calspec]
    L.append("## 플래그된 필터\n")
    L.append(", ".join(flagged["filter"]) if len(flagged) else "없음")
    L.append("\n## 그림\n")
    for p in sorted(D.fig.glob("*.png")): L.append(f"- `figures/{p.name}`")
    (D.out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = ap.parse_args()
    D = Data(Path(args.out_dir))
    if D.frames is None: raise SystemExit("frames.csv missing")
    log(f"loaded: frames {len(D.frames)}, refit {0 if D.refit is None else len(D.refit)}, stars {0 if D.stars is None else len(D.stars)}, "
        f"calspec {0 if D.cal is None else len(D.cal)}; {len(D.filters)} filters")
    T = header_pair_table(D) if D.ct is not None else D.frames[["file", "FILTER", "unit", "obj_type"]]
    fig_H1(D, T); fig_H2(D, T); fig_H3(D, T); log("H figures done")
    P = PA = R7 = pd.DataFrame(); C = C2 = None
    if D.refit is not None:
        P = pooled_slopes(D); fig_R1(D, P); fig_R2(D); fig_R3(D); fig_R4(D, T); fig_R5(D); PA = fig_R6(D); R7 = fig_R7(D); log("R figures done")
    if D.cal is not None and D.syn is not None:
        C = calspec_clean(D)
        if len(C): fig_C1(D, C); C2 = fig_C2(D, C); fig_C3(D, C); log("C figures done")
    S = build_summary(D, T, P, PA, R7, C, C2 if C2 is not None else pd.DataFrame())
    S.to_csv(D.out / "summary_by_filter.csv", index=False)
    write_markdown(D, S, T, C)
    log(f"summary_by_filter.csv ({len(S)} filters) and summary.md written")


if __name__ == "__main__":
    main()
