#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260924_stepA_models.py — Step A, decisive test: which physical model explains the star-to-star pattern of the
standards' offsets?  Uses each standard's EXACT spectrum (CALSPEC / SPSS), never XP, for the model predictors:

  shift  : Delta_s = a + delta * J_s,        J_s = dm/d(delta) of the curve on the star's own spectrum [mag/nm]
  leak   : Delta_s = a - 2.5 log10(1 + L R_s), R_s = (star flux through a flat 650-1050 nm leak x QE x optics x atm)
                                               / (star flux through the band)   [red-leak hypothesis]
  counts : Delta_s = a + c * log10(peak_s)   (photometric count-level effect)
plus the null (a only) and pairwise combinations.  Star-level medians from stepA_frames.csv; weights from the
per-star frame scatter.  Output: stepA_models.csv (per dataset/filter: coefficients, BIC per model), A4 figure.
"""
import importlib
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.table import Table

warnings.filterwarnings("ignore")
sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
TC = importlib.import_module("260921_truthcompare")
AN, TW, R = TC.AN, TC.TW, TC.R
OUT = Path("/home/hhchoi1022/ezphot/log/20260924_stepA")
import sys as _sys
TAG = _sys.argv[1] if len(_sys.argv) > 1 else ""
TS = TC.TS; SRC = TC.SRC

AN._init(); TW._init(40)
curves = TC.build_totals()[("manufacturer", "nobeam")]
atm = pd.read_csv(TS / "atmosphere_elsauce.csv")
qe = pd.read_csv(SRC / "QE.csv")
opt = pd.read_csv(SRC / "optics.efficiency.dr350.dr500.csv", sep=r"\s+", comment="#", skiprows=11)
lw = np.arange(650.0, 1050.0, 1.0)
leak_t = (np.interp(lw, qe.iloc[:, 0], qe.iloc[:, 1]) * np.interp(lw, opt.iloc[:, 0], opt.iloc[:, -1])
          * np.interp(lw, atm.iloc[:, 0], atm["trans"] if "trans" in atm.columns else atm.iloc[:, 1]))

# exact spectra
spec = {}
for obj, st in TW._INFO.items():
    if st["spec"]:
        t = Table.read(TW.CALSPEC_DIR / st["spec"])
        wl = np.asarray(t["WAVELENGTH"], float) / 10.0; fl = np.asarray(t["FLUX"], float) * 1e-2
        m = (wl > 300) & (wl < 1100) & np.isfinite(fl) & (fl > 0)
        spec[("CALSPEC", obj)] = (wl[m], fl[m])
sp = pd.read_csv(AN.SPSS_CSV)
for _, s in sp.iterrows():
    d = AN.load_spss_spec(s.SPSS)
    if d is not None:
        spec[("SPSS", s.Name)] = d


def photon_flux(wl, fl, cw, ct):
    t = np.interp(wl, cw, ct, left=0.0, right=0.0)
    return np.trapezoid(fl * t * wl, wl)


F = pd.read_csv(OUT / f"stepA_frames{TAG}.csv").dropna(subset=["d_gnd"])
norm = {(k[0], AN.norm(k[1])): k for k in spec}
rows, pts = [], []
for (ds, filt), g in F.groupby(["dataset", "filter"]):
    if filt not in curves:
        continue
    fw, ft = curves[filt]
    per = g.groupby("object").agg(d=("d_gnd", "median"), sd=("d_gnd", lambda v: 1.4826 * np.median(np.abs(v - v.median()))),
                                  n=("d_gnd", "size"), peak=("star_fmax", "median"), c=("c_star", "first"))
    J, Rl = [], []
    for obj in per.index:
        key = (ds, obj) if (ds, obj) in spec else norm.get((ds, AN.norm(obj)))
        if key is None:
            J.append(np.nan); Rl.append(np.nan); continue
        wl, fl = spec[key]
        J.append(float(R.syn_mag(wl, fl, fw, ft, 0.5) - R.syn_mag(wl, fl, fw, ft, -0.5)))
        fb = photon_flux(wl, fl, fw, ft)
        m_ = (wl >= 650) & (wl <= 1050)
        fr = np.trapezoid(fl[m_] * np.interp(wl[m_], lw, leak_t) * wl[m_], wl[m_]) if m_.sum() > 10 else np.nan
        Rl.append(fr / fb if fb > 0 else np.nan)
    per["J"] = J; per["Rleak"] = Rl
    per = per.dropna(subset=["J", "Rleak", "peak"])
    per = per[per.n >= 2]
    if len(per) < 5:
        continue
    y = per.d.to_numpy()
    w = 1.0 / np.sqrt(per.sd.clip(lower=0.005).to_numpy() ** 2 / per.n.to_numpy() + 0.010 ** 2)   # 10 mmag floor per star
    X = {"J": per.J.to_numpy(), "logpeak": np.log10(per.peak.clip(lower=1).to_numpy())}

    def fit_lin(cols):
        A = np.column_stack([np.ones(len(y))] + [X[c] for c in cols]) * w[:, None]
        sol, *_ = np.linalg.lstsq(A, y * w, rcond=None)
        chi2 = float(np.sum((A @ sol - y * w) ** 2))
        cov = np.linalg.pinv(A.T @ A) * chi2 / max(len(y) - A.shape[1], 1)
        return sol, np.sqrt(np.diag(cov)), chi2, A.shape[1]

    def fit_leak(extra=None):
        best = None
        for L in np.concatenate([[0.0], np.logspace(-6, -1, 101)]):
            lk = -2.5 * np.log10(1 + L * per.Rleak.to_numpy())
            cols = [np.ones(len(y)), lk] if extra is None else [np.ones(len(y)), lk, X[extra]]
            A = np.column_stack(cols[:1] + cols[2:]) * w[:, None]
            yy = (y - lk) * w
            sol, *_ = np.linalg.lstsq(A, yy, rcond=None)
            chi2 = float(np.sum((A @ sol - yy) ** 2))
            if best is None or chi2 < best[1]:
                best = (L, chi2, sol)
        return best

    n = len(y)
    bic = lambda chi2, k: chi2 + k * np.log(n)
    res = dict(dataset=ds, filter=filt, pivot_nm=int(filt[1:4]), n_stars=n,
               J_min=round(per.J.min(), 4), J_max=round(per.J.max(), 4),
               R_min=float(per.Rleak.min()), R_max=float(per.Rleak.max()))
    s0, e0, c0, k0 = fit_lin([]); res["bic_null"] = bic(c0, k0)
    s1, e1, c1, k1 = fit_lin(["J"]); res.update(delta_nm=s1[1], delta_err=e1[1], bic_shift=bic(c1, k1))
    s2, e2, c2, k2 = fit_lin(["logpeak"]); res.update(c_peak=1e3 * s2[1], c_peak_err=1e3 * e2[1], bic_counts=bic(c2, k2))
    s3, e3, c3, k3 = fit_lin(["J", "logpeak"]); res.update(delta_nm_w_peak=s3[1], delta_err_w_peak=e3[1],
                                                           c_peak_w_J=1e3 * s3[2], c_peak_w_J_err=1e3 * e3[2], bic_shift_counts=bic(c3, k3))
    Lb, cl, _ = fit_leak(); res.update(leak_L=Lb, bic_leak=bic(cl, 2))
    Lb2, cl2, _ = fit_leak("J"); res.update(leak_L_w_J=Lb2, bic_leak_shift=bic(cl2, 3))
    models = {k: res[k] for k in res if k.startswith("bic_")}
    res["best"] = min(models, key=models.get).replace("bic_", "")
    res["dbic_best_vs_null"] = res["bic_null"] - models[min(models, key=models.get)]
    rows.append(res)
    for obj, r in per.iterrows():
        pts.append(dict(dataset=ds, filter=filt, object=obj, d=r.d, n=r.n, J=r.J, Rleak=r.Rleak, peak=r.peak, c=r.c))
M = pd.DataFrame(rows).sort_values(["dataset", "pivot_nm"])
P = pd.DataFrame(pts)
M.to_csv(OUT / f"stepA_models{TAG}.csv", index=False); P.to_csv(OUT / f"stepA_models_points{TAG}.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
show = M[["dataset", "filter", "n_stars", "delta_nm", "delta_err", "c_peak", "c_peak_err", "delta_nm_w_peak", "c_peak_w_J",
          "c_peak_w_J_err", "leak_L", "best", "dbic_best_vs_null", "bic_null", "bic_shift", "bic_counts", "bic_leak",
          "bic_shift_counts"]].copy()
print(show.round(3).to_string(index=False))

# ---- A4: m375w / m386 / m400 / m825 panels
fig, ax = plt.subplots(2, 4, figsize=(19, 8.5))
for j, (ds, filt) in enumerate([("SPSS", "m375w"), ("SPSS", "m386"), ("SPSS", "m400"), ("CALSPEC", "m825")]):
    p = P[(P.dataset == ds) & (P["filter"] == filt)]
    r = M[(M.dataset == ds) & (M["filter"] == filt)]
    if not len(p):
        continue
    r = r.iloc[0]
    sc = ax[0, j].scatter(p.J, 1e3 * p.d, c=p.c, cmap="coolwarm", s=40, edgecolor="k", vmin=-0.5, vmax=2.0)
    xx = np.linspace(p.J.min(), p.J.max(), 50)
    ax[0, j].set(title=f"{ds} {filt}: Delta vs J (shift model)\nbest={r.best}, delta={r.delta_nm:+.2f}+-{r.delta_err:.2f} nm",
                 xlabel="J = dm/d(delta) of own spectrum [mag/nm]", ylabel="Delta [mmag]")
    ax[1, j].scatter(np.log10(p.Rleak), 1e3 * p.d, c=p.c, cmap="coolwarm", s=40, edgecolor="k", vmin=-0.5, vmax=2.0)
    ax[1, j].set(title=f"Delta vs leak ratio R (leak model, L={r.leak_L:.1e})", xlabel="log10 R (650-1050 nm / in-band)",
                 ylabel="Delta [mmag]")
    for a_ in (ax[0, j], ax[1, j]):
        a_.grid(alpha=0.3); a_.axhline(0, color="k", lw=0.5)
fig.colorbar(sc, ax=ax, shrink=0.6, label="BP-RP")
fig.suptitle("A4  Standards' offsets vs model predictors from their exact spectra (colour = BP-RP)", fontsize=12.5)
fig.savefig(OUT / f"A4_models{TAG}.png", dpi=125, bbox_inches="tight")
print("saved A4 + stepA_models.csv")
