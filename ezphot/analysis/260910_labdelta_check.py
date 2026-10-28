#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260910_labdelta_check.py

Cross-check the directly-measured (lab) filter offsets delta_f against (1) our fitted intrinsic offsets
(effective_curves/offset_plus_linear_summary.csv 'offset' column) and (2) the closed loop: does inserting the
lab delta_f into the curve collapse the field colour term and the CALSPEC DA-edge offset the way our fitted
delta_f does?

Four curve variants (corrected XP, standard fields), same syn_mag closed loop as 260910_fourway_refcat.py:
  V_nom  = vendor Final_transmission (= archive M4)
  V_beam = physical total (beam only, n_eff=1.5, delta_f=0)          -> rigid-shift baseline
  V_fit  = V_beam rigidly shifted by our fitted delta_f (offset col)  ~ effective curve
  V_lab  = V_beam rigidly shifted by the lab-measured delta_f
(rigid-shifting the total by delta_f moves the filter edges correctly; the smooth S shift is negligible for a
 colour term.)  V_fit's colour term should match a_offset+k*lambda; V_beam's should match a_k*lambda.

  MPLCONFIGDIR=/tmp/ezphot-matplotlib /home/hhchoi1022/miniconda3/envs/too/bin/python \\
      ezphot/analysis/260910_labdelta_check.py
"""
from __future__ import annotations

import importlib
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402
import pandas as pd                 # noqa: E402
from astropy.table import Table     # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
R = importlib.import_module("260908_zpcolor_shift_recal")

LOG = Path("/home/hhchoi1022/ezphot/log")
ZPC = LOG / "20260908_zp_color_check"
EFF = LOG / "20260910_effective_curves"
TS = LOG / "20260910_transmission_sources"
PHYS = TS / "total_curves_physical"
CALSPEC_DIR = Path("/qso/data3/doabdi/intern/synthetic/calspec_data")
GRID = R.GRID
XP_WAVE = R.XP_WAVE_NM

# user-supplied directly-measured filter offsets (nm), normal incidence, vs vendor curve
LAB_DELTA = {"m400": 1.3, "m425": 1.5, "m450": 0.0, "m475": -1.6, "m500": 1.0, "m525": 1.2,
             "m550": 0.0, "m575": 0.0, "m600": 0.5, "m625": 0.0, "m650": -0.3, "m675": -0.3,
             "m700": 1.0, "m725": 2.3, "m750": 0.0, "m775": 0.0, "m800": -1.0, "m825": 1.0,
             "m850": 0.0, "m875": 2.2}


def shift_curve(lam, trans, d):
    return lam, np.interp(lam - d, lam, trans, left=0.0, right=0.0)


def main():
    fit = pd.read_csv(EFF / "offset_plus_linear_summary.csv").set_index("filter")
    st_fit = pd.read_csv(EFF / "shift_table.csv").set_index("filter")

    # ---- (1) delta_f comparison ----
    cmp = []
    for f in GRID:
        lab = LAB_DELTA[f]
        fd = fit.loc[f, "offset"] if f in fit.index else np.nan
        sig = st_fit.loc[f, "sigma_meas"] if f in st_fit.index else np.nan
        cmp.append(dict(filter=f, pivot=fit.loc[f, "pivot"] if f in fit.index else np.nan,
                        lab=lab, fitted=round(float(fd), 3), sigma=round(float(sig), 2) if np.isfinite(sig) else np.nan,
                        resid=round(lab - float(fd), 3) if np.isfinite(fd) else np.nan))
    cmp = pd.DataFrame(cmp)
    good = cmp.dropna(subset=["resid"])
    r = good.resid.to_numpy()
    print("delta_f: lab vs fitted (offset col)")
    print(cmp.to_string(index=False))
    print(f"\nresidual (lab - fitted): mean {r.mean():+.2f}, rms {np.sqrt((r**2).mean()):.2f} nm, "
          f"corr {np.corrcoef(good.lab, good.fitted)[0,1]:.2f}")
    w = good.dropna(subset=["sigma"])
    zscore = (w.lab - w.fitted) / w.sigma
    print("|lab-fitted|>2sigma:", list(w["filter"][np.abs(zscore) > 2]))

    # ---- (2) closed loop, corrected XP, 4 variants ----
    zf = np.load(TS / "refstar_flux_origcor.npz")
    ids = zf["ids"]; flux_cor = zf["flux_cor"]
    idx_of = {s: i for i, s in enumerate(ids)}
    frames = pd.read_csv(ZPC / "frames.csv")
    bad = set(frames[frames["ZPERR_APER_3"] > 0.08]["file"])
    st = pd.read_csv(ZPC / "refit_stars.csv.gz", dtype={"ref_id": str})
    st = st[(st.catalog == "GAIAXP_CORR_LAMOST") & ~st.file.isin(bad)
            & st["filter"].isin(GRID) & st.ref_id.isin(idx_of)].copy()
    st["row"] = st.ref_id.map(idx_of).astype(int)
    st["sig"] = np.sqrt(st.magerr.to_numpy() ** 2 + 0.01 ** 2)
    cf = R.calspec_clean(ZPC, bad)
    is_da = {o: str(t).startswith("DA") for o, t in cf.groupby("object")["spectral_type"].first().items()}
    syn = pd.read_csv(ZPC / "calspec_synphot.csv").drop_duplicates("object")[["object", "spec"]]
    cal_spec = {}
    for o, sp in zip(syn.object, syn.spec):
        t = Table.read(CALSPEC_DIR / sp)
        wl = np.asarray(t["WAVELENGTH"], float) / 10.0; fl = np.asarray(t["FLUX"], float) * 1e-2
        m = (wl > 300) & (wl < 1000) & np.isfinite(fl)
        cal_spec[o] = (wl[m], fl[m])

    rows = []
    for f in GRID:
        rf = st[st["filter"] == f]
        cff = cf[cf["filter"] == f]
        if len(rf) < 20:
            continue
        row = rf.row.to_numpy()
        pd_ = pd.read_csv(PHYS / f"{f}.csv"); lam, tr = pd_.lam.to_numpy(), pd_.trans.to_numpy()
        variants = {
            "nom": R.load_curve(f),
            "beam": (lam, tr),
            "fit": shift_curve(lam, tr, float(fit.loc[f, "offset"])) if f in fit.index else (lam, tr),
            "lab": shift_curve(lam, tr, LAB_DELTA[f]),
        }
        m_arch = rf.ref_mag.to_numpy()
        for vk, (fw, ft) in variants.items():
            m_ref = R.syn_mag(XP_WAVE, flux_cor[row], fw, ft, 0.0)
            zp_i = rf.zp.to_numpy() + (m_ref - m_arch)
            med, sd, n, keep = R.clipped_zp(rf.file.to_numpy(), zp_i)
            resid = zp_i - rf.file.map(med).to_numpy()
            a, a_err, b, nfit = R.weighted_linfit(rf.bprp.to_numpy()[keep], resid[keep], rf.sig.to_numpy()[keep])
            out = dict(filter=f, pivot=float(fit.loc[f, "pivot"]) if f in fit.index else np.nan,
                       variant=vk, a_bprp=round(a, 4), zp_std=round(float(np.nanmedian(sd.values)), 4))
            if len(cff):
                m_true = {o: float(R.syn_mag(cal_spec[o][0], cal_spec[o][1], fw, ft, 0.0)) for o in cal_spec}
                mobs = cff.mag_inst_APER_3.to_numpy() + cff.file.map(med).to_numpy()
                delta = mobs - cff.object.map(m_true).to_numpy()
                cc = cff.assign(delta=delta).dropna(subset=["delta"])
                per = cc.groupby("object").agg(n=("delta", "size"), d=("delta", "median"))
                per = per[per.n >= 10]
                per["is_da"] = [is_da.get(o, False) for o in per.index]
                da = per[per.is_da].d; ot = per[~per.is_da].d
                out["da_other"] = round(float(da.median() - ot.median()), 4) if len(da) and len(ot) else np.nan
                out["mad"] = round(float(R.mad(per.d.values)), 4) if len(per) >= 3 else np.nan
            rows.append(out)
    res = pd.DataFrame(rows)
    res.to_csv(TS / "labdelta_closedloop.csv", index=False)
    cmp.to_csv(TS / "labdelta_vs_fitted.csv", index=False)

    piv = res.pivot_table(index="filter", columns="variant", values=["a_bprp", "da_other"])
    print("\nclosed loop |colour term| and DA-other by variant (nom/beam/fit/lab):")
    print(piv.round(4).to_string())
    for metric in ["a_bprp", "da_other"]:
        sub = res.dropna(subset=[metric])
        print(f"\nmedian |{metric}| over filters:  "
              + "  ".join(f"{v}={sub[sub.variant==v][metric].abs().median():.4f}" for v in ["nom", "beam", "fit", "lab"]))

    # ---- figure ----
    col = {"nom": "#2ca02c", "beam": "#d62728", "fit": "#1f77b4", "lab": "#ff7f0e"}
    lab_ = {"nom": "nominal (vendor)", "beam": "physical beam-only (df=0)",
            "fit": "beam + fitted df", "lab": "beam + LAB df"}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    ax = axes[0]
    g = cmp.dropna(subset=["fitted"])
    ax.errorbar(g.fitted, g.lab, xerr=g.sigma, fmt="o", ms=6, color="#333", ecolor="#aaa", capsize=2)
    for _, rr in g.iterrows():
        ax.annotate(rr["filter"], (rr["fitted"], rr["lab"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    lim = [-4, 4]; ax.plot(lim, lim, "k--", lw=1, alpha=0.6)
    ax.set(xlim=lim, ylim=lim, xlabel="our fitted delta_f [nm]", ylabel="lab-measured delta_f [nm]",
           title="(a) intrinsic offset: lab vs fitted")
    ax.grid(alpha=0.3)
    for ax, metric, ylab in [(axes[1], "a_bprp", "field colour term a(BP-RP)"),
                             (axes[2], "da_other", "CALSPEC DA - other [mag]")]:
        for v in ["nom", "beam", "fit", "lab"]:
            s = res[res.variant == v].dropna(subset=[metric]).sort_values("pivot")
            ax.plot(s["pivot"], s[metric], "o-", ms=4, color=col[v], label=lab_[v])
        ax.axhline(0, color="k", lw=0.5); ax.set(xlabel="pivot [nm]", ylabel=ylab)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    axes[1].set_title("(b) field colour term"); axes[2].set_title("(c) CALSPEC DA-edge")
    fig.suptitle("F13  do the lab-measured filter offsets explain the on-sky offsets?")
    fig.tight_layout(); fig.savefig(TS / "figures" / "F13_labdelta.png", dpi=150)
    print("\nsaved -> labdelta_vs_fitted.csv, labdelta_closedloop.csv, figures/F13_labdelta.png")


if __name__ == "__main__":
    main()
