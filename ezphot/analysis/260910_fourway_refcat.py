#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260910_fourway_refcat.py

Four synthetic reference catalogues = {NEW physical curve, OLD nominal curve} x {XP Original, XP Corrected(LAMOST)}:
  M1 new x orig, M2 new x corr, M3 old x orig, M4 old x corr (= archive GAIAXP_CORR_LAMOST).

Reuses the closed-loop machinery of 260908_zpcolor_shift_recal.py (syn_mag: 2 nm photon-weighted trapezoid,
3631 Jy AB; clipped_zp; weighted_linfit) to, per filter and per method:
  - field reference stars: per-frame ZP (3s-clipped median) -> ZP scatter (= ZP error) and colour term a(BP-RP);
  - CALSPEC stars: calibrate each frame's standard with that method's ZP, compare to the CALSPEC true spectrum
    synphot through the SAME curve -> Delta, DA-minus-other, between-star MAD.

NEW curve = physical total curves (260910_total_assembly.py, no delta_f, n_eff=1.5).
OLD curve = the archive Final_transmission (R.load_curve). Ref-star original flux is re-extracted
(refstar_flux_origcor.npz); corrected reproduces the npz to 0.0.

  MPLCONFIGDIR=/tmp/ezphot-matplotlib /home/hhchoi1022/miniconda3/envs/too/bin/python \\
      ezphot/analysis/260910_fourway_refcat.py
"""
from __future__ import annotations

import importlib
import sys
import time
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
PHYS = LOG / "20260910_transmission_sources" / "total_curves_physical"
OUT = LOG / "20260910_transmission_sources"
CALSPEC_DIR = Path("/qso/data3/doabdi/intern/synthetic/calspec_data")

GRID = R.GRID                                   # 20 bands m400..m875
XP_WAVE = R.XP_WAVE_NM
MIN_FRAMES = 10
METHODS = ["M1_new_orig", "M2_new_corr", "M3_old_orig", "M4_old_corr"]


def log(m):
    print(time.strftime("[%H:%M:%S] ") + m, flush=True)


def load_new_curve(f):
    d = pd.read_csv(PHYS / f"{f}.csv")
    return d.lam.to_numpy(), d.trans.to_numpy()


def main():
    # ---- field reference stars ----
    zf = np.load(OUT / "refstar_flux_origcor.npz")
    ids = zf["ids"]
    flux = {"orig": zf["flux_orig"], "corr": zf["flux_cor"]}
    idx_of = {s: i for i, s in enumerate(ids)}

    frames = pd.read_csv(ZPC / "frames.csv")
    bad = set(frames[frames["ZPERR_APER_3"] > 0.08]["file"])
    st = pd.read_csv(ZPC / "refit_stars.csv.gz", dtype={"ref_id": str})
    STANDARDS = sorted(st.objname.unique())
    st = st[(st.catalog == "GAIAXP_CORR_LAMOST") & ~st.file.isin(bad)
            & st["filter"].isin(GRID) & st.ref_id.isin(idx_of)].copy()
    st["row"] = st.ref_id.map(idx_of).astype(int)
    st["sig"] = np.sqrt(st.magerr.to_numpy() ** 2 + 0.01 ** 2)
    log(f"field ref stars: {len(st)} rows, {st.file.nunique()} frames, {st['filter'].nunique()} filters")

    # ---- CALSPEC ----
    cf = R.calspec_clean(ZPC, bad)
    is_da = {o: str(t).startswith("DA") for o, t in cf.groupby("object")["spectral_type"].first().items()}
    syn = pd.read_csv(ZPC / "calspec_synphot.csv").drop_duplicates("object")[["object", "spec"]]
    cal_spec = {}
    for o, sp in zip(syn.object, syn.spec):
        t = Table.read(CALSPEC_DIR / sp)
        wl = np.asarray(t["WAVELENGTH"], float) / 10.0
        fl = np.asarray(t["FLUX"], float) * 1e-2
        m = (wl > 300) & (wl < 1000) & np.isfinite(fl)
        cal_spec[o] = (wl[m], fl[m])
    log(f"CALSPEC: {cf.object.nunique()} stars, {len(cf)} frames after clean; spectra loaded {len(cal_spec)}")

    # ---- per filter, per method ----
    field_rows, cal_rows, star_rows = [], [], []
    for f in GRID:
        rf = st[st["filter"] == f]
        cff = cf[cf["filter"] == f]
        if len(rf) < 20:
            continue
        row = rf.row.to_numpy()
        curve_old = R.load_curve(f)
        curve_new = load_new_curve(f)
        # reference-star synthetic mags
        m_ref = {}
        for ckey, curve in [("old", curve_old), ("new", curve_new)]:
            for xkey in ("orig", "corr"):
                m_ref[(ckey, xkey)] = R.syn_mag(XP_WAVE, flux[xkey][row], curve[0], curve[1], 0.0)
        m_arch = rf.ref_mag.to_numpy()                                   # = M4 baseline
        d_M4 = np.nanmax(np.abs(m_ref[("old", "corr")] - m_arch))
        # CALSPEC truth per curve set
        m_true = {}
        for ckey, curve in [("old", curve_old), ("new", curve_new)]:
            m_true[ckey] = {o: float(R.syn_mag(cal_spec[o][0], cal_spec[o][1], curve[0], curve[1], 0.0))
                            for o in cal_spec}

        for method in METHODS:
            ckey = "new" if "new" in method else "old"
            xkey = "orig" if "orig" in method else "corr"
            zp_i = rf.zp.to_numpy() + (m_ref[(ckey, xkey)] - m_arch)
            med, sd, n, keep = R.clipped_zp(rf.file.to_numpy(), zp_i)
            resid = zp_i - rf.file.map(med).to_numpy()
            a, a_err, b, nfit = R.weighted_linfit(rf.bprp.to_numpy()[keep], resid[keep], rf.sig.to_numpy()[keep])
            field_rows.append(dict(filter=f, method=method, n_stars=len(rf), n_frames=int(rf.file.nunique()),
                                   zp_std=round(float(np.nanmedian(sd.values)), 4),
                                   a_bprp=round(a, 4), a_err=round(a_err, 4), n_fit=nfit,
                                   d_M4_reproduce=round(float(d_M4), 6) if method == "M4_old_corr" else np.nan))
            # CALSPEC
            if len(cff):
                mobs = cff.mag_inst_APER_3.to_numpy() + cff.file.map(med).to_numpy()
                delta = mobs - cff.object.map(m_true[ckey]).to_numpy()
                cc = cff.assign(delta=delta).dropna(subset=["delta"])
                per = cc.groupby("object").agg(n=("delta", "size"), d=("delta", "median"),
                                               scat=("delta", R.mad))
                per = per[per.n >= MIN_FRAMES]
                if len(per):
                    per["is_da"] = [is_da.get(o, False) for o in per.index]
                    for o, r_ in per.iterrows():
                        star_rows.append(dict(filter=f, method=method, object=o, is_da=bool(r_.is_da),
                                              n=int(r_.n), delta=round(float(r_.d), 4)))
                    da = per[per.is_da].d; ot = per[~per.is_da].d
                    cal_rows.append(dict(filter=f, method=method, n_stars=len(per),
                                         median_all=round(float(per.d.median()), 4),
                                         da_minus_other=round(float(da.median() - ot.median()), 4)
                                         if len(da) and len(ot) else np.nan,
                                         between_star_mad=round(float(R.mad(per.d.values)), 4) if len(per) >= 3 else np.nan))
    field = pd.DataFrame(field_rows); cal = pd.DataFrame(cal_rows); stars = pd.DataFrame(star_rows)
    field.to_csv(OUT / "fourway_field.csv", index=False)
    cal.to_csv(OUT / "fourway_calspec.csv", index=False)
    stars.to_csv(OUT / "fourway_calspec_stars.csv", index=False)

    dM4 = field[field.method == "M4_old_corr"].d_M4_reproduce
    log(f"M4 reproduces archive ref_mag to max {np.nanmax(dM4):.2e} mag (should be ~1e-3 or below)")
    log("field (ZP scatter, colour term) by method:")
    print(field.pivot_table(index="filter", columns="method", values=["zp_std", "a_bprp"]).round(4).to_string())
    log("CALSPEC (median, DA-other, MAD) by method:")
    print(cal.pivot_table(index="filter", columns="method",
                          values=["median_all", "da_minus_other", "between_star_mad"]).round(4).to_string())

    # ---- figures ----
    field["pivot"] = pd.to_numeric(field["filter"].str[1:], errors="coerce")
    colors = {"M1_new_orig": "#d62728", "M2_new_corr": "#ff7f0e",
              "M3_old_orig": "#1f77b4", "M4_old_corr": "#2ca02c"}
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for method in METHODS:
        fm = field[field.method == method].sort_values("pivot")
        axes[0, 0].plot(fm["pivot"], fm.a_bprp, "o-", ms=4, color=colors[method], label=method)
        axes[0, 1].plot(fm["pivot"], fm.zp_std * 1000, "o-", ms=4, color=colors[method], label=method)
    axes[0, 0].axhline(0, color="k", lw=0.5); axes[0, 0].set(xlabel="pivot [nm]", ylabel="colour term a(BP-RP) [mag/mag]",
                                                             title="field-star colour term")
    axes[0, 1].set(xlabel="pivot [nm]", ylabel="ZP scatter [mmag]", title="field-star ZP scatter (= ZP error)")
    for ax in axes[0]:
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    cal["pivot"] = pd.to_numeric(cal["filter"].str[1:], errors="coerce")
    for method in METHODS:
        cm = cal[cal.method == method].sort_values("pivot")
        axes[1, 0].plot(cm["pivot"], cm.da_minus_other, "o-", ms=4, color=colors[method], label=method)
        axes[1, 1].plot(cm["pivot"], cm.between_star_mad * 1000, "o-", ms=4, color=colors[method], label=method)
    axes[1, 0].axhline(0, color="k", lw=0.5); axes[1, 0].set(xlabel="pivot [nm]", ylabel="DA - other [mag]",
                                                             title="CALSPEC DA white dwarfs minus other standards")
    axes[1, 1].set(xlabel="pivot [nm]", ylabel="between-star MAD [mmag]", title="CALSPEC between-star scatter")
    for ax in axes[1]:
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("F12  four reference catalogues: {new physical, old nominal} x {XP original, XP corrected}")
    fig.tight_layout(); fig.savefig(OUT / "figures" / "F12_fourway.png", dpi=150)
    log(f"saved -> fourway_field.csv, fourway_calspec.csv, fourway_calspec_stars.csv, figures/F12_fourway.png")


if __name__ == "__main__":
    main()
