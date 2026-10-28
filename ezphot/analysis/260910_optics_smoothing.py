#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260910_optics_smoothing.py

Optics-efficiency sensitivity, two tests:
  (A) interpolation method between the 7 nodes (400-1000 nm): linear vs cubic vs Akima -> negligible
      (colour term / DA / median move < 0.001 mag; see optics_interp.csv).
  (B) the BELOW-400 nm extrapolation slope, which is unconstrained (no optics data there and the enhanced-Al
      mirror reflectivity is expected to fall in the UV). Vary the sub-400 slope as a multiple of the node
      slope (400->500), rebuild S = QE x optics x atm, run the same closed loop (corrected XP, beam curve,
      delta_f=0). Among the GRID filters only m400 (band 388-413 nm) reaches below 400 nm, so it is the probe.

Updates figures/F15_optics_smoothing.png with the slope test.

  MPLCONFIGDIR=/tmp/ezphot-matplotlib /home/hhchoi1022/miniconda3/envs/too/bin/python \\
      ezphot/analysis/260910_optics_smoothing.py
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
from scipy.interpolate import CubicSpline, Akima1DInterpolator   # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
R = importlib.import_module("260908_zpcolor_shift_recal")

SRC = Path("~/ezphot/config/common/transmission/Original_sources").expanduser()
LOG = Path("/home/hhchoi1022/ezphot/log")
ZPC = LOG / "20260908_zp_color_check"
TS = LOG / "20260910_transmission_sources"
FBEAM = TS / "filters_beamshift"
CALSPEC_DIR = Path("/qso/data3/doabdi/intern/synthetic/calspec_data")
GRID = R.GRID
XP_WAVE = R.XP_WAVE_NM

# lab-measured intrinsic filter offsets (nm), applied on top of the beam curve
LAB_DELTA = {"m400": 1.3, "m425": 1.5, "m450": 0.0, "m475": -1.6, "m500": 1.0, "m525": 1.2,
             "m550": 0.0, "m575": 0.0, "m600": 0.5, "m625": 0.0, "m650": -0.3, "m675": -0.3,
             "m700": 1.0, "m725": 2.3, "m750": 0.0, "m775": 0.0, "m800": -1.0, "m825": 1.0,
             "m850": 0.0, "m875": 2.2}

_opt = pd.read_csv(SRC / "optics.efficiency.dr350.dr500.csv", sep=r"\s+", comment="#", skiprows=11)
_opt.columns = ["nm", "K9LA", "LaK4L", "FK95N", "VIS", "Al", "Total_pct", "total"]
OX, OY = _opt.nm.to_numpy(float), _opt.total.to_numpy(float)
NODE_SLOPE = (OY[1] - OY[0]) / (OX[1] - OX[0])                     # 400->500 slope


def optics_on(grid, method="linear", blue_mult=1.0):
    if method == "linear":
        v = np.interp(grid, OX, OY)
    elif method == "cubic":
        v = CubicSpline(OX, OY, bc_type="natural")(grid)
    else:
        v = Akima1DInterpolator(OX, OY)(grid)
    v = np.asarray(v, float)
    below = grid < OX[0]
    v[below] = np.maximum(OY[0] + blue_mult * NODE_SLOPE * (grid[below] - OX[0]), 0.0)
    v[grid > OX[-1]] = OY[-1]
    return np.clip(v, 0.0, None)


def load_inputs():
    zf = np.load(TS / "refstar_flux_origcor.npz")
    ids = zf["ids"]; flux_cor = zf["flux_cor"]; idx_of = {s: i for i, s in enumerate(ids)}
    frames = pd.read_csv(ZPC / "frames.csv"); bad = set(frames[frames["ZPERR_APER_3"] > 0.08]["file"])
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
        t = Table.read(CALSPEC_DIR / sp); wl = np.asarray(t["WAVELENGTH"], float) / 10.0
        fl = np.asarray(t["FLUX"], float) * 1e-2; m = (wl > 300) & (wl < 1000) & np.isfinite(fl)
        cal_spec[o] = (wl[m], fl[m])
    return st, cf, is_da, cal_spec, flux_cor


def closed_loop(st, cf, is_da, cal_spec, flux_cor, grid, t_qe, t_atm, optics, variants):
    S = {v: t_qe * optics[v] * t_atm for v in variants}
    rows = []
    for f in GRID:
        rf = st[st["filter"] == f]; cff = cf[cf["filter"] == f]
        if len(rf) < 20:
            continue
        row = rf.row.to_numpy(); m_arch = rf.ref_mag.to_numpy()
        fb = pd.read_csv(FBEAM / f"{f}.csv")
        tr_lab = np.interp(fb.lam - LAB_DELTA[f], fb.lam, fb.trans, left=0, right=0)   # apply lab delta_f
        F = np.interp(grid, fb.lam.to_numpy(), tr_lab, left=0, right=0)
        for v in variants:
            ft = F * S[v]
            m_ref = R.syn_mag(XP_WAVE, flux_cor[row], grid, ft, 0.0)
            zp_i = rf.zp.to_numpy() + (m_ref - m_arch)
            med, sd, n, keep = R.clipped_zp(rf.file.to_numpy(), zp_i)
            resid = zp_i - rf.file.map(med).to_numpy()
            a, a_err, b, nfit = R.weighted_linfit(rf.bprp.to_numpy()[keep], resid[keep], rf.sig.to_numpy()[keep])
            out = dict(filter=f, pivot=float(f[1:4]), variant=v, a_bprp=round(a, 4),
                       zp_std=round(float(np.nanmedian(sd.values)), 4))
            if len(cff):
                m_true = {o: float(R.syn_mag(cal_spec[o][0], cal_spec[o][1], grid, ft, 0.0)) for o in cal_spec}
                mobs = cff.mag_inst_APER_3.to_numpy() + cff.file.map(med).to_numpy()
                cc = cff.assign(delta=mobs - cff.object.map(m_true).to_numpy()).dropna(subset=["delta"])
                per = cc.groupby("object").agg(n=("delta", "size"), d=("delta", "median"))
                per = per[per.n >= 10]; per["is_da"] = [is_da.get(o, False) for o in per.index]
                da = per[per.is_da].d; ot = per[~per.is_da].d
                out["da_other"] = round(float(da.median() - ot.median()), 4) if len(da) and len(ot) else np.nan
                out["median_all"] = round(float(per.d.median()), 4)
            rows.append(out)
    return pd.DataFrame(rows)


def main():
    atm = pd.read_csv(TS / "atmosphere_elsauce.csv")
    grid = atm.lam_air.to_numpy(); sel = (grid >= 330) & (grid <= 1010)
    grid = grid[sel]; t_atm = atm.trans.to_numpy()[sel]
    qe = pd.read_csv(SRC / "QE.csv"); t_qe = np.interp(grid, qe.iloc[:, 0], qe.iloc[:, 1])
    st, cf, is_da, cal_spec, flux_cor = load_inputs()

    # (A) interpolation-method test (kept for the record)
    interp = {m: optics_on(grid, method=m, blue_mult=1.0) for m in ["linear", "cubic", "akima"]}
    resA = closed_loop(st, cf, is_da, cal_spec, flux_cor, grid, t_qe, t_atm, interp, list(interp))
    resA.to_csv(TS / "optics_interp.csv", index=False)
    dcl = resA[resA.variant == "cubic"].a_bprp.values - resA[resA.variant == "linear"].a_bprp.values
    print(f"(A) interp linear vs cubic: colour-term max|delta| {np.nanmax(np.abs(dcl)):.4f} (negligible)")

    # (B) below-400 slope test
    mults = {"flat (0x)": 0.0, "1x (current)": 1.0, "2x": 2.0, "3x": 3.0}
    optics = {k: optics_on(grid, method="linear", blue_mult=m) for k, m in mults.items()}
    variants = list(mults)
    resB = closed_loop(st, cf, is_da, cal_spec, flux_cor, grid, t_qe, t_atm, optics, variants)
    resB.to_csv(TS / "optics_blue_slope.csv", index=False)
    print("\n(B) below-400 slope test -- m400 response:")
    print(resB[resB["filter"] == "m400"][["variant", "a_bprp", "da_other", "median_all", "zp_std"]].to_string(index=False))
    for at in [375, 388]:
        print(f"   optics at {at} nm:  " + "  ".join(
            f"{k}={OY[0] + m*NODE_SLOPE*(at-OX[0]):.3f}" for k, m in mults.items()))

    # ---- figure (corrected-XP CALSPEC only) ----
    col = {"flat (0x)": "#2ca02c", "1x (current)": "#1f77b4", "2x": "#ff7f0e", "3x": "#d62728"}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    ax = axes[0]
    gg = np.linspace(360, 640, 400)
    for k, m in mults.items():
        ax.plot(gg, optics_on(gg, "linear", m), "-", color=col[k], lw=1.6, label=k)
    ax.plot(OX[:3], OY[:3], "ks", ms=9, label="data nodes")
    ax.axvspan(360, 400, color="grey", alpha=0.08); ax.axvline(400, color="k", lw=0.6, ls=":")
    ax.set(xlim=(360, 640), ylim=(0.45, 0.95), xlabel="wavelength [nm]", ylabel="optics total",
           title="(a) below-400 nm extrapolation slope (grey = no data)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    for ax, metric, ylab, ttl in [(axes[1], "da_other", "DA - other [mag]", "(b) CALSPEC DA-edge (corrected)"),
                                   (axes[2], "median_all", "CALSPEC median [mag]", "(c) CALSPEC median (corrected)")]:
        for v in variants:
            s = resB[resB.variant == v].dropna(subset=[metric]).sort_values("pivot")
            ax.plot(s["pivot"], s[metric], "o-", ms=4, color=col[v], label=v)
        ax.axhline(0, color="k", lw=0.5)
        ax.annotate("m400", (400, resB[(resB.variant == "3x") & (resB["filter"] == "m400")][metric].iloc[0]),
                    fontsize=8, color="#d62728", xytext=(6, 4), textcoords="offset points")
        ax.set(xlabel="pivot [nm]", ylabel=ylab, title=ttl); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("F15  optics below-400 nm slope (beam + lab delta_f, corrected XP): CALSPEC response")
    fig.tight_layout(); fig.savefig(TS / "figures" / "F15_optics_smoothing.png", dpi=150)
    print("\nsaved -> optics_interp.csv, optics_blue_slope.csv, figures/F15_optics_smoothing.png")


if __name__ == "__main__":
    main()
