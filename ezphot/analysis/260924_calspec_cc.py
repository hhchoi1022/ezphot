#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260924_calspec_cc.py — CALSPEC offsets BEFORE and AFTER the field colour-term correction, per curve combo.

Per frame and combo (version x beam): ZP from corrected-XP field stars (SNR>40), field colour term a_frame
(weighted fit of ZP residual vs BP-RP over the clipped bright set), c_ref (their weighted mean colour),
standard star Delta = m_inst + ZP - syn_CALSPEC, and the colour-matched version
    Delta_cc = Delta + a_frame * (c_star - c_ref),   c_star = synthetic Gaia BP-RP of the CALSPEC spectrum.

Output: calspec_cc_frames.csv, calspec_cc_summary.csv, C3_calspec_cc.png

  MPLCONFIGDIR=/tmp/ezphot-matplotlib OMP_NUM_THREADS=1 /home/hhchoi1022/miniconda3/envs/too/bin/python -u \\
      ezphot/analysis/260924_calspec_cc.py --n-proc 12
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402
import pandas as pd                 # noqa: E402

warnings.filterwarnings("ignore")
from astropy.coordinates import SkyCoord, match_coordinates_sky   # noqa: E402
from astropy.io import fits          # noqa: E402
from astropy.stats import SigmaClip  # noqa: E402
from astropy.table import Table      # noqa: E402

sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
TC = importlib.import_module("260921_truthcompare")
OT, OZ, TW, K, R = TC.OT, TC.OZ, TC.TW, TC.K, TC.R
XP_WAVE = TC.XP_WAVE
OUT = Path("/home/hhchoi1022/ezphot/log/20260924_colorterm_versions")
COMBOS = [("manufacturer", "nobeam"), ("manufacturer", "beam"), ("version_mhj", "nobeam"), ("version_mhj", "beam")]
MIN_REF = 10


def log(m):
    print(time.strftime("[%H:%M:%S] ") + m, flush=True)


def star_colour(wl, fl):
    import pyphot
    lib = pyphot.get_library()

    def vega(bn):
        f = lib[bn]; fx = f.get_flux(wl * 10.0, fl * 100.0)
        fx = fx.magnitude if hasattr(fx, "magnitude") else float(fx)
        return -2.5 * np.log10(fx) - float(f.Vega_zero_mag)
    return float(vega("Gaia_MAW_BP_bright") - vega("Gaia_MAW_RP"))


def process_cal(args):
    obj, files = args
    st = TW._INFO[obj]
    if not st["spec"]:
        return []
    t = Table.read(TW.CALSPEC_DIR / st["spec"])
    wl = np.asarray(t["WAVELENGTH"], float) / 10.0
    fl = np.asarray(t["FLUX"], float) * 1e-2
    m = (wl > 300) & (wl < 1100) & np.isfinite(fl) & (fl > 0)
    wl, fl = wl[m], fl[m]
    c_star = star_colour(wl, fl)
    got = K.get_reference("GAIAXP_CORR_LAMOST", obj, st["ra"], st["dec"], 1.34, 0.90)
    if got is None:
        return []
    tbl, coords, _ = got
    prepped, need = [], set()
    for fr in files:
        cp = TC.CAL_CATD / (Path(fr).name[:-5] + "_cat.fits")
        if not cp.exists():
            continue
        filt, unit = fr.split("/")[-2], fr.split("/")[-3]
        try:
            with fits.open(cp) as h:
                hdr = h[0].header
            df = OZ.load_ez(cp)
            sel = OT.select_stars_like(df)
            if len(sel) < MIN_REF:
                continue
            src = SkyCoord(sel.ALPHA_J2000.to_numpy(), sel.DELTA_J2000.to_numpy(), unit="deg")
            idx, d2d, _ = match_coordinates_sky(src, coords)
            ok = np.asarray(d2d.arcsec < 2.0)
            if ok.sum() < MIN_REF:
                continue
            mr = tbl[idx[ok]]
            mjd = float(hdr.get("MJD", np.nan))
            dt = (mjd - 51544.5) / 365.25 if np.isfinite(mjd) else 24.5
            pos = SkyCoord(st["ra"] + st["pmra"] * dt / 3.6e6 / np.cos(np.deg2rad(st["dec"])),
                           st["dec"] + st["pmdec"] * dt / 3.6e6, unit="deg")
            call = SkyCoord(df.ALPHA_J2000.to_numpy(), df.DELTA_J2000.to_numpy(), unit="deg")
            sep = call.separation(pos).arcsec
            js = int(np.argmin(sep))
            srow_ = df.iloc[js]
            if not (sep[js] < 1.5 and srow_.FLAGS <= 1 and srow_.MAGERR_APER_3 < 0.05):
                continue
            prepped.append(dict(filt=filt, unit=unit, rid=np.asarray(mr["id"]).astype(str),
                                bprp=np.asarray(mr["bp-rp"], float),
                                inst=sel.MAG_APER_3.to_numpy()[ok], ierr=sel.MAGERR_APER_3.to_numpy()[ok],
                                snr=sel.SNR_APER_3.to_numpy()[ok], star_inst=float(srow_.MAG_APER_3)))
            need.update(prepped[-1]["rid"])
        except Exception:                                    # noqa: BLE001
            continue
    if not prepped:
        return []
    xpf = TW.tile_xp_radec(st["ra"], st["dec"], need)
    sc = SigmaClip(sigma=3.0, maxiters=5)
    rows = []
    for pr in prepped:
        keep = np.array([r in xpf for r in pr["rid"]])
        if keep.sum() < MIN_REF:
            continue
        fc = np.vstack([xpf[r] for r in pr["rid"][keep]])
        bp, inst, snr = pr["bprp"][keep], pr["inst"][keep], pr["snr"][keep]
        sig = np.sqrt(pr["ierr"][keep] ** 2 + 0.01 ** 2)
        m_true = None
        for ver, var in COMBOS:
            curves = TC._TOTALS[(ver, var)]
            if pr["filt"] not in curves:
                continue
            fw, ft = curves[pr["filt"]]
            zp_arr = R.syn_mag(XP_WAVE, fc, fw, ft, 0.0) - inst
            zsel = np.isfinite(zp_arr) & np.isfinite(bp) & (snr > 40.0)
            if zsel.sum() < MIN_REF:
                continue
            cl = sc(zp_arr[zsel])
            zp = float(np.ma.median(cl))
            if not np.isfinite(zp) or float(np.ma.std(cl)) > 0.08:
                continue
            gz = ~np.ma.getmaskarray(cl)
            bz, zz, sz = bp[zsel][gz], zp_arr[zsel][gz], sig[zsel][gz]
            a, _, _, _ = R.weighted_linfit(bz, zz - zp, sz)
            c_ref = float(np.average(bz, weights=1.0 / sz ** 2))
            m_true = float(R.syn_mag(wl, fl, fw, ft, 0.0))
            d = pr["star_inst"] + zp - m_true
            rows.append(dict(object=obj, filter=pr["filt"], unit=pr["unit"], version=ver, variant=var,
                             n_zp=int(gz.sum()), a_frame=round(float(a), 4), c_ref=round(c_ref, 3),
                             c_star=round(c_star, 3), d=round(d, 4),
                             d_cc=round(d + float(a) * (c_star - c_ref), 4)))
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n-proc", type=int, default=12)
    a = ap.parse_args()
    cmeta = pd.read_csv(TC.CAL_META)
    cmeta = cmeta[cmeta.status == "ok"]
    jobs = [(obj, TC.cap_frames([(f.split("/")[-2], f) for f in g.file])) for obj, g in cmeta.groupby("object")]
    log(f"{len(jobs)} CALSPEC objects, cap {TC.FRAME_CAP}/filter")
    rows = []
    with Pool(min(a.n_proc, len(jobs)), initializer=TC._init) as pool:
        for i, r in enumerate(pool.imap_unordered(process_cal, jobs), 1):
            rows += r
            log(f"  [{i}/{len(jobs)}] rows {len(rows)}")
    fr = pd.DataFrame(rows)
    fr.to_csv(OUT / "calspec_cc_frames.csv", index=False)
    fr["combo"] = fr.version + "_" + fr.variant
    per = fr.groupby(["combo", "filter", "object"]).agg(d=("d", "median"), d_cc=("d_cc", "median"),
                                                          a=("a_frame", "median"), n=("d", "size")).reset_index()
    per = per[per.n >= 2]
    S = per.groupby(["combo", "filter"]).agg(
        n_stars=("object", "size"), d_mmag=("d", lambda v: 1e3 * v.median()),
        d_err=("d", lambda v: 1e3 * 1.4826 * np.median(np.abs(v - v.median())) / np.sqrt(len(v))),
        dcc_mmag=("d_cc", lambda v: 1e3 * v.median()),
        dcc_err=("d_cc", lambda v: 1e3 * 1.4826 * np.median(np.abs(v - v.median())) / np.sqrt(len(v))),
        a_frame=("a", "median")).reset_index()
    S["pivot_nm"] = S["filter"].str[1:4].astype(int)
    S = S.sort_values(["combo", "pivot_nm"]).round(4)
    S.to_csv(OUT / "calspec_cc_summary.csv", index=False)
    print(S.to_string(index=False))

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.5), sharex=True, sharey=True)
    for ax, (ver, var) in zip(axes.ravel(), COMBOS):
        s = S[S.combo == f"{ver}_{var}"]
        ax.errorbar(s["pivot_nm"], s.d_mmag, yerr=s.d_err, color="#1f4e8c", marker="o", ms=4.5, lw=1.5,
                    label="Δ_CAL (no colour-term correction)")
        ax.errorbar(s["pivot_nm"], s.dcc_mmag, yerr=s.dcc_err, color="#C25B4E", marker="s", ms=4.5, lw=1.5,
                    ls="--", label="Δ_CAL,cc = Δ + a_frame·(c_star − c_ref)")
        ax.axhline(0, color="k", lw=0.7); ax.grid(alpha=0.3)
        ax.set_title(f"{ver} · {var}", fontsize=11.5)
        ax.legend(fontsize=8.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("[mmag]")
    for ax in axes[1]:
        ax.set_xlabel("pivot wavelength [nm]")
    fig.suptitle("C3  CALSPEC offsets before / after field colour-term correction (CALSPEC truth, ZP from corrected XP)",
                 fontsize=12.5)
    fig.tight_layout()
    fig.savefig(OUT / "C3_calspec_cc.png", dpi=140)
    log("saved -> calspec_cc_frames.csv, calspec_cc_summary.csv, C3_calspec_cc.png")


if __name__ == "__main__":
    main()
