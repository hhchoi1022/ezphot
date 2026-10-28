#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260915_calspec_offset.py

Per-filter curve offset that MINIMIZES the CALSPEC magnitude difference (NO colour term anywhere):
scan a delta grid; at each delta both the field-star ZP (SNR>40, clipped median) and the CALSPEC
synthetic magnitude are computed through the SAME shifted physical curve; Delta(delta) = inst + ZP(delta)
- m_true(delta).  The zero of the star-median Delta(delta) is the CALSPEC-absolute offset — an
independent route to delta_f (colour-term route = F44).

Frames capped per (object, filter) for runtime; ezphot 10" catalogs from 20260914_calspec_ezphot.

  MPLCONFIGDIR=/tmp/ezphot-matplotlib OMP_NUM_THREADS=1 /home/hhchoi1022/miniconda3/envs/too/bin/python -u \\
      ezphot/analysis/260915_calspec_offset.py --n-proc 13
"""
from __future__ import annotations

import argparse
import importlib
import os
import random
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
TW = importlib.import_module("260914_calspec_twoway_ez")
OT = importlib.import_module("260911_offset_timeseries")
OZ = importlib.import_module("260911_offset_ezphot")
R, K, C = TW.R, TW.K, TW.C

OUT = TW.OUT
CATD = TW.CATD
DGRID = np.arange(-6.0, 6.01, 1.5)
CAP = 25                    # frames per (object, filter)
MIN_REF = 15


def log(m):
    print(time.strftime("[%H:%M:%S] ") + m, flush=True)


def _init():
    TW._init(40)            # SNR>40 ZP


def process_object(args):
    obj, files = args
    st = TW._INFO[obj]
    if not st["spec"]:
        return []
    t = Table.read(TW.CALSPEC_DIR / st["spec"])
    wl = np.asarray(t["WAVELENGTH"], float) / 10.0
    fl = np.asarray(t["FLUX"], float) * 1e-2
    m = (wl > 300) & (wl < 1000) & np.isfinite(fl)
    cal_spec = (wl[m], fl[m])
    got = K.get_reference("GAIAXP_CORR_LAMOST", obj, st["ra"], st["dec"], 1.34, 0.90)
    if got is None:
        return []
    tbl, coords, _ = got

    random.seed(20260915)
    byf = {}
    for fr in files:
        byf.setdefault(fr.split("/")[-2], []).append(fr)
    use = []
    for f, fl_ in byf.items():
        use += random.sample(sorted(fl_), min(CAP, len(fl_)))

    prepped, need = [], set()
    for fr in use:
        cp = CATD / (Path(fr).name[:-5] + "_cat.fits")
        if not cp.exists():
            continue
        try:
            with fits.open(cp) as h:
                hdr = h[0].header
                fx = h["LDAC_OBJECTS"].data
                flux_max = np.asarray(fx["FLUX_MAX"], float) if "FLUX_MAX" in fx.columns.names else None
            df = OZ.load_ez(cp)
        except Exception:
            continue
        filt = fr.split("/")[-2]
        if filt not in TW.GRID:
            continue
        sel = OT.select_stars_like(df)
        if len(sel) < MIN_REF:
            continue
        src = SkyCoord(sel.ALPHA_J2000.to_numpy(), sel.DELTA_J2000.to_numpy(), unit="deg")
        idx, d2d, _ = match_coordinates_sky(src, coords)
        ok = np.asarray(d2d.arcsec < 2.0)
        if ok.sum() < MIN_REF:
            continue
        rid = np.asarray(tbl[idx[ok]]["id"]).astype(str)
        di = sel[ok]
        mjd = float(hdr.get("MJD", np.nan))
        dt = (mjd - 51544.5) / 365.25 if np.isfinite(mjd) else 24.5
        ra_s = st["ra"] + st["pmra"] * dt / 3.6e6 / np.cos(np.deg2rad(st["dec"]))
        dec_s = st["dec"] + st["pmdec"] * dt / 3.6e6
        call = SkyCoord(df.ALPHA_J2000.to_numpy(), df.DELTA_J2000.to_numpy(), unit="deg")
        sep = call.separation(SkyCoord(ra_s, dec_s, unit="deg")).arcsec
        j = int(np.argmin(sep))
        srow = df.iloc[j]
        sat = float(hdr.get("SATURATE", np.nan))
        fm = flux_max[j] if flux_max is not None else np.nan
        if not (sep[j] < 1.5 and srow.FLAGS <= 1 and srow.MAGERR_APER_3 < 0.05
                and np.isfinite(fm) and fm < 45000.0 and (not np.isfinite(sat) or fm < sat)):
            continue
        prepped.append(dict(filt=filt, rid=rid, inst=di.MAG_APER_3.to_numpy(),
                            snr=di.SNR_APER_3.to_numpy(), star_inst=float(srow.MAG_APER_3)))
        need.update(rid)
    if not prepped:
        return []

    xp = TW.tile_xp_radec(st["ra"], st["dec"], need)
    sc = SigmaClip(sigma=3.0, maxiters=5)
    rows = []
    for pr in prepped:
        keep = np.array([r in xp for r in pr["rid"]])
        zsel0 = keep & np.array(pr["snr"] > 40.0)
        if zsel0.sum() < 10:
            continue
        fc = np.vstack([xp[r] for r in pr["rid"][keep]])
        inst = pr["inst"][keep]
        bright = (pr["snr"][keep] > 40.0)
        fw, ft = TW._PHYS[pr["filt"]]
        # frame-quality gate at delta=0 (same as stage 2): clipped ZP scatter of bright stars <= 0.08
        zp0 = (R.syn_mag(TW.XP_WAVE, fc, fw, ft, 0.0) - inst)[bright]
        fin0 = np.isfinite(zp0)
        if fin0.sum() < 10:
            continue
        cl0 = sc(zp0[fin0])
        if not np.isfinite(float(np.ma.median(cl0))) or float(np.ma.std(cl0)) > 0.08:
            continue
        row = dict(object=obj, filter=pr["filt"])
        okrow = True
        for dd in DGRID:
            syn = R.syn_mag(TW.XP_WAVE, fc, fw, ft, dd)
            zp_arr = (syn - inst)[bright]
            fin = np.isfinite(zp_arr)
            if fin.sum() < 10:
                okrow = False
                break
            zp = float(np.ma.median(sc(zp_arr[fin])))
            m_true = float(R.syn_mag(cal_spec[0], cal_spec[1], fw, ft, dd))
            row[f"d{dd:+.1f}"] = round(pr["star_inst"] + zp - m_true, 4)
        if okrow:
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n-proc", type=int, default=13)
    a = ap.parse_args()
    meta = pd.read_csv(OUT / "photometry_meta.csv")
    meta = meta[meta.status == "ok"]
    byobj = [(o, sorted(g.file)) for o, g in meta.groupby("object")]
    log(f"{len(byobj)} objects, cap {CAP}/(object,filter), grid {DGRID[0]}..{DGRID[-1]} step 1.5")
    rows = []
    with Pool(min(a.n_proc, len(byobj)), initializer=_init) as pool:
        for r in pool.imap_unordered(process_object, byobj):
            rows += r
            log(f"  +{len(r)} frames (total {len(rows)})")
    fr = pd.DataFrame(rows)
    fr.to_csv(OUT / "calspec_offset_frames.csv", index=False)

    dcols = [f"d{dd:+.1f}" for dd in DGRID]
    res = []
    for f, g in fr.groupby("filter"):
        per = g.groupby("object")[dcols].median()          # star medians per delta
        per = per[g.groupby("object").size() >= 5]
        if len(per) < 3:
            continue
        med = per.median(axis=0).to_numpy()                # median over stars
        mad = 1.4826 * (per - per.median(axis=0)).abs().median(axis=0).to_numpy()
        p = np.polyfit(DGRID, med, 2)                      # mild curvature allowed
        roots = np.roots(p)
        roots = roots[np.isreal(roots)].real
        dstar = float(roots[np.argmin(np.abs(roots))]) if len(roots) else np.nan
        slope = float(np.polyval(np.polyder(p), 0.0))
        err = float(np.median(mad) / np.sqrt(len(per)) / max(abs(slope), 1e-6))
        res.append(dict(filter=f, pivot=int(f[1:4]), n_stars=len(per), n_frames=int(len(g)),
                        d0=round(float(med[np.argmin(np.abs(DGRID))]), 4),
                        slope=round(slope, 5), dstar_calspec=round(dstar, 2), err=round(err, 2)))
    T = pd.DataFrame(res).sort_values("pivot")
    ons = pd.read_csv(TW.OFFCSV).set_index("filter")["offset_nm"]
    lab = pd.read_csv(TW.LABCSV).set_index("filter")["lab"]
    T["onsky_F44"] = [round(float(ons.get(f, np.nan)), 2) for f in T["filter"]]
    T.loc[T["filter"] == "m400", "onsky_F44"] = 3.65       # 2024-copy value for these frames
    T["lab"] = [float(lab.get(f, np.nan)) for f in T["filter"]]
    T.to_csv(OUT / "calspec_offset_needed.csv", index=False)
    print(T.to_string(index=False))

    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    good = T[np.abs(T.slope) > 0.002]
    poor = T[np.abs(T.slope) <= 0.002]
    ax.errorbar(good["pivot"], good.dstar_calspec, yerr=good.err, fmt="o", ms=8, capsize=3,
                color="#9467bd", label="CALSPEC-absolute offset (|slope|>2 mmag/nm)")
    ax.errorbar(poor["pivot"], poor.dstar_calspec, yerr=poor.err, fmt="o", ms=8, mfc="none", capsize=3,
                color="#9467bd", alpha=0.5, label="low leverage")
    ax.plot(T["pivot"], T.onsky_F44, "s--", color="#d62728", ms=6, label="on-sky colour-term route (F44; m400=2024 copy)")
    ax.plot(T["pivot"], T["lab"], "^:", color="#1f77b4", ms=6, label="lab")
    ax.axhline(0, color="k", lw=0.6); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    ax.set(xlabel="pivot [nm]", ylabel="required curve offset [nm] (redward +)", ylim=(-8, 8),
           title="F54  offset minimizing the CALSPEC magnitude difference (no colour term)\n"
                 "vs the colour-term route and lab — three independent measurements")
    for _, s in T.iterrows():
        ax.annotate(s["filter"], (s["pivot"], s.dstar_calspec), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "F54_calspec_offset.png", dpi=145)
    log(f"saved -> calspec_offset_needed.csv, calspec_offset_frames.csv, F54 in {OUT}")


if __name__ == "__main__":
    main()
