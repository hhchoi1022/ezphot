#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260919_spss_offset.py — per (filter, unit) curve offset from the SPSS campaign (absolute route).

Same construction as 260915_calspec_offset.py: scan a delta grid; at each delta both the field ZP
(SNR>40, clipped; frame gate zsd<=0.08 at delta=0) and the standard's synthetic magnitude (truth =
SPSS V2 ground flux table) go through the SAME shifted physical curve; the zero of the star-median
Delta(delta) is the required curve offset. 7DT11 excluded (wheel stuck on r, report 20.2).

Output: spss_offset_frames.csv, spss_offset_needed.csv (filter x unit), figure S5 comparing with the
survey colour-term route (F44), the 2024-08 CALSPEC-absolute route (19.8) and lab.

  MPLCONFIGDIR=/tmp/ezphot-matplotlib OMP_NUM_THREADS=1 /home/hhchoi1022/miniconda3/envs/too/bin/python -u \\
      ezphot/analysis/260919_spss_offset.py --n-proc 14
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

sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
OT = importlib.import_module("260911_offset_timeseries")
AN = importlib.import_module("260919_spss_analysis")
R = OT.R
XP_WAVE = OT.XP_WAVE

OUT = Path("/home/hhchoi1022/ezphot/log/20260918_spss")
DGRID = np.arange(-6.0, 6.01, 2.0)
EXCLUDE_UNITS = {"7DT11"}
MIN_REF = 10


def log(m):
    print(time.strftime("[%H:%M:%S] ") + m, flush=True)


def _init():
    AN._init()


def process_object(args):
    obj, files, srow = args
    sc = SigmaClip(sigma=3.0, maxiters=5)
    spec = AN.load_spss_spec(srow["SPSS"])
    if spec is None:
        return []
    ref = AN.gather_reference(srow["RA_ICRS"], srow["DE_ICRS"])
    if ref is None:
        return []
    sid = str(int(srow["GaiaEDR3"])) if pd.notna(srow["GaiaEDR3"]) else ""
    j = np.where(ref["sid"] == sid)[0]
    if not len(j):
        return []
    j = int(j[0])
    ra_s, dec_s = ref["ra"][j], ref["dec"][j]
    pma, pmd = ref["pmra"][j], ref["pmdec"][j]
    coords = SkyCoord(ref["ra"], ref["dec"], unit="deg")
    rows = []
    for fr in files:
        cp = AN.CATD / (Path(fr).name[:-5] + "_cat.fits")
        filt = Path(fr).name.split("_")[-4]
        unit = Path(fr).name.split("_")[0]
        if filt not in AN._PHYS or unit in EXCLUDE_UNITS:
            continue
        try:
            with fits.open(cp) as h:
                hdr = h[0].header
                dd_ = h["LDAC_OBJECTS"].data
                fmax = np.asarray(dd_["FLUX_MAX"], float) if "FLUX_MAX" in dd_.columns.names else None
            df = AN.load_cat(cp)
            mjd = float(hdr.get("MJD", np.nan))
            sel = OT.select_stars_like(df)
            if len(sel) < MIN_REF:
                continue
            src = SkyCoord(sel.ALPHA_J2000.to_numpy(), sel.DELTA_J2000.to_numpy(), unit="deg")
            idx, d2d, _ = match_coordinates_sky(src, coords)
            ok = np.asarray(d2d.arcsec < 2.0)
            if ok.sum() < MIN_REF:
                continue
            mi = idx[ok]
            fc = ref["flux"][mi]
            inst = sel.MAG_APER_3.to_numpy()[ok]
            snr = sel.SNR_APER_3.to_numpy()[ok]
            bright = snr > 40.0
            if bright.sum() < MIN_REF:
                continue
            # standard star
            dt = (mjd - AN.GAIA_EPOCH_MJD) / 365.25 if np.isfinite(mjd) else 10.7
            pos = SkyCoord(ra_s + pma * dt / 3.6e6 / np.cos(np.deg2rad(dec_s)),
                           dec_s + pmd * dt / 3.6e6, unit="deg")
            call = SkyCoord(df.ALPHA_J2000.to_numpy(), df.DELTA_J2000.to_numpy(), unit="deg")
            sep = call.separation(pos).arcsec
            js = int(np.argmin(sep))
            srow_ = df.iloc[js]
            sat = float(hdr.get("SATURATE", np.nan))
            fm = fmax[js] if fmax is not None else np.nan
            if not (sep[js] < 2.0 and srow_.FLAGS <= 1 and srow_.MAGERR_APER_3 < 0.05
                    and (not np.isfinite(sat) or fm < 0.9 * sat)):
                continue
            fw, ft = AN._PHYS[filt]
            zp0 = (R.syn_mag(XP_WAVE, fc, fw, ft, 0.0) - inst)[bright]
            fin0 = np.isfinite(zp0)
            if fin0.sum() < MIN_REF:
                continue
            cl0 = sc(zp0[fin0])
            if not np.isfinite(float(np.ma.median(cl0))) or float(np.ma.std(cl0)) > 0.08:
                continue
            row = dict(object=obj, filter=filt, unit=unit)
            good = True
            for dd in DGRID:
                syn = R.syn_mag(XP_WAVE, fc, fw, ft, dd)
                zp_arr = (syn - inst)[bright]
                fin = np.isfinite(zp_arr)
                if fin.sum() < MIN_REF:
                    good = False
                    break
                zp = float(np.ma.median(sc(zp_arr[fin])))
                m_true = float(R.syn_mag(spec[0], spec[1], fw, ft, dd))
                row[f"d{dd:+.1f}"] = round(float(srow_.MAG_APER_3) + zp - m_true, 4)
            if good:
                rows.append(row)
        except Exception:                                            # noqa: BLE001
            continue
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n-proc", type=int, default=14)
    a = ap.parse_args()
    spss = pd.read_csv(AN.SPSS_CSV)
    srow_of = {AN.norm(s.Name): s for _, s in spss.iterrows()}
    import glob as _g
    meta = pd.concat([pd.read_csv(m) for m in _g.glob(str(OUT / "process_meta_7DT*.csv"))], ignore_index=True)
    meta = meta[meta.status == "ok"]
    jobs = []
    for obj, g in meta.groupby("object"):
        s = srow_of.get(AN.norm(obj))
        if s is not None:
            jobs.append((obj, sorted(g.file), s))
    log(f"{len(jobs)} objects, grid {DGRID[0]}..{DGRID[-1]} step 2")
    rows = []
    with Pool(min(a.n_proc, len(jobs)), initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(process_object, jobs), 1):
            rows += r
            log(f"  [{i}/{len(jobs)}] total {len(rows)}")
    fr = pd.DataFrame(rows)
    fr.to_csv(OUT / "spss_offset_frames.csv", index=False)

    dcols = [f"d{dd:+.1f}" for dd in DGRID]
    res = []
    for (f, u), g in fr.groupby(["filter", "unit"]):
        per = g.groupby("object")[dcols].median()
        per = per[g.groupby("object").size() >= 2]       # >=2: units with 3-frame-per-star scheduling keep all stars
        if len(per) < 3:
            continue
        med = per.median(axis=0).to_numpy()
        mad = 1.4826 * (per - per.median(axis=0)).abs().median(axis=0).to_numpy()
        p = np.polyfit(DGRID, med, 2)
        roots = np.roots(p); roots = roots[np.isreal(roots)].real
        dstar = float(roots[np.argmin(np.abs(roots))]) if len(roots) else np.nan
        slope = float(np.polyval(np.polyder(p), 0.0))
        err = float(np.median(mad) / np.sqrt(len(per)) / max(abs(slope), 1e-6))
        res.append(dict(filter=f, unit=u, pivot=int(f[1:4]) if f[1:4].isdigit() else 999,
                        n_stars=len(per), n_frames=int(len(g)),
                        d0=round(float(med[np.argmin(np.abs(DGRID))]), 4),
                        slope=round(slope, 5), dstar=round(dstar, 2), err=round(err, 2)))
    T = pd.DataFrame(res).sort_values(["pivot", "unit"])
    # previous measurements for comparison
    f44 = pd.read_csv("/home/hhchoi1022/ezphot/log/20260911_offset_timeseries/"
                      "transmission_offset_needed.csv").set_index("filter")
    cal = pd.read_csv("/home/hhchoi1022/ezphot/log/20260914_calspec_ezphot/"
                      "calspec_offset_needed.csv").set_index("filter")
    lab = pd.read_csv("/home/hhchoi1022/ezphot/log/20260910_transmission_sources/"
                      "labdelta_vs_fitted.csv").set_index("filter")["lab"]
    T["f44_colour"] = [round(float(f44["offset_nm"].get(f, np.nan)), 2) for f in T["filter"]]
    T["calspec_abs"] = [round(float(cal["dstar_calspec"].get(f, np.nan)), 2) for f in T["filter"]]
    T["lab"] = [float(lab.get(f, np.nan)) for f in T["filter"]]
    T.to_csv(OUT / "spss_offset_needed.csv", index=False)
    print(T.to_string(index=False))

    fig, ax = plt.subplots(figsize=(13.5, 7))
    good = T[np.abs(T.slope) > 0.004]
    poor = T[np.abs(T.slope) <= 0.004]
    for d, mk, al, lb in [(good, "o", 1.0, "SPSS 2026-09 (per unit, |slope|>4 mmag/nm)"),
                          (poor, "o", 0.4, "low leverage")]:
        ax.errorbar(d["pivot"] + np.random.default_rng(1).uniform(-2, 2, len(d)), d.dstar, yerr=d.err,
                    fmt=mk, ms=7, capsize=3, color="#2ca02c", alpha=al, mfc=("#2ca02c" if al > 0.5 else "none"),
                    label=lb)
    for _, s in good.iterrows():
        ax.annotate(f"{s['filter']}\n{s['unit'][3:]}", (s["pivot"], s.dstar), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=6.5)
    u = T.drop_duplicates("filter")
    ax.plot(u["pivot"], u.f44_colour, "s--", color="#d62728", ms=5, label="survey colour route (F44)")
    ax.plot(u["pivot"], u.calspec_abs, "^:", color="#9467bd", ms=6, label="CALSPEC absolute 2024-08 (19.8)")
    ax.plot(u["pivot"], u["lab"], "v:", color="#1f77b4", ms=5, label="lab")
    ax.axhline(0, color="k", lw=0.6); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    ax.set(xlabel="pivot [nm]", ylabel="required curve offset [nm] (redward +)", ylim=(-8, 8),
           title="S5  SPSS-campaign absolute offsets (2026-09, truth = SPSS ground spectra)\n"
                 "vs previous routes — 7DT11 excluded")
    fig.tight_layout(); fig.savefig(OUT / "S5_offset_compare.png", dpi=145)
    log("saved -> spss_offset_needed.csv, spss_offset_frames.csv, S5_offset_compare.png")


if __name__ == "__main__":
    main()
