#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260911_m400_timeseries_parallel.py

m400 field colour term time series from 2025-01 onward, ALL units, three curves (nominal / physical /
physical+lab-delta_f, corrected XP).  Parallel over tiles (per-worker get_reference/tile_xp cache).

  MPLCONFIGDIR=/tmp/ezphot-matplotlib OMP_NUM_THREADS=1 /home/hhchoi1022/miniconda3/envs/too/bin/python -u \\
      ezphot/analysis/260911_m400_timeseries_parallel.py --per-night 12 --n-proc 24
"""
from __future__ import annotations

import argparse
import importlib
import os
import random
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402
import pandas as pd                 # noqa: E402
from astropy.coordinates import SkyCoord, match_coordinates_sky   # noqa: E402
from astropy.stats import SigmaClip                                # noqa: E402
from astropy.time import Time                                      # noqa: E402
from multiprocessing import Pool                                   # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
C = importlib.import_module("260911_colorterm_2026")
L = importlib.import_module("260911_colorterm_labdelta")
R, K = C.R, C.K
OUT, TS, XP_WAVE = C.OUT, C.TS, C.XP_WAVE
FILT = "m400"; LAB = L.LAB_DELTA[FILT]     # overridden in main() before the Pool fork
_CURVES = None


def _init():
    global _CURVES
    warnings.filterwarnings("ignore")
    K._refit_init()
    ph = C.load_new_curve(FILT)
    _CURVES = {"nominal": R.load_curve(FILT), "physical": ph, "physlab": L.shift_curve(ph[0], ph[1], LAB)}


def frame_meta(fr):
    m = re.search(r"_(\d{8})_(\d{6})_", fr); u = re.search(r"(7DT\d\d)", fr)
    n = re.search(r"/(20\d\d-\d\d-\d\d)/", fr)
    mjd = np.nan
    if m:
        s = m.group(1); t = m.group(2)
        mjd = Time(f"{s[:4]}-{s[4:6]}-{s[6:]}T{t[:2]}:{t[2:4]}:{t[4:]}").mjd
    return (u.group(1) if u else "?", n.group(1) if n else "?", float(mjd))


def process_tile(args):
    tile, frames = args
    matched, need = [], set()
    for fr in frames:
        try:
            d = C.read_cat(fr[:-5] + "_cat.fits")
        except Exception:
            continue
        q = (d.CLASS_STAR >= 0.8) & (d.FLAGS <= 0) & (d.MAGERR_APER_3 < 0.1) & (d.SNR_APER_3 > 10) & (d.SNR_APER_3 < 150)
        d = d[q]
        if len(d) < 15:
            continue
        ra0, dec0 = float(np.median(d.ALPHA_J2000)), float(np.median(d.DELTA_J2000))
        got = K.get_reference("GAIAXP_CORR_LAMOST", tile, ra0, dec0, 1.34, 0.90)
        if got is None:
            continue
        tbl, coords, _ = got
        magcol, _ = K.resolve_mag_column(FILT, tbl.colnames, "GAIAXP_CORR_LAMOST")
        if magcol is None:
            continue
        src = SkyCoord(d.ALPHA_J2000.to_numpy(), d.DELTA_J2000.to_numpy(), unit="deg")
        idx, d2d, _ = match_coordinates_sky(src, coords)
        ok = np.asarray(d2d.arcsec < 2.0)
        if ok.sum() < 15:
            continue
        mr = tbl[idx[ok]]; di = d[ok]
        rid = np.asarray(mr["id"]).astype(str)
        bprp = np.asarray(mr["bp-rp"], float) if "bp-rp" in mr.colnames else np.full(len(mr), np.nan)
        for k in range(len(di)):
            matched.append((fr, rid[k], bprp[k], float(di.MAG_APER_3.iloc[k]), float(di.MAGERR_APER_3.iloc[k])))
            need.add(rid[k])
    if not matched:
        return []
    xp = C.tile_xp(tile, need)
    M = pd.DataFrame(matched, columns=["file", "ref_id", "bprp", "inst", "ierr"])
    M = M[M.ref_id.isin(xp)]
    if not len(M):
        return []
    M["sig"] = np.sqrt(M.ierr.to_numpy() ** 2 + 0.01 ** 2)
    out = []
    for fr, g in M.groupby("file"):
        if len(g) < 15:
            continue
        fc = np.array([xp[i][1] for i in g.ref_id])
        inst = g.inst.to_numpy(); bp = g.bprp.to_numpy(); sig = g.sig.to_numpy()
        unit, night, mjd = frame_meta(fr)
        rec = dict(file=Path(fr).name, tile=tile, unit=unit, night=night, mjd=mjd, n=len(g))
        for cn, cv in _CURVES.items():
            zp = R.syn_mag(XP_WAVE, fc, cv[0], cv[1], 0.0) - inst
            fin = np.isfinite(zp) & np.isfinite(bp) & np.isfinite(sig)
            if fin.sum() < 15:
                rec[f"a_{cn}"] = np.nan; continue
            cl = SigmaClip(sigma=3.0, maxiters=5)(zp[fin])
            keep = np.where(fin)[0][~np.ma.getmaskarray(cl)]
            if cn == "nominal":
                rec["zp_med"] = float(np.nanmedian(zp[keep]))
            aa, ae, bb, nn = R.weighted_linfit(bp[keep], zp[keep] - np.nanmedian(zp[keep]), sig[keep])
            rec[f"a_{cn}"] = aa
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", default="m400"); ap.add_argument("--per-night", type=int, default=12)
    ap.add_argument("--n-proc", type=int, default=24)
    a = ap.parse_args()
    global FILT, LAB
    FILT = a.filter; LAB = L.LAB_DELTA[FILT]        # set before Pool() so forked workers inherit
    frames = [l.strip() for l in (OUT / f"{FILT}_2025_raw.txt").read_text().splitlines() if l.strip()]
    by_night = defaultdict(list)
    for f in frames:
        n = re.search(r"/(20\d\d-\d\d-\d\d)/", f)
        if n:
            by_night[n.group(1)].append(f)
    random.seed(20260911)
    sample = []
    for night, fl in by_night.items():
        sample += random.sample(fl, min(a.per_night, len(fl)))
    by_tile = defaultdict(list)
    for f in sample:
        by_tile[f.split("/")[-4]].append(f)
    items = sorted(by_tile.items())
    print(f"{len(sample)} m400 frames, {len(by_night)} nights, {len(items)} tiles; {a.n_proc} workers")

    rows = []
    with Pool(a.n_proc, initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(process_tile, items, chunksize=2), 1):
            rows += r
            if i % 200 == 0:
                print(f"  {i}/{len(items)} tiles, {len(rows)} results", flush=True)
    res = pd.DataFrame(rows).sort_values("mjd")
    res.to_csv(OUT / f"{FILT}_timeseries_allunits.csv", index=False)
    print(f"\n{len(res)} {FILT} colour terms; nights {res.night.nunique()}; units {sorted(res.unit.unique())}")
    print("frames per unit:", res.unit.value_counts().to_dict())
    print("MJD range %.0f .. %.0f (%s .. %s)" % (res.mjd.min(), res.mjd.max(), res.night.min(), res.night.max()))

    # ---- figure: colour term vs time, per unit; + overall nightly median ----
    units = sorted(res.unit.unique())
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    # nominal / physical / physlab as three panels, unit-coloured nightly medians
    for pi, cn in enumerate(["nominal", "physical", "physlab"]):
        for ui, u in enumerate(units):
            s = res[res.unit == u]
            ng = s.groupby("night").agg(mjd=("mjd", "median"), a=(f"a_{cn}", "median"))
            ax[pi].plot(ng.mjd, ng.a, "o-", ms=3, lw=0.8, color=cmap(ui % 20), alpha=0.8, label=u if pi == 0 else None)
        ov = res.groupby("night").agg(mjd=("mjd", "median"), a=(f"a_{cn}", "median"))
        ax[pi].plot(ov.mjd, ov.a, "k-", lw=1.5, alpha=0.5)
        ax[pi].axhline(0, color="k", lw=0.5)
        ax[pi].set(ylabel=f"{FILT} a(BP-RP) [{cn}]"); ax[pi].grid(alpha=0.3)
    ax[0].legend(ncol=min(len(units), 10), fontsize=7, loc="upper right")
    ax[0].set_title(f"{FILT} colour term over time (2025-01+), all units, {len(res)} frames — nominal / physical / physical+lab")
    ax[2].set_xlabel("MJD")
    figname = f"F22_{FILT}_timeseries_allunits.png"
    fig.tight_layout(); fig.savefig(TS / "figures" / figname, dpi=150)
    print(f"saved -> {FILT}_timeseries_allunits.csv, figures/{figname}")


if __name__ == "__main__":
    main()
