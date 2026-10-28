#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260911_colorterm_timeseries_all.py

Colour-term time series (2025-01+) for ALL GRID medium bands at once, three curves (nominal / physical /
physical+lab-delta_f, corrected XP), per unit.  Efficient: parallel over TILES, and each tile loads the
GAIAXP_CORR_LAMOST archive + XP once and processes every filter's frames of that tile (amortises the NFS I/O
across filters).  Read-only on /lyman/data2/processed.  All outputs in one folder.

  MPLCONFIGDIR=/tmp/ezphot-matplotlib OMP_NUM_THREADS=1 /home/hhchoi1022/miniconda3/envs/too/bin/python -u \\
      ezphot/analysis/260911_colorterm_timeseries_all.py --per-night 6 --n-proc 64
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
XP_WAVE = C.XP_WAVE
FOLDER = Path("/home/hhchoi1022/ezphot/log/20260911_colorterm_timeseries")
GRID = R.GRID                                   # 20 bands m400..m875
_CURVES = None


FITDF_CSV = "/home/hhchoi1022/ezphot/log/20260910_effective_curves/offset_plus_linear_summary.csv"
CURVE_NAMES = ("nominal", "physical", "physlab", "physfit")


def _init():
    global _CURVES
    warnings.filterwarnings("ignore")
    K._refit_init()
    fit = pd.read_csv(FITDF_CSV).set_index("filter")   # on-sky fitted intrinsic offset delta_f ('offset' col)
    _CURVES = {}
    for f in GRID:
        ph = C.load_new_curve(f)
        d = {"nominal": R.load_curve(f), "physical": ph, "physlab": L.shift_curve(ph[0], ph[1], L.LAB_DELTA[f])}
        off = float(fit.loc[f, "offset"]) if f in fit.index and np.isfinite(fit.loc[f, "offset"]) else 0.0
        d["physfit"] = L.shift_curve(ph[0], ph[1], off)
        _CURVES[f] = d


def meta(fr):
    m = re.search(r"_(\d{8})_(\d{6})_", fr); u = re.search(r"(7DT\d\d)", fr); n = re.search(r"/(20\d\d-\d\d-\d\d)/", fr)
    mjd = np.nan
    if m:
        s, t = m.group(1), m.group(2)
        mjd = Time(f"{s[:4]}-{s[4:6]}-{s[6:]}T{t[:2]}:{t[2:4]}:{t[4:]}").mjd
    return (u.group(1) if u else "?", n.group(1) if n else "?", float(mjd))


def process_tile(args):
    tile, frames = args
    matched = []      # (filter, file, ref_id, bprp, inst, ierr)
    need = set()
    for fr in frames:
        filt = fr.split("/")[-3]
        if filt not in GRID:
            continue
        try:
            d = C.read_cat(fr[:-5] + "_cat.fits")
        except Exception:
            continue
        q = (d.CLASS_STAR >= 0.8) & (d.FLAGS <= 0) & (d.MAGERR_APER_3 < 0.1) & (d.SNR_APER_3 > 10) & (d.SNR_APER_3 < 150)
        d = d[q]
        if len(d) < 15:
            continue
        ra0, dec0 = float(np.median(d.ALPHA_J2000)), float(np.median(d.DELTA_J2000))
        got = K.get_reference("GAIAXP_CORR_LAMOST", tile, ra0, dec0, 1.34, 0.90)   # cached per tile
        if got is None:
            continue
        tbl, coords, _ = got
        magcol, _ = K.resolve_mag_column(filt, tbl.colnames, "GAIAXP_CORR_LAMOST")
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
            matched.append((filt, fr, rid[k], bprp[k], float(di.MAG_APER_3.iloc[k]), float(di.MAGERR_APER_3.iloc[k])))
            need.add(rid[k])
    if not matched:
        return []
    xp = C.tile_xp(tile, need)
    M = pd.DataFrame(matched, columns=["filter", "file", "ref_id", "bprp", "inst", "ierr"])
    M = M[M.ref_id.isin(xp)]
    if not len(M):
        return []
    M["sig"] = np.sqrt(M.ierr.to_numpy() ** 2 + 0.01 ** 2)
    out = []
    for (filt, fr), g in M.groupby(["filter", "file"]):
        if len(g) < 15:
            continue
        fc = np.array([xp[i][1] for i in g.ref_id])
        inst = g.inst.to_numpy(); bp = g.bprp.to_numpy(); sig = g.sig.to_numpy()
        unit, night, mjd = meta(fr)
        rec = dict(filter=filt, unit=unit, night=night, mjd=mjd, tile=tile, n=len(g))
        for cn in CURVE_NAMES:
            fw, ft = _CURVES[filt][cn]
            zp = R.syn_mag(XP_WAVE, fc, fw, ft, 0.0) - inst
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
    ap.add_argument("--per-night", type=int, default=6); ap.add_argument("--n-proc", type=int, default=64)
    a = ap.parse_args()
    frames = [l.strip() for l in (FOLDER / "all_medium_2025.txt").read_text().splitlines() if l.strip()]
    frames = [f for f in frames if f.split("/")[-3] in GRID]
    # sample per (filter, night)
    by_fn = defaultdict(list)
    for f in frames:
        by_fn[(f.split("/")[-3], re.search(r"/(20\d\d-\d\d-\d\d)/", f).group(1))].append(f)
    random.seed(20260911)
    sample = []
    for k, fl in by_fn.items():
        sample += random.sample(fl, min(a.per_night, len(fl)))
    by_tile = defaultdict(list)
    for f in sample:
        by_tile[f.split("/")[-4]].append(f)
    items = sorted(by_tile.items())
    print(f"{len(sample)} frames, {len(GRID)} filters, {len(items)} tiles, {a.n_proc} workers", flush=True)

    rows = []
    with Pool(a.n_proc, initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(process_tile, items, chunksize=2), 1):
            rows += r
            if i % 400 == 0:
                print(f"  {i}/{len(items)} tiles, {len(rows)} results", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(FOLDER / "colorterm_timeseries_all.csv", index=False)
    print(f"\n{len(res)} colour terms; filters {res['filter'].nunique()}; MJD {res['mjd'].min():.0f}..{res['mjd'].max():.0f}", flush=True)

    # ---- per-filter figure (4 curves) + summary ----
    ccol = {"nominal": "#1f77b4", "physical": "#d62728", "physlab": "#2ca02c", "physfit": "#9467bd"}
    summ = []
    for filt in sorted(res["filter"].unique(), key=lambda f: int(f[1:4])):
        s = res[res["filter"] == filt].sort_values("mjd")
        units = sorted(s["unit"].unique())
        fig, ax = plt.subplots(4, 1, figsize=(15, 11), sharex=True)
        for pi, cn in enumerate(CURVE_NAMES):
            lo, hi = np.nanpercentile(s[f"a_{cn}"], [2, 98])
            ng = s.groupby("night").agg(mjd=("mjd", "median"), a=(f"a_{cn}", "median"))
            ax[pi].scatter(s["mjd"], s[f"a_{cn}"], s=4, color=ccol[cn], alpha=0.15)
            ax[pi].plot(ng["mjd"], ng["a"], "o-", ms=2.5, lw=0.7, color=ccol[cn])
            ax[pi].axhline(0, color="k", lw=0.5); ax[pi].set(ylabel=f"a [{cn}]", ylim=(min(lo, -0.02) - 0.02, max(hi, 0.05) + 0.02))
            ax[pi].grid(alpha=0.3)
        ax[0].set_title(f"{filt} colour term over time (2025-01+), {len(s)} frames, units {units} — nominal/physical/physlab/physfit")
        ax[3].set_xlabel("MJD")
        fig.tight_layout(); fig.savefig(FOLDER / "figures" / f"{filt}_timeseries.png", dpi=130); plt.close(fig)
        for u in units:
            su = s[s["unit"] == u]
            summ.append(dict(filter=filt, pivot=int(filt[1:4]), unit=u, n=len(su),
                             night_min=su["night"].min(), night_max=su["night"].max(),
                             a_nominal=round(su["a_nominal"].median(), 4), a_physical=round(su["a_physical"].median(), 4),
                             a_physlab=round(su["a_physlab"].median(), 4), a_physfit=round(su["a_physfit"].median(), 4),
                             zp=round(su["zp_med"].median(), 3)))
    S = pd.DataFrame(summ)
    S.to_csv(FOLDER / "colorterm_timeseries_summary.csv", index=False)
    # per-filter median |a| for the 4 curves (which curve wins)
    med = res.groupby("filter").agg(pivot=("mjd", lambda x: int(res.loc[x.index, "filter"].iloc[0][1:4])))
    for cn in CURVE_NAMES:
        med[cn] = res.groupby("filter")[f"a_{cn}"].median()
    med = med.sort_values("pivot")
    med.to_csv(FOLDER / "colorterm_timeseries_curvecompare.csv")
    print("\n=== per-filter median colour term by curve ===")
    print(med[list(CURVE_NAMES)].round(4).to_string())
    print("\nmedian |a| over filters:  " + "  ".join(f"{cn}={med[cn].abs().median():.4f}" for cn in CURVE_NAMES))
    print(f"\nsaved {res['filter'].nunique()} per-filter figures + all/summary/curvecompare CSVs in {FOLDER}")


if __name__ == "__main__":
    main()
