#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260911_offset_timeseries.py

Instead of measuring the colour term of a FIXED curve, this solves — per frame — for the wavelength offset
delta* that ZEROES the colour term, on top of the physically-predicted curve, and tracks how delta* evolves
over time (2025-01+) per filter/unit.  delta* is "the offset the data wants" at that epoch; drifts reveal filter
swaps / per-copy differences, and its ensemble median should match the global fitted delta_f (physfit).

Method per frame: shift the physical curve by a small grid of delta (syn_mag shift_nm), fit the colour term
a(delta) over a fixed 3-sigma-clipped star set, and root-solve the (near-linear) a(delta)=0 -> delta*.

Read-only on the pipeline.  Parallel over tiles (reference archive + XP loaded once per tile).

  MPLCONFIGDIR=/tmp/ezphot-matplotlib OMP_NUM_THREADS=1 /home/hhchoi1022/miniconda3/envs/too/bin/python -u \\
      ezphot/analysis/260911_offset_timeseries.py --per-night 6 --n-proc 64
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
import astropy.units as u                                          # noqa: E402
from astropy.coordinates import SkyCoord, match_coordinates_sky, search_around_sky   # noqa: E402
from astropy.io import fits                                        # noqa: E402
from astropy.stats import SigmaClip, sigma_clipped_stats           # noqa: E402
from astropy.time import Time                                      # noqa: E402
from multiprocessing import Pool                                   # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
C = importlib.import_module("260911_colorterm_2026")
R, K = C.R, C.K
XP_WAVE = C.XP_WAVE
FOLDER = Path("/home/hhchoi1022/ezphot/log/20260911_offset_timeseries"); FOLDER.mkdir(parents=True, exist_ok=True)
SRC = Path("/home/hhchoi1022/ezphot/log/20260911_colorterm_timeseries/all_medium_2025.txt")
FITDF_CSV = "/home/hhchoi1022/ezphot/log/20260910_effective_curves/offset_plus_linear_summary.csv"
GRID = R.GRID
DGRID = np.arange(-6.0, 6.001, 1.0)                 # nm shift grid applied to the physical curve (1 nm steps)
_PHYS = None
_FITOFF = None


def _init():
    global _PHYS, _FITOFF
    warnings.filterwarnings("ignore")
    K._refit_init()
    _PHYS = {f: C.load_new_curve(f) for f in GRID}                # physically-predicted curve (beam, delta_f=0)
    fit = pd.read_csv(FITDF_CSV).set_index("filter")
    _FITOFF = {f: (float(fit.loc[f, "offset"]) if f in fit.index and np.isfinite(fit.loc[f, "offset"]) else np.nan)
               for f in GRID}


def meta(fr):
    m = re.search(r"_(\d{8})_(\d{6})_", fr); u = re.search(r"(7DT\d\d)", fr); n = re.search(r"/(20\d\d-\d\d-\d\d)/", fr)
    mjd = np.nan
    if m:
        s, t = m.group(1), m.group(2)
        mjd = Time(f"{s[:4]}-{s[4:6]}-{s[6:]}T{t[:2]}:{t[2:4]}:{t[4:]}").mjd
    return (u.group(1) if u else "?", n.group(1) if n else "?", float(mjd))


def read_cat_sel(catpath):
    """Full detection list with the columns needed for the select_stars-aligned cuts."""
    with fits.open(catpath) as h:
        d = h["LDAC_OBJECTS"].data
        cols = {c: np.asarray(d[c], float) for c in
                ["ALPHA_J2000", "DELTA_J2000", "MAG_APER_3", "MAGERR_APER_3", "FLAGS", "CLASS_STAR",
                 "SNR_APER_3", "FWHM_WORLD", "ELONGATION"]}
    return pd.DataFrame(cols)


def select_stars_like(full):
    """ezphot select_stars core cuts (photometriccalibration.py select_stars) minus the inner-region cut:
    quality (CLASS_STAR/FLAGS/SNR/MAGERR) + FWHM 1-15" abs, ELONGATION<1.5, isolation 10" against the FULL
    detection list, then 5-sigma relative clips on FWHM and ELONGATION.  IMAFLAGS_ISO absent in single cats."""
    fwa = full.FWHM_WORLD.to_numpy() * 3600.0
    q = (full.CLASS_STAR.to_numpy() >= 0.8) & (full.FLAGS.to_numpy() <= 0) \
        & (full.MAGERR_APER_3.to_numpy() < 0.1) & (full.SNR_APER_3.to_numpy() > 10) \
        & (full.SNR_APER_3.to_numpy() < 150) & (fwa > 1.0) & (fwa < 15.0) \
        & (full.ELONGATION.to_numpy() < 1.5)
    d = full[q]
    if len(d) < 15:
        return d.iloc[[]]
    allc = SkyCoord(full.ALPHA_J2000.to_numpy(), full.DELTA_J2000.to_numpy(), unit="deg")
    gc = SkyCoord(d.ALPHA_J2000.to_numpy(), d.DELTA_J2000.to_numpy(), unit="deg")
    ig, _, sep2d, _ = search_around_sky(gc, allc, 10.0 * u.arcsec)
    nbr = np.bincount(ig[sep2d.arcsec > 0.5], minlength=len(d))     # exclude the self-match
    d = d[nbr == 0]
    if len(d) < 15:
        return d.iloc[[]]
    fwv = d.FWHM_WORLD.to_numpy() * 3600.0
    _, med, std = sigma_clipped_stats(fwv, sigma=5.0, maxiters=10)
    d = d[np.abs(fwv - med) <= 5.0 * std]
    if len(d) < 15:
        return d.iloc[[]]
    el = d.ELONGATION.to_numpy()
    _, med, std = sigma_clipped_stats(el, sigma=5.0, maxiters=5)
    d = d[np.abs(el - med) <= 5.0 * std]
    return d.reset_index(drop=True)


def solve_offset(fw, ft, fc, inst, bp, sig):
    """(delta*, delta*_err, a(0)): the shift that zeroes the colour term of the physical curve.
    Fixed star set (3-sigma ZP clip at delta=0); LINEAR fit of a(delta) -> root (user choice)."""
    zp0 = R.syn_mag(XP_WAVE, fc, fw, ft, 0.0) - inst
    fin = np.isfinite(zp0) & np.isfinite(bp) & np.isfinite(sig)
    if fin.sum() < 15:
        return np.nan, np.nan, np.nan
    cl = SigmaClip(sigma=3.0, maxiters=5)(zp0[fin])
    keep = np.where(fin)[0][~np.ma.getmaskarray(cl)]
    if len(keep) < 15:
        return np.nan, np.nan, np.nan
    fck, ik, bpk, sk = fc[keep], inst[keep], bp[keep], sig[keep]
    avals, aerr0 = [], np.nan
    for dd in DGRID:
        zp = R.syn_mag(XP_WAVE, fck, fw, ft, dd) - ik
        aa, ae, _, _ = R.weighted_linfit(bpk, zp - np.nanmedian(zp), sk)
        avals.append(aa)
        if dd == 0.0:
            aerr0 = ae
    avals = np.array(avals)
    if not np.all(np.isfinite(avals)):
        return np.nan, np.nan, np.nan
    a0 = float(avals[int(np.argmin(np.abs(DGRID)))])               # a at delta = 0
    m, c = np.polyfit(DGRID, avals, 1)                             # a(delta) ~ m*delta + c
    if abs(m) < 1e-9:
        return np.nan, np.nan, a0
    dstar = float(-c / m)
    derr = float(abs(aerr0 / m)) if np.isfinite(aerr0) else np.nan
    return dstar, derr, a0


def _tile_alarm(signum, frame):
    raise TimeoutError("tile timeout")


def process_tile(args):
    import signal
    signal.signal(signal.SIGALRM, _tile_alarm)
    signal.alarm(900)                                   # straggler guard: skip tiles taking >15 min
    try:
        return _process_tile(args)
    except TimeoutError:
        return []
    finally:
        signal.alarm(0)


def _process_tile(args):
    tile, frames = args
    matched = []
    need = set()
    for fr in frames:
        filt = fr.split("/")[-3]
        if filt not in GRID:
            continue
        try:
            d = select_stars_like(read_cat_sel(fr[:-5] + "_cat.fits"))   # ezphot select_stars-aligned cuts
        except Exception:
            continue
        if len(d) < 15:
            continue
        ra0, dec0 = float(np.median(d.ALPHA_J2000)), float(np.median(d.DELTA_J2000))
        got = K.get_reference("GAIAXP_CORR_LAMOST", tile, ra0, dec0, 1.34, 0.90)
        if got is None:
            continue
        tbl, coords, _ = got
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
        fc = np.vstack([xp[i][1] for i in g.ref_id])
        dstar, derr, a0 = solve_offset(*_PHYS[filt], fc, g.inst.to_numpy(), g.bprp.to_numpy(), g.sig.to_numpy())
        if not np.isfinite(dstar):
            continue
        unit, night, mjd = meta(fr)
        out.append(dict(filter=filt, unit=unit, night=night, mjd=mjd, tile=tile, n=len(g),
                        dstar=dstar, dstar_err=derr, a_phys=a0))
    return out


NEW_FILTERS = ["m375w", "m386", "m438", "m483", "m512", "m534", "m561", "m586", "m615", "m640", "m661", "m769w", "m832w"]


def main():
    global GRID
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-night", type=int, default=6); ap.add_argument("--n-proc", type=int, default=64)
    ap.add_argument("--filters", nargs="*", default=None, help="filter list; 'new' = the 13 non-GRID mediums")
    ap.add_argument("--tag", default="", help="suffix for output filenames")
    ap.add_argument("--src", default=None, help="frame-list file (default: all_medium_2025.txt)")
    a = ap.parse_args()
    if a.filters == ["new"]:
        GRID = NEW_FILTERS
    elif a.filters:
        GRID = a.filters
    print(f"filters ({len(GRID)}): {GRID}", flush=True)
    src = Path(a.src) if a.src else SRC
    frames = [l.strip() for l in src.read_text().splitlines() if l.strip()]
    frames = [f for f in frames if f.split("/")[-3] in GRID]
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
    print(f"{len(sample)} frames, {len(items)} tiles, {a.n_proc} workers", flush=True)

    rows = []
    with Pool(a.n_proc, initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(process_tile, items, chunksize=2), 1):
            rows += r
            if i % 400 == 0:
                print(f"  {i}/{len(items)} tiles, {len(rows)} results", flush=True)
                pd.DataFrame(rows).to_csv(FOLDER / f"offset_timeseries{a.tag}.partial.csv", index=False)
    res = pd.DataFrame(rows)
    res.to_csv(FOLDER / f"offset_timeseries{a.tag}.csv", index=False)
    fitoff = pd.read_csv(FITDF_CSV).set_index("filter")["offset"]
    print(f"\n{len(res)} frames; filters {res['filter'].nunique()}; MJD {res['mjd'].min():.0f}..{res['mjd'].max():.0f}", flush=True)

    # ---- per-filter delta* time series ----
    order = sorted(res["filter"].unique(), key=lambda f: int(f[1:4]))
    units_all = sorted(res["unit"].unique())
    cmap = plt.get_cmap("tab20"); ucol = {un: cmap(i % 20) for i, un in enumerate(units_all)}
    ncol = 4; nrow = int(np.ceil(len(order) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 2.7 * nrow), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    summ = []
    for ax, filt in zip(axes, order):
        s = res[res["filter"] == filt].sort_values("mjd")
        for un in sorted(s["unit"].unique()):
            su = s[s["unit"] == un]
            ax.scatter(su["mjd"], su["dstar"], s=8, color=ucol[un], alpha=0.5, label=un)
        gg = s.groupby("night").agg(mjd=("mjd", "median"), d=("dstar", "median"))
        ax.plot(gg["mjd"], gg["d"], "k-", lw=0.8, alpha=0.7)
        go = float(fitoff.get(filt, np.nan))
        if np.isfinite(go):
            ax.axhline(go, color="red", lw=1.2, ls="--", alpha=0.8)
        ax.axhline(0, color="grey", lw=0.5)
        gtxt = f"global δ_f={go:+.2f}" if np.isfinite(go) else "global δ_f 없음(첫 측정)"
        ax.set_title(f"{filt}  ({gtxt})", fontsize=9)
        ax.set_ylabel("δ* [nm]", fontsize=8); ax.grid(alpha=0.3)
        ax.set_ylim(np.nanpercentile(s["dstar"], 2) - 1, np.nanpercentile(s["dstar"], 98) + 1)
        for un in sorted(s["unit"].unique()):
            su = s[s["unit"] == un]
            err_med = float(np.nanmedian(su["dstar_err"]))
            rob = 1.4826 * float(np.median(np.abs(su["dstar"] - su["dstar"].median())))
            summ.append(dict(filter=filt, pivot=int(filt[1:4]), unit=un, n=len(su),
                             dstar_med=round(su["dstar"].median(), 3), dstar_rob=round(rob, 3),
                             err_med=round(err_med, 3),
                             excess=round(float(np.sqrt(max(0.0, rob ** 2 - err_med ** 2))), 3),
                             global_off=round(go, 3)))
    for ax in axes[len(order):]:
        ax.axis("off")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(len(units_all), 11), fontsize=7, frameon=False)
    fig.suptitle("Optimal offset δ* (zeroes colour term on physical curve) vs time — dashed red = global fitted δ_f", y=0.998)
    fig.tight_layout(rect=[0, 0.03, 1, 0.99]); fig.savefig(FOLDER / f"F28{a.tag}_offset_timeseries.png", dpi=130)
    S = pd.DataFrame(summ); S.to_csv(FOLDER / f"offset_timeseries_summary{a.tag}.csv", index=False)

    print("\n=== per-filter δ* (median over frames) vs global fitted δ_f ===")
    med = res.groupby("filter").agg(pivot=("filter", lambda x: int(x.iloc[0][1:4])),
                                    dstar_med=("dstar", "median"),
                                    dstar_mad=("dstar", lambda v: float(np.median(np.abs(v - np.median(v))))),
                                    err_med=("dstar_err", "median"),
                                    n=("dstar", "size")).sort_values("pivot")
    med["global_off"] = [fitoff.get(f, np.nan) for f in med.index]
    print(med.round(3).to_string())
    print(f"\nsaved -> F28{a.tag}_offset_timeseries.png, offset_timeseries{a.tag}.csv, offset_timeseries_summary{a.tag}.csv in {FOLDER}")


if __name__ == "__main__":
    main()
