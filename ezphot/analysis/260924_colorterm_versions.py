#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260924_colorterm_versions.py — field colour term (corrected XP) of the survey archive re-solved with the
new transmission-curve set.  Same frames (catalogs_ezall, 10" MAG_APER_3), same star selection, same solver
(a(delta)=0 root) as offset_timeseries_ezall; only the curve changes.

Curve combos (old-generation GRID filters):
  phys_orig            : total_curves_physical/<f>.csv  (the curve of the 2024-2026 analysis; regression check)
  manufacturer_beam    : manufacturer x beam(n_eff 1.5) x QE x optics x atm   (should ~= phys_orig)
  manufacturer_nobeam  : manufacturer x QE x optics x atm
  version_mhj_beam     : (manufacturer + lab offset) x beam x QE x optics x atm
  version_mhj_nobeam   : (manufacturer + lab offset) x QE x optics x atm

Output: colorterm_versions_frames.csv (one row per frame x combo: a0 = colour term at delta=0, dstar)

  MPLCONFIGDIR=/tmp/ezphot-matplotlib OMP_NUM_THREADS=1 /home/hhchoi1022/miniconda3/envs/too/bin/python -u \\
      ezphot/analysis/260924_colorterm_versions.py --per-filter 800 --n-proc 32
"""
from __future__ import annotations

import argparse
import importlib
import os
import random
import sys
import time
import warnings
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np                  # noqa: E402
import pandas as pd                 # noqa: E402

warnings.filterwarnings("ignore")
from astropy.coordinates import SkyCoord, match_coordinates_sky   # noqa: E402

sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
sys.path.insert(0, "/home/hhchoi1022/ezphot/log/20260910_transmission_sources")
from dr500_transmission import dr500_transmission   # noqa: E402
OT = importlib.import_module("260911_offset_timeseries")
OZ = importlib.import_module("260911_offset_ezphot")
C, K, R = OT.C, OT.K, OT.R
GRID = list(OT.GRID)

EZALL_CSV = Path("/home/hhchoi1022/ezphot/log/20260911_offset_timeseries/offset_timeseries_ezall.csv")
CATD = Path("/home/hhchoi1022/ezphot/log/20260911_ezphot_photometry/catalogs_ezall")
VBASE = Path("/home/hhchoi1022/ezphot/data/transmission_versions")
SRC = Path("/home/hhchoi1022/ezphot/config/common/transmission/Original_sources")
TS = Path("/home/hhchoi1022/ezphot/log/20260910_transmission_sources")
OUT = Path("/home/hhchoi1022/ezphot/log/20260924_colorterm_versions")
OUT.mkdir(exist_ok=True)
COMBOS = ["phys_orig", "manufacturer_beam", "manufacturer_nobeam", "version_mhj_beam", "version_mhj_nobeam"]

_CURVES = None      # combo -> {filter: (lam, total)}


def log(m):
    print(time.strftime("[%H:%M:%S] ") + m, flush=True)


def build_curves():
    atm = pd.read_csv(TS / "atmosphere_elsauce.csv")
    qe = pd.read_csv(SRC / "QE.csv")
    opt = pd.read_csv(SRC / "optics.efficiency.dr350.dr500.csv", sep=r"\s+", comment="#", skiprows=11)
    atr = atm["trans"] if "trans" in atm.columns else atm.iloc[:, 1]
    out = {c: {} for c in COMBOS}
    for f in GRID:
        out["phys_orig"][f] = C.load_new_curve(f)
        for ver in ("manufacturer", "version_mhj"):
            d = pd.read_csv(VBASE / ver / f"{f}.csv")
            lam = d.lam_nm.to_numpy(float); tr = d.trans.to_numpy(float)
            env = (np.interp(lam, qe.iloc[:, 0], qe.iloc[:, 1]) * np.interp(lam, opt.iloc[:, 0], opt.iloc[:, -1])
                   * np.interp(lam, atm["lam_air"], atr))
            out[f"{ver}_nobeam"][f] = (lam, tr * env)
            te = dr500_transmission(lam, tr, n_eff=1.5, f_number=3.03, obscuration=0.59)
            out[f"{ver}_beam"][f] = (lam, te * env)
    return out


def _init():
    global _CURVES
    warnings.filterwarnings("ignore")
    K._refit_init()
    _CURVES = build_curves()


def process_tile(args):
    tile, frames = args            # frames: list of (file, filter, unit, night, mjd)
    matched, need = [], set()
    for fr, filt, unit, night, mjd in frames:
        cp = CATD / (Path(fr).name[:-5] + "_cat.fits")
        try:
            d = OT.select_stars_like(OZ.load_ez(cp))
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
            matched.append((filt, fr, unit, night, mjd, rid[k], bprp[k],
                            float(di.MAG_APER_3.iloc[k]), float(di.MAGERR_APER_3.iloc[k])))
            need.add(rid[k])
    if not matched:
        return []
    xp = C.tile_xp(tile, need)
    M = pd.DataFrame(matched, columns=["filter", "file", "unit", "night", "mjd", "ref_id", "bprp", "inst", "ierr"])
    M = M[M.ref_id.isin(xp)]
    M["sig"] = np.sqrt(M.ierr.to_numpy() ** 2 + 0.01 ** 2)
    out = []
    for (filt, fr), g in M.groupby(["filter", "file"]):
        if len(g) < 15:
            continue
        fc = np.vstack([xp[i][1] for i in g.ref_id])
        base = dict(filter=filt, file=fr, unit=g.unit.iloc[0], night=g.night.iloc[0],
                    mjd=g.mjd.iloc[0], tile=tile, n=len(g))
        for combo in COMBOS:
            dstar, derr, a0 = OT.solve_offset(*_CURVES[combo][filt], fc, g.inst.to_numpy(),
                                              g.bprp.to_numpy(), g.sig.to_numpy())
            if not np.isfinite(a0):
                continue
            out.append(dict(base, combo=combo, a0=round(float(a0), 5),
                            dstar=round(float(dstar), 3) if np.isfinite(dstar) else np.nan,
                            dstar_err=round(float(derr), 3) if np.isfinite(derr) else np.nan))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-filter", type=int, default=800, help="frames per filter (spread over nights)")
    ap.add_argument("--n-proc", type=int, default=32)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    ez = pd.read_csv(EZALL_CSV)
    ez = ez[ez["filter"].isin(GRID) & ez.file.notna()]
    ez = ez[[(CATD / (Path(f).name[:-5] + "_cat.fits")).exists() for f in ez.file]]
    random.seed(20260924)
    sample = []
    for filt, g in ez.groupby("filter"):
        g = g.sort_values("mjd")
        k = min(a.per_filter, len(g))
        sample.append(g.iloc[np.linspace(0, len(g) - 1, k).astype(int)])   # even spread in time
    S = pd.concat(sample, ignore_index=True)
    by_tile = defaultdict(list)
    for r in S.itertuples():
        by_tile[r.tile].append((r.file, r.filter, r.unit, r.night, r.mjd))
    items = sorted(by_tile.items(), key=lambda kv: -len(kv[1]))
    log(f"{len(S)} frames ({len(ez)} available), {len(items)} tiles, {a.n_proc} workers, combos {COMBOS}")
    rows = []
    t0 = time.time()
    with Pool(a.n_proc, initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(process_tile, items, chunksize=1), 1):
            rows += r
            if i % 100 == 0 or i == len(items):
                log(f"  {i}/{len(items)} tiles, {len(rows)} rows, {(time.time()-t0)/60:.1f} min")
                pd.DataFrame(rows).to_csv(OUT / f"colorterm_versions_frames{a.tag}.partial.csv", index=False)
    fr = pd.DataFrame(rows)
    fr.to_csv(OUT / f"colorterm_versions_frames{a.tag}.csv", index=False)
    summ = (fr.groupby(["filter", "combo"])
              .agg(pivot=("filter", lambda s: int(s.iloc[0][1:4])), n=("a0", "size"),
                   a_median=("a0", "median"), a_mad=("a0", lambda v: 1.4826 * np.median(np.abs(v - v.median()))),
                   dstar_median=("dstar", "median"))
              .reset_index().sort_values(["pivot", "combo"]))
    summ.to_csv(OUT / f"colorterm_versions_summary{a.tag}.csv", index=False)
    print(summ.to_string(index=False))
    log("saved -> colorterm_versions_frames.csv, colorterm_versions_summary.csv")


if __name__ == "__main__":
    main()
