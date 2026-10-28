#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260926_colorterm_time.py — per-image corrected-XP field colour term a(delta = 0) on the whole ezall archive for several
transmission-curve versions, with the Step 1 star selection (260926_delta_star_v2.select_stars_v2: SNR_10" >= 100,
FLUX_MAX <= 50k, FLAGS 0, isolated 10", no shape cuts) and vacuum curves (atmosphere on its lam_air column).

Curve versions (filter x [beam] x QE x optics x atmosphere, vacuum):
  manufacturer_nobeam, manufacturer_beam, mhj_nobeam, mhj_beam      (mhj = manufacturer + lab offset, rigid shift)
Colour term: same as OT.solve_offset at delta = 0 (3-sigma ZP clip, weighted line ZP vs BP-RP), per version.
Output (~/ezphot/log/20260926_colorterm_time/): colorterm_time_frames.csv (filter, file, unit, night, mjd, tile, n,
bprp_lo, bprp_hi, variant, a0, a0_err).  Resume from the .partial.csv; 900 s per-tile guard.

  MPLCONFIGDIR=/tmp/ezphot-matplotlib OMP_NUM_THREADS=1 python -u ezphot/analysis/260926_colorterm_time.py --n-proc 40
"""
import argparse
import importlib
import sys
import time
import warnings
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from astropy.coordinates import SkyCoord, match_coordinates_sky   # noqa: E402
from astropy.stats import SigmaClip                              # noqa: E402

sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
DS = importlib.import_module("260926_delta_star_v2")
C, K, R = DS.C, DS.K, DS.R
OUT = Path("/home/hhchoi1022/ezphot/log/20260926_colorterm_time")
VARIANTS = ["manufacturer_nobeam", "manufacturer_beam", "mhj_nobeam", "mhj_beam"]
_CURVES = None


def build_curves():
    atm = pd.read_csv(DS.TS / "atmosphere_elsauce.csv")
    qe = pd.read_csv(DS.SRC / "QE.csv")
    opt = pd.read_csv(DS.SRC / "optics.efficiency.dr350.dr500.csv", sep=r"\s+", comment="#", skiprows=11)
    out = {v: {} for v in VARIANTS}
    for f in DS.GRID:
        for ver, key in (("manufacturer", "manufacturer"), ("version_mhj", "mhj")):
            d = pd.read_csv(DS.VBASE / ver / f"{f}.csv")
            lam = d.lam_nm.to_numpy(float); tr = d.trans.to_numpy(float)
            env = (np.interp(lam, qe.iloc[:, 0], qe.iloc[:, 1]) * np.interp(lam, opt.iloc[:, 0], opt.iloc[:, -1])
                   * np.interp(lam, atm["lam_air"], atm["trans"]))
            te = DS.dr500_transmission(lam, tr, n_eff=1.5, f_number=3.03, obscuration=0.59)
            lv = DS.air_to_vac_nm(lam)
            out[f"{key}_nobeam"][f] = (lv, tr * env)
            out[f"{key}_beam"][f] = (lv, te * env)
    return out


def colour_term(fw, ft, fc, inst, bp, sig):
    zp0 = R.syn_mag(C.XP_WAVE, fc, fw, ft, 0.0) - inst
    fin = np.isfinite(zp0) & np.isfinite(bp) & np.isfinite(sig)
    if fin.sum() < 15:
        return np.nan, np.nan, 0
    keep = np.where(fin)[0][~np.ma.getmaskarray(SigmaClip(sigma=3.0, maxiters=5)(zp0[fin]))]
    if len(keep) < 15:
        return np.nan, np.nan, 0
    a, ae, _, _ = R.weighted_linfit(bp[keep], zp0[keep] - np.nanmedian(zp0[keep]), sig[keep])
    return a, ae, len(keep)


def _init():
    global _CURVES
    warnings.filterwarnings("ignore")
    K._refit_init()
    _CURVES = build_curves()


def _alarm(signum, frame):
    raise TimeoutError("tile timeout")


def process_tile(args):
    import signal
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(900)
    try:
        return _process_tile(args)
    except TimeoutError:
        return [dict(tile=args[0], status="timeout")]
    except Exception as e:                                           # noqa: BLE001
        return [dict(tile=args[0], status="error:" + str(e)[:80])]
    finally:
        signal.alarm(0)


def _process_tile(args):
    tile, frames = args
    matched, need = [], set()
    for fr, filt, unit, night, mjd in frames:
        try:
            d = DS.select_stars_v2(DS.CATD / (Path(fr).name[:-5] + "_cat.fits"))
        except Exception:
            continue
        if len(d) < 15:
            continue
        got = K.get_reference("GAIAXP_CORR_LAMOST", tile, float(np.median(d.ALPHA_J2000)), float(np.median(d.DELTA_J2000)), 1.34, 0.90)
        if got is None or got[1] is None or len(got[1]) == 0:
            continue
        tbl, coords, _ = got
        idx, d2d, _ = match_coordinates_sky(SkyCoord(d.ALPHA_J2000.to_numpy(), d.DELTA_J2000.to_numpy(), unit="deg"), coords)
        ok = np.asarray(d2d.arcsec < 2.0)
        if ok.sum() < 15:
            continue
        mr, di = tbl[idx[ok]], d[ok]
        bprp = np.asarray(mr["bp-rp"], float) if "bp-rp" in mr.colnames else np.full(len(mr), np.nan)
        for rid, bp, m, e in zip(np.asarray(mr["id"]).astype(str), bprp, di.MAG_APER_3, di.MAGERR_APER_3):
            matched.append((filt, fr, unit, night, mjd, rid, bp, float(m), float(e)))
            need.add(rid)
    if not matched:
        return [dict(tile=tile, status="no_frames")]
    xp = C.tile_xp(tile, need)
    M = pd.DataFrame(matched, columns=["filter", "file", "unit", "night", "mjd", "ref_id", "bprp", "inst", "ierr"])
    M = M[M.ref_id.isin(xp)]
    M["sig"] = np.sqrt(M.ierr.to_numpy() ** 2 + 0.01 ** 2)
    out = []
    for (filt, fr), g in M.groupby(["filter", "file"]):
        if len(g) < 15:
            continue
        fc = np.vstack([xp[i][1] for i in g.ref_id])
        base = dict(filter=filt, file=fr, unit=g.unit.iloc[0], night=g.night.iloc[0], mjd=g.mjd.iloc[0], tile=tile,
                    bprp_lo=float(np.nanpercentile(g.bprp, 10)), bprp_hi=float(np.nanpercentile(g.bprp, 90)), status="ok")
        for v in VARIANTS:
            a, ae, n = colour_term(*_CURVES[v][filt], fc, g.inst.to_numpy(), g.bprp.to_numpy(), g.sig.to_numpy())
            if np.isfinite(a):
                out.append(dict(base, variant=v, n=n, a0=round(float(a), 5), a0_err=round(float(ae), 5)))
    return out or [dict(tile=tile, status="too_few_stars")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=40)
    ap.add_argument("--limit-tiles", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ez = pd.read_csv(DS.EZALL_CSV)
    ez = ez[ez["filter"].isin(DS.GRID) & ez.file.notna()]
    ez = ez[[(DS.CATD / (Path(f).name[:-5] + "_cat.fits")).exists() for f in ez.file]]
    by_tile = defaultdict(list)
    for r in ez.itertuples():
        by_tile[r.tile].append((r.file, r.filter, r.unit, r.night, r.mjd))
    items = sorted(by_tile.items(), key=lambda kv: -len(kv[1]))
    if a.limit_tiles:
        items = items[len(items) // 2: len(items) // 2 + a.limit_tiles]
    part = OUT / "colorterm_time_frames.partial.csv"
    rows = pd.read_csv(part).to_dict("records") if part.exists() else []
    done = {r["tile"] for r in rows}
    todo = [it for it in items if it[0] not in done]
    DS.log(f"{len(ez)} frames, {len(items)} tiles ({len(items) - len(todo)} done), {a.n_proc} workers, variants {VARIANTS}")
    t0 = time.time()
    with Pool(a.n_proc, initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(process_tile, todo, chunksize=1), 1):
            rows += r
            if i % 200 == 0 or i == len(todo):
                DS.log(f"  {i}/{len(todo)} tiles, {sum(1 for x in rows if x.get('status') == 'ok')} rows, {(time.time() - t0) / 60:.1f} min")
                pd.DataFrame(rows).to_csv(part, index=False)
    fr = pd.DataFrame(rows)
    fr.to_csv(part, index=False)
    fr[fr.status == "ok"].drop(columns="status").to_csv(OUT / "colorterm_time_frames.csv", index=False)
    DS.log(f"tile status: {fr.drop_duplicates('tile').status.value_counts().to_dict()}")
    DS.log("saved")


if __name__ == "__main__":
    main()
