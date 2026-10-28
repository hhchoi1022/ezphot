#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260926_delta_star_v2.py — per-frame delta* (curve shift that zeroes the corrected-XP field colour term) re-solved on
the ezall archive with the star selection and curve conventions settled by the background / linearity work.

Changes vs 260924_colorterm_versions.py (same frames = catalogs_ezall, same 10" MAG_APER_3, same solver OT.solve_offset):
  stars  : Gaia-XP (corrected) match < 2", isolated 10" against the full detection list, FLAGS == 0,
           SNR_10" >= 100, FLUX_MAX <= 50000 ADU; NO CLASS_STAR / FWHM / ELONGATION cuts (they bias faint stars) and no
           upper SNR cap (old selection: SNR 10-150 + shape cuts)
  curves : manufacturer x (beam n_eff 1.5 | no beam) x QE x optics x atmosphere, each on AIR wavelengths (as before)
           and converted to VACUUM (lambda_vac = n_air(lambda) lambda_air, VALD/Morton formula) — Gaia XP is on the
           vacuum scale (dispersion anchored to SDSS QSO lines, fluxes to CALSPEC), the curves are lab (air) data
Variants: manufacturer_beam_air, manufacturer_beam_vac, manufacturer_nobeam_air, manufacturer_nobeam_vac
Output (~/ezphot/log/20260926_delta_star_v2/): delta_star_v2_frames.csv (frame x variant: n, a0, dstar, dstar_err)
Resume: tiles already in delta_star_v2_frames.partial.csv are skipped.  Straggler guard: 900 s per tile.

  MPLCONFIGDIR=/tmp/ezphot-matplotlib OMP_NUM_THREADS=1 /home/hhchoi1022/miniconda3/envs/too/bin/python -u \\
      ezphot/analysis/260926_delta_star_v2.py --n-proc 40 [--limit-tiles N] [--tag _test]
"""
from __future__ import annotations

import argparse
import importlib
import os
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
from astropy.coordinates import SkyCoord, match_coordinates_sky, search_around_sky   # noqa: E402
import astropy.units as u           # noqa: E402

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
OUT = Path("/home/hhchoi1022/ezphot/log/20260926_delta_star_v2")
VARIANTS = ["manufacturer_beam_air", "manufacturer_beam_vac", "manufacturer_nobeam_air", "manufacturer_nobeam_vac"]
SNR_MIN, PEAK_MAX, ISO_ARCSEC = 100.0, 50000.0, 10.0
_CURVES = None


def log(m):
    print(time.strftime("[%H:%M:%S] ") + m, flush=True)


def air_to_vac_nm(lam_nm):
    """lambda_vac = n(lambda_air) * lambda_air (VALD / Morton 2000 inverse formula), lambda in nm."""
    s2 = (1e3 / np.asarray(lam_nm, float)) ** 2                      # (1/um)^2
    n = 1 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s2) + 0.0001599740894897 / (38.92568793293 - s2)
    return np.asarray(lam_nm, float) * n


def build_curves():
    atm = pd.read_csv(TS / "atmosphere_elsauce.csv")
    qe = pd.read_csv(SRC / "QE.csv")
    opt = pd.read_csv(SRC / "optics.efficiency.dr350.dr500.csv", sep=r"\s+", comment="#", skiprows=11)
    atr = atm["trans"] if "trans" in atm.columns else atm.iloc[:, 1]
    out = {v: {} for v in VARIANTS}
    for f in GRID:
        d = pd.read_csv(VBASE / "manufacturer" / f"{f}.csv")
        lam = d.lam_nm.to_numpy(float); tr = d.trans.to_numpy(float)
        env = (np.interp(lam, qe.iloc[:, 0], qe.iloc[:, 1]) * np.interp(lam, opt.iloc[:, 0], opt.iloc[:, -1])
               * np.interp(lam, atm["lam_air"], atr))
        te = dr500_transmission(lam, tr, n_eff=1.5, f_number=3.03, obscuration=0.59)
        lv = air_to_vac_nm(lam)
        out["manufacturer_beam_air"][f] = (lam, te * env)
        out["manufacturer_beam_vac"][f] = (lv, te * env)
        out["manufacturer_nobeam_air"][f] = (lam, tr * env)
        out["manufacturer_nobeam_vac"][f] = (lv, tr * env)
    return out


def select_stars_v2(catpath):
    """Gaia-independent part of the new selection: FLAGS 0, SNR >= 100, FLUX_MAX <= 50k, isolated 10" (full list)."""
    from astropy.io import fits
    with fits.open(catpath) as h:
        dd = h["LDAC_OBJECTS"].data
        names = set(dd.columns.names)
    d = OZ.load_ez(catpath)
    d["FLUX_MAX"] = np.asarray(dd["FLUX_MAX"], float) if "FLUX_MAX" in names else np.nan
    allc = SkyCoord(d.ALPHA_J2000.to_numpy(), d.DELTA_J2000.to_numpy(), unit="deg")
    q = (d.FLAGS.to_numpy() == 0) & (d.SNR_APER_3.to_numpy() >= SNR_MIN) & (d.FLUX_MAX.to_numpy() <= PEAK_MAX) \
        & (d.MAG_APER_3.to_numpy() < 90)
    if q.sum() < 15:
        return d.iloc[[]]
    sub = allc[q]
    ig, _, sep2d, _ = search_around_sky(sub, allc, ISO_ARCSEC * u.arcsec)
    nbr = np.bincount(ig[sep2d.arcsec > 0.5], minlength=int(q.sum()))
    return d[q][nbr == 0].reset_index(drop=True)


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
        cp = CATD / (Path(fr).name[:-5] + "_cat.fits")
        try:
            d = select_stars_v2(cp)
        except Exception:
            continue
        if len(d) < 15:
            continue
        ra0, dec0 = float(np.median(d.ALPHA_J2000)), float(np.median(d.DELTA_J2000))
        got = K.get_reference("GAIAXP_CORR_LAMOST", tile, ra0, dec0, 1.34, 0.90)
        if got is None:
            continue
        tbl, coords, _ = got
        if coords is None or len(coords) == 0:
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
            matched.append((filt, fr, unit, night, mjd, rid[k], bprp[k],
                            float(di.MAG_APER_3.iloc[k]), float(di.MAGERR_APER_3.iloc[k])))
            need.add(rid[k])
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
                    n=len(g), bprp_lo=float(np.nanpercentile(g.bprp, 10)), bprp_hi=float(np.nanpercentile(g.bprp, 90)),
                    status="ok")
        for v in VARIANTS:
            dstar, derr, a0 = OT.solve_offset(*_CURVES[v][filt], fc, g.inst.to_numpy(), g.bprp.to_numpy(), g.sig.to_numpy())
            if not np.isfinite(a0):
                continue
            out.append(dict(base, variant=v, a0=round(float(a0), 5),
                            dstar=round(float(dstar), 3) if np.isfinite(dstar) else np.nan,
                            dstar_err=round(float(derr), 3) if np.isfinite(derr) else np.nan))
    return out or [dict(tile=tile, status="too_few_stars")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=40)
    ap.add_argument("--limit-tiles", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ez = pd.read_csv(EZALL_CSV)
    ez = ez[ez["filter"].isin(GRID) & ez.file.notna()]
    ez = ez[[(CATD / (Path(f).name[:-5] + "_cat.fits")).exists() for f in ez.file]]
    by_tile = defaultdict(list)
    for r in ez.itertuples():
        by_tile[r.tile].append((r.file, r.filter, r.unit, r.night, r.mjd))
    items = sorted(by_tile.items(), key=lambda kv: -len(kv[1]))
    if a.limit_tiles:
        items = items[len(items) // 2: len(items) // 2 + a.limit_tiles]
    part = OUT / f"delta_star_v2_frames{a.tag}.partial.csv"
    rows = pd.read_csv(part).to_dict("records") if part.exists() else []
    done = {r["tile"] for r in rows}
    todo = [it for it in items if it[0] not in done]
    log(f"{len(ez)} frames, {len(items)} tiles ({len(items) - len(todo)} done), {a.n_proc} workers, variants {VARIANTS}")
    t0 = time.time()
    with Pool(a.n_proc, initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(process_tile, todo, chunksize=1), 1):
            rows += r
            if i % 100 == 0 or i == len(todo):
                log(f"  {i}/{len(todo)} tiles, {sum(1 for x in rows if x.get('status') == 'ok')} rows, "
                    f"{(time.time() - t0) / 60:.1f} min")
                pd.DataFrame(rows).to_csv(part, index=False)
    fr = pd.DataFrame(rows)
    fr.to_csv(part, index=False)
    ok = fr[fr.status == "ok"].drop(columns="status")
    ok.to_csv(OUT / f"delta_star_v2_frames{a.tag}.csv", index=False)
    log(f"tile status: {fr.drop_duplicates('tile').status.value_counts().to_dict()}")
    summ = (ok.groupby(["filter", "variant"]).agg(n_frames=("a0", "size"), n_stars=("n", "median"),
                                                  a0=("a0", "median"), dstar=("dstar", "median"))
              .reset_index())
    print(summ.pivot_table(index="filter", columns="variant", values="dstar").round(2).to_string())
    log("saved")


if __name__ == "__main__":
    main()
