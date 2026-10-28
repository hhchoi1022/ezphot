#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260911_tilestd_harvest.py

Photometer + reference-match the survey-tile single frames containing a CALSPEC standard, in-memory (the pipeline
does not persist single-frame catalogs).  Per frame: SExtractor photometry -> match to the GAIAXP_CORR_LAMOST
archive (reuse 260908_zpcolor_collect.get_reference) -> per-star ZP + colour term (refit_stars schema) and the
standard's calibrated magnitude vs its CALSPEC synphot (calspec_frames schema).

Four standards in observed tiles: HZ4/T22272, LDS749B/T19136, WD2039-682/T01485, BD-11_3759/T14891.
Single frames only (no coadds).  Corrected (LAMOST) XP only, as decided.

Outputs (append-safe) in /home/hhchoi1022/ezphot/log/20260911_tilestd/:
  refit_stars_tile.csv.gz, calspec_frames_tile.csv

  MPLCONFIGDIR=/tmp/ezphot-matplotlib OMP_NUM_THREADS=1 /home/hhchoi1022/miniconda3/envs/too/bin/python \\
      ezphot/analysis/260911_tilestd_harvest.py --standards HZ4 [--max-frames 4]
"""
from __future__ import annotations

import argparse
import glob
import importlib
import os
import re
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import portalocker


class _NoLock:
    def __init__(self, *a, **k): pass
    def __enter__(self): return None
    def __exit__(self, *a): return False


portalocker.Lock = _NoLock
portalocker.lock = lambda *a, **k: None
portalocker.unlock = lambda *a, **k: None

from astropy.coordinates import SkyCoord, match_coordinates_sky   # noqa: E402
from astropy.stats import SigmaClip                                # noqa: E402
from astropy.table import Table                                    # noqa: E402

sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
K = importlib.import_module("260908_zpcolor_collect")

ROOT = "/lyman/data1/processed_1x1_gain2750"
OUT = Path("/home/hhchoi1022/ezphot/log/20260911_tilestd"); OUT.mkdir(parents=True, exist_ok=True)
META = "/lyman/data1/Calibration/latest_CALSPEC/merged_meta.csv"
ZPC = Path("/home/hhchoi1022/ezphot/log/20260908_zp_color_check")
STD_TILE = {"HZ4": "T22272", "LDS749B": "T19136", "WD2039-682": "T01485", "BD-11_3759": "T14891"}
CATALOG = "GAIAXP_CORR_LAMOST"
MAGERR_FLOOR = 0.01


def log(m):
    print(time.strftime("[%Y-%m-%d %H:%M:%S] ") + m, flush=True)


def std_info():
    meta = pd.read_csv(META).set_index("Name")
    syn = pd.read_csv(ZPC / "calspec_synphot.csv")
    stars = pd.read_csv(Path("/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis/calspec_info.csv"))
    info = {}
    for o in STD_TILE:
        sp = syn[syn.object == o].spec.iloc[0]
        nm = re.split(r"_(mod|stis|stisnic|stiswfc|nic)", sp)[0]
        r = meta.loc[nm]
        g = syn[syn.object == o]
        smap = dict(zip(g["filter"], g["mag_syn"]))
        # spectral type from calspec_info if present
        st = ""
        try:
            st = str(meta.loc[nm, "Sp. T."])
        except Exception:
            pass
        info[o] = dict(ra0=float(r.radeg), dec0=float(r.decdeg), pmra=float(r["PM_R.A."]), pmdec=float(r["PM_Decl."]),
                       spectral_type=st, syn=smap)
    return info


def harvest_frame(path, obj, inf):
    from ezphot.imageobjects import ScienceImage
    img = ScienceImage(path=path, telinfo=None, load=True)
    filt = img.filter
    if filt not in K.MEDIUM if hasattr(K, "MEDIUM") else False:
        pass
    cat = img.photometry_sex(save=False, verbose=False, visualize=False)
    d = cat.data
    need = {"X_WORLD", "Y_WORLD", "X_IMAGE", "Y_IMAGE", "MAG_APER_3", "MAGERR_APER_3", "FLAGS", "CLASS_STAR", "FLUX_MAX"}
    if not need.issubset(d.colnames):
        return [], None
    mjd = float(img.mjd); nx, ny = int(img.naxis1), int(img.naxis2)
    px = float(np.mean(np.atleast_1d(img.pixelscale))); fov_ra, fov_dec = nx * px / 3600, ny * px / 3600
    ra0 = float(np.nanmedian(np.asarray(d["X_WORLD"], float))); dec0 = float(np.nanmedian(np.asarray(d["Y_WORLD"], float)))
    src = SkyCoord(np.asarray(d["X_WORLD"], float), np.asarray(d["Y_WORLD"], float), unit="deg")

    # ---- reference stars -> per-star ZP + colour term ----
    got = K.get_reference(CATALOG, obj, ra0, dec0, fov_ra, fov_dec)
    stars, calrow = [], None
    if got is not None:
        tbl, coords, tiles = got
        magcol, proxy = K.resolve_mag_column(filt, tbl.colnames, CATALOG)
        if magcol is not None:
            qc = (np.asarray(d["CLASS_STAR"], float) >= 0.8) & (np.asarray(d["FLAGS"], float) <= 0) \
                & (np.asarray(d["MAGERR_APER_3"], float) < 0.1) & (np.asarray(d["MAG_APER_3"], float) < 90)
            si = np.where(qc)[0]
            if len(si) >= 5:
                idx, d2d, _ = match_coordinates_sky(src[si], coords)
                ok = d2d.arcsec < 2.5
                mi = si[ok]; mr = tbl[idx[ok]]
                ref_mag = np.asarray(mr[magcol], float)
                inst = np.asarray(d["MAG_APER_3"], float)[mi]; ierr = np.asarray(d["MAGERR_APER_3"], float)[mi]
                bprp = np.asarray(mr["bp-rp"], float) if "bp-rp" in mr.colnames else np.full(len(mr), np.nan)
                ids = mr["id"] if "id" in mr.colnames else np.arange(len(mr))
                zp = ref_mag - inst
                fin = np.isfinite(zp) & np.isfinite(ierr)
                if fin.sum() >= 5:
                    clip = SigmaClip(sigma=3.0, maxiters=5)(zp[fin])
                    keep = np.where(fin)[0][~np.ma.getmaskarray(clip)]
                    zp_med = float(np.nanmedian(zp[keep]))
                    r_arcmin = np.hypot(np.asarray(d["X_IMAGE"], float)[mi] - nx / 2,
                                        np.asarray(d["Y_IMAGE"], float)[mi] - ny / 2) * px / 60
                    for j in keep:
                        stars.append(dict(file=str(path), catalog=CATALOG, filter=filt, telescop=img.telname,
                                          objname=obj, ref_id=str(ids[j]), bprp=float(bprp[j]), gr=np.nan, nbr=np.nan,
                                          zp=float(zp[j]), resid=float(zp[j] - zp_med), magerr=float(ierr[j]),
                                          ref_mag=float(ref_mag[j]), r_arcmin=float(r_arcmin[j])))

    # ---- standard photometry ----
    zp_frame = float(np.nanmedian([s["zp"] for s in stars])) if stars else float(getattr(img, "zp", np.nan))
    dt = (mjd - 51544.5) / 365.25
    rad = inf["dec0"] + inf["pmdec"] * dt / 3.6e6
    raa = inf["ra0"] + inf["pmra"] * dt / 3.6e6 / np.cos(np.deg2rad(inf["dec0"]))
    sep = SkyCoord(raa, rad, unit="deg").separation(src).arcsec
    j = int(np.argmin(sep))
    if sep[j] < 3.0:
        inst = float(np.asarray(d["MAG_APER_3"], float)[j]); ierr = float(np.asarray(d["MAGERR_APER_3"], float)[j])
        magsky = float(d["MAGSKY_APER_3"][j]) if "MAGSKY_APER_3" in d.colnames and np.isfinite(d["MAGSKY_APER_3"][j]) else inst + zp_frame
        calrow = dict(file=str(path), object=obj, filter=filt, telescop=img.telname, mjd=mjd,
                      airmass=float(getattr(img, "airmass", np.nan) or np.nan), exptime=float(img.exptime),
                      seeing=float(getattr(img, "seeing", np.nan) or np.nan), n_refstar=len(stars),
                      mag_syn=inf["syn"].get(filt, np.nan), spectral_type=inf["spectral_type"],
                      zp_APER_3=zp_frame, zperr_APER_3=float(np.nanstd([s["zp"] for s in stars])) if stars else np.nan,
                      sep_arcsec=float(sep[j]), flags=float(d["FLAGS"][j]), flux_max=float(d["FLUX_MAX"][j]),
                      class_star=float(d["CLASS_STAR"][j]),
                      mag_inst_APER_3=inst, mag_obs_APER_3=magsky, magerr_APER_3=ierr,
                      delta_APER_3=magsky - inf["syn"].get(filt, np.nan), status="ok")
    img.clear()
    return stars, calrow


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--standards", nargs="+", default=list(STD_TILE))
    ap.add_argument("--filters", nargs="*", default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    K._refit_init()
    info = std_info()
    all_stars, all_cal = [], []
    for obj in a.standards:
        tile = STD_TILE[obj]
        frames = sorted(f for f in glob.glob(f"{ROOT}/{tile}/*/m*/*.fits") if not f.endswith(".com.fits"))
        if a.filters:
            frames = [f for f in frames if f.split("/")[-2] in a.filters]
        if a.max_frames:
            frames = frames[:a.max_frames]
        log(f"{obj} tile {tile}: {len(frames)} frames")
        for i, f in enumerate(frames, 1):
            try:
                stars, calrow = harvest_frame(f, obj, info[obj])
                all_stars += stars
                if calrow: all_cal.append(calrow)
                if i % 10 == 0 or i == len(frames):
                    log(f"  {obj} {i}/{len(frames)}: cumulative stars {len(all_stars)}, calspec {len(all_cal)}")
            except Exception as e:                  # noqa: BLE001
                log(f"  {obj} frame {i} error: {e!r}"[:160])
    tag = a.tag or "all"
    if all_stars:
        pd.DataFrame(all_stars).to_csv(OUT / f"refit_stars_tile_{tag}.csv.gz", index=False)
    if all_cal:
        cdf = pd.DataFrame(all_cal); cdf.to_csv(OUT / f"calspec_frames_tile_{tag}.csv", index=False)
        log("calspec status: " + cdf.groupby(["object", "filter"]).size().to_string().replace("\n", " | ")[:300])
    log(f"done: {len(all_stars)} ref-star rows, {len(all_cal)} calspec rows -> {OUT} (tag {tag})")


if __name__ == "__main__":
    main()
