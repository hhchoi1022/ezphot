#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260918_spss_process.py

Raw SPSS-campaign frames (obsdata_from_mcs, nights 2026-09-14/15/16) -> ezphot chain up to PHOTOMETRY:
  masters per unit (generate_masterframe from the nights' BIAS/DARK/FLAT)
  -> correct_bdf -> solve_astrometry (astrometry.net) -> photometry_sex (10" = MAG_APER_3)
  -> trimmed catalog (original header attached) saved to 20260918_spss/catalogs/.

Science frames = LIGHT frames whose OBJECT is not BIAS/DARK/FLAT and not a survey tile (7DT-main*).
Frame-level Pool uses fork copy-on-write to share the unit's master frames.
Resume: frames whose catalog already exists are skipped.
Later nights (raw tree /lyman/data1/obsdata/<unit>/<night>): --base, --nights, --calib-nights, --tag (meta file
process_meta_<unit>_<tag>.csv, so earlier meta files are kept) and --spss-only (SPSS table names only).

  MPLCONFIGDIR=/tmp/ezphot-matplotlib /home/hhchoi1022/miniconda3/envs/too/bin/python -u \\
      ezphot/analysis/260918_spss_process.py --units 7DT01 --n-proc 8
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
import numpy as np                  # noqa: E402
import pandas as pd                 # noqa: E402

warnings.filterwarnings("ignore")
import portalocker                  # noqa: E402


class _NoLock:
    def __init__(self, *a, **k): pass
    def __enter__(self): return None
    def __exit__(self, *a): return False


portalocker.Lock = _NoLock
portalocker.lock = lambda *a, **k: None
portalocker.unlock = lambda *a, **k: None

from astropy.io import fits          # noqa: E402
from astropy.table import Table      # noqa: E402

sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot")

BASE = "/lyman/data1/obsdata/obsdata_from_mcs/{unit}/image/{night}"
NIGHTS = ["2026-09-14_gain2750", "2026-09-15_gain2750", "2026-09-16_gain2750"]
CALIB_NIGHTS = NIGHTS
SPSS_ONLY = False
OUT = Path("/home/hhchoi1022/ezphot/log/20260918_spss")
CATD = OUT / "catalogs"
SKIP_OBJ = {"BIAS", "DARK", "FLAT"}

_G = {}


def log(m):
    print(time.strftime("[%H:%M:%S] ") + m, flush=True)


def build_unit(unit):
    """Masters for one unit from all three nights (in memory + archived)."""
    from ezphot.helper import Helper
    from ezphot.imageobjects import CalibrationImage
    from ezphot.methods import Preprocess
    helper = Helper()
    telinfo = helper.get_telinfo(telescope="7DT", ccd="C361K", readoutmode="HIGH", binning=1)
    calib = []
    for n in CALIB_NIGHTS:
        for pat in ("*BIAS*.fits", "*DARK*.fits", "*FLAT*.fits"):
            calib += glob.glob(BASE.format(unit=unit, night=n) + f"/{pat}")
    if not calib:
        return None, None, None
    pre = Preprocess()
    imgs = [CalibrationImage(p, telinfo=telinfo, load=False) for p in sorted(calib)]
    masters = pre.generate_masterframe(imgs, n_proc=8, verbose=False, save=True)
    mset = list(masters.values())[0]
    return telinfo, pre, mset


def science_frames(unit):
    out = []
    if SPSS_ONLY:
        import importlib
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        AN = importlib.import_module("260919_spss_analysis")
        names = {AN.norm(x) for x in pd.read_csv(AN.SPSS_CSV).Name}
    for n in NIGHTS:
        for p in sorted(glob.glob(BASE.format(unit=unit, night=n) + "/*.fits")):
            name = Path(p).name
            parts = name.split("_")
            if len(parts) < 5:
                continue
            obj = "_".join(parts[3:-4]) if len(parts) > 8 else parts[3]
            if obj in SKIP_OBJ or obj.startswith("7DT-main"):
                continue
            if SPSS_ONLY and AN.norm(obj) not in names:
                continue
            out.append((p, obj, parts[-4], n))                       # path, object, filter, night
    return out


def _init_worker(telinfo, mset):
    warnings.filterwarnings("ignore")
    _G["telinfo"] = telinfo
    _G["mset"] = mset
    from ezphot.methods import Preprocess, Platesolve
    _G["pre"] = Preprocess()
    _G["ps"] = Platesolve()


def process(args):
    raw, obj, filt, night = args
    outcat = CATD / (Path(raw).name[:-5] + "_cat.fits")
    if outcat.exists():
        return dict(file=raw, object=obj, filter=filt, night=night, status="done")
    try:
        from ezphot.imageobjects import ScienceImage
        telinfo, mset = _G["telinfo"], _G["mset"]
        mflat = mset["FLAT"].get(filt)
        if mflat is None:
            return dict(file=raw, object=obj, filter=filt, night=night, status="no_flat")
        exp = float(Path(raw).name.split("_")[-2].replace("s", ""))
        dkeys = sorted(mset["DARK"], key=lambda k: abs(float(k) - exp))
        mdark = mset["DARK"][dkeys[0]]
        sci = ScienceImage(raw, telinfo=telinfo, load=False)
        cal = _G["pre"].correct_bdf(target_img=sci, bias_image=mset["BIAS"], dark_image=mdark,
                                    flat_image=mflat, save=True, verbose=False)
        out = _G["ps"].solve_astrometry(cal, overwrite=True, verbose=False)
        if out is None:
            return dict(file=raw, object=obj, filter=filt, night=night, status="astrometry_failed")
        catobj = out.photometry_sex(save=False, verbose=False, visualize=False)
        d = catobj.data
        keep = [c for c in d.colnames if c.startswith(("MAG_", "MAGERR_", "SNR_", "FLUX_MAX", "BACKGROUND"))
                or c in ("X_WORLD", "Y_WORLD", "ALPHA_J2000", "DELTA_J2000", "X_IMAGE", "Y_IMAGE",
                         "FLAGS", "CLASS_STAR", "FWHM_WORLD", "FWHM_IMAGE", "ELONGATION")]
        phdu = fits.PrimaryHDU(header=fits.Header(out.header))
        fits.HDUList([phdu, fits.BinTableHDU(Table(d[keep]), name="LDAC_OBJECTS")]).writeto(outcat, overwrite=True)
        names = set(d.colnames)
        fw = np.asarray(d["FWHM_WORLD"], float) * 3600.0
        cs = np.asarray(d["CLASS_STAR"], float)
        star = (cs > 0.8) & np.isfinite(fw) & (fw > 0.3) & (fw < 15)
        seeing = float(np.median(fw[star])) if star.sum() > 5 else np.nan
        return dict(file=raw, object=obj, filter=filt, night=night, unit=Path(raw).name.split("_")[0],
                    mjd=float(out.header.get("MJD", np.nan)) if "MJD" in out.header else np.nan,
                    n_det=len(d), seeing=round(seeing, 2),
                    skyval=float(np.median(np.asarray(d["BACKGROUND"], float))) if "BACKGROUND" in names else np.nan,
                    status="ok")
    except Exception as e:                                           # noqa: BLE001
        return dict(file=raw, object=obj, filter=filt, night=night, status=("error:" + str(e)[:150]))


def main():
    global BASE, NIGHTS, CALIB_NIGHTS, SPSS_ONLY
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", nargs="*", default=["7DT01"])
    ap.add_argument("--n-proc", type=int, default=8)
    ap.add_argument("--base", default=BASE, help="raw night dir template with {unit} and {night}")
    ap.add_argument("--nights", nargs="*", default=NIGHTS)
    ap.add_argument("--calib-nights", nargs="*", default=None, help="nights for masters (default: --nights)")
    ap.add_argument("--tag", default="", help="meta file suffix: process_meta_<unit>_<tag>.csv")
    ap.add_argument("--spss-only", action="store_true")
    a = ap.parse_args()
    BASE, NIGHTS, SPSS_ONLY = a.base, a.nights, a.spss_only
    CALIB_NIGHTS = a.calib_nights or a.nights
    CATD.mkdir(parents=True, exist_ok=True)
    for unit in a.units:
        frames = science_frames(unit)
        todo = [f for f in frames if not (CATD / (Path(f[0]).name[:-5] + "_cat.fits")).exists()]
        log(f"=== {unit}: {len(frames)} science frames ({len(frames)-len(todo)} done) ===")
        if not todo:
            continue
        log(f"{unit}: building masters ...")
        telinfo, pre, mset = build_unit(unit)
        if mset is None:
            log(f"{unit}: no calibration frames, skipping")
            continue
        log(f"{unit}: masters ready (darks {list(mset['DARK'])}, flats {list(mset['FLAT'])})")
        rows = []
        metap = OUT / (f"process_meta_{unit}_{a.tag}.csv" if a.tag else f"process_meta_{unit}.csv")
        with Pool(a.n_proc, initializer=_init_worker, initargs=(telinfo, mset)) as pool:
            for i, r in enumerate(pool.imap_unordered(process, todo), 1):
                rows.append(r)
                if i % 25 == 0:
                    log(f"  {unit}: {i}/{len(todo)}")
                    pd.DataFrame(rows).to_csv(metap, index=False)
        df = pd.DataFrame(rows)
        df.to_csv(metap, index=False)
        log(f"{unit} status: " + df["status"].str[:24].value_counts().to_dict().__repr__())


if __name__ == "__main__":
    main()
