#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260927_std_rephot.py — re-photometer every CALSPEC and SPSS standard-star frame with the fixed ezphot background.

  background : MaskGenerator.mask_sources(5 sigma, radius x3) -> BackgroundGenerator.estimate_with_photutils(
               bkg_estimator='mean', sigma=5 (default), box 64, filter 3)   (per-mesh 5-sigma-clipped mean; replaces the
               SExtractor mode estimator that left a +~1 ADU/pix pedestal)
  photometry : ScienceImage.photometry_sex(target_bkg=bk)  (10" diameter = MAG_APER_3)
Frames:
  CALSPEC : 20260914_calspec_ezphot/photometry_meta.csv (status ok; 2024, 13 stars)
  SPSS    : 20260918_spss/process_meta_7DT*.csv (status ok, SPSS targets) -> processed frames in the ezphot scidata
            tree (BDF + astrometry done in 260918_spss_process.py); unit 7DT11 excluded (its filter wheel stayed at r during
            the campaign, 260910_transmission_sources_report.md 20.2)
Images are opened read-only; img.savedir points to a scratch work dir so ezphot's temporary products never touch
the data tree.  Output: ~/ezphot/log/20260927_std_newbkg/catalogs/<name>_cat.fits (trimmed catalog, original header +
BKGMEAN/BKGRMS of the new background), rephot_meta.csv.  Resume: existing catalogs are skipped.

  MPLCONFIGDIR=/tmp/ezphot-matplotlib OMP_NUM_THREADS=1 python -u ezphot/analysis/260927_std_rephot.py --n-proc 36 --cap 8
(--cap 8: at most 8 frames per star x filter x unit x exposure time, evenly spaced; 24-36 workers — 64 workers thrash)
"""
from __future__ import annotations

import argparse
import glob
import importlib
import os
import shutil
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np                  # noqa: E402
import pandas as pd                 # noqa: E402

warnings.filterwarnings("ignore")
import portalocker                  # noqa: E402  NFS on wfm: flock fails (EBADF) -> no-op locks


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
sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")

OUT = Path("/home/hhchoi1022/ezphot/log/20260927_std_newbkg")
CATD = OUT / "catalogs"
WORK = Path("/tmp/claude-10000/-lyman-data1-factory-hhchoi-code-ezphot/b1308704-c229-4e49-b19d-bd7bc334e4b7/scratchpad/stdrephot_work")
CAL_META = Path("/home/hhchoi1022/ezphot/log/20260914_calspec_ezphot/photometry_meta.csv")
SPSS_META = "/home/hhchoi1022/ezphot/log/20260918_spss/process_meta_7DT*.csv"
SCI = "/home/hhchoi1022/ezphot/data/scidata/7DT/7DT_C361K_HIGH_1x1"
EXCLUDE_SPSS_UNITS = {"7DT11"}
KEEP_PREFIX = ("MAG_", "MAGERR_", "SNR_", "FLUX_MAX", "BACKGROUND")
KEEP_COLS = ("X_WORLD", "Y_WORLD", "ALPHA_J2000", "DELTA_J2000", "X_IMAGE", "Y_IMAGE", "FLAGS", "CLASS_STAR",
             "FWHM_WORLD", "FWHM_IMAGE", "ELONGATION")


def log(m):
    print(time.strftime("[%H:%M:%S] ") + m, flush=True)


def frame_list():
    c = pd.read_csv(CAL_META)
    c = c[c.status == "ok"]
    cal = pd.DataFrame(dict(dataset="CALSPEC", file=c.file, object=c.object, filter=c["filter"], unit=c.unit))
    AN = importlib.import_module("260919_spss_analysis")
    names = {AN.norm(n) for n in pd.read_csv(AN.SPSS_CSV).Name}
    m = pd.concat([pd.read_csv(f) for f in glob.glob(SPSS_META)], ignore_index=True)
    m = m[(m.status == "ok") & m.object.map(lambda o: AN.norm(o) in names) & ~m.unit.isin(EXCLUDE_SPSS_UNITS)]
    sp = pd.DataFrame(dict(dataset="SPSS", object=m.object, filter=m["filter"], unit=m.unit,
                           file=[f"{SCI}/{o}/{u}/{f}/{os.path.basename(r)}" for o, u, f, r in
                                 zip(m.object, m.unit, m["filter"], m.file)]))
    return pd.concat([cal, sp], ignore_index=True)


def process(args):
    dataset, path, obj, filt, unit = args
    name = Path(path).name
    outp = CATD / (name[:-5] + "_cat.fits")
    row = dict(dataset=dataset, file=path, object=obj, filter=filt, unit=unit)
    if outp.exists():
        return dict(row, status="cached")
    wd = WORK / name[:-5]
    wd.mkdir(parents=True, exist_ok=True)
    try:
        from ezphot.imageobjects import ScienceImage, Mask
        from ezphot.methods import MaskGenerator, BackgroundGenerator
        img = ScienceImage(path=path, telinfo=None, load=True)
        img.savedir = wd
        srcmask = MaskGenerator().mask_sources(img, target_mask=Mask(wd / "srcmask.fits", masktype="source", load=False),
                                               sigma=5.0, mask_radius_factor=3, save=False, verbose=False, visualize=False)
        bk, _ = BackgroundGenerator().estimate_with_photutils(img, target_mask=srcmask, box_size=64, filter_size=3,
                                                             bkg_estimator="mean", save=False, verbose=False,
                                                             visualize=False)
        cat = img.photometry_sex(target_bkg=bk, save=False, verbose=False, visualize=False)
        d = cat.data
        keep = [k for k in d.colnames if k.startswith(KEEP_PREFIX) or k in KEEP_COLS]
        hdr = fits.Header(img.header)
        bdata = np.asarray(bk.data, float)
        hdr["BKGMEAN"] = (float(np.nanmedian(bdata)), "new background: median of per-mesh 5sig mean map [ADU]")
        hdr["BKGMETH"] = ("photutils_mean_s5_b64_f3", "260927_std_rephot.py")
        fits.HDUList([fits.PrimaryHDU(header=hdr), fits.BinTableHDU(Table(d[keep]), name="LDAC_OBJECTS")]).writeto(outp, overwrite=True)
        return dict(row, mjd=float(hdr.get("MJD", np.nan)), exptime=float(hdr.get("EXPTIME", np.nan)),
                    bkgmean=hdr["BKGMEAN"], n_det=len(d), status="ok")
    except Exception as e:                                           # noqa: BLE001
        return dict(row, status="error:" + str(e)[:200])
    finally:
        shutil.rmtree(wd, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dataset", default="", help="CALSPEC or SPSS only")
    ap.add_argument("--cap", type=int, default=0, help="max frames per (dataset, object, filter, unit, exptime), evenly spaced in time")
    a = ap.parse_args()
    CATD.mkdir(parents=True, exist_ok=True); WORK.mkdir(parents=True, exist_ok=True)
    F = frame_list()
    if a.dataset:
        F = F[F.dataset == a.dataset]
    if a.cap:
        F = F.assign(exp=F.file.str.extract(r"_([\d.]+)s_")[0], name=F.file.map(lambda f: Path(f).name)).sort_values("name")
        F = pd.concat([g.iloc[np.linspace(0, len(g) - 1, min(a.cap, len(g))).round().astype(int)]
                       for _, g in F.groupby(["dataset", "object", "filter", "unit", "exp"])], ignore_index=True)
    if a.limit:
        F = F.groupby("dataset").head(a.limit)
    todo = list(F[["dataset", "file", "object", "filter", "unit"]].itertuples(index=False, name=None))
    log(f"{len(todo)} frames ({F.groupby('dataset').size().to_dict()}), {a.n_proc} workers")
    metap = OUT / "rephot_meta.partial.csv"
    rows = pd.read_csv(metap).to_dict("records") if metap.exists() else []
    t0 = time.time()
    with Pool(a.n_proc) as pool:
        for i, r in enumerate(pool.imap_unordered(process, todo, chunksize=1), 1):
            rows.append(r)
            if i % 200 == 0 or i == len(todo):
                log(f"  {i}/{len(todo)}  {(time.time() - t0) / 60:.1f} min")
                pd.DataFrame(rows).to_csv(metap, index=False)
    M = pd.DataFrame(rows)
    M = M.sort_values("status").drop_duplicates("file", keep="last")          # 'ok' beats 'cached' on reruns
    M.to_csv(OUT / "rephot_meta.csv", index=False)
    log(f"status: {M.status.str[:40].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
