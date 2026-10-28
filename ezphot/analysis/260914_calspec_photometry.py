#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260914_calspec_photometry.py  (stage 1 of the CALSPEC redo)

Re-photometer every frame that contains a CALSPEC standard (frame list = unique files of
20260908_zp_color_check/calspec_frames.csv, 10,905 medium-band frames, 13 stars) with ezphot
photometry_sex — same settings as the ezall run (§18-validated).  Trimmed catalogs with the
original image header are saved to a NEW folder; NOTHING in the ezphot tree is deleted or
modified (these frames are originals in the tree, unlike the ezall pipeline copies).

Resume: a frame whose catalog already exists in CATD is skipped.

  MPLCONFIGDIR=/tmp/ezphot-matplotlib OMP_NUM_THREADS=1 /home/hhchoi1022/miniconda3/envs/too/bin/python -u \\
      ezphot/analysis/260914_calspec_photometry.py --n-proc 32
"""
from __future__ import annotations

import argparse
import os
import sys
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

OUT = Path("/home/hhchoi1022/ezphot/log/20260914_calspec_ezphot")
CATD = OUT / "catalogs"
SRC_CSV = Path("/home/hhchoi1022/ezphot/log/20260908_zp_color_check/calspec_frames.csv")
GRID = ["m400", "m425", "m450", "m475", "m500", "m525", "m550", "m575", "m600", "m625",
        "m650", "m675", "m700", "m725", "m750", "m775", "m800", "m825", "m850", "m875"]


def _init():
    warnings.filterwarnings("ignore")


def process(frame):
    try:
        p = Path(frame)
        filt, unit, obj = p.parts[-2], p.parts[-3], p.parts[-4]
        if filt not in GRID or not p.exists():
            return dict(file=frame, object=obj, filter=filt, status="missing")
        hdr = fits.getheader(frame)
        if hdr.get("CTYPE1") is None:
            return dict(file=frame, object=obj, filter=filt, status="no_wcs")
        outp = CATD / (p.name[:-5] + "_cat.fits")
        if not outp.exists():
            from ezphot.imageobjects import ScienceImage
            img = ScienceImage(path=frame, telinfo=None, load=True)
            cat = img.photometry_sex(save=False, verbose=False, visualize=False)
            d = cat.data
            keep = [c for c in d.colnames if c.startswith(("MAG_", "MAGERR_", "SNR_", "FLUX_MAX"))
                    or c in ("X_WORLD", "Y_WORLD", "ALPHA_J2000", "DELTA_J2000", "X_IMAGE", "Y_IMAGE",
                             "FLAGS", "CLASS_STAR", "FWHM_WORLD", "FWHM_IMAGE", "ELONGATION")]
            phdu = fits.PrimaryHDU(header=fits.Header(img.header))   # original image header = provenance
            fits.HDUList([phdu, fits.BinTableHDU(Table(d[keep]), name="LDAC_OBJECTS")]).writeto(
                outp, overwrite=True)
        return dict(file=frame, object=obj, filter=filt, unit=unit,
                    mjd=float(hdr.get("MJD", np.nan)), status="ok")
    except Exception as e:                                   # noqa: BLE001
        return dict(file=frame, object=frame.split("/")[-4], filter=frame.split("/")[-2],
                    status=("error:" + str(e)[:150]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    CATD.mkdir(parents=True, exist_ok=True)
    src = pd.read_csv(SRC_CSV)
    frames = sorted(src.drop_duplicates("file")["file"])
    frames = [f for f in frames if Path(f).parts[-2] in GRID]
    done = {q.name for q in CATD.glob("*_cat.fits")}
    todo = [f for f in frames if (Path(f).name[:-5] + "_cat.fits") not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(todo)} frames to photometer ({len(frames)-len(todo)} already done), {a.n_proc} workers", flush=True)
    rows = []
    metap = OUT / "photometry_meta.partial.csv"
    if metap.exists():
        try:
            rows = pd.read_csv(metap).to_dict("records")
        except Exception:
            rows = []
    have = {r.get("file") for r in rows}
    with Pool(a.n_proc, initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(process, todo), 1):
            if r and r.get("file") not in have:
                rows.append(r)
            if i % 200 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
                pd.DataFrame(rows).to_csv(metap, index=False)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "photometry_meta.csv", index=False)
    print(res["status"].str[:30].value_counts().head(8).to_string())
    print(f"catalogs in {CATD}: {len(list(CATD.glob('*_cat.fits')))}")


if __name__ == "__main__":
    main()
