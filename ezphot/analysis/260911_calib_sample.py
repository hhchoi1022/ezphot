#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260911_calib_sample.py

Run the NATIVE ezphot photometric_calibration on ONE sample pipeline frame and produce its standard figures
(star-selection filtering, ZP vs mag, ZP vs colour, ZP 2D map, reference-mag) so the star-selection behaviour
can be inspected before aligning the delta* time-series cuts with it.

The frame is COPIED to a work dir first — the pipeline tree is never touched; all catalogs/figures land in the
work dir.  Default photometric_calibration parameters (mag 12-16, elong<1.7, FWHM 1-10", FLAGS<=1,
isolation 10", inner 0.9), catalog_type GAIAXP_CORR_LAMOST.

  MPLCONFIGDIR=/tmp/ezphot-matplotlib /home/hhchoi1022/miniconda3/envs/too/bin/python -u \\
      ezphot/analysis/260911_calib_sample.py
"""
from __future__ import annotations

import shutil
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# portalocker NFS EBADF workaround (same as 260911_tilestd_harvest)
import portalocker            # noqa: E402


class _NoLock:
    def __init__(self, *a, **k): pass
    def __enter__(self): return None
    def __exit__(self, *a): return False


portalocker.Lock = _NoLock
portalocker.lock = lambda *a, **k: None
portalocker.unlock = lambda *a, **k: None

sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot")

SRC = Path("/lyman/data2/processed/2025-03-06/T22626/m600/singles/T22626_m600_7DT14_20250307_042355_100s.fits")
WORK = Path("/home/hhchoi1022/ezphot/log/20260911_calib_sample")


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    dst = WORK / SRC.name
    if not dst.exists():
        shutil.copy2(SRC, dst)
    print(f"sample frame copied -> {dst}", flush=True)

    from ezphot.imageobjects import ScienceImage
    from ezphot.methods import PhotometricCalibration

    img = ScienceImage(path=str(dst), telinfo=None, load=True)
    print(f"image: {img.objname} {img.filter} {img.telname} seeing~{img.header.get('SEEING')}", flush=True)

    cat = img.photometry_sex(save=False, verbose=True, visualize=False)
    print(f"photometry: {len(cat.data)} sources", flush=True)

    pc = PhotometricCalibration()
    out = pc.photometric_calibration(
        target_img=img,
        target_catalog=cat,
        catalog_type="GAIAXP_CORR_LAMOST",
        save=False,
        save_refcat=False,
        update_header=False,
        verbose=True,
        visualize=False,
        save_fig=True,
    )
    print("calibration done", flush=True)
    pngs = sorted(WORK.rglob("*.png"))
    print("figures:")
    for p in pngs:
        print("  ", p, flush=True)


if __name__ == "__main__":
    main()
