#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260911_offset_ez2024.py

Fill the earliest medium-band survey epoch (2024-05-02 .. 2024-06-30, 871 frames with NO pipeline catalogs)
by running ezphot direct photometry (validated in §18/§18.1: delta* equivalent to pipeline within ~0.3 nm in
the blue) and solving delta* per frame.  Read-only on the pipeline; ezphot catalogs saved separately under
20260911_ezphot_photometry/catalogs_2024early/; the ezphot tree copy of each frame is deleted right away.

  MPLCONFIGDIR=/tmp/ezphot-matplotlib OMP_NUM_THREADS=1 /home/hhchoi1022/miniconda3/envs/too/bin/python -u \\
      ezphot/analysis/260911_offset_ez2024.py --n-proc 32
"""
from __future__ import annotations

import argparse
import importlib
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
sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
OT = importlib.import_module("260911_offset_timeseries")            # solve_offset, meta, GRID, FOLDER
OZ = importlib.import_module("260911_offset_ezphot")                # dstar_of, PHYS handling
C, K = OT.C, OT.K
GRID = OT.GRID

SRC = OT.FOLDER / "all_medium_2024early_nocat.txt"          # default; override with --src
CATD = Path("/home/hhchoi1022/ezphot/log/20260911_ezphot_photometry/catalogs_2024early")
EZTREE = Path("/home/hhchoi1022/ezphot/data/scidata")
KEEP_COLS = None                                             # resolved per catalog: coords/mags/flags subset


def _init():
    warnings.filterwarnings("ignore")
    K._refit_init()
    OZ.PHYS = {f: C.load_new_curve(f) for f in GRID}


def process(frame):
    try:
        filt = frame.split("/")[-3]; tile = frame.split("/")[-4]
        if filt not in GRID or not os.path.exists(frame):
            return None
        if fits.getheader(frame).get("CTYPE1") is None:       # no astrometry -> no coordinates -> skip
            return None
        from ezphot.imageobjects import ScienceImage
        img = ScienceImage(path=frame, telinfo=None, load=True)
        cat = img.photometry_sex(save=False, verbose=False, visualize=False)
        d = cat.data
        keep = [c for c in d.colnames if c.startswith(("MAG_", "MAGERR_", "SNR_", "FLUX_MAX"))
                or c in ("X_WORLD", "Y_WORLD", "ALPHA_J2000", "DELTA_J2000", "X_IMAGE", "Y_IMAGE",
                         "FLAGS", "CLASS_STAR", "FWHM_WORLD", "FWHM_IMAGE", "ELONGATION")]
        phdu = fits.PrimaryHDU(header=fits.Header(img.header))     # original image header = provenance
        fits.HDUList([phdu, fits.BinTableHDU(Table(d[keep]), name="LDAC_OBJECTS")]).writeto(
            CATD / (Path(frame).name[:-5] + "_cat.fits"), overwrite=True)
        base = Path(frame).name                              # delete the ezphot tree copy (direct path, no crawl)
        try:
            treedir = Path(str(cat.path)).parent
            if str(treedir).startswith(str(EZTREE)) and treedir.is_dir():
                for f in treedir.glob(base + "*"):
                    f.unlink()
        except Exception:
            pass
        names = set(d.colnames)
        rakey = "X_WORLD" if "X_WORLD" in names else "ALPHA_J2000"
        dekey = "Y_WORLD" if "Y_WORLD" in names else "DELTA_J2000"
        df = pd.DataFrame({
            "ALPHA_J2000": np.asarray(d[rakey], float), "DELTA_J2000": np.asarray(d[dekey], float),
            "MAG_APER_3": np.asarray(d["MAG_APER_3"], float),          # ezphot: 10" aperture
            "MAGERR_APER_3": np.asarray(d["MAGERR_APER_3"], float),
            "FLAGS": np.asarray(d["FLAGS"], float), "CLASS_STAR": np.asarray(d["CLASS_STAR"], float),
            "FWHM_WORLD": np.asarray(d["FWHM_WORLD"], float), "ELONGATION": np.asarray(d["ELONGATION"], float)})
        df["SNR_APER_3"] = (np.asarray(d["SNR_APER_3"], float) if "SNR_APER_3" in names
                            else 1.0857 / np.clip(df.MAGERR_APER_3.to_numpy(), 1e-4, None))
        dstar, derr, a0, nstar = OZ.dstar_of(df, tile, filt)
        if not np.isfinite(dstar):
            return None
        unit, night, mjd = OT.meta(frame)
        return dict(filter=filt, unit=unit, night=night, mjd=mjd, tile=tile, n=nstar,
                    dstar=dstar, dstar_err=derr, a_phys=a0, file=frame)
    except Exception as e:                                   # noqa: BLE001
        return dict(filter=frame.split("/")[-3], file=frame, error=str(e)[:200])


def main():
    global SRC, CATD
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=32)
    ap.add_argument("--src", nargs="*", default=None, help="frame-list file(s); default: 2024early no-cat list")
    ap.add_argument("--per-night", type=int, default=0, help="frames per (filter,night); 0 = take all")
    ap.add_argument("--tag", default="ez2024")
    ap.add_argument("--catdir", default=None)
    a = ap.parse_args()
    if a.catdir:
        CATD = Path(a.catdir)
    CATD.mkdir(parents=True, exist_ok=True)
    srcs = a.src if a.src else [str(SRC)]
    frames = []
    for s in srcs:
        frames += [l.strip() for l in Path(s).read_text().splitlines() if l.strip()]
    frames = [f for f in frames if f.split("/")[-3] in GRID]
    if a.per_night > 0:                                       # per-(filter,night) sampling like the pipeline runs
        import random, re
        from collections import defaultdict
        by_fn = defaultdict(list)
        for f in frames:
            m = re.search(r"/(20\d\d-\d\d-\d\d)/", f)
            by_fn[(f.split("/")[-3], m.group(1) if m else "?")].append(f)
        random.seed(20260911)
        frames = []
        for _, fl in by_fn.items():
            frames += random.sample(fl, min(a.per_night, len(fl)))
    rows = []
    donep = OT.FOLDER / f"offset_timeseries_{a.tag}.partial.csv"          # resume: skip already-done frames
    if donep.exists():
        try:
            prev = pd.read_csv(donep)
            if "file" in prev.columns:
                rows = prev.to_dict("records")
        except Exception:
            rows = []
    done = {r.get("file") for r in rows}
    frames = [f for f in frames if f not in done]
    print(f"{len(frames)} frames to process ({len(done)} already done), {a.n_proc} workers", flush=True)
    with Pool(a.n_proc, initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(process, frames), 1):
            if r:
                rows.append(r)
            if i % 200 == 0:
                print(f"  {i}/{len(frames)}, {len(rows)} rows", flush=True)
                pd.DataFrame(rows).to_csv(donep, index=False)
    res = pd.DataFrame(rows)
    if "error" in res.columns:
        err = res[res.error.notna()]
        if len(err):
            print(f"errors: {len(err)}"); print(err.error.str[:80].value_counts().head(5).to_string())
        res = res[res.error.isna()].drop(columns=["error"], errors="ignore")
    res.to_csv(OT.FOLDER / f"offset_timeseries_{a.tag}.csv", index=False)
    if not len(res):
        print("no successful frames"); return
    print(f"\n{len(res)} frames; MJD {res.mjd.min():.0f}..{res.mjd.max():.0f}")
    med = res.groupby("filter").agg(dstar_med=("dstar", "median"), n=("dstar", "size"))
    print(med.reindex([f for f in GRID if f in med.index]).round(2).to_string())
    print(f"\nsaved -> offset_timeseries_{a.tag}.csv in {OT.FOLDER}; catalogs in {CATD}")


if __name__ == "__main__":
    main()
