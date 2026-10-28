#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260924_bkg_pilot.py — pilot re-photometry to decide the pipeline fix for the 10-arcsec background pedestal
(Step A finding).  A small set of SPSS/CALSPEC frames is re-photometered with ezphot photometry_sex under:
  V0  default (BACKPHOTO_TYPE GLOBAL, BACK_SIZE 64)        — must reproduce the existing catalogues
  V1  LOCAL annulus background (annulus_width_arcsec = 12)
  V2  GLOBAL with BACK_SIZE 32
and evaluated per frame:
  p        : pedestal [ADU/pix] from the 5/10-arcsec aperture pair (XP-free, same estimator as Step A)
  dm(SNR)  : m10(Vk) - m10(V0) for the same stars vs SNR, compared with the analysis-side prediction
             -2.5 log10(1 - p_V0 * A10 / F10)
  res(SNR) : corrected-XP field residual (syn - inst - ZP) profile vs SNR (external check)
Originals are only read; catalogues go to ~/ezphot/log/20260924_stepA/pilot/<variant>/.

  MPLCONFIGDIR=/tmp/ezphot-matplotlib OMP_NUM_THREADS=1 /home/hhchoi1022/miniconda3/envs/too/bin/python -u \\
      ezphot/analysis/260924_bkg_pilot.py --n-proc 16
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
import matplotlib.pyplot as plt     # noqa: E402
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

from astropy.coordinates import SkyCoord, match_coordinates_sky   # noqa: E402
from astropy.io import fits          # noqa: E402
from astropy.table import Table      # noqa: E402

sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot")
sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")

LOG = Path("/home/hhchoi1022/ezphot/log/20260924_stepA")
PILOT = LOG / "pilot"
SCI = Path("/home/hhchoi1022/ezphot/data/scidata/7DT/7DT_C361K_HIGH_1x1")
CAL_META = Path("/home/hhchoi1022/ezphot/log/20260914_calspec_ezphot/photometry_meta.csv")
VARIANTS = {"V0": dict(), "V1": dict(annulus_width_arcsec=12.0), "V2": dict(sex_params={"BACK_SIZE": 32})}
KEEP_PREFIX = ("MAG_", "MAGERR_", "SNR_", "FLUX_MAX", "BACKGROUND")
KEEP_COLS = ("X_WORLD", "Y_WORLD", "ALPHA_J2000", "DELTA_J2000", "X_IMAGE", "Y_IMAGE", "FLAGS", "CLASS_STAR",
             "FWHM_WORLD", "FWHM_IMAGE", "ELONGATION")


def select_frames():
    F = pd.read_csv(LOG / "stepA_frames.csv").dropna(subset=["d_gnd"])
    cm = pd.read_csv(CAL_META)
    calpath = {Path(f).name: f for f in cm.file}
    rows = []
    # SPSS: bands carrying the pedestal signal, both exposure times, several fields
    s = F[(F.dataset == "SPSS") & F["filter"].isin(["m450", "m550", "m650", "m750", "m850"])]
    for (obj, filt, exp), g in s.groupby(["object", "filter", "exptime"]):
        rows.append(dict(dataset="SPSS", object=obj, filter=filt, exptime=exp, name=g.file.iloc[len(g) // 2],
                         unit=g.unit.iloc[len(g) // 2]))
    S = pd.DataFrame(rows)
    S = pd.concat([g.sample(min(len(g), 5), random_state=1) for _, g in S.groupby(["filter", "exptime"])])
    c = F[(F.dataset == "CALSPEC") & F["filter"].isin(["m550", "m650", "m750"])]
    crow = []
    for (obj, filt), g in c.groupby(["object", "filter"]):
        nm = g.file.iloc[len(g) // 2]
        if nm in calpath:
            crow.append(dict(dataset="CALSPEC", object=obj, filter=filt, exptime=g.exptime.iloc[0], name=nm,
                             unit=g.unit.iloc[len(g) // 2], path=calpath[nm]))
    C = pd.DataFrame(crow)
    C = pd.concat([g.sample(min(len(g), 5), random_state=1) for _, g in C.groupby("filter")])
    S["path"] = [str(SCI / r.object / r.unit / r["filter"] / r["name"]) for _, r in S.iterrows()]
    out = pd.concat([S, C], ignore_index=True)
    out = out[[Path(p).exists() for p in out.path]]
    return out


def photometer(path):
    """all variants of one frame sequentially in one worker (ezphot reuses a per-image temp catalogue path)."""
    res = []
    for variant in VARIANTS:
        res.append(_photometer_one(path, variant))
    return res


def _photometer_one(path, variant):
    outp = PILOT / variant / (Path(path).name[:-5] + "_cat.fits")
    if outp.exists():
        return str(outp), "cached"
    try:
        from ezphot.imageobjects import ScienceImage
        img = ScienceImage(path=path, telinfo=None, load=True)
        cat = img.photometry_sex(save=False, verbose=False, visualize=False, **VARIANTS[variant])
        d = cat.data
        keep = [k for k in d.colnames if k.startswith(KEEP_PREFIX) or k in KEEP_COLS]
        phdu = fits.PrimaryHDU(header=fits.Header(img.header))
        fits.HDUList([phdu, fits.BinTableHDU(Table(d[keep]), name="LDAC_OBJECTS")]).writeto(outp, overwrite=True)
        return str(outp), "ok"
    except Exception as e:                                           # noqa: BLE001
        return str(outp), f"error: {e}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n-proc", type=int, default=16)
    a = ap.parse_args()
    for v in VARIANTS:
        (PILOT / v).mkdir(parents=True, exist_ok=True)
    fr = select_frames()
    fr.to_csv(PILOT / "pilot_frames.csv", index=False)
    print(f"{len(fr)} frames ({(fr.dataset == 'SPSS').sum()} SPSS, {(fr.dataset == 'CALSPEC').sum()} CALSPEC) x {len(VARIANTS)} variants", flush=True)
    jobs = list(fr.path)
    with Pool(a.n_proc) as pool:
        for i, rs in enumerate(pool.imap_unordered(photometer, jobs), 1):
            for op, st in rs:
                if st.startswith("error"):
                    print(f"  [{i}/{len(jobs)}] {Path(op).name}: {st[:200]}", flush=True)
            if i % 10 == 0:
                print(f"  [{i}/{len(jobs)}] frames done", flush=True)
    print("photometry done", flush=True)


if __name__ == "__main__":
    main()
