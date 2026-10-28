#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260911_tilestd_process.py

Process the survey-tile SINGLE frames (no coadds) that contain a CALSPEC standard, so the standard and the
field reference stars get photometered + calibrated the same way the 2024-08 standard fields were.
Four standards fall in observed tiles: HZ4/T22272, LDS749B/T19136, WD2039-682/T01485, BD-11_3759/T14891.

Same BRIDGE AlertProcessor path as 260910_allbricqs_process.py (radius_arcsec=None -> whole-field reference
stars).  Writes .cat/.refcat/header ZP next to each frame (in-place, as the pipeline does).

  MPLCONFIGDIR=/tmp/ezphot-matplotlib OMP_NUM_THREADS=1 /home/hhchoi1022/miniconda3/envs/too/bin/python \\
      ezphot/analysis/260911_tilestd_process.py --standards HZ4 --n-proc 8 [--max-frames 6] [--dry-run]
"""
from __future__ import annotations

import argparse
import gc
import glob
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import pandas as pd

# portalocker fails to flock the SExtractor param file on this NFS config (EBADF); the lock is only defensive
# against concurrent param-file writes, and we bound concurrency ourselves -> make it a no-op.
import portalocker


class _NoLock:
    def __init__(self, *a, **k): pass
    def __enter__(self): return None
    def __exit__(self, *a): return False


portalocker.Lock = _NoLock
portalocker.lock = lambda *a, **k: None
portalocker.unlock = lambda *a, **k: None

ROOT = "/lyman/data1/processed_1x1_gain2750"
LOGDIR = Path("/home/hhchoi1022/ezphot/log/20260911_tilestd"); LOGDIR.mkdir(parents=True, exist_ok=True)
META = "/lyman/data1/Calibration/latest_CALSPEC/merged_meta.csv"
SYN = "/home/hhchoi1022/ezphot/log/20260908_zp_color_check/calspec_synphot.csv"
STD_TILE = {"HZ4": "T22272", "LDS749B": "T19136", "WD2039-682": "T01485", "BD-11_3759": "T14891"}


def log(m):
    print(time.strftime("[%Y-%m-%d %H:%M:%S] ") + m, flush=True)


def std_coords():
    meta = pd.read_csv(META).set_index("Name")
    syn = pd.read_csv(SYN).drop_duplicates("object")[["object", "spec"]]
    out = {}
    for o, sp in zip(syn.object, syn.spec):
        if o not in STD_TILE:
            continue
        nm = re.split(r"_(mod|stis|stisnic|stiswfc|nic)", sp)[0]
        r = meta.loc[nm]; dt = 2025.5 - 2016.0
        ra = r.radeg + (r["PM_R.A."] / 3.6e6) * dt / np.cos(np.radians(r.decdeg))
        dec = r.decdeg + (r["PM_Decl."] / 3.6e6) * dt
        out[o] = (float(ra), float(dec))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--standards", nargs="+", default=list(STD_TILE))
    ap.add_argument("--filters", nargs="*", default=None)
    ap.add_argument("--n-proc", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=None, help="smoke test: first N frames only")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    coords = std_coords()
    from bridge.objects import Alert
    from bridge.alertmonitor import AlertProcessor
    summary = []
    for obj in a.standards:
        tile = STD_TILE[obj]; ra, dec = coords[obj]
        frames = sorted(f for f in glob.glob(f"{ROOT}/{tile}/*/m*/*.fits") if not f.endswith(".com.fits"))
        if a.filters:
            frames = [f for f in frames if f.split("/")[-2] in a.filters]
        if a.max_frames:
            frames = frames[:a.max_frames]
        paths = [p for p in frames if os.path.exists(p)]
        filts = sorted({p.split("/")[-2] for p in paths})
        log(f"{obj} tile {tile} (ra={ra:.4f} dec={dec:.4f}): {len(paths)} single frames, {len(filts)} filters")
        if a.dry_run:
            continue
        alert = Alert(objname=tile, ra=ra, dec=dec, trigger_time="2000-01-01")
        proc = AlertProcessor()
        proc.config.max_workers = a.n_proc; proc.config.batch_size = max(a.n_proc * 4, 16)
        proc.config.single_process["do_DIA"] = False
        proc.config.single_process["do_forced_photometry"] = False
        proc.config.photcal["radius_arcsec"] = None
        proc.load_images_from_path(paths)
        log(f"{obj}: {len(proc.target_images)} images loaded")
        t0 = time.time()
        try:
            res = proc.pipeline_before_stacking(alert)
        except Exception as e:                       # noqa: BLE001
            log(f"{obj}: pipeline error {e!r}"); res = None
        dt = time.time() - t0
        n_ok = None
        if isinstance(res, (list, tuple)):
            n_ok = sum(1 for r in res if isinstance(r, dict) and r.get("success", False))
            errs = [str(r.get("error"))[:120] for r in res if isinstance(r, dict) and not r.get("success", False)][:3]
            if errs: log(f"{obj}: example errors: {errs}")
        n_cat = sum(os.path.exists(p + ".cat") for p in paths)
        log(f"{obj}: done in {dt/60:.1f} min; success {n_ok}; .cat written {n_cat}/{len(paths)}")
        summary.append(dict(object=obj, tile=tile, n_frames=len(paths), n_ok=n_ok, n_cat=n_cat, minutes=round(dt/60, 1)))
        for img in proc.target_images:
            try: img.clear()
            except Exception: pass
        del proc; gc.collect()
    if summary:
        pd.DataFrame(summary).to_csv(LOGDIR / f"process_summary_{time.strftime('%Y%m%d_%H%M%S')}.csv", index=False)
        log("summary:\n" + pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
