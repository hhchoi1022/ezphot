#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260910_allbricqs_process.py

Single-frame processing (no stacking) of the 7DT survey-tile raw frames that contain AllBRICQS quasars, with the BRIDGE
AlertProcessor exactly as in 260723_proces_images_for_Young.py: preprocess -> platesolve -> masks -> background ->
photometry -> photometric calibration (GAIAXP_CORR_LAMOST).  Frame list from
/home/hhchoi1022/ezphot/log/20260910_allbricqs/portal/qso_raw_frames.csv.gz (GW portal query), medium-band frames only.

  MPLCONFIGDIR=/tmp/ezphot-matplotlib OMP_NUM_THREADS=1 /home/hhchoi1022/anaconda3/envs/py311/bin/python \\
      ezphot/analysis/260910_allbricqs_process.py --tiles T05785 T10796 --n-proc 16 [--max-frames 3] [--dry-run]
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import pandas as pd

PORTAL = Path("/home/hhchoi1022/ezphot/log/20260910_allbricqs/portal")
LOGDIR = Path("/home/hhchoi1022/ezphot/log/20260910_allbricqs/process"); LOGDIR.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(time.strftime("[%Y-%m-%d %H:%M:%S] ") + msg, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiles", nargs="+", required=True); ap.add_argument("--n-proc", type=int, default=16)
    ap.add_argument("--max-frames", type=int, default=None, help="smoke test: process only the first N frames of each tile")
    ap.add_argument("--filters", nargs="*", default=None); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    F = pd.read_csv(PORTAL / "qso_raw_frames.csv.gz"); Q = pd.read_csv(PORTAL.parent / "catalog" / "allbricqs_all.csv")
    from bridge.objects import Alert
    from bridge.alertmonitor import AlertProcessor
    summary = []
    for tile in a.tiles:
        d = F[(F.object_name == tile) & F["filter"].str.match(r"^m\d{3}w?$", na=False)].drop_duplicates("filepath").sort_values(["filter", "mjd"])
        if a.filters: d = d[d["filter"].isin(a.filters)]
        if a.max_frames: d = d.head(a.max_frames)
        qso = d.qso.iloc[0] if len(d) else None
        if qso is None: log(f"{tile}: no medium-band frames in the portal list"); continue
        q = Q[Q.name == qso].iloc[0]
        paths = [p for p in d.filepath if os.path.exists(p)]
        log(f"{tile}: quasar {qso} (z={q.z:.3f}) -> {len(paths)} frames, {d['filter'].nunique()} filters, nights {sorted(d.night.unique())}")
        if a.dry_run: continue
        alert = Alert(objname=tile, ra=float(q.ra), dec=float(q.dec), trigger_time="2000-01-01")
        proc = AlertProcessor()
        proc.config.max_workers = a.n_proc; proc.config.batch_size = max(a.n_proc * 4, 16)
        proc.config.single_process["do_DIA"] = False; proc.config.single_process["do_forced_photometry"] = False
        proc.config.photcal["radius_arcsec"] = None        # reference stars from the whole field, as in the CALSPEC processing
        proc.load_images_from_path(paths)
        log(f"{tile}: {len(proc.target_images)} images loaded; max_workers={proc.config.max_workers}, batch_size={proc.config.batch_size}")
        t0 = time.time()
        try:
            res = proc.pipeline_before_stacking(alert)
        except Exception as e:
            log(f"{tile}: pipeline error {e!r}"); res = None
        dt = time.time() - t0
        n_ok = n_fail = None
        try:
            if isinstance(res, (list, tuple)):
                n_ok = sum(1 for r in res if isinstance(r, dict) and r.get("success", False)); n_fail = len(res) - n_ok
                errs = [str(r.get("error"))[:120] for r in res if isinstance(r, dict) and not r.get("success", False)][:5]
                if errs: log(f"{tile}: example errors: {errs}")
        except Exception: pass
        log(f"{tile}: done in {dt/60:.1f} min; success {n_ok}, failed {n_fail}")
        summary.append(dict(tile=tile, qso=qso, n_frames=len(paths), n_ok=n_ok, n_fail=n_fail, minutes=round(dt / 60, 1)))
        for img in proc.target_images:
            try: img.clear()
            except Exception: pass
        del proc; gc.collect()
    if summary: pd.DataFrame(summary).to_csv(LOGDIR / f"summary_{time.strftime('%Y%m%d_%H%M%S')}.csv", index=False)


if __name__ == "__main__":
    main()
