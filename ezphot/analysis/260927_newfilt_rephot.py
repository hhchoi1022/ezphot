#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260927_newfilt_rephot.py — photometer 2026 survey frames of the new-generation (in-between) medium-band filters with the
fixed ezphot background (same routine as 260927_std_rephot.process: source mask 5 sigma x3, photutils per-mesh 5-sigma mean,
box 64, filter 3; photometry_sex, 10" = MAG_APER_3).

Frames: /lyman/data2/processed/2026-*/<tile>/<filter>/singles/*.fits for the filters in NEW (read-only; img.savedir points to
a scratch work dir).  At most --cap frames per (filter, night), evenly spaced in time.
Output: ~/ezphot/log/20260927_newfilt/catalogs/<name>_cat.fits, frames.csv (selected list), rephot_meta.csv.

  MPLCONFIGDIR=/tmp/ezphot-matplotlib OMP_NUM_THREADS=1 python -u ezphot/analysis/260927_newfilt_rephot.py --n-proc 36 --cap 8
"""
import argparse
import importlib
import os
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
SR = importlib.import_module("260927_std_rephot")          # process(), portalocker patch
PROC = Path("/lyman/data2/processed")
OUT = Path("/home/hhchoi1022/ezphot/log/20260927_newfilt")
NEW = ["m386", "m438", "m466w", "m483", "m512", "m534", "m561", "m586", "m615", "m640", "m661", "m692w", "m710w",
       "m769w", "m832w"]


def frame_list(since="2026-01-01"):
    rows = []
    for n in sorted(d for d in os.listdir(PROC) if re.match(r"\d{4}-\d\d-\d\d$", d) and d >= since):
        for t in os.listdir(PROC / n):
            if not t.startswith("T"):
                continue
            for f in NEW:
                q = PROC / n / t / f / "singles"
                if q.is_dir():
                    for x in sorted(os.listdir(q)):
                        if x.endswith(".fits"):
                            p = x.split("_")
                            rows.append(dict(filter=f, night=n, tile=t, unit=p[2], file=str(q / x), name=x))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=36)
    ap.add_argument("--cap", type=int, default=8)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    SR.CATD = OUT / "catalogs"; SR.CATD.mkdir(parents=True, exist_ok=True)
    SR.WORK = Path(str(SR.WORK) + "_newfilt"); SR.WORK.mkdir(parents=True, exist_ok=True)
    F = frame_list()
    F.to_csv(OUT / "frames_all.csv", index=False)
    F = F.sort_values("name")
    S = pd.concat([g.iloc[np.linspace(0, len(g) - 1, min(a.cap, len(g))).round().astype(int)]
                   for _, g in F.groupby(["filter", "night"])], ignore_index=True)
    S.to_csv(OUT / "frames.csv", index=False)
    SR.log(f"{len(F)} frames found, {len(S)} selected (cap {a.cap} per filter-night): {S.groupby('filter').size().to_dict()}")
    todo = [("SURVEY", r.file, r.tile, r["filter"], r.unit) for _, r in S.iterrows()]
    rows, t0 = [], time.time()
    with Pool(a.n_proc) as pool:
        for i, r in enumerate(pool.imap_unordered(SR.process, todo, chunksize=1), 1):
            rows.append(r)
            if i % 200 == 0 or i == len(todo):
                SR.log(f"  {i}/{len(todo)}  {(time.time() - t0) / 60:.1f} min")
                pd.DataFrame(rows).to_csv(OUT / "rephot_meta.partial.csv", index=False)
    M = pd.DataFrame(rows)
    M.to_csv(OUT / "rephot_meta.csv", index=False)
    SR.log(f"status: {M.status.str[:40].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
