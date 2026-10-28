#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260911_backfill_headers.py

Backfill the original-image header (as the Primary HDU) into the early catalogs_ezall/*_cat.fits that were
written before header-saving was added.  Frame paths come from the ezall partial CSV; catalogs already carrying
a header (EXPTIME present) are skipped, so this is safe to run alongside the resumed main run.

  /home/hhchoi1022/miniconda3/envs/too/bin/python -u ezphot/analysis/260911_backfill_headers.py --n-proc 8
"""
from __future__ import annotations

import argparse
import os
import warnings
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from astropy.io import fits

warnings.filterwarnings("ignore")
CATD = Path("/home/hhchoi1022/ezphot/log/20260911_ezphot_photometry/catalogs_ezall")
PARTIAL = Path("/home/hhchoi1022/ezphot/log/20260911_offset_timeseries/offset_timeseries_ezall.partial.csv")


def backfill(frame):
    try:
        catp = CATD / (Path(frame).name[:-5] + "_cat.fits")
        if not catp.exists():
            return "nocat"
        with fits.open(catp) as h:
            if h[0].header.get("EXPTIME") is not None:
                return "has"
            tbl = h["LDAC_OBJECTS"].copy()
        if not os.path.exists(frame):
            return "noorig"
        hdr = fits.getheader(frame)
        fits.HDUList([fits.PrimaryHDU(header=hdr), tbl]).writeto(catp, overwrite=True)
        return "done"
    except Exception:                                          # noqa: BLE001
        return "err"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n-proc", type=int, default=8)
    a = ap.parse_args()
    p = pd.read_csv(PARTIAL)
    files = sorted(set(p.file.dropna()))
    print(f"{len(files)} catalogs to check", flush=True)
    from collections import Counter
    cnt = Counter()
    with Pool(a.n_proc) as pool:
        for i, s in enumerate(pool.imap_unordered(backfill, files, chunksize=16), 1):
            cnt[s] += 1
            if i % 2000 == 0:
                print(f"  {i}/{len(files)} {dict(cnt)}", flush=True)
    print("result:", dict(cnt))


if __name__ == "__main__":
    main()
