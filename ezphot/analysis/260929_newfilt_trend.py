#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260929_newfilt_trend.py — per (curve variant, filter) summary of the 2026 new-generation-filter field colour terms
(260927_newfilt_colorterm.py output), as used in section 8.5 of the curve catalog.

Per (variant, filter): nightly medians of the per-image colour term a0 (nights with >= 2 images, mmag/mag);
  median      = median of the nightly medians
  scatter     = 1.4826 x MAD of the nightly medians
  slope       = unweighted straight line of the nightly medians against time, per 30 days (>= 5 nights), with its error
  span, first/last UTC date of the nights used
Also prints, for manufacturer·beam, the inverse-variance weighted mean slope over the filters with a slope.
  python 260929_newfilt_trend.py [frames.csv] [out.csv]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.time import Time

OUT = Path("/home/hhchoi1022/ezphot/log/20260927_newfilt")


def mad(v):
    return 1.4826 * np.median(np.abs(v - np.median(v)))


def summarise(F):
    rows = []
    for (var, filt), g in F.groupby(["variant", "filter"]):
        N = g.groupby("night").agg(m=("a0", lambda v: 1e3 * np.median(v)), n=("a0", "size"), mjd=("mjd", "median"))
        N = N[N.n >= 2]
        r = dict(variant=var, filter=filt, n_nights=len(N), median=round(N.m.median(), 1),
                 scatter=round(mad(N.m), 1) if len(N) >= 5 else np.nan, p=int(filt[1:4]))
        if len(N) >= 5:
            x = (N.mjd - N.mjd.min()) / 30.0
            c, cov = np.polyfit(x, N.m, 1, cov=True)
            r.update(slope_per_month=round(c[0], 2), slope_err=round(float(np.sqrt(cov[0, 0])), 2),
                     span_days=round(N.mjd.max() - N.mjd.min(), 0),
                     first=Time(N.mjd.min(), format="mjd").iso[:10], last=Time(N.mjd.max(), format="mjd").iso[:10])
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "colorterm_newfilt_frames.csv"
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT / "trend_summary.csv"
    T = summarise(pd.read_csv(src))
    T.to_csv(dst, index=False)
    pd.set_option("display.width", 220)
    W = T.pivot_table(index="filter", columns="variant", values="median")
    W["best"] = W.abs().idxmin(axis=1)
    print(W.round(1).to_string())
    mb = T[(T.variant == "manufacturer_beam") & T.slope_err.notna()]
    w = 1 / mb.slope_err ** 2
    print(f"manufacturer_beam weighted mean slope over {len(mb)} filters: {np.sum(w * mb.slope_per_month) / w.sum():+.2f} "
          f"+- {1 / np.sqrt(w.sum()):.2f} mmag/mag per month")


if __name__ == "__main__":
    main()
