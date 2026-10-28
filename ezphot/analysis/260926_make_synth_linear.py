#!/usr/bin/env python
# 260926_make_synth_linear.py — perfectly linear synthetic g-band series (null test for 260926_linearity_pairs.py):
# from each unit 100 s frame -> 1,3,5,10,20 s x 5; used for synth_g5 (units 7DT02/05/09/12/15, frame _0002).
# usage: python 260926_make_synth_linear.py <100 s frames...> <outdir>
"""Perfectly linear synthetic g-band series for several units: from each unit's 100 s frame make 1,3,5,10,20 s x 5
frames (Poisson-consistent extra noise).  Base frame copies are NOT used as the long exposure (all synthetic frames
are independently re-noised versions with s < 1)."""
import sys, numpy as np
from astropy.io import fits
from pathlib import Path
srcs, outdir = sys.argv[1:-1], Path(sys.argv[-1]); outdir.mkdir(parents=True, exist_ok=True)
rn = 1.3
for src in srcs:
    h = fits.getheader(src); d = fits.getdata(src).astype(np.float32)
    g = float(h["EGAIN"]); t0 = float(h["EXPTIME"])
    bad = ~np.isfinite(d); d0 = np.where(bad, 0, d).astype(np.float32)
    lev_e = np.clip(d0, 0, None) * g
    name = Path(src).name
    for t in (1, 3, 5, 10, 20):
        s = t / t0
        for k in range(5):
            rng = np.random.default_rng(1000 * t + k + hash(name) % 1000)
            var_add = (s * (1 - s) * lev_e + (1 - s * s) * rn * rn) / g ** 2
            img = (s * d0 + rng.normal(0, np.sqrt(np.clip(var_add, 0, None)))).astype(np.float32)
            img[bad] = np.nan
            hh = h.copy(); hh["EXPTIME"] = float(t)
            fits.writeto(outdir / name.replace("_100.0s_", f"_{t:05.1f}s_").replace(".fits", f"_r{k}.fits"), img, hh, overwrite=True)
    print("done", name, flush=True)
