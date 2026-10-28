#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260911_m400_timeseries.py

Time evolution of the m400 field colour term over 2026-01/02 (all 7DT13), for the three curves
nominal / physical / physical+lab-delta_f (corrected XP).  Samples up to PER_NIGHT frames per night from
20260911_colorterm_2026/m400_frames.txt.  Reuses the matching + XP machinery of 260911_colorterm_2026.

  MPLCONFIGDIR=/tmp/ezphot-matplotlib OMP_NUM_THREADS=1 /home/hhchoi1022/miniconda3/envs/too/bin/python -u \\
      ezphot/analysis/260911_m400_timeseries.py [--per-night 8]
"""
from __future__ import annotations

import argparse
import importlib
import random
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402
import pandas as pd                 # noqa: E402
from astropy.coordinates import SkyCoord, match_coordinates_sky   # noqa: E402
from astropy.stats import SigmaClip                                # noqa: E402
from astropy.time import Time                                      # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
C = importlib.import_module("260911_colorterm_2026")
L = importlib.import_module("260911_colorterm_labdelta")
R, K = C.R, C.K
OUT, TS, XP_WAVE = C.OUT, C.TS, C.XP_WAVE
FILT = "m400"
LAB = L.LAB_DELTA[FILT]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--per-night", type=int, default=8)
    a = ap.parse_args()
    K._refit_init()
    ph = C.load_new_curve(FILT)
    curves = {"nominal": R.load_curve(FILT), "physical": ph, "physlab": L.shift_curve(ph[0], ph[1], LAB)}

    frames = [l.strip() for l in (OUT / "m400_frames.txt").read_text().splitlines() if l.strip()]
    by_night = defaultdict(list)
    for f in frames:
        m = re.search(r"/processed/(2026-\d\d-\d\d)/", f)
        if m:
            by_night[m.group(1)].append(f)
    random.seed(20260911)
    sample = []
    for night, fl in by_night.items():
        sample += random.sample(fl, min(a.per_night, len(fl)))
    print(f"{len(sample)} m400 frames over {len(by_night)} nights")

    rows = []
    for i, fr in enumerate(sample, 1):
        tile = fr.split("/")[-4]
        mm = re.search(r"_(\d{8})_(\d{6})_", fr)
        mjd = Time(f"{mm.group(1)[:4]}-{mm.group(1)[4:6]}-{mm.group(1)[6:]}T"
                   f"{mm.group(2)[:2]}:{mm.group(2)[2:4]}:{mm.group(2)[4:]}").mjd if mm else np.nan
        try:
            d = C.read_cat(fr[:-5] + "_cat.fits")
        except Exception:
            continue
        q = (d.CLASS_STAR >= 0.8) & (d.FLAGS <= 0) & (d.MAGERR_APER_3 < 0.1) & (d.SNR_APER_3 > 10) & (d.SNR_APER_3 < 150)
        d = d[q]
        if len(d) < 15:
            continue
        ra0, dec0 = float(np.median(d.ALPHA_J2000)), float(np.median(d.DELTA_J2000))
        got = K.get_reference("GAIAXP_CORR_LAMOST", tile, ra0, dec0, 1.34, 0.90)
        if got is None:
            continue
        tbl, coords, _ = got
        magcol, _ = K.resolve_mag_column(FILT, tbl.colnames, "GAIAXP_CORR_LAMOST")
        if magcol is None:
            continue
        src = SkyCoord(d.ALPHA_J2000.to_numpy(), d.DELTA_J2000.to_numpy(), unit="deg")
        idx, d2d, _ = match_coordinates_sky(src, coords)
        ok = np.asarray(d2d.arcsec < 2.0)
        if ok.sum() < 15:
            continue
        mr = tbl[idx[ok]]; di = d[ok]
        rid = np.asarray(mr["id"]).astype(str)
        bprp = np.asarray(mr["bp-rp"], float) if "bp-rp" in mr.colnames else np.full(len(mr), np.nan)
        xp = C.tile_xp(tile, set(rid))
        keepmask = np.array([r in xp for r in rid])
        if keepmask.sum() < 15:
            continue
        fc = np.array([xp[r][1] for r in rid[keepmask]])
        inst = di.MAG_APER_3.to_numpy()[keepmask]; bp = bprp[keepmask]
        sig = np.sqrt(di.MAGERR_APER_3.to_numpy()[keepmask] ** 2 + 0.01 ** 2)
        rec = dict(file=Path(fr).name, tile=tile, mjd=float(mjd), night=re.search(r"(2026-\d\d-\d\d)", fr).group(1),
                   n=int(keepmask.sum()))
        for cn, cv in curves.items():
            zp = R.syn_mag(XP_WAVE, fc, cv[0], cv[1], 0.0) - inst
            fin = np.isfinite(zp) & np.isfinite(bp) & np.isfinite(sig)
            cl = SigmaClip(sigma=3.0, maxiters=5)(zp[fin])
            keep = np.where(fin)[0][~np.ma.getmaskarray(cl)]
            if cn == "nominal":
                rec["zp_med"] = float(np.nanmedian(zp[keep]))
            aa, ae, bb, nn = R.weighted_linfit(bp[keep], zp[keep] - np.nanmedian(zp[keep]), sig[keep])
            rec[f"a_{cn}"] = aa
        rows.append(rec)
        if i % 40 == 0:
            print(f"  {i}/{len(sample)}: {len(rows)} results", flush=True)
    res = pd.DataFrame(rows).sort_values("mjd")
    res.to_csv(OUT / "m400_timeseries.csv", index=False)
    print(f"\n{len(res)} m400 frame color terms; nights {res.night.nunique()}")
    nightly = res.groupby("night").agg(mjd=("mjd", "median"), a_nominal=("a_nominal", "median"),
                                       a_physical=("a_physical", "median"), a_physlab=("a_physlab", "median"),
                                       zp=("zp_med", "median"), n=("a_nominal", "size"))
    print("\nnightly median colour term (nominal / physical / physlab):")
    print(nightly[["a_nominal", "a_physical", "a_physlab", "n"]].round(4).to_string())

    col = {"nominal": "#1f77b4", "physical": "#d62728", "physlab": "#2ca02c"}
    fig, ax = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    for cn in ("nominal", "physical", "physlab"):
        ax[0].scatter(res.mjd, res[f"a_{cn}"], s=8, color=col[cn], alpha=0.25)
        ax[0].plot(nightly.mjd, nightly[f"a_{cn}"], "o-", color=col[cn], ms=4, lw=1.3, label=cn)
    ax[0].axhline(0, color="k", lw=0.5); ax[0].set(ylabel="m400 colour term a(BP-RP)",
                                                   title=f"m400 (7DT13) colour term over 2026-01/02 ({len(res)} frames): nightly median + per-frame")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].scatter(res.mjd, res.zp_med, s=8, color="#555", alpha=0.3)
    ax[1].plot(nightly.mjd, nightly.zp, "o-", color="k", ms=4, lw=1.3)
    ax[1].set(xlabel="MJD", ylabel="m400 frame ZP (nominal)", title="m400 zero point over time (throughput/extinction)")
    ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(TS / "figures" / "F20_m400_timeseries.png", dpi=150)
    print("saved -> m400_timeseries.csv, figures/F20_m400_timeseries.png")


if __name__ == "__main__":
    main()
