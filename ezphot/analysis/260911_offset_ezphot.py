#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260911_offset_ezphot.py

Paired delta* comparison on the SAME frames: ezphot direct photometry (10" = MAG_APER_3 of [3,5,7,10,15])
vs the pipeline catalog at the SAME aperture (10" = MAG_APER_5).  Uses the 228 ezphot catalogs saved by
260911_ezphot_photometry.py.  Same solver/selection as the delta* time series (select_stars-aligned cuts,
1 nm grid, linear root).  Answers: does the choice of photometry change the derived filter offset?

Outputs under 20260911_ezphot_photometry/ (separate folder from the pipeline-based delta* results).

  MPLCONFIGDIR=/tmp/ezphot-matplotlib OMP_NUM_THREADS=1 /home/hhchoi1022/miniconda3/envs/too/bin/python -u \\
      ezphot/analysis/260911_offset_ezphot.py --n-proc 16
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
from astropy.coordinates import SkyCoord, match_coordinates_sky   # noqa: E402
from astropy.io import fits          # noqa: E402

sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
OT = importlib.import_module("260911_offset_timeseries")           # solve_offset, select_stars_like, C/K/R
C, K, R = OT.C, OT.K, OT.R
XP_WAVE = OT.XP_WAVE
GRID = OT.GRID

EPH = Path("/home/hhchoi1022/ezphot/log/20260911_ezphot_photometry")
CATD = EPH / "catalogs"
PHYS = {}


def _init():
    global PHYS
    warnings.filterwarnings("ignore")
    K._refit_init()
    PHYS = {f: C.load_new_curve(f) for f in GRID}


def load_ez(catpath):
    """ezphot catalog -> the select_stars_like schema (10\" = MAG_APER_3; SNR from MAGERR if absent)."""
    with fits.open(catpath) as h:
        d = h["LDAC_OBJECTS"].data
        names = set(d.columns.names)
        rakey = "X_WORLD" if "X_WORLD" in names else "ALPHA_J2000"
        dekey = "Y_WORLD" if "Y_WORLD" in names else "DELTA_J2000"
        out = {"ALPHA_J2000": np.asarray(d[rakey], float), "DELTA_J2000": np.asarray(d[dekey], float),
               "MAG_APER_3": np.asarray(d["MAG_APER_3"], float),
               "MAGERR_APER_3": np.asarray(d["MAGERR_APER_3"], float),
               "FLAGS": np.asarray(d["FLAGS"], float), "CLASS_STAR": np.asarray(d["CLASS_STAR"], float),
               "FWHM_WORLD": np.asarray(d["FWHM_WORLD"], float), "ELONGATION": np.asarray(d["ELONGATION"], float)}
        out["SNR_APER_3"] = (np.asarray(d["SNR_APER_3"], float) if "SNR_APER_3" in names
                             else 1.0857 / np.clip(out["MAGERR_APER_3"], 1e-4, None))
    return pd.DataFrame(out)


def load_pipe10(catpath):
    """pipeline catalog remapped so that the 10\" aperture (APER_5) sits in the MAG_APER_3 slots."""
    with fits.open(catpath) as h:
        d = h["LDAC_OBJECTS"].data
        out = {c: np.asarray(d[c], float) for c in
               ["ALPHA_J2000", "DELTA_J2000", "FLAGS", "CLASS_STAR", "FWHM_WORLD", "ELONGATION"]}
        out["MAG_APER_3"] = np.asarray(d["MAG_APER_5"], float)
        out["MAGERR_APER_3"] = np.asarray(d["MAGERR_APER_5"], float)
        out["SNR_APER_3"] = np.asarray(d["SNR_APER_5"], float)
    return pd.DataFrame(out)


def dstar_of(df, tile, filt):
    d = OT.select_stars_like(df)
    if len(d) < 15:
        return (np.nan,) * 3 + (0,)
    ra0, dec0 = float(np.median(d.ALPHA_J2000)), float(np.median(d.DELTA_J2000))
    got = K.get_reference("GAIAXP_CORR_LAMOST", tile, ra0, dec0, 1.34, 0.90)
    if got is None:
        return (np.nan,) * 3 + (0,)
    tbl, coords, _ = got
    src = SkyCoord(d.ALPHA_J2000.to_numpy(), d.DELTA_J2000.to_numpy(), unit="deg")
    idx, d2d, _ = match_coordinates_sky(src, coords)
    ok = np.asarray(d2d.arcsec < 2.0)
    if ok.sum() < 15:
        return (np.nan,) * 3 + (0,)
    mr = tbl[idx[ok]]; di = d[ok]
    rid = np.asarray(mr["id"]).astype(str)
    bprp = np.asarray(mr["bp-rp"], float)
    xp = C.tile_xp(tile, set(rid))
    keep = [k for k in range(len(rid)) if rid[k] in xp]
    if len(keep) < 15:
        return (np.nan,) * 3 + (0,)
    fc = np.vstack([xp[rid[k]][1] for k in keep])
    inst = di.MAG_APER_3.to_numpy()[keep]
    bp = bprp[keep]
    sig = np.sqrt(di.MAGERR_APER_3.to_numpy()[keep] ** 2 + 0.01 ** 2)
    dstar, derr, a0 = OT.solve_offset(*PHYS[filt], fc, inst, bp, sig)
    return dstar, derr, a0, len(keep)


def process(frame):
    try:
        filt = frame.split("/")[-3]; tile = frame.split("/")[-4]
        ezcat = CATD / (Path(frame).name[:-5] + "_cat.fits")
        pcat = frame[:-5] + "_cat.fits"
        if not (ezcat.exists() and os.path.exists(pcat)):
            return None
        de, ee, ae, ne = dstar_of(load_ez(ezcat), tile, filt)
        dp, ep, apn, npi = dstar_of(load_pipe10(pcat), tile, filt)
        if not (np.isfinite(de) and np.isfinite(dp)):
            return None
        return dict(filter=filt, tile=tile, file=frame, dstar_ez=de, err_ez=ee, n_ez=ne,
                    dstar_pipe10=dp, err_pipe10=ep, n_pipe=npi)
    except Exception as e:                                          # noqa: BLE001
        return dict(filter=frame.split("/")[-3], file=frame, error=str(e)[:200])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n-proc", type=int, default=16)
    a = ap.parse_args()
    files = pd.read_csv(EPH / "compare_ezphot_vs_pipeline.csv").file.tolist()
    print(f"{len(files)} paired frames, {a.n_proc} workers", flush=True)
    rows = []
    with Pool(a.n_proc, initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(process, files), 1):
            if r:
                rows.append(r)
            if i % 40 == 0:
                print(f"  {i}/{len(files)}", flush=True)
    res = pd.DataFrame(rows)
    if "error" in res.columns:
        err = res[res.error.notna()]
        if len(err):
            print(f"errors: {len(err)}"); print(err.error.str[:80].value_counts().head(4).to_string())
        res = res[res.error.isna()].drop(columns=["error"])
    res.to_csv(EPH / "dstar_paired_ez_vs_pipe10.csv", index=False)
    print(f"\n{len(res)} paired delta* frames")
    res["ddiff"] = res.dstar_ez - res.dstar_pipe10
    g = res.groupby("filter").agg(n=("file", "size"), dstar_ez=("dstar_ez", "median"),
                                  dstar_pipe10=("dstar_pipe10", "median"), ddiff=("ddiff", "median"),
                                  ddiff_mad=("ddiff", lambda v: float(np.median(np.abs(v - np.median(v))))))
    g = g.reindex([f for f in GRID if f in g.index])
    print(g.round(3).to_string())
    blue = g.loc[[f for f in ["m400", "m425", "m450", "m475", "m500", "m525"] if f in g.index]]
    print(f"\n청색 |Δδ*| 중앙값: {blue.ddiff.abs().median():.3f} nm | 전체: {g.ddiff.abs().median():.3f} nm")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    piv = [int(f[1:4]) for f in g.index]
    ax[0].errorbar(piv, g.ddiff, yerr=g.ddiff_mad / np.sqrt(g.n), fmt="o-", ms=5, capsize=2, color="#d62728")
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set(xlabel="pivot [nm]", ylabel="Δδ* = δ*(ezphot 10\") − δ*(pipeline 10\") [nm]",
              title=f"photometry choice effect on δ* — {len(res)} paired frames")
    sc = ax[1].scatter(res.dstar_pipe10, res.dstar_ez, s=10, alpha=0.5,
                       c=[int(f[1:4]) for f in res["filter"]], cmap="rainbow")
    lim = np.nanpercentile(np.r_[res.dstar_pipe10, res.dstar_ez], [1, 99])
    ax[1].plot(lim, lim, "k--", lw=0.8)
    ax[1].set(xlim=lim, ylim=lim, xlabel="δ* pipeline 10\" [nm]", ylabel="δ* ezphot 10\" [nm]", title="frame-by-frame")
    plt.colorbar(sc, ax=ax[1], label="pivot [nm]")
    for x in ax:
        x.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(EPH / "F31_dstar_paired.png", dpi=140)
    print(f"\nsaved -> F31_dstar_paired.png, dstar_paired_ez_vs_pipe10.csv in {EPH}")


if __name__ == "__main__":
    main()
