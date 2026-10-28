#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260911_ezphot_photometry.py  (stage: compare)

Direct ezphot photometry vs the pipeline catalogs, at the same 10" aperture.
For a stratified sample of survey frames: run ezphot AperturePhotometry (img.photometry_sex, in-memory,
apertures [3,5,7,10,15]" -> MAG_APER_3 = 10") on the ORIGINAL frame (read-only on the pipeline; ezphot works
in its own data tree, and the tree copy is deleted per frame), match star-by-star to the pipeline _cat.fits
(MAG_APER_5 = 10"), and compute — on the SAME reference-matched stars, same GAIAXP_CORR_LAMOST synthetic mags —
per-frame ZP and ZP scatter for both photometries, plus the direct star-by-star mag difference.

ezphot catalogs are SAVED separately (OUT/catalogs/<frame>_cat.fits, LDAC_OBJECTS) for the later delta* rerun.

  MPLCONFIGDIR=/tmp/ezphot-matplotlib OMP_NUM_THREADS=1 /home/hhchoi1022/miniconda3/envs/too/bin/python -u \\
      ezphot/analysis/260911_ezphot_photometry.py --per-filter 15 --n-proc 20
"""
from __future__ import annotations

import argparse
import importlib
import os
import random
import re
import shutil
import sys
import warnings
from collections import defaultdict
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
from astropy.stats import SigmaClip  # noqa: E402
from astropy.table import Table      # noqa: E402

sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot")
sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
C = importlib.import_module("260911_colorterm_2026")
R, K = C.R, C.K
XP_WAVE = C.XP_WAVE

SRC = Path("/home/hhchoi1022/ezphot/log/20260911_colorterm_timeseries/all_medium_2025.txt")
OUT = Path("/home/hhchoi1022/ezphot/log/20260911_ezphot_photometry")
(OUT / "catalogs").mkdir(parents=True, exist_ok=True)
EZTREE = Path("/home/hhchoi1022/ezphot/data/scidata")          # ezphot's own working tree (safe to clean)
GRID = R.GRID
PHYS = {}


def _init():
    global PHYS
    warnings.filterwarnings("ignore")
    K._refit_init()
    PHYS = {f: C.load_new_curve(f) for f in GRID}


def clipped(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 8:
        return np.nan, np.nan, 0
    cl = SigmaClip(sigma=3.0, maxiters=5)(v)
    k = ~np.ma.getmaskarray(cl)
    return float(np.median(v[k])), float(np.std(v[k])), int(k.sum())


def ezphot_cat(frame):
    """Run ezphot photometry on the original frame (in-memory); save LDAC copy; clean the tree copy."""
    from ezphot.imageobjects import ScienceImage
    img = ScienceImage(path=frame, telinfo=None, load=True)
    cat = img.photometry_sex(save=False, verbose=False, visualize=False)
    d = cat.data
    savep = OUT / "catalogs" / (Path(frame).name[:-5] + "_cat.fits")
    t = Table(d)
    hdu = fits.BinTableHDU(t, name="LDAC_OBJECTS")
    hdu.writeto(savep, overwrite=True)
    # clean the ezphot tree copy of THIS frame — direct path from the catalog savepath (36TB tree: NEVER crawl)
    base = Path(frame).name
    try:
        treedir = Path(str(cat.path)).parent
        if str(treedir).startswith(str(EZTREE)) and treedir.is_dir():
            for f in treedir.glob(base + "*"):
                f.unlink()
    except Exception:
        pass
    return d


def process(frame):
    step = "start"
    try:
        filt = frame.split("/")[-3]
        tile = frame.split("/")[-4]
        if not (os.path.exists(frame) and os.path.exists(frame[:-5] + "_cat.fits")):
            return None                              # pipeline tree is live; frames can vanish
        step = "pipe_cat"
        with fits.open(frame[:-5] + "_cat.fits") as h:
            dp = h["LDAC_OBJECTS"].data
            pdf = pd.DataFrame({c: np.asarray(dp[c], float) for c in
                                ["ALPHA_J2000", "DELTA_J2000", "MAG_APER_5", "MAGERR_APER_5", "FLAGS", "CLASS_STAR"]})
        pdf = pdf[(pdf.FLAGS <= 0) & (pdf.CLASS_STAR >= 0.8) & (pdf.MAGERR_APER_5 < 0.1)]
        if len(pdf) < 15:
            return None
        # ezphot photometry (10" = MAG_APER_3 of [3,5,7,10,15])
        step = "ezphot_phot"
        d = ezphot_cat(frame)
        cols = d.colnames
        rakey = "X_WORLD" if "X_WORLD" in cols else "ALPHA_J2000"
        dekey = "Y_WORLD" if "Y_WORLD" in cols else "DELTA_J2000"
        edf = pd.DataFrame({
            "ra": np.asarray(d[rakey], float), "dec": np.asarray(d[dekey], float),
            "mag": np.asarray(d["MAG_APER_3"], float), "err": np.asarray(d["MAGERR_APER_3"], float),
            "flags": np.asarray(d["FLAGS"], float), "cs": np.asarray(d["CLASS_STAR"], float)})
        edf = edf[(edf["flags"] <= 0) & (edf["cs"] >= 0.8) & (edf["err"] < 0.1)]
        if len(edf) < 15:
            return None
        # reference synthetic mags (corrected XP x physical curve) on shared stars
        step = "reference"
        ra0, dec0 = float(pdf.ALPHA_J2000.median()), float(pdf.DELTA_J2000.median())
        got = K.get_reference("GAIAXP_CORR_LAMOST", tile, ra0, dec0, 1.34, 0.90)
        if got is None:
            return None
        tbl, coords, _ = got
        rid_all = np.asarray(tbl["id"]).astype(str)
        pcoord = SkyCoord(pdf.ALPHA_J2000.to_numpy(), pdf.DELTA_J2000.to_numpy(), unit="deg")
        ecoord = SkyCoord(edf.ra.to_numpy(), edf.dec.to_numpy(), unit="deg")
        # common stars: ezphot <-> pipeline pairs (<1")
        i_pe, sep_pe, _ = match_coordinates_sky(ecoord, pcoord)
        okm = np.asarray(sep_pe.arcsec < 1.0)
        em = edf[okm].reset_index(drop=True)
        pm = pdf.iloc[i_pe[okm]].reset_index(drop=True)
        dmed, dstd, ndm = clipped(em["mag"].to_numpy() - pm.MAG_APER_5.to_numpy())
        # ZP for BOTH photometries on the SAME common stars vs the same reference synth mags
        step = "reference_zp"
        cc = SkyCoord(pm.ALPHA_J2000.to_numpy(), pm.DELTA_J2000.to_numpy(), unit="deg")
        ic, sc, _ = match_coordinates_sky(cc, coords)
        okc = np.asarray(sc.arcsec < 2.0)
        rid = rid_all[ic[okc]]
        xp = C.tile_xp(tile, set(rid))
        keep = [k for k in range(len(rid)) if rid[k] in xp]
        if len(keep) < 10:
            return None
        fw, ft = PHYS[filt]
        fc = np.vstack([xp[rid[k]][1] for k in keep])
        syn = R.syn_mag(XP_WAVE, fc, fw, ft, 0.0)
        sel = np.where(okc)[0][keep]
        zpe, zpe_std, ne = clipped(syn - em["mag"].to_numpy()[sel])
        zpp, zpp_std, npp = clipped(syn - pm.MAG_APER_5.to_numpy()[sel])
        hdr = fits.getheader(frame)
        return dict(filter=filt, tile=tile, file=frame, n_ez=len(edf), n_pipe=len(pdf),
                    dmag_med=dmed, dmag_std=dstd, n_dmag=ndm,
                    zp_ez=zpe, zpstd_ez=zpe_std, n_zp_ez=ne,
                    zp_pipe=zpp, zpstd_pipe=zpp_std, n_zp_pipe=npp,
                    ezp5_hdr=float(hdr.get("EZP_5", np.nan)), seeing=float(hdr.get("SEEING", np.nan)))
    except Exception as e:                                      # noqa: BLE001
        return dict(filter=frame.split("/")[-3], file=frame, error=f"[{step}] {e}"[:300])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-filter", type=int, default=15)
    ap.add_argument("--n-proc", type=int, default=20)
    a = ap.parse_args()
    frames = [l.strip() for l in SRC.read_text().splitlines() if l.strip()]
    by_f = defaultdict(list)
    for f in frames:
        filt = f.split("/")[-3]
        if filt in GRID:
            by_f[filt].append(f)
    random.seed(20260911)
    sample = []
    for f in GRID:                              # shuffle, then stat only until enough cat-existing frames found
        pool_f = by_f[f][:]
        random.shuffle(pool_f)
        take = []
        for fr in pool_f:
            if os.path.exists(fr[:-5] + "_cat.fits"):
                take.append(fr)
                if len(take) >= a.per_filter:
                    break
        sample += take
    print(f"{len(sample)} frames ({a.per_filter}/filter), {a.n_proc} workers", flush=True)
    rows = []
    with Pool(a.n_proc, initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(process, sample), 1):
            if r:
                rows.append(r)
            if i % 40 == 0:
                print(f"  {i}/{len(sample)}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "compare_ezphot_vs_pipeline_raw.csv", index=False)      # errors included
    if "error" in res.columns:
        err = res[res.error.notna()]
        if len(err):
            print(f"\nerrors: {len(err)}")
            print(err.error.str[:80].value_counts().head(6).to_string())
        res = res[res.error.isna()].drop(columns=["error"])
    res.to_csv(OUT / "compare_ezphot_vs_pipeline.csv", index=False)
    if not len(res):
        print("no successful frames — see compare_ezphot_vs_pipeline_raw.csv"); return
    print(f"\n{len(res)} frames compared")
    print("\n=== 전체 중앙값 ===")
    for c in ["dmag_med", "dmag_std", "zpstd_ez", "zpstd_pipe", "ezp5_hdr"]:
        print(f"  {c:12s}: {res[c].median():+.4f}")
    print("\n=== 필터별 (ezphot 10\" vs pipeline APER_5) ===")
    g = res.groupby("filter").agg(n=("file", "size"), dmag=("dmag_med", "median"), dstd=("dmag_std", "median"),
                                  zpstd_ez=("zpstd_ez", "median"), zpstd_pipe=("zpstd_pipe", "median"))
    g["better"] = np.where(g.zpstd_ez < g.zpstd_pipe, "ezphot", "pipeline")
    g = g.reindex([f for f in GRID if f in g.index])
    print(g.round(4).to_string())
    print(f"\nezphot ZP산포 우세 필터: {(g.better=='ezphot').sum()}/{len(g)}")

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    ax[0].hist(res.dmag_med * 1000, bins=40, color="#1f77b4")
    ax[0].set(xlabel="median(ezphot10\" - pipeAPER_5) [mmag]", title=f"star-by-star offset (med {1000*res.dmag_med.median():+.1f} mmag)")
    ax[1].scatter(res.zpstd_pipe * 1000, res.zpstd_ez * 1000, s=12, alpha=0.6)
    lim = (0, max(np.nanpercentile(res.zpstd_pipe, 98), np.nanpercentile(res.zpstd_ez, 98)) * 1100)
    ax[1].plot(lim, lim, "k--", lw=0.8)
    ax[1].set(xlim=lim, ylim=lim, xlabel="pipeline ZP scatter [mmag]", ylabel="ezphot ZP scatter [mmag]",
              title=f"per-frame ZP scatter (ezphot better: {100*np.mean(res.zpstd_ez<res.zpstd_pipe):.0f}%)")
    ax[2].scatter(res.seeing, res.dmag_med * 1000, s=10, alpha=0.6)
    ax[2].axhline(0, color="k", lw=0.5); ax[2].set(xlabel="seeing [\"]", ylabel="offset [mmag]", title="offset vs seeing")
    for x in ax:
        x.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "F30_ezphot_vs_pipeline.png", dpi=140)
    print(f"\nsaved -> F30_ezphot_vs_pipeline.png, compare_ezphot_vs_pipeline.csv, catalogs/ in {OUT}")


if __name__ == "__main__":
    main()
