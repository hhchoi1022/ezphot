#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260925_star_profiles.py — where is the real-star deficit?  Stacked radial profiles of REAL faint vs bright stars.

Per pilot frame, after a mean-type background (5-sigma clipped mesh mean, segmentation dilated 4 px; blank sky ~0):
per-pixel mean in annuli out to 30" for every selected star (select_stars_like, V0 positions);
residual(r) = median_faint[ I(r) ] - s * median_bright[ I(r) ],  s = median F5_faint / median F5_bright
  flat negative residual out to 30"   -> the sky around faint stars is locally lower (background-type)
  negative only within ~10", -> 0 out -> faint stars lack wing light relative to bright ones (PSF / pixel-response)
faint: SNR 20-40, bright: SNR 100-150 (as in the metrics).  Output: star_profiles.csv, A12_star_profiles.png
"""
import importlib
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import sep
from astropy.io import fits

warnings.filterwarnings("ignore")
sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
LOG = Path("/home/hhchoi1022/ezphot/log/20260924_stepA")
PILOT = LOG / "pilot"
PIX = 0.505
EDGES = np.array([0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8.5, 10, 12.5, 15, 20, 25, 30]) / PIX   # arcsec -> px
_G = {}


def _init():
    warnings.filterwarnings("ignore")
    _G["TR"] = importlib.import_module("260924_stepA_triage")
    _G["SW"] = importlib.import_module("260925_bkg_recipe_sweep")


def run(args):
    path, dataset = args
    TR, SW = _G["TR"], _G["SW"]
    try:
        from scipy.ndimage import binary_dilation
        data = np.ascontiguousarray(fits.getdata(path).astype(np.float32))
        bad = ~np.isfinite(data); data[bad] = 0.0
        b0 = sep.Background(data, mask=bad, bw=64, bh=64, fw=3, fh=3)
        sep.set_extract_pixstack(10_000_000); sep.set_sub_object_limit(4096)
        _, seg = sep.extract(data - b0.back(), 1.5, err=b0.globalrms, minarea=12, segmentation_map=True, mask=bad)
        back = SW.mesh_bkg(data, binary_dilation(seg > 0, iterations=4) | bad, "mean5")
        sub = data - back
        cat = PILOT / "V0" / (Path(path).name[:-5] + "_cat.fits")
        with fits.open(cat) as h:
            d = h["LDAC_OBJECTS"].data
            xs = np.asarray(d["X_IMAGE"], float); ys = np.asarray(d["Y_IMAGE"], float)       # ezphot catalogues are 0-based
        df = TR.OZ.load_ez(cat) if dataset == "CALSPEC" else TR.AN.load_cat(cat)
        sel = TR.OT.select_stars_like(df.assign(_i=np.arange(len(df))))._i.to_numpy()
        snr = df.SNR_APER_3.to_numpy()[sel]
        x, y = xs[sel], ys[sel]
        ny, nx = data.shape
        inside = (x > 70) & (x < nx - 70) & (y > 70) & (y < ny - 70)
        prof = np.zeros((len(x), len(EDGES) - 1))
        for k in range(len(EDGES) - 1):
            if EDGES[k] == 0:
                s_, _, _ = sep.sum_circle(sub, x, y, EDGES[1], subpix=5)
                a_ = np.pi * EDGES[1] ** 2
            else:
                s_, _, _ = sep.sum_circann(sub, x, y, EDGES[k], EDGES[k + 1], subpix=5)
                a_ = np.pi * (EDGES[k + 1] ** 2 - EDGES[k] ** 2)
            prof[:, k] = s_ / a_
        f5, _, _ = sep.sum_circle(sub, x, y, 2.5 / PIX, subpix=5)
        fa = inside & (snr >= 20) & (snr < 40); br = inside & (snr >= 100) & (snr < 150)
        if fa.sum() < 10 or br.sum() < 5:
            return None
        s = np.median(f5[fa]) / np.median(f5[br])
        pf = np.median(prof[fa], axis=0); pb = np.median(prof[br], axis=0)
        return dict(name=Path(path).name, dataset=dataset, n_faint=int(fa.sum()), n_bright=int(br.sum()), scale=s,
                    **{f"res_{k}": float(pf[k] - s * pb[k]) for k in range(len(EDGES) - 1)},
                    **{f"fa_{k}": float(pf[k]) for k in range(len(EDGES) - 1)},
                    **{f"br_{k}": float(pb[k]) for k in range(len(EDGES) - 1)})
    except Exception as e:                                           # noqa: BLE001
        return dict(name=Path(path).name, error=str(e)[:300])


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fr = pd.read_csv(PILOT / "pilot_frames.csv")
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 8, initializer=_init) as pool:
        rows = [r for r in pool.map(run, [(p, ds) for p, ds in zip(fr.path, fr.dataset)]) if r is not None]
    P = pd.DataFrame(rows)
    fr["name"] = [Path(p).name for p in fr.path]
    P = P.merge(fr[["name", "exptime"]], on="name", how="left")
    P.to_csv(LOG / "star_profiles.csv", index=False)
    if "error" in P.columns:
        print("errors:", P["error"].notna().sum(), P["error"].dropna().head(2).tolist())
        P = P[P["error"].isna()]
    P["grp"] = P.dataset + " " + P.exptime.astype(int).astype(str) + "s"
    rmid = 0.5 * (EDGES[:-1] + EDGES[1:]) * PIX
    rcols = [f"res_{k}" for k in range(len(EDGES) - 1)]
    tab = P.groupby("grp")[rcols].median()
    tab.columns = [f"{r:.1f}\"" for r in rmid]
    pd.set_option("display.width", 250)
    print("residual faint - s*bright [ADU/pix] vs radius:")
    print(tab.round(3).to_string())
    print("\nframes per group:", P.groupby("grp").size().to_dict())
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for grp, row in tab.iterrows():
        ax[0].plot(rmid, row.values, "o-", ms=3, lw=1, label=grp)
    ax[0].axhline(0, color="k", lw=0.6); ax[0].axvspan(2.5, 5.0, color="0.9", zorder=0)
    ax[0].set(xscale="log", xlabel="radius [arcsec]", ylabel="faint − s·bright [ADU/pix]", ylim=(-1.5, 1.0),
              title="(a) residual profile of real faint stars (grey: 5–10″ ring used by p)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    for grp, g in P.groupby("grp"):
        fa = g[[f"fa_{k}" for k in range(len(EDGES) - 1)]].median().values
        brr = g[[f"br_{k}" for k in range(len(EDGES) - 1)]].median().values
        ax[1].plot(rmid, fa / fa[0], "-", lw=1, label=f"{grp} faint")
        ax[1].plot(rmid, brr / brr[0], "--", lw=1, color=ax[1].lines[-1].get_color())
    ax[1].set(xscale="log", yscale="symlog", xlabel="radius [arcsec]", ylabel="I(r)/I(0) (solid faint, dashed bright)",
              title="(b) normalised stacked profiles")
    ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3)
    fig.suptitle("A12  Real faint vs bright star profiles after a mean background (blank sky ≈ 0)", fontsize=10.5)
    fig.tight_layout(); fig.savefig(LOG / "A12_star_profiles.png", dpi=130)
    print("saved")
