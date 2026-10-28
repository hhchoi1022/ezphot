#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260925_bkg_recipe_sweep.py — find a background recipe that removes the 10-arcsec pedestal at its root
(no post-hoc aperture correction).  Read-only on images.

Per pilot frame: detection with the SExtractor-like mode background (sep; mesh 64, filter 3, 1.5 sigma,
minarea 12).  Then photutils Background2D (mesh 64, filter 3) for every (mask, estimator) recipe:
  masks      : seg4   = segmentation dilated 4 px
               ellK   = segmentation + each object's ellipse scaled by K (sep.mask_ellipse), K in {3, 6, 12, 24}
  estimators : mode (SExtractorBackground), median, mean3 (3-sigma clipped mean), mean5 (5-sigma clipped mean)
Aperture sums with sep at the ezphot V0 catalogue positions (5 and 10 arcsec) after subtracting each background:
  p_star    : residual pedestal from the 5/10 aperture growth of isolated stars (want 0)
  faint     : corrected-XP field residual at SNR 20-25 minus SNR >= 110 (want 0; >0 = faint stars too bright)
  blank10   : mean flux per pixel in blank 10-arcsec apertures (want 0)
  dzp       : ZP change vs the sep mode baseline
Output: bkg_recipe_sweep.csv, A10_bkg_recipes.png
"""
import importlib
import sys
import time
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
R5, R10 = 2.5 / PIX, 5.0 / PIX
A5, A10 = np.pi * R5 ** 2, np.pi * R10 ** 2
MASKS = ["seg4", "ell6", "ell12", "ell24"]
ESTS = ["mode", "mean3", "mean5"]
_G = {}


def _init():
    warnings.filterwarnings("ignore")
    TR = importlib.import_module("260924_stepA_triage")
    TR.AN._init()
    _G["TR"] = TR
    _G["curves"] = TR.TC.build_totals()[("manufacturer", "nobeam")]


def mesh_bkg(data, mask, est, box=64, filt=3, max_masked=0.9):
    """SExtractor-style mesh background, vectorised with one sort per box: iterative k-sigma clipping around the
    median using cumulative sums over the sorted, contiguous valid range; then the estimator, 3x3 median filter of
    the mesh, cubic interpolation to full resolution.
    est: 'mode' (2.5 med - 1.5 mean if |mean-med|/sd<0.3 else med; SExtractor rule), 'median', 'mean3', 'mean5'."""
    from scipy.ndimage import median_filter, zoom
    ny, nx = data.shape
    nby, nbx = ny // box, nx // box
    d = data[:nby * box, :nbx * box].reshape(nby, box, nbx, box).transpose(0, 2, 1, 3).reshape(nby * nbx, box * box)
    m = mask[:nby * box, :nbx * box].reshape(nby, box, nbx, box).transpose(0, 2, 1, 3).reshape(nby * nbx, box * box)
    v = np.sort(np.where(m, np.nan, d).astype(np.float32), axis=1)          # NaN sorted to the end
    nvalid = np.isfinite(v).sum(axis=1)
    ctr = np.nanmedian(v[:, ::16], axis=1).astype(np.float32)                 # centre values -> float32 cumsums are exact enough
    z = np.where(np.isfinite(v), v - ctr[:, None], 0.0).astype(np.float32)
    cs1 = np.zeros((len(v), v.shape[1] + 1), dtype=np.float64); cs1[:, 1:] = np.cumsum(z, axis=1, dtype=np.float32)
    cs2 = np.zeros((len(v), v.shape[1] + 1), dtype=np.float64); cs2[:, 1:] = np.cumsum(z * z, axis=1, dtype=np.float32)
    rows = np.arange(len(v))
    lo = np.zeros(len(v), dtype=np.int64); hi = nvalid.astype(np.int64)
    k = 5.0 if est == "mean5" else 3.0

    def stats(lo, hi):
        n = np.maximum(hi - lo, 1)
        mean_c = (cs1[rows, hi] - cs1[rows, lo]) / n
        var = np.maximum((cs2[rows, hi] - cs2[rows, lo]) / n - mean_c ** 2, 0.0)
        med = v[rows, np.clip(lo + (hi - lo) // 2, 0, v.shape[1] - 1)]
        return mean_c + ctr, np.sqrt(var), med

    # rows are sorted; make the whole array globally sorted with a per-row offset so that one searchsorted call
    # returns the clip boundaries of every row at once
    ncol = v.shape[1]
    span = float(np.nanmax(v) - np.nanmin(v)) + 10.0
    base = float(np.nanmin(v))
    vf = np.where(np.isfinite(v), v - base, span - 1.0).astype(np.float64) + rows[:, None] * (span + 1.0)
    vf = vf.ravel()
    off = rows * (span + 1.0)

    def count_below(thr, right=False):
        t = np.clip(thr - base, -0.5, span - 1.5) + off
        return np.searchsorted(vf, t, side="right" if right else "left") - rows * ncol

    for _ in range(10):
        mean, sd, med = stats(lo, hi)
        lo_n = np.minimum(count_below(med - k * sd), nvalid)
        hi_n = np.minimum(count_below(med + k * sd, right=True), nvalid)
        if np.array_equal(lo_n, lo) and np.array_equal(hi_n, hi):
            break
        lo, hi = lo_n, np.maximum(hi_n, lo_n + 1)
    mean, sd, med = stats(lo, hi)
    if est == "mode":
        val = np.where(np.abs(mean - med) / np.maximum(sd, 1e-6) < 0.3, 2.5 * med - 1.5 * mean, med)
    elif est == "median":
        val = med
    else:
        val = mean
    val = np.where(nvalid < (1 - max_masked) * box * box, np.nan, val).reshape(nby, nbx)
    val = np.where(np.isfinite(val), val, np.nanmedian(val))
    val = median_filter(val, size=filt, mode="nearest")
    from scipy.interpolate import RectBivariateSpline
    yc = (np.arange(nby) + 0.5) * box; xc = (np.arange(nbx) + 0.5) * box
    spl = RectBivariateSpline(yc, xc, val, kx=3, ky=3)
    return spl(np.clip(np.arange(ny), yc[0], yc[-1]), np.clip(np.arange(nx), xc[0], xc[-1])).astype(np.float32)


def estimator(name):
    from photutils.background import (MeanBackground, MedianBackground, SExtractorBackground)
    from astropy.stats import SigmaClip
    if name == "mode":
        return SExtractorBackground(), SigmaClip(sigma=3.0, maxiters=10)
    if name == "median":
        return MedianBackground(), SigmaClip(sigma=3.0, maxiters=10)
    if name == "mean3":
        return MeanBackground(), SigmaClip(sigma=3.0, maxiters=10)
    return MeanBackground(), SigmaClip(sigma=5.0, maxiters=5)


def run(args):
    path, dataset, obj, filt = args
    TR = _G["TR"]
    t0 = time.time()
    try:
        from photutils.background import Background2D
        from scipy.ndimage import binary_dilation
        from scipy.spatial import cKDTree
        from astropy.coordinates import SkyCoord, match_coordinates_sky
        data = np.ascontiguousarray(fits.getdata(path).astype(np.float32))
        bad = ~np.isfinite(data); data[bad] = 0.0
        b0 = sep.Background(data, mask=bad, bw=64, bh=64, fw=3, fh=3)
        sep.set_extract_pixstack(10_000_000); sep.set_sub_object_limit(4096)
        objs, seg = sep.extract(data - b0.back(), 1.5, err=b0.globalrms, minarea=12, segmentation_map=True, mask=bad)
        segm = seg > 0
        masks = {"seg4": binary_dilation(segm, iterations=4) | bad}
        for k in (3, 6, 12, 24):
            m = np.zeros(data.shape, dtype=np.bool_)
            sep.mask_ellipse(m, objs["x"], objs["y"], objs["a"], objs["b"], objs["theta"], r=float(k))
            masks[f"ell{k}"] = m | segm | bad
        # catalogue positions + selection + XP truth (as in 260924_bkg_meantest.py)
        cat = PILOT / "V0" / (Path(path).name[:-5] + "_cat.fits")
        with fits.open(cat) as h:
            d = h["LDAC_OBJECTS"].data
            xs = np.asarray(d["X_IMAGE"], float); ys = np.asarray(d["Y_IMAGE"], float)       # ezphot catalogues are 0-based
        df = TR.OZ.load_ez(cat) if dataset == "CALSPEC" else TR.AN.load_cat(cat)
        sel_idx = TR.OT.select_stars_like(df.assign(_i=np.arange(len(df))))._i.to_numpy()
        snr_all = df.SNR_APER_3.to_numpy()
        key = f"CAL_{obj}" if dataset == "CALSPEC" else f"SPSS_{TR.AN.norm(obj)}"
        ref = TR.load_cache(key)
        coords = SkyCoord(ref["ra"], ref["dec"], unit="deg")
        src = SkyCoord(df.ALPHA_J2000.to_numpy()[sel_idx], df.DELTA_J2000.to_numpy()[sel_idx], unit="deg")
        idx, d2d, _ = match_coordinates_sky(src, coords)
        ok = np.asarray(d2d.arcsec < 2.0)
        fw, ft = _G["curves"][filt]
        syn = TR.R.syn_mag(TR.XP_WAVE, ref["cor"][idx[ok]].astype(float), fw, ft, 0.0)
        snr_sel = snr_all[sel_idx]
        # blank positions (no detected source within 20", no segmentation pixel inside 10" aperture)
        ny, nx = data.shape
        rng = np.random.default_rng(2)
        bx = rng.uniform(60, nx - 60, 30000); by = rng.uniform(60, ny - 60, 30000)
        dmin, _ = cKDTree(np.c_[objs["x"], objs["y"]]).query(np.c_[bx, by])
        kk = dmin > 20.0 / PIX
        bx, by = bx[kk], by[kk]
        sseg, _, _ = sep.sum_circle(segm.astype(np.float32), bx, by, R10)
        kk = sseg == 0
        bx, by = bx[kk][:3000], by[kk][:3000]
        def metrics(mk, es, back):
            sub = data - back
            f10, _, _ = sep.sum_circle(sub, xs, ys, R10, subpix=5)
            f5, _, _ = sep.sum_circle(sub, xs, ys, R5, subpix=5)
            p, g, n = TR.estimate_pedestal(f5[sel_idx], f10[sel_idx], A5, A10, snr_sel)
            m10 = np.where(f10 > 0, -2.5 * np.log10(np.clip(f10, 1e-30, None)), np.nan)
            zc = syn - m10[sel_idx][ok]
            snr = snr_sel[ok]
            good = np.isfinite(zc)
            zp, zsd, nz = TR.clipped(zc[good & (snr > 40)])
            res = zc - zp
            faint = good & (snr >= 20) & (snr < 25); bright = good & (snr >= 110)
            fb, _, _ = sep.sum_circle(sub, bx, by, R10, subpix=5)
            return dict(name=Path(path).name, dataset=dataset, filter=filt, mask=mk, est=es,
                        level_minus_mode=float(np.median(back - b0.back())), p_star=p,
                        faint=float(np.median(res[faint]) - np.median(res[bright])) if faint.sum() >= 3 and bright.sum() >= 3 else np.nan,
                        blank10=float(np.mean(fb) / A10), zp=zp, n_faint=int(faint.sum()), n_bright=int(bright.sum()),
                        masked_frac=float(masks[mk].mean()) if mk in masks else np.nan)
        rows = [metrics("sepmode", "none", b0.back())]
        zp_base = rows[0]["zp"]
        for mk in MASKS:
            for es in ESTS:
                rows.append(metrics(mk, es, mesh_bkg(data, masks[mk], es)))
        for r in rows:
            r["dzp_mmag"] = 1e3 * (r["zp"] - zp_base) if zp_base is not None else np.nan
            r["sec"] = time.time() - t0
        return rows
    except Exception as e:                                           # noqa: BLE001
        return [dict(name=Path(path).name, error=str(e)[:300])]


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    fr = pd.read_csv(PILOT / "pilot_frames.csv")
    if limit:
        fr = pd.concat([g.head(limit) for _, g in fr.groupby("dataset")], ignore_index=True)
    jobs = [(r.path, r.dataset, r.object, r["filter"]) for _, r in fr.iterrows()]
    with Pool(nproc, initializer=_init) as pool:
        out = pool.map(run, jobs)
    S = pd.DataFrame([x for r in out for x in r])
    fr["name"] = [Path(p).name for p in fr.path]
    S = S.merge(fr[["name", "exptime"]], on="name", how="left")
    tag = "" if not limit else f"_test{limit}"
    S.to_csv(LOG / f"bkg_recipe_sweep{tag}.csv", index=False)
    if "error" in S.columns and S.error.notna().any():
        print("errors:", S.error.dropna().unique()[:3])
    S = S[S.get("error", pd.Series(index=S.index, dtype=object)).isna()] if "error" in S.columns else S
    S["grp"] = S.dataset + " " + S.exptime.astype(int).astype(str) + "s"
    S["recipe"] = S["mask"] + "/" + S["est"]
    agg = S.groupby(["recipe"]).agg(level=("level_minus_mode", "median"), p_star=("p_star", "median"),
                                   faint_mmag=("faint", lambda v: 1e3 * np.nanmedian(v)),
                                   blank10=("blank10", "median"), dzp=("dzp_mmag", "median"),
                                   masked=("masked_frac", "median"))
    pd.set_option("display.width", 200)
    print(agg.round(3).sort_values("p_star").to_string())
    print("\nper exposure group (faint_mmag):")
    print(S.pivot_table(index="recipe", columns="grp", values="faint", aggfunc=lambda v: 1e3 * np.nanmedian(v)).round(1).to_string())
    print("\nper exposure group (p_star):")
    print(S.pivot_table(index="recipe", columns="grp", values="p_star", aggfunc="median").round(3).to_string())
    print("\nseconds per frame:", round(S.groupby("name").sec.max().median(), 1))
    # figure
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
    for es, mk_ in zip(ESTS, ["o", "s", "^", "D"]):
        s = agg[agg.index.str.endswith("/" + es)]
        order = [f"{m}/{es}" for m in MASKS if f"{m}/{es}" in s.index]
        s = s.loc[order]
        xs_ = np.arange(len(order))
        ax[0].plot(xs_, s.p_star, marker=mk_, label=es); ax[1].plot(xs_, s.faint_mmag, marker=mk_, label=es)
        ax[2].plot(xs_, s.blank10, marker=mk_, label=es)
    for a_, t in zip(ax, ["residual pedestal p from star aperture growth [ADU/pix] (want 0)",
                          "XP residual SNR20-25 minus SNR>=110 [mmag] (want 0)",
                          "blank 10-arcsec aperture mean [ADU/pix] (want 0)"]):
        a_.axhline(0, color="k", lw=0.7); a_.grid(alpha=0.3); a_.set_title(t, fontsize=10)
        a_.set_xticks(range(len(MASKS))); a_.set_xticklabels(MASKS); a_.set_xlabel("source mask"); a_.legend()
    for a_, key in zip(ax, ["p_star", "faint_mmag", "blank10"]):
        a_.axhline(agg.loc[agg.index.str.startswith("sepmode"), key].iloc[0], color="grey", ls=":", lw=1,
                   label="sep mode (ezphot-like)")
    fig.suptitle("A10  Background recipe sweep on the pilot frames (mesh 64, filter 3; 10-arcsec apertures)")
    fig.tight_layout(); fig.savefig(LOG / f"A10_bkg_recipes{tag}.png", dpi=130)
    print("saved")
