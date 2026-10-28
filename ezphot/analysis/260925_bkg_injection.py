#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260925_bkg_injection.py — artificial-star injection–recovery: is the star-side deficit left after a MEAN background
(real stars need 0.2-0.4 ADU/pix less than blank sky) produced by the software path (background + aperture sums) or
by the images themselves?  Read-only on images (injection only in memory; ezphot objects never written).

Per pilot frame:
  PSF     : empirical, median of normalised stamps of bright isolated unsaturated stars (V0 catalogue; local 5-sigma
            clipped annulus mean at 22-29" removed; truncated at 20").
  inject  : SNR_10" = 22 / 50 / 120 (300 / 150 / 100 stars), same isolation as the real-star selection (no V0
            detection within 10"), >= 30" between injected stars; photon noise added (EGAIN).
  routes  : sepmode  = sep.Background (SExtractor mode; ezphot default photometry)
            ezmean   = ezphot MaskGenerator.mask_sources(5 sigma, x3) + BackgroundGenerator.estimate_with_sep(box 64,
                       correct_global_offset=True)
            pumean5  = same mask + BackgroundGenerator.estimate_with_photutils('mean', sigma 5, box 64, filter 3)
                       [all computed on the injected image]
  scores  : injected stars: absolute recovery 2.5 log10(F_meas/F_true) per SNR level (>0 = measured too bright),
            the same as ADU/pix, and the aperture-growth pedestal p_inj from injected stars only (tests the p metric)
            real stars (control, measured on the original pixels with the injected-image background): p_star, faint
            XP residual (SNR 20-25 minus >= 110), blank10
Output: bkg_injection.csv
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
SCR = Path("/tmp/claude-10000/-lyman-data1-factory-hhchoi-code-ezphot/b1308704-c229-4e49-b19d-bd7bc334e4b7/scratchpad")
PIX = 0.505
R5, R10 = 2.5 / PIX, 5.0 / PIX
A5, A10 = np.pi * R5 ** 2, np.pi * R10 ** 2
HALF = 60                                   # stamp half size [px] (121 x 121)
R_PSF = 20.0 / PIX                          # PSF truncation radius
R_ANN = (22.0 / PIX, 29.0 / PIX)            # local background annulus for PSF stamps
LEVELS = [(22.0, 300), (50.0, 150), (120.0, 100)]
_G = {}


def _init():
    warnings.filterwarnings("ignore")
    TR = importlib.import_module("260924_stepA_triage")
    TR.AN._init()
    _G["TR"] = TR
    _G["curves"] = TR.TC.build_totals()[("manufacturer", "nobeam")]


def build_psf(data, bad, xs, ys, snr, flags, cls, fmax):
    from scipy.ndimage import shift as ndshift
    from scipy.spatial import cKDTree
    from astropy.stats import sigma_clipped_stats
    ny, nx = data.shape
    yy, xx = np.mgrid[-HALF:HALF + 1, -HALF:HALF + 1]
    rr = np.hypot(xx, yy)
    ann = (rr >= R_ANN[0]) & (rr <= R_ANN[1])
    tree = cKDTree(np.c_[xs, ys])
    cnt = np.array([len(tree.query_ball_point([x, y], 25.0 / PIX)) for x, y in zip(xs, ys)])
    cand = np.where((snr > 150) & (flags <= 0) & (cls >= 0.8) & (fmax < 20000) & (cnt == 1)
                    & (xs > HALF + 2) & (xs < nx - HALF - 3) & (ys > HALF + 2) & (ys < ny - HALF - 3))[0]
    cand = cand[np.argsort(-snr[cand])][:80]
    stamps = []
    for i in cand:
        ix, iy = int(round(xs[i])), int(round(ys[i]))
        cut = data[iy - HALF:iy + HALF + 1, ix - HALF:ix + HALF + 1].astype(np.float64)
        if bad[iy - HALF:iy + HALF + 1, ix - HALF:ix + HALF + 1].any():
            continue
        cut = ndshift(cut, (iy - ys[i], ix - xs[i]), order=3, mode="nearest")
        cut = cut - sigma_clipped_stats(cut[ann], sigma=5.0, maxiters=5)[0]
        s = cut[rr <= R_PSF].sum()
        if s > 0:
            stamps.append(cut / s)
    if len(stamps) < 8:
        return None, len(stamps)
    psf = np.median(np.array(stamps), axis=0)
    psf[rr > R_PSF] = 0.0
    psf /= psf.sum()
    return psf, len(stamps)


def run(args):
    path, dataset, obj, filt = args
    TR = _G["TR"]
    t0 = time.time()
    try:
        from ezphot.imageobjects import ScienceImage, Mask
        from ezphot.methods import MaskGenerator, BackgroundGenerator
        from scipy.ndimage import shift as ndshift
        from scipy.spatial import cKDTree
        from astropy.coordinates import SkyCoord, match_coordinates_sky
        g = float(fits.getheader(path)["EGAIN"])
        raw = fits.getdata(path).astype(np.float32)
        bad = ~np.isfinite(raw)
        data = np.ascontiguousarray(np.where(bad, 0.0, raw).astype(np.float32))
        ny, nx = data.shape
        cat = PILOT / "V0" / (Path(path).name[:-5] + "_cat.fits")
        with fits.open(cat) as h:
            d = h["LDAC_OBJECTS"].data
            xs = np.asarray(d["X_IMAGE"], float); ys = np.asarray(d["Y_IMAGE"], float)       # ezphot catalogues are 0-based
            snr_v0 = 1.0857 / np.clip(np.asarray(d["MAGERR_APER_3"], float), 1e-4, None)
            psf, npsf = build_psf(data, bad, xs, ys, snr_v0, np.asarray(d["FLAGS"], float),
                                  np.asarray(d["CLASS_STAR"], float), np.asarray(d["FLUX_MAX"], float))
        if psf is None:
            return [dict(name=Path(path).name, error=f"psf stars {npsf}")]
        # --- injection positions: no V0 detection within 10", >= 30" apart, away from edges/bad pixels -------------
        b0 = sep.Background(data, mask=bad, bw=64, bh=64, fw=3, fh=3)
        sig = float(b0.globalrms)
        rng = np.random.default_rng(7)
        tree = cKDTree(np.c_[xs, ys])
        want = sum(n for _, n in LEVELS)
        px, py = [], []
        for _ in range(60):
            cx = rng.uniform(HALF + 5, nx - HALF - 6, 4000); cy = rng.uniform(HALF + 5, ny - HALF - 6, 4000)
            dmin, _ = tree.query(np.c_[cx, cy])
            for x, y, dm in zip(cx, cy, dmin):
                if dm <= 10.0 / PIX or bad[int(y), int(x)]:
                    continue
                if px and np.min(np.hypot(np.array(px) - x, np.array(py) - y)) < 30.0 / PIX:
                    continue
                px.append(x); py.append(y)
                if len(px) >= want:
                    break
            if len(px) >= want:
                break
        px, py = np.array(px), np.array(py)
        lev = np.concatenate([[s] * n for s, n in LEVELS])[:len(px)]
        ftot = (lev ** 2 / g + np.sqrt(lev ** 4 / g ** 2 + 4 * lev ** 2 * A10 * sig ** 2)) / 2.0   # F10 for the SNR
        ee10 = float(sep.sum_circle(psf.astype(np.float32), [HALF], [HALF], R10, subpix=5)[0][0])
        model = np.zeros_like(data, dtype=np.float64)
        for x, y, f in zip(px, py, ftot / ee10):
            ix, iy = int(round(x)), int(round(y))
            st = ndshift(psf, (y - iy, x - ix), order=3, mode="constant", cval=0.0)
            model[iy - HALF:iy + HALF + 1, ix - HALF:ix + HALF + 1] += f * st
        # truth = 10"/5" aperture sums of the noise-free model (ftot/ee10 puts ~ftot inside 10")
        mdl = model.astype(np.float32)
        t10, _, _ = sep.sum_circle(mdl, px, py, R10, subpix=5)
        t5, _, _ = sep.sum_circle(mdl, px, py, R5, subpix=5)
        noise = (rng.poisson(np.clip(model * g, 0, None)) - model * g) / g
        inj = np.ascontiguousarray((data + model + noise).astype(np.float32))
        inj[bad] = 0.0
        # --- background routes on the injected image -------------------------------------------------------------
        backs = {"sepmode": sep.Background(inj, mask=bad, bw=64, bh=64, fw=3, fh=3).back()}
        img = ScienceImage(path=path, telinfo=None, load=True)
        img.data = np.where(bad, np.nan, inj).astype(np.float32)
        ext = Mask(SCR / "ezmask_unused.fits", masktype="source", load=False)
        srcmask = MaskGenerator().mask_sources(img, target_mask=ext, sigma=5.0, mask_radius_factor=3,
                                               save=False, verbose=False, visualize=False)
        bk, _ = BackgroundGenerator().estimate_with_sep(img, target_srcmask=srcmask, box_size=64, filter_size=3,
                                                       correct_global_offset=True, save=False, verbose=False,
                                                       visualize=False)
        backs["ezmean"] = np.where(np.isfinite(bk.data), bk.data, 0.0).astype(np.float32)
        bk5, _ = BackgroundGenerator().estimate_with_photutils(img, target_mask=srcmask, box_size=64, filter_size=3,
                                                              bkg_estimator="mean", save=False, verbose=False,
                                                              visualize=False)
        backs["pumean5"] = np.where(np.isfinite(bk5.data), bk5.data, 0.0).astype(np.float32)
        # --- real-star scoring inputs (as in 260925_bkg_recipe_sweep.py) ------------------------------------------
        df = TR.OZ.load_ez(cat) if dataset == "CALSPEC" else TR.AN.load_cat(cat)
        sel_idx = TR.OT.select_stars_like(df.assign(_i=np.arange(len(df))))._i.to_numpy()
        snr_sel = df.SNR_APER_3.to_numpy()[sel_idx]
        key = f"CAL_{obj}" if dataset == "CALSPEC" else f"SPSS_{TR.AN.norm(obj)}"
        ref = TR.load_cache(key)
        coords = SkyCoord(ref["ra"], ref["dec"], unit="deg")
        src = SkyCoord(df.ALPHA_J2000.to_numpy()[sel_idx], df.DELTA_J2000.to_numpy()[sel_idx], unit="deg")
        idx, d2d, _ = match_coordinates_sky(src, coords)
        ok = np.asarray(d2d.arcsec < 2.0)
        fw, ft = _G["curves"][filt]
        syn = TR.R.syn_mag(TR.XP_WAVE, ref["cor"][idx[ok]].astype(float), fw, ft, 0.0)
        sep.set_extract_pixstack(10_000_000); sep.set_sub_object_limit(4096)
        objs, seg = sep.extract(data - b0.back(), 1.5, err=sig, minarea=12, segmentation_map=True, mask=bad)
        bx = rng.uniform(60, nx - 60, 30000); by = rng.uniform(60, ny - 60, 30000)
        dmin, _ = cKDTree(np.c_[objs["x"], objs["y"]]).query(np.c_[bx, by])
        dinj, _ = cKDTree(np.c_[px, py]).query(np.c_[bx, by])
        kk = (dmin > 20.0 / PIX) & (dinj > 25.0 / PIX)
        bx, by = bx[kk], by[kk]
        sseg, _, _ = sep.sum_circle((seg > 0).astype(np.float32), bx, by, R10)
        bx, by = bx[sseg == 0][:3000], by[sseg == 0][:3000]
        rows = []
        for tag, back in backs.items():
            r = dict(name=Path(path).name, dataset=dataset, filter=filt, route=tag, egain=g, sky_rms=sig, n_psf=npsf,
                     n_inj=len(px), level_minus_mode=float(np.median(back - backs["sepmode"])))
            # injected stars (measured on the injected image)
            sub = inj - back
            f10, _, _ = sep.sum_circle(sub, px, py, R10, subpix=5)
            f5, _, _ = sep.sum_circle(sub, px, py, R5, subpix=5)
            for s, _n in LEVELS:
                m = lev == s
                r[f"inj{int(s)}_mmag"] = 1e3 * float(np.median(2.5 * np.log10(np.clip(f10[m], 1e-3, None) / t10[m])))
                r[f"inj{int(s)}_adupix"] = float(np.median(f10[m] - t10[m]) / A10)
                r[f"inj{int(s)}_mad_mmag"] = 1e3 * float(1.4826 * np.median(np.abs(
                    2.5 * np.log10(np.clip(f10[m], 1e-3, None) / t10[m]) - np.median(2.5 * np.log10(np.clip(f10[m], 1e-3, None) / t10[m])))))
            snr_inj = lev + rng.normal(0, 1.0, len(lev))      # nominal SNR (jitter only to break ties in the bins)
            p_inj, g_inj, _ = TR.estimate_pedestal(f5, f10, A5, A10, snr_inj)
            r["p_inj"] = p_inj
            r["g_inj"] = g_inj
            r["g_true"] = float(np.median(t10 / t5))
            # real stars (original pixels, background from the injected image)
            subr = data - back
            g10, _, _ = sep.sum_circle(subr, xs, ys, R10, subpix=5)
            g5, _, _ = sep.sum_circle(subr, xs, ys, R5, subpix=5)
            p, gg, _ = TR.estimate_pedestal(g5[sel_idx], g10[sel_idx], A5, A10, snr_sel)
            m10 = np.where(g10 > 0, -2.5 * np.log10(np.clip(g10, 1e-30, None)), np.nan)
            zc = syn - m10[sel_idx][ok]
            snr = snr_sel[ok]
            good = np.isfinite(zc)
            zp, _, _ = TR.clipped(zc[good & (snr > 40)])
            res = zc - zp
            fa = good & (snr >= 20) & (snr < 25); br = good & (snr >= 110)
            r["p_star"] = p
            r["faint_real_mmag"] = 1e3 * float(np.median(res[fa]) - np.median(res[br])) if fa.sum() >= 3 and br.sum() >= 3 else np.nan
            r["inj_faint_rel_mmag"] = r["inj22_mmag"] - r["inj120_mmag"]
            fb, _, _ = sep.sum_circle(subr, bx, by, R10, subpix=5)
            r["blank10"] = float(np.mean(fb) / A10)
            r["zp"] = zp
            r["sec"] = time.time() - t0
            rows.append(r)
        return rows
    except Exception as e:                                           # noqa: BLE001
        import traceback
        return [dict(name=Path(path).name, error=(str(e) + " | " + traceback.format_exc()[-500:])[:700])]


if __name__ == "__main__":
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    fr = pd.read_csv(PILOT / "pilot_frames.csv")
    if limit:
        fr = pd.concat([g_.head(limit) for _, g_ in fr.groupby("dataset")], ignore_index=True)
    jobs = [(r_.path, r_.dataset, r_.object, r_["filter"]) for _, r_ in fr.iterrows()]
    out = []
    with Pool(nproc, initializer=_init) as pool:
        for i, rs in enumerate(pool.imap_unordered(run, jobs), 1):
            out.extend(rs)
            print(f"{i}/{len(jobs)} {rs[0].get('name', '')[:50]} {rs[0].get('error', '')[:300]}", flush=True)
    S = pd.DataFrame(out)
    fr["name"] = [Path(p).name for p in fr.path]
    S = S.merge(fr[["name", "exptime"]], on="name", how="left")
    tag = "" if not limit else f"_test{limit}"
    S.to_csv(LOG / f"bkg_injection{tag}.csv", index=False)
    if "error" in S.columns:
        print("errors:", S["error"].notna().sum(), S["error"].dropna().head(3).tolist())
        S = S[S["error"].isna()]
    S["grp"] = S.dataset + " " + S.exptime.astype(int).astype(str) + "s"
    pd.set_option("display.width", 250)
    cols = ["inj22_mmag", "inj50_mmag", "inj120_mmag", "inj22_adupix", "inj50_adupix", "inj120_adupix", "p_inj",
            "g_inj", "g_true", "inj_faint_rel_mmag", "p_star", "faint_real_mmag", "blank10", "level_minus_mode"]
    print(S.groupby("route")[cols].median().round(3).T.to_string())
    for c in ["inj22_mmag", "inj22_adupix", "p_inj", "p_star", "faint_real_mmag", "blank10"]:
        print(f"\nper exposure group ({c}):")
        print(S.pivot_table(index="route", columns="grp", values=c, aggfunc="median").round(3).to_string())
    print("\nmedian inj22 MAD [mmag]:", S.groupby("route").inj22_mad_mmag.median().round(1).to_dict(),
          "| n_psf median", int(S.n_psf.median()), "| seconds per frame:", round(S.groupby("name").sec.max().median(), 1))
