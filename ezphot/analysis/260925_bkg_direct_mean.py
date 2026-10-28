#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260925_bkg_direct_mean.py — change the ESTIMATOR (per-mesh mean) instead of mode + global offset?

Per pilot frame, source mask from the real ezphot MaskGenerator.mask_sources(5 sigma, x3) (+ NaN pixels), then
  sepmode   : sep.Background, no mask (SExtractor mode; current catalogues)
  ez_goff   : ezphot estimate_with_sep(box 64, correct_global_offset=True)          [mode + one global offset]
  pu_mean5  : photutils Background2D(box 64, filter 3, SigmaClip 5, MeanBackground), ezphot mask   [direct mean]
  pu_mean3  : same with SigmaClip 3 (= ezphot estimate_with_photutils 'mean' today, box 64)
  mesh_mean : 260925_bkg_recipe_sweep.mesh_bkg 'mean5' with the same mask (cross-check of pu_mean5)
Scores: blank10 (10000 blank 10" apertures, >= 20" from detections) overall, per 3x3 field cell (sd, range) and
rim (normalised radius > 0.85) minus interior; p_star / faint XP (select_stars_like) and the Gaia set (no shape cuts).
Output: bkg_direct_mean.csv
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
_G = {}


def _init():
    warnings.filterwarnings("ignore")
    TR = importlib.import_module("260924_stepA_triage")
    TR.AN._init()
    _G["TR"] = TR
    _G["SW"] = importlib.import_module("260925_bkg_recipe_sweep")
    _G["curves"] = TR.TC.build_totals()[("manufacturer", "nobeam")]


def run(args):
    path, dataset, obj, filt = args
    TR, SW = _G["TR"], _G["SW"]
    t0 = time.time()
    try:
        from ezphot.imageobjects import ScienceImage, Mask
        from ezphot.methods import MaskGenerator, BackgroundGenerator
        from photutils.background import Background2D, MeanBackground
        from astropy.stats import SigmaClip
        from scipy.spatial import cKDTree
        from astropy.coordinates import SkyCoord, match_coordinates_sky, search_around_sky
        import astropy.units as u
        raw = fits.getdata(path).astype(np.float32)
        bad = ~np.isfinite(raw)
        data = np.ascontiguousarray(np.where(bad, 0.0, raw).astype(np.float32))
        ny, nx = data.shape
        img = ScienceImage(path=path, telinfo=None, load=True)
        ext = Mask(SCR / "ezmask_unused.fits", masktype="source", load=False)
        srcmask = MaskGenerator().mask_sources(img, target_mask=ext, sigma=5.0, mask_radius_factor=3,
                                               save=False, verbose=False, visualize=False)
        msk = np.asarray(srcmask.data).astype(bool) | bad
        backs, secs = {}, {}
        t = time.time(); backs["sepmode"] = sep.Background(data, mask=bad, bw=64, bh=64, fw=3, fh=3).back(); secs["sepmode"] = time.time() - t
        t = time.time()
        bk, _ = BackgroundGenerator().estimate_with_sep(img, target_srcmask=srcmask, box_size=64, filter_size=3,
                                                       correct_global_offset=True, save=False, verbose=False, visualize=False)
        backs["ez_goff"] = np.where(np.isfinite(bk.data), bk.data, 0.0).astype(np.float32); secs["ez_goff"] = time.time() - t
        for tag, sg in [("pu_mean5", 5.0), ("pu_mean3", 3.0)]:
            t = time.time()
            b = Background2D(data, (64, 64), mask=msk, filter_size=(3, 3),
                             sigma_clip=SigmaClip(sigma=sg, maxiters=10), bkg_estimator=MeanBackground())
            backs[tag] = np.asarray(b.background, dtype=np.float32); secs[tag] = time.time() - t
        t = time.time(); backs["mesh_mean"] = SW.mesh_bkg(data, msk, "mean5"); secs["mesh_mean"] = time.time() - t
        # --- stars ------------------------------------------------------------------------------------------------
        cat = PILOT / "V0" / (Path(path).name[:-5] + "_cat.fits")
        with fits.open(cat) as h:
            d = h["LDAC_OBJECTS"].data
            xs = np.asarray(d["X_IMAGE"], float); ys = np.asarray(d["Y_IMAGE"], float)       # 0-based already
        df = TR.OZ.load_ez(cat) if dataset == "CALSPEC" else TR.AN.load_cat(cat)
        n = len(df)
        snr = df.SNR_APER_3.to_numpy()
        cur = np.zeros(n, bool)
        cur[TR.OT.select_stars_like(df.assign(_i=np.arange(n)))._i.to_numpy()] = True
        allc = SkyCoord(df.ALPHA_J2000.to_numpy(), df.DELTA_J2000.to_numpy(), unit="deg")
        i1, _, s2, _ = search_around_sky(allc, allc, 10.0 * u.arcsec)
        nbr = np.bincount(i1[s2.arcsec > 0.5], minlength=n)
        key = f"CAL_{obj}" if dataset == "CALSPEC" else f"SPSS_{TR.AN.norm(obj)}"
        ref = TR.load_cache(key)
        idx, d2d, _ = match_coordinates_sky(allc, SkyCoord(ref["ra"], ref["dec"], unit="deg"))
        gaia_m = np.asarray(d2d.arcsec < 1.0)
        gaia = gaia_m & (nbr == 0) & (df.FLAGS.to_numpy() <= 0) & (snr > 10) & (snr < 150) & (df.MAGERR_APER_3.to_numpy() < 0.1)
        fw, ft = _G["curves"][filt]
        syn = np.full(n, np.nan)
        syn[gaia_m] = TR.R.syn_mag(TR.XP_WAVE, ref["cor"][idx[gaia_m]].astype(float), fw, ft, 0.0)
        # --- blank positions ------------------------------------------------------------------------------------
        sep.set_extract_pixstack(10_000_000); sep.set_sub_object_limit(4096)
        b0 = sep.Background(data, mask=bad, bw=64, bh=64, fw=3, fh=3)
        objs, seg = sep.extract(data - b0.back(), 1.5, err=b0.globalrms, minarea=12, segmentation_map=True, mask=bad)
        rng = np.random.default_rng(3)
        bx = rng.uniform(60, nx - 60, 120000); by = rng.uniform(60, ny - 60, 120000)
        dmin, _ = cKDTree(np.c_[objs["x"], objs["y"]]).query(np.c_[bx, by])
        kk = dmin > 20.0 / PIX
        bx, by = bx[kk], by[kk]
        sseg, _, _ = sep.sum_circle((seg > 0).astype(np.float32), bx, by, R10)
        sbad, _, _ = sep.sum_circle(bad.astype(np.float32), bx, by, R10)
        kk = (sseg == 0) & (sbad == 0)
        bx, by = bx[kk][:10000], by[kk][:10000]
        cell = (np.minimum((by / ny * 3).astype(int), 2) * 3 + np.minimum((bx / nx * 3).astype(int), 2))
        rn = np.hypot((bx - nx / 2) / (nx / 2), (by - ny / 2) / (ny / 2)) / np.sqrt(2)
        rows = []
        for tag, back in backs.items():
            sub = data - back
            fb, _, _ = sep.sum_circle(sub, bx, by, R10, subpix=5)
            pb = fb / A10
            cm = np.array([np.mean(pb[cell == c]) if (cell == c).sum() > 50 else np.nan for c in range(9)])
            f10, _, _ = sep.sum_circle(sub, xs, ys, R10, subpix=5)
            f5, _, _ = sep.sum_circle(sub, xs, ys, R5, subpix=5)
            r = dict(name=Path(path).name, dataset=dataset, filter=filt, route=tag, sec=secs[tag],
                     level_minus_mode=float(np.median(back - backs["sepmode"])), masked_frac=float(msk.mean()),
                     blank10=float(np.mean(pb)), n_blank=len(pb), cell_sd=float(np.nanstd(cm)),
                     cell_range=float(np.nanmax(cm) - np.nanmin(cm)),
                     rim_minus_in=float(np.mean(pb[rn > 0.85]) - np.mean(pb[rn < 0.5])) if (rn > 0.85).sum() > 50 else np.nan)
            m10 = np.where(f10 > 0, -2.5 * np.log10(np.clip(f10, 1e-30, None)), np.nan)
            for sname, m in [("cur", cur), ("gaia", gaia)]:
                p, _, _ = TR.estimate_pedestal(f5[m], f10[m], A5, A10, snr[m])
                zc = syn - m10
                good = m & gaia_m & np.isfinite(zc)
                zp, _, _ = TR.clipped(zc[good & (snr > 40)])
                res = zc - zp
                fa = good & (snr >= 20) & (snr < 25); br = good & (snr >= 110)
                r[f"p_{sname}"] = p
                r[f"faint_{sname}"] = 1e3 * float(np.median(res[fa]) - np.median(res[br])) if fa.sum() >= 3 and br.sum() >= 3 else np.nan
                if sname == "cur":
                    r["zp"] = zp
            rows.append(r)
        for r in rows:
            r["dzp_mmag"] = 1e3 * (r["zp"] - rows[0]["zp"])
        return rows
    except Exception as e:                                           # noqa: BLE001
        import traceback
        return [dict(name=Path(path).name, error=(str(e) + " | " + traceback.format_exc()[-400:])[:600])]


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
    S.to_csv(LOG / f"bkg_direct_mean{tag}.csv", index=False)
    if "error" in S.columns:
        print("errors:", S["error"].notna().sum(), S["error"].dropna().head(3).tolist())
        S = S[S["error"].isna()]
    S["grp"] = S.dataset + " " + S.exptime.astype(int).astype(str) + "s"
    pd.set_option("display.width", 250)
    cols = ["blank10", "cell_sd", "cell_range", "rim_minus_in", "p_cur", "p_gaia", "faint_cur", "faint_gaia",
            "dzp_mmag", "level_minus_mode", "sec"]
    print(S.groupby("route")[cols].median().round(3).to_string())
    print("\nmean |blank10| and 90th pct cell_range:")
    print(S.groupby("route").agg(abs_blank=("blank10", lambda v: np.mean(np.abs(v))),
                                 cell_range_p90=("cell_range", lambda v: np.percentile(v, 90))).round(3).to_string())
    for c in ["blank10", "cell_sd", "faint_gaia", "p_gaia"]:
        print(f"\nper exposure group ({c}):")
        print(S.pivot_table(index="route", columns="grp", values=c, aggfunc="median").round(3).to_string())
