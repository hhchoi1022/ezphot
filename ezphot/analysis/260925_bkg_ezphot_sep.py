#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260925_bkg_ezphot_sep.py — does ezphot's own mean route (MaskGenerator.mask_sources +
BackgroundGenerator.estimate_with_sep(correct_global_offset=True)) remove the 10-arcsec pedestal?
Runs the real ezphot code (no files written: external in-memory Mask, save=False) on the 65 pilot frames and scores
the background maps with the same metrics as 260925_bkg_recipe_sweep.py (p_star, faint, blank10, dzp vs sep mode).
Variants: EZ32_off  : mask_sources(sigma 5, factor 3) + estimate_with_sep(box 32, filter 3, global offset)  [ezphot defaults]
          EZ64_off  : same, box 64 (as the 7DT sexconfig BACK_SIZE)
          EZ64_mode : box 64, correct_global_offset=False (control: masked mode)
Output: bkg_ezphot_sep.csv
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
VARIANTS = {"EZ32_off": dict(box_size=32, correct_global_offset=True),
            "EZ64_off": dict(box_size=64, correct_global_offset=True),
            "EZ64_mode": dict(box_size=64, correct_global_offset=False)}
_G = {}


def _init():
    warnings.filterwarnings("ignore")
    TR = importlib.import_module("260924_stepA_triage")
    TR.AN._init()
    _G["TR"] = TR
    _G["curves"] = TR.TC.build_totals()[("manufacturer", "nobeam")]


def run(args):
    path, dataset, obj, filt = args
    TR = _G["TR"]
    t0 = time.time()
    try:
        from ezphot.imageobjects import ScienceImage, Mask
        from ezphot.methods import MaskGenerator, BackgroundGenerator
        from scipy.spatial import cKDTree
        from astropy.coordinates import SkyCoord, match_coordinates_sky
        data = np.ascontiguousarray(fits.getdata(path).astype(np.float32))
        bad = ~np.isfinite(data); data[bad] = 0.0
        b0 = sep.Background(data, mask=bad, bw=64, bh=64, fw=3, fh=3)
        sep.set_extract_pixstack(10_000_000); sep.set_sub_object_limit(4096)
        objs, seg = sep.extract(data - b0.back(), 1.5, err=b0.globalrms, minarea=12, segmentation_map=True, mask=bad)
        segm = seg > 0
        # --- ezphot's own route -------------------------------------------------------------------------------
        img = ScienceImage(path=path, telinfo=None, load=True)
        ext = Mask(SCR / "ezmask_unused.fits", masktype="source", load=False)          # in-memory, never written
        srcmask = MaskGenerator().mask_sources(img, target_mask=ext, sigma=5.0, mask_radius_factor=3,
                                               save=False, verbose=False, visualize=False)
        backs = {}
        for v, kw in VARIANTS.items():
            bk, _ = BackgroundGenerator().estimate_with_sep(img, target_srcmask=srcmask, filter_size=3, save=False,
                                                           verbose=False, visualize=False, **kw)
            backs[v] = np.asarray(bk.data, dtype=np.float32)
        masked_frac = float(np.asarray(srcmask.data).astype(bool).mean())
        # --- identical scoring to 260925_bkg_recipe_sweep.py ---------------------------------------------------
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
        ny, nx = data.shape
        rng = np.random.default_rng(2)
        bx = rng.uniform(60, nx - 60, 30000); by = rng.uniform(60, ny - 60, 30000)
        dmin, _ = cKDTree(np.c_[objs["x"], objs["y"]]).query(np.c_[bx, by])
        kk = dmin > 20.0 / PIX
        bx, by = bx[kk], by[kk]
        sseg, _, _ = sep.sum_circle(segm.astype(np.float32), bx, by, R10)
        kk = sseg == 0
        bx, by = bx[kk][:3000], by[kk][:3000]

        def metrics(tag, back):
            back = np.where(np.isfinite(back), back, 0.0).astype(np.float32)
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
            return dict(name=Path(path).name, dataset=dataset, filter=filt, variant=tag,
                        level_minus_mode=float(np.median(back - b0.back())), p_star=p,
                        faint=float(np.median(res[faint]) - np.median(res[bright])) if faint.sum() >= 3 and bright.sum() >= 3 else np.nan,
                        blank10=float(np.mean(fb) / A10), zp=zp, masked_frac=masked_frac,
                        n_faint=int(faint.sum()), n_bright=int(bright.sum()))
        rows = [metrics("sepmode", b0.back())]
        for v, bk in backs.items():
            rows.append(metrics(v, bk))
        for r in rows:
            r["dzp_mmag"] = 1e3 * (r["zp"] - rows[0]["zp"])
            r["sec"] = time.time() - t0
        return rows
    except Exception as e:                                           # noqa: BLE001
        import traceback
        return [dict(name=Path(path).name, error=(str(e) + " | " + traceback.format_exc()[-400:])[:600])]


if __name__ == "__main__":
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    fr = pd.read_csv(PILOT / "pilot_frames.csv")
    if limit:
        fr = pd.concat([g.head(limit) for _, g in fr.groupby("dataset")], ignore_index=True)
    jobs = [(r.path, r.dataset, r.object, r["filter"]) for _, r in fr.iterrows()]
    out = []
    with Pool(nproc, initializer=_init) as pool:
        for i, rs in enumerate(pool.imap_unordered(run, jobs), 1):
            out.extend(rs)
            print(f"{i}/{len(jobs)} {rs[0].get('name', '')[:50]} {rs[0].get('error', '')[:200]}", flush=True)
    S = pd.DataFrame(out)
    fr["name"] = [Path(p).name for p in fr.path]
    S = S.merge(fr[["name", "exptime"]], on="name", how="left")
    tag = "" if not limit else f"_test{limit}"
    S.to_csv(LOG / f"bkg_ezphot_sep{tag}.csv", index=False)
    if "error" in S.columns:
        print("errors:", S["error"].notna().sum(), S["error"].dropna().head(2).tolist())
        S = S[S["error"].isna()]
    # add the sweep's mesh-mean recipes for the same frames
    W = pd.read_csv(LOG / "bkg_recipe_sweep.csv")
    W = W[W["mask"].isin(["seg4", "ell24"]) & W["est"].isin(["mean5"])].copy()
    W["variant"] = W["mask"] + "/" + W["est"]
    W = W[W.name.isin(S.name)]
    A = pd.concat([S, W[S.columns.intersection(W.columns)]], ignore_index=True)
    A["grp"] = A.dataset + " " + A.exptime.astype(int).astype(str) + "s"
    pd.set_option("display.width", 220)
    agg = A.groupby("variant").agg(level=("level_minus_mode", "median"), p_star=("p_star", "median"),
                                   faint_mmag=("faint", lambda v: 1e3 * np.nanmedian(v)),
                                   blank10=("blank10", "median"), dzp=("dzp_mmag", "median"),
                                   masked=("masked_frac", "median"), n=("name", "nunique"))
    print(agg.round(3).to_string())
    for key, fn in [("faint", lambda v: 1e3 * np.nanmedian(v)), ("p_star", "median"), ("blank10", "median")]:
        print(f"\nper exposure group ({key}):")
        print(A.pivot_table(index="variant", columns="grp", values=key, aggfunc=fn).round(3).to_string())
    print("\nseconds per frame:", round(S.groupby("name").sec.max().median(), 1))
