#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260925_selection_test.py — is the real-star deficit after a mean background a SELECTION bias of noisy faint stars?
select_stars_like() keeps CLASS_STAR >= 0.8, FWHM/ELONGATION within 5 sigma — measured on the same noisy pixels, so
faint stars whose noise happens to make them look sharp are preferred (less shoulder light -> low aperture flux).

Per pilot frame, background = 5-sigma clipped mesh mean (segmentation + 4 px mask; blank sky ~0), 10" and 5" sep sums at
V0 positions, then p (aperture growth) and the faint XP residual for star sets:
  cur      : select_stars_like (current)
  gaia     : Gaia-XP match (<1") + isolation 10" + FLAGS 0 + SNR 10-150        (no shape cuts)
  gaia_cs  : gaia + CLASS_STAR >= 0.8
  gaia_shp : gaia + FWHM/ELONGATION cuts (as select_stars_like, no CLASS_STAR)
Output: selection_test.csv
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
    try:
        from scipy.ndimage import binary_dilation
        from astropy.coordinates import SkyCoord, match_coordinates_sky, search_around_sky
        from astropy.stats import sigma_clipped_stats
        import astropy.units as u
        data = np.ascontiguousarray(fits.getdata(path).astype(np.float32))
        bad = ~np.isfinite(data); data[bad] = 0.0
        b0 = sep.Background(data, mask=bad, bw=64, bh=64, fw=3, fh=3)
        sep.set_extract_pixstack(10_000_000); sep.set_sub_object_limit(4096)
        _, seg = sep.extract(data - b0.back(), 1.5, err=b0.globalrms, minarea=12, segmentation_map=True, mask=bad)
        backs = {"mode": b0.back(), "mean5": SW.mesh_bkg(data, binary_dilation(seg > 0, iterations=4) | bad, "mean5")}
        cat = PILOT / "V0" / (Path(path).name[:-5] + "_cat.fits")
        with fits.open(cat) as h:
            d = h["LDAC_OBJECTS"].data
            xs = np.asarray(d["X_IMAGE"], float); ys = np.asarray(d["Y_IMAGE"], float)       # ezphot catalogues are 0-based
        df = TR.OZ.load_ez(cat) if dataset == "CALSPEC" else TR.AN.load_cat(cat)
        n = len(df)
        allc = SkyCoord(df.ALPHA_J2000.to_numpy(), df.DELTA_J2000.to_numpy(), unit="deg")
        # isolation 10" against the full detection list
        i1, _, s2, _ = search_around_sky(allc, allc, 10.0 * u.arcsec)
        nbr = np.bincount(i1[s2.arcsec > 0.5], minlength=n)
        snr = df.SNR_APER_3.to_numpy()
        base = (nbr == 0) & (df.FLAGS.to_numpy() <= 0) & (snr > 10) & (snr < 150) & (df.MAGERR_APER_3.to_numpy() < 0.1)
        key = f"CAL_{obj}" if dataset == "CALSPEC" else f"SPSS_{TR.AN.norm(obj)}"
        ref = TR.load_cache(key)
        idx, d2d, _ = match_coordinates_sky(allc, SkyCoord(ref["ra"], ref["dec"], unit="deg"))
        gaia = np.asarray(d2d.arcsec < 1.0)
        fwa = df.FWHM_WORLD.to_numpy() * 3600.0; el = df.ELONGATION.to_numpy()
        g0 = base & gaia
        _, mf, sf = sigma_clipped_stats(fwa[g0], sigma=5.0, maxiters=10)
        _, me, se = sigma_clipped_stats(el[g0], sigma=5.0, maxiters=5)
        shape = (fwa > 1.0) & (fwa < 15.0) & (el < 1.5) & (np.abs(fwa - mf) <= 5 * sf) & (np.abs(el - me) <= 5 * se)
        cur = np.zeros(n, bool)
        cur[TR.OT.select_stars_like(df.assign(_i=np.arange(n)))._i.to_numpy()] = True
        sets = {"cur": cur, "gaia": g0, "gaia_cs": g0 & (df.CLASS_STAR.to_numpy() >= 0.8), "gaia_shp": g0 & shape}
        fw, ft = _G["curves"][filt]
        syn_all = np.full(n, np.nan)
        syn_all[gaia] = TR.R.syn_mag(TR.XP_WAVE, ref["cor"][idx[gaia]].astype(float), fw, ft, 0.0)
        rows = []
        for bname, back in backs.items():
            sub = data - back
            f10, _, _ = sep.sum_circle(sub, xs, ys, R10, subpix=5)
            f5, _, _ = sep.sum_circle(sub, xs, ys, R5, subpix=5)
            f2, _, _ = sep.sum_circann(sub, xs, ys, 1.8 / PIX, 4.5 / PIX, subpix=5)
            f1, _, _ = sep.sum_circle(sub, xs, ys, 1.8 / PIX, subpix=5)
            for sname, m in sets.items():
                p, g, nf = TR.estimate_pedestal(f5[m], f10[m], A5, A10, snr[m])
                mm = m & gaia
                m10 = np.where(f10 > 0, -2.5 * np.log10(np.clip(f10, 1e-30, None)), np.nan)
                zc = syn_all - m10
                good = mm & np.isfinite(zc)
                zp, _, _ = TR.clipped(zc[good & (snr > 40)])
                res = zc - zp
                fa = good & (snr >= 20) & (snr < 25); br = good & (snr >= 110)
                fam = m & (snr >= 20) & (snr < 40); brm = m & (snr >= 100)
                rows.append(dict(name=Path(path).name, dataset=dataset, filter=filt, bkg=bname, set=sname,
                                 n=int(m.sum()), n_faint=int(fa.sum()), p=p,
                                 faint=float(np.median(res[fa]) - np.median(res[br])) if fa.sum() >= 3 and br.sum() >= 3 else np.nan,
                                 # shoulder-to-core ratio (1.8-4.5" ring / inner 1.8") faint vs bright
                                 shoulder_faint=float(np.median(f2[fam] / f1[fam])) if fam.sum() >= 5 else np.nan,
                                 shoulder_bright=float(np.median(f2[brm] / f1[brm])) if brm.sum() >= 5 else np.nan))
        return rows
    except Exception as e:                                           # noqa: BLE001
        return [dict(name=Path(path).name, error=str(e)[:300])]


if __name__ == "__main__":
    fr = pd.read_csv(PILOT / "pilot_frames.csv")
    jobs = [(r.path, r.dataset, r.object, r["filter"]) for _, r in fr.iterrows()]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 8, initializer=_init) as pool:
        out = pool.map(run, jobs)
    S = pd.DataFrame([x for r in out for x in r])
    fr["name"] = [Path(p).name for p in fr.path]
    S = S.merge(fr[["name", "exptime"]], on="name", how="left")
    S.to_csv(LOG / "selection_test.csv", index=False)
    if "error" in S.columns:
        print("errors:", S["error"].notna().sum(), S["error"].dropna().head(2).tolist())
        S = S[S["error"].isna()]
    S["grp"] = S.dataset + " " + S.exptime.astype(int).astype(str) + "s"
    pd.set_option("display.width", 250)
    for b in ["mean5", "mode"]:
        T = S[S.bkg == b]
        print(f"\n=== background {b} ===")
        print(T.groupby("set").agg(n=("n", "median"), n_faint=("n_faint", "median"), p=("p", "median"),
                                   faint_mmag=("faint", lambda v: 1e3 * np.nanmedian(v)),
                                   shoulder_faint=("shoulder_faint", "median"),
                                   shoulder_bright=("shoulder_bright", "median")).round(3).to_string())
        print("faint [mmag] per group:")
        print(T.pivot_table(index="set", columns="grp", values="faint", aggfunc=lambda v: 1e3 * np.nanmedian(v)).round(1).to_string())
        print("p per group:")
        print(T.pivot_table(index="set", columns="grp", values="p", aggfunc="median").round(3).to_string())
