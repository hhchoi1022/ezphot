#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260924_bkg_meantest.py — does a MEAN-based background remove the faint-star bias?  Read-only on images.

Per pilot frame, aperture sums with sep at the V0 (default ezphot) catalogue positions, 5 and 10 arcsec, after
subtracting either
  mode : sep.Background (SExtractor algorithm, mesh 64, filter 3)          -> should reproduce ezphot V0
  mean : photutils Background2D, mesh 64, filter 3, sources masked (1.5 sigma segmentation dilated 4 px),
         5-sigma clipped MEAN
Then the corrected-XP field residual (syn - inst - ZP) vs SNR is compared between ezphot V0, sep-mode, sep-mean.
Output: bkg_meantest_frames.csv, bkg_meantest_bins.csv, A9_bkg_meantest.png
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
SNRB = np.array([20, 25, 30, 40, 60, 80, 110, 150])
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
    try:
        from photutils.background import Background2D, MeanBackground
        from astropy.stats import SigmaClip
        from scipy.ndimage import binary_dilation
        from astropy.coordinates import SkyCoord, match_coordinates_sky
        data = np.ascontiguousarray(fits.getdata(path).astype(np.float32))
        bad = ~np.isfinite(data)
        data[bad] = 0.0
        b_mode = sep.Background(data, mask=bad, bw=64, bh=64, fw=3, fh=3)
        sep.set_extract_pixstack(10_000_000); sep.set_sub_object_limit(4096)
        _, seg = sep.extract(data - b_mode.back(), 1.5, err=b_mode.globalrms, minarea=12, segmentation_map=True, mask=bad)
        msk = binary_dilation(seg > 0, iterations=4) | bad
        b_mean = Background2D(data, (64, 64), mask=msk, filter_size=(3, 3),
                              sigma_clip=SigmaClip(sigma=5.0, maxiters=5), bkg_estimator=MeanBackground())
        cat = PILOT / "V0" / (Path(path).name[:-5] + "_cat.fits")
        with fits.open(cat) as h:
            d = h["LDAC_OBJECTS"].data
            x = np.asarray(d["X_IMAGE"], float); y = np.asarray(d["Y_IMAGE"], float)       # ezphot catalogues are 0-based
            m10_v0 = np.asarray(d["MAG_APER_3"], float)
        out = {}
        for tag, back in [("mode", b_mode.back()), ("mean", b_mean.background)]:
            sub = data - back
            f10, _, _ = sep.sum_circle(sub, x, y, R10, subpix=5)
            out[tag] = np.where(f10 > 0, -2.5 * np.log10(np.clip(f10, 1e-30, None)), 99.0)
        # build a catalogue-like frame with the three magnitude variants
        df = TR.OZ.load_ez(cat) if dataset == "CALSPEC" else TR.AN.load_cat(cat)
        sel_idx = TR.OT.select_stars_like(df.assign(_i=np.arange(len(df))))._i.to_numpy()
        key = f"CAL_{obj}" if dataset == "CALSPEC" else f"SPSS_{TR.AN.norm(obj)}"
        ref = TR.load_cache(key)
        coords = SkyCoord(ref["ra"], ref["dec"], unit="deg")
        src = SkyCoord(df.ALPHA_J2000.to_numpy()[sel_idx], df.DELTA_J2000.to_numpy()[sel_idx], unit="deg")
        idx, d2d, _ = match_coordinates_sky(src, coords)
        ok = np.asarray(d2d.arcsec < 2.0)
        fw, ft = _G["curves"][filt]
        syn = TR.R.syn_mag(TR.XP_WAVE, ref["cor"][idx[ok]].astype(float), fw, ft, 0.0)
        snr = df.SNR_APER_3.to_numpy()[sel_idx][ok]
        rows = []
        frow = dict(name=Path(path).name, dataset=dataset, filter=filt,
                    mean_minus_mode=float(np.median(b_mean.background - b_mode.back())))
        for tag, mags in [("ezV0", m10_v0), ("sep_mode", out["mode"]), ("sep_mean", out["mean"])]:
            inst = mags[sel_idx][ok]
            zc = syn - inst
            good = np.isfinite(zc) & (inst < 90)
            zp, zsd, nz = TR.clipped(zc[good & (snr > 40)])
            frow[f"zp_{tag}"] = zp
            res = zc - zp
            for k in range(len(SNRB) - 1):
                m = good & (snr >= SNRB[k]) & (snr < SNRB[k + 1])
                if m.sum() >= 3:
                    rows.append(dict(name=frow["name"], dataset=dataset, variant=tag, snr_lo=SNRB[k],
                                     res=float(np.median(res[m])), n=int(m.sum())))
        # sanity: sep-mode vs ezphot V0 for bright stars
        bright = (m10_v0 < 90) & (out["mode"] < 90) & (np.asarray(d["SNR_APER_3"] if "SNR_APER_3" in d.columns.names else 1.0857/np.clip(d["MAGERR_APER_3"],1e-4,None), float) > 50)
        frow["sepmode_minus_v0_bright_mmag"] = 1e3 * float(np.median(out["mode"][bright] - m10_v0[bright])) if bright.sum() > 10 else np.nan
        return frow, rows
    except Exception as e:                                           # noqa: BLE001
        return dict(name=Path(path).name, error=str(e)[:300]), []


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fr = pd.read_csv(PILOT / "pilot_frames.csv")
    jobs = [(r.path, r.dataset, r.object, r["filter"]) for _, r in fr.iterrows()]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 8, initializer=_init) as pool:
        res = pool.map(run, jobs)
    FR = pd.DataFrame([r[0] for r in res]); B = pd.DataFrame([x for r in res for x in r[1]])
    fr["name"] = [Path(p).name for p in fr.path]
    FR = FR.merge(fr[["name", "exptime"]], on="name", how="left")
    FR.to_csv(LOG / "bkg_meantest_frames.csv", index=False); B.to_csv(LOG / "bkg_meantest_bins.csv", index=False)
    if "error" in FR.columns:
        print("errors:", FR.error.notna().sum(), FR.error.dropna().head(3).tolist())
    FR["grp"] = FR.dataset + " " + FR.exptime.astype(int).astype(str) + "s"
    FR["dzp_mean_vs_v0"] = 1e3 * (FR.zp_sep_mean - FR.zp_ezV0)
    FR["dzp_mode_vs_v0"] = 1e3 * (FR.zp_sep_mode - FR.zp_ezV0)
    print(FR.groupby("grp")[["mean_minus_mode", "sepmode_minus_v0_bright_mmag", "dzp_mode_vs_v0", "dzp_mean_vs_v0"]].median().round(2).to_string())
    B = B.merge(FR[["name", "grp"]], on="name")
    fig, ax = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
    summary = {}
    for a_, v in zip(ax, ["ezV0", "sep_mode", "sep_mean"]):
        for grp, g in B[B.variant == v].groupby("grp"):
            b = g.groupby("snr_lo").res.median()
            rel = 1e3 * (b - b.iloc[-1])
            a_.plot(b.index, rel, "o-", label=grp)
            summary[(v, grp)] = round(float(rel.iloc[0]), 1)
        a_.axhline(0, color="k", lw=0.6); a_.grid(alpha=0.3); a_.set_xscale("log"); a_.legend(fontsize=8)
        a_.set(title={"ezV0": "ezphot default (SExtractor mode bkg)", "sep_mode": "sep, SExtractor-mode bkg",
                      "sep_mean": "sep, masked 5-sigma-clipped MEAN bkg"}[v], xlabel="SNR")
    ax[0].set_ylabel("XP residual rel. to brightest bin [mmag]")
    fig.suptitle("A9  Faint-star bias vs background estimator (same stars, same apertures, 10 arcsec)")
    fig.tight_layout(); fig.savefig(LOG / "A9_bkg_meantest.png", dpi=130)
    print("faint-end (SNR 20-25) residual rel. to brightest bin [mmag]:")
    for k, v in sorted(summary.items()):
        print("  ", k, v)
