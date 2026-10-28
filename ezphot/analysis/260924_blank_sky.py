#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260924_blank_sky.py — is the 10-arcsec pedestal a background-estimate bias (present in blank sky) or tied to stars?

For each pilot frame: SExtractor-equivalent background with sep (mesh 64, filter 3, same mode estimator), detection
at 1.5 sigma / minarea 12, then circular apertures (5 and 10 arcsec) on random blank positions (no segmentation
pixel inside the aperture, >= 20 arcsec from any detected centroid).  The mean background-subtracted flux per pixel
in blank apertures is the pedestal the background estimate leaves behind; compare with the star-based p (5/10 pair).
Also reports mean - mode of unmasked sky pixels.  Output: blank_sky.csv
"""
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import sep
from astropy.io import fits
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")
LOG = Path("/home/hhchoi1022/ezphot/log/20260924_stepA")
PIX = 0.505
R5, R10 = 2.5 / PIX, 5.0 / PIX


def run(path):
    try:
        data = np.ascontiguousarray(fits.getdata(path).astype(np.float32))
        bkg = sep.Background(data, bw=64, bh=64, fw=3, fh=3)
        sub = data - bkg.back()
        sep.set_extract_pixstack(10_000_000); sep.set_sub_object_limit(4096)
        objs, seg = sep.extract(sub, 1.5, err=bkg.globalrms, minarea=12, segmentation_map=True)
        segf = (seg > 0).astype(np.float32)
        ny, nx = data.shape
        rng = np.random.default_rng(1)
        x = rng.uniform(60, nx - 60, 40000); y = rng.uniform(60, ny - 60, 40000)
        tree = cKDTree(np.c_[objs["x"], objs["y"]])
        dmin, _ = tree.query(np.c_[x, y])
        keep = dmin > 20.0 / PIX
        x, y = x[keep], y[keep]
        s_seg, _, _ = sep.sum_circle(segf, x, y, R10)
        keep = s_seg == 0
        x, y = x[keep][:4000], y[keep][:4000]
        f10, _, _ = sep.sum_circle(sub, x, y, R10, subpix=5)
        f5, _, _ = sep.sum_circle(sub, x, y, R5, subpix=5)
        a10, a5 = np.pi * R10 ** 2, np.pi * R5 ** 2
        sky = data[seg == 0]
        skysub = sub[seg == 0]
        # robust summaries
        def clipmean(v, k=3.0):
            v = v[np.isfinite(v)]
            for _ in range(5):
                m, s = np.mean(v), np.std(v)
                v = v[np.abs(v - m) < k * s]
            return float(np.mean(v))
        return dict(name=Path(path).name, n_blank=len(x), n_obj=len(objs),
                    ped10_mean=float(np.mean(f10) / a10), ped10_med=float(np.median(f10) / a10),
                    ped5_mean=float(np.mean(f5) / a5), ped5_med=float(np.median(f5) / a5),
                    sky_level=float(np.median(sky)), sky_rms=float(bkg.globalrms),
                    mean_minus_back_unmasked=float(np.mean(skysub)), clipmean_minus_back=clipmean(skysub),
                    median_minus_back=float(np.median(skysub)))
    except Exception as e:                                           # noqa: BLE001
        return dict(name=Path(path).name, error=str(e)[:200])


if __name__ == "__main__":
    fr = pd.read_csv(LOG / "pilot" / "pilot_frames.csv")
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as pool:
        rows = pool.map(run, list(fr.path))
    B = pd.DataFrame(rows)
    fr["name"] = [Path(p).name for p in fr.path]
    B = fr[["name", "dataset", "filter", "exptime"]].merge(B, on="name")
    E = pd.read_csv(LOG / "pilot_eval.csv")
    E["name_img"] = E.name.str.replace("_cat.fits", ".fits")
    B = B.merge(E[["name_img", "p_V0"]], left_on="name", right_on="name_img", how="left").drop(columns="name_img")
    B.to_csv(LOG / "blank_sky.csv", index=False)
    B["grp"] = B.dataset + " " + B.exptime.astype(int).astype(str) + "s"
    pd.set_option("display.width", 220)
    print(B.groupby("grp")[["n_blank", "sky_level", "sky_rms", "ped10_mean", "ped10_med", "ped5_mean", "p_V0",
                            "mean_minus_back_unmasked", "clipmean_minus_back", "median_minus_back"]].median().round(3).T.to_string())
    ok = B.dropna(subset=["ped10_mean", "p_V0"])
    print("\nframe-level corr(blank ped10_mean, star p_V0) =", round(np.corrcoef(ok.ped10_mean, ok.p_V0)[0, 1], 3),
          "| median ratio blank/star =", round(float(np.median(ok.ped10_mean / ok.p_V0)), 3))
