#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260925_sky_skew_test.py — is the ~1 ADU/pix pedestal just the SExtractor mode estimator on photon-limited
(low-count, Poisson-skewed) sky?  Read-only on images.

Per pilot frame, on sky pixels (no segmentation pixel, >= 20 arcsec from any detection):
  observed : mean, median, second/third cumulants k2, k3 [ADU], mean - sep background
  Poisson  : k3 predicted from photon noise alone = lambda_e / g^3 with lambda_e = sky mean * g (g = EGAIN)
             and lambda_e solved from k3 (lambda_e = k3 g^3) -> compare with the sky level
  simulate : a noise-only image with the same sky level (Poisson) + Gaussian remainder of the variance,
             same sep background (mesh 64, filter 3) -> mean - background; compare with the blank-aperture pedestal
Output: sky_skew_test.csv
"""
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import sep
from astropy.io import fits
from scipy.ndimage import distance_transform_edt

warnings.filterwarnings("ignore")
LOG = Path("/home/hhchoi1022/ezphot/log/20260924_stepA")
PIX = 0.505


def run(path):
    try:
        h = fits.getheader(path)
        g = float(h["EGAIN"])
        data = np.ascontiguousarray(fits.getdata(path).astype(np.float32))
        bad = ~np.isfinite(data); data[bad] = 0.0
        b0 = sep.Background(data, mask=bad, bw=64, bh=64, fw=3, fh=3)
        back = b0.back()
        sep.set_extract_pixstack(10_000_000); sep.set_sub_object_limit(4096)
        objs, seg = sep.extract(data - back, 1.5, err=b0.globalrms, minarea=12, segmentation_map=True, mask=bad)
        dist = distance_transform_edt(seg == 0) * PIX                   # arcsec to the nearest segmentation pixel
        edge = np.zeros(data.shape, bool); edge[:64] = edge[-64:] = True; edge[:, :64] = edge[:, -64:] = True
        far = (dist > 20.0) & ~bad & ~edge
        r_all = (data - back)
        r = r_all[far].astype(np.float64)
        v = data[far].astype(np.float64)
        med = np.median(r); sig = 1.4826 * np.median(np.abs(r - med))  # robust sigma (MAD)
        core = np.abs(r - med) < 5 * sig
        rc = r[core]
        mu = float(np.median(v)); k2 = rc.var(); k3 = np.mean((rc - rc.mean()) ** 3)
        # decomposition of the far-sky excess (mean - background)
        dec = dict(ex_all=float(r.mean()), ex_core5=float(rc.mean()), ex_tail5=float((r * ~core).sum() / r.size),
                   frac_tail5=float((~core).mean()), ex_core3=float(r[np.abs(r - med) < 3 * sig].mean()),
                   med_minus_back=float(med), sig_mad=float(sig))
        for lo_, hi_ in [(3, 6), (6, 10), (10, 20), (20, 40), (40, 1e9)]:
            m = (dist > lo_) & (dist <= hi_) & ~bad & ~edge
            if m.sum() > 1000:
                rr = r_all[m].astype(np.float64)
                dec[f"ex_d{lo_}"] = float(rr[np.abs(rr - med) < 5 * sig].mean())
                dec[f"n_d{lo_}"] = int(m.sum())
        lam_sky = mu * g                                                 # electrons, if the sky level is all photons
        lam_k3 = k3 * g ** 3                                             # electrons implied by the (core) skewness
        # noise-only simulation with the observed level and variance: Poisson(lam) + Gaussian remainder
        rng = np.random.default_rng(0)
        shape = (2048, 2048)
        out = {}
        for tag, lam in [("sky", lam_sky), ("k3", lam_k3)]:
            if not np.isfinite(lam) or lam <= 0:
                out[tag] = np.nan; continue
            rn2 = max(k2 * g ** 2 - lam, 0.0)
            sim = (rng.poisson(lam, shape) + rng.normal(0, np.sqrt(rn2), shape)) / g
            sim = sim.astype(np.float32)
            bs = sep.Background(sim, bw=64, bh=64, fw=3, fh=3)
            out[tag] = float(sim.mean() - bs.back().mean())
            out[tag + "_rn_e"] = float(np.sqrt(rn2))
        return dict(name=Path(path).name, egain=g, n_far=int(far.sum()), sky_median=mu,
                    sd=float(np.sqrt(k2)), k3=k3, skew=k3 / k2 ** 1.5,
                    k3_poisson=mu / g ** 2 if mu > 0 else np.nan,
                    lam_sky_e=lam_sky, lam_k3_e=lam_k3,
                    obs_mean_minus_back=float(r.mean()),
                    sim_bias_sky=out["sky"], sim_bias_k3=out["k3"], sim_rn_e=out.get("sky_rn_e", np.nan), **dec)
    except Exception as e:                                           # noqa: BLE001
        return dict(name=Path(path).name, error=str(e)[:200])


if __name__ == "__main__":
    fr = pd.read_csv(LOG / "pilot" / "pilot_frames.csv")
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as pool:
        rows = pool.map(run, list(fr.path))
    S = pd.DataFrame(rows)
    fr["name"] = [Path(p).name for p in fr.path]
    S = fr[["name", "dataset", "filter", "exptime", "unit"]].merge(S, on="name")
    B = pd.read_csv(LOG / "blank_sky.csv")[["name", "ped10_mean", "p_V0"]]
    S = S.merge(B, on="name", how="left")
    S.to_csv(LOG / "sky_skew_test.csv", index=False)
    if "error" in S.columns and S.error.notna().any():
        print("errors:", S.error.dropna().unique()[:3])
    S["grp"] = S.dataset + " " + S.exptime.astype(int).astype(str) + "s"
    pd.set_option("display.width", 250)
    cols = ["sky_median", "sig_mad", "sd", "skew", "k3", "k3_poisson", "lam_sky_e", "lam_k3_e", "sim_rn_e",
            "ex_all", "ex_core5", "ex_tail5", "frac_tail5", "ex_core3", "med_minus_back",
            "ex_d3", "ex_d6", "ex_d10", "ex_d20", "ex_d40",
            "sim_bias_sky", "sim_bias_k3", "ped10_mean", "p_V0"]
    print(S.groupby("grp")[cols].median().round(3).T.to_string())
    ok = S.dropna(subset=["obs_mean_minus_back", "sim_bias_k3"])
    print("\ncorr(obs mean-back, sim from k3) =", round(np.corrcoef(ok.obs_mean_minus_back, ok.sim_bias_k3)[0, 1], 3),
          "| median ratio obs/sim_k3 =", round(float(np.median(ok.obs_mean_minus_back / ok.sim_bias_k3)), 3),
          "| median ratio obs/sim_sky =", round(float(np.nanmedian(S.obs_mean_minus_back / S.sim_bias_sky)), 3))
