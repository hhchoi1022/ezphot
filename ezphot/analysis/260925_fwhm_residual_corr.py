#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260925_fwhm_residual_corr.py — at fixed SNR, does the faint-star ZP residual follow FWHM?  (from aper15_stars.csv)

Within each exposure group and SNR bin (0.5-mag bin of m_syn - m(SNR=20) whose median catalogue SNR is nearest to
20 / 50 / 110), residuals and FWHM are demeaned per frame, then
  rho   : Spearman correlation (residual vs FWHM)
  hi-lo : median residual of the broadest minus the narrowest FWHM tercile [mmag] (<0: broader stars fainter)
for the star's own FWHM_IMAGE (noisy, shares pixel noise with the flux) and the local FWHM (15 nearest bright
stars; independent of the star's noise), with the original (d_old10) and new (d_new10) backgrounds.
Also the frame-level correlation of frame FWHM with the SNR~20 residual.  Output: fwhm_residual_corr.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

LOG = Path("/home/hhchoi1022/ezphot/log/20260924_stepA")


def nearest(G, t):
    cen = np.arange(-5.875, 0.9, 0.25)
    ms = np.array([G.loc[(G.dm20 >= c - 0.125) & (G.dm20 < c + 0.125), "snr"].median() for c in cen])
    x0 = cen[np.nanargmin(np.abs(ms - t))]
    return (G.dm20 >= x0 - 0.25) & (G.dm20 < x0 + 0.25)


if __name__ == "__main__":
    S = pd.read_csv(LOG / "aper15_stars.csv")
    S = S[(S.n_bright >= 5) & np.isfinite(S.m20) & np.isfinite(S.d_new10) & (S.own_fwhm > 0.5) & (S.own_fwhm < 15)]
    S["grp"] = S.dataset + " " + S.exptime.astype(int).astype(str) + "s"
    rows = []
    for grp, G in S.groupby("grp"):
        for t in (20, 50, 110):
            H = G[nearest(G, t)].copy()
            if len(H) < 30:
                continue
            for c in ["d_new10", "d_old10", "own_fwhm", "loc_fwhm"]:
                H[c + "_dm"] = H[c] - H.groupby("name")[c].transform("median")
            r = dict(grp=grp, snr=t, n=len(H), own_fwhm_med=H.own_fwhm.median(),
                     own_fwhm_iqr=H.own_fwhm.quantile(.75) - H.own_fwhm.quantile(.25))
            for f in ["own_fwhm", "loc_fwhm"]:
                for d in ["d_new10", "d_old10"]:
                    ok = np.isfinite(H[d + "_dm"]) & np.isfinite(H[f + "_dm"])
                    x = H.loc[ok, f + "_dm"].values; y = 1e3 * H.loc[ok, d + "_dm"].values
                    q1, q2 = np.quantile(x, [1 / 3, 2 / 3])
                    r[f"{f[:3]}_{d[2:5]}_rho"] = spearmanr(x, y).correlation
                    r[f"{f[:3]}_{d[2:5]}_hi_lo"] = np.median(y[x >= q2]) - np.median(y[x <= q1])
                r[f"{f[:3]}_dFWHM"] = np.median(x[x >= q2]) - np.median(x[x <= q1])
            rows.append(r)
    T = pd.DataFrame(rows).round(2)
    T.to_csv(LOG / "fwhm_residual_corr.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.to_string(index=False))
    m = (S.dm20 >= -0.375) & (S.dm20 < 0.125)
    F = S[m].groupby(["name", "grp"]).agg(def_new=("d_new10", lambda v: 1e3 * np.median(v)), n=("d_new10", "size")).reset_index()
    F = F.merge(S[S.snr >= 50].groupby("name").loc_fwhm.median().rename("frame_fwhm"), on="name")
    F = F[F.n >= 10]
    for ds in ["CALSPEC", "SPSS"]:
        G = F[F.grp.str.startswith(ds)]
        print(f"frame level {ds}: n={len(G)} Spearman(frame FWHM, SNR~20 residual) = {spearmanr(G.frame_fwhm, G.def_new).correlation:.2f}")
