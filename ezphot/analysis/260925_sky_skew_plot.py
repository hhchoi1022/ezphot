#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""260925_sky_skew_plot.py — A11: why the SExtractor-mode background leaves a pedestal (uses sky_skew_test.csv)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sep
from astropy.io import fits
from scipy.ndimage import distance_transform_edt

LOG = Path("/home/hhchoi1022/ezphot/log/20260924_stepA")
PIX = 0.505
S = pd.read_csv(LOG / "sky_skew_test.csv")
S["grp"] = S.dataset + " " + S.exptime.astype(int).astype(str) + "s"
fr = pd.read_csv(LOG / "pilot" / "pilot_frames.csv")
row = fr[(fr.dataset == "SPSS") & (fr.exptime == 100)].iloc[0]

data = np.ascontiguousarray(fits.getdata(row.path).astype(np.float32))
g = float(fits.getheader(row.path)["EGAIN"])
bad = ~np.isfinite(data); data[bad] = 0.0
b0 = sep.Background(data, mask=bad, bw=64, bh=64, fw=3, fh=3)
back = b0.back()
sep.set_extract_pixstack(10_000_000); sep.set_sub_object_limit(4096)
_, seg = sep.extract(data - back, 1.5, err=b0.globalrms, minarea=12, segmentation_map=True, mask=bad)
far = (distance_transform_edt(seg == 0) * PIX > 20) & ~bad
far[:64] = far[-64:] = False; far[:, :64] = far[:, -64:] = False
r = (data - back)[far]
lam = float(np.median(data[far])) * g
sd_e = 1.4826 * np.median(np.abs(r - np.median(r))) * g
rn = np.sqrt(max(sd_e ** 2 - lam, 0.0))
rng = np.random.default_rng(0)
sim = (rng.poisson(lam, r.size) + rng.normal(0, rn, r.size)) / g - lam / g

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "lines.linewidth": 1.0})
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
bins = np.arange(-40.5, 50.5, 2.0)
a = ax[0]
a.hist(r, bins=bins, histtype="step", color="k", density=True, label="observed sky (≥20″ from sources) − sep bkg")
a.hist(sim + (np.mean(r) - np.mean(sim)), bins=bins, histtype="step", color="tab:orange", density=True, ls="--",
       label=f"Poisson(λ={lam:.1f} e⁻) + Gauss({rn:.1f} e⁻), same mean")
for v, c, lab in [(0.0, "tab:blue", "sep 'mode' background = 0"), (np.median(r), "tab:green", f"median  +{np.median(r):.2f}"),
                  (np.mean(r[np.abs(r) < 60]), "tab:red", f"mean  +{np.mean(r[np.abs(r) < 60]):.2f} ADU")]:
    a.axvline(v, color=c, lw=1.2, label=lab)
a.set(xlim=(-35, 45), xlabel="pixel value − background [ADU]", ylabel="density",
      title=f"(a) {row['name'][:40]}…\nsky ≈ {lam:.1f} e⁻/pix (gain {g:.3f} e⁻/ADU): photon noise is skewed")
a.legend(fontsize=7.5, loc="upper right")

a = ax[1]
for grp, gg in S.groupby("grp"):
    a.scatter(gg.sim_bias_sky, gg.ex_all, s=16, label=grp)
lim = [0, 1.9]
a.plot(lim, lim, "k-", lw=0.7); a.set(xlim=lim, ylim=lim)
a.set(xlabel="noise-only simulation: mean − sep bkg [ADU/pix]\n(Poisson λ from the sky level, no sources)",
      ylabel="observed far-sky mean − sep bkg [ADU/pix]",
      title=f"(b) photon skew alone reproduces {1/np.nanmedian(S.ex_all/S.sim_bias_sky):.0%} of the pedestal\n"
            f"(with λ from the observed skewness: {1/np.nanmedian(S.ex_all/S.sim_bias_k3):.0%})")
a.legend(fontsize=7.5); a.grid(alpha=0.3)

a = ax[2]
dcols = [("ex_d3", 4.5), ("ex_d6", 8), ("ex_d10", 15), ("ex_d20", 30), ("ex_d40", 55)]
for grp, gg in S.groupby("grp"):
    a.plot([x for _, x in dcols], [gg[c].median() for c, _ in dcols], "o-", ms=4, label=grp)
a.set(xscale="log", xlabel="distance to nearest detected-source pixel [arcsec]",
      ylabel="sky mean − sep bkg [ADU/pix] (5σ-clipped)",
      title="(c) flat plateau far from sources = estimator bias;\nrise within ~10″ = star wings")
a.axhline(0, color="k", lw=0.6); a.grid(alpha=0.3); a.legend(fontsize=7.5)
fig.suptitle("A11  Why the SExtractor-mode background leaves a ~1 ADU/pix pedestal in 7DT medium-band frames", fontsize=10.5)
fig.tight_layout(); fig.savefig(LOG / "A11_sky_skew.png", dpi=130)
print("lam", lam, "rn", rn, "mean", np.mean(r[np.abs(r) < 60]), "median", np.median(r))
