#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
261004_calspec_inspect.py — per-frame image inspection of the CALSPEC 2024 standards with large red-filter offsets
(WD0308-565, WD2149+021, WD2039-682, WD1919+145; m775-m875 and m700 as control), for the stand-alone inspection report.

Per frame (section 8.1 kept frames, manufacturer·beam): a 161 x 161 px cutout around the standard with the 10" / 15"
apertures, the 10-20" annulus and the catalog neighbours (mask r = 1.5 FWHM); the radial profile against the frame PSF
(median profile of bright isolated stars, scaled to the standard's 10" flux); the annulus pixel histogram with the
background estimates (global map at the standard = (own exact 10" sum - SExtractor 10" flux) / area; annulus 3-sigma
median, 5-sigma mean, 5-sigma mean with sources masked); and the offsets Delta of every photometry version of
261004_calspec_annulus.py.
Outputs (~/ezphot/log/20261004_calspec_inspect/): I_<star>_<filter>.png, frames.csv
  MPLCONFIGDIR=/tmp/ezphot-matplotlib OMP_NUM_THREADS=1 python -u ezphot/analysis/261004_calspec_inspect.py --n-proc 16
"""
import argparse
import importlib
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402
from astropy.io import fits       # noqa: E402
from astropy.table import Table   # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
A = importlib.import_module("261004_calspec_annulus")
SS = A.SS
OUT = Path("/home/hhchoi1022/ezphot/log/20261004_calspec_inspect")
SEL = {"WD0308-565": ["m700", "m775", "m800", "m825"], "WD2149+021": ["m700", "m800", "m825", "m850"],
       "WD2039-682": ["m700", "m800", "m825", "m850"], "WD1919+145": ["m700", "m800", "m825", "m850", "m875"]}
MAXROWS = 8
HALF = 80                                            # cutout half size [px] = 40"
R10, R15 = 5 / A.PIX, 7.5 / A.PIX
VERS = ["glb10_bias", "glb10", "mean5m_10", "med3_10", "glb15", "mean5m_15"]


def deltas():
    """Delta per frame [mmag] for every photometry version (same definition as 261004_calspec_annulus.offsets)."""
    out = {}
    for v in VERS:
        tag = v.replace("_bias", "")
        F = pd.read_csv(A.OUT / f"frames_{tag}.csv")
        F = F[F.std_snr > 10].copy()
        bias = SS.expected_bias(F.std_peak_e.to_numpy(), "CALSPEC") / 1e3 if v.endswith("_bias") else 0.0
        F["d"] = (F.d_gnd - F.zp + F.zp_x - bias) * 1e3
        out[v] = F.set_index("file").d
    return pd.DataFrame(out)


def clip(v, k, stat):
    for _ in range(10):
        c = np.median(v) if stat == "median" else np.mean(v)
        sd = np.std(v); keep = np.abs(v - c) <= k * sd
        if keep.all():
            break
        v = v[keep]
    return (np.median(v) if stat == "median" else np.mean(v)), np.std(v)


def profile(cut, cx, cy, rmax=45):
    yy, xx = np.indices(cut.shape); r = np.hypot(xx - cx, yy - cy)
    edges = np.arange(0, rmax + 1, 1.0); mid = 0.5 * (edges[1:] + edges[:-1])
    prof = np.array([np.nanmean(cut[(r >= a) & (r < b)]) for a, b in zip(edges[:-1], edges[1:])])
    return mid, prof


def frame_data(args):
    path, sx, sy = args
    from photutils.aperture import CircularAperture, aperture_photometry
    name = Path(path).name
    g = Table.read(A.CATG / (name[:-5] + "_cat.fits"))
    img = np.asarray(fits.getdata(path, memmap=False), np.float32)
    ny, nx = img.shape
    x0, y0 = sx - 1.0, sy - 1.0
    xi, yi = int(round(x0)), int(round(y0))
    cut = img[yi - HALF:yi + HALF + 1, xi - HALF:xi + HALF + 1].astype(float)
    cx, cy = x0 - (xi - HALF), y0 - (yi - HALF)
    yy, xx = np.indices(cut.shape); rr = np.hypot(xx - cx, yy - cy)
    ann = (rr >= A.R_IN) & (rr <= A.R_OUT)
    gx, gy = np.asarray(g["X_IMAGE"], float) - 1, np.asarray(g["Y_IMAGE"], float) - 1
    fw = np.nan_to_num(np.asarray(g["FWHM_IMAGE"], float), nan=8.0)
    near = (np.abs(gx - x0) < HALF) & (np.abs(gy - y0) < HALF) & (np.hypot(gx - x0, gy - y0) > 2)
    nb = [(gx[j] - (xi - HALF), gy[j] - (yi - HALF), max(1.5 * fw[j], 6.0)) for j in np.where(near)[0]]
    m = ann.copy()
    for bx, by, br in nb:
        m &= np.hypot(xx - bx, yy - by) > br
    v = cut[ann]; v = v[np.isfinite(v)]; vm = cut[m]; vm = vm[np.isfinite(vm)]
    b_med3, _ = clip(v, 3, "median"); b_mean5, _ = clip(v, 5, "mean"); b_mean5m, sd = clip(vm, 5, "mean")
    s10 = float(aperture_photometry(np.nan_to_num(img), CircularAperture([(x0, y0)], r=R10), method="exact")["aperture_sum"][0])
    j = int(np.argmin(np.hypot(gx - x0, gy - y0)))
    f_glb = 10 ** (-0.4 * float(g["MAG_APER_3"][j]))
    b_map = (s10 - f_glb) / (np.pi * R10 ** 2)
    # frame PSF: bright isolated stars, profile normalised by their own 10" flux (local masked-mean background)
    snr = 1.0857 / np.asarray(g["MAGERR_APER_3"], float)
    cand = np.where((np.asarray(g["FLAGS"]) == 0) & (snr > 200) & (np.asarray(g["FLUX_MAX"]) < 40000)
                    & (gx > 60) & (gx < nx - 60) & (gy > 60) & (gy < ny - 60))[0]
    cand = cand[np.argsort(np.hypot(gx[cand] - x0, gy[cand] - y0))][:40]
    profs = []
    for k in cand:
        if np.sum(np.hypot(gx - gx[k], gy - gy[k]) < 25) > 1:
            continue
        kx, ky = int(round(gx[k])), int(round(gy[k]))
        c2 = img[ky - 50:ky + 51, kx - 50:kx + 51].astype(float)
        r2 = np.hypot(*(np.indices(c2.shape)[::-1] - np.array([gx[k] - (kx - 50), gy[k] - (ky - 50)])[:, None, None]))
        bk, _ = clip(c2[(r2 >= A.R_IN) & (r2 <= A.R_OUT)], 5, "mean")
        fk = np.nansum((c2 - bk)[r2 <= R10])
        _, p = profile(c2 - bk, gx[k] - (kx - 50), gy[k] - (ky - 50))
        profs.append(p / fk)
        if len(profs) >= 15:
            break
    psf = np.nanmedian(np.array(profs), axis=0) if profs else None
    f_std = s10 - np.pi * R10 ** 2 * b_mean5m
    rad, prof = profile(cut, cx, cy)
    return dict(file=name, cut=cut, cx=cx, cy=cy, nb=nb, ann_vals=v, b_med3=b_med3, b_mean5=b_mean5, b_mean5m=b_mean5m,
                b_map=b_map, sigpix=sd, rad=rad, prof=prof, psf=psf, f_std=f_std, fwhm=float(g["FWHM_IMAGE"][j]) * A.PIX,
                n_psf=len(profs))


def plot_group(star, filt, rows, D):
    n = len(rows)
    H = 3.1 * n + 0.7
    fig = plt.figure(figsize=(17, H))
    gs = fig.add_gridspec(n, 4, width_ratios=[1.0, 1.35, 1.05, 0.75], hspace=0.42, wspace=0.25,
                          top=1 - 0.55 / H, bottom=0.35 / H, left=0.01, right=0.99)
    for i, (meta, d) in enumerate(rows):
        b = d["b_mean5m"]; sd = d["sigpix"]
        ax = fig.add_subplot(gs[i, 0])
        im = d["cut"] - b
        lo, hi = -2.5 * sd, max(8 * sd, np.nanpercentile(im, 99.7))
        ax.imshow(np.arcsinh(np.clip(im, lo, None) / sd), origin="lower", cmap="gray_r", vmin=np.arcsinh(lo / sd), vmax=np.arcsinh(hi / sd))
        for r, ls, c in ((R10, "-", "#d62728"), (R15, "--", "#ff7f0e"), (A.R_IN, ":", "#1f77b4"), (A.R_OUT, ":", "#1f77b4")):
            ax.add_patch(plt.Circle((d["cx"], d["cy"]), r, fill=False, ls=ls, color=c, lw=0.9))
        for bx, by, br in d["nb"]:
            ax.add_patch(plt.Circle((bx, by), br, fill=False, color="#2ca02c", lw=0.6, alpha=0.8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{meta['file'][:5]} {meta['file'][6:21]}  {meta['exptime']:.0f}s  X {meta['airmass']:.2f}", fontsize=7.5, loc="left")
        # radial profile near the background level
        ax = fig.add_subplot(gs[i, 1])
        ax.plot(d["rad"], d["prof"], "o-", color="k", ms=2.5, lw=0.8, label="standard (raw image)")
        if d["psf"] is not None:
            ax.plot(d["rad"], b + d["psf"] * d["f_std"], "-", color="#9467bd", lw=1.2, label=f"frame PSF x flux + bkg ({d['n_psf']} stars)")
        for val, c, lab in ((d["b_map"], "#d62728", "global map"), (d["b_med3"], "#17becf", "annulus 3σ median"),
                            (d["b_mean5"], "#1f77b4", "annulus 5σ mean"), (b, "#2ca02c", "annulus 5σ mean, masked")):
            ax.axhline(val, color=c, lw=1.0, ls="--", label=f"{lab} {val:+.2f}")
        for r in (R10, R15, A.R_IN, A.R_OUT):
            ax.axvline(r, color="0.8", lw=0.6)
        ax.set_ylim(b - 1.5, b + 4.0)
        ax.set_xlim(0, 45); ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
        ax.set_xlabel("radius [px] (grey: 10\", 15\", annulus)", fontsize=7); ax.set_ylabel("ADU / px", fontsize=7)
        ax.legend(fontsize=5.6, loc="upper right", ncol=1)
        # core profile on a log scale (shape)
        ax2 = ax.inset_axes([0.08, 0.52, 0.32, 0.44])
        ax2.semilogy(d["rad"], np.clip(d["prof"] - b, 1e-2, None), "k.", ms=2)
        if d["psf"] is not None:
            ax2.semilogy(d["rad"], np.clip(d["psf"] * d["f_std"], 1e-2, None), "-", color="#9467bd", lw=1)
        ax2.set_xlim(0, 20); ax2.tick_params(labelsize=5); ax2.set_title("core (− bkg)", fontsize=5.5)
        # annulus histogram
        ax = fig.add_subplot(gs[i, 2])
        v = d["ann_vals"]
        ax.hist(v, bins=np.linspace(b - 5 * sd, b + 5 * sd, 61), color="0.6")
        for val, c in ((d["b_map"], "#d62728"), (d["b_med3"], "#17becf"), (d["b_mean5"], "#1f77b4"), (b, "#2ca02c")):
            ax.axvline(val, color=c, lw=1.1)
        ax.set_xlabel("annulus pixel value [ADU]", fontsize=7); ax.tick_params(labelsize=7)
        ax.set_title(f"σ_pix {sd:.1f} ADU; map − local {d['b_map'] - b:+.2f} ADU", fontsize=7.5, loc="left")
        # numbers
        ax = fig.add_subplot(gs[i, 3]); ax.axis("off")
        lines = [f"unit {meta['unit']}  MJD {meta['mjd']:.4f}", f"SNR {meta['std_snr']:.1f}  peak {meta['std_peak_e']:.0f} e⁻  FWHM {d['fwhm']:.1f}\"",
                 f"field ZP stars {meta['n_zp']:.0f}  zsd {meta['zsd']:.3f}", "Δ [mmag]:"]
        for v_ in VERS:
            val = D.loc[meta["file"], v_] if meta["file"] in D.index else np.nan
            lines.append(f"  {v_:<11s} {val:+7.1f}")
        ax.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=7.6, transform=ax.transAxes)
    fig.suptitle(f"{star} {filt}: one row per frame (section 8.1 kept). Cutout ±40\" (asinh, − masked annulus mean); red 10\", orange 15\", "
                 "blue annulus 10–20\", green masked neighbours", fontsize=10, y=1 - 0.12 / H)
    fig.savefig(OUT / f"I_{star}_{filt}.png", dpi=95); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n-proc", type=int, default=16)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    K = A.frames()
    full = {Path(f).name: f for f in K.file}
    F = pd.read_csv(A.OUT / "frames_glb10.csv"); F = F[F.std_snr > 10]
    D = deltas()
    sel = []
    for star, filts in SEL.items():
        for flt in filts:
            q = F[(F.object == star) & (F["filter"] == flt)].sort_values("mjd")
            if len(q) > MAXROWS:                                  # m700 control: spread over the units and nights
                q = q.iloc[np.linspace(0, len(q) - 1, MAXROWS).round().astype(int)]
            sel.append(q)
    S = pd.concat(sel)
    S = S[S.file.isin(full)]
    with Pool(a.n_proc) as pool:
        res = pool.map(frame_data, [(full[f], x, y) for f, x, y in zip(S.file, S.std_x, S.std_y)])
    R = {r["file"]: r for r in res}
    rows = []
    for (star, flt), q in S.groupby(["object", "filter"], sort=False):
        plot_group(star, flt, [(m, R[m["file"]]) for m in q.to_dict("records")], D)
        for m in q.to_dict("records"):
            d = R[m["file"]]
            rows.append(dict(star=star, filter=flt, file=m["file"], unit=m["unit"], mjd=round(m["mjd"], 5), exptime=m["exptime"],
                             airmass=round(m["airmass"], 3), snr=round(m["std_snr"], 1), peak_e=round(m["std_peak_e"], 0),
                             n_zp=int(m["n_zp"]), zsd=round(m["zsd"], 4),
                             fwhm_arcsec=round(d["fwhm"], 2), sigpix=round(d["sigpix"], 2), b_map=round(d["b_map"], 3),
                             b_med3=round(d["b_med3"], 3), b_mean5=round(d["b_mean5"], 3), b_mean5m=round(d["b_mean5m"], 3),
                             **{v: round(D.loc[m["file"], v], 1) if m["file"] in D.index else np.nan for v in VERS}))
        print(star, flt, len(q), flush=True)
    pd.DataFrame(rows).to_csv(OUT / "frames.csv", index=False)


if __name__ == "__main__":
    main()
