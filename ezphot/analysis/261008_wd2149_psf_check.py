#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
261008_wd2149_psf_check.py — is the 2024 WD2149+021 m825 offset a PSF / aperture effect?  For every frame of the white
dwarf (2024 7DT10 m800 and m825, 7DT11 m850; 2026 7DT12 m825) the white dwarf and isolated field stars near the image
centre (within 2000 px, no detection within 25", FLAGS 0, peak 300 e- ... 30k ADU) are measured on the pipeline-calibrated
frame with the same method:
  centroid: Gaussian-weighted (sigma 2.5 px) iterative centroid on the cutout after a 30-45 px annulus mean is removed;
  curve of growth: photutils exact circular apertures r = 2...25 px; flux(r) / flux(25 px) [25 px = 12.6" radius];
  half-light radius r50 (interpolated on the curve of growth) and FWHM from a circular 2-D Gaussian fit (r < 10 px);
  ellipticity from Gaussian-weighted second moments.
The white dwarf is faint (SNR ~27 per frame), so its curve of growth is also summed over the frames of a set.
"Aperture deficit" = -2.5 log10[ frac10(WD) / frac10(field) ] with frac10 = flux(9.9 px) / flux(25 px), 9.9 px = the 10"
diameter aperture (MAG_APER_3): + means the white dwarf puts a smaller fraction of its light inside 10" than field stars.
Outputs: ~/ezphot/log/20261008_wd2149_psf/ (stars.csv, summary.csv, P1_profiles.png)
"""
from pathlib import Path
from multiprocessing import Pool
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402
from astropy.io import fits       # noqa: E402
from astropy.stats import sigma_clipped_stats    # noqa: E402
from photutils.aperture import CircularAperture, aperture_photometry   # noqa: E402
from scipy.optimize import curve_fit             # noqa: E402

warnings.filterwarnings("ignore")
OUT = Path("/home/hhchoi1022/ezphot/log/20261008_wd2149_psf")
CATD = Path("/home/hhchoi1022/ezphot/log/20260927_std_newbkg/catalogs")
RADII = np.array([2, 3, 4, 5, 6, 7, 8, 9.9, 12, 14.85, 18, 21, 25.0])
R_NORM, R10, HALF, PIX = 25.0, 9.9, 50, 0.505


def cutout(path, x, y):
    with fits.open(path, memmap=True, do_not_scale_image_data=True) as h:
        a = np.asarray(h[0].section[int(y) - HALF:int(y) + HALF + 1, int(x) - HALF:int(x) + HALF + 1], float)
        return a * float(h[0].header.get("BSCALE", 1.0)) + float(h[0].header.get("BZERO", 0.0)), int(x) - HALF, int(y) - HALF


def gauss(xy, a, x0, y0, s, c):
    x, y = xy
    return a * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * s * s)) + c


def measure(img, xc, yc):
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
    r = np.hypot(xx - xc, yy - yc)
    bkg = sigma_clipped_stats(img[(r >= 30) & (r < 45)], sigma=3, maxiters=5)[0]
    im = np.nan_to_num(img - bkg)
    for _ in range(10):                                   # Gaussian-weighted centroid
        w = np.exp(-((xx - xc) ** 2 + (yy - yc) ** 2) / (2 * 2.5 ** 2)) * np.clip(im, 0, None)
        if w.sum() <= 0:
            break
        xn, yn = (w * xx).sum() / w.sum(), (w * yy).sum() / w.sum()
        if np.hypot(xn - xc, yn - yc) < 1e-3:
            break
        xc, yc = xn, yn
    fl = np.array([aperture_photometry(im, CircularAperture((xc, yc), r=rr), method="exact")["aperture_sum"][0] for rr in RADII])
    w = np.exp(-((xx - xc) ** 2 + (yy - yc) ** 2) / (2 * 3.0 ** 2)) * im
    sxx, syy, sxy = (w * (xx - xc) ** 2).sum() / w.sum(), (w * (yy - yc) ** 2).sum() / w.sum(), (w * (xx - xc) * (yy - yc)).sum() / w.sum()
    lam = np.linalg.eigvalsh([[sxx, sxy], [sxy, syy]])
    ell = 1 - np.sqrt(max(lam[0], 1e-9) / max(lam[1], 1e-9)) if lam[1] > 0 else np.nan
    m = r < 10
    try:
        p, _ = curve_fit(gauss, (xx[m], yy[m]), im[m], p0=(im[m].max(), xc, yc, 2.0, 0.0), maxfev=4000)
        fwhm = 2.3548 * abs(p[3]) * PIX
    except Exception:                                     # noqa: BLE001
        fwhm = np.nan
    return dict(xc=xc, yc=yc, bkg=bkg, fwhm=fwhm, ell=ell, **{f"f{rr:g}": f for rr, f in zip(RADII, fl)})


def one(args):
    path, cat, wdx, wdy, meta = args
    rows = []
    df = fits.getdata(cat, "LDAC_OBJECTS")
    x, y = np.asarray(df["X_IMAGE"], float) - 1, np.asarray(df["Y_IMAGE"], float) - 1
    fmax, flags = np.asarray(df["FLUX_MAX"], float), np.asarray(df["FLAGS"])
    egain = meta["egain"]
    near = np.hypot(x - wdx, y - wdy) < 2000
    # isolation 25" (50 px) against every detection
    from scipy.spatial import cKDTree
    t = cKDTree(np.c_[x, y]); cnt = np.array([len(t.query_ball_point((xi, yi), 50)) for xi, yi in zip(x, y)])
    sel = near & (flags == 0) & (fmax * egain > 300) & (fmax < 30000) & (cnt == 1) & (np.hypot(x - wdx, y - wdy) > 5) \
        & (x > HALF + 5) & (y > HALF + 5) & (x < meta["nx"] - HALF - 5) & (y < meta["ny"] - HALF - 5)
    targets = [("WD", wdx, wdy, np.nan)] + [("field", xi, yi, fm * egain) for xi, yi, fm in zip(x[sel], y[sel], fmax[sel])]
    for kind, xi, yi, pk in targets:
        img, x0, y0 = cutout(path, xi, yi)
        r = measure(img, xi - x0, yi - y0)
        rows.append(dict(kind=kind, x=xi, y=yi, peak_e=pk, **r, **{k: meta[k] for k in ("file", "set", "egain")}))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    D = pd.read_csv("/home/hhchoi1022/ezphot/log/20261008_m825_pairs/frames_delta.csv")
    A = pd.read_csv("/home/hhchoi1022/ezphot/log/20261008_m825_pairs/frames_all.csv")
    d = D[(D.obj == "WD2149+021") & (D.curve == "beam") & D.ok].merge(A[["name", "file"]].rename(columns={"name": "file", "file": "path"}), on="file")
    d = d[((d.epoch == 2024) & d["filter"].isin(["m800", "m825", "m850"])) | ((d.epoch == 2026) & (d["filter"] == "m825"))]
    d["set"] = d.epoch.astype(str) + " " + d["filter"] + " " + d.unit
    jobs = [(r.path, CATD / (r.file[:-5] + "_cat.fits"), r.std_x - 1, r.std_y - 1,
             dict(file=r.file, set=r.set, egain=r.egain, nx=r.nx, ny=r.ny)) for r in d.itertuples()]
    with Pool(12) as pool:
        S = pd.DataFrame([x for rows in pool.map(one, jobs) for x in rows])
    S = S.merge(d[["file", "d", "mjd"]], on="file", how="left")
    for rr in RADII:
        S[f"c{rr:g}"] = S[f"f{rr:g}"] / S[f"f{R_NORM:g}"]
    S["frac10"] = S["c9.9"]
    S["r50"] = [np.interp(0.5, np.r_[0, [row[f"c{rr:g}"] for rr in RADII]], np.r_[0, RADII]) for _, row in S.iterrows()]
    S.to_csv(OUT / "stars.csv", index=False)

    rows = []
    for st, g in S.groupby("set"):
        w = g[g.kind == "WD"]; f = g[(g.kind == "field") & (g[f"f{R_NORM:g}"] > 0)]
        # white dwarf: summed curve of growth over the set (faint), field: median over stars of the per-star curve
        wd_sum = np.array([w[f"f{rr:g}"].sum() for rr in RADII]); wd_c = wd_sum / wd_sum[-1]
        fc = np.array([f[f"c{rr:g}"].median() for rr in RADII])
        # bootstrap the WD summed fraction over frames
        rng = np.random.default_rng(1); bs = []
        for _ in range(2000):
            q = w.sample(len(w), replace=True, random_state=int(rng.integers(1e9)))
            bs.append(q["f9.9"].sum() / q[f"f{R_NORM:g}"].sum())
        rows.append(dict(set=st, n_frames=w.file.nunique(), n_field=len(f), field_per_frame=len(f) / max(w.file.nunique(), 1),
                         wd_fwhm=w.fwhm.median(), field_fwhm=f.fwhm.median(), wd_ell=w.ell.median(), field_ell=f.ell.median(),
                         wd_r50=float(np.interp(0.5, np.r_[0, wd_c], np.r_[0, RADII])), field_r50=f.r50.median(),
                         wd_frac10=wd_c[list(RADII).index(9.9)], wd_frac10_err=float(np.std(bs)), field_frac10=fc[list(RADII).index(9.9)],
                         field_frac10_sd=f.frac10.std(), field_peak_med=f.peak_e.median(),
                         aperture_deficit_mmag=-2500 * np.log10(wd_c[list(RADII).index(9.9)] / fc[list(RADII).index(9.9)]),
                         aperture_deficit_err=2500 / np.log(10) * float(np.std(bs)) / wd_c[list(RADII).index(9.9)],
                         wd_centroid_shift_px=float(np.median(np.hypot(w.xc - HALF - (w.x - w.x.astype(int)), w.yc - HALF - (w.y - w.y.astype(int))))),
                         d_med=w.d.median()))
    T = pd.DataFrame(rows); T.to_csv(OUT / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.round(3).to_string(index=False))

    # figure: curves of growth (WD summed vs field median) per set, and FWHM per frame in time order
    sets = ["2024 m800 7DT10", "2024 m825 7DT10", "2024 m850 7DT11", "2026 m825 7DT12"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    cols = {"2024 m800 7DT10": "#4393c3", "2024 m825 7DT10": "#d6604d", "2024 m850 7DT11": "#7f7f7f", "2026 m825 7DT12": "#1b7837"}
    for st in sets:
        g = S[S.set == st]
        if g.empty:
            continue
        w = g[g.kind == "WD"]; f = g[g.kind == "field"]
        wd_sum = np.array([w[f"f{rr:g}"].sum() for rr in RADII]); wd_c = wd_sum / wd_sum[-1]
        fc = np.array([f[f"c{rr:g}"].median() for rr in RADII])
        axes[0].plot(RADII * PIX, fc, "-", color=cols[st], lw=1.2, alpha=0.6)
        axes[0].plot(RADII * PIX, wd_c, "o--", color=cols[st], ms=4, label=f"{st}: WD (o) / field (line)")
        axes[1].plot(RADII * PIX, wd_c - fc, "o-", color=cols[st], ms=4, label=st)
    for ax in axes[:2]:
        ax.axvline(R10 * PIX, color="0.5", ls=":", lw=1); ax.grid(alpha=0.25); ax.set_xlabel("aperture radius [arcsec]")
    axes[0].set_ylabel("flux(r) / flux(12.6\")"); axes[0].legend(fontsize=7)
    axes[1].axhline(0, color="0.5", lw=0.8); axes[1].set_ylabel("WD − field (curve of growth)"); axes[1].legend(fontsize=7)
    axes[0].set_title("Curve of growth: WD (summed over frames) vs field-star median", fontsize=10)
    axes[1].set_title("Difference (dotted line = 10\" aperture radius)", fontsize=10)
    for st in sets:
        g = S[(S.set == st)]
        w = g[g.kind == "WD"].sort_values("mjd"); f = g[g.kind == "field"].groupby("file").fwhm.median()
        t = (w.mjd - S[S.set.str.startswith("2024")].mjd.min()) * 1440 if st.startswith("2024") else None
        if t is None:
            continue
        axes[2].plot(t, w.fwhm, "o", color=cols[st], ms=4, label=f"{st} WD")
        axes[2].plot(t, f.reindex(w.file).to_numpy(), "-", color=cols[st], lw=1, alpha=0.7, label=f"{st} field median")
    axes[2].set_xlabel("minutes since first 2024 frame"); axes[2].set_ylabel("FWHM [arcsec]"); axes[2].grid(alpha=0.25); axes[2].legend(fontsize=6)
    axes[2].set_title("FWHM per frame, 2024 (m850 7DT11, m800 then m825 7DT10)", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "P1_profiles.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
