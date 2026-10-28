#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
261008_wd2149_calib_check.py — does the bias/dark calibration (hot-pixel over-correction) explain the 2024 WD2149+021 m825
offset?  7DT10, night 2024-08-10: m800 (20 x 30 s, 04:40-04:52 UT) and m825 (20 x 30 s, 04:55-05:07 UT) used the SAME
master bias (20240811) and the SAME master dark (dark_30s 20240809, two nights earlier), and their own flats.

Per frame, on a cutout around the white dwarf (raw frames from /lyman/data1/obsdata, masters from
/lyman/data2/master_frame, pipeline-calibrated frames from the ezphot scidata tree; everything read-only):
  1. reproduce the pipeline calibration (raw - bias - dark) / flat and compare it with the pipeline frame;
  2. 10" aperture (r = 9.9 px at the catalogue position) minus a 20-40 px annulus (3-sigma clipped mean), for
     (a) the pipeline frame, (b) the reproduction, (c) the same-night dark (dark_30s 20240811), (d) hot pixels replaced
     by the median of their non-hot 5x5 neighbours, (e) the dark replaced by its local median (no pixel structure),
     (f) the bias replaced by its local median, (g) the 2024-08-09 bias, (h) both bias and dark smooth;
     the flats of five nights (2024-08-08...13) at the white dwarf: aperture vs annulus (local structure) -> flats.csv;
     dm = -2.5 log10(F_x / F_b) [mmag], + = the variant makes the star fainter, i.e. the pipeline made it brighter;
  3. hot pixels of the used dark inside the aperture: count and summed excess over the local dark median, compared with
     the same quantity for random apertures over the chip (is the WD's aperture special?);
  4. a figure with cutouts: raw - bias (median of the 20 m825 frames), bias, used dark, same-night dark, their
     difference, flat, calibrated median, with the m825 (red) and m800 (blue) apertures.
Outputs: ~/ezphot/log/20261008_wd2149_calib/ (frames.csv, random_apertures.csv, C1_cutouts.png, C2_frames.png)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402
from astropy.io import fits       # noqa: E402
from astropy.stats import sigma_clipped_stats   # noqa: E402
from matplotlib.patches import Circle           # noqa: E402
from scipy.ndimage import median_filter         # noqa: E402

OUT = Path("/home/hhchoi1022/ezphot/log/20261008_wd2149_calib")
RAW = Path("/lyman/data1/obsdata/7DT10/2024-08-10_gain2750")
MF = Path("/lyman/data2/master_frame")
BIAS = MF / "2024-08-10/7DT10/bias_7DT10_20240811_1x1_gain2750_C31164.fits"
BIAS_0809 = MF / "2024-08-08/7DT10/bias_7DT10_20240809_1x1_gain2750_C31164.fits"
DARK_USED = MF / "2024-08-08/7DT10/dark_30s_7DT10_20240809_1x1_gain2750_C31164.fits"
DARK_NIGHT = MF / "2024-08-10/7DT10/dark_30s_7DT10_20240811_1x1_gain2750_C31164.fits"
FLAT = {f: MF / f"2024-08-10/7DT10/flat_{f}_7DT10_20240811_1x1_gain2750_C31164.fits" for f in ("m800", "m825")}
R_AP, R_IN, R_OUT, HALF = 9.9, 20, 40, 70                         # px (10" diameter at 0.505"/px)
HOT_K = 5.0


def cut(path, y0, x0, n=2 * HALF + 1, ext=0):
    with fits.open(path, memmap=True, do_not_scale_image_data=True) as h:
        a = np.asarray(h[ext].section[y0:y0 + n, x0:x0 + n], float)
        return a * float(h[ext].header.get("BSCALE", 1.0)) + float(h[ext].header.get("BZERO", 0.0))


def phot(img, xc, yc):
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
    r = np.hypot(xx - xc, yy - yc)
    ann = img[(r >= R_IN) & (r < R_OUT)]
    bkg = sigma_clipped_stats(ann, sigma=3, maxiters=5)[0]
    ap = r < R_AP
    return float(np.sum(img[ap]) - bkg * ap.sum()), float(bkg)


def hot_fix(img, hot):
    med = median_filter(np.where(hot, np.nan, img), size=5, mode="nearest") if False else None
    out = img.copy()
    pad = np.pad(np.where(hot, np.nan, img), 2, mode="edge")
    for y, x in zip(*np.nonzero(hot)):
        w = pad[y:y + 5, x:x + 5]
        out[y, x] = np.nanmedian(w)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    D = pd.read_csv("/home/hhchoi1022/ezphot/log/20261008_m825_pairs/frames_delta.csv")
    A = pd.read_csv("/home/hhchoi1022/ezphot/log/20261008_m825_pairs/frames_all.csv")
    d = D[(D.obj == "WD2149+021") & (D.curve == "beam") & (D.epoch == 2024) & D["filter"].isin(["m800", "m825"])]
    d = d.merge(A[["name", "file"]].rename(columns={"name": "file", "file": "path"}), on="file").sort_values("mjd")
    xm, ym = d[d["filter"] == "m825"].std_x.median() - 1, d[d["filter"] == "m825"].std_y.median() - 1
    x0, y0 = int(round(xm)) - HALF, int(round(ym)) - HALF
    bias = cut(BIAS, y0, x0); dk_u = cut(DARK_USED, y0, x0); dk_n = cut(DARK_NIGHT, y0, x0)
    flats = {f: cut(p, y0, x0) for f, p in FLAT.items()}
    # hot pixels of the used dark: excess over the local (15 px) median beyond HOT_K robust sigma of the whole cutout
    loc = median_filter(dk_u, size=15, mode="nearest")
    b_0809 = cut(BIAS_0809, y0, x0); bloc = median_filter(bias, size=15, mode="nearest")
    ex = dk_u - loc
    sig = 1.4826 * np.median(np.abs(ex - np.median(ex)))
    hot = ex > HOT_K * sig
    rows, cal_stack, raw_stack = [], {"m800": [], "m825": []}, {"m800": [], "m825": []}
    for r in d.itertuples():
        raw_name = Path(r.path).name
        raw = cut(RAW / raw_name, y0, x0)
        pipe = cut(r.path, y0, x0)
        flat = flats[r.filter]
        cal_b = (raw - bias - dk_u) / flat
        cal_c = (raw - bias - dk_n) / flat
        cal_d = hot_fix(cal_b, hot)
        cal_e = (raw - bias - loc) / flat
        cal_f = (raw - bloc - dk_u) / flat; cal_g = (raw - b_0809 - dk_u) / flat; cal_h = (raw - bloc - loc) / flat
        xc, yc = r.std_x - 1 - x0, r.std_y - 1 - y0
        Fa, ba = phot(pipe, xc, yc); Fb, bb = phot(cal_b, xc, yc); Fc, _ = phot(cal_c, xc, yc)
        Fd, _ = phot(cal_d, xc, yc); Fe, _ = phot(cal_e, xc, yc)
        Ff, _ = phot(cal_f, xc, yc); Fg, _ = phot(cal_g, xc, yc); Fh, _ = phot(cal_h, xc, yc)
        yy, xx = np.mgrid[:raw.shape[0], :raw.shape[1]]
        ap = np.hypot(xx - xc, yy - yc) < R_AP
        rows.append(dict(file=r.file, filter=r.filter, mjd=r.mjd, d=r.d, x=r.std_x, y=r.std_y, flux_pipe=Fa, flux_repro=Fb,
                         repro_vs_pipe_mmag=-2500 * np.log10(Fa / Fb), max_abs_pix_diff=float(np.max(np.abs(pipe - cal_b))),
                         dm_same_night_dark=-2500 * np.log10(Fc / Fb), dm_hot_fixed=-2500 * np.log10(Fd / Fb),
                         dm_smooth_dark=-2500 * np.log10(Fe / Fb), dm_smooth_bias=-2500 * np.log10(Ff / Fb),
                         dm_bias_0809=-2500 * np.log10(Fg / Fb), dm_smooth_both=-2500 * np.log10(Fh / Fb), n_hot_ap=int((hot & ap).sum()),
                         dark_excess_ap_adu=float(np.sum(ex[ap & hot])), dark_used_minus_night_ap_adu=float(np.sum((dk_u - dk_n)[ap]) - np.median(dk_u - dk_n) * ap.sum()),
                         n_neg5_ap=int(np.sum((cal_b - bb)[ap] < -5 * 5.75)), bkg_ann=bb))
        cal_stack[r.filter].append(cal_b); raw_stack[r.filter].append(raw - bias)
    F = pd.DataFrame(rows)
    F.to_csv(OUT / "frames.csv", index=False)

    # random apertures over the chip: hot-pixel excess of the used dark (is the WD's spot special?)
    rng = np.random.default_rng(3)
    with fits.open(DARK_USED, memmap=True, do_not_scale_image_data=True) as h:
        ny, nx = h[0].data.shape
        rr = []
        for _ in range(1500):
            xr, yr = rng.integers(200, nx - 200), rng.integers(200, ny - 200)
            c = np.asarray(h[0].section[yr - 30:yr + 31, xr - 30:xr + 31], float)
            lc = median_filter(c, size=15, mode="nearest"); e = c - lc
            yy, xx = np.mgrid[:61, :61]; apr = np.hypot(xx - 30, yy - 30) < R_AP
            hh = e > HOT_K * sig
            rr.append(dict(x=xr, y=yr, n_hot=int((hh & apr).sum()), excess=float(np.sum(e[apr & hh]))))
    RA = pd.DataFrame(rr); RA.to_csv(OUT / "random_apertures.csv", index=False)

    pd.set_option("display.width", 250)
    print(F[["filter", "file", "d", "repro_vs_pipe_mmag", "max_abs_pix_diff", "dm_same_night_dark", "dm_hot_fixed", "dm_smooth_dark",
             "n_hot_ap", "dark_excess_ap_adu", "flux_repro"]].round(2).to_string(index=False))
    print(F.groupby("filter")[["d", "repro_vs_pipe_mmag", "dm_same_night_dark", "dm_hot_fixed", "dm_smooth_dark", "dm_smooth_bias", "dm_bias_0809",
                               "dm_smooth_both", "n_hot_ap", "dark_excess_ap_adu", "flux_repro"]].median().round(2).to_string())
    # flats of five nights at the white dwarf: local structure (aperture / 20-40 px annulus) and level
    rows = []
    for flt in ("m800", "m825"):
        g = F[F["filter"] == flt]; xc, yc = g.x.median() - 1, g.y.median() - 1
        for night in ("2024-08-08", "2024-08-10", "2024-08-11", "2024-08-12", "2024-08-13"):
            pth = sorted((MF / night / "7DT10").glob(f"flat_{flt}_7DT10_*.fits"))
            if not pth:
                continue
            c = cut(pth[0], int(round(yc)) - HALF, int(round(xc)) - HALF)
            yy, xx = np.mgrid[:c.shape[0], :c.shape[1]]; rr_ = np.hypot(xx - (xc - int(round(xc)) + HALF), yy - (yc - int(round(yc)) + HALF))
            rows.append(dict(filter=flt, night=night, flat_ap=np.nanmean(c[rr_ < R_AP]), flat_ann=np.nanmean(c[(rr_ >= R_IN) & (rr_ < R_OUT)])))
    FL = pd.DataFrame(rows); FL["ap_over_ann"] = FL.flat_ap / FL.flat_ann
    FL["local_vs_used_mmag"] = FL.groupby("filter").ap_over_ann.transform(lambda v: -2500 * np.log10(v / v.iloc[1]))
    FL.to_csv(OUT / "flats.csv", index=False); print(FL.round(4).to_string(index=False))
    print("random apertures: n_hot median %.0f (p90 %.0f), excess median %.0f ADU (p90 %.0f, p99 %.0f)" %
          (RA.n_hot.median(), RA.n_hot.quantile(0.9), RA.excess.median(), RA.excess.quantile(0.9), RA.excess.quantile(0.99)))
    print(f"dark robust sigma in the cutout {sig:.2f} ADU; hot pixels in the cutout {int(hot.sum())}")

    # C1: cutouts
    med_raw = np.median(raw_stack["m825"], axis=0); med_cal = np.median(cal_stack["m825"], axis=0); med_cal8 = np.median(cal_stack["m800"], axis=0)
    panels = [("raw − bias, m825 median", med_raw, None), ("master bias − median", bias - np.median(bias), (-6, 12)),
              ("dark used (08-09)", dk_u, (-6, 20)), ("dark same night (08-11)", dk_n, (-6, 20)), ("dark used − same night", dk_u - dk_n, (-10, 10)),
              ("flat m825", flats["m825"], None), ("calibrated, m825 median", med_cal, None), ("calibrated, m800 median", med_cal8, None),
              (f"hot pixels of the used dark (> {HOT_K:.0f}σ)", hot.astype(float), (0, 1))]
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for ax, (t, im, lim) in zip(axes.ravel(), panels):
        if lim is None:
            m_, s_ = np.median(im), 1.4826 * np.median(np.abs(im - np.median(im)))
            lim = (m_ - 3 * s_, m_ + 8 * s_)
        ax.imshow(im, origin="lower", cmap="gray" if "hot" not in t else "Reds", vmin=lim[0], vmax=lim[1], interpolation="nearest")
        for rr_ in d[d["filter"] == "m825"].itertuples():
            ax.add_patch(Circle((rr_.std_x - 1 - x0, rr_.std_y - 1 - y0), R_AP, fill=False, ec="#d6604d", lw=0.5, alpha=0.6))
        for rr_ in d[d["filter"] == "m800"].itertuples():
            ax.add_patch(Circle((rr_.std_x - 1 - x0, rr_.std_y - 1 - y0), R_AP, fill=False, ec="#4393c3", lw=0.5, alpha=0.6))
        ax.set_title(t, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"WD2149+021, 7DT10 2024-08-11, cutout {2 * HALF + 1} px around x={xm + 1:.0f}, y={ym + 1:.0f}; red = m825 apertures, blue = m800", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "C1_cutouts.png", dpi=130); plt.close(fig)

    # C2: per-frame effect of each variant
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for k, (c, lab) in enumerate((("dm_same_night_dark", "same-night dark"), ("dm_hot_fixed", "hot pixels replaced"), ("dm_smooth_dark", "dark without pixel structure"))):
        for f, col in (("m800", "#4393c3"), ("m825", "#d6604d")):
            g = F[F["filter"] == f]
            ax.plot(np.arange(len(g)) + (0 if f == "m800" else 21), g[c], ["o", "s", "^"][k], color=col, mfc="w" if k == 0 else col, ms=5, label=f"{f}: {lab}")
    ax.axhline(0, color="0.5", lw=0.8); ax.set_xlabel("frame (m800 0–19, m825 21–40, time order)"); ax.set_ylabel("Δm vs used calibration [mmag]")
    ax.set_title("Change of the white dwarf's 10\" aperture flux when the dark calibration is changed (+ = star fainter)", fontsize=10)
    ax.legend(fontsize=7, ncol=3); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "C2_frames.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
