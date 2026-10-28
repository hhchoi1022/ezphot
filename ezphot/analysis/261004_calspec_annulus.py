#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
261004_calspec_annulus.py — do the large red-filter CALSPEC offsets of faint stars (section 8.1, S1: m800 / m825 of
WD0308-565, WD2149+021, WD2039-682, WD1919+145) come from a residual background?  Re-measure with a local annulus.

Catalogs: the re-photometry catalogs of 260927_std_rephot.py (global background = photutils per-mesh 5-sigma mean map,
BACKGROUND = that map at the source).  For every source of a frame:
    b_ann  = 3-sigma-clipped median of the image in the annulus R_IN-R_OUT px (10-20")
    b_loc  = b_ann - BACKGROUND                      (what the global map missed at that position)
    F_new  = 10**(-0.4 MAG_APER_k) - pi r_k^2 b_loc  (k = 3: 10" diameter, k = 4: 15")
Implemented as our own exact aperture sums on the image minus pi r^2 b_ann (independent of the global map; SExtractor ran
on image - map with BACK_TYPE MANUAL 0, so its BACKGROUND column is not the map).  A new catalog with MAG_APER_3 /
MAG_APER_4 replaced (originals kept as *_GLB, b_ann as BKG_ANN) is written for every frame, and the section 8.1 frame processing (260927_std_offset.process_cal: field ZP from corrected XP, standard
excluded) is rerun on it, so the field ZP stars and the standard are measured the same way.  Curve: manufacturer·beam
n_eff(lam0).  Frames: those kept in section 8.1 (calspec_frames_flags.csv), filters m600 and redder.
Versions: global / annulus x 10" / 15", each with and without the faint-star bias (SS.expected_bias, calibrated on the
global-background photometry).
Outputs (~/ezphot/log/20261004_calspec_annulus/): catalogs_ann/, frames_<version>.csv, resid.csv, B1_resid.png, B2_snr.png
  MPLCONFIGDIR=/tmp/ezphot-matplotlib OMP_NUM_THREADS=1 python -u ezphot/analysis/261004_calspec_annulus.py --n-proc 24
"""
import argparse
import importlib
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402
from astropy.io import fits       # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
CM = importlib.import_module("261001_calspec_manuf_beam")
SO, SS = CM.SO, CM.SS
OUT = Path("/home/hhchoi1022/ezphot/log/20261004_calspec_annulus")
CAL = Path("/home/hhchoi1022/ezphot/log/20261001_calspec_manuf")
CATA = OUT / "catalogs_ann"                       # default estimator (kept for the first run: med3)
# annulus estimators: med3 = 3-sigma-clipped median; mean5 = 5-sigma-clipped mean; mean5m = 5-sigma-clipped mean with the
# catalog sources masked (r = 1.5 FWHM_IMAGE, >= 6 px) - the global map is a 5-sigma-clipped mean on a source-masked image
ESTS = ["med3", "mean5", "mean5m"]
CATE = {e: OUT / f"catalogs_{e}" for e in ESTS}
CATG = SO.CATD                                      # original (global background) catalogs
PIX = 0.5056                                        # arcsec / px
R_IN, R_OUT = 20, 40                                # annulus [px] = 10-20"
APER = {"MAG_APER_3": 10.0, "MAG_APER_4": 15.0}     # diameter [arcsec]
RED_MIN = 600
FOCUS = ["WD0308-565", "WD2149+021", "WD2039-682", "WD1919+145"]


def frames():
    K = pd.read_csv(CAL / "calspec_frames_flags.csv", usecols=["file", "object", "filter", "unit", "curve", "keep", "std_snr"])
    K = K[(K.curve == "beam") & K.keep & K["filter"].str.startswith("m")]
    K = K[pd.to_numeric(K["filter"].str[1:4], errors="coerce") >= RED_MIN].drop_duplicates("file")
    M = pd.read_csv(SO.OUT / "rephot_meta.csv")                       # full image paths (the flags file keeps base names)
    M = M[(M.dataset == "CALSPEC") & M.status.isin(["ok", "cached"])]
    full = {Path(f).name: f for f in M.file}
    K = K.assign(file=K.file.map(lambda f: full.get(Path(f).name)))
    return K[K.file.notna()]


def _clip(v, k, stat):
    for _ in range(10):
        c = np.median(v) if stat == "median" else np.mean(v)
        sd = np.std(v)
        keep = np.abs(v - c) <= k * sd
        if keep.all():
            break
        v = v[keep]
    return np.median(v) if stat == "median" else np.mean(v)


def annulus_catalog(path):
    """for every estimator write CATE[e]/<name>_cat.fits with MAG_APER_3 / MAG_APER_4 = own exact aperture sum on the image
    minus pi r^2 x annulus level (R_IN-R_OUT px); independent of the global background map (SExtractor ran on image - map
    with BACK_TYPE MANUAL 0).  Original SExtractor magnitudes kept as *_GLB, the annulus level as BKG_ANN."""
    from photutils.aperture import CircularAperture, aperture_photometry
    name = Path(path).name[:-5] + "_cat.fits"
    if all((CATE[e] / name).exists() for e in ESTS):
        return path, "cached"
    try:
        with fits.open(CATG / name) as h:
            hdr0 = h[0].header.copy(); tab = h["LDAC_OBJECTS"].data; cols = h["LDAC_OBJECTS"].columns
            t = {c: np.array(tab[c]) for c in cols.names}
        img = np.asarray(fits.getdata(path, memmap=False), np.float32)
        bad = ~np.isfinite(img)
        ny, nx = img.shape
        yy, xx = np.mgrid[-R_OUT:R_OUT + 1, -R_OUT:R_OUT + 1]
        rr = np.hypot(xx, yy)
        ann = (rr >= R_IN) & (rr <= R_OUT)
        x0, y0 = t["X_IMAGE"] - 1.0, t["Y_IMAGE"] - 1.0                  # 0-based pixel centres
        xi, yi = np.round(x0).astype(int), np.round(y0).astype(int)
        inside = (xi >= R_OUT) & (xi < nx - R_OUT) & (yi >= R_OUT) & (yi < ny - R_OUT)
        rmask = np.maximum(1.5 * np.nan_to_num(t["FWHM_IMAGE"].astype(float), nan=8.0), 6.0)
        # neighbours within R_OUT + rmask of each source, for the masked estimator
        from scipy.spatial import cKDTree
        tree = cKDTree(np.column_stack([x0, y0]))
        b = {e: np.full(len(xi), np.nan) for e in ESTS}
        for i in np.where(inside)[0]:
            cut = img[yi[i] - R_OUT:yi[i] + R_OUT + 1, xi[i] - R_OUT:xi[i] + R_OUT + 1]
            v = cut[ann]; fin = np.isfinite(v)
            b["med3"][i] = _clip(v[fin], 3.0, "median")
            b["mean5"][i] = _clip(v[fin], 5.0, "mean")
            m = ann.copy()
            for j in tree.query_ball_point([x0[i], y0[i]], R_OUT + 20.0):
                if j == i:
                    continue
                m &= np.hypot(xx - (x0[j] - xi[i]), yy - (y0[j] - yi[i])) > rmask[j]
            vm = cut[m]; vm = vm[np.isfinite(vm)]
            b["mean5m"][i] = _clip(vm, 5.0, "mean") if vm.size > 200 else np.nan
        pos = np.column_stack([x0, y0])
        sums = {col: np.asarray(aperture_photometry(np.where(bad, 0.0, img), CircularAperture(pos, r=dia / 2 / PIX), method="exact")["aperture_sum"], float)
                for col, dia in APER.items()}
        for e in ESTS:
            new = dict(t); new["BKG_ANN"] = b[e]
            for col, dia in APER.items():
                f = sums[col] - np.pi * (dia / 2 / PIX) ** 2 * b[e]
                m_ = t[col].astype(float)
                new[col + "_GLB"] = m_
                new[col] = np.where(inside & np.isfinite(b[e]) & (m_ < 90) & (f > 0), -2.5 * np.log10(np.clip(f, 1e-30, None)), 99.0)
            hdu = fits.BinTableHDU.from_columns([fits.Column(name=k, format="J" if v.dtype.kind in "iu" else "D", array=v)
                                                 for k, v in new.items()], name="LDAC_OBJECTS")
            fits.HDUList([fits.PrimaryHDU(header=hdr0), hdu]).writeto(CATE[e] / name, overwrite=True)
        return path, "ok"
    except Exception as e:                                             # noqa: BLE001
        return path, "error:" + str(e)[:200]


def _init(catd, mag):
    warnings.filterwarnings("ignore")
    SO.AN._init()
    SO.TW._init(40)
    SO.CATD = Path(catd)
    SO.MAG = mag
    SO._TOTALS = {("manufacturer", "beam"): CM.manuf_totals()[("manufacturer", "beam")]}


def run(K, catd, mag, n_proc, tag):
    jobs = [(obj, list(zip(g.file, g["filter"], g.unit))) for (obj, _), g in K.groupby(["object", "filter"])]
    rows = []
    with Pool(n_proc, initializer=_init, initargs=(str(catd), mag)) as pool:
        for r in pool.imap_unordered(SO.process_cal, jobs):
            rows += r
    F = pd.DataFrame(rows)
    F.to_csv(OUT / f"frames_{tag}.csv", index=False)
    print(time.strftime("[%H:%M:%S]"), f"{tag}: {len(F)} rows, {F.file.nunique()} frames", flush=True)
    return F


def offsets(F, bias_on):
    F = F[F.std_snr > 10].copy()
    bias = SS.expected_bias(F.std_peak_e.to_numpy(), "CALSPEC") / 1e3 if bias_on else 0.0
    F["d"] = (F.d_gnd - F.zp + F.zp_x - bias) * 1e3
    F = F[F.d.notna() & (F.d.abs() <= 1e3 * SS.D_MAX)]
    return F


def _fc_init(catd):
    FC = importlib.import_module("260927_std_faintcheck")
    FC.CATD = Path(catd); FC.MAG = "MAG_APER_3"
    FC._init()
    globals()["_FC"] = FC


def _fc_one(item):
    return _FC.frame_stars(item)


def faintcheck(items, catd, n_proc):
    """field stars (FLAGS 0, isolated, SNR >= 15, corrected XP) of the same frames: residual vs the SNR >= 100 stars (260927_std_faintcheck)"""
    with Pool(n_proc, initializer=_fc_init, initargs=(str(catd),)) as pool:
        res = [r for r in pool.map(_fc_one, items) if r is not None]
    A = pd.concat(res, ignore_index=True)
    return A[np.isfinite(A.res) & (A.res.abs() < 500)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n-proc", type=int, default=24); ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--fc-frames", type=int, default=400, help="frames sampled for the field-star check")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for d in CATE.values():
        d.mkdir(parents=True, exist_ok=True)
    K = frames()
    print(time.strftime("[%H:%M:%S]"), f"frames (section 8.1 kept, >= m{RED_MIN}): {len(K)}", flush=True)
    with Pool(a.n_proc) as pool:
        st = pd.Series(dict(pool.imap_unordered(annulus_catalog, K.file.tolist())))
    print(time.strftime("[%H:%M:%S]"), "annulus catalogs:", st.str[:5].value_counts().to_dict(), flush=True)
    if (~st.str.startswith(("ok", "cach"))).any():
        print(st[~st.str.startswith(("ok", "cach"))].head().to_string())
    runs = {"glb10": (CATG, "MAG_APER_3"), "glb15": (CATG, "MAG_APER_4")}
    for e in ESTS:
        runs[f"{e}_10"] = (CATE[e], "MAG_APER_3"); runs[f"{e}_15"] = (CATE[e], "MAG_APER_4")
    Fs = {}
    for tag, (catd, mag) in runs.items():
        fc = OUT / f"frames_{tag}.csv"
        Fs[tag] = run(K, catd, mag, a.n_proc, tag) if a.rerun or not fc.exists() else pd.read_csv(fc)
    rows = []
    for tag in Fs:
        for bias_on in (True, False):
            F = offsets(Fs[tag], bias_on)
            P = F.groupby(["object", "filter"]).agg(resid=("d", "median"), n=("d", "size"), snr=("std_snr", "median")).reset_index()
            rows.append(P.assign(version=tag + ("_bias" if bias_on else "")))
    R = pd.concat(rows, ignore_index=True)
    R.round(2).to_csv(OUT / "resid.csv", index=False)
    chk = R[R.version == "glb10_bias"].merge(pd.read_csv(CAL / "calspec_sed_long.csv").query("curve == 'beam'")[["object", "filter", "resid"]],
                                               on=["object", "filter"], suffixes=("", "_81"))
    print("glb10_bias vs section 8.1: max |diff| =", round(float((chk.resid - chk.resid_81).abs().max()), 3), "mmag over", len(chk))
    cols = ["glb10_bias", "glb10"] + [f"{e}_10" for e in ESTS] + ["glb15"] + [f"{e}_15" for e in ESTS]
    W = R.pivot_table(index=["object", "filter"], columns="version", values="resid")[cols]
    W["snr"] = R[R.version == "glb10"].set_index(["object", "filter"]).snr
    fk = lambda f: int(f[1:4])   # noqa: E731
    pd.set_option("display.width", 250)
    for o in FOCUS + sorted(set(R.object) - set(FOCUS)):
        q = W.loc[o]; q = q.reindex(sorted(q.index, key=fk))
        print(f"== {o}\n", q.round(1).to_string())
    W2 = W.reset_index(); W2 = W2[W2["filter"].map(fk) >= 700]
    W2["b"] = pd.cut(W2.snr, [0, 20, 30, 50, 100, 1e9])
    print("standards, m700 and redder, median by SNR bin:\n", W2.groupby("b")[cols].median().round(1).to_string())
    W2.round(2).to_csv(OUT / "resid_red_wide.csv", index=False)

    # field-star check on a sample of the same frames, every background version (10")
    S = K.sample(min(a.fc_frames, len(K)), random_state=1)
    items = [("CALSPEC", f, o, flt) for f, o, flt in zip(S.file, S.object, S["filter"])]
    B = []
    for tag, catd in [("glb", CATG)] + [(e, CATE[e]) for e in ESTS]:
        A = faintcheck(items, catd, a.n_proc)
        A.to_csv(OUT / f"faintcheck_stars_{tag}.csv", index=False)
        for lo, hi in [(15, 20), (20, 30), (30, 50), (50, 100), (100, 200), (200, 1e9)]:
            x = A[(A.snr >= lo) & (A.snr < hi)]
            B.append(dict(version=tag, lo=lo, hi=hi, n=len(x), frames=x.file.nunique(), res=x.res.median(),
                          err=1.2533 * 1.4826 * np.median(np.abs(x.res - x.res.median())) / np.sqrt(max(len(x), 1))))
        print(time.strftime("[%H:%M:%S]"), f"field-star check {tag}: {A.file.nunique()} frames, {len(A)} stars", flush=True)
    B = pd.DataFrame(B); B.round(2).to_csv(OUT / "faintcheck_bins.csv", index=False)
    print("field stars, residual vs SNR >= 100 stars [mmag] (+ = measured too bright):\n",
          B.pivot_table(index=["lo", "hi"], columns="version", values="res").round(1).to_string())

    # B1: per star, red filters, background versions (10")
    stars = FOCUS + [o for o in sorted(R.object.unique()) if o not in FOCUS]
    nc = 4; nr = int(np.ceil(len(stars) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(4.6 * nc, 3.3 * nr), sharex=True, sharey=True, squeeze=False)
    sty = {"glb10_bias": ("k", "o-", "global map + faint bias (section 8.1)"), "glb10": ("0.55", ".--", "global map, no bias"),
           "med3_10": ("#9ecae1", "v:", "annulus 3σ median"), "mean5_10": ("#3182bd", "s-", "annulus 5σ mean"),
           "mean5m_10": ("#e6550d", "D-", "annulus 5σ mean, sources masked")}
    for ax, o in zip(axs.ravel(), stars):
        for j, (v, (c, mk, lab)) in enumerate(sty.items()):
            q = R[(R.object == o) & (R.version == v)].copy()
            q = q.assign(k=q["filter"].map(fk)).sort_values("k")
            ax.plot(q.k + (j - 2) * 1.2, q.resid, mk, color=c, ms=3.5, lw=0.9, label=lab)
        ax.axhline(0, color="k", lw=0.5); ax.axhspan(-10, 10, color="0.92", zorder=0); ax.grid(alpha=0.3); ax.set_ylim(-150, 160)
        ax.set_title(o, fontsize=8.5, loc="left")
    for ax in axs.ravel()[len(stars):]:
        ax.axis("off")
    axs[0, 0].legend(fontsize=6.3, loc="upper left")
    fig.supxlabel("filter centre [nm]"); fig.supylabel("residual obs − syn [mmag], manuf·beam, 10\" aperture")
    fig.suptitle("B1  CALSPEC 2024 red filters: global background map vs local annulus (10–20\") with three estimators; "
                 "field ZP stars and the standard measured the same way", fontsize=10)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.96)); fig.savefig(OUT / "B1_resid.png", dpi=110); plt.close(fig)
    # B2: field-star check and the standards vs SNR
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 4.8))
    cmap = {"glb": "k", "med3": "#9ecae1", "mean5": "#3182bd", "mean5m": "#e6550d"}
    for v, g in B.groupby("version"):
        x = np.sqrt(g.lo * np.where(np.isfinite(g.hi) & (g.hi < 1e8), g.hi, 400))
        axs[0].errorbar(x, g.res, g.err, fmt="o-", color=cmap[v], ms=5, capsize=0, label={"glb": "global map"}.get(v, f"annulus {v}"))
    axs[0].set_xscale("log"); axs[0].axhline(0, color="k", lw=0.5); axs[0].grid(alpha=0.3)
    axs[0].set_xlabel("field-star SNR (10\")"); axs[0].set_ylabel("residual vs SNR ≥ 100 stars [mmag] (+ = too bright)")
    axs[0].set_title(f"(a) field stars with corrected XP in {S.file.nunique()} sampled frames (m600+)", fontsize=9, loc="left"); axs[0].legend(fontsize=8)
    for v, c in (("glb10", "k"), ("mean5m_10", "#e6550d"), ("mean5_10", "#3182bd")):
        axs[1].semilogx(W2.snr, W2[v], "o", color=c, ms=4, alpha=0.8, label={"glb10": "global map, no bias"}.get(v, v))
    axs[1].axhline(0, color="k", lw=0.5); axs[1].axhspan(-10, 10, color="0.92", zorder=0); axs[1].grid(alpha=0.3); axs[1].set_ylim(-150, 160)
    axs[1].set_xlabel("median SNR of the standard"); axs[1].set_ylabel("standard residual [mmag]")
    axs[1].set_title("(b) CALSPEC standards, filters m700 and redder, 10\", no faint bias", fontsize=9, loc="left"); axs[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "B2_snr.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
