#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260926_linearity_test.py — detector linearity from a same-night exposure-time series of one star field
(e.g. 1, 3, 5, 10, 20, 40, 80, 100 s x 5 frames, all units).  XP-free: only Gaia positions and G are used.

Per frame (processed science frames with WCS):
  backgrounds : new = per-mesh 5-sigma clipped mean (mesh_bkg 'mean5', segmentation + 4 px mask); old = sep mode
  stars       : forced 5/10/15" sums at Gaia positions (WCS + median detection offset), isolated (no detection
                within 10" other than itself); peak = max pixel within 3 px minus the new background [ADU and e-]
  frame       : blank 10" aperture residual (new / old), sky median, global rms
Analysis per (unit, filter):
  count rate c = F10/t.  Reference rate of each star = median c over frames where its peak is 400-12000 ADU
  (linear regime: SNR > 100 in that frame, peak < 12000 ADU above sky and raw peak < 15000 ADU).
  Per-frame scale s_f = median(c/c_ref) over reference-range stars (transparency, shutter), iterated.
  Response rho = c / (s_f c_ref) for every star-frame, binned by the star's peak above sky [e-], per exposure time.
    curves overlap in peak  -> loss set by the star's own signal level
    curves split by exposure (sky scales with t) -> loss set by the total pixel level (sky + star)
  Also: bright-end response vs peak ADU, effective exposure offset dt from F_ref = r (t + dt), blank residuals.
Usage:
  python 260926_linearity_test.py "<glob of frames>" | @<frames.txt> <tag> [nproc]
Outputs (~/ezphot/log/20260926_linearity/<tag>/): lin_frames.csv, lin_stars.csv, lin_response.csv, L1/L2 figures
"""
import glob
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
OUTROOT = Path("/home/hhchoi1022/ezphot/log/20260926_linearity")
PIX = 0.505
RAD = {5: 2.5 / PIX, 10: 5.0 / PIX, 15: 7.5 / PIX}
LIN_LO, LIN_HI = 400.0, 12000.0          # reference (linear) peak range above sky [ADU]
_G = {}


def _init(key):
    warnings.filterwarnings("ignore")
    _G["TR"] = importlib.import_module("260924_stepA_triage")
    _G["SW"] = importlib.import_module("260925_bkg_recipe_sweep")
    _G["ref"] = _G["TR"].load_cache(key)


def frame_info(path):
    h = fits.getheader(path)
    name = Path(path).name
    return dict(path=path, name=name, unit=name[:5], filter=str(h.get("FILTER", "")).strip(),
                exptime=float(h.get("EXPTIME", h.get("EXPOSURE", np.nan))), date=str(h.get("DATE-OBS", "")),
                egain=float(h.get("EGAIN", 0.256)), ra=float(h.get("CRVAL1", np.nan)), dec=float(h.get("CRVAL2", np.nan)))


def run(info):
    SW, ref = _G["SW"], _G["ref"]
    try:
        from astropy.wcs import WCS
        from scipy.ndimage import binary_dilation, maximum_filter
        from scipy.spatial import cKDTree
        hdr = fits.getheader(info["path"])
        data = np.ascontiguousarray(fits.getdata(info["path"]).astype(np.float32))
        bad = ~np.isfinite(data); data[bad] = 0.0
        ny, nx = data.shape
        g = info["egain"]
        b0 = sep.Background(data, mask=bad, bw=64, bh=64, fw=3, fh=3)
        sep.set_extract_pixstack(10_000_000); sep.set_sub_object_limit(4096)
        objs, seg = sep.extract(data - b0.back(), 1.5, err=b0.globalrms, minarea=12, segmentation_map=True, mask=bad)
        segm = seg > 0
        back_new = SW.mesh_bkg(data, binary_dilation(segm, iterations=4) | bad, "mean5")
        x, y = WCS(hdr).all_world2pix(np.asarray(ref["ra"], float), np.asarray(ref["dec"], float), 0)
        td = cKDTree(np.c_[objs["x"], objs["y"]])
        dd, ii = td.query(np.c_[x, y])
        gmag = np.asarray(ref["g"], float)
        m = (dd < 4.0) & (gmag < 15.5)
        if m.sum() >= 10:
            x = x + np.median(objs["x"][ii[m]] - x[m]); y = y + np.median(objs["y"][ii[m]] - y[m])
        inside = (x > 60) & (x < nx - 60) & (y > 60) & (y < ny - 60)
        k = np.where(inside)[0]
        x, y = x[k], y[k]
        nnb = np.array([len(td.query_ball_point([a, b], 10.0 / PIX)) for a, b in zip(x, y)])
        dself, _ = td.query(np.c_[x, y])
        iso = (nnb == 0) | ((nnb == 1) & (dself < 1.5 / PIX))
        sub_new = data - back_new
        pk = maximum_filter(sub_new, size=7)                        # peak within +-3 px
        ix = np.clip(np.round(x).astype(int), 0, nx - 1); iy = np.clip(np.round(y).astype(int), 0, ny - 1)
        out = dict(sid=np.asarray(ref["sid"])[k], g=gmag[k], bprp=np.asarray(ref["bprp"], float)[k], x=x, y=y,
                   iso=iso, peak_adu=pk[iy, ix], raw_peak=maximum_filter(data, size=7)[iy, ix],
                   sky_adu=back_new[iy, ix])
        for tag, back in (("new", back_new), ("old", b0.back())):
            sub = data - back
            for r, rad in RAD.items():
                if tag == "old" and r != 10:
                    continue
                f, e, _ = sep.sum_circle(sub, x, y, rad, err=float(b0.globalrms), gain=g, subpix=5)
                out[f"f{r}_{tag}"] = f
                if tag == "new" and r == 10:
                    out["e10"] = e
        S = pd.DataFrame(out)
        for c in ("name", "unit", "filter", "exptime", "egain"):
            S[c] = info[c]
        # blank-sky residuals
        rng = np.random.default_rng(5)
        bx = rng.uniform(60, nx - 60, 40000); by = rng.uniform(60, ny - 60, 40000)
        dmin, _ = td.query(np.c_[bx, by])
        kk = dmin > 20.0 / PIX
        bx, by = bx[kk], by[kk]
        ss, _, _ = sep.sum_circle(segm.astype(np.float32), bx, by, RAD[10])
        bx, by = bx[ss == 0][:4000], by[ss == 0][:4000]
        A10 = np.pi * RAD[10] ** 2
        fr = dict(info)
        for tag, back in (("new", back_new), ("old", b0.back())):
            fb, _, _ = sep.sum_circle(data - back, bx, by, RAD[10], subpix=5)
            fr[f"blank_{tag}"] = float(np.mean(fb) / A10)
        fr.update(sky_med=float(np.median(back_new)), sky_rms=float(b0.globalrms), n_blank=len(bx), n_obj=len(objs))
        return fr, S
    except Exception as e:                                           # noqa: BLE001
        return dict(info, error=str(e)[:300]), None


def analyse_group(S, F):
    """response of every star-frame relative to its linear-regime count rate."""
    S = S[S.iso & np.isfinite(S.f10_new)].copy()
    S["c"] = S.f10_new / S.exptime
    # reference (linear-regime) measurements: SNR > 100 in that frame (noise-independent; implies peak >~150 e-) and
    # below the high-count non-linearity.  Selecting on the measured peak (> 400 ADU) picks noise spikes in
    # bright-sky frames and biases the reference rates of faint stars high.
    S["lin"] = (S.f10_new / S.e10 > 100) & (S.peak_adu < LIN_HI) & (S.raw_peak < 15000)
    S["s_f"] = 1.0
    for _ in range(3):
        cref = S[S.lin].assign(cc=lambda d: d.c / d.s_f).groupby("sid").cc.median().rename("c_ref")
        S = S.drop(columns=[c for c in ("c_ref",) if c in S.columns]).merge(cref, on="sid", how="left")
        sf = S[S.lin & np.isfinite(S.c_ref)].assign(q=lambda d: d.c / d.c_ref).groupby("name").q.median().rename("s_new")
        S = S.merge(sf, on="name", how="left").reset_index(drop=True)
        S["s_f"] = S.s_new.fillna(1.0); S = S.drop(columns="s_new")
    lin = S.lin
    S["rho"] = S.c / (S.s_f * S.c_ref)
    # bin on the EXPECTED peak (reference rate x frame scale x t x frame PSF peak fraction): binning on the measured
    # peak would select noise (low measured peak <-> low measured flux) and fake a low-signal loss
    q = S[S.lin].assign(q=lambda d: d.peak_adu / d.f10_new).groupby("name").q.median().rename("pk_frac")
    S = S.merge(q, on="name", how="left")
    S["peak_e"] = S.c_ref * S.s_f * S.exptime * S.pk_frac * S.egain
    S["peak_meas_e"] = S.peak_adu * S.egain
    S["tot_exp_adu"] = S.peak_e / S.egain + S.sky_adu               # expected total level at the peak (sky + star)
    for r in (5, 15):
        if f"f{r}_new" in S.columns:
            cr = S[S.lin].assign(cc=lambda d: d[f"f{r}_new"] / d.exptime / d.s_f).groupby("sid").cc.median().rename(f"c_ref{r}")
            S = S.merge(cr, on="sid", how="left")
            S[f"rho{r}"] = S[f"f{r}_new"] / S.exptime / (S.s_f * S[f"c_ref{r}"])
    S["rho_old"] = (S.f10_old / S.exptime) / (S.s_f * S.c_ref)
    return S


def binned(S, col, bins, xcol="peak_e"):
    rows = []
    for (t,), g in S.groupby(["exptime"]):
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (g[xcol] >= lo) & (g[xcol] < hi) & np.isfinite(g[col])
            if m.sum() >= 15:
                v = g.loc[m, col]
                rows.append(dict(exptime=t, peak_lo=lo, peak_hi=hi, peak_med=g.loc[m, xcol].median(), n=int(m.sum()),
                                 dmag=-2.5e3 * np.log10(v.median()),
                                 err=abs(2.5e3 / np.log(10) * 1.2533 * v.std() / np.sqrt(m.sum()) / v.median())))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pattern, tag = sys.argv[1], sys.argv[2]
    nproc = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 16
    out = OUTROOT / tag; out.mkdir(parents=True, exist_ok=True)
    REANALYSE = "reanalyse" in sys.argv
    RAWCOLS = ["sid", "g", "bprp", "x", "y", "iso", "peak_adu", "raw_peak", "sky_adu", "f5_new", "f10_new", "f15_new",
               "e10", "f10_old", "name", "unit", "filter", "exptime", "egain"]
    if REANALYSE:
        F = pd.read_csv(out / "lin_frames.csv")
        S = pd.read_csv(out / "lin_stars.csv", usecols=lambda c: c in RAWCOLS)
        S = S[S.iso.astype(bool)] if S.iso.dtype != bool else S
        print("reanalysing", len(S), "star-frames")
    else:
        paths = Path(pattern[1:]).read_text().split() if pattern.startswith("@") else sorted(glob.glob(pattern))
        infos = [frame_info(p) for p in paths]
        print(len(infos), "frames;", pd.DataFrame(infos).groupby(["unit", "filter"]).exptime.agg(lambda v: sorted(set(v))).to_dict())
        TR = importlib.import_module("260924_stepA_triage")
        key = f"LIN_{tag}"
        print("gaia reference:", TR.cache_object((key, np.nanmedian([i["ra"] for i in infos]), np.nanmedian([i["dec"] for i in infos]))))
        with Pool(nproc, initializer=_init, initargs=(key,)) as pool:
            res = pool.map(run, infos)
        F = pd.DataFrame([r[0] for r in res])
        if "error" in F.columns and F["error"].notna().any():
            print("errors:", F["error"].dropna().unique()[:3])
        S = pd.concat([r[1] for r in res if r[1] is not None], ignore_index=True)
        F.to_csv(out / "lin_frames.csv", index=False)
        S[RAWCOLS].to_csv(out / "lin_raw.csv", index=False)
    pd.set_option("display.width", 220)
    print("\nper exposure time (median over frames): sky [ADU], blank residual new / old [ADU/pix]")
    print(F.groupby(["unit", "filter", "exptime"])[["sky_med", "sky_rms", "blank_new", "blank_old"]].median().round(3).to_string())
    bins = np.array([0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384])
    allS, allR = [], []
    for (unit, filt), g in S.groupby(["unit", "filter"]):
        A = analyse_group(g, F[(F.unit == unit) & (F["filter"] == filt)])
        R = binned(A, "rho", bins).assign(unit=unit, filter=filt, aperture=10)
        for r in (5, 15):
            if f"rho{r}" in A.columns:
                R = pd.concat([R, binned(A, f"rho{r}", bins).assign(unit=unit, filter=filt, aperture=r)])
        R = pd.concat([R, binned(A, "rho_old", bins).assign(unit=unit, filter=filt, aperture="10old")])
        tb = np.array([1000, 2000, 4000, 8000, 12000, 16000, 24000, 32000, 48000, 64000, 90000])
        R = pd.concat([R, binned(A, "rho", tb, xcol="tot_exp_adu").assign(unit=unit, filter=filt, aperture="10total")])
        allS.append(A); allR.append(R)
        # effective exposure offset from the frame scale: F_ref ~ r (t + dt)  ->  s_f * t = t + dt  (relative)
        sf = A.groupby(["name", "exptime"]).s_f.first().reset_index()
        tt = sf.exptime.values; yy = sf.s_f.values * tt
        p = np.polyfit(tt, yy, 1)
        print(f"\n{unit} {filt}: frame-scale fit s_f*t = {p[0]:.4f} t + {p[1]:+.3f} s  -> dt ~ {p[1]/p[0]:+.3f} s")
        piv = R[R.aperture == 10].pivot_table(index="peak_lo", columns="exptime", values="dmag").round(1)
        print("response (10\", new bkg) as mmag loss [>0: measured fainter] vs expected star peak above sky [e-] (rows) and exposure [s]:")
        print(piv.to_string())
        pt = R[R.aperture == "10total"].pivot_table(index="peak_lo", columns="exptime", values="dmag").round(1)
        print("bright end: loss [mmag] vs expected TOTAL level at the star peak (sky + star) [ADU]:")
        print(pt.to_string())
    A = pd.concat(allS, ignore_index=True); R = pd.concat(allR, ignore_index=True)
    A.to_csv(out / "lin_stars.csv", index=False); R.to_csv(out / "lin_response.csv", index=False)
    # figures
    groups = sorted(A.groupby(["unit", "filter"]).groups.keys())
    n = len(groups)
    fig, ax = plt.subplots(1, max(n, 1), figsize=(5.2 * max(n, 1), 4.6), squeeze=False)
    cmap = plt.get_cmap("viridis")
    for a_, (unit, filt) in zip(ax[0], groups):
        Rg = R[(R.unit == unit) & (R["filter"] == filt) & (R.aperture == 10)]
        ts = sorted(Rg.exptime.unique())
        for i, t in enumerate(ts):
            r = Rg[Rg.exptime == t]
            a_.errorbar(r.peak_med, r.dmag, r.err, marker="o", ms=3, lw=1, capsize=0, color=cmap(i / max(len(ts) - 1, 1)), label=f"{t:g} s")
        a_.axhline(0, color="k", lw=0.7); a_.set_xscale("log"); a_.grid(alpha=0.3); a_.set_ylim(-60, 120)
        a_.axvspan(LIN_LO * 0.25, LIN_HI * 0.25, color="0.92", zorder=0)
        a_.set(title=f"{unit} {filt}", xlabel="expected star peak above sky [e⁻]")
        a_.legend(fontsize=7, ncol=2)
    ax[0, 0].set_ylabel("flux deficit vs linear-regime rate [mmag]\n(>0: measured too faint)")
    fig.suptitle("L1  Response vs star peak level, by exposure time (10″, new background; grey = reference range)", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "L1_response.png", dpi=125)
    print("saved", out)
