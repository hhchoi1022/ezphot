#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260925_exposure_ratio_test.py — XP-free test of the faint-star residual: flux ratio of the same stars in two exposure
times (same field, unit and filter; the short exposure is CALSPEC 2024, the long one SPSS 2026).

Per frame: forced 10" (and 5") aperture sums at Gaia positions (projected with the frame WCS, shifted by the median
detection offset), with
  new : per-mesh 5-sigma clipped mean background (mesh_bkg 'mean5', segmentation + 4 px mask)
  old : sep/SExtractor mode background (current catalogues)
Stars: isolated (no detection within 10" other than itself), in the frame, Gaia G used only to sort stars.
Per side, fluxes are put on a common scale per frame (bright-star median) and median-combined over frames; per star
R = F_long / F_short, dR = 2.5 log10(R / R_ref) [mmag]; reference = stars with short-exposure SNR 150-400 (the
brightest stars are saturated / non-linear in the 100 s frames: raw peaks up to ~70k ADU), binned on Gaia G.
SNR here is background-only (sep err = globalrms); ratio SNR 20 ~ catalogue SNR 15.
  image-side loss that is larger in the short (2024) exposure -> dR > 0 for faint stars (rises toward faint)
  comparison-side (XP) cause only                              -> dR flat
  old background pedestal                                       -> dR < 0 for faint stars (short exposure gains more)
Output: exposure_ratio_stars.csv, exposure_ratio_bins.csv, A17_exposure_ratio.png
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
SCI = Path("/home/hhchoi1022/ezphot/data/scidata/7DT/7DT_C361K_HIGH_1x1")
CAL_META = Path("/home/hhchoi1022/ezphot/log/20260914_calspec_ezphot/photometry_meta.csv")
PIX = 0.505
R5, R10 = 2.5 / PIX, 5.0 / PIX
NMAX = 8
_G = {}


def _init():
    warnings.filterwarnings("ignore")
    _G["TR"] = importlib.import_module("260924_stepA_triage")
    _G["SW"] = importlib.import_module("260925_bkg_recipe_sweep")


def frame_list():
    F = pd.read_csv(LOG / "stepA_frames.csv")
    F["objn"] = F.object.str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    calpath = {Path(f).name: f for f in pd.read_csv(CAL_META).file}
    jobs = []
    for (objn, unit, filt), g in F.groupby(["objn", "unit", "filter"]):
        exps = sorted(set(g.exptime))
        if len(exps) < 2:
            continue
        pair = f"{objn} {unit} {filt}"
        for side, e in zip(("short", "long"), (exps[0], exps[-1])):
            h = g[g.exptime == e].sort_values("mjd")
            h = h.iloc[np.unique(np.linspace(0, len(h) - 1, min(NMAX, len(h))).round().astype(int))]
            for _, r in h.iterrows():
                p = calpath.get(r.file) if r.dataset == "CALSPEC" else str(SCI / r.object / r.unit / r["filter"] / r.file)
                if p and Path(p).exists():
                    key = f"CAL_{'FEIGE110' if objn == 'FEIGE110' else 'WD2039-682'}"
                    jobs.append(dict(pair=pair, side=side, exptime=e, dataset=r.dataset, path=p, key=key, filter=filt))
    return jobs


def run(job):
    TR, SW = _G["TR"], _G["SW"]
    try:
        from astropy.wcs import WCS
        from scipy.ndimage import binary_dilation
        from scipy.spatial import cKDTree
        hdr = fits.getheader(job["path"])
        data = np.ascontiguousarray(fits.getdata(job["path"]).astype(np.float32))
        bad = ~np.isfinite(data); data[bad] = 0.0
        ny, nx = data.shape
        b0 = sep.Background(data, mask=bad, bw=64, bh=64, fw=3, fh=3)
        sep.set_extract_pixstack(10_000_000); sep.set_sub_object_limit(4096)
        objs, seg = sep.extract(data - b0.back(), 1.5, err=b0.globalrms, minarea=12, segmentation_map=True, mask=bad)
        back_new = SW.mesh_bkg(data, binary_dilation(seg > 0, iterations=4) | bad, "mean5")
        ref = TR.load_cache(job["key"])
        x, y = WCS(hdr).all_world2pix(np.asarray(ref["ra"], float), np.asarray(ref["dec"], float), 0)
        # astrometric refinement from bright matches
        td = cKDTree(np.c_[objs["x"], objs["y"]])
        dd, ii = td.query(np.c_[x, y])
        g = np.asarray(ref["g"], float)
        m = (dd < 4.0) & (g < 15.5)
        if m.sum() >= 10:
            x = x + np.median(objs["x"][ii[m]] - x[m]); y = y + np.median(objs["y"][ii[m]] - y[m])
        inside = (x > 60) & (x < nx - 60) & (y > 60) & (y < ny - 60)
        nnb = np.array([len(td.query_ball_point([a, b], 10.0 / PIX)) for a, b in zip(x, y)])
        dself, _ = td.query(np.c_[x, y])
        iso = (nnb == 0) | ((nnb == 1) & (dself < 1.5 / PIX))
        k = np.where(inside)[0]
        out = dict(sid=np.asarray(ref["sid"])[k], g=g[k], bprp=np.asarray(ref["bprp"], float)[k], iso=iso[k],
                   detected=(dself[k] < 1.5 / PIX))
        for tag, back in (("new", back_new), ("old", b0.back())):
            sub = data - back
            f10, e10, _ = sep.sum_circle(sub, x[k], y[k], R10, err=float(b0.globalrms), subpix=5)
            f5, _, _ = sep.sum_circle(sub, x[k], y[k], R5, subpix=5)
            out[f"f10_{tag}"] = f10; out[f"f5_{tag}"] = f5
            if tag == "new":
                out["e10"] = e10
        D = pd.DataFrame(out)
        for c in ("pair", "side", "exptime", "dataset", "filter"):
            D[c] = job[c]
        D["frame"] = Path(job["path"]).name
        D["sky_rms"] = float(b0.globalrms)
        return D
    except Exception as e:                                           # noqa: BLE001
        return pd.DataFrame([dict(pair=job["pair"], frame=Path(job["path"]).name, error=str(e)[:300])])


def combine(D, col):
    """per side: common flux scale per frame (bright stars), median over frames -> one flux per star."""
    out = {}
    for side, g in D.groupby("side"):
        P = g.pivot_table(index="sid", columns="frame", values=col, aggfunc="first")
        ref = P.median(axis=1)
        bright = ref.sort_values(ascending=False).index[20:80]         # skip the brightest (saturation)
        scale = (P.loc[bright].div(ref.loc[bright], axis=0)).median(axis=0)
        out[side] = P.div(scale, axis=1).median(axis=1)
        out[side + "_nfr"] = P.notna().sum(axis=1)
    return pd.DataFrame(out)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if len(sys.argv) > 2 and sys.argv[2] == "reuse":
        A = pd.read_csv(LOG / "exposure_ratio_frames_stars.csv")
    else:
        jobs = frame_list()
        print(len(jobs), "frames in", len({j["pair"] for j in jobs}), "pairs", flush=True)
        with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 16, initializer=_init) as pool:
            parts = pool.map(run, jobs)
        A = pd.concat(parts, ignore_index=True)
    if "error" in A.columns and A["error"].notna().any():
        print("errors:", A["error"].dropna().unique()[:3])
        A = A[A["error"].isna()]
    A.to_csv(LOG / "exposure_ratio_frames_stars.csv", index=False)
    rows, bins = [], []
    for pair, D in A.groupby("pair"):
        meta = D.groupby("sid")[["g", "bprp"]].first()
        iso = D.groupby("sid").iso.all()
        snr_s = D[D.side == "short"].assign(s=lambda d: d.f10_new / d.e10).groupby("sid").s.median()
        C = {tag: combine(D, f"f10_{tag}") for tag in ("new", "old")}
        S = meta.join(iso).join(snr_s.rename("snr_short"))
        for tag in ("new", "old"):
            S[f"R_{tag}"] = C[tag]["long"] / C[tag]["short"]
        S["nfr_short"] = C["new"]["short_nfr"]; S["nfr_long"] = C["new"]["long_nfr"]
        S = S[S.iso & (S.R_new > 0) & (S.R_old > 0) & (S.nfr_short >= 2) & (S.nfr_long >= 2) & np.isfinite(S.snr_short)]
        br = S[(S.snr_short >= 150) & (S.snr_short < 400)]
        for tag in ("new", "old"):
            S[f"dR_{tag}"] = 2.5 * np.log10(S[f"R_{tag}"] / np.median(br[f"R_{tag}"]))
        # colour term from the bright half (SNR_short > 50), removed from all
        cb = S[(S.snr_short > 50) & (S.snr_short < 400)]
        for tag in ("new", "old"):
            sl = np.polyfit(cb.bprp - cb.bprp.median(), cb[f"dR_{tag}"], 1)[0] if len(cb) > 20 else 0.0
            S[f"dRc_{tag}"] = S[f"dR_{tag}"] - sl * (S.bprp - cb.bprp.median())
            S[f"slope_{tag}"] = sl
        e = np.arange(9, 18.01, 0.5); cc = pd.cut(S.g, e)
        med = S.groupby(cc, observed=True).snr_short.median()
        S["dg"] = S.g - np.array([iv.mid for iv in med.index])[np.argmin(np.abs(med.values - 20))]
        S["pair"] = pair
        S["exp_short"] = D[D.side == "short"].exptime.iloc[0]; S["exp_long"] = D[D.side == "long"].exptime.iloc[0]
        rows.append(S.reset_index())
        for lo, hi in [(15, 20), (20, 30), (30, 50), (50, 80), (80, 150), (150, 1e9)]:
            m = (S.snr_short >= lo) & (S.snr_short < hi)
            if m.sum() >= 8:
                v = S.loc[m]
                bs = [np.median(np.random.default_rng(i).choice(v.dRc_new, len(v))) for i in range(200)]
                bins.append(dict(pair=pair, snr_lo=lo, snr_hi=hi, n=int(m.sum()), snr_med=v.snr_short.median(),
                                 G_med=v.g.median(), new=1e3 * v.dRc_new.median(), new_err=1e3 * np.std(bs),
                                 old=1e3 * v.dRc_old.median()))
    S = pd.concat(rows, ignore_index=True); B = pd.DataFrame(bins)
    S.to_csv(LOG / "exposure_ratio_stars.csv", index=False); B.to_csv(LOG / "exposure_ratio_bins.csv", index=False)
    pd.set_option("display.width", 220)
    print(B.round(1).to_string(index=False))
    # pooled over pairs by field
    B["field"] = B.pair.str.split().str[0]
    S["field"] = S.pair.str.split().str[0]
    print("\npooled per field (median over stars, colour-corrected) [mmag]:")
    for fld, G in S.groupby("field"):
        out = []
        for lo, hi in [(15, 25), (25, 40), (40, 80), (80, 150)]:
            m = (G.snr_short >= lo) & (G.snr_short < hi)
            if m.sum() >= 10:
                v = G.loc[m]
                bs = [np.median(np.random.default_rng(i).choice(v.dRc_new, len(v))) for i in range(300)]
                out.append(f"SNR {lo}-{hi}: new {1e3*v.dRc_new.median():+.1f}±{1e3*np.std(bs):.1f}, old {1e3*v.dRc_old.median():+.1f} (n={m.sum()})")
        print(" ", fld, " | ".join(out))
    # truth-side (Gaia G) binning, pooled by field and band
    S["band"] = np.where(S.pair.str.contains("m400|m425|m450"), "blue", "red")
    GB = []
    for (fld, band), H in S.groupby(["field", "band"]):
        for lo, hi in [(-0.5, 0.25), (-1.25, -0.5), (-2.0, -1.25), (-3.0, -2.0), (-5.0, -3.0)]:
            m = (H.dg >= lo) & (H.dg < hi)
            if m.sum() >= 8:
                v = H.loc[m]
                bs = [np.median(np.random.default_rng(i).choice(v.dRc_new, len(v))) for i in range(300)]
                GB.append(dict(field=fld, band=band, dg_lo=lo, dg_hi=hi, n=int(m.sum()), snr_short=v.snr_short.median(),
                               new=1e3 * v.dRc_new.median(), err=1e3 * np.std(bs), new_uncorr=1e3 * v.dR_new.median(),
                               old=1e3 * v.dRc_old.median()))
    GB = pd.DataFrame(GB); GB.to_csv(LOG / "exposure_ratio_gbins.csv", index=False)
    print("\nGaia-G binning (relative to G at short SNR~20), clean reference:")
    print(GB.round(1).to_string(index=False))
    # figure
    pairs = sorted(S.pair.unique())
    fig, ax = plt.subplots(2, 5, figsize=(21, 8), sharey=True)
    for a_, pair in zip(ax.flat, pairs):
        G = S[S.pair == pair]
        a_.scatter(G.snr_short, 1e3 * G.dRc_new, s=5, color="tab:blue", alpha=0.25, lw=0)
        for col, c, lab in [("dRc_new", "tab:blue", "new background"), ("dRc_old", "tab:orange", "original (mode)")]:
            xs_, ys_, es_ = [], [], []
            for lo, hi in [(-0.5, 0.25), (-1.25, -0.5), (-2.0, -1.25), (-3.0, -2.0), (-5.0, -3.0)]:
                m = (G.dg >= lo) & (G.dg < hi)
                if m.sum() >= 8:
                    xs_.append(G.loc[m, "snr_short"].median()); ys_.append(1e3 * G.loc[m, col].median())
                    es_.append(1e3 * 1.2533 * G.loc[m, col].std() / np.sqrt(m.sum()))
            a_.errorbar(xs_, ys_, es_, color=c, marker="o", ms=4, capsize=2, label=lab)
        a_.axhline(0, color="k", lw=0.7); a_.set_xscale("log"); a_.grid(alpha=0.3); a_.set_ylim(-150, 150)
        e = G.iloc[0]
        a_.set_title(f"{pair}\n{int(e.exp_long)} s (2026) / {int(e.exp_short)} s (2024), {len(G)} stars", fontsize=8.5)
        a_.set_xlim(3, 3000)
    for a_ in ax[1]:
        a_.set_xlabel("single-frame SNR, short exposure (10″; background-only)")
    for a_ in ax[:, 0]:
        a_.set_ylabel("2.5 log(F_long/F_short) − bright [mmag]\n(>0: short exposure lost more)")
    ax[0, 0].legend(fontsize=7.5)
    fig.suptitle("A17  XP-free test: flux ratio of the same stars in two exposures (forced 10″ at Gaia positions, isolated, "
                 "colour term removed; points = Gaia-G bins, reference = SNR 150–400 stars)", fontsize=10)
    fig.tight_layout(); fig.savefig(LOG / "A17_exposure_ratio.png", dpi=125)
    print("saved")
