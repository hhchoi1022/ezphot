#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260925_bkg_new_vs_old.py — ezphot photometry with the new background vs the original, per-star ZP vs magnitude.

  old (V0) : photometry_sex() default = SExtractor internal background (mode estimator)       [pilot/V0]
  new (V3) : MaskGenerator.mask_sources(5 sigma, x3) -> BackgroundGenerator.estimate_with_photutils(
             bkg_estimator='mean', sigma=5 (default), box 64, filter 3) -> photometry_sex(target_bkg=bk)   [pilot/V3]
Each frame is copied to the scratchpad, but ezphot derives savedir from config + header, so the temporary
background-subtracted image is still written (and removed) in the data directory; set img.savedir to avoid that.
Per star (Gaia-XP match < 1", isolated 10" in the V0 detection list, FLAGS 0 — no CLASS_STAR/shape cuts):
  ZP_i = m_syn(corrected XP, manufacturer curve) - MAG_APER_3 (10")
Output: new_vs_old_stars.csv, A13_zp_vs_mag.png

  MPLCONFIGDIR=/tmp/claude-10000 OMP_NUM_THREADS=1 python ezphot/analysis/260925_bkg_new_vs_old.py phot 16
  MPLCONFIGDIR=/tmp/claude-10000 python ezphot/analysis/260925_bkg_new_vs_old.py plot
"""
import importlib
import shutil
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

warnings.filterwarnings("ignore")
import portalocker                                   # noqa: E402  NFS on wfm: flock fails (EBADF) -> no-op locks


class _NoLock:
    def __init__(self, *a, **k): pass
    def __enter__(self): return None
    def __exit__(self, *a): return False


portalocker.Lock = _NoLock
portalocker.lock = lambda *a, **k: None
portalocker.unlock = lambda *a, **k: None

sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
LOG = Path("/home/hhchoi1022/ezphot/log/20260924_stepA")
PILOT = LOG / "pilot"
WORK = Path("/tmp/claude-10000/-lyman-data1-factory-hhchoi-code-ezphot/b1308704-c229-4e49-b19d-bd7bc334e4b7/scratchpad/bkgfix/work")
KEEP_PREFIX = ("MAG_", "MAGERR_", "SNR_", "FLUX_MAX", "BACKGROUND")
KEEP_COLS = ("X_WORLD", "Y_WORLD", "ALPHA_J2000", "DELTA_J2000", "X_IMAGE", "Y_IMAGE", "FLAGS", "CLASS_STAR",
             "FWHM_WORLD", "FWHM_IMAGE", "ELONGATION")
PIX = 0.505


def photometer_new(path):
    name = Path(path).name
    outp = PILOT / "V3" / (name[:-5] + "_cat.fits")
    if outp.exists():
        return name, "cached"
    wd = WORK / name[:-5]
    wd.mkdir(parents=True, exist_ok=True)
    local = wd / name
    try:
        shutil.copy2(path, local)
        from ezphot.imageobjects import ScienceImage, Mask
        from ezphot.methods import MaskGenerator, BackgroundGenerator
        img = ScienceImage(path=str(local), telinfo=None, load=True)
        srcmask = MaskGenerator().mask_sources(img, target_mask=Mask(wd / "srcmask.fits", masktype="source", load=False),
                                               sigma=5.0, mask_radius_factor=3, save=False, verbose=False, visualize=False)
        bk, _ = BackgroundGenerator().estimate_with_photutils(img, target_mask=srcmask, box_size=64, filter_size=3,
                                                             bkg_estimator="mean", save=False, verbose=False,
                                                             visualize=False)
        cat = img.photometry_sex(target_bkg=bk, save=False, verbose=False, visualize=False)
        d = cat.data
        keep = [k for k in d.colnames if k.startswith(KEEP_PREFIX) or k in KEEP_COLS]
        phdu = fits.PrimaryHDU(header=fits.Header(img.header))
        fits.HDUList([phdu, fits.BinTableHDU(Table(d[keep]), name="LDAC_OBJECTS")]).writeto(outp, overwrite=True)
        return name, "ok"
    except Exception as e:                                           # noqa: BLE001
        return name, f"error: {str(e)[:300]}"
    finally:
        shutil.rmtree(wd, ignore_errors=True)


_G = {}


def _init():
    warnings.filterwarnings("ignore")
    TR = importlib.import_module("260924_stepA_triage")
    TR.AN._init()
    _G["TR"] = TR
    _G["curves"] = TR.TC.build_totals()[("manufacturer", "nobeam")]


def stars(args):
    path, dataset, obj, filt, exptime = args
    TR = _G["TR"]
    from astropy.coordinates import SkyCoord, match_coordinates_sky, search_around_sky
    import astropy.units as u
    from scipy.spatial import cKDTree
    name = Path(path).name
    c0 = PILOT / "V0" / (name[:-5] + "_cat.fits"); c3 = PILOT / "V3" / (name[:-5] + "_cat.fits")
    if not c3.exists():
        return None
    o = Table.read(c0, hdu="LDAC_OBJECTS"); nw = Table.read(c3, hdu="LDAC_OBJECTS")
    n = len(o)
    allc = SkyCoord(np.asarray(o["ALPHA_J2000"]), np.asarray(o["DELTA_J2000"]), unit="deg")
    i1, _, s2, _ = search_around_sky(allc, allc, 10.0 * u.arcsec)
    nbr = np.bincount(i1[s2.arcsec > 0.5], minlength=n)
    key = f"CAL_{obj}" if dataset == "CALSPEC" else f"SPSS_{TR.AN.norm(obj)}"
    ref = TR.load_cache(key)
    idx, d2d, _ = match_coordinates_sky(allc, SkyCoord(ref["ra"], ref["dec"], unit="deg"))
    m_old = np.asarray(o["MAG_APER_3"], float)
    snr = 1.0857 / np.clip(np.asarray(o["MAGERR_APER_3"], float), 1e-4, None)
    sel = (np.asarray(d2d.arcsec) < 1.0) & (nbr == 0) & (np.asarray(o["FLAGS"]) == 0) & (m_old < 90) \
        & (np.asarray(o["FLUX_MAX"], float) < 40000) & (snr > 5)
    ii = np.where(sel)[0]
    # same stars in the new catalogue (positions agree to << 1 px)
    dd, jj = cKDTree(np.c_[np.asarray(nw["X_IMAGE"]), np.asarray(nw["Y_IMAGE"])]).query(
        np.c_[np.asarray(o["X_IMAGE"])[ii], np.asarray(o["Y_IMAGE"])[ii]])
    m_new = np.where(dd < 0.5, np.asarray(nw["MAG_APER_3"], float)[jj], np.nan)
    fw, ft = _G["curves"][filt]
    syn = TR.R.syn_mag(TR.XP_WAVE, ref["cor"][idx[ii]].astype(float), fw, ft, 0.0)
    return pd.DataFrame(dict(name=name, dataset=dataset, filter=filt, exptime=exptime, object=obj,
                             m_syn=syn, bprp=ref["bprp"][idx[ii]], snr_old=snr[ii],
                             zp_old=syn - m_old[ii], zp_new=np.where(m_new < 90, syn - m_new, np.nan)))


def binned(x, y, edges, min_n=5):
    xc, med, err, nn = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x < b) & np.isfinite(y)
        if m.sum() >= min_n:
            v = y[m]
            xc.append(np.median(x[m])); med.append(np.median(v))
            err.append(1.2533 * 1.4826 * np.median(np.abs(v - np.median(v))) / np.sqrt(m.sum())); nn.append(m.sum())
    return np.array(xc), np.array(med), np.array(err), np.array(nn)


if __name__ == "__main__":
    stage = sys.argv[1]
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    fr = pd.read_csv(PILOT / "pilot_frames.csv")
    if stage == "phot":
        (PILOT / "V3").mkdir(exist_ok=True); WORK.mkdir(parents=True, exist_ok=True)
        with Pool(nproc) as pool:
            for i, (nm, st) in enumerate(pool.imap_unordered(photometer_new, list(fr.path)), 1):
                print(f"{i}/{len(fr)} {nm[:55]} {st}", flush=True)
        sys.exit(0)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    jobs = [(r.path, r.dataset, r.object, r["filter"], r.exptime) for _, r in fr.iterrows()]
    with Pool(nproc, initializer=_init) as pool:
        S = pd.concat([d for d in pool.map(stars, jobs) if d is not None], ignore_index=True)
    S["grp"] = S.dataset + " " + S.exptime.astype(int).astype(str) + "s"
    # per-frame references: bright ZP (SNR >= 100, per method) and the magnitude where the old SNR is ~20
    ref = []
    for nm, g in S.groupby("name"):
        br = g[g.snr_old >= 100]
        m20 = g[(g.snr_old >= 18) & (g.snr_old < 22)].m_syn.median()
        ref.append(dict(name=nm, zpb_old=br.zp_old.median(), zpb_new=br.zp_new.median(), n_bright=len(br), m20=m20))
    S = S.merge(pd.DataFrame(ref), on="name")
    S["dzp_old"] = S.zp_old - S.zpb_old
    S["dzp_new"] = S.zp_new - S.zpb_new
    S["dm20"] = S.m_syn - S.m20
    S.to_csv(LOG / "new_vs_old_stars.csv", index=False)
    S = S[(S.n_bright >= 5) & np.isfinite(S.m20)]
    grps = ["CALSPEC 10s", "CALSPEC 20s", "CALSPEC 30s", "SPSS 30s", "SPSS 100s"]
    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "lines.linewidth": 1.0})
    fig, ax = plt.subplots(2, 5, figsize=(21, 8.2))
    edges = np.arange(-6.0, 2.01, 0.25)
    summary = []
    for k, grp in enumerate(grps):
        G = S[S.grp == grp]
        if G.empty:
            continue
        # top: the best-populated frame of the group, absolute ZP vs magnitude
        nm = G.groupby("name").size().idxmax()
        g = G[G.name == nm]
        a = ax[0, k]
        a.scatter(g.m_syn, g.zp_old, s=5, color="tab:orange", alpha=0.35, lw=0, label="original (SExtractor mode)")
        a.scatter(g.m_syn, g.zp_new, s=5, color="tab:blue", alpha=0.35, lw=0, label="new (mean, 5σ)")
        e = np.arange(np.floor(g.m_syn.min()), np.ceil(g.m_syn.max()) + 0.01, 0.25)
        for col, c in [("zp_old", "tab:orange"), ("zp_new", "tab:blue")]:
            xc, md, er, _ = binned(g.m_syn.to_numpy(), g[col].to_numpy(), e)
            a.errorbar(xc, md, er, color=c, marker="o", ms=3, lw=1.2, capsize=0)
        a.axhline(g.zpb_old.iloc[0], color="tab:orange", ls="--", lw=0.8)
        a.axhline(g.zpb_new.iloc[0], color="tab:blue", ls="--", lw=0.8)
        for s_, ls in [(20, ":"), (50, "-.")]:
            mm = g[(g.snr_old >= s_ * 0.9) & (g.snr_old < s_ * 1.1)].m_syn.median()
            if np.isfinite(mm):
                a.axvline(mm, color="0.4", ls=ls, lw=0.8)
                a.text(mm, g.zpb_new.iloc[0] + 0.23, f"SNR {s_}", fontsize=7, ha="center", color="0.3")
        a.set(ylim=(g.zpb_new.iloc[0] - 0.25, g.zpb_new.iloc[0] + 0.25), xlabel=f"corrected-XP synthetic mag ({g['filter'].iloc[0]})",
              title=f"{grp}: {g.object.iloc[0]} {g['filter'].iloc[0]} ({len(g)} stars)")
        if k == 0:
            a.set_ylabel("ZP = m_syn − MAG_APER_3 (10″)")
            a.legend(fontsize=7, loc="lower left", markerscale=2)
        a.grid(alpha=0.3)
        # bottom: all frames of the group, ZP relative to the bright-star ZP vs magnitude relative to SNR-20 level
        a = ax[1, k]
        for col, c, lab in [("dzp_old", "tab:orange", "original"), ("dzp_new", "tab:blue", "new")]:
            xc, md, er, nn = binned(G.dm20.to_numpy(), 1e3 * G[col].to_numpy(), edges, min_n=15)
            a.errorbar(xc, md, er, color=c, marker="o", ms=3, lw=1.2, capsize=0, label=lab)
            # 0.5-mag bins centred where the median catalogue SNR is closest to 20 / 50 / 110
            cen = np.arange(-5.875, 0.9, 0.25)
            msnr = np.array([G.loc[(G.dm20 >= c_ - 0.125) & (G.dm20 < c_ + 0.125), "snr_old"].median() for c_ in cen])
            for tgt in (20, 50, 110):
                x0 = cen[np.nanargmin(np.abs(msnr - tgt))]
                m = (G.dm20 >= x0 - 0.25) & (G.dm20 < x0 + 0.25)
                summary.append(dict(grp=grp, method=lab, bin=f"snr{tgt}", snr_med=float(G.loc[m, "snr_old"].median()),
                                    dzp_mmag=1e3 * np.nanmedian(G.loc[m, col]), n=int(m.sum())))
        a.axhline(0, color="k", lw=0.7); a.axvline(0, color="0.4", ls=":", lw=0.8)
        a.set(ylim=(-80, 180), xlim=(-5.5, 1.5), xlabel="m_syn − m(SNR=20) of the frame [mag]",
              title=f"{grp}: {G.name.nunique()} frames, {len(G)} stars")
        if k == 0:
            a.set_ylabel("ZP − ZP(bright, SNR≥100) [mmag]  (>0: star measured too bright)")
            a.legend(fontsize=7.5)
        a.grid(alpha=0.3)
    fig.suptitle("A13  Per-star ZP vs magnitude: original ezphot photometry (SExtractor mode background) vs new "
                 "(masked per-mesh mean, 5σ clip) — same stars (Gaia-XP matched, isolated, no shape cuts), 10″ aperture",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(LOG / "A13_zp_vs_mag.png", dpi=125)
    T = pd.DataFrame(summary).pivot_table(index=["grp", "bin"], columns="method", values=["dzp_mmag", "snr_med"]).round(1)
    pd.set_option("display.width", 200)
    print(T.to_string())
    print("\nbright ZP new - old [mmag] per group:",
          (1e3 * (S.groupby("grp").zpb_new.median() - S.groupby("grp").zpb_old.median())).round(1).to_dict())
