#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260925_aper15_check.py — is the faint-star deficit a PSF-width (FWHM varying over the field) effect?

Same pilot catalogues as A13: original (pilot/V0, SExtractor mode background) and new (pilot/V3, per-mesh 5-sigma mean).
Per star: ZP = m_syn(corrected XP) - MAG_APER_3 (10") and - MAG_APER_4 (15" diameter), relative to each
(method, aperture) bright ZP (SNR10 >= 100).  Star set: Gaia-XP < 1", isolated 15" (no V0 detection), FLAGS 0,
FLUX_MAX < 40000.  Local FWHM = median FWHM_IMAGE of the 15 nearest bright stars (SNR10 >= 50) in the same frame.
  aperture-loss (PSF width) effect -> deficit shrinks at 15" and follows local FWHM
  background-like (additive) effect -> deficit grows ~x2.25 (area) at 15"
Output: aper15_stars.csv, A16_aper15.png
"""
import importlib
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.table import Table

warnings.filterwarnings("ignore")
sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
LOG = Path("/home/hhchoi1022/ezphot/log/20260924_stepA")
PILOT = LOG / "pilot"
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
    i1, _, s2, _ = search_around_sky(allc, allc, 15.0 * u.arcsec)
    nbr = np.bincount(i1[s2.arcsec > 0.5], minlength=n)
    key = f"CAL_{obj}" if dataset == "CALSPEC" else f"SPSS_{TR.AN.norm(obj)}"
    ref = TR.load_cache(key)
    idx, d2d, _ = match_coordinates_sky(allc, SkyCoord(ref["ra"], ref["dec"], unit="deg"))
    m10o = np.asarray(o["MAG_APER_3"], float); m15o = np.asarray(o["MAG_APER_4"], float)
    snr = 1.0857 / np.clip(np.asarray(o["MAGERR_APER_3"], float), 1e-4, None)
    fmax = np.asarray(o["FLUX_MAX"], float); flags = np.asarray(o["FLAGS"])
    x = np.asarray(o["X_IMAGE"], float); y = np.asarray(o["Y_IMAGE"], float)
    fwhm = np.asarray(o["FWHM_IMAGE"], float)
    sel = (np.asarray(d2d.arcsec) < 1.0) & (nbr == 0) & (flags == 0) & (m10o < 90) & (m15o < 90) & (fmax < 40000) & (snr > 5)
    ii = np.where(sel)[0]
    # local FWHM from bright, unsaturated, unflagged stars
    br = (snr >= 50) & (fmax < 40000) & (flags == 0) & (fwhm > 0.5) & (fwhm < 30)
    tb = cKDTree(np.c_[x[br], y[br]])
    _, nn = tb.query(np.c_[x[ii], y[ii]], k=min(15, int(br.sum())))
    loc_fwhm = np.median(fwhm[br][nn], axis=1) * 0.505
    dd, jj = cKDTree(np.c_[np.asarray(nw["X_IMAGE"]), np.asarray(nw["Y_IMAGE"])]).query(np.c_[x[ii], y[ii]])
    ok3 = dd < 0.5
    m10n = np.where(ok3, np.asarray(nw["MAG_APER_3"], float)[jj], np.nan)
    m15n = np.where(ok3, np.asarray(nw["MAG_APER_4"], float)[jj], np.nan)
    fw, ft = _G["curves"][filt]
    syn = TR.R.syn_mag(TR.XP_WAVE, ref["cor"][idx[ii]].astype(float), fw, ft, 0.0)
    nx = float(np.nanmax(x)); ny = float(np.nanmax(y))
    rn = np.hypot((x[ii] - nx / 2) / (nx / 2), (y[ii] - ny / 2) / (ny / 2)) / np.sqrt(2)
    def z(m):
        return np.where(m < 90, syn - m, np.nan)
    return pd.DataFrame(dict(name=name, dataset=dataset, filter=filt, exptime=exptime, m_syn=syn, snr=snr[ii],
                             loc_fwhm=loc_fwhm, own_fwhm=fwhm[ii] * 0.505, rnorm=rn,
                             zp_old10=z(m10o[ii]), zp_old15=z(m15o[ii]), zp_new10=z(m10n), zp_new15=z(m15n)))


COMBOS = ["old10", "old15", "new10", "new15"]


def nearest_bins(G, targets=(20, 50, 110)):
    cen = np.arange(-5.875, 0.9, 0.25)
    msnr = np.array([G.loc[(G.dm20 >= c - 0.125) & (G.dm20 < c + 0.125), "snr"].median() for c in cen])
    out = {}
    for t in targets:
        x0 = cen[np.nanargmin(np.abs(msnr - t))]
        out[t] = (G.dm20 >= x0 - 0.25) & (G.dm20 < x0 + 0.25)
    return out


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fr = pd.read_csv(PILOT / "pilot_frames.csv")
    jobs = [(r.path, r.dataset, r.object, r["filter"], r.exptime) for _, r in fr.iterrows()]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 16, initializer=_init) as pool:
        S = pd.concat([d for d in pool.map(stars, jobs) if d is not None], ignore_index=True)
    S["grp"] = S.dataset + " " + S.exptime.astype(int).astype(str) + "s"
    refs = []
    for nm, g in S.groupby("name"):
        b = g[g.snr >= 100]
        r = dict(name=nm, n_bright=len(b), m20=g[(g.snr >= 18) & (g.snr < 22)].m_syn.median())
        for c in COMBOS:
            r[f"zb_{c}"] = b[f"zp_{c}"].median()
        # per-frame FWHM tercile edges (local FWHM)
        q1, q2 = np.nanpercentile(g.loc_fwhm, [33.3, 66.7])
        r["fw_q1"], r["fw_q2"] = q1, q2
        refs.append(r)
    S = S.merge(pd.DataFrame(refs), on="name")
    for c in COMBOS:
        S[f"d_{c}"] = S[f"zp_{c}"] - S[f"zb_{c}"]
    S["dm20"] = S.m_syn - S.m20
    S["fwbin"] = np.where(S.loc_fwhm <= S.fw_q1, "narrow", np.where(S.loc_fwhm >= S.fw_q2, "broad", "mid"))
    S.to_csv(LOG / "aper15_stars.csv", index=False)
    S = S[(S.n_bright >= 5) & np.isfinite(S.m20)]
    grps = ["CALSPEC 10s", "CALSPEC 20s", "CALSPEC 30s", "SPSS 30s", "SPSS 100s"]
    pd.set_option("display.width", 230)
    # --- tables ---------------------------------------------------------------------------------------------------
    rows = []
    for grp in grps:
        G = S[S.grp == grp]
        if G.empty:
            continue
        nb = nearest_bins(G)
        for t, m in nb.items():
            r = dict(grp=grp, snr=t, snr_med=round(G.loc[m, "snr"].median()), n=int(m.sum()))
            for c in COMBOS:
                r[c] = round(1e3 * G.loc[m, f"d_{c}"].median(), 1)
            for fb in ("narrow", "broad"):
                mm = m & (G.fwbin == fb)
                r[f"new10_{fb}"] = round(1e3 * G.loc[mm, "d_new10"].median(), 1)
                r[f"new15_{fb}"] = round(1e3 * G.loc[mm, "d_new15"].median(), 1)
            rows.append(r)
    T = pd.DataFrame(rows)
    print("frames:", S.groupby("grp").name.nunique().to_dict())
    print("\nZP - bright ZP [mmag] by aperture/method at SNR ~20/50/110 (same stars; 15\"-isolated):")
    print(T[["grp", "snr", "snr_med", "n"] + COMBOS].to_string(index=False))
    print("\nnew background, split by local FWHM tercile (per frame):")
    print(T[["grp", "snr", "new10_narrow", "new10_broad", "new15_narrow", "new15_broad"]].to_string(index=False))
    # FWHM variation and bright-star aperture loss
    B = S[S.snr >= 100].copy()
    B["ap_loss"] = 1e3 * (B.zp_new10 - B.zp_new15)          # >0: 10" misses light relative to 15"
    fwr = S.groupby("name").loc_fwhm.agg(lambda v: np.nanpercentile(v, 90) - np.nanpercentile(v, 10))
    print("\nlocal-FWHM spread p10-p90 per frame [arcsec]: median %.2f (range %.2f-%.2f); median local FWHM %.2f\""
          % (fwr.median(), fwr.min(), fwr.max(), S.loc_fwhm.median()))
    print("bright-star 10\"-vs-15\" aperture loss [mmag] by FWHM tercile:",
          B.groupby("fwbin").ap_loss.median().round(1).to_dict())
    # --- figure ---------------------------------------------------------------------------------------------------
    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "lines.linewidth": 1.1})
    fig, ax = plt.subplots(2, 5, figsize=(21, 8.2), sharex=True)
    edges = np.arange(-6.0, 1.51, 0.25)

    def curve(a_, G, col, **kw):
        xc, md = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (G.dm20 >= lo) & (G.dm20 < hi) & np.isfinite(G[col])
            if m.sum() >= 15:
                xc.append(G.dm20[m].median()); md.append(1e3 * G.loc[m, col].median())
        a_.plot(xc, md, **kw)
    for k, grp in enumerate(grps):
        G = S[S.grp == grp]
        if G.empty:
            continue
        a_ = ax[0, k]
        curve(a_, G, "d_old10", color="tab:orange", ls="-", marker="o", ms=3, label="original 10″")
        curve(a_, G, "d_old15", color="tab:orange", ls="--", marker="s", ms=3, label="original 15″")
        curve(a_, G, "d_new10", color="tab:blue", ls="-", marker="o", ms=3, label="new 10″")
        curve(a_, G, "d_new15", color="tab:blue", ls="--", marker="s", ms=3, label="new 15″")
        a_.set(title=f"{grp}: {G.name.nunique()} frames, {len(G)} stars", ylim=(-150, 250))
        a_ = ax[1, k]
        for fb, c in [("narrow", "tab:green"), ("broad", "tab:purple")]:
            H = G[G.fwbin == fb]
            curve(a_, H, "d_new10", color=c, ls="-", marker="o", ms=3, label=f"new 10″, {fb} FWHM")
            curve(a_, H, "d_new15", color=c, ls="--", marker="s", ms=3, label=f"new 15″, {fb} FWHM")
        a_.set(ylim=(-150, 60), xlabel="m_syn − m(SNR=20) of the frame [mag]")
        for a_ in ax[:, k]:
            a_.axhline(0, color="k", lw=0.7); a_.axvline(0, color="0.5", ls=":", lw=0.8); a_.grid(alpha=0.3)
    ax[0, 0].set_ylabel("ZP − ZP(bright SNR≥100) [mmag]  (>0: too bright)")
    ax[1, 0].set_ylabel("new background: split by local FWHM tercile [mmag]")
    ax[0, 0].legend(fontsize=7.5); ax[1, 0].legend(fontsize=7)
    fig.suptitle("A16  10″ vs 15″ diameter apertures, original vs new background, and by local FWHM "
                 "(same stars: Gaia-XP matched, isolated 15″, no shape cuts)", fontsize=10)
    fig.tight_layout(); fig.savefig(LOG / "A16_aper15.png", dpi=125)
    print("saved")
