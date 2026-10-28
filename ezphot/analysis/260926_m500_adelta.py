#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260926_m500_adelta.py — why delta* is not identifiable in m500: the colour term a(delta) as a function of the curve
shift, on a wide fine grid, for m450 / m500 / m525 frames of the same tiles (same stars/curves as 260926_delta_star_v2:
SNR >= 100, corrected XP, manufacturer x beam x QE x optics x atm, vacuum).

  (a) a(delta) per frame (thin) and median (thick), with the typical per-frame 1-sigma error of a
  (b) ln(F_red / F_blue): median XP spectrum of red stars (BP-RP 1.2-1.5) over blue stars (0.6-0.8); the colour-term
      sensitivity da/d(delta) is -(2.5/ln10) x the band-averaged slope of this curve per unit colour difference of the
      two groups, so where it is flat inside a band the shift cannot be measured
Output (~/ezphot/log/20260926_delta_star_v2/): m500_adelta.csv, D2_m500_adelta.png
"""
import importlib
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from astropy.coordinates import SkyCoord, match_coordinates_sky   # noqa: E402

sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
DS = importlib.import_module("260926_delta_star_v2")
OT, C, K, R = DS.OT, DS.C, DS.K, DS.R
OUT = DS.OUT
FILTS = ["m450", "m500", "m525"]
VAR = "manufacturer_beam_vac"
DG = np.arange(-15.0, 15.01, 0.5)
N_TILES = 16


def tile_frames(args):
    """a(delta) on DG for every m450/m500/m525 frame of one tile + the stars' spectra for panel (b)."""
    tile, frames = args
    matched, need = [], set()
    for fr, filt in frames:
        cp = DS.CATD / (Path(fr).name[:-5] + "_cat.fits")
        d = DS.select_stars_v2(cp)
        if len(d) < 15:
            continue
        got = K.get_reference("GAIAXP_CORR_LAMOST", tile, float(np.median(d.ALPHA_J2000)), float(np.median(d.DELTA_J2000)), 1.34, 0.90)
        if got is None or got[1] is None or len(got[1]) == 0:
            continue
        tbl, coords, _ = got
        idx, d2d, _ = match_coordinates_sky(SkyCoord(d.ALPHA_J2000.to_numpy(), d.DELTA_J2000.to_numpy(), unit="deg"), coords)
        ok = np.asarray(d2d.arcsec < 2.0)
        mr, di = tbl[idx[ok]], d[ok]
        for rid, bp, m, e in zip(np.asarray(mr["id"]).astype(str), np.asarray(mr["bp-rp"], float), di.MAG_APER_3, di.MAGERR_APER_3):
            matched.append((filt, fr, rid, bp, float(m), float(e)))
            need.add(rid)
    if not matched:
        return [], None
    xp = C.tile_xp(tile, need)
    M = pd.DataFrame(matched, columns=["filter", "file", "ref_id", "bprp", "inst", "ierr"])
    M = M[M.ref_id.isin(xp)]
    M["sig"] = np.sqrt(M.ierr ** 2 + 0.01 ** 2)
    rows = []
    for (filt, fr), g in M.groupby(["filter", "file"]):
        if len(g) < 15:
            continue
        fw, ft = DS._CURVES[VAR][filt]
        fc = np.vstack([xp[i][1] for i in g.ref_id])
        zp0 = R.syn_mag(C.XP_WAVE, fc, fw, ft, 0.0) - g.inst.to_numpy()
        fin = np.isfinite(zp0) & np.isfinite(g.bprp.to_numpy())
        # same star set as OT.solve_offset: 3-sigma ZP clip at delta = 0
        from astropy.stats import SigmaClip
        keep = np.where(fin)[0][~np.ma.getmaskarray(SigmaClip(sigma=3.0, maxiters=5)(zp0[fin]))]
        gk, fk = g.iloc[keep], fc[keep]
        for dd in DG:
            zp = R.syn_mag(C.XP_WAVE, fk, fw, ft, dd) - gk.inst.to_numpy()
            a, ae, _, _ = R.weighted_linfit(gk.bprp.to_numpy(), zp - np.nanmedian(zp), gk.sig.to_numpy())
            rows.append(dict(tile=tile, filter=filt, file=fr, n=len(gk), delta=dd, a=a, a_err=ae))
    # spectra for panel (b): one row per unique star
    S = M.drop_duplicates("ref_id")
    spec = (S.bprp.to_numpy(), np.vstack([xp[i][1] for i in S.ref_id]))
    return rows, spec


def init():
    DS._init()


if __name__ == "__main__":
    F = pd.read_csv(OUT / "delta_star_v2_frames.csv")
    F = F[(F.variant == VAR) & F["filter"].isin(FILTS) & (F.n >= 80)]
    ok = F.groupby("tile")["filter"].nunique()
    tiles = F[F.tile.isin(ok[ok == 3].index) & (F["filter"] == "m500")].groupby("tile").n.median().sort_values(ascending=False).index
    tiles = list(tiles[:: max(1, len(tiles) // N_TILES)][:N_TILES])     # spread over the star-count ranking
    items = [(t, list(F[F.tile == t][["file", "filter"]].itertuples(index=False, name=None))) for t in tiles]
    with Pool(len(items), initializer=init) as pool:
        res = pool.map(tile_frames, items)
    A = pd.DataFrame([r for rows, _ in res for r in rows])
    A.to_csv(OUT / "m500_adelta.csv", index=False)
    bp = np.concatenate([s[0] for _, s in res if s is not None]); fl = np.vstack([s[1] for _, s in res if s is not None])
    # summary numbers
    print(f"{A.file.nunique()} frames, {len(tiles)} tiles")
    for filt, g in A.groupby("filter"):
        W = g.pivot_table(index="file", columns="delta", values="a")
        ncross = ((np.sign(W.to_numpy()[:, 1:]) != np.sign(W.to_numpy()[:, :-1])).sum(axis=1))
        med = W.median(axis=0)
        e0 = g[g.delta == 0].a_err.median()
        lin = np.polyfit(DG[np.abs(DG) <= 6], med[np.abs(DG) <= 6], 1)[0]
        print(f"  {filt}: n_frames {len(W)}, median a(0) {1e3 * med[0.0]:+.1f}, a(-6) {1e3 * med[-6.0]:+.1f}, "
              f"a(+6) {1e3 * med[6.0]:+.1f}, a(-15) {1e3 * med[-15.0]:+.1f}, a(+15) {1e3 * med[15.0]:+.1f} mmag/mag; "
              f"slope(|d|<=6) {1e3 * lin:+.2f} mmag/mag/nm; per-frame err(a) {1e3 * e0:.1f}; "
              f"zero crossings per frame in +-15 nm: median {np.median(ncross):.0f}, >1 in {np.mean(ncross > 1) * 100:.0f}%")
    # figure
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.2))
    col = dict(m450="tab:blue", m500="tab:red", m525="tab:green")
    for filt, g in A.groupby("filter"):
        W = g.pivot_table(index="file", columns="delta", values="a") * 1e3
        for _, r in W.iterrows():
            ax[0].plot(DG, r.to_numpy(), color=col[filt], lw=0.4, alpha=0.25)
        med = W.median(axis=0).to_numpy(); e0 = 1e3 * g[g.delta == 0].a_err.median()
        ax[0].plot(DG, med, color=col[filt], lw=2.4, label=f"{filt} (median of {len(W)} frames)")
        ax[0].fill_between(DG, med - e0, med + e0, color=col[filt], alpha=0.15)
    ax[0].axhline(0, color="k", lw=0.8); ax[0].axvline(0, color="k", lw=0.5, ls=":")
    ax[0].set_ylim(-150, 150); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
    ax[0].set(xlabel="curve shift δ [nm] (+ = redward)", ylabel="colour term a(δ) [mmag / mag of BP−RP]",
              title="(a) colour term vs curve shift: δ* = where the curve crosses 0\n(band = typical per-frame ±1σ of a)")
    w = C.XP_WAVE
    lam0 = np.argmin(np.abs(w - 550))
    blue = fl[(bp > 0.6) & (bp < 0.8)]; red = fl[(bp > 1.2) & (bp < 1.5)]
    rb = np.log(np.median(red / red[:, [lam0]], axis=0) / np.median(blue / blue[:, [lam0]], axis=0))
    ax[1].plot(w, rb, "k-", lw=1.5, label=f"ln(F_red / F_blue): median XP, BP−RP 1.2–1.5 ({len(red)}) / 0.6–0.8 ({len(blue)})")
    DS._CURVES = DS.build_curves()
    for filt in ["m425", "m450", "m475", "m500", "m525", "m550", "m575"]:
        fw, ft = DS._CURVES[VAR][filt]
        ax[1].fill_between(fw, rb.min() - 0.05, rb.min() - 0.05 + 0.25 * ft / ft.max(), color=col.get(filt, "0.7"),
                           alpha=0.5 if filt in col else 0.25, lw=0)
        ax[1].text(fw[np.argmax(ft)], rb.min() - 0.05 + 0.27, filt, ha="center", fontsize=7)
    for lv, nm in [(486.27, "Hβ"), (517.4, "Mg b"), (434.2, "Hγ"), (431.0, "G band")]:
        ax[1].axvline(lv, color="0.5", lw=0.6, ls="--"); ax[1].text(lv, rb.max(), nm, fontsize=7, rotation=90, va="top")
    ax[1].set_xlim(400, 600); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=7, loc="upper left")
    sel = (w > 400) & (w < 600)
    ax[1].set_ylim(rb.min() - 0.08, rb[sel].max() + 0.05)
    ax[1].set(xlabel="wavelength [nm] (vacuum)", ylabel="ln(red star / blue star), normalised at 550 nm",
              title="(b) colour-term sensitivity ∝ −(slope of this curve inside the band)\nflat inside m500 → shifting the curve does not change the colour term")
    fig.suptitle("D2  Why δ* cannot be determined in m500 (16 tiles with m450, m500 and m525 frames; SNR ≥ 100 stars, beam, vacuum)", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "D2_m500_adelta.png", dpi=125)
    # band-averaged slope of rb (per nm) -> predicted sensitivity
    for filt in ["m425", "m450", "m475", "m500", "m525", "m550", "m575"]:
        fw, ft = DS._CURVES[VAR][filt]
        t = np.interp(w, fw, ft, left=0, right=0); s = np.gradient(rb, w)
        dcol = np.median(bp[(bp > 1.2) & (bp < 1.5)]) - np.median(bp[(bp > 0.6) & (bp < 0.8)])
        pred = -2.5 / np.log(10) * np.sum(s * t * w) / np.sum(t * w) / dcol
        print(f"  {filt}: predicted da/d(delta) from panel (b) = {pred * 1e3:+.1f} mmag/mag/nm")
    print("saved", OUT / "D2_m500_adelta.png")
