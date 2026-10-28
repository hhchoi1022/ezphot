#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260911_lamost_vs_xp.py

Do the GAIAXP-corrected spectra agree with the LAMOST DR7 spectra?  (No 7DT photometry in the loop.)
For reference stars that have BOTH a corrected-XP spectrum and a LAMOST spectrum, synthesise each through the
same medium bands and compare per band.  This isolates the XP-vs-LAMOST agreement — telling us whether the
~59 mmag floor / arm-junction wiggle in the 7DT-vs-LAMOST test (16.2) is 7DT photometry noise or a genuine
XP-vs-LAMOST systematic.

  MPLCONFIGDIR=/tmp/ezphot-matplotlib OMP_NUM_THREADS=1 /home/hhchoi1022/miniconda3/envs/too/bin/python -u \\
      ezphot/analysis/260911_lamost_vs_xp.py --n-tiles 40
"""
from __future__ import annotations

import argparse
import importlib
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402
import pandas as pd                 # noqa: E402
import astropy.units as u           # noqa: E402
from astropy.coordinates import SkyCoord, match_coordinates_sky, search_around_sky   # noqa: E402
from astroquery.vizier import Vizier                              # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
RC = importlib.import_module("260911_lamost_refcat")              # reuse curves, LAMOST download, modules
C, R, K = RC.C, RC.R, RC.K
XP_WAVE = RC.XP_WAVE
GRID = RC.GRID
TILES = RC.TILES
OUT = RC.OUT
LAMOST_SNR = 30.0
ISO_ARCSEC = 15.0


def isolate_ref(coords):
    """Boolean mask: reference stars with no other reference star within ISO_ARCSEC."""
    idx, sep, _ = match_coordinates_sky(coords, coords, nthneighbor=2)
    return np.asarray(sep.arcsec > ISO_ARCSEC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tiles", type=int, default=40)
    ap.add_argument("--curve", default="physfit")
    a = ap.parse_args()
    K._refit_init()
    cv = RC.build_curves()
    tiles = RC.sample_equatorial_tiles(a.n_tiles)
    print(f"{len(tiles)} tiles, curve={a.curve}", flush=True)

    Vizier.ROW_LIMIT = -1
    pairs = []            # (tile, ref_id, obsid, subclass, snrr)
    for tile in tiles:
        t = TILES[TILES.id == tile]
        if not len(t):
            continue
        ra0, dec0 = float(t.ra.iloc[0]), float(t.dec.iloc[0])
        got = K.get_reference("GAIAXP_CORR_LAMOST", tile, ra0, dec0, 1.34, 0.90)
        if got is None:
            continue
        tbl, coords, _ = got
        rid = np.asarray(tbl["id"]).astype(str)
        iso = isolate_ref(coords)
        try:
            lam = Vizier(columns=["ObsID", "RAJ2000", "DEJ2000", "Class", "subClass", "snrr"]).query_region(
                f"{ra0} {dec0}", radius="0.8 deg", catalog="V/156/dr7lrs")[0].to_pandas()
        except Exception:
            continue
        lam = lam[(lam["Class"] == "STAR") & (lam.snrr > LAMOST_SNR)]
        if not len(lam):
            continue
        lc = SkyCoord(lam.RAJ2000.to_numpy(), lam.DEJ2000.to_numpy(), unit="deg")
        idx, sep, _ = match_coordinates_sky(coords, lc)
        ok = (sep.arcsec < 2.0) & iso
        for k in np.where(ok)[0]:
            pairs.append((tile, rid[k], int(lam.ObsID.to_numpy()[idx[k]]),
                          lam.subClass.to_numpy()[idx[k]], float(lam.snrr.to_numpy()[idx[k]])))
    P = pd.DataFrame(pairs, columns=["tile", "ref_id", "obsid", "subclass", "snrr"]).drop_duplicates("obsid")
    print(f"{len(P)} isolated stars with BOTH corrected-XP and LAMOST spectra", flush=True)

    # per star: synth both spectra through each band, compare
    rows = []             # (pivot, diff_mmag)
    star_rms = []
    examples = []
    for tile, g in P.groupby("tile"):
        xp = C.tile_xp(tile, set(g.ref_id))
        for _, s in g.iterrows():
            if s.ref_id not in xp:
                continue
            xflux = xp[s.ref_id][1]                                # corrected XP flux on XP_WAVE
            try:
                lw, lf = RC.get_lamost_spec(int(s.obsid))
            except Exception:
                continue
            xsyn = np.array([R.syn_mag(XP_WAVE, xflux, *cv[b][a.curve], 0.0) for b in GRID])
            lsyn = np.array([R.syn_mag(lw, lf, *cv[b][a.curve], 0.0) for b in GRID])
            good = np.isfinite(xsyn) & np.isfinite(lsyn)
            if good.sum() < 10:
                continue
            grey = np.median((xsyn - lsyn)[good])
            d = xsyn - lsyn - grey
            for b, dv, gd in zip(GRID, d, good):
                if gd:
                    rows.append((int(b[1:4]), 1000 * dv))
            star_rms.append(round(1000 * np.std(d[good]), 1))
            if len(examples) < 4 and s.snrr > 80:
                examples.append((s, xflux, lw, lf, grey))
    RB = pd.DataFrame(rows, columns=["pivot", "diff_mmag"])
    RB.to_csv(OUT / "xp_vs_lamost_bandresid.csv", index=False)
    per_band = RB.groupby("pivot").diff_mmag.agg(["mean", "std", "count"])
    per_band["sem"] = per_band["std"] / np.sqrt(per_band["count"])
    print(f"\nstars used: {len(star_rms)}; median per-star band-diff rms: {np.median(star_rms):.1f} mmag")
    print(f"per-band |mean(XP-LAMOST)| median: {np.median(np.abs(per_band['mean'])):.1f} mmag")
    print("\nper-band mean(XP_corr - LAMOST) [mmag]:")
    print(per_band["mean"].round(1).to_string())

    # figure
    fig, ax = plt.subplots(2, 1, figsize=(13, 9))
    ax[0].errorbar(per_band.index, per_band["mean"], yerr=per_band["sem"], fmt="o-", color="#1f77b4", capsize=2, ms=5)
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].axvspan(555, 585, color="orange", alpha=0.15, label="LAMOST blue/red arm join (~580 nm)")
    ax[0].set(xlabel="pivot wavelength [nm]", ylabel="mean(XP_corr_syn - LAMOST_syn) [mmag]",
              title=f"GAIAXP-corrected vs LAMOST DR7 (synth through 7DT bands, {a.curve}) — "
                    f"{len(star_rms)} isolated stars, per-star rms {np.median(star_rms):.0f} mmag")
    ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)
    # example overlay in relative AB mag
    if examples:
        s, xflux, lw, lf, grey = examples[0]
        # relative AB mag: m = -2.5 log10(fnu); fnu ∝ flam * lam^2
        xm = -2.5 * np.log10(np.clip(xflux, 1e-30, None) * XP_WAVE ** 2)
        lm = -2.5 * np.log10(np.clip(lf, 1e-30, None) * lw ** 2)
        xm = xm - np.median(xm[(XP_WAVE > 500) & (XP_WAVE < 800)])
        lm = lm - np.median(lm[(lw > 500) & (lw < 800)])
        ax[1].plot(lw, lm, color="#888", lw=0.5, alpha=0.8, label=f"LAMOST {s.subclass} (snr {s.snrr:.0f})")
        ax[1].plot(XP_WAVE, xm, color="#d62728", lw=1.8, label="GAIAXP corrected")
        ax[1].set(xlim=(395, 905), ylim=(1.2, -1.4), xlabel="wavelength [nm]", ylabel="relative AB mag (norm 500-800nm)")
        ax[1].legend(fontsize=9, loc="lower right"); ax[1].grid(alpha=0.3)
        ax[1].set_title(f"example overlay — obsid {int(s.obsid)}")
    fig.tight_layout(); fig.savefig(OUT / "F27_xp_vs_lamost.png", dpi=140)
    print(f"\nsaved -> F27_xp_vs_lamost.png, xp_vs_lamost_bandresid.csv in {OUT}")


if __name__ == "__main__":
    main()
