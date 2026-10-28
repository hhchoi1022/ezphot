#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260911_colorterm_2026.py

Field-star colour term on the 2026-01+ pipeline single frames (/lyman/data2/processed/{date}/T?????/{filter}/
singles/*.fits), comparing the ORIGINAL (nominal Final_transmission) vs NEW (physical) transmission curves.
The pipeline already wrote LDAC catalogues (<frame>_cat.fits, HDU 'LDAC_OBJECTS'), so NO re-photometry: read the
catalogue -> match to the GAIAXP_CORR_LAMOST archive -> per-star ZP = syn_mag(curve, XP_cor) - MAG_APER_3 ->
3-sigma-clipped frame ZP -> colour term a(BP-RP).  Random sample (seed 20260911).  Corrected XP only.

Read-only on /lyman/data2/processed; all outputs to /home/hhchoi1022/ezphot/log/20260911_colorterm_2026/.

  MPLCONFIGDIR=/tmp/ezphot-matplotlib OMP_NUM_THREADS=1 /home/hhchoi1022/miniconda3/envs/too/bin/python \\
      ezphot/analysis/260911_colorterm_2026.py [--n 1000]
"""
from __future__ import annotations

import argparse
import importlib
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402
import pandas as pd                 # noqa: E402
from astropy.coordinates import SkyCoord, match_coordinates_sky   # noqa: E402
from astropy.io import fits         # noqa: E402
from astropy.stats import SigmaClip                                # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
R = importlib.import_module("260908_zpcolor_shift_recal")
K = importlib.import_module("260908_zpcolor_collect")

OUT = Path("/home/hhchoi1022/ezphot/log/20260911_colorterm_2026")
TS = Path("/home/hhchoi1022/ezphot/log/20260910_transmission_sources")
PHYS = TS / "total_curves_physical"
TILESRC = Path("/lyman/data1/factory_doabdi/GAIA_BEST/final_result")
TILES = pd.read_csv("/lyman/data1/factory_hhchoi/code/ezphot/ezphot/utils/tileinfo/7-DT/final_tiles.txt", sep=r"\s+")
GRID = R.GRID
XP_WAVE = R.XP_WAVE_NM


def load_new_curve(f):
    d = pd.read_csv(PHYS / f"{f}.csv"); return d.lam.to_numpy(), d.trans.to_numpy()


def read_cat(catpath):
    with fits.open(catpath) as h:
        d = h["LDAC_OBJECTS"].data
        cols = {c: np.asarray(d[c], float) for c in
                ["ALPHA_J2000", "DELTA_J2000", "MAG_APER_3", "MAGERR_APER_3", "FLAGS", "CLASS_STAR", "SNR_APER_3"]}
    return pd.DataFrame(cols)


def tile_xp(tilename, want_ids):
    """(flux_orig[col 11], flux_cor[col 13]) for the wanted ids, scanning the tile + neighbours in final_result."""
    if tilename not in set(TILES.id):
        near = TILES
    else:
        t = TILES[TILES.id == tilename].iloc[0]
        dd = np.abs(TILES.dec.values - t.dec)
        dr = np.abs(((TILES.ra.values - t.ra + 180) % 360) - 180) * np.cos(np.radians(t.dec))
        near = TILES[(dd < 1.2) & (dr < 1.2)]
    want = set(str(i) for i in want_ids); out = {}
    for tn in near.id.values:
        p = TILESRC / f"{tn}.csv"
        if not p.exists() or not want - set(out):
            continue
        with open(p) as fh:
            fh.readline()
            for line in fh:
                c = line.split(",", 14)
                sid = c[0].lstrip("﻿")
                if sid in want and sid not in out:
                    vo = np.fromstring(c[11], sep=";")     # original (uncalibrated) XP flux
                    vc = np.fromstring(c[13], sep=";")     # LAMOST-corrected flux
                    if vo.size == 343 and vc.size == 343:
                        out[sid] = (vo, vc)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    a = ap.parse_args()
    K._refit_init()
    frames = [f for f in (OUT / "sample_frames.txt").read_text().splitlines() if f.strip()][:a.n]
    # load nominal + physical curves for every medium band that has a physical curve (37 bands, not just GRID)
    curves = {"nominal": {}, "physical": {}}
    for pf in sorted(PHYS.glob("m*.csv")):
        f = pf.stem
        try:
            curves["physical"][f] = load_new_curve(f); curves["nominal"][f] = R.load_curve(f)
        except Exception:
            pass
    print(f"curves loaded for {len(curves['physical'])} medium bands")

    by_tile = defaultdict(list)
    for f in frames:
        by_tile[f.split("/")[-4]].append(f)
    print(f"{len(frames)} frames, {len(by_tile)} tiles")

    rows = []
    for ti, (tile, flist) in enumerate(sorted(by_tile.items()), 1):
        # match every frame of this tile, collect needed archive ids
        matched = []      # (frame, filt, ref_id, bprp, inst, ierr, ref_mag)
        need = set()
        for fr in flist:
            filt = fr.split("/")[-3]
            if filt not in curves["nominal"]:
                continue
            cat = fr[:-5] + "_cat.fits"
            try:
                d = read_cat(cat)
            except Exception:
                continue
            q = (d.CLASS_STAR >= 0.8) & (d.FLAGS <= 0) & (d.MAGERR_APER_3 < 0.1) \
                & (d.SNR_APER_3 > 10) & (d.SNR_APER_3 < 150)
            d = d[q]
            if len(d) < 10:
                continue
            ra0, dec0 = float(np.median(d.ALPHA_J2000)), float(np.median(d.DELTA_J2000))
            got = K.get_reference("GAIAXP_CORR_LAMOST", tile, ra0, dec0, 1.34, 0.90)
            if got is None:
                continue
            tbl, coords, _ = got
            magcol, _ = K.resolve_mag_column(filt, tbl.colnames, "GAIAXP_CORR_LAMOST")
            if magcol is None:
                continue
            src = SkyCoord(d.ALPHA_J2000.to_numpy(), d.DELTA_J2000.to_numpy(), unit="deg")
            idx, d2d, _ = match_coordinates_sky(src, coords)
            ok = d2d.arcsec < 2.0
            if ok.sum() < 10:
                continue
            mr = tbl[idx[ok.to_numpy() if hasattr(ok, "to_numpy") else ok]]
            di = d[np.asarray(ok)]
            rid = np.asarray(mr["id"]).astype(str) if "id" in mr.colnames else None
            if rid is None:
                continue
            bprp = np.asarray(mr["bp-rp"], float) if "bp-rp" in mr.colnames else np.full(len(mr), np.nan)
            refm = np.asarray(mr[magcol], float)
            for k in range(len(di)):
                matched.append((fr, filt, rid[k], bprp[k], float(di.MAG_APER_3.iloc[k]),
                                float(di.MAGERR_APER_3.iloc[k]), refm[k]))
                need.add(rid[k])
        if not matched:
            continue
        xp = tile_xp(tile, need)
        M = pd.DataFrame(matched, columns=["file", "filter", "ref_id", "bprp", "inst", "ierr", "ref_mag"])
        M = M[M.ref_id.isin(xp)]
        if not len(M):
            continue
        M["sig"] = np.sqrt(M.ierr.to_numpy() ** 2 + 0.01 ** 2)
        # per frame, per curve, per XP version (corrected vs original) colour term
        for (fr, filt), g in M.groupby(["file", "filter"]):
            if len(g) < 15:
                continue
            ids = list(g.ref_id)
            fo = np.array([xp[i][0] for i in ids]); fc = np.array([xp[i][1] for i in ids])
            inst = g.inst.to_numpy(); bprp = g.bprp.to_numpy(); sig = g.sig.to_numpy()
            rec = dict(file=fr, filter=filt, tile=tile, n=len(g))
            for cn in ("nominal", "physical"):
                fw, ft = curves[cn][filt]
                for xn, flx in (("cor", fc), ("orig", fo)):
                    m_ref = R.syn_mag(XP_WAVE, flx, fw, ft, 0.0)
                    zp = m_ref - inst
                    fin = np.isfinite(zp) & np.isfinite(bprp) & np.isfinite(sig)
                    if fin.sum() < 15:
                        rec[f"a_{cn}_{xn}"] = np.nan; continue
                    cl = SigmaClip(sigma=3.0, maxiters=5)(zp[fin])
                    keep = np.where(fin)[0][~np.ma.getmaskarray(cl)]
                    resid = zp[keep] - np.nanmedian(zp[keep])
                    aa, ae, bb, nn = R.weighted_linfit(bprp[keep], resid, sig[keep])
                    rec[f"a_{cn}_{xn}"] = round(aa, 4)
            rec["n_fit"] = int(len(g))
            rows.append(rec)
        if ti % 20 == 0:
            print(f"  {ti}/{len(by_tile)} tiles, {len(rows)} frame-results")
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "colorterm_frames.csv", index=False)

    # ---- aggregate per filter: 4 combos = {nominal, physical} x {corrected, original} XP ----
    COMBOS = ["nominal_cor", "physical_cor", "nominal_orig", "physical_orig"]
    d = {"n_frames": res.groupby("filter")["file"].size()}
    for cb in COMBOS:
        d[f"a_{cb}_med"] = res.groupby("filter")[f"a_{cb}"].median()
        d[f"a_{cb}_mad"] = res.groupby("filter")[f"a_{cb}"].apply(lambda x: R.mad(x.values))
    agg = pd.DataFrame(d)
    agg["pivot"] = [float(f[1:4]) for f in agg.index]
    agg = agg.sort_values("pivot"); agg.to_csv(OUT / "colorterm_by_filter.csv")
    print("\n=== colour term by filter (median over frames), 4 combos ===")
    print(agg[["n_frames"] + [f"a_{cb}_med" for cb in COMBOS]].round(4).to_string())
    print("\nmedian |a| over filters:  " + "  ".join(f"{cb}={agg[f'a_{cb}_med'].abs().median():.4f}" for cb in COMBOS))

    col = {"nominal_cor": "#1f77b4", "physical_cor": "#d62728", "nominal_orig": "#9467bd", "physical_orig": "#ff7f0e"}
    lab = {"nominal_cor": "nominal x corrected (= archive)", "physical_cor": "physical x corrected",
           "nominal_orig": "nominal x original", "physical_orig": "physical x original"}
    fig, ax = plt.subplots(1, 2, figsize=(16, 5.8))
    for i, cb in enumerate(COMBOS):
        ax[0].errorbar(agg["pivot"] + (i - 1.5) * 2.5, agg[f"a_{cb}_med"], yerr=agg[f"a_{cb}_mad"], fmt="o-",
                       color=col[cb], ms=4, capsize=2, lw=1, label=lab[cb])
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set(xlabel="pivot [nm]", ylabel="colour term a(BP-RP) [mag/mag]",
              title="(a) field colour term, 2026-01+ survey (median +/- MAD)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    for cb in COMBOS:
        ax[1].plot(agg["pivot"], agg[f"a_{cb}_med"].abs(), "o-", ms=4, color=col[cb], label=lab[cb])
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set(xlabel="pivot [nm]", ylabel="|colour term| [mag/mag]", title="(b) |a| by combination (lower = better)")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.suptitle(f"F18  transmission curve (original vs new) x XP (original vs corrected): colour term on {len(res)} 2026 survey frames")
    fig.tight_layout(); fig.savefig(TS / "figures" / "F18_colorterm_2026.png", dpi=150)
    print("\nsaved -> colorterm_frames.csv, colorterm_by_filter.csv, figures/F18_colorterm_2026.png")


if __name__ == "__main__":
    main()
