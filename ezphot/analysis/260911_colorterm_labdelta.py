#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260911_colorterm_labdelta.py

Same 2026-01+ survey colour-term measurement as 260911_colorterm_2026.py, but comparing THREE curves with
corrected (LAMOST) XP: nominal (Final_transmission) / physical (beam only, delta_f=0) /
physical+lab (physical rigidly shifted by the lab-measured intrinsic offset delta_f, GRID filters only).
Reuses the same sample_frames.txt and the heavy helpers (read_cat, tile_xp) from 260911_colorterm_2026.

  MPLCONFIGDIR=/tmp/ezphot-matplotlib OMP_NUM_THREADS=1 /home/hhchoi1022/miniconda3/envs/too/bin/python -u \\
      ezphot/analysis/260911_colorterm_labdelta.py [--n 1000]
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
from astropy.stats import SigmaClip                                # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
C = importlib.import_module("260911_colorterm_2026")
R, K = C.R, C.K
OUT, TS, PHYS, XP_WAVE, GRID = C.OUT, C.TS, C.PHYS, C.XP_WAVE, C.GRID

LAB_DELTA = {"m400": 1.3, "m425": 1.5, "m450": 0.0, "m475": -1.6, "m500": 1.0, "m525": 1.2,
             "m550": 0.0, "m575": 0.0, "m600": 0.5, "m625": 0.0, "m650": -0.3, "m675": -0.3,
             "m700": 1.0, "m725": 2.3, "m750": 0.0, "m775": 0.0, "m800": -1.0, "m825": 1.0,
             "m850": 0.0, "m875": 2.2}


def shift_curve(lam, trans, d):
    return lam, np.interp(lam - d, lam, trans, left=0.0, right=0.0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=1000)
    a = ap.parse_args()
    K._refit_init()
    frames = [f for f in (OUT / "sample_frames.txt").read_text().splitlines() if f.strip()][:a.n]
    curves = {"nominal": {}, "physical": {}, "physlab": {}}
    for pf in sorted(PHYS.glob("m*.csv")):
        f = pf.stem
        try:
            ph = C.load_new_curve(f)
            curves["physical"][f] = ph; curves["nominal"][f] = R.load_curve(f)
            if f in LAB_DELTA:
                curves["physlab"][f] = shift_curve(ph[0], ph[1], LAB_DELTA[f])
        except Exception:
            pass
    print(f"curves: nominal {len(curves['nominal'])}, physical {len(curves['physical'])}, physlab {len(curves['physlab'])}")

    by_tile = defaultdict(list)
    for f in frames:
        by_tile[f.split("/")[-4]].append(f)
    print(f"{len(frames)} frames, {len(by_tile)} tiles")

    rows = []
    for ti, (tile, flist) in enumerate(sorted(by_tile.items()), 1):
        matched, need = [], set()
        for fr in flist:
            filt = fr.split("/")[-3]
            if filt not in curves["physlab"]:      # GRID filters with a lab offset only
                continue
            try:
                d = C.read_cat(fr[:-5] + "_cat.fits")
            except Exception:
                continue
            q = (d.CLASS_STAR >= 0.8) & (d.FLAGS <= 0) & (d.MAGERR_APER_3 < 0.1) & (d.SNR_APER_3 > 10) & (d.SNR_APER_3 < 150)
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
            ok = np.asarray(d2d.arcsec < 2.0)
            if ok.sum() < 10:
                continue
            mr = tbl[idx[ok]]; di = d[ok]
            rid = np.asarray(mr["id"]).astype(str) if "id" in mr.colnames else None
            if rid is None:
                continue
            bprp = np.asarray(mr["bp-rp"], float) if "bp-rp" in mr.colnames else np.full(len(mr), np.nan)
            for k in range(len(di)):
                matched.append((fr, filt, rid[k], bprp[k], float(di.MAG_APER_3.iloc[k]), float(di.MAGERR_APER_3.iloc[k])))
                need.add(rid[k])
        if not matched:
            continue
        xp = C.tile_xp(tile, need)
        M = pd.DataFrame(matched, columns=["file", "filter", "ref_id", "bprp", "inst", "ierr"])
        M = M[M.ref_id.isin(xp)]
        if not len(M):
            continue
        M["sig"] = np.sqrt(M.ierr.to_numpy() ** 2 + 0.01 ** 2)
        for (fr, filt), g in M.groupby(["file", "filter"]):
            if len(g) < 15:
                continue
            fc = np.array([xp[i][1] for i in g.ref_id])       # corrected XP
            inst = g.inst.to_numpy(); bprp = g.bprp.to_numpy(); sig = g.sig.to_numpy()
            rec = dict(file=fr, filter=filt, tile=tile, n=len(g))
            for cn in ("nominal", "physical", "physlab"):
                fw, ft = curves[cn][filt]
                zp = R.syn_mag(XP_WAVE, fc, fw, ft, 0.0) - inst
                fin = np.isfinite(zp) & np.isfinite(bprp) & np.isfinite(sig)
                if fin.sum() < 15:
                    rec[f"a_{cn}"] = np.nan; continue
                cl = SigmaClip(sigma=3.0, maxiters=5)(zp[fin])
                keep = np.where(fin)[0][~np.ma.getmaskarray(cl)]
                aa, ae, bb, nn = R.weighted_linfit(bprp[keep], zp[keep] - np.nanmedian(zp[keep]), sig[keep])
                rec[f"a_{cn}"] = round(aa, 4)
            rows.append(rec)
        if ti % 40 == 0:
            print(f"  {ti}/{len(by_tile)} tiles, {len(rows)} frame-results", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "colorterm_labdelta_frames.csv", index=False)

    CV = ["nominal", "physical", "physlab"]
    d = {"n_frames": res.groupby("filter")["file"].size()}
    for cn in CV:
        d[f"a_{cn}_med"] = res.groupby("filter")[f"a_{cn}"].median()
        d[f"a_{cn}_mad"] = res.groupby("filter")[f"a_{cn}"].apply(lambda x: R.mad(x.values))
    agg = pd.DataFrame(d); agg["pivot"] = [float(f[1:4]) for f in agg.index]
    agg = agg.sort_values("pivot"); agg.to_csv(OUT / "colorterm_labdelta_by_filter.csv")
    print("\n=== colour term by filter (corrected XP), 3 curves ===")
    print(agg[["n_frames"] + [f"a_{cn}_med" for cn in CV]].round(4).to_string())
    print("\nmedian |a| over filters:  " + "  ".join(f"{cn}={agg[f'a_{cn}_med'].abs().median():.4f}" for cn in CV))

    col = {"nominal": "#1f77b4", "physical": "#d62728", "physlab": "#2ca02c"}
    lab = {"nominal": "nominal (Final_transmission)", "physical": "physical (beam, delta_f=0)",
           "physlab": "physical + lab delta_f"}
    fig, ax = plt.subplots(1, 2, figsize=(16, 5.8))
    for i, cn in enumerate(CV):
        ax[0].errorbar(agg["pivot"] + (i - 1) * 3, agg[f"a_{cn}_med"], yerr=agg[f"a_{cn}_mad"], fmt="o-",
                       color=col[cn], ms=4, capsize=2, lw=1, label=lab[cn])
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set(xlabel="pivot [nm]", ylabel="colour term a(BP-RP) [mag/mag]",
              title="(a) field colour term, 2026 survey, corrected XP (median +/- MAD)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    for cn in CV:
        ax[1].plot(agg["pivot"], agg[f"a_{cn}_med"].abs(), "o-", ms=4, color=col[cn], label=lab[cn])
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set(xlabel="pivot [nm]", ylabel="|colour term| [mag/mag]", title="(b) |a| by curve (lower = better)")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.suptitle(f"F19  nominal vs physical vs physical+lab-delta_f: colour term on {len(res)} 2026 survey frames (corrected XP)")
    fig.tight_layout(); fig.savefig(TS / "figures" / "F19_colorterm_labdelta.png", dpi=150)
    print("\nsaved -> colorterm_labdelta_frames.csv, colorterm_labdelta_by_filter.csv, figures/F19_colorterm_labdelta.png")


if __name__ == "__main__":
    main()
