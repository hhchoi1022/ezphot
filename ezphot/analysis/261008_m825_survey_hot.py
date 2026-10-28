#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
261008_m825_survey_hot.py — do hot stars come out too faint with the 7DT10 m825 filter anywhere on the chip, or only the
CALSPEC targets at the image centre?

Survey frames of the colour-term time series (20260911_colorterm_timeseries/colorterm_timeseries_all.csv, m825 rows:
7DT04 2025-01..06, 7DT10 2025-01..11, 7DT07 2025-11..2026-01, 7DT08 2026-01..04) with a pipeline catalogue
(/lyman/data2/processed/<night>/<tile>/m825/singles/*_cat.fits, read-only).  Per frame: stars matched to corrected Gaia XP
(GAIAXP_CORR_LAMOST, 2"), FLAGS 0, 10" aperture (pipeline MAG_APER_5), FLUX_MAX < 40k ADU; synthetic m825 through the
manufacturer beam curve (per-filter n_eff, as section 8); frame ZP = 3-sigma-clipped median of (syn - inst) over stars
with BP-RP 0.5-2.0 and SNR >= 50; Delta_i = inst + ZP - syn [mmag] (+ = measured too faint, the white-dwarf convention).
Summary per unit: median Delta in colour bins for SNR >= 50 (and SNR 20-50), hot stars (BP-RP < 0.3) also split by the
distance from the chip centre.
Outputs: ~/ezphot/log/20261008_m825_survey/ (stars.csv.gz, colour_bins.csv, hot_by_radius.csv, S1_survey_hot.png)

  MPLCONFIGDIR=/tmp/ezphot-matplotlib python -u ezphot/analysis/261008_m825_survey_hot.py --n-proc 32
"""
import argparse
import glob
import importlib
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402
from astropy.coordinates import SkyCoord, match_coordinates_sky   # noqa: E402
from astropy.io import fits       # noqa: E402
from astropy.stats import sigma_clipped_stats                     # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
CT = importlib.import_module("260911_colorterm_2026")
CM = importlib.import_module("261001_calspec_manuf_beam")
K, R = CT.K, CM.R
OUT = Path("/home/hhchoi1022/ezphot/log/20261008_m825_survey")
TS = Path("/home/hhchoi1022/ezphot/log/20260911_colorterm_timeseries/colorterm_timeseries_all.csv")
EGAIN = 0.2562
NX, NY = 9576, 6388
_C = {}


def _init():
    warnings.filterwarnings("ignore")
    K._refit_init()
    _C["curve"] = CM.manuf_totals()[("manufacturer", "beam")]["m825"]


def read(cp):
    with fits.open(cp) as h:
        d = h["LDAC_OBJECTS"].data
        cols = {c: np.asarray(d[c], float) for c in ("ALPHA_J2000", "DELTA_J2000", "X_IMAGE", "Y_IMAGE", "MAG_APER_5", "MAGERR_APER_5",
                                                     "SNR_APER_5", "FLUX_MAX", "FLAGS")}
    return pd.DataFrame(cols)


def process_tile(args):
    tile, cats = args
    out = []
    try:
        frames = []
        for cp in cats:
            try:
                d = read(cp)
            except Exception:                                   # noqa: BLE001
                continue
            d = d[(d.FLAGS == 0) & (d.MAG_APER_5 < 90) & np.isfinite(d.MAG_APER_5) & (d.SNR_APER_5 >= 20) & (d.FLUX_MAX < 40000)]
            if len(d) >= 20:
                frames.append((cp, d))
        if not frames:
            return out
        ra0 = float(np.median(frames[0][1].ALPHA_J2000)); dec0 = float(np.median(frames[0][1].DELTA_J2000))
        got = K.get_reference("GAIAXP_CORR_LAMOST", tile, ra0, dec0, 1.34, 0.90)
        if got is None:
            return out
        tbl, coords, _ = got
        mt, need = [], set()
        for cp, d in frames:
            idx, d2d, _ = match_coordinates_sky(SkyCoord(d.ALPHA_J2000.to_numpy(), d.DELTA_J2000.to_numpy(), unit="deg"), coords)
            ok = np.asarray(d2d.arcsec < 2.0)
            if ok.sum() < 20:
                continue
            dd = d[ok].reset_index(drop=True)
            dd["rid"] = np.asarray(tbl[idx[ok]]["id"]).astype(str)
            dd["bprp"] = np.asarray(tbl[idx[ok]]["bp-rp"], float)
            dd["G"] = np.asarray(tbl[idx[ok]]["g_mean"], float)
            mt.append((cp, dd)); need.update(dd.rid)
        if not mt:
            return out
        xp = CT.tile_xp(tile, need)
        fw, ft = _C["curve"]
        for cp, dd in mt:
            dd = dd[dd.rid.isin(xp)].reset_index(drop=True)
            if len(dd) < 20:
                continue
            fc = np.vstack([xp[r][1] for r in dd.rid])
            syn = R.syn_mag(CT.XP_WAVE, fc, fw, ft, 0.0)
            zp_i = syn - dd.MAG_APER_5.to_numpy()
            ref = np.isfinite(zp_i) & (dd.bprp.between(0.5, 2.0)).to_numpy() & (dd.SNR_APER_5 >= 50).to_numpy()
            if ref.sum() < 10:
                continue
            zp = sigma_clipped_stats(zp_i[ref], sigma=3, maxiters=5)[1]
            name = Path(cp).name
            unit = name.split("_")[2]
            dd["delta"] = 1e3 * (zp - zp_i)
            dd["unit"], dd["tile"], dd["file"], dd["n_ref"] = unit, tile, name, int(ref.sum())
            dd["r_px"] = np.hypot(dd.X_IMAGE - NX / 2, dd.Y_IMAGE - NY / 2)
            dd["peak_e"] = dd.FLUX_MAX * EGAIN
            keep = (dd.bprp < 0.7) | (np.random.default_rng(abs(hash(name)) % 2**32).random(len(dd)) < 0.15)
            out.append(dd[keep & np.isfinite(dd.delta)][["unit", "tile", "file", "rid", "G", "bprp", "SNR_APER_5", "peak_e", "r_px", "delta", "n_ref"]])
    except Exception as e:                                      # noqa: BLE001
        print("tile", tile, "error", str(e)[:120], flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n-proc", type=int, default=32); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    T = pd.read_csv(TS); T = T[T["filter"] == "m825"]
    jobs = {}
    for r in T.drop_duplicates(["unit", "night", "tile"]).itertuples():
        for cp in glob.glob(f"/lyman/data2/processed/{r.night}/{r.tile}/m825/singles/{r.tile}_m825_{r.unit}_*_cat.fits"):
            jobs.setdefault(r.tile, []).append(cp)
    jobs = sorted(jobs.items(), key=lambda kv: -len(kv[1]))
    print(f"{len(jobs)} tiles, {sum(len(v) for _, v in jobs)} catalogues", flush=True)
    rows = []
    with Pool(a.n_proc, initializer=_init) as pool:
        for i, res in enumerate(pool.imap_unordered(process_tile, jobs), 1):
            rows += res
            if i % 20 == 0:
                print(f"  {i}/{len(jobs)} tiles", flush=True)
    S = pd.concat(rows, ignore_index=True)
    S.to_csv(OUT / "stars.csv.gz", index=False)
    edges = [-1.0, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.2, 1.6, 2.2, 4]
    S["cbin"] = pd.cut(S.bprp, edges)
    S["sbin"] = np.where(S.SNR_APER_5 >= 50, "SNR>=50", "SNR 20-50")
    agg = lambda q: pd.Series(dict(n=len(q), n_stars=q.rid.nunique(), frames=q.file.nunique(), col=q.bprp.median(), delta=q.delta.median(),
                                   err=1.2533 * 1.4826 * np.median(np.abs(q.delta - q.delta.median())) / np.sqrt(max(len(q), 1)), peak=q.peak_e.median()))
    CB = S.groupby(["unit", "sbin", "cbin"], observed=True).apply(agg).reset_index()
    CB.to_csv(OUT / "colour_bins.csv", index=False)
    H = S[(S.bprp < 0.3) & (S.SNR_APER_5 >= 50)].assign(rbin=lambda q: pd.cut(q.r_px, [0, 1000, 2500, 6000]))
    HR = H.groupby(["unit", "rbin"], observed=True).apply(agg).reset_index(); HR.to_csv(OUT / "hot_by_radius.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    print(CB[CB.sbin == "SNR>=50"].round(1).to_string(index=False))
    print(HR.round(1).to_string(index=False))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    cols = {"7DT04": "#8c6bb1", "7DT07": "#41ab5d", "7DT08": "#2b8cbe", "7DT10": "#d6604d"}
    for ax, sb in zip(axes, ("SNR>=50", "SNR 20-50")):
        for u, q in CB[CB.sbin == sb].groupby("unit"):
            q = q[q.n >= 15]
            ax.errorbar(q.col, q.delta, q.err, fmt="o-", color=cols.get(u, "k"), ms=4, label=f"{u}")
        ax.axhline(0, color="0.5", lw=0.8); ax.axvspan(-1, 0.3, color="0.9", zorder=0); ax.grid(alpha=0.25); ax.set_xlim(-0.6, 3)
        ax.set_xlabel("BP − RP"); ax.set_ylabel("Δ = inst + ZP − syn [mmag]"); ax.set_title(f"Survey m825, field stars {sb} (shaded: hot, BP − RP < 0.3)", fontsize=10)
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "S1_survey_hot.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
