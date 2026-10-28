#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260911_lamost_refcat.py

LAMOST DR7 validation of the transmission curves, but with the 7DT magnitudes RE-CALIBRATED from the
corrected XP reference catalogue (not the old-pipeline header ZP).  For each candidate curve
(nominal[=old pipeline] / physical / physfit) we:

  1. per band frame: match stars to GAIAXP_CORR_LAMOST reference, synth ref mag = syn_mag(curve, corrected XP),
     frame ZP = 3-sigma clipped median(ref_syn - MAG_APER_5), calibrated 7DT mag = MAG_APER_5 + ZP;
  2. assemble the multi-band 7DT SED per curve (keyed by reference id);
  3. crossmatch to LAMOST DR7, download the spectrum, synth LAMOST mag through the SAME curve;
  4. residual per band = cal_7DT(curve) - LAMOST_syn(curve) - grey; the curve that flattens it wins.

Uses the 10" aperture (MAG_APER_5, total flux; established cleaner than 3" for SED work).  Read-only on the
pipeline; outputs to /home/hhchoi1022/ezphot/log/20260911_lamost/.

  MPLCONFIGDIR=/tmp/ezphot-matplotlib OMP_NUM_THREADS=1 /home/hhchoi1022/miniconda3/envs/too/bin/python -u \\
      ezphot/analysis/260911_lamost_refcat.py --aper 5
"""
from __future__ import annotations

import argparse
import gzip
import importlib
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402
import pandas as pd                 # noqa: E402
import requests                     # noqa: E402
import astropy.units as u          # noqa: E402
from astropy.coordinates import SkyCoord, match_coordinates_sky, search_around_sky   # noqa: E402
from astropy.io import fits         # noqa: E402
from astropy.stats import sigma_clip                              # noqa: E402
from astroquery.vizier import Vizier                              # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
C = importlib.import_module("260911_colorterm_2026")
L = importlib.import_module("260911_colorterm_labdelta")
R, K = C.R, C.K
XP_WAVE = C.XP_WAVE                 # nm grid of the XP sampled spectra (336..1020, 2 nm)
GRID = R.GRID
TILES = C.TILES
ENUM = Path("/home/hhchoi1022/ezphot/log/20260911_colorterm_timeseries/all_medium_2025.txt")
OUT = Path("/home/hhchoi1022/ezphot/log/20260911_lamost"); (OUT / "spec").mkdir(parents=True, exist_ok=True)
FIT = pd.read_csv("/home/hhchoi1022/ezphot/log/20260910_effective_curves/offset_plus_linear_summary.csv").set_index("filter")
CURVE_NAMES = ("nominal", "physical", "physlab", "physfit")  # curves whose ZP we recompute from corrected XP
ALL_CAL = ("header",) + CURVE_NAMES                          # + old-pipeline header ZP baseline
SYNTH_CURVE = {"header": "nominal", "nominal": "nominal", "physical": "physical",
               "physlab": "physlab", "physfit": "physfit"}
CCOL = {"header": "#000000", "nominal": "#1f77b4", "physical": "#d62728", "physlab": "#2ca02c", "physfit": "#9467bd"}
# default equatorial tiles with dense LAMOST coverage (from the overlap scan)
DEF_TILES = ["T22626", "T25077", "T21768", "T23440", "T22392", "T24762", "T24505", "T24553"]
APER = 5
SNR_MIN = 50.0            # 7DT aperture SNR floor (bright, low-noise stars)
ISO_ARCSEC = 15.0         # reject stars with a comparably-bright neighbour within this radius (10" aper vs 3.3" LAMOST fiber)
LAMOST_SNR = 30.0         # LAMOST spectrum SNR (snrr) floor


def read_cat_aper(catpath):
    with fits.open(catpath) as h:
        d = h["LDAC_OBJECTS"].data
        out = {c: np.asarray(d[c], float) for c in ["ALPHA_J2000", "DELTA_J2000", "FLAGS", "CLASS_STAR"]}
        out["inst"] = np.asarray(d[f"MAG_APER_{APER}"], float)
        out["ierr"] = np.asarray(d[f"MAGERR_APER_{APER}"], float)
        out["snr"] = np.asarray(d[f"SNR_APER_{APER}"], float)
    return pd.DataFrame(out)


def build_curves():
    cv = {}
    for f in GRID:
        ph = C.load_new_curve(f)
        off = float(FIT.loc[f, "offset"]) if f in FIT.index and np.isfinite(FIT.loc[f, "offset"]) else 0.0
        cv[f] = {"nominal": R.load_curve(f), "physical": ph,
                 "physlab": L.shift_curve(ph[0], ph[1], L.LAB_DELTA[f]),   # physical + lab-measured delta_f
                 "physfit": L.shift_curve(ph[0], ph[1], off)}              # physical + on-sky fitted delta_f
    return cv


def select_isolated(full):
    """Good point sources (bright, unflagged) with NO comparably-bright neighbour within ISO_ARCSEC.
    Neighbour check uses the full detection list so that sources SExtractor split into separate objects
    (not blend-flagged) but sitting inside the 10" aperture are still caught."""
    good = full[(full.FLAGS <= 0) & (full.CLASS_STAR >= 0.8) & (full.snr > SNR_MIN) & (full.ierr < 0.05)].reset_index(drop=True)
    if len(good) < 15 or len(full) < 2:
        return good.iloc[[]]
    allc = SkyCoord(full.ALPHA_J2000.to_numpy(), full.DELTA_J2000.to_numpy(), unit="deg")
    gc = SkyCoord(good.ALPHA_J2000.to_numpy(), good.DELTA_J2000.to_numpy(), unit="deg")
    ig, ia, sep2d, _ = search_around_sky(gc, allc, ISO_ARCSEC * u.arcsec)
    other = sep2d.arcsec > 0.5                                       # drop self-match
    bright = full.inst.to_numpy()[ia] < good.inst.to_numpy()[ig] + 2.5   # neighbour within 2.5 mag (<=10x fainter)
    contam = np.unique(ig[other & bright])
    keep = np.setdiff1d(np.arange(len(good)), contam)
    return good.iloc[keep].reset_index(drop=True)


def frame_zp(syn, inst):
    d = np.asarray(syn) - np.asarray(inst)
    d = d[np.isfinite(d)]
    if len(d) < 8:
        return np.nan
    return float(np.median(sigma_clip(d, sigma=3, maxiters=5).compressed()))


def get_lamost_spec(obsid):
    cache = OUT / "spec" / f"{obsid}.fits"
    if not cache.exists():
        r = requests.get(f"http://dr7.lamost.org/v1.3/spectrum/fits/{obsid}", timeout=60)
        raw = gzip.decompress(r.content) if r.content[:2] == b"\x1f\x8b" else r.content
        cache.write_bytes(raw)
    dd = fits.open(cache)[0].data
    return dd[2].astype(float) / 10.0, dd[0].astype(float)     # wave nm, flux (relative)


def process_tile(tile, cv):
    """Return DataFrame: one row per (ref_id, band) with inst + calibrated mag per curve, and ra/dec."""
    t = TILES[TILES.id == tile]
    if not len(t):
        return None
    ra0, dec0 = float(t.ra.iloc[0]), float(t.dec.iloc[0])
    got = K.get_reference("GAIAXP_CORR_LAMOST", tile, ra0, dec0, 1.34, 0.90)
    if got is None:
        return None
    tbl, coords, _ = got
    rid_all = np.asarray(tbl["id"]).astype(str)
    frames = [l.strip() for l in ENUM.read_text().splitlines() if f"/{tile}/" in l]
    by_band = {}
    for fr in frames:
        b = fr.split("/")[-3]
        if b in GRID:
            by_band.setdefault(b, []).append(fr)
    rows = []
    need = set()
    for b, frs in by_band.items():
        # best frame = most good stars
        best = None
        for fr in frs:
            cat = fr[:-5] + "_cat.fits"
            if not Path(cat).exists():
                continue
            try:
                d = select_isolated(read_cat_aper(cat))
            except Exception:                                  # noqa: BLE001
                continue
            if best is None or len(d) > len(best[1]):
                best = (fr, d.reset_index(drop=True))
        if best is None:
            continue
        fr, d = best
        if len(d) < 15:
            continue
        zp_hdr = float(fits.getheader(fr).get(f"ZP_{APER}", np.nan))
        src = SkyCoord(d.ALPHA_J2000.to_numpy(), d.DELTA_J2000.to_numpy(), unit="deg")
        idx, d2d, _ = match_coordinates_sky(src, coords)
        ok = np.asarray(d2d.arcsec < 2.0)
        if ok.sum() < 15:
            continue
        di = d[ok].reset_index(drop=True); mid = rid_all[idx[ok]]
        for k in range(len(di)):
            rows.append((b, mid[k], float(di.inst[k]), float(di.ALPHA_J2000[k]), float(di.DELTA_J2000[k]), zp_hdr))
            need.add(mid[k])
    if not rows:
        return None
    M = pd.DataFrame(rows, columns=["band", "ref_id", "inst", "ra", "dec", "zp_hdr"])
    xp = C.tile_xp(tile, need)                                 # {id: (orig, cor)}
    M = M[M.ref_id.isin(xp)].reset_index(drop=True)
    if not len(M):
        return None
    # per band: frame ZP per curve, then calibrated mag
    for cn in CURVE_NAMES:
        M[f"cal_{cn}"] = np.nan
    for b, g in M.groupby("band"):
        fc = np.vstack([xp[i][1] for i in g.ref_id])           # (n,343) corrected XP flux
        inst = g.inst.to_numpy()
        for cn in CURVE_NAMES:
            fw, ft = cv[b][cn]
            syn = R.syn_mag(XP_WAVE, fc, fw, ft, 0.0)
            zp = frame_zp(syn, inst)
            M.loc[g.index, f"cal_{cn}"] = inst + zp
    M["cal_header"] = M.inst + M.zp_hdr                         # old-pipeline baseline (header ZP)
    M["tile"] = tile
    return M


def sample_equatorial_tiles(n, seed=11):
    """Observed tiles with dec in [0, 19.5] (the 7DT x LAMOST overlap band)."""
    tset = TILES.set_index("id")
    obs = sorted({l.split("/")[-4] for l in ENUM.read_text().splitlines()
                  if l.strip() and l.split("/")[-3] in GRID})
    eq = [t for t in obs if t in tset.index and 0 < tset.loc[t, "dec"] < 19.5]
    rng = np.random.default_rng(seed)
    return list(rng.choice(eq, size=min(n, len(eq)), replace=False))


def main():
    global APER, SNR_MIN, ISO_ARCSEC, LAMOST_SNR
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", nargs="*", default=None)
    ap.add_argument("--n-tiles", type=int, default=40)
    ap.add_argument("--aper", type=int, default=5)
    ap.add_argument("--snr-min", type=float, default=SNR_MIN)
    ap.add_argument("--iso-arcsec", type=float, default=ISO_ARCSEC)
    ap.add_argument("--lamost-snr", type=float, default=LAMOST_SNR)
    a = ap.parse_args()
    APER = a.aper; SNR_MIN = a.snr_min; ISO_ARCSEC = a.iso_arcsec; LAMOST_SNR = a.lamost_snr
    print(f"cuts: aper={APER} snr>{SNR_MIN} iso>{ISO_ARCSEC}\" lamost_snr>{LAMOST_SNR}", flush=True)
    K._refit_init()
    cv = build_curves()
    tiles = a.tiles if a.tiles else sample_equatorial_tiles(a.n_tiles)
    print(f"{len(tiles)} tiles, aper {APER}", flush=True)

    # 1) per tile: calibrated 7DT SED per curve, keyed by ref_id
    allM = []
    for tile in tiles:
        M = process_tile(tile, cv)
        if M is not None:
            allM.append(M)
            print(f"  {tile}: {M.ref_id.nunique()} stars, {len(M)} star-bands", flush=True)
    M = pd.concat(allM, ignore_index=True)

    # 2) LAMOST crossmatch per tile (unique star positions)
    star = M.groupby(["tile", "ref_id"]).agg(ra=("ra", "median"), dec=("dec", "median")).reset_index()
    Vizier.ROW_LIMIT = -1
    star["obsid"] = -1; star["subclass"] = ""; star["snrr"] = np.nan
    for tile, g in star.groupby("tile"):
        t = TILES[TILES.id == tile].iloc[0]
        try:
            lam = Vizier(columns=["ObsID", "RAJ2000", "DEJ2000", "Class", "subClass", "snrr"]).query_region(
                f"{t.ra} {t.dec}", radius="0.8 deg", catalog="V/156/dr7lrs")[0].to_pandas()
        except Exception:
            continue
        lam = lam[(lam["Class"] == "STAR") & (lam.snrr > LAMOST_SNR)]
        if not len(lam):
            continue
        lc = SkyCoord(lam.RAJ2000.to_numpy(), lam.DEJ2000.to_numpy(), unit="deg")
        gc = SkyCoord(g.ra.to_numpy(), g.dec.to_numpy(), unit="deg")
        idx, sep, _ = match_coordinates_sky(gc, lc)
        ok = sep.arcsec < 2.0
        star.loc[g.index[ok], "obsid"] = lam.ObsID.to_numpy()[idx[ok]]
        star.loc[g.index[ok], "subclass"] = lam.subClass.to_numpy()[idx[ok]]
        star.loc[g.index[ok], "snrr"] = lam.snrr.to_numpy()[idx[ok]]
    matched = star[star.obsid > 0].reset_index(drop=True)
    print(f"\n{len(matched)} stars matched to LAMOST across {matched.tile.nunique()} tiles", flush=True)

    # 3) per matched star: residual per band per curve
    resid_rows = []       # (pivot, curve, resid_mmag)
    star_rms = []         # (obsid, subclass, curve, rms_mmag, nband)
    for _, s in matched.iterrows():
        try:
            wl_nm, flux = get_lamost_spec(int(s.obsid))
        except Exception as e:                                 # noqa: BLE001
            print("spec fail", s.obsid, e); continue
        sub = M[(M.tile == s.tile) & (M.ref_id == s.ref_id)]
        bands = sub.band.tolist()
        for cn in ALL_CAL:
            cal = sub[f"cal_{cn}"].to_numpy()
            lsyn = np.array([R.syn_mag(wl_nm, flux, *cv[b][SYNTH_CURVE[cn]], 0.0) for b in bands])
            good = np.isfinite(cal) & np.isfinite(lsyn)
            if good.sum() < 8:
                continue
            grey = np.median((cal - lsyn)[good])
            r = (cal - lsyn - grey)
            for b, rv, gd in zip(bands, r, good):
                if gd:
                    resid_rows.append((int(b[1:4]), cn, 1000 * rv))
            star_rms.append((int(s.obsid), s.subclass, cn, round(1000 * np.std(r[good]), 1), int(good.sum())))
    RS = pd.DataFrame(star_rms, columns=["obsid", "subclass", "curve", "rms_mmag", "nband"])
    RB = pd.DataFrame(resid_rows, columns=["pivot", "curve", "resid_mmag"])
    tag = f"aper{APER}_snr{int(SNR_MIN)}_iso{int(ISO_ARCSEC)}"
    RS.to_csv(OUT / f"refcat_starrms_{tag}.csv", index=False)
    RB.to_csv(OUT / f"refcat_bandresid_{tag}.csv", index=False)

    print("\n=== median per-star band-residual rms [mmag] by calibration ===")
    print(RS.groupby("curve").rms_mmag.median().reindex(ALL_CAL).round(1).to_string())

    # 4) figure: per-band mean residual vs wavelength, one line per calibration
    apstr = {3: '3"', 5: '10"'}.get(APER, f"APER{APER}")
    fig, ax = plt.subplots(2, 1, figsize=(13, 9))
    for cn in ALL_CAL:
        g = RB[RB.curve == cn].groupby("pivot").resid_mmag
        mean = g.mean(); sem = g.std() / np.sqrt(g.count())
        ax[0].errorbar(mean.index, mean.values, yerr=sem.values, fmt="o-", ms=4, lw=1, color=CCOL[cn],
                       capsize=2, label=f"{cn} (|mean| med {np.median(np.abs(mean)):.1f} mmag)")
    ax[0].axhline(0, color="k", lw=0.5); ax[0].set(xlabel="pivot wavelength [nm]", ylabel="mean(7DT_cal - LAMOST_syn) [mmag]")
    ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)
    ax[0].set_title(f"Per-band residual: 7DT (re-cal from corrected XP × curve) vs LAMOST DR7 — {len(matched)} stars, "
                    f"{apstr} aper, SNR>{int(SNR_MIN)}, isolated>{int(ISO_ARCSEC)}\".  header = old-pipeline ZP")
    # per-calibration star-rms distribution
    for cn in ALL_CAL:
        v = RS[RS.curve == cn].rms_mmag.to_numpy()
        ax[1].hist(v, bins=np.arange(0, 200, 12), histtype="step", lw=1.8, color=CCOL[cn],
                   label=f"{cn}: median {np.median(v):.1f}")
    ax[1].set(xlabel="per-star band-residual rms [mmag]", ylabel="stars"); ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / f"F26_refcat_lamost_{tag}.png", dpi=140)
    print(f"\nsaved -> F26_refcat_lamost_{tag}.png, refcat_starrms/bandresid_{tag} CSVs in {OUT}")


if __name__ == "__main__":
    main()
