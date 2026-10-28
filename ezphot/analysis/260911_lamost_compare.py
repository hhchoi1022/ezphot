#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260911_lamost_compare.py

Demonstrate comparing the 7DT medium-band SED with a real LAMOST DR7 spectrum for stars observed by both.
For one equatorial tile: assemble multi-band 7DT photometry (MAG_APER_3 + header ZP per band, AB), crossmatch to
LAMOST DR7 (VizieR V/156/dr7lrs), download the LAMOST spectrum (dr7.lamost.org, no login, 370-906 nm), and
overplot the 7DT SED on the LAMOST spectrum + synthesise the LAMOST spectrum through the nominal/physfit curves.

Read-only on the pipeline; outputs to /home/hhchoi1022/ezphot/log/20260911_lamost/.

  MPLCONFIGDIR=/tmp/ezphot-matplotlib /home/hhchoi1022/miniconda3/envs/too/bin/python \\
      ezphot/analysis/260911_lamost_compare.py --tile T22626 --n-stars 6
"""
from __future__ import annotations

import argparse
import gzip
import importlib
import io
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402
import pandas as pd                 # noqa: E402
import requests                     # noqa: E402
from astropy.coordinates import SkyCoord, match_coordinates_sky   # noqa: E402
from astropy.io import fits         # noqa: E402
import astropy.units as u           # noqa: E402
from astroquery.vizier import Vizier                              # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
R = importlib.import_module("260908_zpcolor_shift_recal")
C = importlib.import_module("260911_colorterm_2026")

ENUM = Path("/home/hhchoi1022/ezphot/log/20260911_colorterm_timeseries/all_medium_2025.txt")
OUT = Path("/home/hhchoi1022/ezphot/log/20260911_lamost"); (OUT / "spec").mkdir(parents=True, exist_ok=True)
PHYS = Path("/home/hhchoi1022/ezphot/log/20260910_transmission_sources/total_curves_physical")
FIT = pd.read_csv("/home/hhchoi1022/ezphot/log/20260910_effective_curves/offset_plus_linear_summary.csv").set_index("filter")
GRID = R.GRID
AB0 = 3.631e-20                     # erg/s/cm^2/Hz (3631 Jy)
C_AA = 2.99792458e18               # c in AA/s


APER = 3   # SExtractor fixed-aperture index: 3 = 3", 5 = 10"; overridden by --aper


def read_cat_full(catpath):
    with fits.open(catpath) as h:
        d = h["LDAC_OBJECTS"].data
        out = {c: np.asarray(d[c], float) for c in ["ALPHA_J2000", "DELTA_J2000", "FLAGS", "CLASS_STAR"]}
        out["MAG"] = np.asarray(d[f"MAG_APER_{APER}"], float)
        out["MAGERR"] = np.asarray(d[f"MAGERR_APER_{APER}"], float)
        out["SNR"] = np.asarray(d[f"SNR_APER_{APER}"], float)
        return pd.DataFrame(out)


def band_zp(fr):
    return float(fits.getheader(fr).get(f"ZP_{APER}", np.nan))


def pivot_aa(f):
    d = pd.read_csv(PHYS / f"{f}.csv"); lam = d.lam.to_numpy(); tr = d.trans.to_numpy()
    return 10.0 * np.sqrt(np.trapezoid(tr * lam, lam) / np.trapezoid(tr / lam, lam))   # nm->AA pivot wavelength


def load_curve_fine(f, shift_nm=0.0):
    d = pd.read_csv(PHYS / f"{f}.csv")
    lam = d.lam.to_numpy(); tr = np.interp(d.lam.to_numpy() - shift_nm, d.lam.to_numpy(), d.trans.to_numpy(), left=0, right=0)
    return lam, tr


def syn_ab(wave_aa, flam, curve_nm, curve_tr):
    """AB mag of a spectrum (flam, arbitrary scale) through a curve; returns mag up to the unknown LAMOST scale."""
    wl_nm = wave_aa / 10.0
    t = np.interp(wl_nm, curve_nm, curve_tr, left=0, right=0)
    fnu = flam * (wave_aa ** 2) / C_AA          # flam(/AA) -> fnu (same arbitrary scale)
    num = np.trapezoid(fnu * t / wl_nm, wl_nm); den = np.trapezoid(t / wl_nm, wl_nm)
    return -2.5 * np.log10(num / den) if num > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", default="T22626"); ap.add_argument("--n-stars", type=int, default=6)
    ap.add_argument("--aper", type=int, default=3, help='SExtractor fixed-aperture index: 3=3", 5=10"')
    a = ap.parse_args()
    global APER; APER = a.aper
    frames = [l.strip() for l in ENUM.read_text().splitlines() if f"/{a.tile}/" in l]
    by_band = {}
    for fr in frames:
        b = fr.split("/")[-3]
        if b in GRID:
            by_band.setdefault(b, []).append(fr)
    bands = sorted(by_band, key=lambda f: int(f[1:4]))
    print(f"{a.tile}: {len(bands)} bands {bands}")

    # ---- assemble 7DT multiband photometry (one good frame per band) ----
    perband = {}
    for b in bands:
        best = None
        for fr in by_band[b]:
            zp = band_zp(fr)
            if not np.isfinite(zp):
                continue
            d = read_cat_full(fr[:-5] + "_cat.fits")
            d = d[(d.FLAGS <= 0) & (d.CLASS_STAR >= 0.8) & (d.SNR > 20) & (d.MAGERR < 0.05)]
            if best is None or len(d) > len(best[1]):
                best = (zp, d.reset_index(drop=True))
        if best:
            z, d = best
            d["mag_ab"] = d.MAG + z
            perband[b] = d
    ref_b = bands[len(bands) // 2]
    base = perband[ref_b][["ALPHA_J2000", "DELTA_J2000"]].copy()
    coords = SkyCoord(base.ALPHA_J2000.to_numpy(), base.DELTA_J2000.to_numpy(), unit="deg")
    sed = {b: np.full(len(base), np.nan) for b in bands}
    for b in bands:
        d = perband[b]; c2 = SkyCoord(d.ALPHA_J2000.to_numpy(), d.DELTA_J2000.to_numpy(), unit="deg")
        idx, sep, _ = match_coordinates_sky(coords, c2)
        ok = sep.arcsec < 1.5
        sed[b][ok] = d.mag_ab.to_numpy()[idx[ok]]
    nband = np.sum([np.isfinite(sed[b]) for b in bands], axis=0)
    base["nband"] = nband; base = base[base.nband >= max(8, len(bands) - 4)]
    print(f"stars with >= {max(8,len(bands)-4)} bands: {len(base)}")

    # ---- LAMOST crossmatch ----
    ra0, dec0 = float(base.ALPHA_J2000.median()), float(base.DELTA_J2000.median())
    Vizier.ROW_LIMIT = -1
    lam = Vizier(columns=["ObsID", "RAJ2000", "DEJ2000", "Class", "subClass", "snrg", "snrr"]).query_region(
        f"{ra0} {dec0}", radius="0.75 deg", catalog="V/156/dr7lrs")[0].to_pandas()
    lam = lam[(lam["Class"] == "STAR") & (lam.snrr > 10)]
    lc = SkyCoord(lam.RAJ2000.to_numpy(), lam.DEJ2000.to_numpy(), unit="deg")
    bc = SkyCoord(base.ALPHA_J2000.to_numpy(), base.DELTA_J2000.to_numpy(), unit="deg")
    idx, sep, _ = match_coordinates_sky(bc, lc)
    base["lam_ok"] = sep.arcsec < 2.0; base["obsid"] = lam.ObsID.to_numpy()[idx]
    base["subclass"] = lam.subClass.to_numpy()[idx]; base["snrr"] = lam.snrr.to_numpy()[idx]
    base["lam_sep"] = sep.arcsec
    m = base[base.lam_ok].sort_values("snrr", ascending=False)
    print(f"7DT stars matched to LAMOST spectra: {len(m)}")

    # curves
    cv = {b: {"nominal": load_curve_fine(b, 0.0),
              "physfit": load_curve_fine(b, float(FIT.loc[b, "offset"]) if b in FIT.index else 0.0)} for b in bands}
    piv = {b: pivot_aa(b) for b in bands}

    fig, axes = plt.subplots(min(a.n_stars, len(m)), 1, figsize=(13, 3.0 * min(a.n_stars, len(m))))
    if min(a.n_stars, len(m)) == 1:
        axes = [axes]
    rows = []
    for ax, (_, s) in zip(axes, m.head(a.n_stars).iterrows()):
        obsid = int(s.obsid)
        cache = OUT / "spec" / f"{obsid}.fits"
        try:
            if not cache.exists():
                r = requests.get(f"http://dr7.lamost.org/v1.3/spectrum/fits/{obsid}", timeout=60)
                raw = gzip.decompress(r.content) if r.content[:2] == b"\x1f\x8b" else r.content
                cache.write_bytes(raw)
            hd = fits.open(cache)
            dd = hd[0].data; flux = dd[0].astype(float); wave = dd[2].astype(float)
        except Exception as e:                     # noqa: BLE001
            print("spec fail", obsid, e); continue
        # 7DT SED points
        wl = np.array([piv[b] for b in bands]); mag = np.array([sed[b][s.name] for b in bands])
        good = np.isfinite(mag)
        # LAMOST relative flux -> relative AB mag per pixel; grey-shift so band-synth matches the 7DT mags
        posf = flux > 0
        mag_lam = np.full_like(wave, np.nan)
        mag_lam[posf] = -2.5 * np.log10(flux[posf] * wave[posf] ** 2 / C_AA)
        syn_nom = np.array([syn_ab(wave, flux, *cv[b]["nominal"]) for b in bands])
        off = np.nanmedian(mag[good] - syn_nom[good])
        ax.plot(wave[posf], mag_lam[posf] + off, color="#888", lw=0.5, alpha=0.8,
                label=f"LAMOST {s.subclass} (obsid {obsid}, snr {s.snrr:.0f})")
        ax.plot(wl[good], mag[good], "o", color="#d62728", ms=7, label="7DT medium-band SED")
        # residuals per band (7DT - LAMOST synth), nominal vs physfit
        for cn in ("nominal", "physfit"):
            syn_c = np.array([syn_ab(wave, flux, *cv[b][cn]) for b in bands])
            o = np.nanmedian(mag[good] - syn_c[good])
            resid = (mag - syn_c - o)
            rows.append(dict(obsid=obsid, subclass=s.subclass, curve=cn,
                             rms_mmag=round(1000 * np.nanstd(resid[good]), 1)))
        lo, hi = np.nanmin(mag[good]), np.nanmax(mag[good])
        ax.set(xlim=(3700, 9100), ylim=(hi + 0.7, lo - 0.7), ylabel="AB mag")
        ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3)
    axes[-1].set_xlabel("wavelength [AA]")
    apstr = {3: '3"', 5: '10"'}.get(APER, f"APER{APER}")
    fig.suptitle(f"7DT medium-band SED ({apstr} aper) vs LAMOST DR7 spectrum — {a.tile} ({len(m)} matched stars)")
    fig.tight_layout(); fig.savefig(OUT / f"lamost_sed_{a.tile}_aper{APER}.png", dpi=150)
    rr = pd.DataFrame(rows)
    rr.to_csv(OUT / f"lamost_resid_{a.tile}_aper{APER}.csv", index=False)
    print("\nper-star band-residual rms (7DT - LAMOST synth), nominal vs physfit:")
    if len(rr):
        print(rr.pivot_table(index=["obsid", "subclass"], columns="curve", values="rms_mmag").to_string())
        print("\nmedian band-residual rms [mmag]:  " +
              "  ".join(f"{cn}={rr[rr.curve == cn].rms_mmag.median():.1f}" for cn in ("nominal", "physfit")))
    print(f"\nsaved -> lamost_sed_{a.tile}_aper{APER}.png, lamost_resid_{a.tile}_aper{APER}.csv in {OUT}")


if __name__ == "__main__":
    main()
