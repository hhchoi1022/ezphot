#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260910_atmosphere_elsauce.py

Build an average-conditions atmospheric transmission curve for 7DT at El Sauce (~1580 m) and compare it with
the curve used so far (Original_sources/skytable.fits = ESO SkyCalc, Paranal 2640 m, airmass 1.3, PWV 2.5 mm).

Recipe:
  1. ESO SkyCalc API runs at the closest available site (La Silla, 2400 m): airmass X = 1.23 (the reference the
     effective-curve work uses, X0), PWV = 7.5 mm (fixed by decision, 2026-09-10) with PWV = 5.0 / 10.0 mm as a
     variability bracket.
  2. Altitude rescale 2400 m -> 1580 m (pressure ratio 839/761 = 1.102) applied to the pressure-proportional
     components only: Rayleigh (T_rs ** 1.102, exact for a continuum) and the O2 bands inside trans_ma
     (gamma 627-632, B 686-697, A 758-772 nm; power scaling slightly overestimates saturated-line growth).
     H2O is set by the PWV parameter; ozone and aerosol are left at the SkyCalc site climatology.
  3. Output: skytable_elsauce_pwv*.fits (raw SkyCalc), atmosphere_elsauce.csv (lam_vac, lam_air [Edlen], components,
     total for the three PWV cases), figure F9_atmosphere_elsauce.png, per-band table atmosphere_band_effects.csv.

  MPLCONFIGDIR=/tmp/ezphot-matplotlib /home/hhchoi1022/miniconda3/envs/too/bin/python \\
      ezphot/analysis/260910_atmosphere_elsauce.py [--skip-download]
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402
import pandas as pd                 # noqa: E402
import requests                     # noqa: E402
from astropy.io import fits         # noqa: E402

SRC = Path("~/ezphot/config/common/transmission/Original_sources").expanduser()
OUT = Path("~/ezphot/log/20260910_transmission_sources").expanduser()
SERVER = "https://etimecalret-002.eso.org"

X_REF, PWV_MAIN, PWV_BRACKET = 1.23, 7.5, (5.0, 10.0)
P_ELSAUCE, P_LASILLA = 839.0, 761.0            # hPa, P = 1013.25 exp(-h/8400), h = 1580 / 2400 m
R_P = P_ELSAUCE / P_LASILLA                     # 1.102
O2_BANDS = [(627.0, 632.5), (686.0, 697.0), (758.0, 772.0)]


def log(msg):
    print(time.strftime("[%Y-%m-%d %H:%M:%S] ") + msg, flush=True)


def skycalc(pwv, out_fits):
    params = dict(observatory="lasilla", airmass=X_REF, pwv_mode="pwv", pwv=pwv, msolflux=130.0,
                  incl_moon="Y", moon_sun_sep=90.0, moon_target_sep=45.0, moon_alt=45.0, moon_earth_dist=1.0,
                  incl_starlight="Y", incl_zodiacal="Y", ecl_lon=135.0, ecl_lat=90.0,
                  incl_loweratm="Y", incl_upperatm="Y", incl_airglow="Y", incl_therm="N",
                  therm_t1=0.0, therm_e1=0.0, therm_t2=0.0, therm_e2=0.0, therm_t3=0.0, therm_e3=0.0,
                  vacair="vac", wmin=300.0, wmax=1100.0, wgrid_mode="fixed_spectral_resolution",
                  wdelta=0.1, wres=20000, lsf_type="none", lsf_gauss_fwhm=5.0, lsf_boxcar_fwhm=5.0)
    r = requests.post(SERVER + "/observing/etc/api/skycalc", data=json.dumps(params), timeout=300)
    r.raise_for_status()
    res = r.json()
    assert res["status"] == "success", res
    f = requests.get(SERVER + "/observing/etc/tmp/" + res["tmpdir"] + "/skytable.fits", timeout=300)
    f.raise_for_status()
    out_fits.write_bytes(f.content)
    log(f"SkyCalc lasilla X={X_REF} PWV={pwv} -> {out_fits.name} ({len(f.content)/1e6:.1f} MB)")


def load_sky(path):
    with fits.open(path) as h:
        return pd.DataFrame({c: np.asarray(h[1].data[c], dtype="f8")
                             for c in ["lam", "trans", "trans_ma", "trans_o3", "trans_rs", "trans_ms"]})


def to_elsauce(s):
    """Altitude rescale La Silla -> El Sauce on the pressure-proportional components."""
    rs = s.trans_rs ** R_P
    ma = s.trans_ma.copy()
    for lo, hi in O2_BANDS:
        m = (s.lam >= lo) & (s.lam <= hi)
        ma[m] = ma[m] ** R_P
    total = rs * s.trans_o3 * s.trans_ms * ma
    return pd.DataFrame(dict(lam=s.lam, trans=total, trans_rs=rs, trans_o3=s.trans_o3,
                             trans_ms=s.trans_ms, trans_ma=ma))


def vac_to_air(lam_nm):
    s2 = (1e3 / lam_nm) ** 2                    # (1/um)^2
    n = 1 + (8342.54 + 2406147.0 / (130.0 - s2) + 15998.0 / (38.9 - s2)) * 1e-8
    return lam_nm / n


def smooth(lam, tr, width_nm=2.0):
    n = max(1, int(width_nm / np.median(np.diff(lam))))
    return pd.Series(tr).rolling(n, center=True, min_periods=1).median().to_numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-download", action="store_true")
    a = ap.parse_args()

    cases = {p: OUT / f"skytable_elsauce_pwv{p}.fits" for p in [PWV_MAIN, *PWV_BRACKET]}
    if not a.skip_download:
        for pwv, f in cases.items():
            if not f.exists():
                skycalc(pwv, f)

    old = load_sky(SRC / "skytable.fits")                          # Paranal 2640 m, X=1.3, PWV 2.5
    new = {pwv: to_elsauce(load_sky(f)) for pwv, f in cases.items()}
    main_new = new[PWV_MAIN]
    lo, hi = PWV_BRACKET

    # combined CSV (vacuum + air wavelengths)
    dfo = pd.DataFrame(dict(lam_vac=main_new.lam, lam_air=vac_to_air(main_new.lam),
                            trans=main_new.trans, trans_rs=main_new.trans_rs, trans_o3=main_new.trans_o3,
                            trans_ms=main_new.trans_ms, trans_ma=main_new.trans_ma))
    dfo[f"trans_pwv{str(lo).replace('.', 'p')}"] = np.interp(main_new.lam, new[lo].lam, new[lo].trans)
    dfo[f"trans_pwv{str(hi).replace('.', 'p')}"] = np.interp(main_new.lam, new[hi].lam, new[hi].trans)
    dfo.to_csv(OUT / "atmosphere_elsauce.csv", index=False)

    # per-band effects: weight = vendor filter curve only
    filt_files = sorted(p for p in SRC.glob("m*.csv") if re.fullmatch(r"m\d{3}w?", p.stem))
    grid = old.lam[(old.lam >= 320) & (old.lam <= 1080)].to_numpy()
    t_old = np.interp(grid, old.lam, old.trans)
    t_new = np.interp(grid, main_new.lam, main_new.trans)
    t_lo = np.interp(grid, new[lo].lam, new[lo].trans)
    t_hi = np.interp(grid, new[hi].lam, new[hi].trans)
    rows = []
    for p in filt_files:
        d = pd.read_csv(p); lam = d.iloc[:, 0].to_numpy(float); tr = d.iloc[:, 1].to_numpy(float)
        o = np.argsort(lam); lam, tr = lam[o], tr[o]
        F = np.interp(grid, lam, tr, left=0, right=0)
        if F.max() < 0.1: continue
        w_old, w_new = F * t_old, F * t_new
        piv = np.trapz(grid * F, grid) / np.trapz(F, grid)
        dm = -2.5 * np.log10(np.trapz(w_new, grid) / np.trapz(w_old, grid))
        dm_lo = -2.5 * np.log10(np.trapz(F * t_lo, grid) / np.trapz(w_old, grid))
        dm_hi = -2.5 * np.log10(np.trapz(F * t_hi, grid) / np.trapz(w_old, grid))
        dleff = (np.trapz(grid * w_new, grid) / np.trapz(w_new, grid)
                 - np.trapz(grid * w_old, grid) / np.trapz(w_old, grid))
        rows.append(dict(filter=p.stem, pivot_nm=round(piv, 1), dmag=round(dm, 4),
                         dmag_pwv_lo=round(dm_lo, 4), dmag_pwv_hi=round(dm_hi, 4), dlam_eff_nm=round(dleff, 3)))
    band = pd.DataFrame(rows).sort_values("pivot_nm")
    band.to_csv(OUT / "atmosphere_band_effects.csv", index=False)
    print(band.to_string(index=False))

    # ---- figure ----
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    ax = axes[0]
    sel = (old.lam >= 320) & (old.lam <= 1080)
    ax.plot(old.lam[sel], old.trans[sel], lw=0.25, color="#999")
    ax.plot(grid, smooth(grid, t_old), lw=2, color="k", label="old: Paranal 2640 m, X=1.30, PWV 2.5 (2 nm median)")
    ax.fill_between(grid, smooth(grid, t_lo), smooth(grid, t_hi), color="#2ca02c", alpha=0.18,
                    label=f"new: PWV {lo}-{hi} bracket")
    ax.plot(grid, smooth(grid, t_new), lw=2, color="#2ca02c",
            label=f"new: El Sauce 1580 m, X={X_REF}, PWV {PWV_MAIN} (2 nm median)")
    ax.set(xlim=(320, 1080), ylim=(0.3, 1.02), ylabel="transmission", title="F9  atmosphere: old vs new (El Sauce)")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.axhline(1, color="k", lw=0.5)
    ax.plot(grid, smooth(grid, t_new / t_old), lw=1.5, color="#2ca02c", label="new / old (2 nm median)")
    ax.fill_between(grid, smooth(grid, t_lo / t_old), smooth(grid, t_hi / t_old), color="#2ca02c", alpha=0.18)
    ax.set(xlim=(320, 1080), ylim=(0.85, 1.1), ylabel="ratio", xlabel="")
    ax.legend(loc="lower left"); ax.grid(alpha=0.3)

    ax = axes[2]
    colors = plt.get_cmap("turbo")((band.pivot_nm - 370) / (890 - 370))
    ax.bar(band.pivot_nm, band.dmag * 1000, width=16, color=colors, edgecolor="k", lw=0.4)
    ax.errorbar(band.pivot_nm, band.dmag * 1000,
                yerr=[(band.dmag - band.dmag_pwv_lo).abs() * 1000,
                      (band.dmag_pwv_hi - band.dmag).abs() * 1000], fmt="none", ecolor="#555", lw=1, capsize=2)
    ax.axhline(0, color="k", lw=0.5)
    ax.set(xlim=(340, 960), xlabel="wavelength [nm]", ylabel="band d(mag) [mmag]",
           title=f"band-integrated change old -> new (bars; whiskers = PWV {lo}-{hi}); grey part is absorbed by per-band ZP")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "F9_atmosphere_elsauce.png", dpi=150)
    log(f"saved -> {OUT/'figures'/'F9_atmosphere_elsauce.png'}, atmosphere_elsauce.csv, atmosphere_band_effects.csv")


if __name__ == "__main__":
    main()
