#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260910_effcurve_common.py

Shared pieces of the effective-transmission-curve estimation (plan of 2026-09-09):

  T_eff,f(lambda) = Beam[F_f](lambda - delta_f) * S(lambda)

* F_f  : filter-only curve (/qso/data3/doabdi/intern/synthetic/filter_only, nm, 0.2 nm), assumed normal incidence
* S    : filter-independent system factor (QE x optics x atmosphere at an unknown airmass X0), recovered as
         nominal / filter_only over all 42 filters and decomposed with the camera QE, the optics efficiency and an
         ESO SkyCalc atmosphere (7DT-Simulator files) to estimate X0
* Beam : angle average over the f/3 converging beam with central obstruction eps (user: 0.59 by diameter);
         an interference filter at incidence theta transmits T0(lambda / sqrt(1 - sin^2 theta / n_eff^2))
* rigid: the alternative, simpler family F_f(lambda - Delta)

Everything is read-only with respect to ezphot; the synthetic-photometry recipe is imported from
260908_zpcolor_shift_recal.py (syn_mag: photon-weighted trapezoid on the spectrum grid, AB 3631 Jy).
"""
from __future__ import annotations

import importlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
R = importlib.import_module("260908_zpcolor_shift_recal")

PREV_OUT = Path("/home/hhchoi1022/ezphot/log/20260908_zp_color_check")          # products of the 2026-09-08 analysis
DEFAULT_OUT = Path("/home/hhchoi1022/ezphot/log/20260910_effective_curves")
REPO_CURVES = Path(__file__).resolve().parent / "effective_curves"
FILTER_ONLY_DIR = Path("/qso/data3/doabdi/intern/synthetic/filter_only")
NOMINAL_DIR = R.TRANSMISSION_DIR                                                  # Final_transmission (nm, 0.1 nm)
QE_FILE = Path("/lyman/data1/Calibration/7DT-Simulator/conf/QE.csv")
OPTICS_FILE = Path("/lyman/data1/Calibration/7DT-Simulator/util/optics.efficiency.dr350.dr500.csv")
SKYTABLE = Path("/lyman/data1/Calibration/7DT-Simulator/util/skytable.fits")     # ESO SkyCalc, airmass 1.3, PWV 2.5 mm
SKY_AIRMASS = 1.3

FINE = np.round(np.arange(300.0, 1000.0 + 1e-9, 0.1), 1)                         # nm grid of the delivered curves
MEDIUM = sorted([p.stem for p in FILTER_ONLY_DIR.glob("m*.csv") if re.match(r"^m\d{3}w?$", p.stem)],
                key=lambda s: int(re.sub(r"\D", "", s)))                          # 37 medium bands
ALL_FILTERS = MEDIUM + ["u", "g", "r", "i", "z"]
EPS_DEFAULT, FRATIO_DEFAULT = 0.59, 3.0
N_EFF_PROVISIONAL = 1.59                       # from eps 0.59 and the 2026-09-08 result dlam/lam = -0.36 %
MISLABELED = [("7DT07", "m800", 60531.0, 60546.5), ("7DT08", "m825", 60531.0, 60546.5)]   # unit, filter, MJD range


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


# --------------------------------------------------------------------------------------
# curves
# --------------------------------------------------------------------------------------
def load_filter_only(name: str):
    """Filter-only transmission on the FINE grid (nm). m466w.csv is stored in descending wavelength order."""
    d = pd.read_csv(FILTER_ONLY_DIR / f"{name}.csv")
    w = d.iloc[:, 0].to_numpy(float); t = np.clip(d.iloc[:, 1].to_numpy(float), 0, None)
    o = np.argsort(w); w, t = w[o], t[o]
    return np.interp(FINE, w, t, left=0.0, right=0.0)


def load_nominal(name: str):
    """Nominal system curve (Final_transmission) on the FINE grid."""
    w, t = R.load_curve(name)
    return np.interp(FINE, w, t, left=0.0, right=0.0)


def pivot_nm(w, t):
    t = np.clip(t, 0, None)
    return float(np.sqrt(np.trapezoid(t * w, w) / np.trapezoid(t / w, w)))


def recover_S(min_f: float = 0.3):
    """S(lambda) = nominal / filter_only, median over the filters whose filter-only transmission exceeds min_f.
    Returns S, the number of contributing filters, and the relative spread (max-min)/median per wavelength."""
    ratios = []
    for f in ALL_FILTERS:
        F = load_filter_only(f); N = load_nominal(f)
        r = np.where(F > min_f, N / np.where(F > min_f, F, 1.0), np.nan)
        ratios.append(r)
    A = np.vstack(ratios)
    n = np.sum(np.isfinite(A), axis=0)
    with np.errstate(all="ignore"):
        S = np.nanmedian(A, axis=0)
        spread = (np.nanmax(A, axis=0) - np.nanmin(A, axis=0)) / S
    return S, n, spread


def load_components():
    """Camera QE, optics efficiency and SkyCalc atmospheric transmission components on the FINE grid."""
    from astropy.io import fits
    from astropy.table import Table
    q = pd.read_csv(QE_FILE); qe = np.interp(FINE, q.iloc[:, 0].to_numpy(float), q.iloc[:, 1].to_numpy(float))
    o = Table.read(OPTICS_FILE, format="ascii.ecsv")
    ow = np.asarray(o["nm"], float); ot = np.asarray(o["total"], float)
    optics = np.interp(FINE, ow, ot)
    with fits.open(SKYTABLE) as h:
        d = h[1].data; lam = np.asarray(d["lam"], float)
        comp = {}
        for c in ("trans_rs", "trans_ms", "trans_o3", "trans_ma"):
            v = np.asarray(d[c], float)
            # box-average the R=20000 table onto the 0.1 nm grid (so that band depths are preserved)
            idx = np.clip(np.searchsorted(FINE - 0.05, lam) - 1, 0, len(FINE) - 1)
            s = np.bincount(idx, weights=v, minlength=len(FINE)); cnt = np.bincount(idx, minlength=len(FINE))
            comp[c] = np.where(cnt > 0, s / np.maximum(cnt, 1), np.interp(FINE, lam, v))
    return qe, optics, comp


def atm_transmission(comp, X: float, gamma_ma: float = 1.0):
    """Atmospheric transmission at airmass X from the SkyCalc components tabulated at airmass SKY_AIRMASS
    (Beer-Lambert scaling; molecular absorption gets an optional exponent)."""
    t = np.ones_like(FINE)
    for c in ("trans_rs", "trans_ms", "trans_o3"):
        t = t * comp[c] ** (X / SKY_AIRMASS)
    t = t * comp["trans_ma"] ** ((X / SKY_AIRMASS) ** gamma_ma)
    return t


def decompose_S(S, qe, optics, comp):
    """Fit S ~ g * QE * optics * atm(X0).  X0 from the O2 A-band equivalent width (755-775 nm, continuum from
    740-750 and 780-790), g from the 500-650 nm continuum.  Returns dict(g, X0, ew_S, ew_model(X0), resid_rms)."""
    def ew(y):
        cont_m = ((FINE >= 740) & (FINE <= 750)) | ((FINE >= 780) & (FINE <= 790))
        p = np.polyfit(FINE[cont_m], y[cont_m], 1); cont = np.polyval(p, FINE)
        band = (FINE >= 755) & (FINE <= 775)
        return float(np.trapezoid(1 - y[band] / cont[band], FINE[band]))
    ew_S = ew(S)
    Xs = np.arange(0.6, 3.01, 0.02)
    ews = np.array([ew(qe * optics * atm_transmission(comp, X)) for X in Xs])
    X0 = float(np.interp(ew_S, ews, Xs)) if ews.min() <= ew_S <= ews.max() else np.nan
    flag = "" if np.isfinite(X0) else "X0 outside 0.6-3.0: fixed to 1.3"
    if not np.isfinite(X0): X0 = SKY_AIRMASS
    model = qe * optics * atm_transmission(comp, X0)
    cont = (FINE >= 500) & (FINE <= 650)
    g = float(np.median(S[cont] / model[cont]))
    band = (FINE >= 400) & (FINE <= 900)
    with np.errstate(all="ignore"):
        resid = S[band] / (g * model[band]) - 1
    return dict(g=g, X0=X0, ew_S_nm=ew_S, ew_model_nm=float(ew(model)), resid_rms=float(np.nanstd(resid)), flag=flag), g * model


def system_S():
    """Recovered S(lambda), filled outside the filter coverage with the decomposition model, plus the decomposition
    products.  Returns S_filled, info(dict), parts(dict of FINE-grid arrays: S_raw, n, spread, qe, optics, atm_X0, model)."""
    S_raw, n, spread = recover_S()
    qe, optics, comp = load_components()
    info, model = decompose_S(S_raw, qe, optics, comp)
    S = np.where(np.isfinite(S_raw), S_raw, model)
    parts = dict(S_raw=S_raw, n=n, spread=spread, qe=qe, optics=optics, atm_X0=atm_transmission(comp, info["X0"]), model=model, comp=comp)
    return S, info, parts


# --------------------------------------------------------------------------------------
# beam model
# --------------------------------------------------------------------------------------
def beam_s_range(eps: float = EPS_DEFAULT, fratio: float = FRATIO_DEFAULT):
    s_max = np.sin(np.arctan(1.0 / (2.0 * fratio))) ** 2
    return eps ** 2 * s_max, s_max


def beam_mean_shift(n_eff: float, eps: float = EPS_DEFAULT, fratio: float = FRATIO_DEFAULT):
    """zeta = <dlambda/lambda> of the angle-averaged interference filter (negative)."""
    s_lo, s_hi = beam_s_range(eps, fratio)
    return -(s_lo + s_hi) / 2.0 / (2.0 * n_eff ** 2)


def n_eff_from_zeta(zeta: float, eps: float = EPS_DEFAULT, fratio: float = FRATIO_DEFAULT):
    s_lo, s_hi = beam_s_range(eps, fratio)
    return float(np.sqrt(-(s_lo + s_hi) / (4.0 * zeta))) if zeta < 0 else np.nan


def beam_average(F, n_eff: float, eps: float = EPS_DEFAULT, fratio: float = FRATIO_DEFAULT, n_s: int = 25):
    """Angle average of the filter-only curve F (FINE grid) for a beam whose sin^2(theta) is uniform on
    [eps^2 s_max, s_max]: T(lambda) = mean_k F(lambda / sqrt(1 - s_k / n_eff^2))."""
    s_lo, s_hi = beam_s_range(eps, fratio)
    s = s_lo + (np.arange(n_s) + 0.5) / n_s * (s_hi - s_lo)
    out = np.zeros_like(FINE)
    for sk in s:
        g = np.sqrt(1.0 - sk / n_eff ** 2)
        out += np.interp(FINE / g, FINE, F, left=0.0, right=0.0)
    return out / n_s


def shifted(curve, delta_nm: float):
    """Rigid shift of a FINE-grid curve to longer wavelengths by delta_nm (negative = blueward)."""
    return np.interp(FINE, FINE + delta_nm, curve, left=0.0, right=0.0)


class CurveFamily:
    """Curves of one filter as a function of the total effective shift Delta (nm).
    rigid: F(lambda - Delta) * S ;  beam: Beam_n[F](lambda - (Delta - zeta * pivot)) * S."""

    def __init__(self, name: str, S, n_eff: float = N_EFF_PROVISIONAL, eps: float = EPS_DEFAULT, fratio: float = FRATIO_DEFAULT):
        self.name = name; self.S = S; self.n_eff = n_eff; self.eps = eps; self.fratio = fratio
        self.F = load_filter_only(name)
        self.nominal = load_nominal(name)
        self.pivot = pivot_nm(FINE, self.nominal)
        self.zeta = beam_mean_shift(n_eff, eps, fratio)
        self.F_beam = beam_average(self.F, n_eff, eps, fratio)
        self.pivot_beam = pivot_nm(FINE, self.F_beam * S) if np.any(self.F_beam > 0) else np.nan

    def curve(self, delta: float, form: str = "rigid"):
        if form == "rigid":
            return shifted(self.F, delta) * self.S
        elif form == "beam":
            return shifted(self.F_beam, delta - self.zeta * self.pivot) * self.S
        raise ValueError(form)


# --------------------------------------------------------------------------------------
# curve I/O
# --------------------------------------------------------------------------------------
def write_curve_nm(path: Path, trans, extra_cols: dict | None = None):
    d = {"lam": FINE, "trans": np.round(np.clip(trans, 0, None), 6)}
    for k, v in (extra_cols or {}).items(): d[k] = np.round(v, 6)
    pd.DataFrame(d).to_csv(path, index=False, float_format="%.6f")


def write_curve_AA(path: Path, trans):
    """Config format: wavelength [A] 3000-11000 at 1 A, response (values < 0.001 zeroed as in the live config)."""
    w = np.arange(3000, 11001, 1.0)
    t = np.interp(w, FINE * 10.0, trans, left=0.0, right=0.0)
    t = np.where(t < 0.001, 0.0, t)
    pd.DataFrame({"wavelength": w, "response": np.round(t, 5)}).to_csv(path, index=False, float_format="%.5f")


def write_meta(path: Path, meta: dict):
    path.write_text(json.dumps(meta, indent=1, ensure_ascii=False, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o)), encoding="utf-8")


# --------------------------------------------------------------------------------------
# data selection helpers (shared by fit and validate)
# --------------------------------------------------------------------------------------
def is_mislabeled(telescop, filt, mjd):
    for u, f, m0, m1 in MISLABELED:
        if telescop == u and filt == f and m0 <= mjd <= m1: return True
    return False


def load_frames(kind: str = "single", prev_out: Path = PREV_OUT):
    """Frame headers of the 2026-09-08 harvest (single frames or coadds) with the mislabeled frames flagged."""
    p = prev_out / "frames.csv" if kind == "single" else prev_out / "coadd" / "frames.csv"
    F = pd.read_csv(p, low_memory=False)
    F["is_tile"] = F["OBJECT"].astype(str).str.match(r"^T\d{5}$")
    F["mislabeled"] = [is_mislabeled(u, f, m) for u, f, m in zip(F["TELESCOP"], F["FILTER"], F["MJD"])]
    F["bad_zperr"] = F["ZPERR_APER_3"] > R.ZPERR_ANOMALY
    return F


if __name__ == "__main__":
    # quick self-test: S recovery, decomposition and the beam model
    S, info, parts = system_S(); n, spread = parts["n"], parts["spread"]
    band = (FINE >= 350) & (FINE <= 950)
    log(f"S(lambda): contributing filters median {np.median(n[band]):.0f}, relative spread median {np.nanmedian(spread[band]):.4f}, "
        f"max {np.nanmax(spread[band]):.3f}; S(500)={S[np.argmin(np.abs(FINE-500))]:.4f}, S(761)={S[np.argmin(np.abs(FINE-761))]:.4f}")
    log(f"decomposition: {info}")
    for f in ("m475", "m675", "m875"):
        fam = CurveFamily(f, S)
        log(f"{f}: pivot {fam.pivot:.2f} nm, zeta {fam.zeta*100:+.3f} %, beam pivot {fam.pivot_beam:.2f} (shift {fam.pivot_beam-fam.pivot:+.2f} nm)")
