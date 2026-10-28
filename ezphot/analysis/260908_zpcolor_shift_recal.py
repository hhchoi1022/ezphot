#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260908_zpcolor_shift_recal.py

Closed-loop test of the transmission-curve shifts found by 260908_zpcolor_calspec_shift.py:
regenerate the reference photometry of the standard-star fields with *shifted* curves,
recompute the zero points of every standard-field frame from the same reference stars,
recompute the CALSPEC synthetic magnitudes with the same shifted curves, and see whether
the star-to-star CALSPEC residual pattern (DA white dwarfs vs the rest) collapses without
degrading the field-star diagnostics (ZP scatter, colour term).

Nothing outside the standard-star fields is touched: only the 2,129 reference stars that the
pipeline actually used (refit_stars.csv.gz, catalogue GAIAXP_CORR_LAMOST) are re-synthesised,
from their LAMOST-corrected XP spectra in /lyman/data1/factory_doabdi/GAIA_BEST/final_result.

Synthesis follows the recipe that produced the GAIAXP_CORR_LAMOST v1 archive (doabdi
syn_calib.ipynb): photon-weighted trapezoid on the spectrum's own grid, Final_transmission
curves (nm), AB reference 3631 Jy.  At zero shift it reproduces the archive to 1e-12 mag.
Only *differences* m(shift) - m(0) are applied to the existing per-star zero points and to
the existing CALSPEC synthetic magnitudes, so the zero-shift case is exactly the 2026-09-08
result and the shifted case differs only through the curve shift.

Cases per filter: 'none' (0 nm), 'c4' (best shift of 260908_zpcolor_calspec_shift.py, 0 where
undetermined), 'scan' (-5..+2 nm), and 'rec' = the consensus of four diagnostics of the scan
(field colour term = 0, minimum frame ZP scatter, DA-minus-other = 0, minimum between-star MAD).

Stages (--stage):
  extract : pull the spectra of the reference stars from the 34 tiles -> shift_recal/refstar_spectra.npz
  compute : shifts x filters -> shift_recal/recal_summary.csv, recal_calspec_stars.csv, recal_frames.csv.gz,
            recal_calspec_syn.csv, recal_recommended.csv, figures/C5_recal_overplot.png, figures/C6_recal_scan.png
  all     : both

  MPLCONFIGDIR=/tmp/ezphot-matplotlib /home/hhchoi1022/anaconda3/envs/py311/bin/python \\
      ezphot/analysis/260908_zpcolor_shift_recal.py --stage all
"""
from __future__ import annotations

import argparse
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
from astropy.table import Table          # noqa: E402

warnings.filterwarnings("ignore")

DEFAULT_OUT_DIR = Path("/home/hhchoi1022/ezphot/log/20260908_zp_color_check")
SPECTRA_DIR = Path("/lyman/data1/factory_doabdi/GAIA_BEST/final_result")   # row set of the v1 archive
TRANSMISSION_DIR = Path("/lyman/data1/factory_doabdi/Final_transmission")  # nm, used for the v1 archive
CALSPEC_DIR = Path("/qso/data3/doabdi/intern/synthetic/calspec_data")
GRID = [f"m{w}" for w in range(400, 876, 25)]
XP_WAVE_NM = np.arange(336, 1022, 2).astype(float)          # 343 samples of the XP sampled spectra
SCAN_NM = np.array([-5, -4, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 2], float)
SCAN_EDGE = 4.9                                              # optima at |shift| >= this are treated as unconstrained
CATALOG = "GAIAXP_CORR_LAMOST"
STANDARDS = ["BD+02_3375", "BD-11_3759", "FEIGE110", "GD71", "GJ754.1A", "HZ4", "LDS749B", "WD0227+050",
             "WD0308-565", "WD1544-377", "WD1919+145", "WD2039-682", "WD2149+021"]
# same cleaning as 260908_zpcolor_plot.py / _calspec_shift.py
ZPERR_ANOMALY, CAL_MAX_MAGERR, CAL_MAX_FLUX, MIN_FRAMES = 0.08, 0.05, 45000.0, 10
MAGERR_FLOOR, ERR_FLOOR = 0.01, 0.005

# AB reference spectrum in W m^-2 nm^-1 (doabdi recipe)
C_NM_S = 2.99792458e17
F_NU_AB = 3.631e-20


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def mad(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan


def weighted_linfit(x, y, sigma, nsig=3.0, maxiter=5):
    x = np.asarray(x, float); y = np.asarray(y, float); sigma = np.asarray(sigma, float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(sigma) & (sigma > 0)
    a = b = a_err = np.nan
    for _ in range(maxiter):
        if m.sum() < 5 or np.ptp(x[m]) == 0:
            return np.nan, np.nan, np.nan, int(m.sum())
        (a, b), cov = np.polyfit(x[m], y[m], 1, w=1.0 / sigma[m], cov=True)
        a_err = float(np.sqrt(cov[0, 0]))
        res = y - (a * x + b); s = max(np.std(res[m]), 1e-4)
        new = m & (np.abs(res) < nsig * s)
        if new.sum() == m.sum(): break
        m = new
    return float(a), a_err, float(b), int(m.sum())


def zero_cross(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float); m = np.isfinite(y); x = x[m]; y = y[m]
    if len(x) < 3: return np.nan
    s = np.sign(y); idx = np.where(np.diff(s) != 0)[0]
    if len(idx) == 0: return np.nan
    i = idx[np.argmin(np.abs(x[idx]))]
    return float(x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))


def argmin_interp(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float); m = np.isfinite(y); x = x[m]; y = y[m]
    if len(x) < 3: return np.nan
    i = int(np.argmin(y))
    if 0 < i < len(x) - 1:
        a, b, c = np.polyfit(x[i - 1:i + 2], y[i - 1:i + 2], 2)
        return float(-b / (2 * a)) if a > 0 else float(x[i])
    return float(x[i])


# --------------------------------------------------------------------------------------
# synthetic photometry, doabdi recipe
# --------------------------------------------------------------------------------------
def load_curve(name):
    d = pd.read_csv(TRANSMISSION_DIR / f"{name}.csv")
    w = d.iloc[:, 0].to_numpy(float); t = np.clip(d.iloc[:, 1].to_numpy(float), 0, None)
    if np.nanmedian(w) > 2000: w = w / 10.0
    return w, t


def syn_mag(wave_nm, flux, fw, ft, shift_nm=0.0, dlam=0.1):
    """AB magnitude(s) through the curve (fw+shift, ft).
    flux: (..., n) in W m^-2 nm^-1 on wave_nm (increasing).  Photon-weighted, 3631 Jy reference.
    The spectrum is taken as linear between its nodes and the curve is integrated on a fine grid (the union of the
    spectrum nodes, the curve nodes and a dlam-nm grid over the curve support): the integral reduces to fixed weights on
    the spectrum nodes, so the cost is one dot product per star.  (2026-09-29: before, the curve was sampled only at the
    spectrum nodes; on the 2-nm Gaia XP grid sharp filter edges aliased into colour-dependent errors, e.g. +5 to +9
    mmag/mag in the colour term of m438 ky nobeam and m386 manufacturer nobeam.)"""
    wave_nm = np.asarray(wave_nm, float)
    fw = np.asarray(fw, float) + shift_nm
    ft = np.asarray(ft, float)
    nz = np.nonzero(ft > 0)[0]
    lo = max(fw[max(nz[0] - 1, 0)], wave_nm[0]) if len(nz) else wave_nm[0]
    hi = min(fw[min(nz[-1] + 1, len(fw) - 1)], wave_nm[-1]) if len(nz) else wave_nm[0]
    if not len(nz) or hi <= lo:
        return np.full(np.shape(flux)[:-1], np.nan) if np.ndim(flux) > 1 else np.nan
    g = np.unique(np.concatenate([np.arange(lo, hi, dlam), [hi], wave_nm[(wave_nm > lo) & (wave_nm < hi)],
                                  fw[(fw > lo) & (fw < hi)]]))
    t = np.interp(g, fw, ft, left=0.0, right=0.0)
    dg = np.diff(g)
    q = np.zeros_like(g)                                   # trapezoid weights on g
    q[:-1] += dg / 2
    q[1:] += dg / 2
    k = np.clip(np.searchsorted(wave_nm, g, side="right") - 1, 0, len(wave_nm) - 2)
    u = (g - wave_nm[k]) / (wave_nm[k + 1] - wave_nm[k])
    tq = t * g * q
    w = (np.bincount(k, tq * (1 - u), minlength=len(wave_nm))
         + np.bincount(k + 1, tq * u, minlength=len(wave_nm)))   # weights on the spectrum nodes
    num = np.asarray(flux, float) @ w
    den = np.sum(F_NU_AB * C_NM_S / g ** 2 / 1e3 * tq)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = -2.5 * np.log10(num / den)
    return np.where(np.isfinite(m) & (num > 0), m, np.nan)


# --------------------------------------------------------------------------------------
# stage extract
# --------------------------------------------------------------------------------------
def _extract_tile(args):
    tile, wanted = args
    path = SPECTRA_DIR / f"{tile}.csv"
    out = {}
    if not path.exists():
        return tile, out, "missing"
    with open(path, "r", errors="ignore") as fh:
        header = fh.readline().lstrip("﻿").rstrip("\n").split(",")
        try:
            i_cor = header.index("correctedxpspecv1_flux_cor")
        except ValueError:
            return tile, out, "no flux_cor column"
        for line in fh:
            sid = line[:line.find(",")]
            if sid not in wanted: continue
            parts = line.rstrip("\n").split(",")
            vals = parts[i_cor].split(";")
            if len(vals) != len(XP_WAVE_NM): continue
            out[sid] = np.asarray(vals, float)
    return tile, out, "ok"


def stage_extract(out_dir: Path, n_proc: int):
    rf = pd.read_csv(out_dir / "refit_frames.csv", low_memory=False)
    rf = rf[(rf.catalog == CATALOG) & rf.objname.isin(STANDARDS) & rf.ref_tiles.notna()]
    tiles_of = {o: sorted({t for s in g.ref_tiles.unique() for t in str(s).split(";") if t}) for o, g in rf.groupby("objname")}
    st = pd.read_csv(out_dir / "refit_stars.csv.gz", usecols=["catalog", "objname", "ref_id"], dtype={"ref_id": str})
    st = st[(st.catalog == CATALOG) & st.objname.isin(STANDARDS)]
    ids_of = {o: set(g.ref_id.unique()) for o, g in st.groupby("objname")}
    jobs = [(t, ids_of[o]) for o in STANDARDS if o in tiles_of for t in tiles_of[o]]
    log(f"extract: {sum(len(v) for v in ids_of.values())} reference stars from {len(jobs)} tiles")
    spectra, tile_of = {}, {}
    with Pool(min(n_proc, len(jobs))) as pool:
        for tile, found, status in pool.imap_unordered(_extract_tile, jobs):
            log(f"  {tile}: {status}, {len(found)} stars")
            for sid, fl in found.items():
                spectra[sid] = fl; tile_of[sid] = tile
    ids = sorted(spectra)
    flux = np.vstack([spectra[s] for s in ids]) if ids else np.zeros((0, len(XP_WAVE_NM)))
    obj_of = {s: o for o, ss in ids_of.items() for s in ss}
    d = out_dir / "shift_recal"; d.mkdir(exist_ok=True)
    np.savez_compressed(d / "refstar_spectra.npz", ids=np.array(ids), flux=flux, wave_nm=XP_WAVE_NM,
                        objname=np.array([obj_of[s] for s in ids]), tile=np.array([tile_of[s] for s in ids]))
    missing = sum(len(v) for v in ids_of.values()) - len(ids)
    log(f"extract: {len(ids)} spectra saved ({missing} reference stars not found in the tiles)")


# --------------------------------------------------------------------------------------
# stage compute
# --------------------------------------------------------------------------------------
def clipped_zp(frame, x, nsig=3.0, maxiters=5):
    """Per-frame 3-sigma clipped median/std (astropy SigmaClip defaults: cenfunc median, stdfunc std)."""
    frame = np.asarray(frame); x = np.asarray(x, float)
    mask = np.isfinite(x)
    for _ in range(maxiters):
        s = pd.Series(x[mask]).groupby(frame[mask])
        med = s.median(); sd = s.std(ddof=0)
        m = pd.Series(frame).map(med).to_numpy(); d = pd.Series(frame).map(sd).to_numpy()
        new = mask & np.isfinite(m) & (np.abs(x - m) <= nsig * d)
        if new.sum() == mask.sum(): break
        mask = new
    s = pd.Series(x[mask]).groupby(frame[mask])
    return s.median(), s.std(ddof=0), s.size(), mask


def calspec_clean(out_dir: Path, bad: set):
    c = pd.read_csv(out_dir / "calspec_frames.csv")
    c = c[(c["status"] == "ok") & ~c["file"].isin(bad)]
    c = c[(c["flags"].fillna(0) <= 1) & (~c["saturated"].astype(bool)) & (c["sep_arcsec"] < 1.5)
          & (c["magerr_APER_3"] < CAL_MAX_MAGERR) & (c["flux_max"] < CAL_MAX_FLUX)]
    return c


def stage_compute(out_dir: Path):
    d = out_dir / "shift_recal"; d.mkdir(exist_ok=True)
    fig_dir = out_dir / "figures"; fig_dir.mkdir(exist_ok=True)
    npz = np.load(d / "refstar_spectra.npz", allow_pickle=False)
    ids = npz["ids"].astype(str); flux = npz["flux"]; wave = npz["wave_nm"]
    idx_of = {s: i for i, s in enumerate(ids)}
    curves = {f: load_curve(f) for f in GRID}
    fit = pd.read_csv(out_dir / "calspec_shift_fit.csv").set_index("filter")
    c4 = {f: (float(fit.loc[f, "best_shift_nm"]) if f in fit.index and np.isfinite(fit.loc[f, "shift_err_nm"]) else 0.0) for f in GRID}

    frames = pd.read_csv(out_dir / "frames.csv", low_memory=False)
    bad = set(frames[frames["ZPERR_APER_3"] > ZPERR_ANOMALY]["file"])
    st = pd.read_csv(out_dir / "refit_stars.csv.gz", dtype={"ref_id": str})
    st = st[(st.catalog == CATALOG) & st.objname.isin(STANDARDS) & ~st.file.isin(bad) & st["filter"].isin(GRID)]
    st = st[st.ref_id.isin(idx_of)].copy()
    st["row"] = st.ref_id.map(idx_of)
    st["sig"] = np.sqrt(st.magerr ** 2 + MAGERR_FLOOR ** 2)
    log(f"compute: {len(st)} star measurements in {st.file.nunique()} frames, {st.ref_id.nunique()} stars")

    cal = calspec_clean(out_dir, bad)
    syn = pd.read_csv(out_dir / "calspec_synphot.csv")
    spec_file = dict(zip(syn["object"], syn["spec"]))
    cal_spec = {}
    for o in STANDARDS:
        if o not in spec_file: continue
        t = Table.read(CALSPEC_DIR / spec_file[o])
        wl = np.asarray(t["WAVELENGTH"], float) / 10.0; fl = np.asarray(t["FLUX"], float) * 1e-2   # nm, W m^-2 nm^-1
        m = (wl > 300) & (wl < 1000) & np.isfinite(fl)
        cal_spec[o] = (wl[m], fl[m])
    sptype = cal.groupby("object")["spectral_type"].first().to_dict()
    is_da = {o: str(sptype.get(o, "")).startswith("DA") for o in STANDARDS}

    # --- zero-shift reproduction check
    worst = 0.0
    for f in GRID:
        rows = st[st["filter"] == f]
        if not len(rows): continue
        m0 = syn_mag(wave, flux[rows.row.to_numpy()], *curves[f], 0.0)
        worst = max(worst, float(np.nanmax(np.abs(m0 - rows.ref_mag.to_numpy()))))
    log(f"compute: zero-shift re-synthesis reproduces the archive reference magnitudes to {worst:.2e} mag (max)")
    cal_check = []
    for o, (wl, fl) in cal_spec.items():
        for f in GRID:
            m0 = float(syn_mag(wl, fl, *curves[f], 0.0))
            ref = syn[(syn.object == o) & (syn["filter"] == f)]["mag_syn"]
            if len(ref): cal_check.append(m0 - float(ref.iloc[0]))
    log(f"compute: CALSPEC zero-shift trapezoid vs pyphot synphot: median {np.median(cal_check):+.4f}, max |d| {np.max(np.abs(cal_check)):.4f} mag")

    # --- loop over filters and shifts
    summary, star_rows, frame_rows, syn_rows, recommended = [], [], [], [], []
    for f in GRID:
        rows = st[st["filter"] == f].copy()
        if not len(rows): continue
        fw, ft = curves[f]
        m_ref0 = syn_mag(wave, flux[rows.row.to_numpy()], fw, ft, 0.0)
        zp0_frame = rows.groupby("file").zp.median()
        cf = cal[cal["filter"] == f].copy()
        cal0 = {o: float(syn_mag(*cal_spec[o], fw, ft, 0.0)) for o in cal_spec}
        syn0 = {o: float(syn[(syn.object == o) & (syn["filter"] == f)]["mag_syn"].iloc[0]) for o in cal_spec
                if len(syn[(syn.object == o) & (syn["filter"] == f)])}

        def evaluate(sh):
            dm_ref = syn_mag(wave, flux[rows.row.to_numpy()], fw, ft, sh) - m_ref0
            zp_i = rows.zp.to_numpy() + dm_ref
            med, sd, n, keep = clipped_zp(rows.file.to_numpy(), zp_i)
            resid = zp_i - rows.file.map(med).to_numpy()
            a, a_err, _, n_fit = weighted_linfit(rows.bprp.to_numpy()[keep], resid[keep], rows.sig.to_numpy()[keep])
            dm_cal = {o: float(syn_mag(*cal_spec[o], fw, ft, sh)) - cal0[o] for o in cal_spec}
            c = cf.copy()
            c["delta_new"] = c["delta_APER_3"] + (c.file.map(med) - c.file.map(zp0_frame)).to_numpy() - c.object.map(dm_cal).to_numpy()
            ok = c.dropna(subset=["delta_new"])
            per_star = ok.groupby("object").agg(n=("delta_new", "size"), delta=("delta_new", "median"), scatter=("delta_new", mad))
            per_star = per_star.reindex(columns=["n", "delta", "scatter"])      # keep the columns even when no star survives
            per_star = per_star[per_star.n >= MIN_FRAMES]
            per_star["err"] = np.maximum(per_star.scatter / np.sqrt(per_star.n), ERR_FLOOR)
            per_star["is_da"] = np.array([is_da[o] for o in per_star.index], dtype=bool)   # bool dtype even when empty
            da = per_star.loc[per_star["is_da"], "delta"]; other = per_star.loc[~per_star["is_da"], "delta"]
            w = 1 / per_star.err ** 2
            rec = dict(filter=f, shift_nm=sh, n_frames=int(len(med)), n_stars_field=int(rows.ref_id.nunique()),
                       zp_change_median=float((med - zp0_frame).median()), zp_std_median=float(sd.median()),
                       a_bprp=a, a_bprp_err=a_err, n_fit=n_fit, cal_n_stars=int(len(per_star)),
                       cal_median_all=float(per_star.delta.median()) if len(per_star) else np.nan,
                       cal_da_minus_other=float(da.median() - other.median()) if len(da) and len(other) else np.nan,
                       cal_between_star_mad=mad(per_star.delta) if len(per_star) >= 3 else np.nan,
                       cal_chi2_const=float(np.sum(w * (per_star.delta - np.sum(w * per_star.delta) / np.sum(w)) ** 2)) if len(per_star) else np.nan)
            return rec, per_star, med, sd, n, dm_cal

        def record(sh, cases, ev):
            rec, per_star, med, sd, n, dm_cal = ev
            for case in cases:
                summary.append(dict(rec, case=case))
                for o, r in per_star.iterrows():
                    star_rows.append(dict(filter=f, shift_nm=sh, case=case, object=o, is_da=bool(r.is_da), n=int(r.n), delta=float(r.delta), err=float(r.err)))
                if case != "scan":
                    for fpath, v in med.items():
                        frame_rows.append(dict(filter=f, shift_nm=sh, case=case, file=fpath, zp=float(v), zp_std=float(sd[fpath]), n_used=int(n[fpath])))
                    for o, dm in dm_cal.items():
                        if o in syn0:
                            syn_rows.append(dict(filter=f, shift_nm=sh, case=case, object=o, dm_syn=dm, mag_syn_shifted=syn0[o] + dm))

        shifts = sorted(set(SCAN_NM.tolist()) | {c4[f]})
        filter_rows = []
        for sh in shifts:
            cases = (["none"] if sh == 0 else []) + (["c4"] if abs(sh - c4[f]) < 1e-9 else [])
            if not cases: cases = ["scan"]
            ev = evaluate(sh); record(sh, cases, ev); filter_rows.append(dict(ev[0], shift_nm=sh))
        # consensus of the scan -> recommended shift
        sc = pd.DataFrame(filter_rows).drop_duplicates("shift_nm").sort_values("shift_nm")
        cand = dict(colorterm_zero=zero_cross(sc.shift_nm, sc.a_bprp), zpstd_min=argmin_interp(sc.shift_nm, sc.zp_std_median),
                    cal_da_zero=zero_cross(sc.shift_nm, sc.cal_da_minus_other), cal_mad_min=argmin_interp(sc.shift_nm, sc.cal_between_star_mad))
        vals = [v for v in cand.values() if np.isfinite(v) and abs(v) < SCAN_EDGE]
        rec_shift = float(np.median(vals)) if len(vals) >= 2 else np.nan
        recommended.append(dict(filter=f, c4=c4[f], **cand, n_diag=len(vals), spread=(max(vals) - min(vals)) if vals else np.nan,
                                recommended=rec_shift, applied=0.0 if not np.isfinite(rec_shift) else rec_shift))
        record(0.0 if not np.isfinite(rec_shift) else rec_shift, ["rec"], evaluate(0.0 if not np.isfinite(rec_shift) else rec_shift))
        log(f"  {f}: c4 {c4[f]:+.1f} nm, recommended {rec_shift:+.1f} nm ({len(vals)} diagnostics), {len(zp0_frame)} frames")
    S = pd.DataFrame(summary); S.to_csv(d / "recal_summary.csv", index=False)
    pd.DataFrame(star_rows).to_csv(d / "recal_calspec_stars.csv", index=False)
    pd.DataFrame(frame_rows).to_csv(d / "recal_frames.csv.gz", index=False, compression="gzip")
    pd.DataFrame(syn_rows).to_csv(d / "recal_calspec_syn.csv", index=False)
    pd.DataFrame(recommended).to_csv(d / "recal_recommended.csv", index=False)

    # --- figures
    piv = {f: float(np.sqrt(np.trapezoid(curves[f][1] * curves[f][0], curves[f][0]) / np.trapezoid(curves[f][1] / curves[f][0], curves[f][0]))) for f in GRID}
    make_overplot(S, pd.DataFrame(star_rows), c4, piv, fig_dir / "C5_recal_overplot.png")
    make_scan(S, c4, piv, fig_dir / "C6_recal_scan.png")
    print_table(S, c4)


def make_overplot(S, stars, c4, piv, path):
    cols = {"none": ("0.55", "o", "no shift"), "c4": ("tab:blue", "D", "C4 shift"), "rec": ("tab:orange", "s", "recommended shift")}
    fig = plt.figure(figsize=(18, 20)); gs = fig.add_gridspec(6, 5, height_ratios=[1.3, 1.3, 1, 1, 1, 1])
    metrics = [("cal_da_minus_other", "DA − other stars [mag]", (-0.07, 0.04)), ("cal_between_star_mad", "between-star MAD [mag]", (0, 0.05)),
               ("cal_median_all", "median residual, all stars [mag]", (-0.04, 0.04)), ("a_bprp", "field colour term dZP/d(BP−RP)", (-0.05, 0.05)),
               ("zp_std_median", "median ZP scatter per frame [mag]", (0.01, 0.03))]
    for k, (col, lab, yl) in enumerate(metrics):
        ax = fig.add_subplot(gs[0, k])
        for case, (c, mk, lb) in cols.items():
            dd = S[S.case == case]
            ax.plot([piv[f] for f in dd["filter"]], dd[col], mk, color=c, ms=5, label=lb, alpha=.9)
            if col == "a_bprp": ax.errorbar([piv[f] for f in dd["filter"]], dd[col], yerr=dd["a_bprp_err"], fmt="none", ecolor=c, alpha=.5)
        ax.axhline(0, color="k", lw=.5); ax.set_ylim(*yl); ax.grid(alpha=.3); ax.set_title(lab, fontsize=9); ax.set_xlabel("pivot [nm]", fontsize=8)
        if k == 0: ax.legend(fontsize=8)
    ax = fig.add_subplot(gs[1, 0:3])
    for case, (c, mk, lb) in (("c4", cols["c4"]), ("rec", cols["rec"])):
        dd = S[S.case == case]
        ax.plot([piv[f] for f in dd["filter"]], dd["shift_nm"], mk + "-", color=c, ms=5, label=lb)
    ax.axhline(0, color="k", lw=.5); ax.set_title("applied shift per filter [nm]", fontsize=9); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax = fig.add_subplot(gs[1, 3:5])
    for case, (c, mk, lb) in cols.items():
        dd = S[S.case == case]
        ax.plot([piv[f] for f in dd["filter"]], dd["cal_chi2_const"], mk, color=c, ms=5, label=lb)
    ax.set_yscale("log"); ax.set_title("χ² of the star-to-star residuals about a constant", fontsize=9); ax.grid(alpha=.3); ax.legend(fontsize=8)
    for k, f in enumerate(GRID):
        a = fig.add_subplot(gs[2 + k // 5, k % 5])
        d0 = stars[(stars["filter"] == f) & (stars.case == "none")].set_index("object")
        d1 = stars[(stars["filter"] == f) & (stars.case == "c4")].set_index("object")
        d2 = stars[(stars["filter"] == f) & (stars.case == "rec")].set_index("object")
        objs = [o for o in STANDARDS if o in d0.index]
        x = np.arange(len(objs))
        for i, o in enumerate(objs):
            a.errorbar(i - 0.2, d0.loc[o, "delta"], yerr=d0.loc[o, "err"], fmt="o", color="0.55", ms=5, capsize=2, zorder=3)
            if o in d1.index: a.errorbar(i, d1.loc[o, "delta"], yerr=d1.loc[o, "err"], fmt="D", color="tab:blue", ms=5, capsize=2, zorder=4)
            if o in d2.index: a.errorbar(i + 0.2, d2.loc[o, "delta"], yerr=d2.loc[o, "err"], fmt="s", color="tab:orange", ms=5, capsize=2, zorder=4)
            if d0.loc[o, "is_da"]: a.axvspan(i - 0.5, i + 0.5, color="tab:blue", alpha=.06, lw=0)
        s0 = S[(S["filter"] == f) & (S.case == "none")].iloc[0]; s1 = S[(S["filter"] == f) & (S.case == "c4")].iloc[0]
        s2 = S[(S["filter"] == f) & (S.case == "rec")].iloc[0]
        a.axhline(0, color="k", lw=.5); a.set_ylim(-0.1, 0.1); a.grid(alpha=.3)
        a.set_xticks(x); a.set_xticklabels(objs, rotation=90, fontsize=6)
        a.set_title(f"{f}  C4 {s1.shift_nm:+.1f} / rec {s2.shift_nm:+.1f} nm | DA−other {s0.cal_da_minus_other:+.3f} → {s1.cal_da_minus_other:+.3f} / {s2.cal_da_minus_other:+.3f}", fontsize=7)
        if k % 5 == 0: a.set_ylabel("m_obs − m_syn [mag]", fontsize=8)
    fig.suptitle("Closed-loop test: reference photometry and CALSPEC synthetic magnitudes recomputed with shifted curves\n"
                 "grey circles = no shift, blue diamonds = C4 per-filter shift, orange squares = recommended shift (scan consensus); shaded = DA white dwarfs", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def make_scan(S, c4, piv, path):
    fig, axes = plt.subplots(4, 5, figsize=(18, 13), sharex=True)
    for ax, f in zip(axes.ravel(), GRID):
        dd = S[(S["filter"] == f) & (S.case != "rec")].drop_duplicates("shift_nm").sort_values("shift_nm")
        rec = S[(S["filter"] == f) & (S.case == "rec")]
        ax.plot(dd.shift_nm, dd.cal_da_minus_other, "o-", color="tab:blue", ms=3, label="DA − other")
        ax.plot(dd.shift_nm, dd.cal_between_star_mad, "s-", color="tab:green", ms=3, label="between-star MAD")
        ax.plot(dd.shift_nm, dd.cal_median_all, "^-", color="0.4", ms=3, label="median all stars")
        ax2 = ax.twinx()
        ax2.plot(dd.shift_nm, dd.a_bprp, "d--", color="tab:red", ms=3, label="field colour term")
        ax2.set_ylim(-0.06, 0.06); ax2.tick_params(labelsize=7, colors="tab:red")
        ax.axvline(c4[f], color="tab:blue", lw=.8, ls=":")
        if len(rec): ax.axvline(rec.shift_nm.iloc[0], color="tab:orange", lw=1.2, ls="--")
        ax.axhline(0, color="k", lw=.5); ax.axvline(0, color="k", lw=.5)
        ax.set_ylim(-0.08, 0.08); ax.grid(alpha=.3); ax.tick_params(labelsize=7)
        ax.set_title(f"{f}  (C4 {c4[f]:+.1f} nm, rec {rec.shift_nm.iloc[0] if len(rec) else float('nan'):+.1f} nm)", fontsize=9)
        if f == GRID[0]:
            h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, fontsize=6, loc="lower right")
    for ax in axes[-1]: ax.set_xlabel("curve shift [nm]", fontsize=8)
    fig.suptitle("Scan of the closed loop: CALSPEC diagnostics (left axis) and field colour term (right axis, red) vs curve shift; dotted = C4, dashed = recommended", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def print_table(S, c4):
    print("\nfilter  c4/rec  | DA-other: none -> c4 -> rec | star MAD: none -> c4 -> rec | colour term: none -> c4 -> rec | ZP std: none -> rec | dZP(rec)")
    for f in GRID:
        s0 = S[(S["filter"] == f) & (S.case == "none")]; s1 = S[(S["filter"] == f) & (S.case == "c4")]; s2 = S[(S["filter"] == f) & (S.case == "rec")]
        if not len(s0) or not len(s1) or not len(s2): continue
        s0 = s0.iloc[0]; s1 = s1.iloc[0]; s2 = s2.iloc[0]
        print(f"{f:6s} {s1.shift_nm:+5.1f}/{s2.shift_nm:+5.1f} | {s0.cal_da_minus_other:+.3f} -> {s1.cal_da_minus_other:+.3f} -> {s2.cal_da_minus_other:+.3f} | "
              f"{s0.cal_between_star_mad:.3f} -> {s1.cal_between_star_mad:.3f} -> {s2.cal_between_star_mad:.3f} | {s0.a_bprp:+.4f} -> {s1.a_bprp:+.4f} -> {s2.a_bprp:+.4f} | "
              f"{s0.zp_std_median:.4f} -> {s2.zp_std_median:.4f} | {s2.zp_change_median:+.4f}")


def main():
    global MIN_FRAMES, GRID
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["extract", "compute", "all"])
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--n-proc", type=int, default=16)
    ap.add_argument("--min-frames", type=int, default=MIN_FRAMES, help="frames per star and filter required (use 1 for coadds)")
    ap.add_argument("--filters", default="", help="comma-separated filters (default: the 25 nm grid)")
    args = ap.parse_args()
    MIN_FRAMES = args.min_frames
    if args.filters: GRID = [f.strip() for f in args.filters.split(",") if f.strip()]
    out = Path(args.out_dir)
    t0 = time.time()
    if args.stage in ("extract", "all"):
        stage_extract(out, args.n_proc)
    if args.stage in ("compute", "all"):
        stage_compute(out)
    log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
