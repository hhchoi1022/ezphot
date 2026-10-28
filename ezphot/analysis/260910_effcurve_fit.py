#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260910_effcurve_fit.py

Estimate the effective transmission curve of every 7DT medium band (plan of 2026-09-09).

Model (per filter f):  T_eff = Family[F_f](Delta_f) * S(lambda), Family = rigid shift or beam-averaged filter-only
curve, S = filter-independent system factor (QE x optics x atmosphere) held fixed, so that the telluric bands of S
do not move with the filter.  One parameter per filter, the total effective shift Delta_f, is measured from
  (1) the standard-star fields: per-frame zero points re-derived from the same reference stars with re-synthesised
      reference magnitudes -> field colour term a(Delta), ZP scatter, per-unit splits, CALSPEC per-star residuals;
  (2) the survey-tile coadds: the same with the tile reference stars (XP spectra extracted here);
combined as chi^2(Delta) = min_c sum_s [(Delta'_s - c)/sigma_s]^2 + [a_std/sigma_a]^2 + [a_tile/sigma_a]^2,
and tied across filters by a shrinkage prior Delta = k * pivot (tau = excess scatter of the well-measured filters).

Stages (--stage):
  extract : XP spectra of the tile-coadd reference stars       -> tilestar_spectra.npz
  scan    : diagnostics vs Delta for both families              -> scan_channels.csv, scan_units.csv, scan_calspec.csv, family_closure.csv
  combine : per-filter estimates, prior, final values, flags    -> shift_table.csv, model_params.json
  write   : curve files (nm / AA / uncertainty / alternative form) + metadata, closure check, repo copy
  all     : extract -> scan -> combine -> write

  MPLCONFIGDIR=/tmp/ezphot-matplotlib /home/hhchoi1022/anaconda3/envs/py311/bin/python \\
      ezphot/analysis/260910_effcurve_fit.py --stage all --n-proc 16
"""
from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent))
C = importlib.import_module("260910_effcurve_common")
R = C.R
warnings.filterwarnings("ignore")

SCAN_DEFAULT = np.round(np.arange(-6.0, 3.0 + 1e-9, 0.25), 2)
FORMS = ("rigid", "beam")
SIGMA_A_DEFAULT = 0.003
SIGMA_A_SPECIAL = {"m400": 0.010, "m375w": 0.010, "m386": 0.010}   # XP blue end
SIGMA_STAR_FLOOR = 0.012                                            # per-star CALSPEC systematic floor [mag]
NO_PRIOR = {"m400", "m425"}                                          # opposite sign: not shrunk, not in the prior fit
PRIOR_MIN_PIVOT, PRIOR_MAX_SIGMA, TAU_FLOOR = 440.0, 0.8, 0.3
GRID = list(R.GRID)

log = C.log


# --------------------------------------------------------------------------------------
# stage extract
# --------------------------------------------------------------------------------------
def extract_tile(args):
    tile, wanted = args
    path = R.SPECTRA_DIR / f"{tile}.csv"
    out = {}
    if not path.exists():
        return tile, out, "missing"
    with open(path, "r", errors="ignore") as fh:
        header = fh.readline().lstrip("﻿").rstrip("\n").split(",")
        try:
            i_cor = header.index("correctedxpspecv1_flux_cor"); i_err = header.index("correctedxpspecv1_flux_error")
        except ValueError:
            return tile, out, "no flux_cor column"
        i_cau = header.index("correctedxpspecv1_caution") if "correctedxpspecv1_caution" in header else -1
        for line in fh:
            sid = line[:line.find(",")]
            if sid not in wanted: continue
            parts = line.rstrip("\n").split(",")
            vals = parts[i_cor].split(";")
            if len(vals) != len(R.XP_WAVE_NM): continue
            err = parts[i_err].split(";")
            err = np.asarray(err, float) if len(err) == len(vals) else np.full(len(vals), np.nan)
            cau = parts[i_cau] if i_cau >= 0 and i_cau < len(parts) else ""
            out[sid] = (np.asarray(vals, float), err, cau)
    return tile, out, "ok"


def stage_extract(out: Path, n_proc: int):
    rf = pd.read_csv(C.PREV_OUT / "coadd" / "refit_frames.csv", low_memory=False)
    rf = rf[(rf.catalog == R.CATALOG) & rf.objname.astype(str).str.match(r"^T\d{5}$") & rf.ref_tiles.notna()]
    tiles_of = {o: sorted({t for s in g.ref_tiles.unique() for t in str(s).split(";") if t}) for o, g in rf.groupby("objname")}
    st = pd.read_csv(C.PREV_OUT / "coadd" / "refit_stars.csv.gz", usecols=["catalog", "objname", "ref_id"], dtype={"ref_id": str})
    st = st[(st.catalog == R.CATALOG) & st.objname.isin(tiles_of)]
    ids_of = {o: set(g.ref_id.unique()) for o, g in st.groupby("objname")}
    # one job per raw tile with the union of the stars of every coadd object that references it
    wanted_by_tile = {}
    for o, ids in ids_of.items():
        for t in tiles_of[o]:
            wanted_by_tile.setdefault(t, set()).update(ids)
    jobs = sorted(wanted_by_tile.items())
    log(f"extract: {sum(len(v) for v in ids_of.values())} tile reference stars from {len(jobs)} raw tiles")
    spectra, errors, cautions, tile_of = {}, {}, {}, {}
    with Pool(min(n_proc, len(jobs))) as pool:
        for tile, found, status in pool.imap_unordered(extract_tile, jobs):
            log(f"  {tile}: {status}, {len(found)} stars")
            for sid, (fl, er, cau) in found.items():
                spectra[sid] = fl; errors[sid] = er; cautions[sid] = cau; tile_of[sid] = tile
    ids = sorted(spectra)
    obj_of = {}
    for o, ss in ids_of.items():
        for s in ss: obj_of.setdefault(s, o)
    np.savez_compressed(out / "tilestar_spectra.npz", ids=np.array(ids), flux=np.vstack([spectra[s] for s in ids]),
                        flux_err=np.vstack([errors[s] for s in ids]), caution=np.array([cautions[s] for s in ids]),
                        wave_nm=R.XP_WAVE_NM, objname=np.array([obj_of[s] for s in ids]), tile=np.array([tile_of[s] for s in ids]))
    missing = sum(len(v) for v in ids_of.values()) - len(ids)
    log(f"extract: {len(ids)} spectra saved ({missing} reference stars not found in the tiles)")


# --------------------------------------------------------------------------------------
# stage scan
# --------------------------------------------------------------------------------------
_G = {}   # data shared with the worker processes (fork)


def load_channel(kind: str, out: Path):
    """Per-star zero points of one channel joined to their XP spectra. kind = 'std' (single frames of the
    standard fields) or 'tile' (survey-tile coadds)."""
    if kind == "std":
        frames = C.load_frames(kind="single")
        st = pd.read_csv(C.PREV_OUT / "refit_stars.csv.gz", dtype={"ref_id": str})
        npz = np.load(C.PREV_OUT / "shift_recal" / "refstar_spectra.npz", allow_pickle=False)
        st = st[st.objname.isin(R.STANDARDS)]
    else:
        p = out / "tilestar_spectra.npz"
        if not p.exists():
            log("tile channel: tilestar_spectra.npz missing (run --stage extract); tile channel skipped"); return None
        frames = C.load_frames(kind="coadd")
        st = pd.read_csv(C.PREV_OUT / "coadd" / "refit_stars.csv.gz", dtype={"ref_id": str})
        npz = np.load(p, allow_pickle=False)
        st = st[st.objname.astype(str).str.match(r"^T\d{5}$")]
    bad = set(frames[frames.bad_zperr | frames.mislabeled]["file"])
    ids = npz["ids"].astype(str); flux = npz["flux"]; wave = npz["wave_nm"]
    idx_of = {s: i for i, s in enumerate(ids)}
    st = st[(st.catalog == R.CATALOG) & ~st.file.isin(bad) & st.ref_id.isin(idx_of)].copy()
    st["row"] = st.ref_id.map(idx_of).astype(int)
    st["sig"] = np.sqrt(st.magerr ** 2 + R.MAGERR_FLOOR ** 2)
    mjd = frames.set_index("file")["MJD"]; st["mjd"] = st.file.map(mjd)
    log(f"{kind} channel: {len(st)} star measurements, {st.file.nunique()} frames, {st.ref_id.nunique()} stars, {st['filter'].nunique()} filters")
    return dict(st=st, flux=flux, wave=wave)


def load_calspec():
    frames = C.load_frames(kind="single")
    bad = set(frames[frames.bad_zperr | frames.mislabeled]["file"])
    cal = R.calspec_clean(C.PREV_OUT, bad)
    syn = pd.read_csv(C.PREV_OUT / "calspec_synphot.csv")
    spec_file = dict(zip(syn["object"], syn["spec"]))
    spectra = {}
    for o in R.STANDARDS:
        if o not in spec_file: continue
        t = Table.read(R.CALSPEC_DIR / spec_file[o])
        wl = np.asarray(t["WAVELENGTH"], float) / 10.0; fl = np.asarray(t["FLUX"], float) * 1e-2
        m = (wl > 300) & (wl < 1000) & np.isfinite(fl); spectra[o] = (wl[m], fl[m])
    sptype = cal.groupby("object")["spectral_type"].first().to_dict()
    is_da = {o: str(sptype.get(o, "")).startswith("DA") for o in R.STANDARDS}
    return dict(cal=cal, spectra=spectra, is_da=is_da)


def channel_diagnostics(rows, flux, wave, curve0, curve, per_unit=True):
    """Re-synthesise the reference magnitudes with `curve` (relative to `curve0`), re-derive the per-frame ZPs and
    return the colour term, ZP scatter and ZP change, overall and per unit."""
    u_rows, inv = np.unique(rows.row.to_numpy(), return_inverse=True)
    dm = (R.syn_mag(wave, flux[u_rows], C.FINE, curve, 0.0) - R.syn_mag(wave, flux[u_rows], C.FINE, curve0, 0.0))[inv]
    zp_i = rows.zp.to_numpy() + dm
    med, sd, n, keep = R.clipped_zp(rows.file.to_numpy(), zp_i)
    resid = zp_i - rows.file.map(med).to_numpy()
    a, a_err, _, n_fit = R.weighted_linfit(rows.bprp.to_numpy()[keep], resid[keep], rows.sig.to_numpy()[keep])
    zp0 = rows.groupby("file").zp.median()
    rec = dict(n_frames=int(len(med)), n_stars=int(rows.ref_id.nunique()), a=a, a_err=a_err, n_fit=n_fit,
               zp_std=float(sd.median()), dzp=float((med - zp0).median()))
    units = []
    if per_unit:
        tel = rows.telescop.to_numpy()
        for u in sorted(set(tel)):
            m = (tel == u) & keep
            if m.sum() < 30: continue
            au, au_err, _, nu = R.weighted_linfit(rows.bprp.to_numpy()[m], resid[m], rows.sig.to_numpy()[m])
            fr = set(rows.file.to_numpy()[tel == u])
            units.append(dict(telescop=u, n_frames=len(fr), a=au, a_err=au_err, n_fit=nu, zp_std=float(sd[list(fr)].median())))
    return rec, units, med, zp0


def calspec_residuals(cf, spectra, is_da, curve0, curve, med, zp0, cal0):
    """CALSPEC per-star residuals with the re-derived frame ZPs (closed loop): Delta' = Delta + (ZP' - ZP0) - dm_cal."""
    dm_cal = {o: float(R.syn_mag(*spectra[o], C.FINE, curve, 0.0)) - cal0[o] for o in spectra}
    c = cf.copy()
    c["delta_new"] = c["delta_APER_3"] + (c.file.map(med) - c.file.map(zp0)).to_numpy() - c.object.map(dm_cal).to_numpy()
    c = c.dropna(subset=["delta_new"])
    per = c.groupby("object").agg(n=("delta_new", "size"), delta=("delta_new", "median"), scatter=("delta_new", R.mad), magerr=("magerr_APER_3", "median"))
    per = per[per.n >= R.MIN_FRAMES]
    per["err"] = np.sqrt((per.scatter / np.sqrt(per.n)) ** 2 + (per.magerr / np.sqrt(per.n)) ** 2).clip(lower=R.ERR_FLOOR)
    per["is_da"] = [bool(is_da.get(o, False)) for o in per.index]
    return per, dm_cal


def scan_filter(args):
    f, scan, inject = args
    S = _G["S"]; std = _G["std"]; tile = _G["tile"]; cal = _G["cal"]; n_eff = _G["n_eff"]; eps = _G["eps"]; fratio = _G["fratio"]
    fam = C.CurveFamily(f, S, n_eff=n_eff, eps=eps, fratio=fratio)
    chan_rows, unit_rows, cal_rows, closure = [], [], [], []
    rows_std = std["st"][std["st"]["filter"] == f] if std is not None else None
    rows_tile = tile["st"][tile["st"]["filter"] == f] if tile is not None else None
    if (rows_std is None or not len(rows_std)) and (rows_tile is None or not len(rows_tile)):
        return f, chan_rows, unit_rows, cal_rows, closure
    cf = cal["cal"][cal["cal"]["filter"] == f] if cal is not None else None
    for form in FORMS:
        curve0 = fam.curve(0.0, form)
        # closure of the family at zero shift against the archive reference magnitudes (rigid: F*S vs nominal)
        if rows_std is not None and len(rows_std):
            u_rows, inv = np.unique(rows_std.row.to_numpy(), return_inverse=True)
            m0 = R.syn_mag(std["wave"], std["flux"][u_rows], C.FINE, curve0, 0.0)[inv]
            d = m0 - rows_std.ref_mag.to_numpy()
            closure.append(dict(filter=f, form=form, channel="std", median_offset=float(np.nanmedian(d)), max_abs_dev=float(np.nanmax(np.abs(d - np.nanmedian(d))))))
        if inject:   # injection test: the observations behave as if the true curve were shifted by `inject` on top of the fitted shift
            base = _G.get("base_shift", {}).get(f, 0.0)
            inj = fam.curve(base + inject, form); curve_b = fam.curve(base, form)
            if rows_std is not None and len(rows_std):
                u_rows, inv = np.unique(rows_std.row.to_numpy(), return_inverse=True)
                rows_std = rows_std.copy(); rows_std["zp"] = rows_std.zp.to_numpy() - (R.syn_mag(std["wave"], std["flux"][u_rows], C.FINE, inj, 0.0) - R.syn_mag(std["wave"], std["flux"][u_rows], C.FINE, curve_b, 0.0))[inv]
            if rows_tile is not None and len(rows_tile):
                u_rows, inv = np.unique(rows_tile.row.to_numpy(), return_inverse=True)
                rows_tile = rows_tile.copy(); rows_tile["zp"] = rows_tile.zp.to_numpy() - (R.syn_mag(tile["wave"], tile["flux"][u_rows], C.FINE, inj, 0.0) - R.syn_mag(tile["wave"], tile["flux"][u_rows], C.FINE, curve_b, 0.0))[inv]
            if cf is not None and len(cf):
                cf = cf.copy(); cal_inj = {o: float(R.syn_mag(*cal["spectra"][o], C.FINE, inj, 0.0) - R.syn_mag(*cal["spectra"][o], C.FINE, curve_b, 0.0)) for o in cal["spectra"]}
                cf["delta_APER_3"] = cf["delta_APER_3"] + cf.object.map(cal_inj).to_numpy()
        cal0 = {o: float(R.syn_mag(*cal["spectra"][o], C.FINE, curve0, 0.0)) for o in cal["spectra"]} if cal is not None else {}
        for delta in scan:
            curve = fam.curve(float(delta), form)
            if rows_std is not None and len(rows_std):
                rec, units, med, zp0 = channel_diagnostics(rows_std, std["flux"], std["wave"], curve0, curve)
                chan_rows.append(dict(filter=f, form=form, delta=float(delta), channel="std", **rec))
                unit_rows += [dict(filter=f, form=form, delta=float(delta), channel="std", **u) for u in units]
                if cf is not None and len(cf) and cal["spectra"]:
                    per, _ = calspec_residuals(cf, cal["spectra"], cal["is_da"], curve0, curve, med, zp0, cal0)
                    cal_rows += [dict(filter=f, form=form, delta=float(delta), object=o, is_da=bool(r.is_da), n=int(r.n), delta_star=float(r.delta), err=float(r.err)) for o, r in per.iterrows()]
            if rows_tile is not None and len(rows_tile):
                rec, units, _, _ = channel_diagnostics(rows_tile, tile["flux"], tile["wave"], curve0, curve)
                chan_rows.append(dict(filter=f, form=form, delta=float(delta), channel="tile", **rec))
                unit_rows += [dict(filter=f, form=form, delta=float(delta), channel="tile", **u) for u in units]
    return f, chan_rows, unit_rows, cal_rows, closure


def system_factor(model: str = "nominal", airmass: float = 1.3):
    """S(lambda) variants for sensitivity tests: nominal (recovered), flat (filter-only synthetic photometry),
    noatm (g*QE*optics), atmX (g*QE*optics*atm(airmass))."""
    S, info, parts = C.system_S()
    if model == "nominal": return S, info
    if model == "flat": return np.ones_like(S), info
    base = info["g"] * parts["qe"] * parts["optics"]
    if model == "noatm": return base, info
    if model == "atmX": return base * C.atm_transmission(parts["comp"], airmass), info
    raise ValueError(model)


def stage_scan(out: Path, filters, scan, n_proc: int, n_eff: float, eps: float, fratio: float, inject: float = 0.0, s_model: str = "nominal", s_airmass: float = 1.3):
    S, info = system_factor(s_model, s_airmass)
    log(f"S(lambda) model '{s_model}'" + (f" (airmass {s_airmass})" if s_model == "atmX" else "") + f"; decomposition X0 = {info['X0']:.2f}, g = {info['g']:.3f}, resid rms {info['resid_rms']:.3f}")
    _G.update(S=S, std=load_channel("std", out), tile=load_channel("tile", out), cal=load_calspec(), n_eff=n_eff, eps=eps, fratio=fratio)
    if inject:
        base_table = out.parent / "shift_table.csv"
        if base_table.exists():
            bt = pd.read_csv(base_table).set_index("filter"); _G["base_shift"] = {f: float(v) for f, v in bt["delta_final"].items() if np.isfinite(v)}
            log(f"injection {inject:+.2f} nm applied on top of the fitted shifts of {base_table}")
    jobs = [(f, scan, inject) for f in filters]
    chan, units, cals, clos = [], [], [], []
    with Pool(min(n_proc, len(jobs))) as pool:
        for f, a, b, c, d in pool.imap_unordered(scan_filter, jobs):
            chan += a; units += b; cals += c; clos += d
            n_std = sum(1 for r in a if r["channel"] == "std"); n_tile = sum(1 for r in a if r["channel"] == "tile")
            log(f"  {f}: std {'yes' if n_std else 'no'}, tile {'yes' if n_tile else 'no'}, calspec stars {len({r['object'] for r in c})}")
    pd.DataFrame(chan).to_csv(out / "scan_channels.csv", index=False)
    pd.DataFrame(units).to_csv(out / "scan_units.csv", index=False)
    pd.DataFrame(cals).to_csv(out / "scan_calspec.csv", index=False)
    pd.DataFrame(clos).to_csv(out / "family_closure.csv", index=False)
    json.dump(dict(info, n_eff_scan=n_eff, eps=eps, fratio=fratio, inject=inject), open(out / "scan_params.json", "w"), indent=1)
    if clos:
        cl = pd.DataFrame(clos); r = cl[cl.form == "rigid"]
        log(f"family closure (rigid F*S vs archive): median offset {r.median_offset.abs().max():.2e}, max deviation {r.max_abs_dev.max():.2e} mag")


# --------------------------------------------------------------------------------------
# stage combine
# --------------------------------------------------------------------------------------
def refine(x, chi):
    """Parabolic refinement of the minimum; returns (x_min, sigma, chi_min) with sigma from delta chi^2 = 1."""
    x = np.asarray(x, float); chi = np.asarray(chi, float); m = np.isfinite(chi)
    x, chi = x[m], chi[m]
    if len(x) < 3: return np.nan, np.nan, np.nan
    i = int(np.argmin(chi))
    if 0 < i < len(x) - 1:
        a, b, c = np.polyfit(x[i - 1:i + 2], chi[i - 1:i + 2], 2)
        if a > 0:
            xm = -b / (2 * a)
            return float(xm), float(np.sqrt(1 / a)), float(c - b * b / (4 * a))
    return float(x[i]), np.nan, float(chi[i])


def combine_filter(f, form, ch, un, cs, sigma_a, scan_edge=5.9):
    g = ch[(ch["filter"] == f) & (ch.form == form)]
    if not len(g): return None
    deltas = np.array(sorted(g.delta.unique()))
    def series(channel):
        s = g[g.channel == channel].set_index("delta")
        return (s["a"].reindex(deltas).to_numpy(), s["a_err"].reindex(deltas).to_numpy()) if len(s) else (np.full(len(deltas), np.nan),) * 2
    a_std, ae_std = series("std"); a_tile, ae_tile = series("tile")
    c = cs[(cs["filter"] == f) & (cs.form == form)]
    stars = sorted(c.object.unique())
    D = np.full((len(stars), len(deltas)), np.nan); E = np.full_like(D, np.nan)
    for k, o in enumerate(stars):
        s = c[c.object == o].set_index("delta")
        D[k] = s["delta_star"].reindex(deltas).to_numpy(); E[k] = s["err"].reindex(deltas).to_numpy()
    chi = np.zeros(len(deltas)); n_terms = 0
    if len(stars) >= 2:
        w = 1 / (E ** 2 + SIGMA_STAR_FLOOR ** 2)
        cstar = np.nansum(w * D, axis=0) / np.nansum(w, axis=0)
        chi += np.nansum(w * (D - cstar) ** 2, axis=0); n_terms += len(stars) - 1
    for a, ae in ((a_std, ae_std), (a_tile, ae_tile)):
        if np.isfinite(a).any():
            chi += np.where(np.isfinite(a), a ** 2 / (sigma_a ** 2 + np.nan_to_num(ae) ** 2), 0.0); n_terms += 1
    if n_terms == 0: return None
    dof = max(n_terms - 1, 1)
    scale = max(float(np.nanmin(chi)) / dof, 1.0)
    d_joint, s_joint, chi_min = refine(deltas, chi / scale)
    res = dict(filter=f, form=form, n_cal_stars=len(stars), n_terms=n_terms, chi2_min=chi_min * scale, chi2_dof=float(np.nanmin(chi) / dof),
               delta_joint=d_joint, sigma_joint=s_joint, edge=bool(np.isfinite(d_joint) and abs(d_joint) >= scan_edge))
    # channel-only estimates
    res["delta_std"] = R.zero_cross(deltas, a_std) if np.isfinite(a_std).any() else np.nan
    res["delta_tile"] = R.zero_cross(deltas, a_tile) if np.isfinite(a_tile).any() else np.nan
    res["a0_std"] = float(a_std[np.argmin(np.abs(deltas))]) if np.isfinite(a_std).any() else np.nan
    res["a0_tile"] = float(a_tile[np.argmin(np.abs(deltas))]) if np.isfinite(a_tile).any() else np.nan
    # colour-term sensitivity da/dDelta near 0 (used by the validation time series)
    for name, a in (("std", a_std), ("tile", a_tile)):
        m = np.isfinite(a) & (np.abs(deltas) <= 2.0)
        res[f"dadd_{name}"] = float(np.polyfit(deltas[m], a[m], 1)[0]) if m.sum() >= 3 else np.nan
    if len(stars) >= 2:
        w = 1 / (E ** 2 + SIGMA_STAR_FLOOR ** 2)
        cstar = np.nansum(w * D, axis=0) / np.nansum(w, axis=0)
        chi_c = np.nansum(w * (D - cstar) ** 2, axis=0); sc = max(float(np.nanmin(chi_c)) / max(len(stars) - 1, 1), 1.0)
        res["delta_cal_const"], res["sigma_cal_const"], _ = refine(deltas, chi_c / sc)
        chi_a = np.nansum(w * D ** 2, axis=0); sa = max(float(np.nanmin(chi_a)) / max(len(stars), 1), 1.0)
        res["delta_cal_abs"], res["sigma_cal_abs"], _ = refine(deltas, chi_a / sa)
        da = np.array([o for o in stars]); isda = c.groupby("object").is_da.first().reindex(stars).to_numpy().astype(bool)
        if isda.sum() >= 1 and (~isda).sum() >= 1:
            da_minus = np.nanmedian(D[isda], axis=0) - np.nanmedian(D[~isda], axis=0)
            res["delta_da_zero"] = R.zero_cross(deltas, da_minus); res["da_minus_other_0"] = float(da_minus[np.argmin(np.abs(deltas))])
        res["cal_median_0"] = float(np.nanmedian(D[:, np.argmin(np.abs(deltas))]))
    # per-unit zero crossings
    u = un[(un["filter"] == f) & (un.form == form)]
    units = {}
    for (chn, tel), s in u.groupby(["channel", "telescop"]):
        s = s.set_index("delta")["a"].reindex(deltas).to_numpy()
        units[f"{chn}:{tel}"] = R.zero_cross(deltas, s)
    res["units"] = json.dumps({k: (None if not np.isfinite(v) else round(float(v), 2)) for k, v in units.items()})
    spreads = []
    for chn in ("std", "tile"):
        vals = [v for k, v in units.items() if k.startswith(chn) and np.isfinite(v) and abs(v) < scan_edge]
        res[f"unit_spread_{chn}"] = (max(vals) - min(vals)) if len(vals) >= 2 else np.nan
        if len(vals) >= 2: spreads.append(max(vals) - min(vals))
    res["unit_spread"] = max(spreads) if spreads else np.nan
    return res


def stage_combine(out: Path, sigma_a_default: float, eps: float, fratio: float):
    ch = pd.read_csv(out / "scan_channels.csv"); un = pd.read_csv(out / "scan_units.csv"); cs = pd.read_csv(out / "scan_calspec.csv")
    S, info, parts = C.system_S()
    piv = {f: C.pivot_nm(C.FINE, C.load_nominal(f)) for f in C.MEDIUM}
    rows = []
    for f in C.MEDIUM:
        for form in FORMS:
            r = combine_filter(f, form, ch, un, cs, SIGMA_A_SPECIAL.get(f, sigma_a_default))
            if r: rows.append(r)
    T = pd.DataFrame(rows)
    # form preference from the CALSPEC pattern (chi2 of the star-to-star pattern at each form's optimum)
    pref = {}
    for f in T["filter"].unique():
        t = T[T["filter"] == f].set_index("form")
        if "rigid" in t.index and "beam" in t.index and np.isfinite(t.loc["beam", "chi2_min"]) and np.isfinite(t.loc["rigid", "chi2_min"]) and t.loc["rigid", "n_cal_stars"] >= 3:
            pref[f] = "beam" if t.loc["beam", "chi2_min"] < t.loc["rigid", "chi2_min"] - 1.0 else "rigid"
        else:
            pref[f] = "rigid"
    prev_rec = pd.read_csv(C.PREV_OUT / "shift_recal" / "recal_recommended.csv").set_index("filter") if (C.PREV_OUT / "shift_recal" / "recal_recommended.csv").exists() else None
    prev_line = pd.read_csv(C.PREV_OUT / "lineonly" / "lineonly_shift.csv").set_index("filter") if (C.PREV_OUT / "lineonly" / "lineonly_shift.csv").exists() else None
    frames = C.load_frames("single"); coadds = C.load_frames("coadd")
    n_single = frames[~frames.mislabeled & ~frames.bad_zperr].groupby("FILTER").size()
    n_coadd = coadds[coadds.is_tile & ~coadds.bad_zperr].groupby("FILTER").size()
    final = []
    for f in C.MEDIUM:
        t = T[(T["filter"] == f) & (T.form == pref.get(f, "rigid"))]
        rec = dict(filter=f, pivot_nm=piv[f], form=pref.get(f, "rigid"), n_single_frames=int(n_single.get(f, 0)), n_tile_coadds=int(n_coadd.get(f, 0)))
        if len(t):
            r = t.iloc[0].to_dict(); r.pop("filter"); r.pop("form"); rec.update(r)
            sig = max(float(r["sigma_joint"]) if np.isfinite(r["sigma_joint"]) else 9.9, 0.25)
            # expected precision of a colour-term zero crossing per channel: sigma_a / |da/dDelta|
            sa = SIGMA_A_SPECIAL.get(f, sigma_a_default)
            e = {chn: (abs(sa / r[f"dadd_{chn}"]) if np.isfinite(r.get(f"dadd_{chn}", np.nan)) and r[f"dadd_{chn}"] != 0 else np.inf) for chn in ("std", "tile")}
            exc = [max(0.0, r[f"unit_spread_{chn}"] - 2 * np.sqrt(2) * e[chn]) / 2 for chn in ("std", "tile") if np.isfinite(r.get(f"unit_spread_{chn}", np.nan))]
            if np.isfinite(r["delta_std"]) and np.isfinite(r["delta_tile"]):
                exc.append(max(0.0, abs(r["delta_std"] - r["delta_tile"]) - 2 * np.sqrt(e["std"] ** 2 + e["tile"] ** 2)) / 2)
            # colour-term channel combined (std, tile) and its expected precision; disagreement with CALSPEC -> half difference
            cvals = [(r[f"delta_{chn}"], e[chn]) for chn in ("std", "tile") if np.isfinite(r[f"delta_{chn}"]) and np.isfinite(e[chn])]
            if cvals:
                wc = np.array([1 / max(ec, 0.05) ** 2 for _, ec in cvals]); rec["delta_C"] = float(np.sum(wc * np.array([v for v, _ in cvals])) / wc.sum()); rec["e_C"] = float(1 / np.sqrt(wc.sum()))
            else:
                rec["delta_C"] = np.nan; rec["e_C"] = np.nan
            if np.isfinite(r.get("delta_cal_const", np.nan)) and np.isfinite(rec["delta_C"]) and r.get("n_cal_stars", 0) >= 3:
                exc.append(abs(r["delta_cal_const"] - rec["delta_C"]) / 2)
            sig = max([sig] + exc)
            rec["delta_meas"] = float(r["delta_joint"]); rec["sigma_meas"] = float(sig); rec["sigma_excess"] = float(max(exc) if exc else 0.0)
            if r["edge"]: rec["delta_meas"] = np.nan; rec["sigma_meas"] = np.nan
        else:
            rec["delta_meas"] = np.nan; rec["sigma_meas"] = np.nan
        rec["prev_recommended"] = float(prev_rec.loc[f, "recommended"]) if prev_rec is not None and f in prev_rec.index else np.nan
        rec["prev_lineonly"] = float(prev_line.loc[f, "shift_abs"]) if prev_line is not None and f in prev_line.index and "shift_abs" in prev_line else np.nan
        final.append(rec)
    F = pd.DataFrame(final)
    # prior: Delta = k * pivot over the well-measured filters
    ok = F[np.isfinite(F.delta_meas) & (F.sigma_meas <= PRIOR_MAX_SIGMA) & (F.pivot_nm >= PRIOR_MIN_PIVOT) & ~F["filter"].isin(NO_PRIOR)
           & (F.get("n_cal_stars", pd.Series(0, index=F.index)).fillna(0) >= 3)]
    x = ok.pivot_nm.to_numpy(); y = ok.delta_meas.to_numpy(); w = 1 / ok.sigma_meas.to_numpy() ** 2
    k = float(np.sum(w * x * y) / np.sum(w * x * x)) if len(ok) >= 3 else np.nan
    for _ in range(5):   # Huber reweighting
        r = y - k * x; s = max(R.mad(r), 1e-3); hw = np.where(np.abs(r) <= 1.345 * s, 1.0, 1.345 * s / np.abs(r))
        k = float(np.sum(w * hw * x * y) / np.sum(w * hw * x * x))
    resid = y - k * x
    tau = max(float(np.sqrt(np.mean(resid ** 2))), TAU_FLOOR) if len(ok) >= 3 else np.nan
    k_err = float(np.sqrt(1 / np.sum(w * x * x))) if len(ok) >= 3 else np.nan
    n_eff = C.n_eff_from_zeta(k, eps, fratio) if np.isfinite(k) and k < 0 else np.nan
    log(f"prior: k = {k*100:+.3f} +- {k_err*100:.3f} % (n_eff {n_eff:.2f} at eps {eps}), tau = {tau:.2f} nm from {len(ok)} filters")
    # final values
    out_rows = []
    for _, r in F.iterrows():
        f = r["filter"]; prior = k * r.pivot_nm; prior_sig = np.sqrt(tau ** 2 + (k_err * r.pivot_nm) ** 2)
        flags = []
        if np.isfinite(r.delta_meas):
            if f in NO_PRIOR:
                d, s = r.delta_meas, r.sigma_meas; flags.append("no shrinkage (sign opposite to the beam model)")
            else:
                wm, wp = 1 / r.sigma_meas ** 2, 1 / prior_sig ** 2
                d, s = (wm * r.delta_meas + wp * prior) / (wm + wp), 1 / np.sqrt(wm + wp)
            has_line = np.isfinite(r.get("delta_cal_const", np.nan)) and r.get("n_cal_stars", 0) >= 3
            tol = max(0.5, 2 * np.sqrt(np.nan_to_num(r.get("sigma_cal_const", np.nan), nan=0.5) ** 2 + np.nan_to_num(r.get("e_C", np.nan), nan=9.0) ** 2))
            consistent = (not has_line) or (not np.isfinite(r.get("delta_C", np.nan))) or abs(r.delta_cal_const - r.delta_C) <= tol
            if has_line and r.sigma_meas <= 0.5 and consistent: cls = "determined"
            elif r.sigma_meas > 1.5 or (r.n_single_frames + r.n_tile_coadds) < 10: cls = "undetermined"
            else: cls = "prior+continuum"
            if has_line and not consistent: flags.append("channels disagree (CALSPEC vs field colour term)")
            if r.n_cal_stars == 0 or not has_line: flags.append("colour-term channel only")
            if np.isfinite(r.unit_spread) and r.unit_spread > 1.0: flags.append(f"unit-to-unit spread {r.unit_spread:.1f} nm")
            if r.n_single_frames == 0: flags.append("no standard-field frames (tile coadds only)")
            if f in SIGMA_A_SPECIAL: flags.append("XP blue end: colour-term floor 0.010")
        else:
            d, s, cls = prior, prior_sig, "predicted"
            if (r.n_single_frames + r.n_tile_coadds) > 0 and bool(r.get("edge", False)):
                flags.append("measurement runs to the scan edge (unconstrained by the data): prior k*pivot only")
            else:
                flags.append("no usable frames: predicted from the prior k*pivot")
        if f == "m425": flags.append("redward: offset of the nominal curve itself, not a beam effect")
        out_rows.append(dict(r.to_dict(), prior_delta=prior, prior_sigma=prior_sig, delta_final=float(d), sigma_final=float(s), cls=cls, flags="; ".join(flags)))
    F2 = pd.DataFrame(out_rows)
    F2.to_csv(out / "shift_table.csv", index=False)
    T.to_csv(out / "combine_forms.csv", index=False)
    params = dict(k=k, k_err=k_err, tau_nm=tau, n_eff=n_eff, eps=eps, fratio=fratio, X0=info["X0"], g=info["g"], S_resid_rms=info["resid_rms"],
                  sigma_a_default=sigma_a_default, sigma_star_floor=SIGMA_STAR_FLOOR, n_prior_filters=int(len(ok)), form_preference=pref)
    json.dump(params, open(out / "model_params.json", "w"), indent=1)
    cols = ["filter", "pivot_nm", "form", "cls", "delta_meas", "sigma_meas", "delta_std", "delta_tile", "delta_cal_const", "delta_cal_abs", "delta_final", "sigma_final", "prev_recommended", "prev_lineonly", "unit_spread"]
    print(F2[cols].round(2).to_string(index=False))
    return F2, params


# --------------------------------------------------------------------------------------
# stage write
# --------------------------------------------------------------------------------------
def stage_write(out: Path, eps: float, fratio: float):
    F = pd.read_csv(out / "shift_table.csv"); P = json.load(open(out / "model_params.json"))
    S, info, parts = C.system_S()
    n_eff = P["n_eff"] if np.isfinite(P.get("n_eff", np.nan)) else C.N_EFF_PROVISIONAL
    for sub in ("nm", "AA", "unc", "alt", "meta"): (out / "curves" / sub).mkdir(parents=True, exist_ok=True)
    C.write_curve_nm(out / "curves" / "S_system.csv", S, dict(qe=parts["qe"], optics=parts["optics"], atm_X0=parts["atm_X0"], model=parts["model"]))
    std = load_channel("std", out)
    closure = []
    for _, r in F.iterrows():
        f = r["filter"]; fam = C.CurveFamily(f, S, n_eff=n_eff, eps=eps, fratio=fratio)
        main = r["form"] if isinstance(r["form"], str) else "rigid"; other = "beam" if main == "rigid" else "rigid"
        d, s = float(r.delta_final), float(r.sigma_final)
        T = fam.curve(d, main); lo = np.minimum(fam.curve(d - s, main), fam.curve(d + s, main)); hi = np.maximum(fam.curve(d - s, main), fam.curve(d + s, main))
        C.write_curve_nm(out / "curves" / "nm" / f"{f}.csv", T)
        C.write_curve_AA(out / "curves" / "AA" / f"{f}.csv", T)
        C.write_curve_nm(out / "curves" / "unc" / f"{f}.csv", T, dict(trans_lo=lo, trans_hi=hi))
        C.write_curve_nm(out / "curves" / "alt" / f"{f}_{other}.csv", fam.curve(d, other))
        meta = dict(filter=f, nominal_source="filter_only (doabdi, 0.2 nm) x S = Final_transmission / filter_only", pivot_nominal_nm=round(fam.pivot, 2),
                    pivot_effective_nm=round(C.pivot_nm(C.FINE, T), 2), shift_nm=round(d, 3), shift_sigma_nm=round(s, 3), form=main,
                    cls=r.cls, delta_measured_nm=None if not np.isfinite(r.delta_meas) else round(float(r.delta_meas), 3),
                    sigma_measured_nm=None if not np.isfinite(r.sigma_meas) else round(float(r.sigma_meas), 3),
                    delta_std_nm=None if not np.isfinite(r.get("delta_std", np.nan)) else round(float(r.delta_std), 3),
                    delta_tile_nm=None if not np.isfinite(r.get("delta_tile", np.nan)) else round(float(r.delta_tile), 3),
                    delta_calspec_nm=None if not np.isfinite(r.get("delta_cal_const", np.nan)) else round(float(r.delta_cal_const), 3),
                    per_unit_shift_nm=r.get("units", ""), prior_k_percent=round(P["k"] * 100, 4), prior_k_err_percent=round(P["k_err"] * 100, 4), prior_tau_nm=round(P["tau_nm"], 3),
                    n_eff=round(n_eff, 3), eps=eps, fratio=fratio, X0=round(P["X0"], 3), X_ref="X0 (same atmosphere as the nominal curves)",
                    n_single_frames=int(r.n_single_frames), n_tile_coadds=int(r.n_tile_coadds), n_calspec_stars=int(r.get("n_cal_stars", 0) if np.isfinite(r.get("n_cal_stars", np.nan)) else 0),
                    flags=r.flags if isinstance(r.flags, str) else "", epoch="2024-07 .. 2026-07", date=pd.Timestamp.now().strftime("%Y-%m-%d"),
                    scripts="ezphot/analysis/260910_effcurve_{common,fit,validate}.py")
        C.write_meta(out / "curves" / "meta" / f"{f}.json", meta)
        # closure: the written nm file must reproduce the in-memory family magnitudes
        if std is not None:
            rows = std["st"][std["st"]["filter"] == f]
            if len(rows):
                u_rows = np.unique(rows.row.to_numpy())[:200]
                w2, t2 = R.load_curve(f) if False else (None, None)
                d2 = pd.read_csv(out / "curves" / "nm" / f"{f}.csv"); fw = d2["lam"].to_numpy(float); ft = d2["trans"].to_numpy(float)
                m_file = R.syn_mag(std["wave"], std["flux"][u_rows], fw, ft, 0.0); m_mem = R.syn_mag(std["wave"], std["flux"][u_rows], C.FINE, T, 0.0)
                closure.append(dict(filter=f, max_abs_diff=float(np.nanmax(np.abs(m_file - m_mem)))))
    pd.DataFrame(closure).to_csv(out / "curves" / "closure_check.csv", index=False)
    if closure: log(f"closure: written files reproduce the in-memory magnitudes to {max(c['max_abs_diff'] for c in closure):.2e} mag")
    meta_all = [json.load(open(out / "curves" / "meta" / f"{f}.json")) for f in F["filter"]]
    pd.DataFrame(meta_all).to_csv(out / "curves" / "curves_metadata.csv", index=False)
    # repository copy (nm curves + metadata)
    C.REPO_CURVES.mkdir(exist_ok=True)
    for f in F["filter"]:
        shutil.copy(out / "curves" / "nm" / f"{f}.csv", C.REPO_CURVES / f"{f}.csv")
    shutil.copy(out / "curves" / "curves_metadata.csv", C.REPO_CURVES / "curves_metadata.csv")
    shutil.copy(out / "curves" / "S_system.csv", C.REPO_CURVES / "S_system.csv")
    shutil.copy(out / "model_params.json", C.REPO_CURVES / "model_params.json")
    (C.REPO_CURVES / "README.md").write_text(
        "# 7DT effective medium-band transmission curves (2026-09-10)\n\n"
        "One file per filter (`lam` nm, `trans`): filter-only curve shifted by the fitted effective shift and multiplied by the\n"
        "filter-independent system factor S(lambda) (QE x optics x atmosphere at airmass X0, same as the nominal curves).\n"
        "`curves_metadata.csv` gives the shift, its uncertainty, the class (determined / prior+continuum / undetermined / predicted)\n"
        "and flags per filter; `model_params.json` the global prior (k = dlambda/lambda, tau) and the beam model (n_eff, eps).\n"
        f"Full products (AA format, uncertainty bands, alternative form, scans, validation): {C.DEFAULT_OUT}\n"
        "Generated by ezphot/analysis/260910_effcurve_fit.py; validation by 260910_effcurve_validate.py.\n", encoding="utf-8")
    log(f"curves written to {out / 'curves'} and copied to {C.REPO_CURVES}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all", choices=["extract", "scan", "combine", "write", "all"])
    ap.add_argument("--out-dir", default=str(C.DEFAULT_OUT))
    ap.add_argument("--filters", default="", help="comma-separated filters (default: all 37 medium bands)")
    ap.add_argument("--n-proc", type=int, default=16)
    ap.add_argument("--scan", default="", help="min,max,step in nm (default -6,3,0.25)")
    ap.add_argument("--sigma-a", type=float, default=SIGMA_A_DEFAULT, help="systematic floor of the field colour term [mag/mag]")
    ap.add_argument("--eps", type=float, default=C.EPS_DEFAULT, help="central obstruction ratio (diameter)")
    ap.add_argument("--fratio", type=float, default=C.FRATIO_DEFAULT)
    ap.add_argument("--n-eff", type=float, default=C.N_EFF_PROVISIONAL, help="effective index used for the beam family in the scan")
    ap.add_argument("--inject", type=float, default=0.0, help="injection test: shift the observations by this many nm (outputs in inject_<x>/)")
    ap.add_argument("--s-model", default="nominal", choices=["nominal", "flat", "noatm", "atmX"], help="system factor S used in the scan (sensitivity tests; outputs in svar_<model>/)")
    ap.add_argument("--s-airmass", type=float, default=1.3, help="airmass of the atmosphere for --s-model atmX")
    args = ap.parse_args()
    out = Path(args.out_dir)
    if args.inject: out = out / f"inject_{args.inject:+.2f}"
    if args.s_model != "nominal": out = out / (f"svar_{args.s_model}" + (f"_{args.s_airmass:.1f}" if args.s_model == "atmX" else ""))
    out.mkdir(parents=True, exist_ok=True)
    filters = [f.strip() for f in args.filters.split(",") if f.strip()] or list(C.MEDIUM)
    scan = SCAN_DEFAULT
    if args.scan:
        lo, hi, st = (float(v) for v in args.scan.split(",")); scan = np.round(np.arange(lo, hi + 1e-9, st), 3)
    stages = ["extract", "scan", "combine", "write"] if args.stage == "all" else [args.stage]
    base = Path(args.out_dir)
    for s in stages:
        log(f"=== stage {s} ===")
        if s == "extract":
            if (base / "tilestar_spectra.npz").exists(): log("tilestar_spectra.npz exists, skipped (delete to redo)")
            else: stage_extract(base, args.n_proc)
        elif s == "scan":
            if args.inject and not (out / "tilestar_spectra.npz").exists() and (base / "tilestar_spectra.npz").exists():
                (out / "tilestar_spectra.npz").symlink_to(base / "tilestar_spectra.npz")
            if args.s_model != "nominal" and not (out / "tilestar_spectra.npz").exists() and (base / "tilestar_spectra.npz").exists():
                (out / "tilestar_spectra.npz").symlink_to(base / "tilestar_spectra.npz")
            stage_scan(out, filters, scan, args.n_proc, args.n_eff, args.eps, args.fratio, args.inject, args.s_model, args.s_airmass)
        elif s == "combine":
            stage_combine(out, args.sigma_a, args.eps, args.fratio)
        elif s == "write":
            stage_write(out, args.eps, args.fratio)


if __name__ == "__main__":
    main()
