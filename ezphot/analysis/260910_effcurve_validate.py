#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260910_effcurve_validate.py

Validation of the effective curves written by 260910_effcurve_fit.py, using the *written* curve files:
  * closed loop on the standard-field frames and the tile coadds (colour term, ZP scatter, per unit) before/after
  * CALSPEC per-star residuals before/after (DA - other, between-star MAD, held-out continuum stars)
  * single-frame time series of the colour-term shift estimate per (unit, filter): drift, airmass slope, field position
  * gate (header ZP reproduction), injection tests (if inject_*/ directories exist), XP wavelength-scale check
  * validation table V1-V13, report.md and figures F1-F9

  MPLCONFIGDIR=/tmp/ezphot-matplotlib /home/hhchoi1022/anaconda3/envs/py311/bin/python \\
      ezphot/analysis/260910_effcurve_validate.py
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                         # noqa: E402
import pandas as pd                        # noqa: E402
from astropy.table import Table            # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
C = importlib.import_module("260910_effcurve_common")
FIT = importlib.import_module("260910_effcurve_fit")
L = importlib.import_module("260909_zpcolor_lineonly_shift")
R = C.R
warnings.filterwarnings("ignore")
log = C.log

DA_FILTERS = ["m425", "m450", "m475", "m500", "m650", "m675"]
CALSPEC_XP = Path("/lyman/data1/Calibration/latest_CALSPEC/XP_SAMPLED_COMBINED.csv")


def load_written(out: Path):
    curves = {}
    for p in sorted((out / "curves" / "nm").glob("*.csv")):
        d = pd.read_csv(p); curves[p.stem] = np.interp(C.FINE, d["lam"].to_numpy(float), d["trans"].to_numpy(float), left=0.0, right=0.0)
    return curves


def closed_loop(out: Path, curves, table):
    """Before/after diagnostics per filter and per unit for both channels, plus CALSPEC."""
    std = FIT.load_channel("std", out); tile = FIT.load_channel("tile", out); cal = FIT.load_calspec()
    rows, urows, crows = [], [], []
    for f in C.MEDIUM:
        if f not in curves: continue
        nom = C.load_nominal(f); eff = curves[f]
        for kind, ch in (("std", std), ("tile", tile)):
            if ch is None: continue
            rs = ch["st"][ch["st"]["filter"] == f]
            if not len(rs): continue
            before, ub, med0, zp0 = FIT.channel_diagnostics(rs, ch["flux"], ch["wave"], nom, nom)
            after, ua, med1, _ = FIT.channel_diagnostics(rs, ch["flux"], ch["wave"], nom, eff)
            rows.append(dict(filter=f, channel=kind, n_frames=before["n_frames"], n_stars=before["n_stars"], a_before=before["a"], a_after=after["a"], a_err=after["a_err"],
                             zp_std_before=before["zp_std"], zp_std_after=after["zp_std"], dzp=after["dzp"]))
            for b, a in zip(ub, ua):
                urows.append(dict(filter=f, channel=kind, telescop=a["telescop"], n_frames=a["n_frames"], a_before=b["a"], a_after=a["a"], zp_std_before=b["zp_std"], zp_std_after=a["zp_std"]))
            if kind == "std" and cal["spectra"]:
                cf = cal["cal"][cal["cal"]["filter"] == f]
                if len(cf):
                    cal0 = {o: float(R.syn_mag(*cal["spectra"][o], C.FINE, nom, 0.0)) for o in cal["spectra"]}
                    p0, _ = FIT.calspec_residuals(cf, cal["spectra"], cal["is_da"], nom, nom, med0, zp0, cal0)
                    p1, _ = FIT.calspec_residuals(cf, cal["spectra"], cal["is_da"], nom, eff, med1, zp0, cal0)
                    for o in p0.index:
                        crows.append(dict(filter=f, object=o, is_da=bool(p0.loc[o, "is_da"]), n=int(p0.loc[o, "n"]), before=float(p0.loc[o, "delta"]),
                                          after=float(p1.loc[o, "delta"]) if o in p1.index else np.nan, err=float(p0.loc[o, "err"])))
        log(f"  {f}: closed loop done")
    return pd.DataFrame(rows), pd.DataFrame(urows), pd.DataFrame(crows)


def calspec_summary(cr, sline):
    rows = []
    for f, g in cr.groupby("filter"):
        da = g[g.is_da]; ot = g[~g.is_da]
        cont = g[g.object.map(lambda o: sline.get((f, o), np.nan) <= 0.003)]
        rows.append(dict(filter=f, n_stars=len(g), da_minus_other_before=float(da.before.median() - ot.before.median()) if len(da) and len(ot) else np.nan,
                         da_minus_other_after=float(da.after.median() - ot.after.median()) if len(da) and len(ot) else np.nan,
                         mad_before=R.mad(g.before) if len(g) >= 3 else np.nan, mad_after=R.mad(g.after) if len(g) >= 3 else np.nan,
                         median_all_before=float(g.before.median()), median_all_after=float(g.after.median()),
                         n_cont=len(cont), cont_median_before=float(cont.before.median()) if len(cont) else np.nan, cont_median_after=float(cont.after.median()) if len(cont) else np.nan))
    return pd.DataFrame(rows)


def time_series(out: Path, table):
    """Single-frame colour terms (2026-09-08 refit, CORR_LAMOST, APER_3) converted to shift estimates with the
    colour-term sensitivity of each filter."""
    forms = pd.read_csv(out / "combine_forms.csv")
    dadd = {}
    for f, g in forms.groupby("filter"):
        g = g[g.form == table.set_index("filter").loc[f, "form"]] if f in table["filter"].values else g
        v = g["dadd_std"].dropna(); v2 = g["dadd_tile"].dropna()
        dadd[f] = float(v.iloc[0]) if len(v) else (float(v2.iloc[0]) if len(v2) else np.nan)
    frames = C.load_frames("single")
    rf = pd.read_csv(C.PREV_OUT / "refit_frames.csv", low_memory=False)
    rf = rf[(rf.catalog == R.CATALOG) & (rf.aper == "APER_3") & (rf.status == "ok")]
    rf = rf.merge(frames[["file", "mislabeled", "bad_zperr", "is_tile", "CRVAL1", "CRVAL2"]], on="file", how="left")
    rf = rf[~rf.mislabeled & ~rf.bad_zperr & rf["filter"].isin(dadd)]
    rf["dadd"] = rf["filter"].map(dadd)
    rf = rf[rf["dadd"].abs() >= 0.004].copy()          # only filters whose colour term is sensitive to the shift (<= ~1 nm per frame)
    rf["delta_frame"] = -rf["a_bprp"] / rf["dadd"]
    rf["delta_frame_err"] = rf["a_bprp_err"].abs() / rf["dadd"].abs()
    rf["flag_mismatch"] = (rf["zp_std"] > 0.08) | (rf["delta_frame"].abs() > 8.0)
    rf["night"] = np.floor(rf["mjd"] - 0.5).astype(int)
    rows = []
    for (u, f), g in rf[~rf.flag_mismatch].groupby(["telescop", "filter"]):
        if len(g) < 20 or not np.isfinite(g.delta_frame).all(): continue
        w = 1 / np.maximum(g.delta_frame_err, 0.2) ** 2
        mean = float(np.sum(w * g.delta_frame) / np.sum(w)); scatter = R.mad(g.delta_frame)
        nights = g.groupby("night").delta_frame.median(); night_rms = float(nights.std()) if len(nights) >= 3 else np.nan
        slope_t = slope_x = err_t = err_x = np.nan
        if g.night.nunique() >= 3 and np.ptp(g.mjd) > 30:
            (p, cov) = np.polyfit(g.mjd / 365.25, g.delta_frame, 1, cov=True); slope_t, err_t = float(p[0]), float(np.sqrt(cov[0, 0]))
        if np.ptp(g.airmass) > 0.3:
            (p, cov) = np.polyfit(g.airmass, g.delta_frame, 1, cov=True); slope_x, err_x = float(p[0]), float(np.sqrt(cov[0, 0]))
        inner_outer = float((g.a_bprp_inner - g.a_bprp_outer).median() / -dadd[f]) if "a_bprp_inner" in g and g.a_bprp_inner.notna().sum() >= 5 else np.nan
        rows.append(dict(telescop=u, filter=f, n_frames=len(g), n_nights=int(g.night.nunique()), span_days=float(np.ptp(g.mjd)), tiles=int(g.is_tile.sum()),
                         delta_mean=mean, delta_mad=scatter, night_rms=night_rms, slope_nm_per_yr=slope_t, slope_err=err_t,
                         slope_nm_per_airmass=slope_x, slope_x_err=err_x, inner_minus_outer_nm=inner_outer))
    return rf, pd.DataFrame(rows)


def xp_wavelength_check(cal):
    """Cross-correlate the Gaia XP sampled spectra of the CALSPEC DA stars with the CALSPEC spectra convolved with a
    Gaussian LSF around H-beta and H-alpha; returns the median lag (nm) = XP wavelength scale relative to CALSPEC."""
    if not CALSPEC_XP.exists(): return pd.DataFrame()
    xp = pd.read_csv(CALSPEC_XP)
    info = pd.read_csv(Path(__file__).resolve().parent / "calspec_info.csv")
    norm = lambda s: str(s).replace(" ", "").replace("_", "").upper()
    rows = []
    for o, (wl, fl) in cal["spectra"].items():
        if not cal["is_da"].get(o, False): continue
        r = info[(info["star_name_1"].map(norm) == norm(o)) | (info["calv_name"].map(norm) == norm(o)) | (info["star_name_2"].map(norm) == norm(o))]
        if not len(r): continue
        ra, dec = float(r["ra_j2000_deg"].iloc[0]), float(r["dec_j2000_deg"].iloc[0])
        g = xp.groupby("source_id").first()
        d = np.hypot((g.ra - ra) * np.cos(np.radians(dec)), g.dec - dec) * 3600
        if d.min() > 15: continue
        sid = d.idxmin(); s = xp[xp.source_id == sid].sort_values("wavelength")
        xw, xf = s.wavelength.to_numpy(float), s.flux.to_numpy(float)
        for fwhm in (6.0, 8.0, 10.0, 12.0, 15.0):
            grid = np.arange(400, 700, 0.5); f0 = np.interp(grid, wl, fl)
            sig = fwhm / 2.3548; k = np.exp(-0.5 * (np.arange(-4 * sig, 4 * sig + 0.5, 0.5) / sig) ** 2); k /= k.sum()
            conv = np.convolve(f0, k, mode="same")
            for name, lo, hi in (("Hbeta", 455, 515), ("Halpha", 625, 690)):
                m = (xw >= lo) & (xw <= hi)
                best = (np.nan, -np.inf)
                for lag in np.arange(-1.5, 1.51, 0.05):
                    model = np.interp(xw[m] - lag, grid, conv)
                    c = np.corrcoef(np.log(np.clip(xf[m], 1e-30, None)), np.log(np.clip(model, 1e-30, None)))[0, 1]
                    if c > best[1]: best = (float(lag), float(c))
                rows.append(dict(object=o, source_id=str(sid), fwhm=fwhm, line=name, lag_nm=best[0], corr=best[1]))
    T = pd.DataFrame(rows)
    if not len(T): return T
    # keep the LSF width that correlates best per star and line
    T = T.sort_values("corr", ascending=False).drop_duplicates(["object", "line"])
    return T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(C.DEFAULT_OUT))
    args = ap.parse_args()
    out = Path(args.out_dir); fig_dir = out / "figures"; fig_dir.mkdir(exist_ok=True)
    table = pd.read_csv(out / "shift_table.csv"); P = json.load(open(out / "model_params.json"))
    sens = out / "sens_sigma005" / "shift_table.csv"
    if sens.exists():
        sb = pd.read_csv(sens).set_index("filter")["delta_final"]
        table["delta_final_sa005"] = table["filter"].map(sb)
        table["sigma_a_sensitivity_nm"] = (table["delta_final_sa005"] - table["delta_final"]).abs()
        meta = out / "curves" / "curves_metadata.csv"
        if meta.exists():
            m = pd.read_csv(meta); m["shift_nm_sigma_a_0.005"] = m["filter"].map(sb).round(3); m["sigma_a_sensitivity_nm"] = (m["shift_nm_sigma_a_0.005"] - m["shift_nm"]).abs().round(3)
            m.to_csv(meta, index=False); m.to_csv(C.REPO_CURVES / "curves_metadata.csv", index=False)
    curves = load_written(out)
    log(f"{len(curves)} written curves loaded")
    S, info, parts = C.system_S()

    # ---- closed loop with the written curves
    ch, un, cr = closed_loop(out, curves, table)
    ch.to_csv(out / "validation_channels.csv", index=False); un.to_csv(out / "validation_units.csv", index=False); cr.to_csv(out / "validation_calspec_stars.csv", index=False)
    sline = {}
    p = C.PREV_OUT / "lineonly" / "lineonly_stars.csv"
    if p.exists():
        ls = pd.read_csv(p); sline = {(f, o): v for f, o, v in zip(ls["filter"], ls["object"], ls["s_line"])}
    cs = calspec_summary(cr, sline) if len(cr) else pd.DataFrame()
    cs.to_csv(out / "validation_calspec.csv", index=False)

    # ---- time series
    rf, ts = time_series(out, table)
    rf.to_csv(out / "validation_frames.csv.gz", index=False, compression="gzip"); ts.to_csv(out / "validation_timeseries.csv", index=False)

    # ---- gate V1 from the 2026-09-08 refit
    rfa = pd.read_csv(C.PREV_OUT / "refit_frames.csv", low_memory=False)
    rfa = rfa[(rfa.catalog == R.CATALOG) & (rfa.aper == "APER_3") & (rfa.status == "ok")]
    gate = float(((rfa.zp_median - rfa.zp_header).abs() <= 0.002).mean())

    # ---- injection tests
    inj_rows = []
    for d in sorted(out.glob("inject_*")):
        if not (d / "shift_table.csv").exists(): continue
        x = float(d.name.split("_")[1]); t = pd.read_csv(d / "shift_table.csv").set_index("filter")
        for f in t.index:
            if f in table.set_index("filter").index and np.isfinite(t.loc[f, "delta_meas"]) and np.isfinite(table.set_index("filter").loc[f, "delta_meas"]):
                inj_rows.append(dict(filter=f, injected=x, recovered=float(t.loc[f, "delta_meas"] - table.set_index("filter").loc[f, "delta_meas"])))
    inj = pd.DataFrame(inj_rows); inj.to_csv(out / "validation_injection.csv", index=False)

    # ---- XP wavelength scale
    cal = FIT.load_calspec()
    xw = xp_wavelength_check(cal); xw.to_csv(out / "validation_xp_wavelength.csv", index=False)

    # ---- validation table
    T = table.set_index("filter")
    chs = ch[ch.channel == "std"].set_index("filter") if len(ch) else pd.DataFrame(); cht = ch[ch.channel == "tile"].set_index("filter") if len(ch) else pd.DataFrame()
    csi = cs.set_index("filter") if len(cs) else pd.DataFrame()
    V = []
    def add(code, name, value, thresh, ok, note=""):
        V.append(dict(code=code, test=name, value=value, threshold=thresh, passed=bool(ok), note=note))
    add("V1", "header ZP reproduced within 0.002 mag (fraction of frames)", round(gate, 3), ">= 0.98", gate >= 0.98)
    if len(inj):
        bias = float((inj.recovered - inj.injected).abs().median()); add("V2", "injection recovery |bias| median [nm]", round(bias, 2), "<= 0.2", round(bias, 2) <= 0.2, f"{inj['filter'].nunique()} filters, {inj.injected.nunique()} amplitudes")
    else:
        add("V2", "injection recovery", np.nan, "<= 0.2", False, "no inject_* runs found")
    if len(chs):
        worst = chs.a_after.abs(); lim = pd.Series({f: (0.02 if f in FIT.SIGMA_A_SPECIAL else 0.005) for f in chs.index})
        fails = [f for f in chs.index if worst[f] > lim[f]]
        add("V3a", "std colour term after |a| <= 0.005 (0.02 for the XP blue end)", round(float(worst.max()), 4), "all filters", len(fails) == 0, "fails: " + ",".join(fails) if fails else "")
    if len(cht):
        fails = [f for f in cht.index if abs(cht.a_after[f]) > (0.02 if f in FIT.SIGMA_A_SPECIAL else 0.008)]
        add("V3b", "tile colour term after |a| <= 0.008", round(float(cht.a_after.abs().max()), 4), "all filters", len(fails) == 0, "fails: " + ",".join(fails) if fails else "")
    if len(chs):
        ratio = chs.zp_std_after / chs.zp_std_before; fails = [f for f in chs.index if ratio[f] > 1.02]
        add("V4", "std ZP scatter after/before <= 1.02", round(float(ratio.max()), 3), "all filters", len(fails) == 0, "fails: " + ",".join(fails) if fails else f"median ratio {ratio.median():.3f}")
    if len(csi):
        sub = csi.loc[[f for f in DA_FILTERS if f in csi.index]]
        worst = float(sub.da_minus_other_after.abs().max()) if len(sub) else np.nan
        add("V5a", "CALSPEC |DA - other| after <= 0.010 (m425-m500, m650-m675)", round(worst, 4), "<= 0.010", worst <= 0.010, "; ".join(f"{f} {sub.da_minus_other_before[f]:+.3f}->{sub.da_minus_other_after[f]:+.3f}" for f in sub.index))
        m4 = csi[csi.n_stars >= 4]; worst = float(m4.mad_after.max()) if len(m4) else np.nan
        add("V5b", "CALSPEC between-star MAD after <= 0.012 (>= 4 stars)", round(worst, 4), "<= 0.012", worst <= 0.012, "fails: " + ",".join(m4.index[m4.mad_after > 0.012]))
        dc = csi.cont_median_after.abs(); worst = float(dc.max()); fails = [f"{f} {v:+.3f}" for f, v in csi.cont_median_after.items() if abs(v) > 0.010]
        add("V5c", "continuum CALSPEC stars: |median residual after| <= 0.010 (grey term)", round(worst, 4), "<= 0.010", worst <= 0.010, "; ".join(fails))
    if "sigma_excess" in T:
        ex = T.sigma_excess.dropna(); fails = [f"{f} {v:.1f}" for f, v in ex.items() if v > 0.5]
        add("V6", "unit/tile/channel disagreement beyond the expected precision (half difference) <= 0.5 nm", round(float(ex.max()), 2) if len(ex) else np.nan, "<= 0.5", len(fails) == 0, "; ".join(fails))
    both = T[np.isfinite(T.delta_std) & np.isfinite(T.delta_tile)]
    if len(both):
        d = (both.delta_std - both.delta_tile).abs(); lim = np.where(both.sigma_meas <= 0.5, 1.0, 2.0); fails = [f for f, v, l in zip(both.index, d, lim) if v > l]
        add("V7", "tile vs standard-field shift agreement", round(float(d.max()), 2), "<= 1 nm (sigma<=0.5) / 2 nm", len(fails) == 0, "fails: " + ",".join(fails) if fails else "")
    if len(ts):
        t2 = ts[np.isfinite(ts.slope_nm_per_yr)]; sig = (t2.slope_nm_per_yr.abs() > 1.0 + 2 * t2.slope_err)
        add("V8a", "time drift |d(shift)/dt| <= 1 nm/yr at 2 sigma", int(sig.sum()), "0 (unit,filter) pairs", sig.sum() == 0, f"{len(t2)} pairs with >30 d baselines; " + "; ".join(f"{u}/{f} {s:+.1f}+-{e:.1f}" for u, f, s, e in zip(t2.telescop[sig], t2["filter"][sig], t2.slope_nm_per_yr[sig], t2.slope_err[sig])))
        nr = ts.night_rms.dropna(); add("V8b", "night-to-night rms of the shift estimate <= 1.5 nm (median over pairs; nights = different fields)", round(float(nr.median()), 2) if len(nr) else np.nan, "<= 1.5", len(nr) > 0 and nr.median() <= 1.5, f"{int((nr > 1.5).sum())} of {len(nr)} pairs above 1.5")
        t3 = ts[np.isfinite(ts.slope_nm_per_airmass)]; sig = (t3.slope_nm_per_airmass.abs() > 0.3 + 2 * t3.slope_x_err)
        add("V9", "airmass slope <= 0.3 nm/airmass at 2 sigma", int(sig.sum()), "0 pairs", sig.sum() == 0, "; ".join(f"{u}/{f} {s:+.2f}" for u, f, s in zip(t3.telescop[sig], t3["filter"][sig], t3.slope_nm_per_airmass[sig])))
        io = ts.inner_minus_outer_nm.dropna(); add("V10", "field position: |inner - outer| median <= 0.3 nm", round(float(io.abs().median()), 2) if len(io) else np.nan, "<= 0.3", len(io) > 0 and io.abs().median() <= 0.3)
    both = T[np.isfinite(T.get("delta_cal_const", np.nan)) & np.isfinite(T.get("delta_C", np.nan)) & (T.n_cal_stars >= 3)]
    if len(both):
        d = (both.delta_cal_const - both.delta_C).abs()
        lim = np.maximum(0.5, 2 * np.sqrt(both.sigma_cal_const.fillna(0.5) ** 2 + both.e_C.fillna(9.0) ** 2)); fails = [f"{f} ({v:.1f} > {l:.1f})" for f, v, l in zip(both.index, d, lim) if v > l]
        add("V11", "CALSPEC pattern vs field colour-term channel: |difference| <= max(0.5, 2 sqrt(sigma_cal^2 + e_C^2))", round(float(d.max()), 2), "per filter", len(fails) == 0, "fails: " + ", ".join(fails) if fails else "; ".join(f"{f} {v:+.1f}" for f, v in (both.delta_cal_const - both.delta_C).items()))
    clo = pd.read_csv(out / "curves" / "closure_check.csv") if (out / "curves" / "closure_check.csv").exists() else pd.DataFrame()
    fc = pd.read_csv(out / "family_closure.csv") if (out / "family_closure.csv").exists() else pd.DataFrame()
    add("V12a", "written files reproduce the in-memory magnitudes [mag]", float(clo.max_abs_diff.max()) if len(clo) else np.nan, "< 1e-4", len(clo) > 0 and clo.max_abs_diff.max() < 1e-4)
    add("V12b", "zero-shift family F*S vs archive reference magnitudes, max deviation [mag]", float(fc[fc.form == "rigid"].max_abs_dev.max()) if len(fc) else np.nan, "< 1e-3", len(fc) > 0 and fc[fc.form == "rigid"].max_abs_dev.max() < 1e-3)
    add("V13", "prior k = dlambda/lambda [%] and tau [nm]", f"{P['k']*100:+.3f} +- {P['k_err']*100:.3f}, tau {P['tau_nm']:.2f}", "k in [-0.5,-0.2], tau <= 1.0", (-0.5 <= P["k"] * 100 <= -0.2) and P["tau_nm"] <= 1.0, f"{P['n_prior_filters']} filters, n_eff {P['n_eff']:.2f}")
    if len(xw):
        add("XP", "XP wavelength scale vs CALSPEC (median lag, nm; vacuum/air = +0.14..+0.22)", round(float(xw.lag_nm.median()), 2), "info", True, "; ".join(f"{o} {l} {v:+.2f}" for o, l, v in zip(xw.object, xw.line, xw.lag_nm)))
    VT = pd.DataFrame(V); VT.to_csv(out / "validation.csv", index=False)
    print(VT.to_string(index=False))

    # ---- figures
    make_figures(out, fig_dir, table, P, S, parts, curves, ch, un, cr, cs, ts, rf, inj)
    write_report(out, table, P, info, VT, ch, cs, ts, inj, xw)
    log("validation done")


def make_figures(out, fig_dir, table, P, S, parts, curves, ch, un, cr, cs, ts, rf, inj):
    T = table.set_index("filter")
    # F1: S(lambda) recovery and decomposition
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
    ax[0].plot(C.FINE, parts["S_raw"], color="k", lw=1, label="S = nominal / filter-only (median over filters)")
    ax[0].plot(C.FINE, parts["model"], color="tab:red", lw=1, ls="--", label=f"g·QE·optics·atm(X0={P['X0']:.2f}), g={P['g']:.3f}")
    ax[0].plot(C.FINE, parts["qe"], color="tab:blue", lw=.8, alpha=.7, label="QE (IMX455)"); ax[0].plot(C.FINE, parts["optics"], color="tab:green", lw=.8, alpha=.7, label="optics")
    ax[0].plot(C.FINE, parts["atm_X0"], color="tab:orange", lw=.8, alpha=.7, label="atmosphere at X0 (SkyCalc)")
    ax[0].set_ylabel("throughput"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3); ax[0].set_title("F1  Filter-independent system factor S(λ) of the nominal curves and its decomposition")
    with np.errstate(all="ignore"): ax[1].plot(C.FINE, parts["S_raw"] / parts["model"] - 1, color="k", lw=.8)
    ax[1].axhline(0, color="grey"); ax[1].set_ylim(-0.3, 0.3); ax[1].set_ylabel("S / model − 1"); ax[1].set_xlabel("wavelength [nm]"); ax[1].grid(alpha=.3); ax[1].set_xlim(320, 1000)
    fig.tight_layout(); fig.savefig(fig_dir / "F1_S_decomposition.png", dpi=130); plt.close(fig)
    # F2: shifts vs pivot
    fig, ax = plt.subplots(figsize=(11, 6))
    x = T.pivot_nm
    ax.plot(x, P["k"] * x, color="grey", ls="--", label=f"prior k·λ, k = {P['k']*100:+.3f} %, τ = {P['tau_nm']:.2f} nm")
    ax.fill_between(np.sort(x), P["k"] * np.sort(x) - P["tau_nm"], P["k"] * np.sort(x) + P["tau_nm"], color="grey", alpha=.12)
    for col, mk, lab, colr in (("delta_std", "o", "field colour term, standard fields", "tab:blue"), ("delta_cal_const", "^", "CALSPEC pattern (free constant)", "tab:red")):   # tile-coadd channel not drawn (user request 2026-09-10); it still enters the joint estimate
        m = np.isfinite(T[col]); ax.plot(x[m], T[col][m], mk, mfc="none", color=colr, ms=7, label=lab)
    m = np.isfinite(T.delta_meas); ax.errorbar(x[m], T.delta_meas[m], yerr=T.sigma_meas[m], fmt="D", color="k", ms=5, capsize=2, label="joint measurement ± σ")
    ax.errorbar(x, T.delta_final, yerr=T.sigma_final, fmt="*", color="tab:orange", ms=11, capsize=2, zorder=5, label="final (shrinkage to the prior)")
    for f, xv, yv in zip(T.index, x, T.delta_final): ax.annotate(f, (xv, yv), xytext=(0, 8), textcoords="offset points", fontsize=6, ha="center", color="tab:orange")
    ax.axhline(0, color="k", lw=.6); ax.set_xlabel("pivot wavelength [nm]"); ax.set_ylabel("effective shift Δ [nm]"); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="lower left")
    ax.set_title("F2  Effective curve shift per filter: channels, joint measurement, prior and final value")
    fig.tight_layout(); fig.savefig(fig_dir / "F2_shift_vs_pivot.png", dpi=130); plt.close(fig)
    # F3: time series per (unit, filter) with >= 3 nights
    if len(ts):
        sel = ts[ts.n_nights >= 3].sort_values("span_days", ascending=False).head(24)
        n = len(sel); ncol = 4; nrow = int(np.ceil(n / ncol)) if n else 1
        fig, axes = plt.subplots(nrow, ncol, figsize=(16, 2.8 * nrow), sharey=True, squeeze=False)
        for k, (_, r) in enumerate(sel.iterrows()):
            ax = axes[k // ncol][k % ncol]; g = rf[(rf.telescop == r.telescop) & (rf["filter"] == r["filter"]) & ~rf.flag_mismatch]
            ax.errorbar(g.mjd, g.delta_frame, yerr=np.minimum(g.delta_frame_err, 5), fmt=".", ms=3, alpha=.5, color="tab:blue" if True else "k")
            ax.plot(g[g.is_tile].mjd, g[g.is_tile].delta_frame, "s", ms=3, color="tab:green", alpha=.7)
            if r["filter"] in T.index: ax.axhline(T.loc[r["filter"], "delta_final"], color="tab:orange", lw=1.2)
            ax.set_ylim(-8, 5); ax.grid(alpha=.3); ax.set_title(f"{r.telescop} {r['filter']}: {r.n_nights} nights, drift {r.slope_nm_per_yr:+.2f}±{r.slope_err:.2f} nm/yr", fontsize=8)
        for k in range(n, nrow * ncol): axes[k // ncol][k % ncol].axis("off")
        fig.suptitle("F3  Per-frame shift estimate from the field colour term (blue = standard fields, green = tiles) vs MJD; orange = final shift", fontsize=10)
        fig.tight_layout(); fig.savefig(fig_dir / "F3_timeseries.png", dpi=110); plt.close(fig)
        # F4: airmass dependence
        fig, ax = plt.subplots(figsize=(10, 5))
        t3 = ts[np.isfinite(ts.slope_nm_per_airmass)]
        for f, g in t3.groupby("filter"):
            ax.errorbar([T.loc[f, "pivot_nm"]] * len(g), g.slope_nm_per_airmass, yerr=g.slope_x_err, fmt="o", ms=4, capsize=2, alpha=.7)
        ax.axhline(0, color="k", lw=.6); ax.axhspan(-0.3, 0.3, color="grey", alpha=.15); ax.set_xlabel("pivot [nm]"); ax.set_ylabel("d(shift)/d(airmass) [nm per airmass]"); ax.grid(alpha=.3)
        ax.set_title("F4  Airmass dependence of the per-frame shift estimate per (unit, filter)"); fig.tight_layout(); fig.savefig(fig_dir / "F4_airmass.png", dpi=130); plt.close(fig)
    # F5: CALSPEC per star before/after
    if len(cr):
        filters = [f for f in C.MEDIUM if f in set(cr["filter"])]
        ncol = 5; nrow = int(np.ceil(len(filters) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow), squeeze=False)
        for k, f in enumerate(filters):
            ax = axes[k // ncol][k % ncol]; g = cr[cr["filter"] == f].sort_values("object")
            xk = np.arange(len(g)); ax.axhspan(-0.02, 0.02, color="grey", alpha=.15); ax.axhline(0, color="k", lw=.6)
            for i, (_, r) in enumerate(g.iterrows()):
                if r.is_da: ax.axvspan(i - .5, i + .5, color="tab:blue", alpha=.08)
            ax.errorbar(xk - .12, g.before, yerr=g.err, fmt="o", color="0.5", ms=4, capsize=2, label="nominal curves")
            ax.errorbar(xk + .12, g.after, yerr=g.err, fmt="s", color="tab:orange", ms=4, capsize=2, label="effective curves")
            ax.set_xticks(xk); ax.set_xticklabels(g.object, rotation=90, fontsize=6); ax.set_ylim(-0.1, 0.1); ax.grid(alpha=.3)
            ax.set_title(f"{f}  Δ = {T.loc[f, 'delta_final']:+.2f} nm", fontsize=9)
            if k == 0: ax.legend(fontsize=7)
        for k in range(len(filters), nrow * ncol): axes[k // ncol][k % ncol].axis("off")
        fig.suptitle("F5  CALSPEC residuals (observed − synthetic) with the nominal and the effective curves; shaded = DA white dwarfs", fontsize=11)
        fig.tight_layout(); fig.savefig(fig_dir / "F5_calspec_before_after.png", dpi=110); plt.close(fig)
    # F6: colour terms and ZP scatter before/after
    if len(ch):
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        for chn, mk in (("std", "o"), ("tile", "s")):
            g = ch[ch.channel == chn]
            if not len(g): continue
            xs = [T.loc[f, "pivot_nm"] for f in g["filter"]]
            ax[0].plot(xs, g.a_before, mk, mfc="none", color="0.4", ms=6, label=f"{chn}: nominal curves")
            ax[0].plot(xs, g.a_after, mk, color="tab:orange", ms=6, label=f"{chn}: effective curves")
            ax[1].plot(xs, g.zp_std_after / g.zp_std_before, mk, color="tab:orange" if chn == "std" else "tab:green", ms=6, label=chn)
        ax[0].axhline(0, color="k", lw=.6); ax[0].axhspan(-0.005, 0.005, color="grey", alpha=.15); ax[0].set_ylabel("field colour term dZP/d(BP−RP)"); ax[0].set_xlabel("pivot [nm]"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
        ax[1].axhline(1, color="k", lw=.6); ax[1].set_ylabel("ZP scatter after / before"); ax[1].set_xlabel("pivot [nm]"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
        fig.suptitle("F6  Closed loop with the written curves: field colour terms and per-frame ZP scatter"); fig.tight_layout(); fig.savefig(fig_dir / "F6_colourterms.png", dpi=130); plt.close(fig)
    # F7: injection
    if len(inj):
        fig, ax = plt.subplots(figsize=(7, 5))
        for x, g in inj.groupby("injected"): ax.plot([T.loc[f, "pivot_nm"] for f in g["filter"]], g.recovered - g.injected, "o", ms=5, label=f"injected {x:+.1f} nm")
        ax.axhline(0, color="k", lw=.6); ax.axhspan(-0.2, 0.2, color="grey", alpha=.15); ax.set_xlabel("pivot [nm]"); ax.set_ylabel("recovered − injected [nm]"); ax.grid(alpha=.3); ax.legend(fontsize=8)
        ax.set_title("F7  Injection test of the estimator"); fig.tight_layout(); fig.savefig(fig_dir / "F7_injection.png", dpi=130); plt.close(fig)
    # F8: curves nominal vs effective (37 panels)
    filters = [f for f in C.MEDIUM if f in curves]
    ncol = 6; nrow = int(np.ceil(len(filters) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.6 * nrow), squeeze=False)
    for k, f in enumerate(filters):
        ax = axes[k // ncol][k % ncol]; nom = C.load_nominal(f); eff = curves[f]; piv = T.loc[f, "pivot_nm"]
        u = pd.read_csv(out / "curves" / "unc" / f"{f}.csv")
        m = (C.FINE > piv - 30) & (C.FINE < piv + 30)
        ax.plot(C.FINE[m], nom[m], color="0.5", lw=1, label="nominal"); ax.plot(C.FINE[m], eff[m], color="tab:orange", lw=1.2, label="effective")
        ax.fill_between(u.lam[m], u.trans_lo[m], u.trans_hi[m], color="tab:orange", alpha=.25)
        ax.set_title(f"{f}: {T.loc[f, 'delta_final']:+.2f} ± {T.loc[f, 'sigma_final']:.2f} nm ({T.loc[f, 'cls']})", fontsize=7); ax.grid(alpha=.3); ax.tick_params(labelsize=6)
        if k == 0: ax.legend(fontsize=6)
    for k in range(len(filters), nrow * ncol): axes[k // ncol][k % ncol].axis("off")
    fig.suptitle("F8  Nominal (grey) and effective (orange, ±σ band) system curves", fontsize=11); fig.tight_layout(); fig.savefig(fig_dir / "F8_curves.png", dpi=110); plt.close(fig)
    # F9: unit comparison
    if len(un):
        fig, ax = plt.subplots(figsize=(11, 5))
        for f in T.index:
            u = json.loads(T.loc[f, "units"]) if isinstance(T.loc[f, "units"], str) else {}
            for k, v in u.items():
                if v is None: continue
                ax.plot(T.loc[f, "pivot_nm"], v, "o" if k.startswith("std") else "s", mfc="none", color="tab:blue" if k.startswith("std") else "tab:green", ms=5)
                ax.annotate(k.split(":")[1].replace("7DT", ""), (T.loc[f, "pivot_nm"], v), fontsize=5, xytext=(3, 0), textcoords="offset points")
        ax.errorbar(T.pivot_nm, T.delta_final, yerr=T.sigma_final, fmt="*", color="tab:orange", ms=9, capsize=2, zorder=5, label="final")
        ax.plot([], [], "o", mfc="none", color="tab:blue", label="per unit, standard fields"); ax.plot([], [], "s", mfc="none", color="tab:green", label="per unit, tile coadds")
        ax.axhline(0, color="k", lw=.6); ax.set_xlabel("pivot [nm]"); ax.set_ylabel("colour-term zero crossing per unit [nm]"); ax.grid(alpha=.3); ax.legend(fontsize=8)
        ax.set_title("F9  Unit-to-unit consistency of the shift (colour-term channel)"); fig.tight_layout(); fig.savefig(fig_dir / "F9_units.png", dpi=130); plt.close(fig)


def write_report(out, table, P, info, VT, ch, cs, ts, inj, xw):
    T = table.set_index("filter")
    lines = ["# 7DT 유효 투과곡선 검증 보고서 (2026-09-10)", "",
             f"모델: T_eff = Family[F_filter](Δ) × S(λ). S(λ) = 공칭/필터단독(필터 무관, 분해: g = {P['g']:.3f}, X₀ = {P['X0']:.2f}, 잔차 rms {P['S_resid_rms']:.3f}). "
             f"사전분포 k = {P['k']*100:+.3f} ± {P['k_err']*100:.3f} % (n_eff = {P['n_eff']:.2f}, ε = {P['eps']}), τ = {P['tau_nm']:.2f} nm ({P['n_prior_filters']}개 필터). 색항 계통 바닥 σ_a = {P['sigma_a_default']}.", "",
             "## 필터별 결과", "",
             "| 필터 | pivot | 형태 | 등급 | 측정 Δ ± σ | 표준성 색항 | 타일 색항 | CALSPEC | 최종 Δ ± σ | σ_a=0.005 | 이전(권고/선별) | 플래그 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda v, p=2: "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:+.{p}f}"
    for f, r in T.iterrows():
        lines.append(f"| {f} | {r.pivot_nm:.0f} | {r.form} | {r.cls} | {fmt(r.delta_meas)} ± {'' if not np.isfinite(r.sigma_meas) else f'{r.sigma_meas:.2f}'} | {fmt(r.get('delta_std', np.nan))} | {fmt(r.get('delta_tile', np.nan))} | {fmt(r.get('delta_cal_const', np.nan))} | {r.delta_final:+.2f} ± {r.sigma_final:.2f} | {fmt(r.get('delta_final_sa005', np.nan))} | {fmt(r.prev_recommended, 1)} / {fmt(r.prev_lineonly, 1)} | {r.flags if isinstance(r.flags, str) else ''} |")
    lines += ["", "## 검증표", "", "| 코드 | 검사 | 값 | 기준 | 통과 | 비고 |", "|---|---|---|---|---|---|"]
    for _, v in VT.iterrows():
        lines.append(f"| {v.code} | {v.test} | {v.value} | {v.threshold} | {'예' if v.passed else '아니오'} | {v.note} |")
    if len(cs):
        lines += ["", "## CALSPEC 전후", "", "| 필터 | 별 | DA−기타 전 | 후 | 별 간 MAD 전 | 후 | 전체 중앙값 전 | 후 | 연속 별 중앙값 전 | 후 |", "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in cs.iterrows():
            lines.append(f"| {r['filter']} | {r.n_stars} | {fmt(r.da_minus_other_before, 3)} | {fmt(r.da_minus_other_after, 3)} | {'' if not np.isfinite(r.mad_before) else f'{r.mad_before:.3f}'} | {'' if not np.isfinite(r.mad_after) else f'{r.mad_after:.3f}'} | {fmt(r.median_all_before, 3)} | {fmt(r.median_all_after, 3)} | {fmt(r.cont_median_before, 3)} | {fmt(r.cont_median_after, 3)} |")
    if len(ch):
        lines += ["", "## 폐루프: 필드 색항과 영점 산포", "", "| 필터 | 채널 | 프레임 | 색항 전 | 후 | 영점 산포 전 | 후 | 영점 변화 |", "|---|---|---|---|---|---|---|---|"]
        for _, r in ch.iterrows():
            lines.append(f"| {r['filter']} | {r.channel} | {r.n_frames} | {r.a_before:+.4f} | {r.a_after:+.4f} | {r.zp_std_before:.4f} | {r.zp_std_after:.4f} | {r.dzp:+.4f} |")
    if len(ts):
        lines += ["", "## 시간·대기질량 안정성 (단일 프레임 색항 → 이동량, 유닛·필터별)", "", "| 유닛 | 필터 | 프레임 | 밤 | 기간 [일] | 평균 Δ | MAD | 밤별 rms | 추세 [nm/yr] | 대기질량 기울기 [nm/X] | 안−밖 [nm] |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in ts.sort_values(["filter", "telescop"]).iterrows():
            lines.append(f"| {r.telescop} | {r['filter']} | {r.n_frames} | {r.n_nights} | {r.span_days:.0f} | {r.delta_mean:+.2f} | {r.delta_mad:.2f} | {'' if not np.isfinite(r.night_rms) else f'{r.night_rms:.2f}'} | {'' if not np.isfinite(r.slope_nm_per_yr) else f'{r.slope_nm_per_yr:+.2f} ± {r.slope_err:.2f}'} | {'' if not np.isfinite(r.slope_nm_per_airmass) else f'{r.slope_nm_per_airmass:+.2f} ± {r.slope_x_err:.2f}'} | {'' if not np.isfinite(r.inner_minus_outer_nm) else f'{r.inner_minus_outer_nm:+.2f}'} |")
    if len(xw):
        lines += ["", "## XP 파장 척도 점검 (CALSPEC DA 별의 XP vs CALSPEC⊗LSF 교차상관)", "", "| 별 | 선 | LSF FWHM | 지연 [nm] | 상관 |", "|---|---|---|---|---|"]
        for _, r in xw.iterrows(): lines.append(f"| {r['object']} | {r['line']} | {r['fwhm']:.0f} | {r['lag_nm']:+.2f} | {r['corr']:.3f} |")
    lines += ["", "## 그림", "", "F1 S(λ) 분해 · F2 이동량 vs pivot · F3 시간열 · F4 대기질량 · F5 CALSPEC 전후 · F6 색항·영점 산포 전후 · F7 주입 검사 · F8 곡선 · F9 유닛 비교 (`figures/`)", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
