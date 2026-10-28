#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260909_zpcolor_lineonly_shift.py

Curve shift per filter determined from the CALSPEC stars that are NOT continuum-dominated,
then applied in the closed loop and validated on the continuum stars that were left out.

"Continuum-dominated" is defined per star and filter: the synthetic magnitude of the star and of its
own continuum (upper envelope of the spectrum: running maximum then running median over ~40 nm) are
both computed with the curve shifted by +-3 nm; the excess sensitivity
    s_line = max |[m(shift) - m(0)]_star - [m(shift) - m(0)]_continuum| / 3 nm     [mag/nm]
measures how much of the star's response to a band-edge shift comes from features the continuum
does not describe (Balmer lines, molecular band heads, jumps).  Stars with s_line below
--sens-thresh are continuum-dominated: their residual moves with the field-star zero point and
carries no edge information, so they are excluded from the fit and used only for validation.

Per filter and shift (same machinery as 260908_zpcolor_shift_recal.py, re-used via importlib):
  ZP'(frame) from the shifted reference photometry of the same stars, CALSPEC residual
  Delta'(star) = m_obs + ZP' - m_syn(shifted curve).  Objective on the line-dominated set L:
    abs   : chi2 = sum_L (Delta'/err)^2              (no free offset: residual should vanish)
    const : chi2 = sum_L ((Delta' - c)/err)^2        (free common offset, needs >= 2 stars)
The best shift is the chi2 minimum (parabolic refinement); its error is the Delta chi2 = 1
interval after scaling chi2 so that the minimum reduced chi2 is 1.

Outputs (in --out-dir/lineonly/): lineonly_shift.csv (per filter), lineonly_stars.csv (per star),
recal_frames_line.csv.gz and recal_calspec_syn_line.csv (case 'line' for 260908_zpcolor_calspec_sed.py
--shift-case line), figures/C9_lineonly_scan.png, C10_lineonly_summary.png.

  MPLCONFIGDIR=/tmp/ezphot-matplotlib /home/hhchoi1022/anaconda3/envs/py311/bin/python \\
      ezphot/analysis/260909_zpcolor_lineonly_shift.py
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
from astropy.table import Table          # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
R = importlib.import_module("260908_zpcolor_shift_recal")   # syn_mag, load_curve, clipped_zp, weighted_linfit, mad, calspec_clean, constants

DEFAULT_OUT_DIR = R.DEFAULT_OUT_DIR
SCAN = np.arange(-6.0, 3.01, 0.25)
PIVOT = {"m400": 401.3, "m425": 425.5, "m450": 450.8, "m475": 475.2, "m500": 500.3, "m525": 524.8, "m550": 550.1, "m575": 574.9,
         "m600": 600.1, "m625": 624.8, "m650": 650.1, "m675": 674.5, "m700": 700.0, "m725": 724.7, "m750": 749.0, "m775": 775.3,
         "m800": 799.4, "m825": 824.4, "m850": 849.4, "m875": 873.8, "u": 371.4, "g": 479.4, "r": 616.2, "i": 758.7, "z": 876.6}


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def continuum_envelope(wl, fl, window_nm=40.0, step_nm=0.5):
    """Upper envelope of a spectrum (running max, then running median) on a regular grid: the 'continuum'
    that absorption features (lines, band heads) depart from."""
    grid = np.arange(wl.min(), wl.max(), step_nm)
    f = np.interp(grid, wl, fl)
    k = max(int(window_nm / step_nm) | 1, 3)
    env = pd.Series(f).rolling(k, center=True, min_periods=1).max().rolling(k, center=True, min_periods=1).median().to_numpy()
    return grid, env


def refine(x, chi):
    i = int(np.argmin(chi))
    if 0 < i < len(x) - 1:
        a, b, c = np.polyfit(x[i - 1:i + 2], chi[i - 1:i + 2], 2)
        if a > 0:
            return float(-b / (2 * a)), float(np.sqrt(1 / a)), float(chi[i]), i
    return float(x[i]), np.nan, float(chi[i]), i


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--filters", default="", help="comma-separated filters (default: the 25 nm grid)")
    ap.add_argument("--exclude-filters", default="")
    ap.add_argument("--sens-thresh", type=float, default=0.003, help="excess sensitivity s_line above this [mag/nm] = line-dominated")
    ap.add_argument("--sens-baseline-nm", type=float, default=3.0, help="shift used to evaluate s_line (captures non-linear wings)")
    ap.add_argument("--min-frames", type=int, default=R.MIN_FRAMES)
    ap.add_argument("--min-stars", type=int, default=2, help="minimum line-dominated stars for a fit")
    args = ap.parse_args()
    out = Path(args.out_dir); d = out / "lineonly"; d.mkdir(exist_ok=True); fig_dir = out / "figures"; fig_dir.mkdir(exist_ok=True)
    grid = [f.strip() for f in args.filters.split(",") if f.strip()] or list(R.GRID)
    excl = {f.strip() for f in args.exclude_filters.split(",") if f.strip()}
    grid = [f for f in grid if f not in excl]

    npz = np.load(d.parent / "shift_recal" / "refstar_spectra.npz", allow_pickle=False)
    ids = npz["ids"].astype(str); flux = npz["flux"]; wave = npz["wave_nm"]; idx_of = {s: i for i, s in enumerate(ids)}
    curves = {f: R.load_curve(f) for f in grid}
    frames = pd.read_csv(out / "frames.csv", low_memory=False)
    bad = set(frames[frames["ZPERR_APER_3"] > R.ZPERR_ANOMALY]["file"])
    st = pd.read_csv(out / "refit_stars.csv.gz", dtype={"ref_id": str})
    st = st[(st.catalog == R.CATALOG) & st.objname.isin(R.STANDARDS) & ~st.file.isin(bad) & st["filter"].isin(grid) & st.ref_id.isin(idx_of)].copy()
    st["row"] = st.ref_id.map(idx_of); st["sig"] = np.sqrt(st.magerr ** 2 + R.MAGERR_FLOOR ** 2)
    cal = R.calspec_clean(out, bad)
    syn = pd.read_csv(out / "calspec_synphot.csv")
    spec_file = dict(zip(syn["object"], syn["spec"]))
    cal_spec = {}
    for o in R.STANDARDS:
        if o not in spec_file: continue
        t = Table.read(R.CALSPEC_DIR / spec_file[o])
        wl = np.asarray(t["WAVELENGTH"], float) / 10.0; fl = np.asarray(t["FLUX"], float) * 1e-2
        m = (wl > 300) & (wl < 1000) & np.isfinite(fl); cal_spec[o] = (wl[m], fl[m])
    cal_cont = {o: continuum_envelope(*cal_spec[o]) for o in cal_spec}
    sptype = cal.groupby("object")["spectral_type"].first().to_dict()
    prev_c4 = pd.read_csv(out / "calspec_shift_fit.csv").set_index("filter") if (out / "calspec_shift_fit.csv").exists() else None
    prev_rec = pd.read_csv(out / "shift_recal" / "recal_recommended.csv").set_index("filter") if (out / "shift_recal" / "recal_recommended.csv").exists() else None

    rows, star_rows, frame_rows, syn_rows, detail = [], [], [], [], {}
    for f in grid:
        rs = st[st["filter"] == f]
        cf = cal[cal["filter"] == f]
        if not len(rs) or not len(cf): log(f"{f}: no data"); continue
        fw, ft = curves[f]
        fl_ref = flux[rs.row.to_numpy()]
        m0_ref = R.syn_mag(wave, fl_ref, fw, ft, 0.0)
        s_field = (R.syn_mag(wave, fl_ref, fw, ft, 0.5) - R.syn_mag(wave, fl_ref, fw, ft, -0.5))      # mag/nm per reference star
        s_field_med = float(np.nanmedian(s_field)); s_field_mad = R.mad(s_field)
        zp0 = rs.groupby("file").zp.median()
        objs = [o for o in R.STANDARDS if o in set(cf.object) and o in cal_spec and (cf.object == o).sum() >= args.min_frames]
        if not objs: log(f"{f}: no standards"); continue
        s_star = {o: float(R.syn_mag(*cal_spec[o], fw, ft, 0.5) - R.syn_mag(*cal_spec[o], fw, ft, -0.5)) for o in objs}
        B = args.sens_baseline_nm
        def excess(o):
            vals = []
            for sh in (-B, +B):
                d_star = float(R.syn_mag(*cal_spec[o], fw, ft, sh) - R.syn_mag(*cal_spec[o], fw, ft, 0.0))
                d_cont = float(R.syn_mag(*cal_cont[o], fw, ft, sh) - R.syn_mag(*cal_cont[o], fw, ft, 0.0))
                vals.append(abs(d_star - d_cont) / B)
            return max(vals)
        s_line = {o: excess(o) for o in objs}
        line = [o for o in objs if s_line[o] > args.sens_thresh]
        cont = [o for o in objs if o not in line]
        cal0 = {o: float(R.syn_mag(*cal_spec[o], fw, ft, 0.0)) for o in objs}
        # scan
        table = {}   # shift -> dict(delta per star, err per star, a, zp_std, dzp, med per frame, dm_cal)
        for sh in SCAN:
            dm_ref = R.syn_mag(wave, fl_ref, fw, ft, sh) - m0_ref
            zp_i = rs.zp.to_numpy() + dm_ref
            med, sd, n, keep = R.clipped_zp(rs.file.to_numpy(), zp_i)
            resid = zp_i - rs.file.map(med).to_numpy()
            a, a_err, _, _ = R.weighted_linfit(rs.bprp.to_numpy()[keep], resid[keep], rs.sig.to_numpy()[keep])
            dm_cal = {o: float(R.syn_mag(*cal_spec[o], fw, ft, sh)) - cal0[o] for o in objs}
            c = cf.copy()
            c["delta_new"] = c["delta_APER_3"] + (c.file.map(med) - c.file.map(zp0)).to_numpy() - c.object.map(dm_cal).to_numpy()
            c = c.dropna(subset=["delta_new"])
            per = c.groupby("object")["delta_new"].agg(["median", "size", R.mad])
            per.columns = ["delta", "n", "scatter"]
            magerr = c.groupby("object")["magerr_APER_3"].median()
            err = np.sqrt((per["scatter"] / np.sqrt(per["n"])) ** 2 + (magerr.reindex(per.index) / np.sqrt(per["n"])) ** 2).clip(lower=0.005)
            table[sh] = dict(delta=per["delta"].to_dict(), err=err.to_dict(), a=a, a_err=a_err, zp_std=float(sd.median()),
                             dzp=float((med - zp0).median()), med=med, dm_cal=dm_cal,
                             all_mad=R.mad(per["delta"]) if len(per) >= 3 else np.nan,
                             cont_med=float(np.median([per["delta"][o] for o in cont if o in per.index])) if cont and any(o in per.index for o in cont) else np.nan,
                             line_med=float(np.median([per["delta"][o] for o in line if o in per.index])) if line and any(o in per.index for o in line) else np.nan)
        rec = dict(filter=f, pivot=PIVOT.get(f, np.nan), n_stars=len(objs), n_line=len(line), n_cont=len(cont),
                   line_stars=";".join(line), cont_stars=";".join(cont), s_field_median=s_field_med, s_field_mad=s_field_mad,
                   sens_thresh=args.sens_thresh)
        for o in objs:
            star_rows.append(dict(filter=f, object=o, sptype=sptype.get(o, ""), s_star=s_star[o], s_minus_field=s_star[o] - s_field_med,
                                  s_line=s_line[o], is_line=o in line, delta0=table[0.0]["delta"].get(o, np.nan), err0=table[0.0]["err"].get(o, np.nan)))
        best = {}
        if len(line) >= args.min_stars or (len(line) >= 1 and args.min_stars <= 1):
            L = [o for o in line if o in table[0.0]["delta"]]
            chi_abs, chi_const = [], []
            for sh in SCAN:
                y = np.array([table[sh]["delta"][o] for o in L]); e = np.array([table[sh]["err"][o] for o in L]); w = 1 / e ** 2
                chi_abs.append(float(np.sum(w * y ** 2)))
                cc = np.sum(w * y) / np.sum(w); chi_const.append(float(np.sum(w * (y - cc) ** 2)))
            for name, chi, dof in (("abs", np.array(chi_abs), max(len(L), 1)), ("const", np.array(chi_const), max(len(L) - 1, 1))):
                if name == "const" and len(L) < 2: continue
                scale = max(chi.min() / dof, 1.0)
                b, e, cmin, i = refine(SCAN, chi / scale)
                best[name] = (b, e, cmin * scale, i)
                rec.update({f"shift_{name}": b, f"shift_{name}_err": e, f"chi2min_{name}": cmin * scale, f"chi2_0_{name}": float(chi[np.argmin(np.abs(SCAN))])})
        rec["c4_shift"] = float(prev_c4.loc[f, "best_shift_nm"]) if prev_c4 is not None and f in prev_c4.index and np.isfinite(prev_c4.loc[f, "shift_err_nm"]) else np.nan
        rec["rec_shift"] = float(prev_rec.loc[f, "recommended"]) if prev_rec is not None and f in prev_rec.index else np.nan
        # diagnostics at 0 and at the line-only (abs) shift
        for tag, sh in (("at0", 0.0), ("atline", best["abs"][0] if "abs" in best else np.nan)):
            if not np.isfinite(sh): continue
            key = SCAN[np.argmin(np.abs(SCAN - sh))]
            tb = table[key]
            rec.update({f"{tag}_shift_used": float(key), f"{tag}_colour_term": tb["a"], f"{tag}_zp_std": tb["zp_std"], f"{tag}_dzp": tb["dzp"],
                        f"{tag}_line_median": tb["line_med"], f"{tag}_cont_median": tb["cont_med"], f"{tag}_all_mad": tb["all_mad"]})
            if tag == "atline":
                for fpath, v in tb["med"].items():
                    frame_rows.append(dict(filter=f, shift_nm=float(key), case="line", file=fpath, zp=float(v)))
                for o, dm in tb["dm_cal"].items():
                    s0 = syn[(syn.object == o) & (syn["filter"] == f)]["mag_syn"]
                    if len(s0): syn_rows.append(dict(filter=f, shift_nm=float(key), case="line", object=o, dm_syn=dm, mag_syn_shifted=float(s0.iloc[0]) + dm))
        rows.append(rec)
        detail[f] = dict(table=table, line=line, cont=cont, best=best)
        log(f"{f}: line stars {line} | cont {len(cont)} | s_field {s_field_med:+.4f}/nm | shift abs {rec.get('shift_abs', np.nan):+.2f} ± {rec.get('shift_abs_err', np.nan):.2f} nm, "
            f"const {rec.get('shift_const', np.nan):+.2f} | C4 {rec['c4_shift']:+.1f}, rec {rec['rec_shift']:+.1f} | cont median 0 -> line: {rec.get('at0_cont_median', np.nan):+.3f} -> {rec.get('atline_cont_median', np.nan):+.3f}")
    S = pd.DataFrame(rows)
    for c in ("shift_abs", "shift_abs_err", "chi2min_abs", "chi2_0_abs", "shift_const", "shift_const_err", "chi2min_const", "chi2_0_const"):
        if c not in S: S[c] = np.nan  # no filter had enough line-dominated stars (e.g. broad bands)
    S.to_csv(d / "lineonly_shift.csv", index=False)
    pd.DataFrame(star_rows).to_csv(d / "lineonly_stars.csv", index=False)
    pd.DataFrame(frame_rows).to_csv(d / "recal_frames_line.csv.gz", index=False, compression="gzip")
    pd.DataFrame(syn_rows).to_csv(d / "recal_calspec_syn_line.csv", index=False)

    if not S["shift_abs"].notna().any():
        log("no filter with a determined line-only shift: figures C9/C10 skipped"); return
    # ---- figures
    n = len(detail); ncol = 5; nrow = max(int(np.ceil(n / ncol)), 1)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.0 * nrow), squeeze=False, sharex=True)
    for ax, f in zip(axes.ravel(), detail):
        dt = detail[f]; tb = dt["table"]
        for o in dt["line"] + dt["cont"]:
            y = [tb[sh]["delta"].get(o, np.nan) for sh in SCAN]
            ax.plot(SCAN, y, "-", color="tab:blue" if o in dt["line"] else "0.6", lw=1.3 if o in dt["line"] else 0.8, alpha=.9 if o in dt["line"] else .6)
        ax.axhspan(-0.02, 0.02, color="grey", alpha=.12); ax.axhline(0, color="k", lw=.5); ax.axvline(0, color="k", lw=.5)
        if "abs" in dt["best"]: ax.axvline(dt["best"]["abs"][0], color="tab:red", lw=1.2, label=f"line-only {dt['best']['abs'][0]:+.2f} nm")
        r = S[S["filter"] == f].iloc[0]
        if np.isfinite(r["rec_shift"]): ax.axvline(r["rec_shift"], color="tab:orange", lw=1, ls="--", label=f"consensus {r['rec_shift']:+.1f}")
        ax.set_ylim(-0.1, 0.1); ax.grid(alpha=.3); ax.set_title(f"{f}  ({len(dt['line'])} line / {len(dt['cont'])} cont)", fontsize=9); ax.legend(fontsize=6, loc="lower left")
    for ax in axes.ravel()[n:]: ax.axis("off")
    for ax in axes[-1]: ax.set_xlabel("curve shift [nm]", fontsize=8)
    fig.supylabel("CALSPEC residual after re-calibration [mag]")
    fig.suptitle("Residual of each standard vs curve shift (blue = line-dominated, used for the fit; grey = continuum, validation only)", fontsize=10)
    fig.tight_layout(); fig.savefig(fig_dir / "C9_lineonly_scan.png", dpi=120); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    for col, lab, mk, c in (("shift_abs", "line stars only, no offset", "o", "tab:red"), ("shift_const", "line stars only, free offset", "s", "tab:purple"),
                            ("c4_shift", "all stars, free offset (C4)", "D", "tab:blue"), ("rec_shift", "scan consensus (rec)", "^", "tab:orange")):
        if col not in S: continue
        e = S[f"{col}_err"] if f"{col}_err" in S else None
        ax.errorbar(S["pivot"], S[col], yerr=e, fmt=mk, color=c, ms=5, capsize=2, label=lab, alpha=.9)
    ax.axhline(0, color="k", lw=.5); ax.set_xlabel("pivot [nm]"); ax.set_ylabel("best shift [nm]"); ax.grid(alpha=.3); ax.legend(fontsize=8); ax.set_ylim(-6, 3.5)
    ax.set_title("Curve shift per filter by method", fontsize=10)
    ax = axes[1]
    ax.plot(S["pivot"], S.get("at0_cont_median"), "o", color="0.5", label="continuum stars, no shift")
    ax.plot(S["pivot"], S.get("atline_cont_median"), "o", color="tab:red", label="continuum stars, line-only shift applied")
    ax.plot(S["pivot"], S.get("at0_line_median"), "s", color="0.5", mfc="none", label="line stars, no shift")
    ax.plot(S["pivot"], S.get("atline_line_median"), "s", color="tab:red", mfc="none", label="line stars, line-only shift applied")
    ax.axhspan(-0.02, 0.02, color="grey", alpha=.12); ax.axhline(0, color="k", lw=.5); ax.set_ylim(-0.08, 0.08); ax.grid(alpha=.3); ax.legend(fontsize=7)
    ax.set_xlabel("pivot [nm]"); ax.set_ylabel("median residual [mag]"); ax.set_title("Validation: residuals of the excluded continuum stars", fontsize=10)
    fig.tight_layout(); fig.savefig(fig_dir / "C10_lineonly_summary.png", dpi=130); plt.close(fig)
    log(f"lineonly/ and figures C9, C10 written ({len(S)} filters)")


if __name__ == "__main__":
    main()
