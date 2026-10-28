#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260910_xp_mdwarfs_photometry.py

Do the effective (shifted) transmission curves reproduce the observed 7DT magnitudes of M-type stars better than the
nominal curves?  The M stars of 260910_xp_mdwarfs.py (BP-RP >= 1.8, redder than the pipeline's reference-star range)
are matched in the full detection lists (.cat) of the medium-band single frames; their observed magnitudes
(MAG_APER_3 + header ZP) are compared with synthetic magnitudes from their XP spectra through
  (a) the nominal curves with the pipeline zero point, and
  (b) the effective curves (ezphot/analysis/effective_curves) with the zero point re-derived from the frame's own
      reference stars (closed loop, ZP' = ZP + median[m_ref(eff) - m_ref(nom)]).

Stages (--stage):
  match   : cross-match the M stars with the .cat detections of every usable medium-band frame -> mstar_phot.csv.gz
  compute : residuals per star and filter, binned by colour, dwarfs/giants; figures F5, F6 -> mstar_resid_*.csv, figures/
  all     : both

  MPLCONFIGDIR=/tmp/ezphot-matplotlib /home/hhchoi1022/anaconda3/envs/py311/bin/python \\
      ezphot/analysis/260910_xp_mdwarfs_photometry.py --stage all --n-proc 32
"""
from __future__ import annotations

import argparse
import importlib
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402
import pandas as pd                 # noqa: E402
from astropy.coordinates import SkyCoord, match_coordinates_sky   # noqa: E402
import astropy.units as u           # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
C = importlib.import_module("260910_effcurve_common")
M = importlib.import_module("260910_xp_mdwarfs")
R = C.R
warnings.filterwarnings("ignore")
log = C.log

OUT = M.DEFAULT_OUT
GAIA_EPOCH = 2016.0
MATCH_ARCSEC, FOV_RADIUS_DEG = 1.5, 0.8
CAT_COLS = ["X_WORLD", "Y_WORLD", "MAG_APER_3", "MAGERR_APER_3", "FLAGS", "FLUX_MAX", "CLASS_STAR"]
_G = {}


def match_frame(args):
    file, filt, tel, mjd, airmass, ra0, dec0, zp, zperr = args
    st = _G["stars"]
    d = np.degrees(np.arccos(np.clip(np.sin(np.radians(dec0)) * np.sin(np.radians(st["dec"])) + np.cos(np.radians(dec0)) * np.cos(np.radians(st["dec"])) * np.cos(np.radians(st["ra"] - ra0)), -1, 1)))
    cand = np.where(d < FOV_RADIUS_DEG)[0]
    if not len(cand): return []
    p = Path(file + ".cat")
    if not p.exists(): return []
    try:
        cat = pd.read_csv(p, sep=r"\s+", usecols=CAT_COLS)
    except Exception:
        return []
    if not len(cat): return []
    dt = (mjd - 51544.5) / 365.25 + 2000.0 - GAIA_EPOCH
    ra = st["ra"][cand] + np.nan_to_num(st["pmra"][cand]) / 3.6e6 * dt / np.cos(np.radians(st["dec"][cand]))
    dec = st["dec"][cand] + np.nan_to_num(st["pmdec"][cand]) / 3.6e6 * dt
    obj = SkyCoord(ra * u.deg, dec * u.deg); sky = SkyCoord(cat["X_WORLD"].to_numpy() * u.deg, cat["Y_WORLD"].to_numpy() * u.deg)
    idx, sep, _ = match_coordinates_sky(obj, sky)
    ok = sep.arcsec <= MATCH_ARCSEC
    rows = []
    for k, i, s in zip(cand[ok], idx[ok], sep.arcsec[ok]):
        c = cat.iloc[i]
        rows.append(dict(file=file, filter=filt, telescop=tel, mjd=mjd, airmass=airmass, sid=st["ids"][k], mag_inst=float(c["MAG_APER_3"]), magerr=float(c["MAGERR_APER_3"]),
                         flags=int(c["FLAGS"]), flux_max=float(c["FLUX_MAX"]), class_star=float(c["CLASS_STAR"]), sep_arcsec=float(s), r_deg=float(d[k]), zp=zp, zperr=zperr))
    return rows


def stage_match(out: Path, n_proc: int, m450_max: float):
    z = np.load(out / "mstars.npz", allow_pickle=True)
    wave, flux = z["wave_nm"], z["flux"]
    m450 = R.syn_mag(wave, flux, *R.load_curve("m450"), 0.0)
    sel = np.isfinite(m450) & (m450 <= m450_max) & np.all(np.isfinite(flux), axis=1)
    _G["stars"] = {k: z[k][sel] for k in ("ids", "ra", "dec", "pmra", "pmdec")}
    _G["stars"]["ids"] = _G["stars"]["ids"].astype(str)
    log(f"match: {sel.sum()} M stars with m450 <= {m450_max}")
    F = C.load_frames("single")
    F = F[F["FILTER"].isin(C.MEDIUM) & F["ZP_APER_3"].notna() & ~F["mislabeled"] & ~F["bad_zperr"]]
    jobs = [(r.file, r.FILTER, r.TELESCOP, float(r.MJD), float(r.AIRMASS), float(r.CRVAL1), float(r.CRVAL2), float(r.ZP_APER_3), float(r.ZPERR_APER_3)) for r in F.itertuples()]
    log(f"match: {len(jobs)} medium-band single frames")
    rows = []
    with Pool(n_proc) as pool:
        for k, res in enumerate(pool.imap_unordered(match_frame, jobs, chunksize=8)):
            rows += res
            if (k + 1) % 2000 == 0: log(f"  {k + 1} frames, {len(rows)} matches")
    D = pd.DataFrame(rows); D.to_csv(out / "mstar_phot.csv.gz", index=False, compression="gzip")
    log(f"match: {len(D)} detections of {D.sid.nunique() if len(D) else 0} stars in {D.file.nunique() if len(D) else 0} frames")


def frame_zp_change(out: Path, filters):
    """ZP' - ZP per single frame from the frame's own reference stars re-synthesised with the effective curves."""
    st = pd.read_csv(C.PREV_OUT / "refit_stars.csv.gz", dtype={"ref_id": str}, usecols=["file", "catalog", "filter", "ref_id"])
    st = st[(st.catalog == R.CATALOG) & st["filter"].isin(filters)]
    spectra, ids = [], []
    for p in (C.PREV_OUT / "shift_recal" / "refstar_spectra.npz", C.DEFAULT_OUT / "tilestar_spectra.npz"):
        if p.exists():
            zz = np.load(p, allow_pickle=False); spectra.append(zz["flux"]); ids.append(zz["ids"].astype(str)); wave = zz["wave_nm"]
    flux = np.vstack(spectra); ids = np.concatenate(ids); _, first = np.unique(ids, return_index=True); flux, ids = flux[first], ids[first]
    idx = {s: i for i, s in enumerate(ids)}
    st = st[st.ref_id.isin(idx)].copy(); st["row"] = st.ref_id.map(idx)
    out_rows = []
    for f in filters:
        nom = C.load_nominal(f); eff = load_effective(f)
        if eff is None: continue
        g = st[st["filter"] == f]
        if not len(g): continue
        u_rows, inv = np.unique(g.row.to_numpy(), return_inverse=True)
        dm = (R.syn_mag(wave, flux[u_rows], C.FINE, eff, 0.0) - R.syn_mag(wave, flux[u_rows], C.FINE, nom, 0.0))[inv]
        t = pd.DataFrame({"file": g.file.to_numpy(), "dm": dm}).groupby("file")["dm"].agg(["median", "size"])
        t = t[t["size"] >= 5]
        out_rows.append(pd.DataFrame({"file": t.index, "filter": f, "dzp": t["median"].to_numpy(), "n_ref": t["size"].to_numpy()}))
    return pd.concat(out_rows) if out_rows else pd.DataFrame(columns=["file", "filter", "dzp", "n_ref"])


def load_effective(f):
    p = C.REPO_CURVES / f"{f}.csv"
    if not p.exists(): return None
    d = pd.read_csv(p); return np.interp(C.FINE, d["lam"].to_numpy(float), d["trans"].to_numpy(float), left=0.0, right=0.0)


def stage_compute(out: Path, fig_dir: Path, m450_max: float):
    D = pd.read_csv(out / "mstar_phot.csv.gz", dtype={"sid": str})
    D = D[(D["flags"] <= 1) & (D["flux_max"] < 45000) & (D["magerr"] < 0.1) & (D["sep_arcsec"] <= MATCH_ARCSEC)]
    z = np.load(out / "mstars.npz", allow_pickle=True)
    ids = z["ids"].astype(str); idx = {s: i for i, s in enumerate(ids)}; wave, flux = z["wave_nm"], z["flux"]
    info = pd.DataFrame({"sid": ids, "bp_rp": z["bp_rp"], "g": z["g"], "parallax": z["parallax"]}).set_index("sid")
    info["MG"] = info.g + 5 * np.log10(np.clip(info.parallax, 1e-3, None) / 100.0)
    info["lum"] = np.where(np.isfinite(info.parallax) & (info.parallax > 1) & (info.MG > 6), "dwarf", np.where(np.isfinite(info.parallax) & (info.MG < 4), "giant", "unknown"))
    filters = [f for f in C.MEDIUM if f in set(D["filter"])]
    shifts = pd.read_csv(C.DEFAULT_OUT / "shift_table.csv").set_index("filter")["delta_final"].to_dict() if (C.DEFAULT_OUT / "shift_table.csv").exists() else {}
    dzp = frame_zp_change(out, filters); D = D.merge(dzp, on=["file", "filter"], how="left")
    log(f"compute: {len(D)} detections, {D.sid.nunique()} stars, {len(filters)} filters; frames with a closed-loop ZP: {D.dzp.notna().mean():.2f}")
    star_rows = []
    for f in filters:
        g = D[D["filter"] == f]
        if not len(g): continue
        nom = C.load_nominal(f); eff = load_effective(f)
        u_ids = g.sid.unique(); rows = np.array([idx[s] for s in u_ids])
        m_nom = pd.Series(R.syn_mag(wave, flux[rows], C.FINE, nom, 0.0), index=u_ids)
        m_eff = pd.Series(R.syn_mag(wave, flux[rows], C.FINE, eff, 0.0), index=u_ids) if eff is not None else m_nom
        obs = g.mag_inst + g.zp
        r_nom = obs - g.sid.map(m_nom).to_numpy(); r_eff = obs + g.dzp - g.sid.map(m_eff).to_numpy()
        t = pd.DataFrame({"sid": g.sid, "r_nom": r_nom, "r_eff": r_eff, "magerr": g.magerr}).groupby("sid").agg(n=("r_nom", "size"), r_nom=("r_nom", "median"), r_eff=("r_eff", "median"), sc=("r_nom", R.mad), magerr=("magerr", "median"))
        t = t[t.n >= 2]; t["filter"] = f; t["m_syn_nom"] = m_nom.reindex(t.index).to_numpy(); t["dm_curve"] = (m_eff - m_nom).reindex(t.index).to_numpy()
        star_rows.append(t.reset_index())
    S = pd.concat(star_rows).merge(info, left_on="sid", right_index=True, how="left")
    S.to_csv(out / "mstar_resid_stars.csv.gz", index=False, compression="gzip")
    # ---- summaries: per filter, all stars and dwarfs, slopes vs colour, colour bins
    def robust_slope(x, y):
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 20: return np.nan, np.nan
        a, ae, b, n = R.weighted_linfit(x[m], y[m], np.full(m.sum(), 0.02)); return a, ae
    summ, binned = [], []
    for f in filters:
        t = S[(S["filter"] == f) & np.isfinite(S.r_eff)]
        if len(t) < 20: continue
        rec = dict(filter=f, pivot=C.pivot_nm(C.FINE, C.load_nominal(f)), n_stars=len(t), n_dwarf=int((t.lum == "dwarf").sum()), n_giant=int((t.lum == "giant").sum()),
                   curve_change=float((t.r_eff - t.r_nom).median()), shift_nm=float(shifts.get(f, np.nan)))
        for tag, sub in (("all", t), ("dwarf", t[t.lum == "dwarf"]), ("giant", t[t.lum == "giant"])):
            if len(sub) < 20:
                rec.update({f"{tag}_median_nom": np.nan, f"{tag}_median_eff": np.nan, f"{tag}_mad_nom": np.nan, f"{tag}_mad_eff": np.nan, f"{tag}_slope_nom": np.nan, f"{tag}_slope_eff": np.nan}); continue
            sn, _ = robust_slope(sub.bp_rp.to_numpy(), sub.r_nom.to_numpy()); se, _ = robust_slope(sub.bp_rp.to_numpy(), sub.r_eff.to_numpy())
            rec.update({f"{tag}_median_nom": float(sub.r_nom.median()), f"{tag}_median_eff": float(sub.r_eff.median()), f"{tag}_mad_nom": R.mad(sub.r_nom), f"{tag}_mad_eff": R.mad(sub.r_eff),
                        f"{tag}_slope_nom": sn, f"{tag}_slope_eff": se})
        summ.append(rec)
        for lo, hi, lab in M.COLOUR_BINS:
            sub = t[(t.bp_rp >= lo) & (t.bp_rp < hi)]
            if len(sub) < 5: continue
            binned.append(dict(filter=f, bin=lab, lo=lo, hi=hi, n=len(sub), r_nom=float(sub.r_nom.median()), r_eff=float(sub.r_eff.median()), err=R.mad(sub.r_nom) / np.sqrt(len(sub)),
                               dm_curve=float(sub.dm_curve.median()), n_dwarf=int((sub.lum == "dwarf").sum()), r_nom_dwarf=float(sub[sub.lum == "dwarf"].r_nom.median()) if (sub.lum == "dwarf").sum() >= 5 else np.nan,
                               r_eff_dwarf=float(sub[sub.lum == "dwarf"].r_eff.median()) if (sub.lum == "dwarf").sum() >= 5 else np.nan))
    SM = pd.DataFrame(summ); B = pd.DataFrame(binned); SM.to_csv(out / "mstar_resid_summary.csv", index=False); B.to_csv(out / "mstar_resid_bins.csv", index=False)
    pd.set_option("display.width", 250)
    print(SM[["filter", "shift_nm", "n_stars", "n_dwarf", "curve_change", "all_median_nom", "all_median_eff", "all_mad_nom", "all_mad_eff", "all_slope_nom", "all_slope_eff", "dwarf_median_nom", "dwarf_median_eff", "giant_median_nom", "giant_median_eff"]].round(4).to_string(index=False))
    # ---- adjacent-band diagnostic: per-star residual correlation between neighbouring grid bands (XP band-head smearing)
    grid = [f for f in filters if f in C.MEDIUM and len(f) == 4]
    piv = {f: C.pivot_nm(C.FINE, C.load_nominal(f)) for f in grid}; grid = sorted(grid, key=piv.get)
    W = S.pivot_table(index="sid", columns="filter", values="r_nom"); col = S.groupby("sid").bp_rp.first()
    adj = []
    for a, b in zip(grid[:-1], grid[1:]):
        if abs(piv[b] - piv[a]) > 30: continue
        t = W[[a, b]].dropna(); c = col.reindex(t.index).to_numpy()
        if len(t) < 30: continue
        ra_, rb_ = t[a].to_numpy(), t[b].to_numpy()
        sd, _ = robust_slope(c, ra_ - rb_); ss, _ = robust_slope(c, ra_ + rb_)
        adj.append(dict(pair=f"{a}-{b}", boundary_nm=(piv[a] + piv[b]) / 2, n=len(t), corr=float(np.corrcoef(ra_, rb_)[0, 1]), slope_diff=sd, slope_sum=ss, mad_diff=R.mad(ra_ - rb_), mad_sum=R.mad(ra_ + rb_)))
    A = pd.DataFrame(adj); A.to_csv(out / "mstar_adjacent_bands.csv", index=False)
    print(A.round(3).to_string(index=False))
    # ---- F5: binned residual vs colour per filter
    filters = [f for f in filters if f in set(SM["filter"])]
    ncol = 5; nrow = int(np.ceil(len(filters) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.1 * nrow), sharex=True, sharey=True, squeeze=False)
    for k, f in enumerate(filters):
        ax = axes[k // ncol][k % ncol]; t = S[(S["filter"] == f) & np.isfinite(S.r_eff)]
        ax.axvspan(0.6, 2.0, color="tab:blue", alpha=.06); ax.axhline(0, color="k", lw=.6); ax.axhspan(-0.02, 0.02, color="grey", alpha=.15)
        ax.plot(t.bp_rp, t.r_nom, ".", ms=2, color="0.75", alpha=.3)
        b = B[B["filter"] == f]
        x = (b.lo + b.hi) / 2
        ax.errorbar(x - 0.03, b.r_nom, yerr=b.err, fmt="o", color="0.4", ms=5, capsize=2, label="nominal curves")
        ax.errorbar(x + 0.03, b.r_eff, yerr=b.err, fmt="s", color="tab:orange", ms=5, capsize=2, label="effective curves")
        if b.r_nom_dwarf.notna().any(): ax.plot(x + 0.03, b.r_eff_dwarf, "^", color="tab:red", ms=5, label="effective, dwarfs only")
        sm = SM[SM["filter"] == f].iloc[0]
        ax.set_title(f"{f}: n={sm.n_stars}  slope {sm.all_slope_nom:+.3f} → {sm.all_slope_eff:+.3f}", fontsize=8.5); ax.grid(alpha=.3); ax.set_xlim(1.7, 4.0); ax.set_ylim(-0.15, 0.15)
        if k == 0: ax.legend(fontsize=7, loc="lower left")
    for k in range(len(filters), nrow * ncol): axes[k // ncol][k % ncol].axis("off")
    for ax in axes[-1]: ax.set_xlabel("BP−RP")
    for row in axes: row[0].set_ylabel("observed − synthetic [mag]")
    fig.suptitle(f"M-type stars (m450 ≤ {m450_max:g}): observed 7DT magnitude minus XP-synthetic magnitude with the nominal (grey) and effective (orange) curves; blue = reference-star colour range", fontsize=11)
    fig.tight_layout(); fig.savefig(fig_dir / "F5_mstars_residual_vs_colour.png", dpi=110); plt.close(fig)
    # ---- F6: summary vs pivot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for lab, mk in (("M0–M2", "o"), ("M2–M3", "s"), ("M3–M4", "D")):
        b = B[B.bin == lab].merge(SM[["filter", "pivot"]], on="filter")
        axes[0].plot(b["pivot"] - 3, b.r_nom, mk, mfc="none", color="0.4", ms=6); axes[0].plot(b["pivot"] + 3, b.r_eff, mk, color="tab:orange", ms=6, label=f"{lab}")
    axes[0].plot([], [], "o", mfc="none", color="0.4", label="nominal (open)"); axes[0].axhline(0, color="k", lw=.6); axes[0].axhspan(-0.02, 0.02, color="grey", alpha=.15)
    axes[0].set_xlabel("pivot [nm]"); axes[0].set_ylabel("median residual of the colour bin [mag]"); axes[0].grid(alpha=.3); axes[0].legend(fontsize=8); axes[0].set_title("median residual per colour bin: nominal (open) vs effective (filled)", fontsize=9)
    axes[1].plot(SM["pivot"] - 3, SM.all_slope_nom, "o", mfc="none", color="0.4", label="nominal"); axes[1].plot(SM["pivot"] + 3, SM.all_slope_eff, "o", color="tab:orange", label="effective")
    axes[1].axhline(0, color="k", lw=.6); axes[1].set_xlabel("pivot [nm]"); axes[1].set_ylabel("d(residual)/d(BP−RP) over 1.8–3.5 [mag/mag]"); axes[1].grid(alpha=.3); axes[1].legend(fontsize=8); axes[1].set_title("colour dependence of the residual among M stars", fontsize=9)
    fig.tight_layout(); fig.savefig(fig_dir / "F6_mstars_residual_summary.png", dpi=130); plt.close(fig)
    # ---- F7: does the nominal M-star residual look like what the effective curves predict?  r_nom = -curve_change if the effective
    #         curves were right and XP were perfect; regression slope beta (1 = fully confirmed, 0 = no evidence)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5)); fits = []
    for ax, tag, lab in zip(axes, ("all", "dwarf", "giant"), ("all M stars", "dwarfs (parallax > 1 mas, M_G > 6)", "giants (M_G < 4)")):
        cc = S.groupby("filter").apply(lambda t: float((t.r_eff - t.r_nom).median()) if tag == "all" else float((t[t.lum == tag].r_eff - t[t.lum == tag].r_nom).median()) if (t.lum == tag).sum() >= 20 else np.nan)
        y = SM.set_index("filter")[f"{tag}_median_nom"]; x = -cc.reindex(y.index); m = np.isfinite(x) & np.isfinite(y) & ~y.index.isin(["m400", "m425"])
        beta, beta_err, b0, n = R.weighted_linfit(x[m].to_numpy(), y[m].to_numpy(), np.full(m.sum(), 0.01))
        rho = float(np.corrcoef(x[m], y[m])[0, 1]); fits.append(dict(subset=tag, n=int(m.sum()), beta=beta, beta_err=beta_err, intercept=b0, rho=rho))
        ax.plot([-0.03, 0.03], [-0.03, 0.03], "k--", lw=.8, label="β = 1 (effective curves exactly right)"); ax.axhline(0, color="k", lw=.5); ax.axvline(0, color="k", lw=.5)
        for f in y.index[m]: ax.annotate(f, (x[f], y[f]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.plot(x[m], y[m], "o", color="tab:blue", ms=5); ax.plot(x[~m & np.isfinite(x) & np.isfinite(y)], y[~m & np.isfinite(x) & np.isfinite(y)], "o", mfc="none", color="tab:blue", ms=5, label="m400, m425 (excluded)")
        xx = np.linspace(-0.03, 0.03, 2); ax.plot(xx, b0 + beta * xx, color="tab:orange", label=f"fit: β = {beta:.2f} ± {beta_err:.2f}, ρ = {rho:.2f} (n={m.sum()})")
        ax.set_xlabel("predicted M-star residual with the nominal curves = −(curve-induced change) [mag]"); ax.set_ylabel("observed median residual, nominal curves [mag]"); ax.set_title(lab, fontsize=10)
        ax.set_xlim(-0.03, 0.035); ax.set_ylim(-0.06, 0.06); ax.grid(alpha=.3); ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("M stars: is the residual of the nominal calibration the one the effective curves predict?  (per filter, median over stars; XP systematics not removed)", fontsize=10)
    fig.tight_layout(); fig.savefig(fig_dir / "F7_mstars_prediction_test.png", dpi=130); plt.close(fig)
    FT = pd.DataFrame(fits); FT.to_csv(out / "mstar_prediction_fit.csv", index=False); print(FT.round(3).to_string(index=False))
    log("figures F5, F6, F7 written")


def stage_bd11(out: Path, fig_dir: Path):
    """BD-11 3759 (M3.5V, BP-RP 2.92): the only M star with a CALSPEC spectrum AND 7DT standard-field frames.  Same frames, two
    references: CALSPEC (validation_calspec_stars.csv, closed loop of the effective curves) and XP flux_cor (this script)."""
    sid = "6324325225803432320"
    S = pd.read_csv(out / "mstar_resid_stars.csv.gz", dtype={"sid": str}); b = S[S.sid == sid]
    V = pd.read_csv(C.DEFAULT_OUT / "validation_calspec_stars.csv"); v = V[V.object == "BD-11_3759"]
    m = b.merge(v, on="filter", how="inner"); m["pivot"] = [C.pivot_nm(C.FINE, C.load_nominal(f)) for f in m["filter"]]; m = m.sort_values("pivot")
    m["xp_err_implied"] = m["before"] - m["r_nom"]
    m[["filter", "pivot", "n_x", "before", "after", "r_nom", "r_eff", "xp_err_implied", "dm_curve"]].rename(columns={"n_x": "n_frames", "before": "calspec_nominal", "after": "calspec_effective", "r_nom": "xp_nominal", "r_eff": "xp_effective"}).to_csv(out / "bd11d3759_residuals.csv", index=False)
    rms = lambda x: float(np.sqrt(np.nanmean(np.asarray(x, float) ** 2)))
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.5), sharex=True)
    ax = axes[0]
    ax.plot(m["pivot"] - 2, m.before, "o", mfc="none", color="tab:red", ms=7, label=f"CALSPEC reference, nominal curves (rms {rms(m.before):.3f})")
    ax.plot(m["pivot"] + 2, m.after, "o", color="tab:red", ms=7, label=f"CALSPEC reference, effective curves (rms {rms(m.after):.3f})")
    for x, y0, y1 in zip(m["pivot"], m.before, m.after): ax.annotate("", (x + 2, y1), (x - 2, y0), arrowprops=dict(arrowstyle="->", color="tab:red", lw=.7, alpha=.6))
    ax.axhline(0, color="k", lw=.6); ax.axhspan(-0.01, 0.01, color="grey", alpha=.15); ax.set_ylim(-0.06, 0.06); ax.grid(alpha=.3); ax.legend(fontsize=9)
    ax.set_ylabel("observed − synthetic [mag]"); ax.set_title("BD-11 3759 (M3.5V, BP−RP = 2.92), 7DT standard-field frames: residual with the TRUE spectrum (CALSPEC) before → after the curve shift", fontsize=10)
    ax = axes[1]
    ax.plot(m["pivot"] - 2, m.r_nom, "s", mfc="none", color="tab:blue", ms=7, label=f"XP (flux_cor) reference, nominal curves (rms {rms(m.r_nom):.3f})")
    ax.plot(m["pivot"] + 2, m.r_eff, "s", color="tab:blue", ms=7, label=f"XP reference, effective curves (rms {rms(m.r_eff):.3f})")
    ax.plot(m["pivot"], m.xp_err_implied, "_", color="k", ms=16, mew=2, label="implied XP error  m_XP − m_CALSPEC (= CALSPEC residual − XP residual)")
    ax.axhline(0, color="k", lw=.6); ax.axhspan(-0.01, 0.01, color="grey", alpha=.15); ax.set_ylim(-0.2, 0.2); ax.grid(alpha=.3); ax.legend(fontsize=9)
    ax.set_xlabel("pivot wavelength [nm]"); ax.set_ylabel("observed − synthetic [mag]")
    ax.set_title("same star, same frames, XP spectrum as reference: the XP synthetic magnitude itself is off by up to 0.17 mag where TiO band heads fall in a band", fontsize=10)
    for name, x in M.FEATURES:
        for a in axes: a.axvline(x, color="tab:green", lw=.5, alpha=.4)
    fig.tight_layout(); fig.savefig(fig_dir / "F9_bd11d3759_calspec_vs_xp.png", dpi=130); plt.close(fig)
    print(m[["filter", "before", "after", "r_nom", "r_eff", "xp_err_implied"]].round(4).to_string(index=False)); log("F9 written")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all", choices=["match", "compute", "bd11", "all"])
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--n-proc", type=int, default=32)
    ap.add_argument("--m450-max", type=float, default=17.0)
    args = ap.parse_args()
    out = Path(args.out_dir); fig_dir = out / f"figures_m450le{args.m450_max:g}"; fig_dir.mkdir(parents=True, exist_ok=True)
    if args.stage in ("match", "all"): stage_match(out, args.n_proc, args.m450_max)
    if args.stage in ("compute", "all"): stage_compute(out, fig_dir, args.m450_max)
    if args.stage in ("bd11", "all"): stage_bd11(out, fig_dir)


if __name__ == "__main__":
    main()
