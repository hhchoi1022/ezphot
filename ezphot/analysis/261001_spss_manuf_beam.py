#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
261001_spss_manuf_beam.py — section 8.2: Gaia SPSS standards (2026-09 campaign) calibrated with the same curves and the
same frame cleaning as section 8.1 (261001_calspec_manuf_beam.py, CALSPEC 2024).

Curves (filter x QE x optics x El Sauce atmosphere, vacuum wavelengths): manufacturer nobeam / beam n_eff(lam0) /
beam n_eff 1.5 (reference), and the lab curve with the per-filter n_eff beam ("lab_beam"; CM.n_eff_filter: measured n_eff
for 15 filters since 2026-10-07, provisional n_eff(lam0) for the rest): version_mhj for the old-generation
25-nm filters, version_hyp for the new-generation filters (m386, m438, m466w, m483, m512, m534, m586, m615, m661, m692w,
m710w, m769w).  m832w (defective) and the broad g / r are left out.
Frames: every SPSS frame of the re-photometered catalogs (260927_std_rephot.py; 7DT11 excluded there, its wheel stayed
at r), processed with 260927_std_offset.process_spss (field ZP from corrected Gaia XP through the same curve, 10" aperture
MAG_APER_3).
Cleaning and filter identity: 261001_calspec_manuf_beam.clean_frames / filter_identity rules (cloudy nights, colour-term
outliers beyond 10 x max(1.4826 MAD, 10 mmag/mag), wrong filters; m375w exempt from the identity flag: its vendor curve is
~7 nm off, see ID_EXEMPT); in addition a whole night is dropped when the within-night scatter of the frame ZP residuals
(all units and filters) exceeds 3x the median night and 0.03 mag (UNSTABLE_K / UNSTABLE_MIN; only 2026-09-16/17 UT).  A frame
flagged with any curve is dropped for all curves.
The standard: FLAGS 0, peak <= 50k ADU, SNR >= 100 (the SPSS cut of section 9); the faint-star bias (faintcheck_bins,
SPSS rows) is removed as in 8.1.
Offset per frame: Delta = MAG(std) + global field ZP (all ZP stars of the frame, the standard excluded: zp_x) - syn(SPSS V2
spectrum, same curve); per star / filter / curve the median over frames.  (Until 2026-10-01 the local ZP within 1500 px was
used; that version is kept in 20261001_spss_manuf_v12_localzp/.)
Outputs (~/ezphot/log/20261001_spss_manuf/): spss_manuf_frames.csv, spss_frames_flags.csv, filter_id_frames.csv,
  filter_id_unitnight.csv, spss_zp_ct_summary.csv, spss_sed_long.csv, spss_summary.csv, spss_vs_calspec.csv,
  spss_unit_offsets.csv, spss_unit_pairs.csv, lab_offsets.csv, ky_offsets.csv / ky_compare.csv / P4_spss_ky.png
  (version_ky = lab set 1 for m438 / m483 / m512, with and without the beam),
  P6_spss_ct_vs_hotcool.png / spss_ct_vs_hotcool.csv (field colour term vs hot - cool: response ratio r and the
  curve-independent gap), spss_hot_cool.csv (hot WD/sdO - cool FGKM per filter),
  spss_hot_cool_shift.csv (curve shift that zeroes hot - cool, vs the lab offset),
  hot_calspec_vs_spss.csv (hot stars, CALSPEC 2024 - SPSS 2026), common_stars_calspec_vs_spss.csv (the four stars in
  both campaigns: CALSPEC - SPSS split into the observation and the spectrum parts),
  Z3_spss_zp.png, Z4_spss_colorterm.png (default view) and Z4_spss_colorterm_<bits>.png (16 views: mhj·beam, hyp·beam,
  ky·beam, CALSPEC medians of Z2 on/off, bits in Z4_SERIES order), P1_spss_sed.png,
  P2a_spss_offsets.png + P2a_spss_offsets_<bits>.png (all-star median per filter, same 16 views as Z4),
  P2b_spss_hotcool.png + P2b_spss_hotcool_<bits>.png (hot - cool per filter, 8 views: mhj·beam, hyp·beam, ky·beam on/off),
  P2c_spss_levels.png (the curve-independent levels), P3_spss_units.png
  MPLCONFIGDIR=/tmp/ezphot-matplotlib OMP_NUM_THREADS=1 python -u ezphot/analysis/261001_spss_manuf_beam.py --n-proc 32
"""
import argparse
import importlib
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
CM = importlib.import_module("261001_calspec_manuf_beam")
SO, SS, MB, SF, R = CM.SO, CM.SS, CM.MB, CM.SF, CM.R
OUT = Path("/home/hhchoi1022/ezphot/log/20261001_spss_manuf")
CAL = Path("/home/hhchoi1022/ezphot/log/20261001_calspec_manuf")
REG = Path("/home/hhchoi1022/ezphot/log/20260927_std_newbkg/spss_registered_bprp.csv")
NEFF = Path("/home/hhchoi1022/ezphot/log/20261001_neff/neff_filters.csv")
SNR_MIN = 100.0
SKIP = {"g", "r", "m832w"}
# m375w: the vendor curve is ~7 nm too blue (Step A, red edge), so the field-star test prefers the redder m386 although no
# candidate fits the standards (m375w +274, m386 +110, m400 -229 mmag on GD71); not treated as a wrong filter
ID_EXEMPT = {"m375w"}
# hot = blue stars (BP-RP < 0.3: WD, sdO/sdB); cool = F G K stars with field-like colours (BP-RP <= 1.3; the field ZP stars
# have a median BP-RP of 0.88, 5-95% of frames 0.75-1.42), so a curve error cancels for them. Neither group: the DA9
# WD0018-267 (BP-RP 0.90, field colour with a WD spectrum) and the red LTT377 (K9, 1.87) and LP885-23 (M0, 2.66)
HOT_BPRP, COOL_BPRP_MAX = 0.3, 1.3
UNSTABLE_K, UNSTABLE_MIN = 3.0, 0.03
COOL = {"FGK", "M"}
# stars observed in both CALSPEC 2024 and SPSS 2026 (SPSS name: CALSPEC name)
COMMON = {"GD71": "GD71", "WD2039-682": "WD2039-682", "EG131": "GJ754.1A", "G93-48": "WD2149+021", "Feige110": "FEIGE110"}
VARIANTS = {"nobeam": ("manuf·nobeam", "0.25", "o"), "beam": ("manuf·beam, per-filter n_eff", "#C25B4E", "o"),
            "lab_beam": ("lab (mhj / hyp)·beam, per-filter n_eff", "#7B3FA0", "D"),
            "beam_neff1.5": ("manuf·beam, n_eff 1.5 (previous)", "#7F9CC9", "s"),
            "ky_beam": ("lab set 1 (ky)·beam, per-filter n_eff", "#2F5D9E", "^"), "ky_nobeam": ("lab set 1 (ky)·nobeam", "#8FB3DE", "v")}
CURVE_KEY = {"nobeam": ("manufacturer", "nobeam"), "beam": ("manufacturer", "beam"), "lab_beam": ("lab", "beam"),
             "beam_neff1.5": ("manufacturer", "beam_neff1.5"), "ky_beam": ("version_ky", "beam"),
             "ky_nobeam": ("version_ky", "nobeam")}
KY_FILTERS = ["m438", "m483", "m512"]
# curves defined for every filter, used to measure how hot - cool follows the field colour term (ct_vs_hotcool)
CT_CURVES = ["nobeam", "beam", "beam_neff1.5", "lab_beam"]
# Z4 / P2a switches (file name Z4_spss_colorterm_<bits>.png / P2a_spss_offsets_<bits>.png, one bit per switch in this
# order) and the default view
Z4_SERIES = ["mhj", "hyp", "ky", "cal"]
Z4_DEFAULT = "1100"
P2A_DEFAULT = "1101"    # P2a opens with the CALSPEC medians shown, as the former P2 (a)
# P2b (hot - cool) switches: P2b_spss_hotcool_<bits>.png; no CALSPEC hot - cool exists
HC_SERIES = ["mhj", "hyp", "ky"]
HC_DEFAULT = "110"
CT_MIN_SPREAD = 8.0     # mmag: r is fitted only where the curves spread the predicted hot - cool by at least this
mad = CM.mad
fkey = lambda f: int(f[1:4])   # noqa: E731


def totals():
    """CM.manuf_totals, with the lab curve = version_mhj (old generation) + version_hyp (new generation), both with the
    n_eff(lam0) beam (lam0 = centroid of that lab curve)."""
    TC, DS = SO.TC, SO.DS
    out = CM.manuf_totals()
    lab = dict(out.pop(("version_mhj", "beam")))
    out.pop(("version_mhj", "nobeam"), None)      # added to manuf_totals for the CALSPEC figures (2026-10-05); not used here
    atm = pd.read_csv(TC.TS / "atmosphere_elsauce.csv"); qe = pd.read_csv(TC.SRC / "QE.csv")
    opt = pd.read_csv(TC.SRC / "optics.efficiency.dr350.dr500.csv", sep=r"\s+", comment="#", skiprows=11)
    for p in sorted((TC.VBASE / "version_hyp").glob("m*.csv")):
        if p.stem in TC.BAD_FILTERS or p.stem in lab:
            continue
        d = pd.read_csv(p)
        lam = d.lam_nm.to_numpy(float); tr = d.trans.to_numpy(float)
        env = (np.interp(lam, qe.iloc[:, 0], qe.iloc[:, 1]) * np.interp(lam, opt.iloc[:, 0], opt.iloc[:, -1])
               * np.interp(lam, atm["lam_air"], atm["trans"]))
        lam0 = np.trapz(lam * tr, lam) / np.trapz(tr, lam)
        lab[p.stem] = (DS.air_to_vac_nm(lam), TC.dr500_transmission(lam, tr, n_eff=CM.n_eff_filter(p.stem, lam0), f_number=3.03,
                                                                   obscuration=0.59) * env)
    out[("lab", "beam")] = lab
    # version_ky (lab set 1, roi200: m438 / m483 / m512, band core only), with and without the n_eff(lam0) beam
    kb, kn = {}, {}
    for p in sorted((TC.VBASE / "version_ky").glob("m*.csv")):
        d = pd.read_csv(p)
        lam = d.lam_nm.to_numpy(float); tr = np.clip(d.trans.to_numpy(float), 0, None)
        env = (np.interp(lam, qe.iloc[:, 0], qe.iloc[:, 1]) * np.interp(lam, opt.iloc[:, 0], opt.iloc[:, -1])
               * np.interp(lam, atm["lam_air"], atm["trans"]))
        lam0 = np.trapz(lam * tr, lam) / np.trapz(tr, lam)
        kn[p.stem] = (DS.air_to_vac_nm(lam), tr * env)
        kb[p.stem] = (DS.air_to_vac_nm(lam), TC.dr500_transmission(lam, tr, n_eff=CM.n_eff_filter(p.stem, lam0), f_number=3.03,
                                                                  obscuration=0.59) * env)
    out[("version_ky", "beam")], out[("version_ky", "nobeam")] = kb, kn
    return out


def lab_offsets():
    """lab - manufacturer centroid of the bare filter curve [nm], and which lab version is used."""
    TC = SO.TC
    rows = []
    for p in sorted((TC.VBASE / "manufacturer").glob("m*.csv")):
        f = p.stem
        for ver in ("version_mhj", "version_hyp"):
            q = TC.VBASE / ver / f"{f}.csv"
            if q.exists() and f not in TC.BAD_FILTERS:
                a, b = pd.read_csv(p), pd.read_csv(q)
                ca = np.trapz(a.lam_nm * a.trans, a.lam_nm) / np.trapz(a.trans, a.lam_nm)
                cb = np.trapz(b.lam_nm * b.trans, b.lam_nm) / np.trapz(b.trans, b.lam_nm)
                rows.append(dict(filter=f, lab=ver.split("_")[1], manuf_centroid=round(ca, 2), lab_centroid=round(cb, 2),
                                 lab_offset_nm=round(cb - ca, 2)))
                break
    return pd.DataFrame(rows)


def ky_offsets():
    """version_ky - manufacturer centroid of the bare filter curve [nm] (above 1 % of the peak, as in section 5; curves
    interpolated to 0.01 nm first: on the 1-nm hyp grid the 1 % cut moves the centroid by up to 0.06 nm)."""
    TC = SO.TC
    def cen(f):
        d = pd.read_csv(f); l0, t0 = d.lam_nm.to_numpy(float), np.clip(d.trans.to_numpy(float), 0, None)
        l = np.arange(l0.min(), l0.max(), 0.01); t = np.interp(l, l0, t0)
        m = t > 0.01 * t.max()
        return np.trapz(l[m] * t[m], l[m]) / np.trapz(t[m], l[m])
    rows = []
    for f in KY_FILTERS:
        a, k = cen(TC.VBASE / "manufacturer" / f"{f}.csv"), cen(TC.VBASE / "version_ky" / f"{f}.csv")
        h = cen(TC.VBASE / "version_hyp" / f"{f}.csv")
        rows.append(dict(filter=f, manuf_centroid=round(a, 2), ky_centroid=round(k, 2), hyp_centroid=round(h, 2),
                         ky_offset_nm=round(k - a, 2), hyp_offset_nm=round(h - a, 2)))
    return pd.DataFrame(rows)


def plot_ky(HC, Sm, HS, KO):
    """P4: the three filters with a version_ky curve: hot - cool and the all-star level for every curve, and the sky
    curve shift (hot - cool converted, relative to manuf·beam) against the ky and hyp offsets."""
    order = ["nobeam", "beam", "beam_neff1.5", "lab_beam", "ky_nobeam", "ky_beam"]
    short = {"nobeam": "manuf\nnobeam", "beam": "manuf\nbeam", "beam_neff1.5": "manuf\nbeam 1.5", "lab_beam": "hyp\nbeam",
             "ky_nobeam": "ky\nnobeam", "ky_beam": "ky\nbeam"}
    fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharey="row")
    for k, f in enumerate(KY_FILTERS):
        for row, (tab, val, err, ttl) in enumerate([(HC, "hot_minus_cool", "err", "hot − cool"), (Sm, "median", "scatter", "all-star level")]):
            ax = axs[row, k]
            q = tab[tab["filter"] == f].set_index("curve")
            for j, v in enumerate(order):
                if v not in q.index:
                    continue
                lab, c, mk = VARIANTS[v]
                ax.errorbar(j, q.loc[v, val], q.loc[v, err], fmt=mk, color=c, ms=8, capsize=3, mec="k" if v.startswith("ky") else c)
            ax.axhline(0, color="k", lw=0.6); ax.axhspan(-5, 5, color="0.92", zorder=0); ax.grid(alpha=0.3, axis="y")
            ax.set_xticks(range(len(order))); ax.set_xticklabels([short[v] for v in order], fontsize=8)
            if row == 0:
                hs = HS.set_index("filter").loc[f]; ko = KO.set_index("filter").loc[f]
                ax.set_title(f"{f}: sky shift {hs.shift_nm:+.2f} ± {hs.shift_err:.2f} nm; ky {ko.ky_offset_nm:+.2f}, hyp {ko.hyp_offset_nm:+.2f} nm",
                             fontsize=9.5, loc="left")
            if k == 0:
                ax.set_ylabel(f"{ttl} [mmag]")
    fig.suptitle("P4  SPSS: the three filters with a version_ky (lab set 1) curve. Top: hot (BP−RP < 0.3) − cool (F G K) median "
                 "with bootstrap error; bottom: all-star median ± scatter", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(OUT / "P4_spss_ky.png", dpi=110); plt.close(fig)


def ct_vs_hotcool(L, HS, nboot=1000):
    """Field colour term against hot - cool.  pred = what the field colour term predicts for the residual of a standard,
    (zp_x - zg_c0) - a_x (BP-RP_std - bprp0) per frame (Delta - pred is the residual after applying the colour term),
    median per star like resid.  Per filter, over CT_CURVES: hot - cool(resid) = r * hot - cool(pred) + gap.
    r = how much of the linear BP-RP extrapolation of the field term the standards see when the curve changes;
    gap = hot - cool of the curve whose field colour term would be zero (curve-independent part: field (corrected XP)
    and standards (SPSS spectra) disagree, or the samples differ); gap_nm = -gap / slope (hot - cool per nm, from
    spss_hot_cool_shift.csv) = how much redder (+) the standards want the curve than the field.  Paired bootstrap:
    the same hot and cool stars for every curve; every resample is fitted (frac_valid = share of resamples whose curves
    spread hot - cool(pred) by >= CT_MIN_SPREAD).  r is not reported where the point estimate has a spread below
    CT_MIN_SPREAD.  after_ct = hot - cool of the per-star median of (Delta - pred), i.e. after applying the colour term,
    manuf·beam.  r depends on synthetic photometry only (the observed magnitudes are the same for every curve)."""
    sl = HS.set_index("filter").slope_mmag_per_nm
    rows = []
    for f, g in L[L.curve.isin(CT_CURVES)].groupby("filter"):
        W = g.pivot_table(index="object", columns="curve", values=["resid", "pred", "resid_ct"])
        grp = g.drop_duplicates("object").set_index("object").grp.reindex(W.index)
        h, c = W.index[grp == "hot"], W.index[grp == "cool"]
        if len(h) < 3 or len(c) < 3 or not set(CT_CURVES) <= set(W["resid"].columns):
            continue
        R_, P_ = W["resid"][CT_CURVES], W["pred"][CT_CURVES]
        def fit(hh, cc):
            y = (R_.loc[hh].median() - R_.loc[cc].median()).to_numpy(float)
            x = (P_.loc[hh].median() - P_.loc[cc].median()).to_numpy(float)
            r, gp = np.polyfit(x, y, 1) if np.ptp(x) > 0 else (np.nan, np.nan)
            return y, x, r, gp, np.ptp(x) >= CT_MIN_SPREAD
        y, x, r, gp, ok0 = fit(h, c)
        if not ok0:
            r = gp = np.nan
        rng = np.random.default_rng(1)
        bs = [fit(rng.choice(h, len(h)), rng.choice(c, len(c)))[2:] for _ in range(nboot)]
        valid = np.mean([v[2] for v in bs])
        ab = np.array([v[:2] for v in bs], float); ab = ab[np.isfinite(ab[:, 0])]
        b = CT_CURVES.index("beam")
        lo, hi = (np.percentile(ab, [16, 84], axis=0) if np.isfinite(r) else np.full((2, 2), np.nan))
        RC = W["resid_ct"]["beam"]
        slope = sl.get(f, np.nan)
        rows.append(dict(filter=f, lam=round(float(g[g.curve == "beam"].lam.median()), 2), n_hot=len(h), n_cool=len(c),
                         dbprp=round(float(g[g.grp == "hot"].bprp.median() - g[g.grp == "cool"].bprp.median()), 3),
                         pred_beam=x[b], hc_beam=y[b], after_ct_beam=RC.loc[h].median() - RC.loc[c].median(),
                         spread_pred=np.ptp(x), spread_hc=np.ptp(y), frac_valid=valid,
                         r=r, r_lo=lo[0], r_hi=hi[0], gap=gp, gap_lo=lo[1], gap_hi=hi[1],
                         slope_mmag_per_nm=slope, gap_nm=-gp / slope if np.isfinite(slope) and abs(slope) >= 2 else np.nan))
    T = pd.DataFrame(rows).sort_values("filter", key=lambda q: q.map(fkey))
    fig, axs = plt.subplots(3, 1, figsize=(14, 11), sharex=True, gridspec_kw=dict(height_ratios=[1.3, 1, 1]))
    ax = axs[0]
    ax.plot(T.lam, T.pred_beam, "s--", color="0.45", ms=5, lw=0.9, label="field colour term, linear in BP−RP: −a·ΔBP−RP (manuf·beam)")
    ax.plot(T.lam, T.hc_beam, "o-", color="#C25B4E", ms=5, lw=1.0, label="observed hot − cool (manuf·beam)")
    ax.plot(T.lam, T.after_ct_beam, "^:", color="#2F5D9E", ms=5, lw=0.9, label="hot − cool after applying the field colour term")
    ax.axhline(0, color="k", lw=0.5); ax.axhspan(-5, 5, color="0.92", zorder=0); ax.grid(alpha=0.3); ax.set_ylim(-70, 100)
    ax.set_ylabel("[mmag]"); ax.legend(fontsize=8, loc="upper right")
    ax.set_title("(a) prediction of the field colour term for hot − cool, the observed hot − cool, and the difference", fontsize=9, loc="left")
    ok = T.r.notna()
    # bootstrap 16-84 % interval; it need not contain the point estimate (medians of few stars), hence the clip
    axs[1].errorbar(T.lam[ok], T.r[ok], np.clip([T.r[ok] - T.r_lo[ok], T.r_hi[ok] - T.r[ok]], 0, None), fmt="o", color="#7B3FA0", ms=5, capsize=0)
    axs[1].plot(T.lam[~ok], np.zeros((~ok).sum()) + 0.05, "x", color="0.5", ms=6, label=f"not fitted (curves spread the prediction < {CT_MIN_SPREAD:.0f} mmag)")
    axs[1].axhline(1, color="k", lw=0.6, ls="--"); axs[1].axhline(0, color="k", lw=0.5); axs[1].set_ylim(-1.0, 2.5); axs[1].grid(alpha=0.3)
    axs[1].set_ylabel("r"); axs[1].legend(fontsize=8, loc="upper left")
    axs[1].set_title(f"(b) r: change of hot − cool per change of the prediction when the curve changes ({len(CT_CURVES)} curves, paired bootstrap 16–84 %, all resamples)",
                     fontsize=9, loc="left")
    axs[2].errorbar(T.lam[ok], T.gap[ok], np.clip([T.gap[ok] - T.gap_lo[ok], T.gap_hi[ok] - T.gap[ok]], 0, None), fmt="D", color="#3C7A5A", ms=5, capsize=0)
    axs[2].axhline(0, color="k", lw=0.5); axs[2].axhspan(-5, 5, color="0.92", zorder=0); axs[2].grid(alpha=0.3); axs[2].set_ylim(-60, 60)
    axs[2].set_ylabel("gap [mmag]"); axs[2].set_xlabel("filter pivot wavelength [nm]")
    axs[2].set_title("(c) gap: hot − cool of the curve whose field colour term is zero (the part a curve shift cannot remove)", fontsize=9, loc="left")
    sec = axs[0].secondary_xaxis("top"); sec.set_xticks(T.lam.to_numpy()); sec.set_xticklabels(T["filter"], rotation=90, fontsize=6.5)
    fig.suptitle("P6  SPSS: field colour term (corrected Gaia XP field stars) vs hot − cool of the standards: hot − cool = r × prediction + gap",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(OUT / "P6_spss_ct_vs_hotcool.png", dpi=110); plt.close(fig)
    return T


def _init():
    warnings.filterwarnings("ignore")
    SO.AN._init()
    SO.TW._init(40)
    SO._TOTALS = totals()


def spss_jobs():
    M = pd.read_csv(SO.OUT / "rephot_meta.csv")
    M = M[M.status.isin(["ok", "cached"]) & (M.dataset == "SPSS") & ~M["filter"].isin(SKIP)].drop_duplicates("file")
    M = M[[(SO.CATD / (Path(f).name[:-5] + "_cat.fits")).exists() for f in M.file]]
    sp = pd.read_csv(SO.AN.SPSS_CSV)
    srow_of = {SO.AN.norm(s.Name): s for _, s in sp.iterrows()}
    jobs = [(obj, list(zip(g.file, g["filter"], g.unit)), srow_of[SO.AN.norm(obj)])
            for (obj, _), g in M.groupby(["object", "filter"]) if SO.AN.norm(obj) in srow_of]
    return M, jobs


def run_frames(n_proc):
    M, jobs = spss_jobs()
    print(time.strftime("[%H:%M:%S]"), f"SPSS frames {len(M)}, stars {M.object.nunique()}, jobs {len(jobs)}", flush=True)
    rows = []
    with Pool(n_proc, initializer=_init) as pool:
        for r in pool.imap_unordered(SO.process_spss, jobs):
            rows += r
    F = pd.DataFrame(rows)
    F.to_csv(OUT / "spss_manuf_frames.csv", index=False)
    print(time.strftime("[%H:%M:%S]"), f"saved {len(F)} rows, {F.file.nunique()} frames", flush=True)
    return F


def filter_identity(n_proc):
    M, jobs = spss_jobs()
    rows = []
    with Pool(n_proc, initializer=CM._init_id, initargs=(sorted(M["filter"].unique()),)) as pool:
        for r in pool.imap_unordered(SO.process_spss, jobs):
            rows += [dict(file=x["file"], object=x["object"], filter=x["filter"], unit=x["unit"], mjd=x["mjd"],
                          cand=x["version"], zsd=x["zsd"], a_x=x["a_x"]) for x in r]
    I = pd.DataFrame(rows)
    I.to_csv(OUT / "filter_id_frames.csv", index=False)
    print(time.strftime("[%H:%M:%S]"), f"filter identity: {I.file.nunique()} frames", flush=True)
    return I


def plot_zp_ct(F, LO):
    filters = sorted(F["filter"].unique(), key=fkey)
    units = sorted(F.unit.unique()); cmap = plt.get_cmap("tab20")
    nc = 6; nr = int(np.ceil(len(filters) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(3.6 * nc, 2.7 * nr), squeeze=False)
    b = F[F.curve == "nobeam"]
    for ax, flt in zip(axs.ravel(), filters):
        g = b[b["filter"] == flt].sort_values("mjd")
        for u, q in g.groupby("unit"):
            ax.plot(q.mjd - 61000, q.zp_model_res, ".", ms=3, color=cmap(units.index(u) % 20), label=u)
        bad = g[~g.keep]
        ax.plot(bad.mjd - 61000, bad.zp_model_res, "x", color="r", ms=4, mew=0.8)
        ax.axhline(0, color="k", lw=0.5); ax.set_ylim(-0.4, 0.25); ax.grid(alpha=0.3); ax.tick_params(labelsize=6.5)
        ax.set_title(f"{flt}: {len(g)} frames, {len(bad)} removed", fontsize=8, loc="left")
        ax.legend(fontsize=5.5, loc="lower left", ncol=2, markerscale=2)
    for ax in axs.ravel()[len(filters):]:
        ax.axis("off")
    fig.supxlabel("MJD − 61000"); fig.supylabel("ZP per second − (median night + k·airmass) [mag]")
    fig.suptitle("Z3  SPSS frames: field ZP per frame (corrected Gaia XP, manufacturer·nobeam) relative to the per-unit "
                 "airmass model; red x = removed (cloudy or unstable night, colour-term outlier or wrong filter)", fontsize=10)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.96)); fig.savefig(OUT / "Z3_spss_zp.png", dpi=100); plt.close(fig)

    # Z4 in 16 versions for the report's checkboxes: manufacturer nobeam / beam always, plus any of the lab curves
    # (mhj = old-generation filters, hyp = new-generation filters), ky (0 deg)·beam and the CALSPEC 2024 medians of Z2
    for bits in range(16):
        on = dict(zip(Z4_SERIES, [(bits >> (3 - k)) & 1 == 1 for k in range(4)]))
        tag = "".join("1" if on[k] else "0" for k in Z4_SERIES)
        plot_z4(F, LO, filters, on, OUT / f"Z4_spss_colorterm_{tag}.png")
        if tag == Z4_DEFAULT:
            plot_z4(F, LO, filters, on, OUT / "Z4_spss_colorterm.png")


def plot_z4(F, LO, filters, on, path):
    """Z4: field colour term per frame (dots) and the median per filter (lines) for manuf·nobeam / manuf·beam and the
    series switched on in `on` (mhj·beam, hyp·beam, ky·beam, CALSPEC 2024 medians).  Each series has its own fixed
    jitter so that a dot stays where it is whatever else is drawn."""
    fam = LO.set_index("filter")["lab"]
    series = [("nobeam", "nobeam", None, "manuf·nobeam", "0.25", "o", 1),
              ("beam", "beam", None, "manuf·beam, per-filter n_eff", "#C25B4E", "o", 2),
              ("mhj", "lab_beam", "mhj", "mhj·beam, per-filter n_eff (old-generation filters)", "#7B3FA0", "D", 3),
              ("hyp", "lab_beam", "hyp", "hyp·beam, per-filter n_eff (new-generation and wide filters)", "#2E8B7A", "h", 4),
              ("ky", "ky_beam", None, "ky (0°)·beam, per-filter n_eff", "#2F5D9E", "^", 5)]
    off = {"nobeam": -0.27, "beam": -0.09, "mhj": 0.09, "hyp": 0.09, "ky": 0.27}
    pv = {f: k for k, f in enumerate(filters)}
    fig, ax = plt.subplots(figsize=(16, 5.5))
    for key, curve, lab_ver, lab, c, mk, seed in series:
        if key in on and not on[key]:
            continue
        g = F[F.curve == curve].sort_values("file")
        if lab_ver is not None:
            g = g[g["filter"].map(fam) == lab_ver]
        x = g["filter"].map(pv) + off[key] + np.random.default_rng(seed).normal(0, 0.03, len(g))
        ax.plot(x[g.keep], 1e3 * g.a_x[g.keep], ".", ms=1.5, color=c, alpha=0.12, zorder=1)
        ax.plot(x[~g.keep], 1e3 * g.a_x[~g.keep], "x", ms=2.5, mew=0.6, color="r", alpha=0.12, zorder=2)
        m = g[g.keep].groupby("filter").a_x.median()
        o = sorted(m.index, key=pv.get)
        # mhj / hyp: one line through all their filters (2026-10-05, user); ky (three filters) joins neighbours only
        xs = np.array([pv[f] + off[key] for f in o]); ys = 1e3 * m[o].to_numpy()
        brk = np.r_[False, np.diff([pv[f] for f in o]) > 1] if key == "ky" else np.zeros(len(o), bool)
        ax.plot(np.insert(xs, np.where(brk)[0], np.nan), np.insert(ys, np.where(brk)[0], np.nan), mk + "-", color=c, ms=7, lw=2.2,
                mec="white", mew=0.8, zorder=5, label=f"{lab}: median")
    if on["cal"]:
        Zc = pd.read_csv(CAL / "calspec_zp_ct_summary.csv").set_index("filter")
        for key, col, lab, c, mk in (("nobeam", "ct_nobeam_mmag", "manuf·nobeam", "0.25", "o"), ("beam", "ct_beam_mmag", "manuf·beam", "#C25B4E", "o"),
                                     ("mhj", "ct_mhj_beam_mmag", "mhj·beam", "#7B3FA0", "D")):
            if key == "mhj" and not on["mhj"]:
                continue
            o = sorted([f for f in Zc.index if f in pv], key=pv.get)
            ax.plot([pv[f] + off[key] for f in o], Zc.loc[o, col].to_numpy(), mk + "--", color=c, mfc="white", mec=c, mew=1.4,
                    ms=7, lw=1.6, zorder=6, label=f"CALSPEC 2024 (Z2), {lab}: median")
    ax.axhline(0, color="k", lw=0.5); ax.axhspan(-5, 5, color="0.92", zorder=0); ax.set_ylim(-60, 80); ax.grid(alpha=0.3)
    ax.set_xlabel("filter"); ax.set_ylabel("field colour term a [mmag per mag BP−RP]"); ax.legend(fontsize=7.5, ncol=2, loc="upper right")
    ax.set_xticks(range(len(filters))); ax.set_xticklabels(filters, rotation=45, fontsize=7)
    ax.set_title("Z4  Field colour term per frame (ZP_i vs BP−RP of the field stars, the standard excluded), SPSS frames; dots = "
                 "kept frames, red x = removed; grey band ±5 mmag/mag", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_sed(L, spec):
    stars = L.drop_duplicates("object").sort_values("bprp").object.tolist()
    nc = 5; nr = int(np.ceil(len(stars) / nc))
    fig = plt.figure(figsize=(4.8 * nc, 5.4 * nr))
    outer = fig.add_gridspec(nr, nc, hspace=0.32, wspace=0.28)
    for i, o in enumerate(stars):
        rr, cc = divmod(i, nc)
        inner = outer[rr, cc].subgridspec(2, 1, height_ratios=[2.2, 1], hspace=0.06)
        ax = fig.add_subplot(inner[0]); axr = fig.add_subplot(inner[1], sharex=ax)
        g = L[L.object == o]
        w, f = spec[o]
        m = (w > 360) & (w < 900)
        ax.plot(w[m], MB.ab_spectrum(w[m], f[m]), "-", color="0.72", lw=0.7, zorder=1, label="SPSS V2 spectrum")
        for j, v in enumerate(["nobeam", "beam", "lab_beam"]):
            q = g[g.curve == v].sort_values("lam")
            lab, c, mk = VARIANTS[v]
            dx = (j - 1) * 1.2
            ax.plot(q.lam + dx, q.m_obs, mk, color=c, ms=3.4, zorder=3, label=f"7DT, {lab}")
            ax.plot(q.lam + dx, q.m_syn, mk, mfc="none", mec=c, ms=4.6, mew=0.7, zorder=2)
            axr.plot(q.lam + dx, q.resid, mk + "-", color=c, ms=3, lw=0.8, label=lab)
        ax.invert_yaxis(); ax.grid(alpha=0.3); ax.tick_params(labelsize=7, labelbottom=False)
        fl = g.flag.iloc[0]
        ax.set_title(f"{o} ({g.sptype.iloc[0]}, BP−RP {g.bprp.iloc[0]:.2f}, flag {fl:.0f})", fontsize=8, loc="left")
        ax.set_ylabel("AB mag", fontsize=7)
        axr.axhline(0, color="k", lw=0.5); axr.axhspan(-10, 10, color="0.92", zorder=0); axr.grid(alpha=0.3)
        axr.tick_params(labelsize=7)
        lo, hi = np.nanpercentile(g.resid, [2, 98]) if len(g) > 3 else (-50, 50)
        pad = max(10, 0.15 * (hi - lo))
        axr.set_ylim(max(-150, lo - pad), min(150, hi + pad))
        axr.set_ylabel("obs − syn [mmag]", fontsize=7); axr.set_xlim(360, 900)
        if i == 0:
            ax.legend(fontsize=6, loc="upper right")
    fig.suptitle("P1  SPSS standards (2026-09) calibrated with the manufacturer curve: filled = observed 7DT magnitude (MAG_APER_3 + "
                 "field ZP), open = synthetic photometry of the SPSS V2 spectrum through the same curve, grey = spectrum;\n"
                 "red = manufacturer·beam and purple = lab (mhj / hyp)·beam, both with per-filter n_eff; black = manufacturer, no beam; "
                 "bottom = residual obs − syn (grey band ±10 mmag); flag = SPSS spectral-quality flag (0 clean, 1 <3%, 2 >3% defects)",
                 fontsize=10)
    fig.savefig(OUT / "P1_spss_sed.png", dpi=95, bbox_inches="tight")
    plt.close(fig)


def hot_cool(L, nboot=500):
    """per filter / curve: median residual of the hot stars (BP-RP < 0.3) minus that of the cool stars (F G K, BP-RP <= 1.3), with a
    bootstrap error (stars resampled within each group); needs >= 3 stars in each group.  The field ZP is set by FGK field
    stars, so a curve error largely cancels for the cool standards and shows up in the hot ones; the flux-scale difference
    between the SPSS spectra and corrected XP at field colours cancels in the difference."""
    rows = []
    for (flt, cv), g in L.groupby(["filter", "curve"]):
        h, c = g[g.grp == "hot"].resid.to_numpy(), g[g.grp == "cool"].resid.to_numpy()
        if len(h) < 3 or len(c) < 3:
            continue
        rng = np.random.default_rng(0)
        bs = [np.median(rng.choice(h, len(h))) - np.median(rng.choice(c, len(c))) for _ in range(nboot)]
        rows.append(dict(filter=flt, curve=cv, lam=g.lam.median(), n_hot=len(h), n_cool=len(c), hot=np.median(h),
                         cool=np.median(c), hot_minus_cool=np.median(h) - np.median(c), err=np.std(bs)))
    return pd.DataFrame(rows)


def hc_shift(HC, LO):
    """curve shift [nm] (relative to manuf·beam n_eff(lam0)) that zeroes hot - cool, by linear interpolation between the
    n_eff(lam0) and n_eff 1.5 beam curves (their centroid difference from 261001_neff_filters.py); compared with the lab
    offset.  Undetermined where the two curves differ by < 0.2 nm (red filters) or hot - cool hardly responds (|slope| < 2)."""
    N = pd.read_csv(NEFF).set_index("filter")
    W = HC.pivot_table(index="filter", columns="curve", values=["hot_minus_cool", "err"])
    rows = []
    for f in W.index:
        if f not in N.index:
            continue
        dc = N.loc[f, "dcentroid_neff1p5_nm"] - N.loc[f, "dcentroid_nm"]
        a, b = W.loc[f, ("hot_minus_cool", "beam")], W.loc[f, ("hot_minus_cool", "beam_neff1.5")]
        sl = (b - a) / dc if abs(dc) >= 0.2 else np.nan
        ok = np.isfinite(sl) and abs(sl) >= 2
        rows.append(dict(filter=f, dc_neff1p5_nm=round(dc, 2), hot_minus_cool=round(a, 2), err=round(W.loc[f, ("err", "beam")], 2),
                         slope_mmag_per_nm=round(sl, 2), shift_nm=round(-a / sl, 2) if ok else np.nan,
                         shift_err=round(W.loc[f, ("err", "beam")] / abs(sl), 2) if ok else np.nan,
                         lab_offset_nm=LO.set_index("filter").lab_offset_nm.get(f, np.nan)))
    return pd.DataFrame(rows).sort_values("filter", key=lambda x: x.map(fkey))


def same_star(L, CL, spec, tot):
    """stars observed in both campaigns: resid(CALSPEC) - resid(SPSS) = [obs 2024 - obs 2026] - [syn(CALSPEC spectrum) -
    syn(SPSS spectrum)], per star / filter / curve."""
    calspec = {o: v for (_, o), v in SF.spectra([], list(COMMON.values())).items()}
    rows = []
    for so, co in COMMON.items():
        a = L[L.object == so].set_index(["filter", "curve"]).resid
        b = CL[CL.object == co].set_index(["filter", "curve"]).resid
        for (flt, cv) in a.index.intersection(b.index):
            fw, ft = tot[CURVE_KEY[cv]][flt]
            dsyn = 1e3 * float(R.syn_mag(*calspec[co], fw, ft) - R.syn_mag(*spec[so], fw, ft))
            tot_d = float(b[(flt, cv)] - a[(flt, cv)])
            rows.append(dict(star=so, calspec_name=co, filter=flt, curve=cv, cal_minus_spss=tot_d, spectra=-dsyn,
                             observation=tot_d + dsyn))
    return pd.DataFrame(rows)


def plot_offsets(Sm, C, LO, on, path):
    """P2a: median residual over all SPSS stars +- scatter per filter (>= 3 stars) for manuf·nobeam / manuf·beam (and
    the n_eff 1.5 beam as a dashed reference), plus the series switched on in `on`: mhj·beam (old-generation filters),
    hyp·beam (new-generation and wide filters), ky (0 deg)·beam and the CALSPEC 2024 medians of section 8.1."""
    Sp = Sm[Sm.n >= 3]
    fam = LO.set_index("filter")["lab"]
    allf = Sp[Sp.curve == "beam"].sort_values("lam")
    order = {f: k for k, f in enumerate(allf["filter"])}
    fig, ax = plt.subplots(figsize=(15, 6.2))
    q = Sp[Sp.curve == "beam_neff1.5"].sort_values("lam")
    h_ref = ax.plot(q.lam, q["median"], "--", color=VARIANTS["beam_neff1.5"][1], lw=1.0, label=VARIANTS["beam_neff1.5"][0])[0]
    handles = []
    series = [("nobeam", "nobeam", None, "manuf·nobeam", "0.25", "o", -1.5),
              ("beam", "beam", None, "manuf·beam, per-filter n_eff", "#C25B4E", "o", 0.0),
              ("mhj", "lab_beam", "mhj", "mhj·beam, per-filter n_eff (old-generation filters)", "#7B3FA0", "D", 1.5),
              ("hyp", "lab_beam", "hyp", "hyp·beam, per-filter n_eff (new-generation and wide filters)", "#2E8B7A", "h", 1.5),
              ("ky", "ky_beam", None, "ky (0°)·beam, per-filter n_eff", "#2F5D9E", "^", 3.0)]
    for key, curve, lab_ver, lab, c, mk, dx in series:
        if key in on and not on[key]:
            continue
        q = Sp[Sp.curve == curve].sort_values("lam")
        if lab_ver is not None:
            q = q[q["filter"].map(fam) == lab_ver]
        # the line joins neighbouring filters only
        brk = np.r_[False, np.diff(q["filter"].map(order).to_numpy()) > 1]
        idx = np.where(brk)[0]
        ax.plot(np.insert(q.lam.to_numpy() + dx, idx, np.nan), np.insert(q["median"].to_numpy(), idx, np.nan), "-", color=c, lw=1.0)
        handles.append(ax.errorbar(q.lam + dx, q["median"], q.scatter, fmt=mk, color=c, ms=5.5 if key != "ky" else 7, capsize=0,
                                   mec="white", mew=0.6, label=lab))
    handles.append(h_ref)
    if on["cal"]:
        for key, ck, lab, c, mk in (("nobeam", "nobeam", "manuf·nobeam", "0.25", "o"), ("beam", "beam", "manuf·beam", "#C25B4E", "*"),
                                    ("mhj", "mhj_beam", "mhj·beam", "#7B3FA0", "P")):
            if key == "mhj" and not on["mhj"]:
                continue
            q = C[(C.curve == ck) & (C.n >= 3)].sort_values("lam")
            handles.append(ax.plot(q.lam, q["median"], mk, mfc="none", color=c, ms=9, mew=1.2, label=f"CALSPEC 2024 (8.1), {lab}")[0])
    ax.axhline(0, color="k", lw=0.5); ax.axhspan(-10, 10, color="0.92", zorder=0); ax.set_ylim(-60, 60); ax.grid(alpha=0.3)
    ax.set_ylabel("residual obs − syn [mmag]"); ax.set_xlabel("filter pivot wavelength [nm]")
    ax.legend(handles=handles, fontsize=7.5, loc="upper left", ncol=2)
    sec = ax.secondary_xaxis("top"); sec.set_xticks(allf.lam.to_numpy()); sec.set_xticklabels(allf["filter"], rotation=90, fontsize=6.5)
    ax.set_title("P2a  SPSS: median residual over all stars ± scatter (1.4826 MAD) per filter, filters with ≥ 3 stars; "
                 "open symbols = CALSPEC 2024 (section 8.1); m375w (+145) is above the plot", fontsize=9.5, loc="left")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_hotcool(HC, LO, on, path):
    """P2b: hot (BP-RP < 0.3) - cool (F G K, BP-RP <= 1.3) median difference per filter with bootstrap error, for
    manuf·nobeam / manuf·beam / manuf·beam n_eff 1.5, plus the series switched on in `on`: mhj·beam (old-generation
    filters), hyp·beam (new-generation and wide filters) and ky (0 deg)·beam."""
    fam = LO.set_index("filter")["lab"]
    allf = HC[HC.curve == "beam"].sort_values("lam")
    order = {f: k for k, f in enumerate(allf["filter"])}
    fig, ax = plt.subplots(figsize=(15, 6.2))
    handles = []
    series = [("nobeam", "nobeam", None, "manuf·nobeam", "0.25", "o", -1.8, "-"),
              ("beam", "beam", None, "manuf·beam, per-filter n_eff", "#C25B4E", "o", -0.6, "-"),
              ("n15", "beam_neff1.5", None, "manuf·beam, n_eff 1.5 (previous)", "#7F9CC9", "s", 0.6, "--"),
              ("mhj", "lab_beam", "mhj", "mhj·beam, per-filter n_eff (old-generation filters)", "#7B3FA0", "D", 1.8, "-"),
              ("hyp", "lab_beam", "hyp", "hyp·beam, per-filter n_eff (new-generation and wide filters)", "#2E8B7A", "h", 1.8, "-"),
              ("ky", "ky_beam", None, "ky (0°)·beam, per-filter n_eff", "#2F5D9E", "^", 3.0, "-")]
    for key, curve, lab_ver, lab, c, mk, dx, ls in series:
        if key in on and not on[key]:
            continue
        q = HC[HC.curve == curve].sort_values("lam")
        if lab_ver is not None:
            q = q[q["filter"].map(fam) == lab_ver]
        # the line joins neighbouring filters only
        idx = np.where(np.r_[False, np.diff(q["filter"].map(order).to_numpy()) > 1])[0]
        ax.plot(np.insert(q.lam.to_numpy() + dx, idx, np.nan), np.insert(q.hot_minus_cool.to_numpy(), idx, np.nan), ls, color=c,
                lw=0.9)
        handles.append(ax.errorbar(q.lam + dx, q.hot_minus_cool, q.err, fmt=mk, color=c, ms=5.5 if key != "ky" else 7, capsize=0,
                                   mfc="white" if key == "n15" else c, mec=c if key == "n15" else "white", mew=0.8, label=lab))
    ax.axhline(0, color="k", lw=0.5); ax.axhspan(-5, 5, color="0.92", zorder=0); ax.set_ylim(-40, 30); ax.grid(alpha=0.3)
    ax.set_ylabel("hot − cool [mmag]"); ax.set_xlabel("filter pivot wavelength [nm]")
    ax.legend(handles=handles, fontsize=7.5, loc="lower left", ncol=3)
    sec = ax.secondary_xaxis("top"); sec.set_xticks(allf.lam.to_numpy()); sec.set_xticklabels(allf["filter"], rotation=90, fontsize=6.5)
    ax.set_title("P2b  SPSS: hot stars (BP−RP < 0.3: WD, sdO) − cool stars (F G K, BP−RP ≤ 1.3), median difference with bootstrap "
                 "error, filters with ≥ 3 stars in each group", fontsize=9.5, loc="left")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_levels(HC, CH):
    """P2c: the levels that hardly depend on the curve: the cool-star median, and hot stars CALSPEC 2024 - SPSS 2026."""
    fig, ax = plt.subplots(figsize=(15, 4.6))
    q = HC[HC.curve == "beam"].sort_values("lam")
    ax.plot(q.lam, q.cool, "o-", color="#3C7A5A", ms=5, label="cool SPSS stars, manuf·beam (SPSS spectra vs corrected XP at field colours)")
    q2 = HC[HC.curve == "nobeam"].sort_values("lam")
    ax.plot(q2.lam, q2.cool, "s", mfc="none", color="#3C7A5A", ms=5, label="cool SPSS stars, manuf·nobeam")
    q3 = CH[CH.curve == "beam"].sort_values("lam")
    ax.plot(q3.lam, q3.cal_minus_spss, "*-", color="#B07A1E", ms=7, label="hot stars: CALSPEC 2024 − SPSS 2026, manuf·beam")
    ax.axhline(0, color="k", lw=0.5); ax.axhspan(-10, 10, color="0.92", zorder=0); ax.set_ylim(-40, 40); ax.grid(alpha=0.3)
    ax.set_ylabel("[mmag]"); ax.legend(fontsize=7.5, loc="lower left"); ax.set_xlabel("filter pivot wavelength [nm]")
    sec = ax.secondary_xaxis("top"); sec.set_xticks(q.lam.to_numpy()); sec.set_xticklabels(q["filter"], rotation=90, fontsize=6.5)
    ax.set_title("P2c  SPSS: levels that hardly depend on the curve: cool-star median, and hot-star CALSPEC − SPSS", fontsize=9.5, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "P2c_spss_levels.png", dpi=110); plt.close(fig)


def plot_units(per, pairs):
    filters = sorted(per["filter"].unique(), key=fkey)
    pv = {f: k for k, f in enumerate(filters)}
    fig, axs = plt.subplots(2, 1, figsize=(17, 8.5), sharex=True, gridspec_kw=dict(height_ratios=[1.4, 1]))
    q = per[per.curve == "beam"]
    cmap = plt.get_cmap("tab20"); units = sorted(q.unit.unique())
    for flt, g in q.groupby("filter"):
        g = g.sort_values("unit")
        xs = pv[flt] + np.linspace(-0.18, 0.18, len(g)) if len(g) > 1 else np.array([pv[flt]])
        axs[0].plot(xs, g["median"], "-", color="0.6", lw=0.8, zorder=1)
        for x, r in zip(xs, g.itertuples()):
            y = float(np.clip(r.median, -55, 60))               # m375w (+270) drawn at the edge with its value
            axs[0].errorbar(x, y, r.scatter if y == r.median else 0, fmt="o" if y == r.median else "^",
                            color=cmap(units.index(r.unit) % 20), ms=5, capsize=0, zorder=2)
            lab = f"{r.unit[3:]}\n{r.n}" + ("" if y == r.median else f"\n{r.median:+.0f}")
            axs[0].text(x, y + 3, lab, fontsize=6, ha="center", va="bottom", clip_on=True)
    axs[0].axhline(0, color="k", lw=0.5); axs[0].axhspan(-10, 10, color="0.92", zorder=0); axs[0].set_ylim(-60, 75)
    axs[0].grid(alpha=0.3); axs[0].set_ylabel("residual obs − syn [mmag]")
    axs[0].set_title("(a) per filter copy: median over its stars ± scatter, manuf·beam per-filter n_eff; label = unit number / stars",
                     fontsize=9, loc="left")
    for j, (cv, mk, c) in enumerate([("beam", "o", "#C25B4E"), ("nobeam", "s", "0.3"), ("lab_beam", "D", "#7B3FA0")]):
        t = pairs[pairs.curve == cv]
        axs[1].errorbar(t["filter"].map(pv) + (j - 1) * 0.12, t["diff"], t.diff_err, fmt=mk, color=c, ms=5, capsize=0,
                        label={"beam": "manuf·beam per-filter n_eff", "nobeam": "manuf·nobeam", "lab_beam": "lab·beam per-filter n_eff"}[cv])
    for r in pairs[pairs.curve == "beam"].itertuples():
        axs[1].text(pv[r.filter], min(r.diff + r.diff_err + 2, 38), f"{r.unit_a[3:]}−{r.unit_b[3:]}\n({r.n_common})", fontsize=6,
                    ha="center", clip_on=True)
    axs[1].axhline(0, color="k", lw=0.5); axs[1].axhspan(-10, 10, color="0.92", zorder=0); axs[1].grid(alpha=0.3)
    axs[1].set_ylim(-35, 45)
    axs[1].set_ylabel("copy A − copy B [mmag]"); axs[1].legend(fontsize=8)
    axs[1].set_title("(b) difference between the two copies of a filter, paired over the stars observed with both (median, bootstrap error)",
                     fontsize=9, loc="left")
    axs[1].set_xticks(range(len(filters))); axs[1].set_xticklabels(filters, fontsize=7.5, rotation=45)
    fig.suptitle("P3  SPSS offsets per filter copy (unit), 2026-09", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(OUT / "P3_spss_units.png", dpi=110); plt.close(fig)


def metrics(Sm, filters=None):
    out = {}
    for v in VARIANTS:
        q = Sm[(Sm.curve == v) & (Sm.n >= 3)]
        if filters is not None:
            q = q[q["filter"].isin(filters)]
        out[v] = dict(n_filters=len(q), median_of_medians=q["median"].median(), median_abs=q["median"].abs().median(),
                      rms=np.sqrt(np.mean(q["median"] ** 2)), median_scatter=q.scatter.median())
    return pd.DataFrame(out).T.round(1)


def metrics_hc(HC, filters=None):
    out = {}
    for v in VARIANTS:
        q = HC[HC.curve == v]
        if filters is not None:
            q = q[q["filter"].isin(filters)]
        out[v] = dict(n_filters=len(q), median=q.hot_minus_cool.median(), median_abs=q.hot_minus_cool.abs().median(),
                      rms=np.sqrt(np.mean(q.hot_minus_cool ** 2)), chi2_per_filter=np.mean((q.hot_minus_cool / q.err) ** 2))
    return pd.DataFrame(out).T.round(1)


def main():
    global OUT
    ap = argparse.ArgumentParser(); ap.add_argument("--n-proc", type=int, default=32)
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--filter-id", action="store_true", help="recompute the filter identity check")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
    LO = lab_offsets(); LO.to_csv(OUT / "lab_offsets.csv", index=False)
    fc = OUT / "spss_manuf_frames.csv"
    F = run_frames(a.n_proc) if a.rerun or not fc.exists() else pd.read_csv(fc)
    F["curve"] = np.where(F.version == "lab", "lab_beam", np.where(F.version == "version_ky", "ky_" + F.variant, F.variant))

    fi = OUT / "filter_id_frames.csv"
    I = filter_identity(a.n_proc) if a.filter_id or not fi.exists() else pd.read_csv(fi)
    Fr, U = CM.filter_flags(I, I[["file", "object", "filter", "unit", "mjd"]])
    U["exempt"] = U["filter"].isin(ID_EXEMPT)
    Fr.loc[Fr["filter"].isin(ID_EXEMPT), "flag_filter"] = False
    U.round(3).to_csv(OUT / "filter_id_unitnight.csv", index=False)
    Fr.round(4).to_csv(OUT / "filter_id_frame_flags.csv", index=False)
    pd.set_option("display.width", 220)
    print("unit-nights flagged / doubtful:\n", U[U.flag | (U.share_wrong > 0.2)].round(3).to_string(index=False))
    print(f"labelled filter = best candidate in {np.mean(Fr.best == Fr['filter']):.1%} of {len(Fr)} frames")
    # frames whose labelled curve never gave a ZP (not in F) but were checked
    miss = sorted(set(Fr.file) - set(F.file))
    print(f"frames checked but without a ZP in the labelled curve: {len(miss)}")

    F, info = CM.clean_frames(F, frozenset(Fr[Fr.flag_filter].file))
    # unstable night (2026-10-02, user: drop the night with many deviating frames in Z3): robust scatter of the frame ZP
    # residuals within a night (nobeam, all units and filters) > UNSTABLE_K x the median night and > UNSTABLE_MIN mag
    ns = F[F.curve == "nobeam"].groupby("night").zp_model_res.agg(lambda v: 1.4826 * np.nanmedian(np.abs(v - np.nanmedian(v))))
    bad_n = set(ns[(ns > UNSTABLE_K * ns.median()) & (ns > UNSTABLE_MIN)].index)
    F["flag_unstable"] = F.night.isin(bad_n)
    F["flag_night"] = F.flag_night | F.flag_unstable
    F.loc[F.flag_unstable, "keep"] = False
    print("night ZP scatter [mag]:", ns.round(3).to_dict(), "-> unstable nights dropped:", sorted(bad_n))
    plot_zp_ct(F, LO)
    ct = F[F.keep].groupby(["filter", "curve"]).a_x.agg(lambda v: 1e3 * np.nanmedian(v)).unstack("curve")
    B = F[(F.curve == "nobeam") & (F.std_snr >= SNR_MIN)].assign(
        std_cut=lambda q: (q.std_flags != 0) | (q.std_peak > SS.PEAK_MAX), used=lambda q: q.keep & q.zp_x.notna())
    rem = B.groupby("filter").agg(n_frames=("file", "size"), removed=("keep", lambda k: int((~k).sum())),
                                  night=("flag_night", "sum"), colour=("flag_ct", "sum"), filt=("flag_filter", "sum"),
                                  std_cut=("std_cut", "sum"), used=("used", "sum"))
    Z = info.groupby("filter").agg(units=("unit", lambda u: ",".join(sorted(x[3:] for x in u))), k_air=("k_air", "median"),
                                   frame_sigma=("frame_sigma", "median"), night_scatter=("night_scatter", "median")).join(rem).join(
        ct.rename(columns=lambda c: f"ct_{c}_mmag")).reset_index()
    Z = Z.iloc[np.argsort([fkey(f) for f in Z["filter"]])]
    Z.round(4).to_csv(OUT / "spss_zp_ct_summary.csv", index=False)
    info.to_csv(OUT / "spss_zp_unit_fit.csv", index=False)
    F.drop(columns=[c for c in F.columns if c.startswith("d_") or c.startswith("dr_")]).to_csv(OUT / "spss_frames_flags.csv", index=False)
    print(Z.round(3).to_string(index=False))
    n_all = F[F.curve == "beam"].file.nunique()
    n_snr = F[(F.curve == "beam") & (F.std_snr >= SNR_MIN)].file.nunique()
    F = F[F.keep & (F.std_snr >= SNR_MIN)]
    print(f"SPSS frames {n_all}; std SNR >= 100: {n_snr}; after cleaning {F[F.curve == 'beam'].file.nunique()}")

    bias = SS.expected_bias(F.std_peak_e.to_numpy(), "SPSS") / 1e3
    F["d"] = (F.d_gnd - F.zp + F.zp_x - bias) * 1e3                  # global field ZP without the standard
    F = F[F.d.notna() & (F.d.abs() <= 1e3 * SS.D_MAX)]
    tot = totals()
    spec = {o: v for (_, o), v in SF.spectra(F.object.unique(), []).items()}
    reg = pd.read_csv(REG); reg["k"] = reg["name"].map(SO.AN.norm)
    info_s = {k: (t, c, b, fl) for k, t, c, b, fl in zip(reg.k, reg.sptype, reg.cls, reg.bprp, reg.flag)}
    # what the field colour term predicts for the residual (Delta - pred = residual after applying the colour term)
    bp_std = F.object.map(lambda o: info_s.get(SO.AN.norm(o), (None, None, np.nan, None))[2]).astype(float)
    F["p"] = (F.zp_x - F.zg_c0 - F.a_x * (bp_std - F.bprp0)) * 1e3
    F["dct"] = F.d - F.p

    P = F.groupby(["object", "filter", "curve"]).agg(d=("d", "median"), p=("p", "median"), dct=("dct", "median"), n_frames=("d", "size"), snr=("std_snr", "median"),
                                                       units=("unit", lambda u: ",".join(sorted(set(u))))).reset_index()
    rows = []
    for r in P.itertuples(index=False):
        if r.object not in spec or r.filter not in tot[CURVE_KEY[r.curve]]:
            continue
        w, f = spec[r.object]
        fw, ft = tot[CURVE_KEY[r.curve]][r.filter]
        m_syn = float(R.syn_mag(w, f, fw, ft))
        spt, cls, bp, fl = info_s.get(SO.AN.norm(r.object), ("?", "?", np.nan, np.nan))
        rows.append(dict(object=r.object, sptype=spt, cls=cls, flag=fl, bprp=bp, filter=r.filter, curve=r.curve,
                         lam=round(MB.pivot(fw, ft), 2), m_syn=round(m_syn, 4), m_obs=round(m_syn + r.d / 1e3, 4),
                         resid=round(r.d, 2), pred=round(r.p, 2), resid_ct=round(r.dct, 2), n_frames=r.n_frames, snr=round(r.snr, 1), units=r.units))
    L = pd.DataFrame(rows)
    L["grp"] = np.where(L.bprp < HOT_BPRP, "hot", np.where(L.cls.isin(COOL) & (L.bprp <= COOL_BPRP_MAX), "cool", "other"))
    L.to_csv(OUT / "spss_sed_long.csv", index=False)
    plot_sed(L, spec)

    agg = dict(lam=("lam", "median"), n=("resid", "size"), median=("resid", "median"), scatter=("resid", mad),
               snr=("snr", "median"))
    Sm = L.groupby(["filter", "curve"]).agg(**agg).reset_index()
    for tag, sel in (("hot", L.grp == "hot"), ("cool", L.grp == "cool")):
        s = L[sel].groupby(["filter", "curve"]).resid.agg(["size", "median"]).rename(
            columns={"size": f"n_{tag}", "median": f"median_{tag}"})
        Sm = Sm.merge(s.reset_index(), on=["filter", "curve"], how="left")
    Sm.round(2).to_csv(OUT / "spss_summary.csv", index=False)
    C = pd.read_csv(CAL / "calspec_summary.csv")
    C2 = C.assign(curve=C.curve.replace({"mhj_beam": "lab_beam"}))
    X = Sm.merge(C2[["filter", "curve", "n", "median", "scatter"]], on=["filter", "curve"], how="inner", suffixes=("_spss", "_cal"))
    X["cal_minus_spss"] = X.median_cal - X.median_spss
    X.round(2).to_csv(OUT / "spss_vs_calspec.csv", index=False)
    HC = hot_cool(L)
    HC.round(2).to_csv(OUT / "spss_hot_cool.csv", index=False)
    HS = hc_shift(HC, LO)
    KO = ky_offsets(); KO.to_csv(OUT / "ky_offsets.csv", index=False)
    HS.to_csv(OUT / "spss_hot_cool_shift.csv", index=False)
    # hot stars only: CALSPEC 2024 - SPSS 2026 (same star type, so mostly the dataset / epoch difference)
    CL = pd.read_csv(CAL / "calspec_sed_long.csv")
    CL = CL[CL.bprp < HOT_BPRP].assign(curve=lambda q: q.curve.replace({"mhj_beam": "lab_beam"}))
    ch = CL.groupby(["filter", "curve"]).agg(n_cal=("resid", "size"), cal=("resid", "median"), lam=("lam", "median"))
    sh = L[L.grp == "hot"].groupby(["filter", "curve"]).agg(n_spss=("resid", "size"), spss=("resid", "median"))
    CH = ch.join(sh, how="inner").reset_index()
    CH = CH[(CH.n_cal >= 3) & (CH.n_spss >= 3)].assign(cal_minus_spss=lambda q: q.cal - q.spss)
    CH.round(2).to_csv(OUT / "hot_calspec_vs_spss.csv", index=False)
    plot_levels(HC, CH)
    for bits in range(8):
        on = dict(zip(HC_SERIES, [(bits >> (2 - k)) & 1 == 1 for k in range(3)]))
        tag = "".join("1" if on[k] else "0" for k in HC_SERIES)
        plot_hotcool(HC, LO, on, OUT / f"P2b_spss_hotcool_{tag}.png")
        if tag == HC_DEFAULT:
            plot_hotcool(HC, LO, on, OUT / "P2b_spss_hotcool.png")
    for bits in range(16):
        on = dict(zip(Z4_SERIES, [(bits >> (3 - k)) & 1 == 1 for k in range(4)]))
        tag = "".join("1" if on[k] else "0" for k in Z4_SERIES)
        plot_offsets(Sm, C, LO, on, OUT / f"P2a_spss_offsets_{tag}.png")
        if tag == P2A_DEFAULT:
            plot_offsets(Sm, C, LO, on, OUT / "P2a_spss_offsets.png")
    plot_ky(HC, Sm, HS, KO)
    CT = ct_vs_hotcool(L, HS)
    CT.round(3).to_csv(OUT / "spss_ct_vs_hotcool.csv", index=False)
    print("field colour term vs hot - cool:\n", CT.round(2).to_string(index=False))
    KC = Sm[Sm["filter"].isin(KY_FILTERS)].merge(HC[["filter", "curve", "hot", "cool", "hot_minus_cool", "err"]],
                                                   on=["filter", "curve"], how="left")
    KC.round(2).to_csv(OUT / "ky_compare.csv", index=False)
    print("version_ky comparison:\n", KC[["filter", "curve", "n", "median", "scatter", "hot_minus_cool", "err"]].round(1).to_string(index=False))
    SS2 = same_star(L, CL, spec, tot)
    SS2.round(2).to_csv(OUT / "common_stars_calspec_vs_spss.csv", index=False)

    per, pairs = CM.unit_offsets(F)
    per.round(2).to_csv(OUT / "spss_unit_offsets.csv", index=False)
    pairs.round(2).to_csv(OUT / "spss_unit_pairs.csv", index=False)
    plot_units(per, pairs)

    order = sorted(Sm["filter"].unique(), key=fkey)
    print("lab offsets:\n", LO.to_string(index=False))
    print(Sm.pivot_table(index="filter", columns="curve", values=["median", "scatter", "n"]).round(1).reindex(order).to_string())
    print("hot - cool:\n", HC.pivot_table(index="filter", columns="curve", values=["hot_minus_cool", "err"]).round(1)
          .reindex([f for f in order if f in set(HC["filter"])]).to_string())
    print("cool median:\n", HC.pivot_table(index="filter", columns="curve", values="cool").round(1)
          .reindex([f for f in order if f in set(HC["filter"])]).to_string())
    print("common stars (manuf beam), median over stars:\n", SS2[SS2.curve == "beam"].groupby("filter")[["cal_minus_spss", "spectra",
          "observation"]].agg(["median", "size"]).round(1).reindex([f for f in order if f in set(SS2["filter"])]).to_string())
    print("hot CALSPEC - SPSS:\n", CH.pivot_table(index="filter", columns="curve", values="cal_minus_spss").round(1)
          .reindex([f for f in order if f in set(CH["filter"])]).to_string())
    old = sorted(set(C["filter"]))
    print("all filters:\n", metrics(Sm).to_string())
    print("old-gen 8.1 filters:\n", metrics(Sm, old).to_string())
    print("new-gen / wide:\n", metrics(Sm, [f for f in order if f not in old]).to_string())
    print("curve shift from hot - cool:\n", HS.to_string(index=False))
    print("hot - cool, all:\n", metrics_hc(HC).to_string())
    print("hot - cool, old-gen:\n", metrics_hc(HC, old).to_string())
    print("hot - cool, new-gen / wide:\n", metrics_hc(HC, [f for f in order if f not in old]).to_string())
    print("CALSPEC - SPSS:\n", X.pivot_table(index="filter", columns="curve", values="cal_minus_spss").round(1)
          .reindex([f for f in order if f in old]).to_string())
    print("copy differences (manuf·beam):\n", pairs[pairs.curve == "beam"].drop(columns="stars").round(1).to_string(index=False))
    print("stars:", L.drop_duplicates("object")[["object", "sptype", "flag", "bprp"]].sort_values("bprp").to_string(index=False))
    print("frames used per curve:", F.groupby("curve").size().to_dict())


if __name__ == "__main__":
    main()
