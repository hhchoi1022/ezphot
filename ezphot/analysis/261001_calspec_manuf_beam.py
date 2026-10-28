#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
261001_calspec_manuf_beam.py — CALSPEC standard-star frames calibrated with the manufacturer curve, with and without the
f/3.03 beam, the beam using the per-filter n_eff: measured values (NEFF_MEASURED, 15 filters, 2026-10-07) and, until the
rest are measured, the provisional n_eff(lam0) = 1.962 - 0.147 (lam0 - 480) / 100 (section 6.8; the first
formula 2.2898 - 0.002651 (lam0 - 480) is discarded).

Curves (filter x QE x optics x El Sauce atmosphere, vacuum wavelengths, as 260926_truthcompare_v2.build_totals):
  manufacturer: nobeam, beam (n_eff(lam0), lam0 = centroid of the normal-incidence filter curve), beam_neff1.5 (previous,
  reference only); version_mhj (manufacturer + lab offset): nobeam and beam (n_eff(lam0)) -> curve labels "mhj_nobeam",
  "mhj_beam" (mhj_nobeam added 2026-10-05; display only, it does not enter the frame selection)
Frames: every CALSPEC frame of the re-photometered catalogs (260927_std_rephot.py), processed with
260927_std_offset.process_cal (field ZP from corrected Gaia XP through the same curve, 10" aperture MAG_APER_3).
Outlier frames, per filter and unit (clean_frames; revised 2026-10-01 at the user's request: no frame-ZP and no in-frame
ZP-scatter cut, colour-term cut loosened): field ZP per second (zp - 2.5 log t) fitted as night level + k * airmass
(3-sigma clipped, for the night levels only); a night is dropped if its level is more than 3 sigma (floor 0.03 mag) below
the median night; a frame is dropped if its field colour term (a_x, standard excluded) is beyond CT_NMAD = 10 x
max(1.4826 MAD, CT_FLOOR = 10 mmag/mag) of the filter/unit/curve median.  Night flags come from the nobeam curve; a frame
flagged with any curve is dropped for all curves (same frames); mhj_nobeam (display only) does not take part in the flagging.
Filter identity (filter_identity): every frame's field ZP is recomputed with every candidate curve (manufacturer medium
bands with the n_eff(lam0) beam, broad u g r i z); the true filter gives the smallest field-star ZP scatter.  A frame is
flagged when its labelled curve fails the ZP (scatter > 0.08) or its scatter exceeds the best candidate's by > 25%, and a
whole unit-night when more than half of its frames fail or exceed the best by > 15%.
The standard: FLAGS 0, peak <= 50k ADU, SNR > 10.
Offset per frame: Delta = MAG(std) + global field ZP (all ZP stars of the frame, the standard itself excluded: zp_x) -
syn(CALSPEC spectrum, same curve), minus the faint-star bias (faintcheck_bins.csv); per star / filter / curve the median over
frames.  (Until 2026-10-01 the local ZP within 1500 px was used; that version is kept in 20261001_calspec_manuf_v11_localzp/.)
Outputs (~/ezphot/log/20261001_calspec_manuf/): calspec_manuf_frames.csv, calspec_sed_long.csv, calspec_summary.csv,
  S1_calspec_sed.png (per star: spectrum + calibrated medium-band SED, beam vs nobeam, and residuals),
  S2_calspec_residual.png (median +- scatter over stars per filter),
  Z1_calspec_zp.png (field ZP per second vs airmass model, per filter; outliers), Z2_calspec_colorterm.png (field colour
  term per frame, beam vs nobeam; outliers), calspec_zp_ct_summary.csv, filter_id_frames.csv / filter_id_unitnight.csv
  (filter identity), S3_calspec_units.png + calspec_unit_offsets.csv (offset per filter copy, paired over common stars)
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
SO = importlib.import_module("260927_std_offset")
SS = importlib.import_module("260927_std_offset_summary")
MB = importlib.import_module("260930_std_mb_sed")
SF, R = MB.SF, MB.R
OUT = Path("/home/hhchoi1022/ezphot/log/20261001_calspec_manuf")
NEFF0, NEFF1 = 1.962, -0.00147
VARIANTS = {"nobeam": ("manuf·nobeam", "0.25", "o"), "beam": ("manuf·beam, n_eff(λ0)", "#C25B4E", "o"),
            "mhj_nobeam": ("lab (mhj)·nobeam", "#C48FDB", "d"), "mhj_beam": ("lab (mhj)·beam, n_eff(λ0)", "#7B3FA0", "D"),
            "beam_neff1.5": ("manuf·beam, n_eff 1.5 (previous)", "#7F9CC9", "s")}
CURVE_KEY = {"nobeam": ("manufacturer", "nobeam"), "beam": ("manufacturer", "beam"), "mhj_nobeam": ("version_mhj", "nobeam"),
             "mhj_beam": ("version_mhj", "beam"),
             "beam_neff1.5": ("manufacturer", "beam_neff1.5")}
CT_NMAD, CT_FLOOR = 10.0, 0.010   # colour-term outlier: beyond CT_NMAD x max(1.4826 MAD, CT_FLOOR) [mag/mag]
mad = lambda v: 1.4826 * np.median(np.abs(v - np.median(v)))   # noqa: E731


def n_eff(lam0):
    return NEFF0 + NEFF1 * (lam0 - 480.0)


# measured effective index per filter (lab tilt measurements, provided by the user 2026-10-07; part number, CWL [nm],
# FWHM [nm]).  The linear n_eff(lam0) relation above turned out to be wrong; it is kept ONLY as a provisional value for
# the filters not measured yet (results for the rest expected within 3-4 days).
NEFF_MEASURED = {
    "m386": (1.81, "#33-322", 386.0, 27), "m438": (1.95, "#67-040", 438.0, 28), "m483": (1.97, "#67-042", 482.5, 36),
    "m512": (1.91, "#33-325", 512.0, 30), "m534": (1.97, "#84-115", 534.0, 25), "m561": (1.69, "#86-994", 561.0, 21),
    "m586": (1.94, "#84-117", 586.0, 26), "m615": (1.70, "#87-767", 615.0, 26), "m640": (1.83, "#86-995", 640.0, 20),
    "m661": (1.83, "#87-769", 661.0, 26), "m466w": (1.77, "#86-363", 466.0, 45), "m692w": (1.79, "#67-052", 692.0, 47),
    "m710w": (1.83, "#84-120", 710.0, 47), "m769w": (1.88, "#84-121", 769.0, 48), "m832w": (1.91, "#84-123", 832.0, 45)}


def n_eff_filter(filt, lam0):
    """measured n_eff where available (NEFF_MEASURED), otherwise the provisional n_eff(lam0)."""
    return NEFF_MEASURED[filt][0] if filt in NEFF_MEASURED else n_eff(lam0)


def manuf_totals():
    """filter x QE x optics x atmosphere (as V2.build_totals): manufacturer nobeam / beam n_eff(lam0) / beam 1.5 and
    version_mhj (lab offset) nobeam / beam n_eff(lam0)."""
    TC, DS = SO.TC, SO.DS
    atm = pd.read_csv(TC.TS / "atmosphere_elsauce.csv")
    qe = pd.read_csv(TC.SRC / "QE.csv")
    opt = pd.read_csv(TC.SRC / "optics.efficiency.dr350.dr500.csv", sep=r"\s+", comment="#", skiprows=11)
    out = {k: {} for k in CURVE_KEY.values()}
    for p in sorted((TC.VBASE / "version_mhj").glob("m*.csv")):
        if p.stem in TC.BAD_FILTERS:
            continue
        d = pd.read_csv(p)
        lam = d.lam_nm.to_numpy(float); tr = d.trans.to_numpy(float)
        env = (np.interp(lam, qe.iloc[:, 0], qe.iloc[:, 1]) * np.interp(lam, opt.iloc[:, 0], opt.iloc[:, -1])
               * np.interp(lam, atm["lam_air"], atm["trans"]))
        lam0 = np.trapz(lam * tr, lam) / np.trapz(tr, lam)
        out[("version_mhj", "nobeam")][p.stem] = (DS.air_to_vac_nm(lam), tr * env)
        out[("version_mhj", "beam")][p.stem] = (DS.air_to_vac_nm(lam), TC.dr500_transmission(
            lam, tr, n_eff=n_eff_filter(p.stem, lam0), f_number=3.03, obscuration=0.59) * env)
    for p in sorted((TC.VBASE / "manufacturer").glob("m*.csv")):
        f = p.stem
        if f in TC.BAD_FILTERS:
            continue
        d = pd.read_csv(p)
        lam = d.lam_nm.to_numpy(float); tr = d.trans.to_numpy(float)
        env = (np.interp(lam, qe.iloc[:, 0], qe.iloc[:, 1]) * np.interp(lam, opt.iloc[:, 0], opt.iloc[:, -1])
               * np.interp(lam, atm["lam_air"], atm["trans"]))
        lv = DS.air_to_vac_nm(lam)
        lam0 = np.trapz(lam * tr, lam) / np.trapz(tr, lam)
        out[("manufacturer", "nobeam")][f] = (lv, tr * env)
        out[("manufacturer", "beam")][f] = (lv, TC.dr500_transmission(lam, tr, n_eff=n_eff_filter(f, lam0), f_number=3.03,
                                                                       obscuration=0.59) * env)
        out[("manufacturer", "beam_neff1.5")][f] = (lv, TC.dr500_transmission(lam, tr, n_eff=1.5, f_number=3.03,
                                                                               obscuration=0.59) * env)
    return out


def _init():
    warnings.filterwarnings("ignore")
    SO.AN._init()
    SO.TW._init(40)
    SO._TOTALS = manuf_totals()


def candidate_totals(labels):
    """every manufacturer medium band (n_eff(lam0) beam) and broad u g r i z, each placed under every labelled filter."""
    TC, DS = SO.TC, SO.DS
    atm = pd.read_csv(TC.TS / "atmosphere_elsauce.csv"); qe = pd.read_csv(TC.SRC / "QE.csv")
    opt = pd.read_csv(TC.SRC / "optics.efficiency.dr350.dr500.csv", sep=r"\s+", comment="#", skiprows=11)
    cand = dict(manuf_totals()[("manufacturer", "beam")])
    src = Path("~/ezphot/config/common/transmission/Original_sources").expanduser()
    for b in "ugriz":
        d = pd.read_csv(src / f"{b}.csv"); lam = d.lam.to_numpy(float); tr = np.clip(d.trans.to_numpy(float), 0, None)
        env = (np.interp(lam, qe.iloc[:, 0], qe.iloc[:, 1]) * np.interp(lam, opt.iloc[:, 0], opt.iloc[:, -1])
               * np.interp(lam, atm["lam_air"], atm["trans"]))
        cand[b] = (DS.air_to_vac_nm(lam), tr * env)
    return {(c, "id"): {lf: curve for lf in labels} for c, curve in cand.items()}


def _init_id(labels):
    warnings.filterwarnings("ignore")
    SO.AN._init()
    SO.TW._init(40)
    SO._TOTALS = candidate_totals(labels)


def filter_identity(n_proc):
    M = pd.read_csv(SO.OUT / "rephot_meta.csv")
    M = M[M.status.isin(["ok", "cached"]) & (M.dataset == "CALSPEC")]
    M = M[[(SO.CATD / (Path(f).name[:-5] + "_cat.fits")).exists() for f in M.file]]
    jobs = [(obj, list(zip(g.file, g["filter"], g.unit))) for (obj, _), g in M.groupby(["object", "filter"])]
    rows = []
    with Pool(n_proc, initializer=_init_id, initargs=(sorted(M["filter"].unique()),)) as pool:
        for r in pool.imap_unordered(SO.process_cal, jobs):
            rows += [dict(file=x["file"], object=x["object"], filter=x["filter"], unit=x["unit"], mjd=x["mjd"],
                          cand=x["version"], zsd=x["zsd"], a_x=x["a_x"]) for x in r]
    I = pd.DataFrame(rows)
    I.to_csv(OUT / "filter_id_frames.csv", index=False)
    return I


def filter_flags(I, frames):
    """per frame: labelled-curve scatter vs the best candidate; per unit-night: share of frames flagged."""
    best = I.loc[I.groupby("file").zsd.idxmin()][["file", "cand", "zsd"]].rename(columns={"cand": "best", "zsd": "zsd_best"})
    lab = I[I.cand == I["filter"]][["file", "zsd"]].rename(columns={"zsd": "zsd_label"})
    Fr = frames[["file", "object", "filter", "unit", "mjd"]].drop_duplicates("file").merge(best, on="file", how="left").merge(lab, on="file", how="left")
    Fr["ratio"] = Fr.zsd_label / Fr.zsd_best
    Fr["wrong"] = Fr.zsd_label.isna() | (Fr.ratio > 1.25)
    Fr["doubt"] = Fr.zsd_label.isna() | (Fr.ratio > 1.15)
    Fr["night"] = np.floor(Fr.mjd - 16 / 24).astype(int)
    U = Fr.groupby(["filter", "unit", "night"]).agg(n=("file", "size"), share_wrong=("doubt", "mean"),
                                                   best=("best", lambda v: v.value_counts().index[0] if v.notna().any() else ""),
                                                   ratio_med=("ratio", "median")).reset_index()
    U["flag"] = U.share_wrong > 0.5
    bad = set(map(tuple, U[U.flag][["filter", "unit", "night"]].to_numpy()))
    Fr["flag_filter"] = Fr.wrong | np.array([k in bad for k in zip(Fr["filter"], Fr.unit, Fr.night)], bool)
    return Fr, U


def run_frames(n_proc):
    M = pd.read_csv(SO.OUT / "rephot_meta.csv")
    M = M[M.status.isin(["ok", "cached"]) & (M.dataset == "CALSPEC")]
    M = M[[(SO.CATD / (Path(f).name[:-5] + "_cat.fits")).exists() for f in M.file]]
    jobs = [(obj, list(zip(g.file, g["filter"], g.unit))) for (obj, _), g in M.groupby(["object", "filter"])]
    print(time.strftime("[%H:%M:%S]"), f"CALSPEC frames {len(M)}, jobs {len(jobs)}", flush=True)
    rows = []
    with Pool(n_proc, initializer=_init) as pool:
        for r in pool.imap_unordered(SO.process_cal, jobs):
            rows += r
    F = pd.DataFrame(rows)
    F.to_csv(OUT / "calspec_manuf_frames.csv", index=False)
    print(time.strftime("[%H:%M:%S]"), f"saved {len(F)} rows, {F.file.nunique()} frames", flush=True)
    return F


def robust_night_fit(zpn, X, night):
    """zpn = c_night + k X, 3-sigma clipped; returns k, c per night, residual, sigma."""
    nights = np.unique(night); ni = np.searchsorted(nights, night)
    use_k = np.ptp(X) > 0.1
    keep = np.ones(len(zpn), bool)
    for _ in range(8):
        A = np.vstack([(ni == j).astype(float) for j in range(len(nights))] + ([X] if use_k else [])).T
        sol = np.linalg.lstsq(A[keep], zpn[keep], rcond=None)[0]
        res = zpn - A @ sol
        sig = max(1.4826 * np.median(np.abs(res[keep] - np.median(res[keep]))), 0.01)
        new = np.abs(res) <= 3 * sig
        if (new == keep).all():
            break
        keep = new
    k = sol[-1] if use_k else 0.0
    # night level = median over all frames of that night (a night whose frames were all clipped still gets a real level)
    lev = {n: float(np.median((zpn - k * X)[ni == j])) for j, n in enumerate(nights)}
    res = zpn - (np.array([lev[n] for n in night]) + k * X)
    return k, lev, res, sig


def clean_frames(F, filt_bad=frozenset()):
    F = F.copy()
    F["night"] = np.floor(F.mjd - 16 / 24).astype(int)               # local noon (UT 16 h) splits El Sauce nights
    F["zpn"] = F.zp - 2.5 * np.log10(F.exptime)
    night_flag, info = set(), []
    base = F[F.curve == "nobeam"]
    for (flt, unit), g in base.groupby(["filter", "unit"]):
        k, cn, res, sig = robust_night_fit(g.zpn.to_numpy(), g.airmass.to_numpy(), g.night.to_numpy())
        cv = np.array(list(cn.values()))
        cmed = np.median(cv); csig = max(1.4826 * np.median(np.abs(cv - cmed)), 0.03) if len(cv) >= 3 else 0.03
        bad_n = {n for n, c in cn.items() if c < cmed - 3 * csig}
        night_flag |= set(g.file[g.night.isin(bad_n)])
        F.loc[F.file.isin(g.file) & (F["filter"] == flt), "zp_model_res"] = F.loc[F.file.isin(g.file) & (F["filter"] == flt), "file"].map(
            dict(zip(g.file, g.zpn.to_numpy() - (cmed + k * g.airmass.to_numpy()))))
        info.append(dict(filter=flt, unit=unit, n_frames=len(g), n_nights=len(cn), k_air=round(k, 3), night_scatter=round(csig, 3),
                         frame_sigma=round(sig, 4), bad_nights=len(bad_n)))
    ct_flag = set()
    for (flt, unit, var), g in F[F.curve != "mhj_nobeam"].groupby(["filter", "unit", "curve"]):
        am = g.a_x.median(); asg = max(1.4826 * np.nanmedian(np.abs(g.a_x - am)), CT_FLOOR)
        ct_flag |= set(g.file[(g.a_x - am).abs() > CT_NMAD * asg])
    F["flag_night"] = F.file.isin(night_flag); F["flag_ct"] = F.file.isin(ct_flag)
    F["flag_filter"] = F.file.isin(filt_bad)
    F["keep"] = ~(F.flag_night | F.flag_ct | F.flag_filter) & (F.std_flags == 0) & (F.std_peak <= SS.PEAK_MAX)
    return F, pd.DataFrame(info)


def plot_zp_ct(F, info):
    filters = sorted(F["filter"].unique(), key=lambda f: int(f[1:4]))
    units = sorted(F.unit.unique()); cmap = plt.get_cmap("tab10")
    nc = 5; nr = int(np.ceil(len(filters) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(4.0 * nc, 2.8 * nr), squeeze=False)
    b = F[F.curve == "nobeam"]
    for ax, flt in zip(axs.ravel(), filters):
        g = b[b["filter"] == flt].sort_values("mjd")
        for u, q in g.groupby("unit"):
            ax.plot(q.mjd - 60000, q.zp_model_res, ".", ms=3, color=cmap(units.index(u) % 10), label=u)
        bad = g[~g.keep]
        ax.plot(bad.mjd - 60000, bad.zp_model_res, "x", color="r", ms=4, mew=0.8)
        ax.axhline(0, color="k", lw=0.5); ax.set_ylim(-0.4, 0.25); ax.grid(alpha=0.3); ax.tick_params(labelsize=6.5)
        ax.set_title(f"{flt}: {len(g)} frames, {len(bad)} removed", fontsize=8, loc="left")
        ax.legend(fontsize=5.5, loc="lower left", ncol=2, markerscale=2)
    for ax in axs.ravel()[len(filters):]:
        ax.axis("off")
    fig.supxlabel("MJD − 60000"); fig.supylabel("ZP per second − (median night + k·airmass) [mag]")
    fig.suptitle("Z1  Field ZP per frame (corrected Gaia XP, manufacturer·nobeam) relative to the per-unit airmass model; red x = "
                 "removed (cloudy night, colour-term outlier or wrong filter)", fontsize=10)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.96)); fig.savefig(OUT / "Z1_calspec_zp.png", dpi=105); plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 5.5))
    pv = {f: int(f[1:4]) for f in filters}
    rng = np.random.default_rng(0)
    for j, v in enumerate(["nobeam", "beam", "mhj_nobeam", "mhj_beam"]):
        lab, c, mk = VARIANTS[v]
        g = F[F.curve == v]
        x = g["filter"].map(pv) + (j - 1.5) * 3 + rng.normal(0, 0.5, len(g))
        ax.plot(x[g.keep], 1e3 * g.a_x[g.keep], ".", ms=1.5, color=c, alpha=0.12, zorder=1)
        ax.plot(x[~g.keep], 1e3 * g.a_x[~g.keep], "x", ms=3, color="r", alpha=0.35, zorder=2)
        m = g[g.keep].groupby("filter").a_x.median()
        ax.plot([pv[f] + (j - 1.5) * 3 for f in m.index], 1e3 * m.values, mk + "-", color=c, ms=7, lw=2.2, mec="white", mew=0.8,
                zorder=5, label=f"{lab}: median")
    ax.axhline(0, color="k", lw=0.5); ax.axhspan(-5, 5, color="0.92", zorder=0); ax.set_ylim(-60, 70); ax.grid(alpha=0.3)
    ax.set_xlabel("filter"); ax.set_ylabel("field colour term a [mmag per mag BP−RP]"); ax.legend(fontsize=8)
    ax.set_xticks([pv[f] for f in filters]); ax.set_xticklabels(filters, rotation=45, fontsize=7)
    ax.set_title("Z2  Field colour term per frame (ZP_i vs BP−RP of the field stars, the standard excluded), CALSPEC frames; dots = "
                 "kept frames, red x = removed; grey band ±5 mmag/mag", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "Z2_calspec_colorterm.png", dpi=110); plt.close(fig)


def unit_offsets(F):
    """offset per filter copy (unit): median over stars of the per-star medians, and copy-to-copy differences paired over
    the stars observed with both copies (star selection cancels)."""
    P = F.groupby(["filter", "unit", "curve", "object"]).d.median().reset_index()
    per = P.groupby(["filter", "unit", "curve"]).agg(n=("d", "size"), median=("d", "median"), scatter=("d", mad)).reset_index()
    pairs = []
    for (flt, cv), g in P.groupby(["filter", "curve"]):
        us = sorted(g.unit.unique())
        for i, ua in enumerate(us):
            for ub in us[i + 1:]:
                a = g[g.unit == ua].set_index("object").d; b = g[g.unit == ub].set_index("object").d
                com = a.index.intersection(b.index)
                if len(com) < 2:
                    continue
                dd = (a[com] - b[com]).to_numpy()
                rng = np.random.default_rng(0)
                bs = [np.median(dd[rng.integers(0, len(dd), len(dd))]) for _ in range(500)]
                pairs.append(dict(filter=flt, curve=cv, unit_a=ua, unit_b=ub, n_common=len(com), diff=float(np.median(dd)),
                                  diff_err=float(np.std(bs)), stars=",".join(com)))
    return per, pd.DataFrame(pairs)


def plot_units(per, pairs):
    filters = sorted(per["filter"].unique(), key=lambda f: int(f[1:4]))
    pv = {f: k for k, f in enumerate(filters)}
    fig, axs = plt.subplots(2, 1, figsize=(15, 8.5), sharex=True, gridspec_kw=dict(height_ratios=[1.4, 1]))
    q = per[per.curve == "beam"]
    cmap = plt.get_cmap("tab10"); units = sorted(q.unit.unique())
    for (flt), g in q.groupby("filter"):
        g = g.sort_values("unit")
        xs = pv[flt] + np.linspace(-0.18, 0.18, len(g)) if len(g) > 1 else np.array([pv[flt]])
        axs[0].plot(xs, g["median"], "-", color="0.6", lw=0.8, zorder=1)
        for x, r in zip(xs, g.itertuples()):
            axs[0].errorbar(x, r.median, r.scatter, fmt="o", color=cmap(units.index(r.unit) % 10), ms=5, capsize=0, zorder=2)
            axs[0].text(x, r.median + 3, f"{r.unit[3:]}\n{r.n}", fontsize=6, ha="center", va="bottom")
    axs[0].axhline(0, color="k", lw=0.5); axs[0].axhspan(-10, 10, color="0.92", zorder=0); axs[0].set_ylim(-65, 100)
    axs[0].grid(alpha=0.3); axs[0].set_ylabel("residual obs − syn [mmag]")
    axs[0].set_title("(a) per filter copy: median over its stars ± scatter, manuf·beam n_eff(λ0); label = unit number / stars",
                     fontsize=9, loc="left")
    for j, (cv, mk, c) in enumerate([("beam", "o", "#C25B4E"), ("nobeam", "s", "0.3"), ("mhj_nobeam", "d", "#C48FDB"),
                                     ("mhj_beam", "D", "#7B3FA0")]):
        t = pairs[pairs.curve == cv]
        axs[1].errorbar(t["filter"].map(pv) + (j - 1.5) * 0.1, t["diff"], t.diff_err, fmt=mk, color=c, ms=5, capsize=0,
                        label={"beam": "manuf·beam n_eff(λ0)", "nobeam": "manuf·nobeam", "mhj_nobeam": "mhj·nobeam",
                               "mhj_beam": "mhj·beam n_eff(λ0)"}[cv])
    for r in pairs[pairs.curve == "beam"].itertuples():
        axs[1].text(pv[r.filter], r.diff + r.diff_err + 2, f"{r.unit_a[3:]}−{r.unit_b[3:]}\n({r.n_common})", fontsize=6, ha="center")
    axs[1].axhline(0, color="k", lw=0.5); axs[1].axhspan(-10, 10, color="0.92", zorder=0); axs[1].grid(alpha=0.3)
    axs[1].set_ylim(-30, 52)
    axs[1].set_ylabel("copy A − copy B [mmag]"); axs[1].legend(fontsize=8)
    axs[1].set_title("(b) difference between the two copies of a filter, paired over the stars observed with both (median, bootstrap error)",
                     fontsize=9, loc="left")
    axs[1].set_xticks(range(len(filters))); axs[1].set_xticklabels(filters, fontsize=8)
    fig.suptitle("S3  CALSPEC offsets per filter copy (unit)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(OUT / "S3_calspec_units.png", dpi=110); plt.close(fig)


def main():
    global OUT
    ap = argparse.ArgumentParser(); ap.add_argument("--n-proc", type=int, default=16)
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--filter-id", action="store_true", help="recompute the filter identity check")
    a = ap.parse_args()
    OUT = Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)
    fc = OUT / "calspec_manuf_frames.csv"
    F = run_frames(a.n_proc) if a.rerun or not fc.exists() else pd.read_csv(fc)
    F["curve"] = np.where(F.version == "version_mhj", "mhj_" + F.variant.astype(str), F.variant)

    # filter identity: is the labelled filter the one that fits the field stars best?
    fi = OUT / "filter_id_frames.csv"
    I = filter_identity(a.n_proc) if a.filter_id or not fi.exists() else pd.read_csv(fi)
    Fr, U = filter_flags(I, I[["file", "object", "filter", "unit", "mjd"]])
    U.round(3).to_csv(OUT / "filter_id_unitnight.csv", index=False)
    print("unit-nights flagged as wrong filter:\n", U[U.flag | (U.share_wrong > 0.2)].round(3).to_string(index=False))

    # per-filter ZP and colour term, outlier frames
    F, info = clean_frames(F, frozenset(Fr[Fr.flag_filter].file))
    plot_zp_ct(F, info)
    ct = F[F.keep].groupby(["filter", "curve"]).a_x.agg(lambda v: 1e3 * np.nanmedian(v)).unstack("curve")
    B = F[(F.curve == "nobeam") & (F.std_snr > 10)].assign(std_cut=lambda q: (q.std_flags != 0) | (q.std_peak > SS.PEAK_MAX),
                                                           used=lambda q: q.keep & q.zp_x.notna())
    rem = B.groupby("filter").agg(n_frames=("file", "size"), removed=("keep", lambda k: int((~k).sum())),
                                  night=("flag_night", "sum"), colour=("flag_ct", "sum"), filt=("flag_filter", "sum"),
                                  std_cut=("std_cut", "sum"), used=("used", "sum"))
    Z = info.groupby("filter").agg(k_air=("k_air", "median"), frame_sigma=("frame_sigma", "median"),
                                   night_scatter=("night_scatter", "median")).join(rem).join(
        ct.rename(columns=lambda c: f"ct_{c}_mmag")).reset_index()
    Z = Z.iloc[np.argsort([int(f[1:4]) for f in Z["filter"]])]
    Z.round(4).to_csv(OUT / "calspec_zp_ct_summary.csv", index=False)
    info.to_csv(OUT / "calspec_zp_unit_fit.csv", index=False)
    F.drop(columns=[c for c in F.columns if c.startswith("d_") or c.startswith("dr_")]).to_csv(OUT / "calspec_frames_flags.csv", index=False)
    print(Z.round(3).to_string(index=False))
    n_before = F[(F.curve == "beam") & (F.std_snr > 10)].file.nunique()
    F = F[F.keep & (F.std_snr > 10)]
    print(f"frames with std SNR > 10: {n_before} -> {F[F.curve == 'beam'].file.nunique()} after outlier removal")

    # per-frame offsets
    bias = SS.expected_bias(F.std_peak_e.to_numpy(), "CALSPEC") / 1e3
    F["d"] = (F.d_gnd - F.zp + F.zp_x - bias) * 1e3                  # global field ZP without the standard
    F = F[F.d.notna() & (F.d.abs() <= 1e3 * SS.D_MAX)]
    tot = manuf_totals()
    spec = {o: v for (_, o), v in SF.spectra([], F.object.unique()).items()}
    S = pd.read_csv(SS.D / "std_colours.csv")
    info = {k: (t, c) for d_, k, t, c in zip(S.dataset, S.key, S.sptype, S.cls) if d_ == "CALSPEC"}
    cb = pd.read_csv(Path("/home/hhchoi1022/ezphot/log/20260930_std_ct_applied/cal_bprp_gaia.csv")).set_index("object").bprp_gaia

    P = F.groupby(["object", "filter", "curve"]).agg(d=("d", "median"), n_frames=("d", "size"),
                                                       snr=("std_snr", "median"), units=("unit", lambda u: ",".join(sorted(set(u))))).reset_index()
    rows = []
    for r in P.itertuples(index=False):
        if r.object not in spec or r.filter not in tot[CURVE_KEY[r.curve]]:
            continue
        w, f = spec[r.object]
        fw, ft = tot[CURVE_KEY[r.curve]][r.filter]
        m_syn = float(R.syn_mag(w, f, fw, ft))
        spt, cls = info.get(r.object, ("?", "?"))
        rows.append(dict(object=r.object, sptype=spt, cls=cls, bprp=cb.get(r.object, np.nan), filter=r.filter, curve=r.curve,
                         lam=round(MB.pivot(fw, ft), 2), m_syn=round(m_syn, 4), m_obs=round(m_syn + r.d / 1e3, 4),
                         resid=round(r.d, 2), n_frames=r.n_frames, snr=round(r.snr, 1), units=r.units))
    L = pd.DataFrame(rows)
    L.to_csv(OUT / "calspec_sed_long.csv", index=False)

    # S1: per-star SED and residual, beam (n_eff(lam0)) vs nobeam
    stars = L.drop_duplicates("object").sort_values("bprp").object.tolist()
    nc = 4; nr = int(np.ceil(len(stars) / nc))
    fig = plt.figure(figsize=(5.2 * nc, 6.0 * nr))
    outer = fig.add_gridspec(nr, nc, hspace=0.32, wspace=0.28)
    for i, o in enumerate(stars):
        rr, cc = divmod(i, nc)
        inner = outer[rr, cc].subgridspec(2, 1, height_ratios=[2.2, 1], hspace=0.06)
        ax = fig.add_subplot(inner[0]); axr = fig.add_subplot(inner[1], sharex=ax)
        g = L[L.object == o]
        w, f = spec[o]
        m = (w > 360) & (w < 900)
        ax.plot(w[m], MB.ab_spectrum(w[m], f[m]), "-", color="0.72", lw=0.7, zorder=1, label="CALSPEC spectrum")
        for j, v in enumerate(["nobeam", "beam", "mhj_nobeam", "mhj_beam"]):
            q = g[g.curve == v].sort_values("lam")
            lab, c, mk = VARIANTS[v]
            dx = (j - 1.5) * 1.0
            ax.plot(q.lam + dx, q.m_obs, mk, color=c, ms=3.6, zorder=3, label=f"7DT, {lab}")
            ax.plot(q.lam + dx, q.m_syn, mk, mfc="none", mec=c, ms=5, mew=0.7, zorder=2)
            axr.plot(q.lam + dx, q.resid, mk + "-", color=c, ms=3, lw=0.8, label=lab)
        ax.invert_yaxis(); ax.grid(alpha=0.3); ax.tick_params(labelsize=7, labelbottom=False)
        ax.set_title(f"{o} ({g.sptype.iloc[0]}, BP−RP {g.bprp.iloc[0]:.2f})", fontsize=8, loc="left")
        ax.set_ylabel("AB mag", fontsize=7)
        axr.axhline(0, color="k", lw=0.5); axr.axhspan(-10, 10, color="0.92", zorder=0); axr.grid(alpha=0.3)
        axr.tick_params(labelsize=7)
        gp = g[g.curve.isin(["nobeam", "beam", "mhj_nobeam", "mhj_beam"])]          # the plotted curves only
        lo, hi = (gp.resid.min(), gp.resid.max()) if len(gp) > 3 else (-50, 50)
        pad = max(10, 0.15 * (hi - lo))
        axr.set_ylim(max(-150, lo - pad), min(150, hi + pad))
        axr.set_ylabel("obs − syn [mmag]", fontsize=7); axr.set_xlim(360, 900)
        if i == 0:
            ax.legend(fontsize=6, loc="upper right")
    fig.suptitle("S1  CALSPEC standards calibrated with the manufacturer curve: filled = observed 7DT magnitude (MAG_APER_3 + "
                 "field ZP), open = synthetic photometry of the CALSPEC spectrum through the same curve, grey = spectrum;\n"
                 "red = manufacturer·beam and purple = lab-offset (mhj)·beam, both with n_eff(λ0) = 1.962 − 0.147 (λ0 − 480)/100; "
                 "black = manufacturer, no beam; light purple = lab-offset (mhj), no beam; bottom = residual obs − syn (grey band ±10 mmag)",
                 fontsize=10)
    fig.savefig(OUT / "S1_calspec_sed.png", dpi=105, bbox_inches="tight")
    plt.close(fig)

    # S2: median +- scatter over stars per filter
    Sm = L.groupby(["filter", "curve"]).agg(lam=("lam", "median"), n=("resid", "size"), median=("resid", "median"),
                                              scatter=("resid", mad), snr=("snr", "median")).reset_index()
    Sm.round(2).to_csv(OUT / "calspec_summary.csv", index=False)
    Sp = Sm[Sm.n >= 3]
    fig, axs = plt.subplots(2, 1, figsize=(14, 8.5), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
    ax = axs[0]
    for j, v in enumerate(["nobeam", "beam", "mhj_nobeam", "mhj_beam", "beam_neff1.5"]):
        q = Sp[Sp.curve == v].sort_values("lam")
        lab, c, mk = VARIANTS[v]
        if v == "beam_neff1.5":
            ax.plot(q.lam, q["median"], "--", color=c, lw=1.0, label=lab)
            continue
        ax.errorbar(q.lam + (j - 1.5) * 1.2, q["median"], q.scatter, fmt=mk + "-", color=c, ms=5, lw=1.0, capsize=0, label=lab)
    ax.axhline(0, color="k", lw=0.5); ax.axhspan(-10, 10, color="0.92", zorder=0); ax.set_ylim(-60, 60); ax.grid(alpha=0.3)
    ax.set_ylabel("residual obs − syn [mmag]")
    ax.set_title(f"median over CALSPEC stars ± scatter (1.4826 MAD); filters with ≥ 3 stars ({L.object.nunique()} stars)",
                 fontsize=9, loc="left")
    ax.legend(fontsize=8, loc="upper left")
    W = Sp.pivot_table(index="filter", columns="curve", values=["median", "lam"])
    ax = axs[1]
    ax.bar(W[("lam", "beam")] - 4.5, W[("median", "beam")] - W[("median", "nobeam")], width=3, color="#C25B4E", alpha=0.7,
           label="manuf·beam n_eff(λ0) − manuf·nobeam")
    ax.bar(W[("lam", "beam")] - 1.5, W[("median", "mhj_nobeam")] - W[("median", "nobeam")], width=3, color="#C48FDB", alpha=0.8,
           label="mhj·nobeam − manuf·nobeam (lab offset alone)")
    ax.bar(W[("lam", "beam")] + 1.5, W[("median", "mhj_beam")] - W[("median", "nobeam")], width=3, color="#7B3FA0", alpha=0.6,
           label="mhj·beam n_eff(λ0) − manuf·nobeam")
    ax.plot(W[("lam", "beam")] + 4.5, W[("median", "beam_neff1.5")] - W[("median", "nobeam")], "s", mfc="none", color="#7F9CC9",
            ms=5, label="manuf·beam n_eff 1.5 − manuf·nobeam")
    ax.axhline(0, color="k", lw=0.5); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_xlabel("filter pivot wavelength [nm]"); ax.set_ylabel("curve − manuf·nobeam [mmag]")
    fig.suptitle("S2  CALSPEC residuals per filter: manufacturer and lab-offset (mhj) curves, each with and without the beam",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(OUT / "S2_calspec_residual.png", dpi=115)
    plt.close(fig)

    per, pairs = unit_offsets(F)
    per.round(2).to_csv(OUT / "calspec_unit_offsets.csv", index=False)
    pairs.round(2).to_csv(OUT / "calspec_unit_pairs.csv", index=False)
    plot_units(per, pairs)

    pd.set_option("display.width", 220)
    print("copy differences (manuf·beam):\n", pairs[pairs.curve == "beam"].drop(columns="stars").round(1).to_string(index=False))
    print(Sm.pivot_table(index="filter", columns="curve", values=["median", "scatter", "n"]).round(1)
          .reindex(sorted(Sm["filter"].unique(), key=lambda s: int(s[1:4]))).to_string())
    for v in VARIANTS:
        q = Sp[Sp.curve == v]
        print(f"{v:14s} filters {len(q)}  median |median| {q['median'].abs().median():.1f}  median scatter {q.scatter.median():.1f}"
              f"  median of medians {q['median'].median():+.1f}")
    print("stars:", L.drop_duplicates("object")[["object", "sptype", "bprp"]].to_string(index=False))
    print("frames used per curve:", F.groupby("curve").size().to_dict())


if __name__ == "__main__":
    main()
