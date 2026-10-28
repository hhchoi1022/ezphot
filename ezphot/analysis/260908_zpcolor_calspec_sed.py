#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260908_zpcolor_calspec_sed.py

One figure per CALSPEC standard, on a wavelength axis:
  - the CALSPEC spectrum converted to AB magnitude,
  - the synthetic magnitudes through our nominal transmission curves (calspec_synphot.csv),
  - the calibrated magnitudes measured on the 7DT frames (calspec_frames.csv, median over frames),
  - the same measurements re-calibrated with the shifted-curve reference photometry of the
    closed-loop test (shift_recal/recal_frames.csv.gz, case 'rec' = recommended per-filter shift),
  - the same measurements with the header colour term (g-r) applied as apply_color_terms would
    (colorterm_test/colorterm_stars.csv from 260908_zpcolor_colorterm_test.py, if present).
A lower panel shows observed - synthetic per filter for both calibrations (the shifted case
is compared with the synthetic magnitude through the shifted curve).

Inputs (--out-dir): calspec_synphot.csv, calspec_frames.csv, frames.csv,
                    shift_recal/recal_frames.csv.gz, shift_recal/recal_calspec_syn.csv (optional)
Outputs: figures/sed/SED_<object>.png (13 files), figures/sed/sed_index.csv

  MPLCONFIGDIR=/tmp/ezphot-matplotlib /home/hhchoi1022/anaconda3/envs/py311/bin/python \\
      ezphot/analysis/260908_zpcolor_calspec_sed.py
"""
from __future__ import annotations

import argparse
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

DEFAULT_OUT_DIR = Path("/home/hhchoi1022/ezphot/log/20260908_zp_color_check")
CALSPEC_DIR = Path("/qso/data3/doabdi/intern/synthetic/calspec_data")
STANDARDS = ["BD+02_3375", "BD-11_3759", "FEIGE110", "GD71", "GJ754.1A", "HZ4", "LDS749B", "WD0227+050",
             "WD0308-565", "WD1544-377", "WD1919+145", "WD2039-682", "WD2149+021"]
ZPERR_ANOMALY, CAL_MAX_MAGERR, CAL_MAX_FLUX, MIN_FRAMES = 0.08, 0.05, 45000.0, 3
C_AA = 2.99792458e18          # Angstrom / s
WL_RANGE_NM = (370.0, 920.0)
SHIFT_CASE = "rec"            # case label in shift_recal/recal_frames.csv.gz to use for the re-calibrated points


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def mad(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan


def spectrum_ab(spec_file: Path, bin_nm=0.5):
    """CALSPEC spectrum as AB magnitude vs wavelength [nm], binned for plotting."""
    t = Table.read(spec_file)
    wl = np.asarray(t["WAVELENGTH"], float); fl = np.asarray(t["FLUX"], float)
    m = (wl > WL_RANGE_NM[0] * 10) & (wl < WL_RANGE_NM[1] * 10) & np.isfinite(fl) & (fl > 0)
    wl, fl = wl[m], fl[m]
    fnu = fl * wl ** 2 / C_AA                      # erg s^-1 cm^-2 Hz^-1
    ab = -2.5 * np.log10(fnu) - 48.60
    edges = np.arange(WL_RANGE_NM[0], WL_RANGE_NM[1] + bin_nm, bin_nm)
    idx = np.digitize(wl / 10.0, edges)
    cen, val = [], []
    for i in range(1, len(edges)):
        s = ab[idx == i]
        if len(s): cen.append(0.5 * (edges[i - 1] + edges[i])); val.append(np.median(s))
    return np.array(cen), np.array(val), len(wl)


def observed_table(out_dir: Path):
    """Per (object, filter): median calibrated magnitude with the pipeline ZP and with the shifted-curve ZP."""
    c = pd.read_csv(out_dir / "calspec_frames.csv")
    fr = pd.read_csv(out_dir / "frames.csv", low_memory=False)
    bad = set(fr[fr["ZPERR_APER_3"] > ZPERR_ANOMALY]["file"])
    c = c[(c["status"] == "ok") & ~c["file"].isin(bad)]
    c = c[(c["flags"].fillna(0) <= 1) & (~c["saturated"].astype(bool)) & (c["sep_arcsec"] < 1.5)
          & (c["magerr_APER_3"] < CAL_MAX_MAGERR) & (c["flux_max"] < CAL_MAX_FLUX)].copy()
    rec_path = out_dir / ("lineonly" if SHIFT_CASE == "line" else "shift_recal") / ("recal_frames_line.csv.gz" if SHIFT_CASE == "line" else "recal_frames.csv.gz")
    c["mag_rec"] = np.nan
    if rec_path.exists():
        rf = pd.read_csv(rec_path); rf = rf[rf["case"] == SHIFT_CASE]
        c["mag_rec"] = c["mag_inst_APER_3"] + c["file"].map(dict(zip(rf["file"], rf["zp"])))
    g = c.groupby(["object", "filter"]).agg(n=("mag_obs_APER_3", "size"), mag=("mag_obs_APER_3", "median"),
                                             scatter=("mag_obs_APER_3", mad), sptype=("spectral_type", "first"),
                                             magerr=("magerr_APER_3", "median"),
                                             n_rec=("mag_rec", "count"), mag_rec=("mag_rec", "median"),
                                             scatter_rec=("mag_rec", mad)).reset_index()
    g = g[g["n"] >= MIN_FRAMES]
    # error of the median: frame-to-frame scatter and the photometric error of the measurements, both / sqrt(n);
    # for a single frame (coadds) this is just the photometric error of that measurement
    g["err"] = np.sqrt((g["scatter"] / np.sqrt(g["n"])) ** 2 + (g["magerr"] / np.sqrt(g["n"])) ** 2)
    nr = g["n_rec"].clip(lower=1)
    g["err_rec"] = np.sqrt((g["scatter_rec"].fillna(0) / np.sqrt(nr)) ** 2 + (g["magerr"] / np.sqrt(nr)) ** 2)
    return g


def colorterm_table(out_dir: Path):
    """Per (object, filter) residual after applying the header colour term (g-r); from the colour-term test."""
    p = out_dir / "colorterm_test" / "colorterm_stars.csv"
    if not p.exists(): return pd.DataFrame(columns=["object", "filter", "ct_gr", "ct_gr_err"])
    t = pd.read_csv(p); return t[["object", "filter", "ct_gr", "ct_gr_err"]]


def shifted_synthetic(out_dir: Path):
    p = out_dir / ("lineonly" if SHIFT_CASE == "line" else "shift_recal") / ("recal_calspec_syn_line.csv" if SHIFT_CASE == "line" else "recal_calspec_syn.csv")
    if not p.exists(): return pd.DataFrame(columns=["object", "filter", "shift_nm", "mag_syn_shifted"])
    s = pd.read_csv(p); return s[s["case"] == SHIFT_CASE][["object", "filter", "shift_nm", "mag_syn_shifted"]]


def main():
    global MIN_FRAMES
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--exclude-filters", default="", help="comma-separated filters to drop (e.g. m800,m825 for coadds)")
    ap.add_argument("--shift-case", default="rec", help="case label of the shifted-curve re-calibration: 'rec' (shift_recal/) or 'line' (lineonly/)")
    ap.add_argument("--sed-subdir", default="sed", help="output sub-directory under figures/")
    ap.add_argument("--min-frames", type=int, default=MIN_FRAMES, help="frames per star and filter required (use 1 for coadds)")
    args = ap.parse_args()
    global SHIFT_CASE
    MIN_FRAMES = args.min_frames; SHIFT_CASE = args.shift_case
    out = Path(args.out_dir); fig_dir = out / "figures" / args.sed_subdir; fig_dir.mkdir(parents=True, exist_ok=True)
    syn = pd.read_csv(out / "calspec_synphot.csv")
    syn = syn[syn["filter"].str.match(r"^(m\d{3}w?|[ugriz])$")]  # medium and broad bands (what can be observed)
    excl = {f.strip() for f in args.exclude_filters.split(",") if f.strip()}
    syn = syn[~syn["filter"].isin(excl)]
    obs = observed_table(out)
    ssyn = shifted_synthetic(out)
    ctab = colorterm_table(out)
    index = []
    for o in STANDARDS:
        s = syn[syn["object"] == o]
        if not len(s): log(f"{o}: no synthetic photometry, skipped"); continue
        spec = s["spec"].iloc[0]
        wl, ab, n_pts = spectrum_ab(CALSPEC_DIR / spec)
        ob = obs[obs["object"] == o].set_index("filter")
        s = s.set_index("filter")
        sh = ssyn[ssyn["object"] == o].set_index("filter")
        ctt = ctab[ctab["object"] == o].set_index("filter")
        common = [f for f in s.index if f in ob.index]
        ct_ok = [f for f in common if f in ctt.index and np.isfinite(ctt.loc[f, "ct_gr"])]
        rec_ok = [f for f in common if f in sh.index and np.isfinite(ob.loc[f, "mag_rec"])]
        sptype = ob["sptype"].dropna().iloc[0] if len(ob) and ob["sptype"].notna().any() else ""
        resid = pd.Series({f: ob.loc[f, "mag"] - s.loc[f, "mag_syn"] for f in common})
        resid_rec = pd.Series({f: ob.loc[f, "mag_rec"] - sh.loc[f, "mag_syn_shifted"] for f in rec_ok})
        resid_ct = pd.Series({f: ctt.loc[f, "ct_gr"] for f in ct_ok})                    # (obs + colour term) - syn(nominal)
        mag_ct = {f: s.loc[f, "mag_syn"] + ctt.loc[f, "ct_gr"] for f in ct_ok}

        fig, (ax, ar) = plt.subplots(2, 1, figsize=(11, 7.4), sharex=True, gridspec_kw=dict(height_ratios=[3.2, 1.1], hspace=0.05))
        ax.plot(wl, ab, color="0.35", lw=0.9, label=f"CALSPEC {spec} (AB, 0.5 nm bins)")
        # red diamonds = synthetic magnitude with the shifted curve (nominal curve where no shift is defined);
        # small grey diamonds = nominal-curve synthetic magnitude, the reference of the blue and green residuals
        syn_plot = [sh.loc[f, "mag_syn_shifted"] if (f in sh.index and np.isfinite(sh.loc[f, "mag_syn_shifted"])) else s.loc[f, "mag_syn"] for f in s.index]
        ax.plot(s["wl_pivot_nm"], s["mag_syn"], "D", mfc="none", mec="0.55", mew=0.9, ms=5, label="synthetic magnitude, nominal 7DT curve (reference of the blue and green residuals)")
        ax.plot(s["wl_pivot_nm"], syn_plot, "D", mfc="none", mec="tab:red", mew=1.4, ms=8,
                label=f"synthetic magnitude, shifted 7DT curve ({'line-star shift' if SHIFT_CASE == 'line' else 'recommended shift'}; nominal where no shift)")
        if len(common):
            xs = [s.loc[f, "wl_pivot_nm"] for f in common]
            ax.errorbar(xs, [ob.loc[f, "mag"] for f in common], yerr=[max(ob.loc[f, "err"], 0.001) for f in common], fmt="o", color="tab:blue",
                        ms=5.5, capsize=2.5, zorder=5, label="observed, pipeline calibration (MAGSKY_APER_3, median over frames)")
            for f in common:
                ax.annotate(f"{int(ob.loc[f, 'n'])}", (s.loc[f, "wl_pivot_nm"], ob.loc[f, "mag"]), xytext=(0, -11), textcoords="offset points",
                            fontsize=6, color="tab:blue", ha="center")
        if len(ct_ok):
            ax.errorbar([s.loc[f, "wl_pivot_nm"] - 2.5 for f in ct_ok], [mag_ct[f] for f in ct_ok], yerr=[max(ctt.loc[f, "ct_gr_err"], 0.001) for f in ct_ok],
                        fmt="^", color="tab:green", ms=5.5, capsize=2.5, zorder=5, label="observed, + header colour term (g−r) applied")
        if len(rec_ok):
            xr = [s.loc[f, "wl_pivot_nm"] + 2.5 for f in rec_ok]
            ax.errorbar(xr, [ob.loc[f, "mag_rec"] for f in rec_ok], yerr=[max(ob.loc[f, "err_rec"], 0.001) for f in rec_ok], fmt="s",
                        color="tab:orange", ms=5.5, capsize=2.5, zorder=6, label=f"observed, re-calibrated with shifted curves ({'line-star shift' if SHIFT_CASE == 'line' else 'recommended shift'})")
        lo, hi = np.nanpercentile(ab, [1, 99]); pad = 0.12 * (hi - lo) + 0.05
        ax.set_ylim(hi + pad, lo - pad)            # brighter up
        ax.set_ylabel("AB magnitude"); ax.grid(alpha=.3); ax.legend(fontsize=7.5, loc="lower right")
        ax.set_title(f"{o}   {sptype}   |   {len(common)} filters, {int(ob['n'].sum()) if len(ob) else 0} frames (numbers = frames per filter)", fontsize=10)
        ar.axhspan(-0.02, 0.02, color="grey", alpha=.15); ar.axhline(0, color="k", lw=.6)
        if len(common):
            ar.errorbar([s.loc[f, "wl_pivot_nm"] for f in common], resid.values, yerr=[max(ob.loc[f, "err"], 0.001) for f in common],
                        fmt="o", color="tab:blue", ms=5, capsize=2.5, label="pipeline: obs − syn(nominal curve)")
        if len(ct_ok):
            ar.errorbar([s.loc[f, "wl_pivot_nm"] - 2.5 for f in ct_ok], resid_ct.values, yerr=[max(ctt.loc[f, "ct_gr_err"], 0.001) for f in ct_ok],
                        fmt="^", color="tab:green", ms=5, capsize=2.5, label="+ colour term: obs′ − syn(nominal)")
        if len(rec_ok):
            ar.errorbar([s.loc[f, "wl_pivot_nm"] + 2.5 for f in rec_ok], resid_rec.values, yerr=[max(ob.loc[f, "err_rec"], 0.001) for f in rec_ok],
                        fmt="s", color="tab:orange", ms=5, capsize=2.5, label="shifted: obs′ − syn(shifted curve)")
            for f in rec_ok:
                ar.annotate(f"{sh.loc[f, 'shift_nm']:+.1f}", (s.loc[f, "wl_pivot_nm"] + 2.5, resid_rec[f]), xytext=(0, 7), textcoords="offset points",
                            fontsize=5.5, color="tab:orange", ha="center")
        ar.set_ylim(-0.1, 0.1); ar.set_ylabel("observed − synthetic [mag]"); ar.set_xlabel("wavelength [nm]"); ar.grid(alpha=.3)
        ar.set_xlim(*WL_RANGE_NM); ar.legend(fontsize=6.5, loc="upper right", ncol=3)
        fig.savefig(fig_dir / f"SED_{o}.png", dpi=120, bbox_inches="tight"); plt.close(fig)
        index.append(dict(object=o, spectral_type=sptype, spec=spec, n_filters=len(common), n_frames=int(ob["n"].sum()) if len(ob) else 0,
                          resid_median=float(resid.median()) if len(resid) else np.nan, resid_mad=mad(resid) if len(resid) >= 3 else np.nan,
                          resid_max_abs=float(resid.abs().max()) if len(resid) else np.nan,
                          resid_rec_median=float(resid_rec.median()) if len(resid_rec) else np.nan,
                          resid_rec_mad=mad(resid_rec) if len(resid_rec) >= 3 else np.nan,
                          resid_rec_max_abs=float(resid_rec.abs().max()) if len(resid_rec) else np.nan,
                          resid_ct_median=float(resid_ct.median()) if len(resid_ct) else np.nan,
                          resid_ct_mad=mad(resid_ct) if len(resid_ct) >= 3 else np.nan,
                          resid_ct_max_abs=float(resid_ct.abs().max()) if len(resid_ct) else np.nan, n_spec_points=n_pts))
        log(f"{o}: {len(common)} filters | pipeline MAD {mad(resid) if len(resid) >= 3 else float('nan'):.3f}, max {resid.abs().max() if len(resid) else float('nan'):.3f}"
            f" | colour term MAD {mad(resid_ct) if len(resid_ct) >= 3 else float('nan'):.3f}, max {resid_ct.abs().max() if len(resid_ct) else float('nan'):.3f}"
            f" | shifted MAD {mad(resid_rec) if len(resid_rec) >= 3 else float('nan'):.3f}, max {resid_rec.abs().max() if len(resid_rec) else float('nan'):.3f}")
    pd.DataFrame(index).to_csv(fig_dir / "sed_index.csv", index=False)
    log(f"{len(index)} figures in {fig_dir}")


if __name__ == "__main__":
    main()
