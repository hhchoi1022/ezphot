#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260908_zpcolor_calspec_shift.py

Use the CALSPEC standards (in particular the DA white dwarfs, whose broad Balmer lines make
the synthetic magnitude very sensitive to where a band edge sits) to ask by how much each
medium-band transmission curve would have to be shifted (or widened) to reproduce the
observed star-to-star pattern of  m_obs - m_syn.

Model per filter:  Delta_obs,star = [m_syn(curve perturbed by p) - m_syn(curve)]_star + c
where c is a free constant (the zero point is set by field stars, so a common offset is
not constrained here).  p is either a rigid wavelength shift (nm) or a symmetric width
change (percent).  Best p from a chi^2 scan; the error is the Delta chi^2 = 1 interval.

Inputs (from --out-dir): calspec_frames.csv, calspec_synphot.csv, frames.csv
Outputs: calspec_shift_fit.csv, figures/C4_calspec_curve_shift.png

  MPLCONFIGDIR=/tmp/ezphot-matplotlib /home/hhchoi1022/anaconda3/envs/py311/bin/python \\
      ezphot/analysis/260908_zpcolor_calspec_shift.py
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
TRANSMISSION_DIR = Path("/home/hhchoi1022/ezphot/config/common/transmission")
GRID = [f"m{w}" for w in range(400, 876, 25)]
SHIFTS_NM = np.arange(-6.0, 6.01, 0.5)
WIDTHS_PCT = np.arange(-15.0, 15.01, 1.0)
MIN_FRAMES = 10
MIN_STARS = 4
ERR_FLOOR = 0.005
# same cleaning as 260908_zpcolor_plot.py
ZPERR_ANOMALY, CAL_MAX_MAGERR, CAL_MAX_FLUX = 0.08, 0.05, 45000.0
DA_TYPES = ("DA",)   # spectral types treated as hydrogen-line stars in the plot


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def mad(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan


def load_curve(name):
    d = pd.read_csv(TRANSMISSION_DIR / f"{name}.csv")
    return d.iloc[:, 0].to_numpy(float), np.clip(d.iloc[:, 1].to_numpy(float), 0, None)


def perturbed_filter(name, wave, trans, shift_nm=0.0, width_pct=0.0):
    from pyphot import unit
    from pyphot.phot import Filter
    w = wave.copy()
    if width_pct:
        piv = np.sqrt(np.trapezoid(trans * w, w) / np.trapezoid(trans / w, w))
        w = piv + (1 + width_pct / 100.0) * (w - piv)
    w = w + 10.0 * shift_nm
    return Filter(w * unit["AA"], trans, name=name)


def observed_table(out_dir: Path):
    c = pd.read_csv(out_dir / "calspec_frames.csv")
    fr = pd.read_csv(out_dir / "frames.csv", low_memory=False)
    bad = set(fr[fr["ZPERR_APER_3"] > ZPERR_ANOMALY]["file"])
    c = c[(c["status"] == "ok") & ~c["file"].isin(bad)]
    c = c[(c["flags"].fillna(0) <= 1) & (~c["saturated"].astype(bool)) & (c["sep_arcsec"] < 1.5)
          & (c["magerr_APER_3"] < CAL_MAX_MAGERR) & (c["flux_max"] < CAL_MAX_FLUX)]
    g = c.groupby(["filter", "object"]).agg(n=("delta_APER_3", "size"), delta=("delta_APER_3", "median"),
                                             scatter=("delta_APER_3", mad), sptype=("spectral_type", "first")).reset_index()
    g = g[g["n"] >= MIN_FRAMES]
    g["err"] = np.maximum(g["scatter"] / np.sqrt(g["n"]), ERR_FLOOR)
    return g


def main():
    global MIN_FRAMES, GRID
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--min-frames", type=int, default=MIN_FRAMES, help="frames per star and filter required (use 1 for coadds)")
    ap.add_argument("--filters", default="", help="comma-separated filters (default: the 25 nm grid)")
    args = ap.parse_args()
    MIN_FRAMES = args.min_frames
    if args.filters: GRID = [f.strip() for f in args.filters.split(",") if f.strip()]
    out = Path(args.out_dir)
    from ezphot.dataobjects import Spectrum

    obs = observed_table(out)
    syn = pd.read_csv(out / "calspec_synphot.csv")
    spec_file = dict(zip(syn["object"], syn["spec"]))
    stars = sorted(set(obs["object"]))
    log(f"{len(stars)} standards with >= {MIN_FRAMES} frames in some filter: {stars}")

    # CALSPEC spectra (3000-11000 A) as ezphot Spectrum objects
    spectra = {}
    for s in stars:
        t = Table.read(CALSPEC_DIR / spec_file[s])
        wl = np.asarray(t["WAVELENGTH"], float); fl = np.asarray(t["FLUX"], float)
        m = (wl > 3000) & (wl < 11000) & np.isfinite(fl)
        spectra[s] = Spectrum(wavelength=wl[m], flux=fl[m], wavelength_unit="AA", flux_unit="flamb", verbose=False)

    curves = {f: load_curve(f) for f in GRID}

    def synth(shift=0.0, width=0.0):
        """dict star -> {filter: mag} for all GRID filters under one perturbation."""
        filt = {f: perturbed_filter(f, *curves[f], shift_nm=shift, width_pct=width) for f in GRID}
        res = {}
        for s in stars:
            phot = spectra[s].synphot(filterset=GRID, visualize=False, pyphot_filters=filt)[0]
            res[s] = {f: float(phot[f]["mag"]) for f in GRID}
        return res

    log("synthesising reference (unperturbed) magnitudes")
    base = synth()
    log(f"scanning {len(SHIFTS_NM)} shifts and {len(WIDTHS_PCT)} widths")
    shift_syn = {d: synth(shift=d) for d in SHIFTS_NM}
    width_syn = {w: synth(width=w) for w in WIDTHS_PCT}

    rows, detail = [], {}
    for f in GRID:
        o = obs[obs["filter"] == f].set_index("object")
        use = [s for s in stars if s in o.index]
        if len(use) < MIN_STARS: continue
        y = o.loc[use, "delta"].to_numpy(); e = o.loc[use, "err"].to_numpy(); w = 1 / e ** 2

        def scan(table, params):
            chi = []
            for p in params:
                pred = np.array([table[p][s][f] - base[s][f] for s in use])
                c = np.sum(w * (y - pred)) / np.sum(w)
                chi.append(np.sum(w * (y - pred - c) ** 2))
            chi = np.array(chi); i = int(np.argmin(chi))
            # parabolic refinement + Delta chi2 = 1 error
            if 0 < i < len(params) - 1:
                a, b, cc = np.polyfit(params[i - 1:i + 2], chi[i - 1:i + 2], 2)
                best = -b / (2 * a) if a > 0 else params[i]; err = np.sqrt(1 / a) if a > 0 else np.nan
            else:
                best, err = params[i], np.nan
            return best, err, chi, i

        c0 = np.sum(w * y) / np.sum(w)
        chi2_0 = float(np.sum(w * (y - c0) ** 2))
        s_best, s_err, s_chi, s_i = scan(shift_syn, SHIFTS_NM)
        w_best, w_err, w_chi, w_i = scan(width_syn, WIDTHS_PCT)
        rows.append(dict(filter=f, n_stars=len(use), chi2_const=chi2_0, dof_const=len(use) - 1,
                         best_shift_nm=s_best, shift_err_nm=s_err, chi2_shift=float(s_chi[s_i]),
                         best_width_pct=w_best, width_err_pct=w_err, chi2_width=float(w_chi[w_i]),
                         da_minus_other=float(np.nanmedian(y[[str(o.loc[s, "sptype"]).startswith(DA_TYPES) for s in use]])
                                              - np.nanmedian(y[[not str(o.loc[s, "sptype"]).startswith(DA_TYPES) for s in use]]))
                         if any(str(o.loc[s, "sptype"]).startswith(DA_TYPES) for s in use)
                         and any(not str(o.loc[s, "sptype"]).startswith(DA_TYPES) for s in use) else np.nan))
        pred = np.array([shift_syn[SHIFTS_NM[s_i]][s][f] - base[s][f] for s in use])
        c = np.sum(w * (y - pred)) / np.sum(w)
        detail[f] = dict(stars=use, y=y, e=e, pred=pred + c, c0=c0, sptype=[str(o.loc[s, "sptype"]) for s in use],
                         shift=SHIFTS_NM[s_i], chi=s_chi)
    R = pd.DataFrame(rows)
    R.to_csv(out / "calspec_shift_fit.csv", index=False)
    print(R.round(3).to_string(index=False))

    # ---- figure
    piv = {f: float(np.sqrt(np.trapezoid(curves[f][1] * curves[f][0], curves[f][0]) / np.trapezoid(curves[f][1] / curves[f][0], curves[f][0])) / 10) for f in GRID}
    fig = plt.figure(figsize=(16, 14)); gs = fig.add_gridspec(4, 5, height_ratios=[1.2, 1, 1, 1])
    ax = fig.add_subplot(gs[0, :3])
    ax.errorbar([piv[f] for f in R["filter"]], R["best_shift_nm"], yerr=R["shift_err_nm"].fillna(0), fmt="o", color="tab:red", capsize=3)
    for _, r in R.iterrows():
        ax.annotate(f"{r['filter']}\nΔχ²={r['chi2_const'] - r['chi2_shift']:.0f}", (piv[r["filter"]], r["best_shift_nm"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="k", lw=.6); ax.set_ylabel("best rigid shift of the curve [nm]"); ax.set_xlabel("pivot wavelength [nm]"); ax.grid(alpha=.3)
    ax.set_title("Curve shift that best reproduces the CALSPEC star-to-star residual pattern (Δχ² vs constant model)", fontsize=9)
    ax2 = fig.add_subplot(gs[0, 3:])
    ax2.errorbar([piv[f] for f in R["filter"]], R["best_width_pct"], yerr=R["width_err_pct"].fillna(0), fmt="s", color="tab:purple", capsize=3)
    ax2.axhline(0, color="k", lw=.6); ax2.set_ylabel("best symmetric width change [%]"); ax2.set_xlabel("pivot wavelength [nm]"); ax2.grid(alpha=.3)
    ax2.set_title("Same, with a width change instead of a shift", fontsize=9)
    for k, f in enumerate([f for f in GRID if f in detail][:15]):
        d = detail[f]; a = fig.add_subplot(gs[1 + k // 5, k % 5])
        x = np.arange(len(d["stars"]))
        col = ["tab:blue" if t.startswith(DA_TYPES) else "tab:gray" for t in d["sptype"]]
        a.errorbar(x, d["y"], yerr=d["e"], fmt="o", color="k", ms=3, capsize=2, zorder=3)
        a.scatter(x, d["y"], c=col, s=30, zorder=4)
        a.plot(x, d["pred"], "r_", ms=12, mew=2, label=f"shift {d['shift']:+.1f} nm")
        a.axhline(d["c0"], color="grey", lw=.7, ls="--")
        a.set_xticks(x); a.set_xticklabels([s.replace("WD", "WD\n") if False else s for s in d["stars"]], rotation=90, fontsize=6)
        a.set_title(f, fontsize=9); a.set_ylim(-0.1, 0.1); a.grid(alpha=.3); a.legend(fontsize=6, loc="upper right")
        if k % 5 == 0: a.set_ylabel("m_obs - m_syn [mag]", fontsize=8)
    fig.suptitle("CALSPEC residual pattern per star (blue = DA white dwarf, grey = other) and the prediction of the best-shifted curve (red)", fontsize=10)
    fig.tight_layout(); (out / "figures").mkdir(exist_ok=True)
    fig.savefig(out / "figures" / "C4_calspec_curve_shift.png", dpi=130); plt.close(fig)
    log("calspec_shift_fit.csv and figures/C4_calspec_curve_shift.png written")


if __name__ == "__main__":
    main()
