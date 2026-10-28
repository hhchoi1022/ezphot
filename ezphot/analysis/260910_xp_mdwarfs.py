#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260910_xp_mdwarfs.py

M-type stars from the Gaia XP tiles (LAMOST-corrected sampled spectra, 336-1020 nm at 2 nm) and the shape of their
spectra against the 7DT medium bands.

Stages (--stage):
  extract : scan the raw XP tiles used by the standard fields and the survey tiles for sources with BP-RP >= --bprp-min
            and G <= --g-max  -> mstars.npz (flux_cor, flux_error, bp_rp, g, parallax, pm, ruwe, caution, tile)
  plot    : F1 individual spectra by colour bin (normalised at 750-800 nm), F2 median spectra per colour bin with the
            7DT medium bands and the TiO/VO/CaH band heads, F3 dwarf/giant split (absolute G from parallax),
            F4 comparison with the CALSPEC M3.5V standard BD-11 3759 (true resolution vs XP)
  all     : both

  MPLCONFIGDIR=/tmp/ezphot-matplotlib /home/hhchoi1022/anaconda3/envs/py311/bin/python \\
      ezphot/analysis/260910_xp_mdwarfs.py --stage all --n-proc 16
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
from astropy.table import Table     # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
C = importlib.import_module("260910_effcurve_common")
R = C.R
warnings.filterwarnings("ignore")
log = C.log

DEFAULT_OUT = Path("/home/hhchoi1022/ezphot/log/20260910_xp_mdwarfs")
XP_WAVE = R.XP_WAVE_NM
COLOUR_BINS = [(1.8, 2.0, "K7–M0"), (2.0, 2.3, "M0–M2"), (2.3, 2.6, "M2–M3"), (2.6, 3.0, "M3–M4"), (3.0, 4.5, "M4+")]
# molecular band heads and lines of M dwarfs [nm]
FEATURES = [("Na I", 589.4), ("TiO", 615.0), ("TiO", 659.0), ("CaH", 680.0), ("TiO", 705.0), ("TiO", 715.0), ("VO", 740.0), ("TiO", 766.0),
            ("VO", 790.0), ("TiO", 843.0), ("Ca II", 855.0), ("TiO", 886.0)]
_G = {}


def scan_tile(args):
    tile, bprp_min, g_max, cap = args
    path = R.SPECTRA_DIR / f"{tile}.csv"
    out = []
    if not path.exists(): return tile, out, "missing"
    with open(path, "r", errors="ignore") as fh:
        header = fh.readline().lstrip("﻿").rstrip("\n").split(",")
        col = {c: header.index(c) for c in header}
        pre = "gaiaxp_photinfo_replenish_"
        try:
            i_bp, i_rp, i_g = col[pre + "bp_mag"], col[pre + "rp_mag"], col[pre + "g_mag"]
            i_cor, i_err = col["correctedxpspecv1_flux_cor"], col["correctedxpspecv1_flux_error"]
            i_plx, i_pmra, i_pmdec, i_ruwe = col[pre + "parallax"], col[pre + "pmra"], col[pre + "pmdec"], col[pre + "ruwe"]
            i_cau = col.get("correctedxpspecv1_caution", -1); i_ra, i_dec = col[pre + "ra"], col[pre + "dec"]
        except KeyError as e:
            return tile, out, f"missing column {e}"
        for line in fh:
            parts = line.rstrip("\n").split(",")
            try:
                bp, rp, g = float(parts[i_bp]), float(parts[i_rp]), float(parts[i_g])
            except (ValueError, IndexError):
                continue
            if not (np.isfinite(bp) and np.isfinite(rp) and np.isfinite(g)) or bp - rp < bprp_min or g > g_max: continue
            vals = parts[i_cor].split(";")
            if len(vals) != len(XP_WAVE): continue
            err = parts[i_err].split(";"); err = np.asarray(err, float) if len(err) == len(vals) else np.full(len(vals), np.nan)
            def fl(i):
                try: return float(parts[i])
                except (ValueError, IndexError): return np.nan
            out.append(dict(source_id=parts[0], ra=fl(i_ra), dec=fl(i_dec), bp_rp=bp - rp, g=g, parallax=fl(i_plx), pmra=fl(i_pmra), pmdec=fl(i_pmdec), ruwe=fl(i_ruwe),
                            caution=parts[i_cau] if 0 <= i_cau < len(parts) else "", flux=np.asarray(vals, float), flux_err=err))
            if len(out) >= cap: break
    return tile, out, "ok"


def stage_extract(out: Path, n_proc: int, bprp_min: float, g_max: float, cap: int, extra_tiles):
    z1 = np.load(C.PREV_OUT / "shift_recal" / "refstar_spectra.npz", allow_pickle=False); tiles = set(z1["tile"].astype(str))
    p2 = C.DEFAULT_OUT / "tilestar_spectra.npz"
    if p2.exists(): tiles |= set(np.load(p2, allow_pickle=False)["tile"].astype(str))
    tiles |= set(extra_tiles)
    jobs = [(t, bprp_min, g_max, cap) for t in sorted(tiles)]
    log(f"extract: scanning {len(jobs)} tiles for BP-RP >= {bprp_min}, G <= {g_max} (cap {cap} per tile)")
    rows = []
    with Pool(min(n_proc, len(jobs))) as pool:
        for tile, found, status in pool.imap_unordered(scan_tile, jobs):
            log(f"  {tile}: {status}, {len(found)} stars"); rows += [dict(r, tile=tile) for r in found]
    if not rows: log("no stars found"); return
    meta = pd.DataFrame([{k: v for k, v in r.items() if k not in ("flux", "flux_err")} for r in rows])
    meta = meta.drop_duplicates("source_id"); keep = meta.index.to_numpy()
    flux = np.vstack([rows[i]["flux"] for i in keep]); ferr = np.vstack([rows[i]["flux_err"] for i in keep])
    np.savez_compressed(out / "mstars.npz", ids=meta.source_id.to_numpy(str), flux=flux, flux_err=ferr, wave_nm=XP_WAVE,
                        **{c: meta[c].to_numpy() for c in ("ra", "dec", "bp_rp", "g", "parallax", "pmra", "pmdec", "ruwe", "tile")}, caution=meta.caution.to_numpy(str))
    meta.to_csv(out / "mstars.csv", index=False)
    log(f"extract: {len(meta)} unique stars saved; BP-RP quantiles {np.percentile(meta.bp_rp, [5, 50, 95]).round(2)}, G quantiles {np.percentile(meta.g, [5, 50, 95]).round(1)}")


def normalise(flux, wave, lo=750.0, hi=800.0):
    m = (wave >= lo) & (wave <= hi)
    ref = np.nanmedian(flux[:, m], axis=1)
    return flux / ref[:, None]


def stage_plot(out: Path, fig_dir: Path, m450_max: float = np.inf):
    z = np.load(out / "mstars.npz", allow_pickle=True)
    wave = z["wave_nm"]; flux = z["flux"]; bprp = z["bp_rp"]; g = z["g"]; plx = z["parallax"]
    good = np.all(np.isfinite(flux), axis=1) & (np.nanmin(flux[:, (wave > 600)], axis=1) > 0)
    if np.isfinite(m450_max):   # keep only stars brighter than m450_max in the synthetic m450 magnitude (nominal curve)
        m450 = R.syn_mag(wave, flux, *R.load_curve("m450"), 0.0)
        good &= np.isfinite(m450) & (m450 <= m450_max)
    flux, bprp, g, plx = flux[good], bprp[good], g[good], plx[good]
    nf = normalise(flux, wave)
    log(f"plot: {len(nf)} usable spectra" + (f" with m450 <= {m450_max}" if np.isfinite(m450_max) else ""))
    bands = [(f"m{w}", w - 12.5, w + 12.5) for w in range(400, 876, 25)]
    # ---- F1: individual spectra per colour bin
    fig, axes = plt.subplots(len(COLOUR_BINS), 1, figsize=(12, 2.6 * len(COLOUR_BINS)), sharex=True)
    for ax, (lo, hi, lab) in zip(axes, COLOUR_BINS):
        m = (bprp >= lo) & (bprp < hi); idx = np.where(m)[0]
        rng = np.random.default_rng(1); pick = rng.choice(idx, size=min(60, len(idx)), replace=False) if len(idx) else []
        for i in pick: ax.plot(wave, nf[i], color="tab:red", lw=.5, alpha=.25)
        if len(idx): ax.plot(wave, np.nanmedian(nf[idx], axis=0), color="k", lw=1.4, label=f"median of {len(idx)}")
        for name, x in FEATURES: ax.axvline(x, color="tab:blue", lw=.5, alpha=.5)
        ax.set_ylim(0, 1.6); ax.set_ylabel("F / F(750–800)"); ax.set_title(f"BP−RP {lo:.1f}–{hi:.1f} ({lab}): {len(idx)} stars, G ≤ {g.max():.1f}", fontsize=9, loc="left"); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="upper left")
    axes[-1].set_xlabel("wavelength [nm]"); axes[-1].set_xlim(380, 1000)
    fig.suptitle("Gaia XP (LAMOST-corrected) spectra of M-type stars by colour bin; blue lines = TiO/VO/CaH band heads, Na I, Ca II triplet", fontsize=11)
    fig.tight_layout(); fig.savefig(fig_dir / "F1_mstars_individual.png", dpi=120); plt.close(fig)
    # ---- F2: median spectra with the 7DT bands
    fig, ax = plt.subplots(figsize=(13, 6))
    cmap = plt.get_cmap("autumn_r")
    for k, (lo, hi, lab) in enumerate(COLOUR_BINS):
        m = (bprp >= lo) & (bprp < hi)
        if m.sum() < 3: continue
        med = np.nanmedian(nf[m], axis=0); q1, q3 = np.nanpercentile(nf[m], [25, 75], axis=0)
        ax.fill_between(wave, q1, q3, color=cmap((k + 1) / (len(COLOUR_BINS) + 1)), alpha=.15)
        ax.plot(wave, med, color=cmap((k + 1) / (len(COLOUR_BINS) + 1)), lw=1.6, label=f"BP−RP {lo:.1f}–{hi:.1f} ({lab}), n={m.sum()}")
    for j, (f, a, b) in enumerate(bands):
        ax.axvspan(a, b, color="grey" if j % 2 == 0 else "lightgrey", alpha=.18); ax.text((a + b) / 2, 1.52, f, ha="center", fontsize=6, rotation=90, color="0.3")
    for name, x in FEATURES:
        ax.axvline(x, color="tab:blue", lw=.7, alpha=.6); ax.text(x, 0.03, name, rotation=90, fontsize=7, color="tab:blue", ha="center", va="bottom")
    ax.set_xlim(380, 1000); ax.set_ylim(0, 1.6); ax.set_xlabel("wavelength [nm]"); ax.set_ylabel("flux normalised at 750–800 nm"); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Median XP spectra of M-type stars per colour bin (band = interquartile range); grey stripes = 7DT 25 nm medium bands", fontsize=10)
    fig.tight_layout(); fig.savefig(fig_dir / "F2_mstars_median_bands.png", dpi=130); plt.close(fig)
    # ---- F3: dwarfs vs giants (absolute G) in a common colour range
    ok = np.isfinite(plx) & (plx > 1.0)
    MG = g + 5 * np.log10(np.clip(plx, 1e-3, None) / 100.0)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw=dict(width_ratios=[1, 1.6]))
    axes[0].scatter(bprp[ok], MG[ok], s=4, c=bprp[ok], cmap="autumn_r", alpha=.6); axes[0].invert_yaxis(); axes[0].set_xlabel("BP−RP"); axes[0].set_ylabel("M_G (parallax > 1 mas)"); axes[0].grid(alpha=.3)
    axes[0].axhline(6.0, color="k", lw=.7, ls="--"); axes[0].set_title("colour–absolute magnitude: dwarfs below, giants above the dashed line", fontsize=9)
    for sel, lab, col in ((ok & (MG > 6) & (bprp >= 2.0) & (bprp < 2.6), "dwarfs, BP−RP 2.0–2.6", "tab:red"), (ok & (MG < 4) & (bprp >= 2.0) & (bprp < 2.6), "giants, BP−RP 2.0–2.6", "tab:purple")):
        if sel.sum() >= 3: axes[1].plot(wave, np.nanmedian(nf[sel], axis=0), color=col, lw=1.6, label=f"{lab} (n={sel.sum()})")
    for name, x in FEATURES: axes[1].axvline(x, color="tab:blue", lw=.6, alpha=.5)
    axes[1].set_xlim(380, 1000); axes[1].set_ylim(0, 1.6); axes[1].set_xlabel("wavelength [nm]"); axes[1].set_ylabel("flux normalised at 750–800 nm"); axes[1].grid(alpha=.3); axes[1].legend(fontsize=8)
    axes[1].set_title("median spectra: M dwarfs vs M giants of the same colour", fontsize=9)
    fig.tight_layout(); fig.savefig(fig_dir / "F3_dwarfs_vs_giants.png", dpi=130); plt.close(fig)
    # ---- F4: XP vs CALSPEC (BD-11 3759, M3.5V) — resolution
    syn = pd.read_csv(C.PREV_OUT / "calspec_synphot.csv"); spec = syn[syn.object == "BD-11_3759"]["spec"]
    if len(spec):
        t = Table.read(R.CALSPEC_DIR / spec.iloc[0]); wl = np.asarray(t["WAVELENGTH"], float) / 10.0; fl = np.asarray(t["FLUX"], float)
        m = (wl > 380) & (wl < 1000); wl, fl = wl[m], fl[m]; ref = np.median(fl[(wl >= 750) & (wl <= 800)])
        # XP-like smoothing: 10 nm Gaussian
        grid = np.arange(380, 1000, 0.5); f0 = np.interp(grid, wl, fl) / ref; sig = 10 / 2.3548; k = np.exp(-0.5 * (np.arange(-4 * sig, 4 * sig + .5, .5) / sig) ** 2); k /= k.sum()
        sm = np.convolve(f0, k, mode="same")
        sel = (bprp >= 2.3) & (bprp < 2.7)
        fig, ax = plt.subplots(figsize=(13, 5.5))
        ax.plot(wl, fl / ref, color="0.6", lw=.6, label="CALSPEC BD-11 3759 (M3.5V), full resolution")
        ax.plot(grid, sm, color="k", lw=1.4, label="same, smoothed with a 10 nm Gaussian (XP-like resolution)")
        if sel.sum() >= 3: ax.plot(wave, np.nanmedian(nf[sel], axis=0), color="tab:red", lw=1.6, label=f"median XP spectrum, BP−RP 2.3–2.7 (n={sel.sum()})")
        for j, (f, a, b) in enumerate(bands): ax.axvspan(a, b, color="grey" if j % 2 == 0 else "lightgrey", alpha=.18)
        for name, x in FEATURES: ax.axvline(x, color="tab:blue", lw=.6, alpha=.5)
        ax.set_xlim(380, 1000); ax.set_ylim(0, 1.8); ax.set_xlabel("wavelength [nm]"); ax.set_ylabel("flux normalised at 750–800 nm"); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="upper left")
        ax.set_title("What XP keeps of an M-dwarf spectrum: band heads of 2–5 nm are smoothed to ~10 nm, band depths survive", fontsize=10)
        fig.tight_layout(); fig.savefig(fig_dir / "F4_xp_vs_calspec_mdwarf.png", dpi=130); plt.close(fig)
    # ---- summary table per bin: band depths measured in the XP spectra
    rows = []
    for lo, hi, lab in COLOUR_BINS:
        m = (bprp >= lo) & (bprp < hi)
        if m.sum() < 3: continue
        med = np.nanmedian(nf[m], axis=0)
        def at(x): return float(np.interp(x, wave, med))
        rows.append(dict(bin=f"{lo:.1f}-{hi:.1f}", type=lab, n=int(m.sum()), G_median=float(np.median(g[m])), f500=at(500), f600=at(600), f650=at(650), f700=at(700), f850=at(850), f950=at(950),
                         TiO715_depth=1 - at(716) / (0.5 * (at(700) + at(735))), TiO766_depth=1 - at(768) / (0.5 * (at(752) + at(790))), VO790_depth=1 - at(795) / (0.5 * (at(780) + at(815)))))
    S = pd.DataFrame(rows); S.to_csv(fig_dir / "mstars_bins.csv", index=False); print(S.round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all", choices=["extract", "plot", "all"])
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--n-proc", type=int, default=16)
    ap.add_argument("--bprp-min", type=float, default=1.8)
    ap.add_argument("--g-max", type=float, default=17.0)
    ap.add_argument("--cap", type=int, default=3000, help="maximum stars per tile")
    ap.add_argument("--tiles", default="", help="extra tile ids (comma-separated)")
    ap.add_argument("--m450-max", type=float, default=np.inf, help="plot stage: keep only stars with synthetic m450 <= this (figures go to figures_m450le<x>/)")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / ("figures" if not np.isfinite(args.m450_max) else f"figures_m450le{args.m450_max:g}"); fig_dir.mkdir(exist_ok=True)
    extra = [t.strip() for t in args.tiles.split(",") if t.strip()]
    if args.stage in ("extract", "all"): stage_extract(out, args.n_proc, args.bprp_min, args.g_max, args.cap, extra)
    if args.stage in ("plot", "all"): stage_plot(out, fig_dir, args.m450_max)


if __name__ == "__main__":
    main()
