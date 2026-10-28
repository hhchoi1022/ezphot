#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260910_transmission_sources_plot.py

Plot the raw transmission-curve source materials in ~/ezphot/config/common/transmission/Original_sources
(vendor filter curves, camera QE, optics efficiency, SkyCalc atmosphere) as they are, before any combination:
F1 medium-band filters, F2 broad bands, F3 QE, F4 optics, F5 atmosphere, F6 overview, plus inventory.csv.
Sources are read-only; outputs go to the log directory.

  MPLCONFIGDIR=/tmp/ezphot-matplotlib /home/hhchoi1022/miniconda3/envs/too/bin/python \\
      ezphot/analysis/260910_transmission_sources_plot.py [--out ~/ezphot/log/20260910_transmission_sources]
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402
import pandas as pd                 # noqa: E402
from astropy.io import fits         # noqa: E402

SRC = Path("~/ezphot/config/common/transmission/Original_sources").expanduser()


def log(msg):
    print(time.strftime("[%Y-%m-%d %H:%M:%S] ") + msg, flush=True)


def load_curve(path):
    d = pd.read_csv(path)
    lam, tr = d.iloc[:, 0].to_numpy(float), d.iloc[:, 1].to_numpy(float)
    order = np.argsort(lam)                      # m466w is stored descending
    return lam[order], tr[order]


def pivot_wavelength(lam, tr):
    return np.sqrt(np.trapz(tr * lam, lam) / np.trapz(tr / lam, lam))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(Path("~/ezphot/log/20260910_transmission_sources").expanduser()))
    a = ap.parse_args()
    out = Path(a.out); fig_dir = out / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)

    # ---- load everything -------------------------------------------------
    filter_files = sorted(p for p in SRC.glob("*.csv") if re.fullmatch(r"(m\d{3}w?|[ugriz])", p.stem))
    medium = [p for p in filter_files if p.stem.startswith("m")]
    broad = [p for p in filter_files if not p.stem.startswith("m")]
    curves, inv = {}, []
    for p in filter_files + [SRC / "QE.csv"]:
        lam, tr = load_curve(p)
        curves[p.stem] = (lam, tr)
        inv.append(dict(name=p.stem, kind="filter" if p in filter_files else "camera QE",
                        n=len(lam), lam_min=lam.min(), lam_max=lam.max(),
                        step_med=np.median(np.diff(lam)), peak=tr.max(),
                        pivot_nm=pivot_wavelength(lam, tr) if p in filter_files else np.nan))

    opt = pd.read_csv(SRC / "optics.efficiency.dr350.dr500.csv", sep=r"\s+", comment="#", skiprows=11)
    opt.columns = ["nm", "H-K9LA", "H-LaK4L", "H-FK95N", "VIS", "Al coating", "Total_pct", "total"]
    inv.append(dict(name="optics.efficiency.dr350.dr500", kind="optics", n=len(opt),
                    lam_min=opt.nm.min(), lam_max=opt.nm.max(), step_med=np.median(np.diff(opt.nm)),
                    peak=opt.total.max(), pivot_nm=np.nan))

    with fits.open(SRC / "skytable.fits") as hdul:
        sky = pd.DataFrame({c: np.asarray(hdul[1].data[c], dtype="f8") for c in
                            ["lam", "trans", "trans_ma", "trans_o3", "trans_rs", "trans_ms"]})
        # SkyCalc writes its input parameters as KEY=VALUE lines inside the COMMENT block
        sky_cards = [tuple(line.split("=", 1)) for line in str(hdul[1].header.get("COMMENT", "")).splitlines()
                     if "=" in line]
    inv.append(dict(name="skytable", kind="atmosphere", n=len(sky), lam_min=sky.lam.min(),
                    lam_max=sky.lam.max(), step_med=np.median(np.diff(sky.lam)),
                    peak=sky.trans.max(), pivot_nm=np.nan))
    inv = pd.DataFrame(inv)
    inv.to_csv(out / "inventory.csv", index=False)
    log("inventory:\n" + inv.round(3).to_string(index=False))
    log("skytable SkyCalc input parameters:")
    for k, v in sky_cards:
        print(f"    {k} = {v}", flush=True)

    def sky_param(*subs):
        for k, v in sky_cards:
            if any(s in k.upper() for s in subs):
                return f"{k}={v}"
        return None

    # ---- F1 medium-band filters -----------------------------------------
    piv = {p.stem: pivot_wavelength(*curves[p.stem]) for p in medium}
    order = sorted(piv, key=piv.get)
    cmap = plt.get_cmap("turbo")
    fig, ax = plt.subplots(figsize=(16, 6))
    for name in order:
        lam, tr = curves[name]
        c = cmap((piv[name] - 370) / (890 - 370))
        wide = name.endswith("w")
        ax.plot(lam, tr, color=c, lw=1.4 if wide else 1.0, ls="--" if wide else "-")
        k = np.argmax(tr)
        ax.annotate(name, (lam[k], tr[k] + 0.01), color=c, fontsize=6, rotation=90,
                    ha="center", va="bottom")
    ax.set(xlim=(340, 960), ylim=(0, 1.12), xlabel="wavelength [nm]", ylabel="transmission",
           title=f"F1  vendor filter curves — {len(medium)} medium bands (dashed = wide)")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "F1_filters_medium.png", dpi=150); plt.close(fig)

    # ---- F2 broad bands ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))
    bcol = dict(u="#7b2fbe", g="#2ca02c", r="#d62728", i="#8c564b", z="#444444")
    for name in ["u", "g", "r", "i", "z"]:
        lam, tr = curves[name]
        ax.plot(lam, tr, color=bcol[name], lw=1.2, label=f"{name} (pivot {pivot_wavelength(lam, tr):.0f} nm)")
    ax.set(xlim=(300, 1150), ylim=(0, 1.05), xlabel="wavelength [nm]", ylabel="transmission",
           title="F2  vendor broad-band filter curves")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "F2_filters_broad.png", dpi=150); plt.close(fig)

    # ---- F3 camera QE -----------------------------------------------------
    lam, qe = curves["QE"]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(lam, qe, "o", ms=4, color="#1f77b4", label=f"QE.csv data points (n={len(lam)}, ~10 nm step)")
    ax.plot(lam, qe, "-", lw=0.8, color="#1f77b4", alpha=0.6, label="linear interpolation")
    ax.axvspan(340, 960, color="orange", alpha=0.06, label="7DT medium-band range")
    ax.set(xlim=(250, 1050), ylim=(0, 1.0), xlabel="wavelength [nm]", ylabel="QE",
           title=f"F3  camera QE (IMX455) — peak {qe.max():.3f} at {lam[np.argmax(qe)]:.0f} nm")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "F3_qe.png", dpi=150); plt.close(fig)

    # ---- F4 optics ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))
    for col, c in [("H-K9LA", "#999"), ("H-LaK4L", "#bbb"), ("H-FK95N", "#777"),
                   ("VIS", "#2ca02c"), ("Al coating", "#d62728")]:
        ax.plot(opt.nm, opt[col], "o--", ms=4, lw=0.8, color=c, label=col)
    ax.plot(opt.nm, opt.total, "s-", ms=6, lw=2, color="k", label="total")
    for r in opt.itertuples():
        ax.annotate(f"{r.total:.2f}", (r.nm, r.total - 0.05), ha="center", fontsize=8)
    ax.set(xlim=(360, 1040), ylim=(0.3, 1.05), xlabel="wavelength [nm]", ylabel="efficiency",
           title="F4  optics efficiency (dr350/dr500) — only 7 points on a 100 nm grid")
    ax.legend(ncol=3); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "F4_optics.png", dpi=150); plt.close(fig)

    # ---- F5 atmosphere ------------------------------------------------------
    s = sky[(sky.lam >= 300) & (sky.lam <= 1050)]
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(s.lam, s.trans, lw=0.3, color="k", label="total transmission")
    axes[0].set(ylim=(0, 1.05), ylabel="transmission", title="F5  SkyCalc atmosphere — total")
    for col, c, lab in [("trans_rs", "#1f77b4", "Rayleigh"), ("trans_o3", "#9467bd", "ozone"),
                        ("trans_ms", "#8c564b", "aerosol (Mie)"), ("trans_ma", "#d62728", "molecular abs.")]:
        axes[1].plot(s.lam, s[col], lw=0.5 if col == "trans_ma" else 1.2, color=c, label=lab)
    axes[1].set(ylim=(0, 1.05), xlabel="wavelength [nm]", ylabel="transmission", title="components")
    params = [p for p in (sky_param("AIRMASS"), sky_param("PWV"), sky_param("SEASON"),
                          sky_param("VACAIR"), sky_param("SITE.HEIGHT"), sky_param("RESOLUTION")) if p]
    for ax_ in axes:
        ax_.legend(loc="lower right"); ax_.grid(alpha=0.3)
    if params:
        axes[0].text(0.01, 0.05, " | ".join(params), transform=axes[0].transAxes, fontsize=8,
                     bbox=dict(fc="white", alpha=0.7, ec="none"))
    fig.tight_layout(); fig.savefig(fig_dir / "F5_atmosphere.png", dpi=150); plt.close(fig)

    # ---- F6 overview ---------------------------------------------------------
    grid = np.arange(350, 1000.1, 0.5)
    qe_i = np.interp(grid, *curves["QE"])
    opt_i = np.interp(grid, opt.nm, opt.total)
    n_med = max(1, int(2.0 / np.median(np.diff(s.lam))))          # ~2 nm display smoothing
    atm_sm = np.interp(grid, s.lam, s.trans.rolling(n_med, center=True, min_periods=1).median())
    fig, ax = plt.subplots(figsize=(16, 6))
    for name in order:
        lam, tr = curves[name]
        ax.plot(lam, tr, color=cmap((piv[name] - 370) / (890 - 370)), lw=0.5, alpha=0.35)
    ax.plot(grid, qe_i, color="#1f77b4", lw=2, label="camera QE (interp.)")
    ax.plot(grid, opt_i, color="k", lw=2, ls="--", label="optics total (interp. from 7 pts)")
    ax.plot(grid, atm_sm, color="#2ca02c", lw=1.5, label="atmosphere (2 nm median)")
    ax.plot(grid, qe_i * opt_i * atm_sm, color="#d62728", lw=2.5,
            label="QE x optics x atmosphere (= S candidate)")
    ax.set(xlim=(340, 1000), ylim=(0, 1.05), xlabel="wavelength [nm]", ylabel="transmission / efficiency",
           title="F6  overview — system components over the vendor filter curves")
    ax.legend(loc="upper right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "F6_overview.png", dpi=150); plt.close(fig)

    log(f"done -> {fig_dir} (6 figures), {out / 'inventory.csv'}")


if __name__ == "__main__":
    main()
