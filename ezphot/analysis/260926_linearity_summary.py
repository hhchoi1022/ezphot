#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260926_linearity_summary.py — pooled summary of 260926_linearity_test.py outputs (all units of one run).

  (a) low-signal response vs EXPECTED star peak above sky [e-], short exposures, pooled over units
      (reference: 2024 medium-band XP residual vs peak: +51/+37/+23/+10/0 mmag at 3-8/8-16/16-32/32-64/>64 e-)
  (b) bright end: response vs expected total level at the star peak (sky + star) [ADU], pooled over units
  (c) effective exposure offset dt per unit (frame scale s_f t = a t + b -> dt = b / a)
  (d) blank 10" residual after the old (SExtractor mode) and new (mean) backgrounds, in e-/pix, vs sky level
  (e) aperture dependence (5/10/15") of the low-signal response
Usage: python 260926_linearity_summary.py <tag>      -> L2_summary.png, lin_summary_*.csv in the run directory
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTROOT = Path("/home/hhchoi1022/ezphot/log/20260926_linearity")
REF2024 = [(3, 8, 51), (8, 16, 37), (16, 32, 23), (32, 64, 10), (64, 300, 0)]


def boot_med(v, n=300):
    """median of the flux RATIO (negative ratios kept: dropping them biases low-SNR bins bright), in mmag."""
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 10:
        return np.nan, np.nan
    rng = np.random.default_rng(0)
    b = np.array([np.median(rng.choice(v, len(v))) for _ in range(n)])
    med = np.median(v)
    if med <= 0:
        return np.nan, np.nan
    return float(mmag(med)), float(np.std(mmag(np.clip(b, 1e-6, None))))


def mmag(rho):
    return -2.5e3 * np.log10(rho)


if __name__ == "__main__":
    tag = sys.argv[1]
    d = OUTROOT / tag
    A = pd.read_csv(d / "lin_stars.csv"); F = pd.read_csv(d / "lin_frames.csv")
    A = A[np.isfinite(A.rho) & np.isfinite(A.peak_e)]
    for c in ("rho", "rho5", "rho15", "rho_old"):
        if c in A.columns:
            A["m_" + c] = A[c]                     # ratios; boot_med converts the median to mmag
    pd.set_option("display.width", 220)
    # (a) low-signal, pooled over units, per exposure
    pbins = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    rows = []
    for t, g in A.groupby("exptime"):
        for lo, hi in zip(pbins[:-1], pbins[1:]):
            m = (g.peak_e >= lo) & (g.peak_e < hi)
            med, err = boot_med(g.loc[m, "m_rho"])
            r = dict(exptime=t, peak_lo=lo, peak_hi=hi, n=int(m.sum()), mmag=med, err=err,
                     sky_e=float((g.sky_adu * g.egain).median()))
            for ap in ("rho5", "rho15"):
                if "m_" + ap in g.columns:
                    r["mmag_" + ap] = boot_med(g.loc[m, "m_" + ap])[0]
            rows.append(r)
    L = pd.DataFrame(rows); L.to_csv(d / "lin_summary_lowsignal.csv", index=False)
    print("(a) low-signal response pooled over units [mmag, >0 = measured fainter]; rows = expected peak [e-]:")
    print(L.pivot_table(index="peak_lo", columns="exptime", values="mmag").round(1).to_string())
    print("   errors:")
    print(L.pivot_table(index="peak_lo", columns="exptime", values="err").round(1).to_string())
    print("   sky [e-/pix] per exposure:", L.groupby("exptime").sky_e.first().round(1).to_dict())
    # per-unit spread at low peak for the shortest exposures
    U = []
    for (u, t), g in A[A.exptime <= 5].groupby(["unit", "exptime"]):
        for lo, hi in [(4, 16), (16, 64), (64, 256)]:
            m = (g.peak_e >= lo) & (g.peak_e < hi)
            U.append(dict(unit=u, exptime=t, peak=f"{lo}-{hi}", mmag=boot_med(g.loc[m, "m_rho"])[0], n=int(m.sum())))
    U = pd.DataFrame(U)
    print("\n   per unit, exposures <= 5 s pooled over exposure (median of unit medians):")
    print(U.groupby(["unit", "peak"]).mmag.median().unstack().round(1).to_string())
    # (b) bright end vs total level, exposures >= 10 s pooled
    tb = [2000, 4000, 8000, 12000, 16000, 24000, 32000, 48000, 60000, 90000]
    rows = []
    for lo, hi in zip(tb[:-1], tb[1:]):
        m = (A.tot_exp_adu >= lo) & (A.tot_exp_adu < hi) & (A.exptime >= 10)
        med, err = boot_med(A.loc[m, "m_rho"])
        rows.append(dict(tot_lo=lo, tot_hi=hi, n=int(m.sum()), mmag=med, err=err))
    Bt = pd.DataFrame(rows); Bt.to_csv(d / "lin_summary_brightend.csv", index=False)
    print("\n(b) bright end, exposures >= 10 s pooled, vs expected total level at the peak [ADU]:")
    print(Bt.round(1).to_string(index=False))
    # (c) shutter offset per unit
    rows = []
    for u, g in A.groupby("unit"):
        sf = g.groupby(["name", "exptime"]).s_f.first().reset_index()
        p = np.polyfit(sf.exptime, sf.s_f * sf.exptime, 1)
        rows.append(dict(unit=u, slope=p[0], dt=p[1] / p[0]))
    Dt = pd.DataFrame(rows); Dt.to_csv(d / "lin_summary_shutter.csv", index=False)
    print("\n(c) effective exposure offset dt [s] per unit:", Dt.set_index("unit").dt.round(3).to_dict(),
          "| median %.3f s" % Dt.dt.median())
    # (d) blank residuals in e-
    F["sky_e"] = F.sky_med * F.egain
    F["blank_new_e"] = F.blank_new * F.egain; F["blank_old_e"] = F.blank_old * F.egain
    Bk = F.groupby("exptime")[["sky_e", "blank_old_e", "blank_new_e"]].median()
    print("\n(d) blank 10\" residual [e-/pix] (median over units):"); print(Bk.round(3).to_string())
    # figure
    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9})
    fig, ax = plt.subplots(1, 4, figsize=(21, 4.8))
    cmap = plt.get_cmap("viridis"); ts = sorted(L.exptime.unique())
    for i, t in enumerate(ts):
        r = L[(L.exptime == t) & (L.n >= 30)]
        pk = np.sqrt(r.peak_lo * r.peak_hi)
        ax[0].errorbar(pk, r.mmag, r.err, marker="o", ms=3, lw=1, capsize=0, color=cmap(i / max(len(ts) - 1, 1)),
                       label=f"{t:g} s (sky {r.sky_e.iloc[0]:.0f} e⁻)" if len(r) else None)
    rx = [np.sqrt(a * b) for a, b, _ in REF2024]; ry = [c for _, _, c in REF2024]
    ax[0].plot(rx, ry, "k--", lw=1.2, label="2024 medium band (XP-based)")
    ax[0].axhline(0, color="k", lw=0.6); ax[0].set_xscale("log"); ax[0].set_ylim(-40, 80); ax[0].grid(alpha=0.3)
    ax[0].set(xlabel="expected star peak above sky [e⁻]", ylabel="flux deficit [mmag] (>0: too faint)",
              title="(a) low-signal response, all units pooled")
    ax[0].legend(fontsize=6.5, ncol=2)
    ax[1].errorbar(np.sqrt(Bt.tot_lo * Bt.tot_hi), Bt.mmag, Bt.err, marker="o", color="tab:red")
    ax[1].axhline(0, color="k", lw=0.6); ax[1].set_xscale("log"); ax[1].grid(alpha=0.3)
    ax[1].set(xlabel="expected total level at the star peak (sky + star) [ADU]", ylabel="flux deficit [mmag]",
              title="(b) bright end (exposures ≥ 10 s)")
    ax[2].bar(Dt.unit.str[3:], Dt.dt, color="tab:purple"); ax[2].axhline(0, color="k", lw=0.6)
    ax[2].set(xlabel="unit", ylabel="effective exposure offset dt [s]", title="(c) shutter: t_eff = t + dt")
    ax[3].plot(Bk.sky_e, Bk.blank_old_e, "o-", color="tab:orange", label="SExtractor mode (current)")
    ax[3].plot(Bk.sky_e, Bk.blank_new_e, "o-", color="tab:blue", label="per-mesh mean 5σ (new)")
    ax[3].axhline(0.42, color="0.5", ls=":", lw=1, label="Poisson mode bias 0.42 e⁻")
    ax[3].axhline(0, color="k", lw=0.6); ax[3].set_xscale("log"); ax[3].grid(alpha=0.3); ax[3].legend(fontsize=7)
    ax[3].set(xlabel="sky [e⁻/pix]", ylabel="blank 10″ residual [e⁻/pix]", title="(d) background residual vs sky level")
    fig.suptitle(f"L2  Linearity series {tag}: g band, WD2039-682, 15 units, 1–200 s × 5 (forced 10″, new background)", fontsize=10)
    fig.tight_layout(); fig.savefig(d / "L2_summary.png", dpi=125)
    print("saved", d / "L2_summary.png")
