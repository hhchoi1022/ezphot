#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260927_std_offset_summary.py — summaries and figures for 260927_std_offset.py (CALSPEC / SPSS magnitude offsets on the
re-photometered catalogs).

Offsets use the local ZP (PFX = "dl": ZP stars within 1500 px of the standard, >= 5 stars), because the standards sit
at the image centre and the flat-field leaves an epoch-dependent radial ZP pattern (see 260927_std_offset.py).
Screens (every summary):
  - field colour term within 0.1 mag/mag of its (dataset, filter, version, variant) median (wrong-filter frames)
  - standard FLAGS == 0 and FLUX_MAX <= 50000 ADU; per-frame |Delta| <= 0.5 mag before the star medians (broken truth spectrum)
CALSPEC figure O1: standards SNR > 10 (user choice 2026-09-27; field ZP stars still SNR >= 100), bootstrap errors on every
point, expected low-signal bias per filter from the field-star check (std_offset_summary_cal10.csv: bias_exp).
Standard SNR cuts: 'snr100' (SNR >= 100, adopted) and 'snr50' (SNR >= 50, reference for red filters; field-star check
260927_std_faintcheck.py shows stars at SNR 50-100 measured ~3 mmag too faint on these catalogs).
O3 = field-star residual vs SNR and vs peak [e-] from 260927_std_faintcheck.py (faintcheck_stars.csv).
Outputs (~/ezphot/log/20260927_std_newbkg/): std_offset_summary_<cut>.csv,
O1_std_offset_calspec.png, O2_std_offset_spss.png, O3_std_offset_snr.png, O4_std_offset_position.png
"""
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

D = Path("/home/hhchoi1022/ezphot/log/20260927_std_newbkg")
FIGD = Path("/home/hhchoi1022/ezphot/log/20260921_curve_versions")
TRU = [("gnd", "#1f4e8c"), ("cor", "#C25B4E"), ("raw", "#E09A3E")]
VARS = ["nobeam", "beam"]
A_TOL, PEAK_MAX, D_MAX, NMIN = 0.1, 50000.0, 0.5, 5
CUTS = {"snr100": 100.0, "snr50": 50.0}
KEY = ["dataset", "filter", "version", "variant"]
PFX = "dl"   # offsets with the local ZP (field stars within 1500 px of the standard); "d" = global ZP, "dr" = radial fit


def mad(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return 1.4826 * np.median(np.abs(v - np.median(v))) if len(v) else np.nan


def screened(F):
    med = F.groupby(KEY).a_field.transform("median")
    return F[((F.a_field - med).abs() <= A_TOL) & (F.std_flags == 0) & (F.std_peak <= PEAK_MAX)]


def summarise(F):
    res = []
    for k, g in F.groupby(KEY):
        row = dict(zip(KEY, k), pivot=int(k[1][1:4]), n_frames=len(g), n_stars=g.object.nunique())
        for tag, _ in TRU:
            c = f"{PFX}_{tag}"
            if c not in g.columns:
                continue
            gg = g.dropna(subset=[c])
            gg = gg[gg[c].abs() <= D_MAX]
            per = gg.groupby("object")[c].median()
            if len(per) < 2:
                continue
            row[f"{tag}_mmag"] = round(1e3 * per.median(), 1)
            row[f"{tag}_err"] = round(1e3 * 1.2533 * mad(per) / np.sqrt(len(per)), 1)
            row[f"{tag}_n"] = len(per)
        res.append(row)
    return pd.DataFrame(res).sort_values(["dataset", "pivot", "version", "variant"])


def expected_bias(peak_e, ds, sfx=""):
    """low-signal bias [mmag, + = measured too faint -> Delta too large] from the field-star check (faintcheck_bins.csv,
    residual vs peak above sky), interpolated in log(peak); sfx="_auto" for the MAG_AUTO check."""
    B = pd.read_csv(D / f"faintcheck_bins{sfx}.csv")
    B = B[(B.dataset == ds) & (B.kind == "peak_e")].sort_values("peak_e_med")
    return -np.interp(np.log10(np.clip(peak_e, 1, None)), np.log10(B.peak_e_med), B.res)


def summarise_boot(F, nboot=500, seed=0):
    """filter value = median over stars of the per-star median Delta; error = std of that statistic when both the stars
    and, within each star, the frames are resampled (works for 1-4 stars, where the star-to-star MAD is undefined)."""
    rng = np.random.default_rng(seed)
    res = []
    for k, g in F.groupby(KEY):
        row = dict(zip(KEY, k), pivot=int(k[1][1:4]), n_frames=len(g), snr_med=round(g.std_snr.median(), 1),
                   peak_e_med=round(g.std_peak_e.median(), 1),
                   bias_exp=round(float(np.median(expected_bias(g.std_peak_e.to_numpy(), k[0]))), 1))
        for tag, _ in TRU:
            c = f"{PFX}_{tag}"
            if c not in g.columns:
                continue
            gg = g.dropna(subset=[c])
            gg = gg[gg[c].abs() <= D_MAX]
            if gg.empty:
                continue
            groups = [1e3 * v.to_numpy() for _, v in gg.groupby("object")[c]]
            per = np.array([np.median(v) for v in groups])
            boots = np.empty(nboot)
            for b in range(nboot):
                pick = rng.integers(0, len(groups), len(groups))
                boots[b] = np.median([np.median(groups[i][rng.integers(0, len(groups[i]), len(groups[i]))]) for i in pick])
            row[f"{tag}_mmag"] = round(float(np.median(per)), 1)
            row[f"{tag}_err"] = round(float(np.std(boots)), 1)
            row[f"{tag}_n"] = len(per)
        res.append(row)
    return pd.DataFrame(res).sort_values(["dataset", "pivot", "version", "variant"])


def grid_figure(S, S0, ds, gnd_label, fname, code, cut_label, ylim, all_err=False):
    vers = [v for v in ["manufacturer", "version_mhj", "version_hyp", "version_ky"] if ((S.dataset == ds) & (S.version == v)).any()]
    fig, axes = plt.subplots(len(vers), 2, figsize=(16.5, 4.4 * len(vers)), sharex=True, sharey=True)
    axes = np.atleast_2d(axes).reshape(len(vers), 2)
    for i, ver in enumerate(vers):
        for j, var in enumerate(VARS):
            ax = axes[i, j]
            T = S[(S.dataset == ds) & (S.version == ver) & (S.variant == var)]
            for t_i, (tag, col) in enumerate(TRU):
                lab = {"gnd": gnd_label, "cor": "corrected XP", "raw": "raw XP"}[tag]
                g = T.dropna(subset=[f"{tag}_mmag"]).sort_values("pivot")
                x = g["pivot"] + (t_i - 1) * 2.0
                ax.plot(x, g[f"{tag}_mmag"], "-", color=col, lw=1.2, alpha=0.7)
                if all_err:
                    ax.errorbar(x, g[f"{tag}_mmag"], yerr=g[f"{tag}_err"], ls="none", marker="o", ms=4.5, capsize=2,
                                color=col, label=lab)
                else:
                    ok = g[f"{tag}_n"] >= NMIN
                    ax.errorbar(x[ok], g[f"{tag}_mmag"][ok], yerr=g[f"{tag}_err"][ok], ls="none", marker="o", ms=4.5,
                                color=col, label=lab)
                    ax.plot(x[~ok], g[f"{tag}_mmag"][~ok], "o", ms=5, mfc="none", color=col, alpha=0.8)
            if all_err:
                g = T.dropna(subset=["gnd_mmag"]).sort_values("pivot")
                ax.plot(g["pivot"], g.bias_exp, "--", color="0.45", lw=1.0,
                        label="expected low-signal bias (field-star check, standards' peak e⁻)")
                for _, r in g.iterrows():
                    ax.text(r["pivot"], ylim[0] + 0.04 * (ylim[1] - ylim[0]), f"{int(r.gnd_n)}★\n{r.snr_med:.0f}",
                            ha="center", va="bottom", fontsize=6.5, color="0.35")
            ax.axhline(0, color="k", lw=0.7); ax.axhspan(-10, 10, color="0.93", zorder=0); ax.grid(alpha=0.3)
            ax.set_title(f"{ver} · {var}", fontsize=11.5)
            if j == 0:
                ax.set_ylabel("m_cal − syn_truth [mmag]")
            if i == 0 and j == 0:
                ax.legend(fontsize=9)
    axes[0, 0].set_ylim(*ylim)
    for j in range(2):
        axes[-1, j].set_xlabel("pivot wavelength [nm]")
    note = ("error bars = bootstrap over stars and frames; bottom labels = standards (★) / median standard SNR"
            if all_err else f"open = fewer than {NMIN} standards")
    fig.suptitle(f"{code}  {ds} magnitude offsets — curve version × f/3 beam (new background; ZP from SNR ≥ 100 field stars within 1500 px of the standard; vacuum)\n"
                 f"(standards {cut_label}, FLAGS 0, peak ≤ 50k ADU; {note}; grey band = ±10 mmag)", fontsize=12)
    fig.tight_layout()
    fig.savefig(D / fname, dpi=130)
    shutil.copy2(D / fname, FIGD / fname)
    print("saved", fname)


if __name__ == "__main__":
    adopt = sys.argv[1] if len(sys.argv) > 1 else "snr100"
    F = pd.read_csv(D / "std_offset_frames.csv")
    Fs = screened(F)
    print(f"rows {len(F)} -> screened {len(Fs)}; frames {F.file.nunique()} -> {Fs.file.nunique()}")
    for name, thr in CUTS.items():
        S = summarise(Fs[Fs.std_snr >= thr]); S.to_csv(D / f"std_offset_summary_{name}.csv", index=False)
    # CALSPEC: standards SNR > 10 (field ZP stars stay at SNR >= 100), bootstrap errors on every point
    C10 = summarise_boot(Fs[(Fs.dataset == "CALSPEC") & (Fs.std_snr > 10)])
    C10.to_csv(D / "std_offset_summary_cal10.csv", index=False)
    grid_figure(C10, None, "CALSPEC", "CALSPEC spectrum", "O1_std_offset_calspec.png", "O1", "SNR > 10", (-80, 130), all_err=True)
    S = pd.read_csv(D / f"std_offset_summary_{adopt}.csv")
    lab = f"SNR ≥ {CUTS[adopt]:.0f}"
    grid_figure(S, None, "SPSS", "SPSS flux table", "O2_std_offset_spss.png", "O2", lab, (-60, 60))
    # O3: field-star residual vs SNR / peak (260927_std_faintcheck.py)
    A = pd.read_csv(D / "faintcheck_stars.csv")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ds, col in (("CALSPEC", "tab:blue"), ("SPSS", "tab:orange")):
        g = A[A.dataset == ds]
        for k, (x, edges) in enumerate((("snr", np.geomspace(15, 1000, 14)), ("peak_e", np.geomspace(8, 20000, 14)))):
            xc, med, err = [], [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                v = g[(g[x] >= lo) & (g[x] < hi)].res
                if len(v) >= 50:
                    xc.append(np.sqrt(lo * hi)); med.append(v.median()); err.append(1.2533 * mad(v) / np.sqrt(len(v)))
            ax[k].errorbar(xc, med, err, marker="o", ms=5, capsize=0, color=col,
                           label=f"{ds} ({'2024' if ds == 'CALSPEC' else '2026'} frames, {g.file.nunique()} frames)")
    for k, lab in enumerate(("field-star SNR (10\" aperture)", "field-star peak above sky [e⁻]")):
        ax[k].axhline(0, color="k", lw=0.7); ax[k].axhspan(-3, 3, color="0.92", zorder=0)
        ax[k].set_xscale("log"); ax[k].grid(alpha=0.3); ax[k].set_ylim(-45, 15); ax[k].set_xlabel(lab); ax[k].legend(fontsize=8)
    ax[0].axvline(100, color="0.4", ls=":", lw=0.8); ax[0].axvline(50, color="0.4", ls=":", lw=0.8)
    ax[0].set_ylabel("ZP_i − median ZP(SNR ≥ 100) [mmag]  (− = measured too faint)")
    fig.suptitle("O3  Faint stars on the re-photometered standard-star frames (new background; corrected XP, manufacturer, no beam)", fontsize=11)
    fig.tight_layout(); fig.savefig(D / "O3_std_offset_snr.png", dpi=130); shutil.copy2(D / "O3_std_offset_snr.png", FIGD / "O3_std_offset_snr.png")
    print("saved O3")
    # O4: radial ZP pattern (260927_std_radial_check.py) and CALSPEC - SPSS per filter, global vs local ZP
    Rc = pd.read_csv(D / "radial_check.csv")
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.8))
    edges = np.array([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000])
    for ds, col in (("CALSPEC", "tab:blue"), ("SPSS", "tab:orange")):
        g = Rc[Rc.dataset == ds]
        xc, med, err = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            v = g[(g.rad >= lo) & (g.rad < hi)].res
            if len(v) >= 30:
                xc.append((lo + hi) / 2); med.append(v.median()); err.append(1.2533 * mad(v) / np.sqrt(len(v)))
        ax[0].errorbar(xc, med, err, marker="o", ms=5, capsize=0, color=col, label=f"{ds} ({'2024' if ds == 'CALSPEC' else '2026'}, {g.file.nunique()} frames)")
    ax[0].axvspan(0, 1500, color="0.93", zorder=0, label="local-ZP radius around the standard")
    ax[0].axhline(0, color="k", lw=0.7); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
    ax[0].set(xlabel="distance from image centre [px] (standards: median 112–149 px)", ylabel="field-star ZP − frame median [mmag]",
              title="(a) radial ZP pattern of SNR ≥ 100 field stars")
    common = ["m425", "m450", "m475", "m500", "m525", "m550", "m575", "m600", "m625", "m650"]
    G = Fs[(Fs.std_snr >= np.where(Fs.dataset == "CALSPEC", 10.0001, 100)) & (Fs.version == "manufacturer") & (Fs.variant == "beam")
           & Fs["filter"].isin(common)]
    for pre, mk, lab in (("d", "s", "frame-median ZP"), ("dl", "o", "local ZP (≤ 1500 px)")):
        per = G.groupby(["dataset", "filter", "object"])[f"{pre}_gnd"].median().reset_index()
        fm = per.groupby(["dataset", "filter"])[f"{pre}_gnd"].median().unstack("dataset") * 1e3
        x = [int(f[1:]) for f in fm.index]
        ax[1].plot(np.array(x) + (-2 if pre == "d" else 2), fm.CALSPEC - fm.SPSS, mk + "-", ms=6, label=lab)
    ax[1].axhline(0, color="k", lw=0.7); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
    ax[1].set(xlabel="filter [nm]", ylabel="CALSPEC − SPSS offset [mmag]", title="(b) CALSPEC (SNR > 10) − SPSS (SNR ≥ 100), manufacturer·beam, ground")
    fig.suptitle("O4  Standards at the image centre vs field-median ZP", fontsize=11)
    fig.tight_layout(); fig.savefig(D / "O4_std_offset_position.png", dpi=130); shutil.copy2(D / "O4_std_offset_position.png", FIGD / "O4_std_offset_position.png")
    print("saved O4")

