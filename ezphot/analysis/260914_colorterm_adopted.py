#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260914_colorterm_adopted.py

필터별 대표(채택) 색항 정의 + F37v3 스타일 그림 (F43).
- m400: 2026-01-01(교체) 이후 데이터만 / 나머지 필터: 2024-05+ 전체
- outlier 제거: 프레임 오차컷(err<3x중앙) 후 3σ-클리핑 평균 (astropy sigma_clipped_stats)
- 각 패널에 채택 평균선(주황)과 값 표기; 극단 클러스터는 축 분할 스트립; 밤 중앙값 선은 >=5프레임 밤만
- 산출: colorterm_adopted.csv (filter, a_mean, a_err, a_median, n, epoch)

  MPLCONFIGDIR=/tmp/ezphot-matplotlib /home/hhchoi1022/miniconda3/envs/too/bin/python -u \\
      ezphot/analysis/260914_colorterm_adopted.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402
import pandas as pd                 # noqa: E402
from astropy.stats import sigma_clipped_stats   # noqa: E402

warnings.filterwarnings("ignore")
plt.rcParams["axes.unicode_minus"] = False
FOLDER = Path("/home/hhchoi1022/ezphot/log/20260911_offset_timeseries")
MJD_2025, MJD_2026 = 60676.0, 61041.0
XLIM = (60465, 61150)
MIN_NIGHT = 5


def breakmarks(ax_top, ax_bot):
    d = 0.018
    kw = dict(color="k", clip_on=False, lw=0.9)
    ax_top.plot((-d, +d), (-d, +d), transform=ax_top.transAxes, **kw)
    ax_top.plot((1 - d, 1 + d), (-d, +d), transform=ax_top.transAxes, **kw)
    ax_bot.plot((-d, +d), (1 - d, 1 + d), transform=ax_bot.transAxes, **kw)
    ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_bot.transAxes, **kw)


def main():
    r = pd.read_csv(FOLDER / "offset_timeseries_ezall.csv")
    r = r[np.isfinite(r["a_phys"]) & np.isfinite(r["dstar_err"])]
    thr = r.groupby("filter")["dstar_err"].transform("median") * 3.0
    r = r[r["dstar_err"] < thr]

    order = sorted(r["filter"].unique(), key=lambda f: int(f[1:4]))
    units_all = sorted(r["unit"].unique())
    cmap = plt.get_cmap("tab20"); ucol = {un: cmap(i % 20) for i, un in enumerate(units_all)}

    adopted = []
    fig = plt.figure(figsize=(22, 15))
    outer = fig.add_gridspec(5, 4, hspace=0.55, wspace=0.27, top=0.955, bottom=0.06, left=0.045, right=0.99)
    for k, filt in enumerate(order):
        s = r[r["filter"] == filt].sort_values("mjd")
        # 채택 epoch: m400은 2026+, 나머지는 전체
        sel = s[s["mjd"] >= MJD_2026] if filt == "m400" else s
        mean_c, med_c, std_c = sigma_clipped_stats(sel["a_phys"].to_numpy(), sigma=3.0, maxiters=10)
        n_eff = int((np.abs(sel["a_phys"] - med_c) < 3 * std_c).sum())
        aerr = std_c / np.sqrt(max(n_eff, 1))
        adopted.append(dict(filter=filt, pivot=int(filt[1:4]), a_mean=round(float(mean_c), 4),
                            a_err=round(float(aerr), 4), a_median=round(float(med_c), 4),
                            a_std=round(float(std_c), 4), n=n_eff,
                            epoch=("2026+" if filt == "m400" else "2024+")))

        med = s["a_phys"].median()
        mad = 1.4826 * np.median(np.abs(s["a_phys"] - med))
        lo_b, hi_b = med - 7 * max(mad, 1e-3), med + 7 * max(mad, 1e-3)
        bulk = s[(s["a_phys"] >= lo_b) & (s["a_phys"] <= hi_b)]
        up = s[s["a_phys"] > hi_b]; dn = s[s["a_phys"] < lo_b]
        need_up = len(up) >= 15; need_dn = len(dn) >= 15
        heights = ([0.30] if need_up else []) + [1.0] + ([0.30] if need_dn else [])
        inner = outer[k // 4, k % 4].subgridspec(len(heights), 1, height_ratios=heights, hspace=0.09)
        axs = [fig.add_subplot(inner[i]) for i in range(len(heights))]
        axm = axs[1 if need_up else 0]

        blo, bhi = np.nanpercentile(bulk["a_phys"], [0.5, 99.5])
        pad = 0.10 * (bhi - blo)
        for un in sorted(bulk["unit"].unique()):
            su = bulk[bulk["unit"] == un]
            axm.scatter(su["mjd"], su["a_phys"], s=5, color=ucol[un], alpha=0.3, rasterized=True)
        ng = bulk.groupby("night").agg(mjd=("mjd", "median"), a=("a_phys", "median"), n=("a_phys", "size"))
        ng = ng[ng["n"] >= MIN_NIGHT].sort_values("mjd")
        axm.plot(ng["mjd"], ng["a"], "k-", lw=0.9, alpha=0.85)
        axm.axhline(0, color="grey", lw=0.6, ls="--", alpha=0.6)
        axm.axvline(MJD_2026, color="k", lw=0.6, ls=":", alpha=0.6)
        # 채택 평균선 (m400은 2026+ 구간에만 그림)
        x0 = MJD_2026 if filt == "m400" else XLIM[0]
        axm.plot([x0, XLIM[1]], [mean_c, mean_c], color="#ff7f0e", lw=2.0, alpha=0.9)
        axm.annotate(f"{mean_c:+.3f}", xy=(XLIM[1], mean_c), xytext=(-2, 3), textcoords="offset points",
                     ha="right", fontsize=8.5, color="#ff7f0e", weight="bold")
        axm.set_ylim(min(blo - pad, mean_c - 3 * pad), max(bhi + pad, mean_c + 3 * pad))
        axm.set_ylabel("a", fontsize=8)
        axm.grid(alpha=0.3)

        if need_up:
            axu = axs[0]
            ulo, uhi = np.nanpercentile(up["a_phys"], [2, 98]); upad = 0.2 * max(uhi - ulo, 1e-3)
            for un in sorted(up["unit"].unique()):
                su = up[up["unit"] == un]
                axu.scatter(su["mjd"], su["a_phys"], s=6, color=ucol[un], alpha=0.5, rasterized=True)
            axu.set_ylim(ulo - upad, uhi + upad); axu.set_yticks([round(float(np.median(up["a_phys"])), 2)])
            axu.text(0.99, 0.8, f"n={len(up)}", transform=axu.transAxes, ha="right", fontsize=6.5, color="#666")
            axu.grid(alpha=0.2); breakmarks(axu, axm)
            axu.spines["bottom"].set_visible(False); axm.spines["top"].set_visible(False)
            axu.tick_params(labelbottom=False, bottom=False, labelsize=7)
        if need_dn:
            axd = axs[-1]
            dlo, dhi = np.nanpercentile(dn["a_phys"], [2, 98]); dpad = 0.2 * max(dhi - dlo, 1e-3)
            for un in sorted(dn["unit"].unique()):
                su = dn[dn["unit"] == un]
                axd.scatter(su["mjd"], su["a_phys"], s=6, color=ucol[un], alpha=0.5, rasterized=True)
            axd.set_ylim(dlo - dpad, dhi + dpad); axd.set_yticks([round(float(np.median(dn["a_phys"])), 2)])
            axd.text(0.99, 0.8, f"n={len(dn)}", transform=axd.transAxes, ha="right", fontsize=6.5, color="#666")
            axd.grid(alpha=0.2); breakmarks(axm, axd)
            axm.spines["bottom"].set_visible(False); axd.spines["top"].set_visible(False)
            axm.tick_params(labelbottom=False, bottom=False)
        for a_ in axs:
            a_.set_xlim(*XLIM); a_.tick_params(labelsize=7)
        for a_ in axs[:-1]:
            a_.tick_params(labelbottom=False)
        ep = "2026+" if filt == "m400" else "2024+"
        axs[0].set_title(f"{filt}  adopted a = {mean_c:+.4f} ± {aerr:.4f}  ({ep}, n={n_eff})", fontsize=8.5)

    h = [plt.Line2D([0], [0], marker="o", ls="", color=ucol[un]) for un in units_all]
    fig.legend(h, units_all, loc="lower center", ncol=min(len(units_all), 16), fontsize=8, frameon=False)
    fig.suptitle("Adopted mean colour term per filter — orange line = 3σ-clipped mean "
                 "(m400: 2026+ only; others: all 2024+); err<3×med, nightly median ≥5 frames", y=0.985)
    fig.savefig(FOLDER / "F43_colorterm_adopted.png", dpi=130)
    A = pd.DataFrame(adopted)
    A.to_csv(FOLDER / "colorterm_adopted.csv", index=False)
    print(A.to_string(index=False))
    print(f"\nsaved -> F43_colorterm_adopted.png, colorterm_adopted.csv in {FOLDER}")


if __name__ == "__main__":
    main()
