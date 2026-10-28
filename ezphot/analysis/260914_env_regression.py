#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260914_env_regression.py

색항 잔차(copy 중심)를 온도·수증기압(PWV 프록시)·seeing에 대해 필터별 3변수 동시 회귀 (F40).
판별 논리:
  - PWV가 원인이면 beta_E가 물 흡수 대역이 밴드 안에 있는 필터(m725: 715-735nm, m825: 810-840nm)에서만 돌출
  - seeing이 원인이면 beta_S가 밴드-무관하게 유의
  - 3변수 동시 적합이라 서로의 계절 공변을 부분적으로 분리 (부분회귀계수)
수증기압 e = 6.112*exp(17.62*Td/(243.12+Td)) [hPa] (Magnus, DEWPOINT 사용 — RH보다 PWV에 좋은 프록시).

  MPLCONFIGDIR=/tmp/ezphot-matplotlib /home/hhchoi1022/miniconda3/envs/too/bin/python -u \\
      ezphot/analysis/260914_env_regression.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402
import pandas as pd                 # noqa: E402

warnings.filterwarnings("ignore")
FOLDER = Path("/home/hhchoi1022/ezphot/log/20260911_offset_timeseries")
MJD_2026 = 61041.0
WATER = [(715, 735), (810, 840), (890, 920)]        # 주요 H2O 흡수 대역 [nm]


def main():
    r = pd.read_csv(FOLDER / "offset_timeseries_ezall.csv")
    env = pd.read_csv(FOLDER / "env_ezall.csv")
    r = r.merge(env, on="file", how="inner")
    r = r[np.isfinite(r["a_phys"]) & np.isfinite(r["dstar_err"])]
    thr = r.groupby("filter")["dstar_err"].transform("median") * 3.0
    r = r[r["dstar_err"] < thr].copy()
    r["copy"] = r["filter"] + "|" + r["unit"] + np.where((r["filter"] == "m400") & (r["mjd"] >= MJD_2026), "|26", "")
    r["resid"] = r["a_phys"] - r.groupby("copy")["a_phys"].transform("median")
    # bulk cut: drop coherent outlier clusters (early-2024 m450/m475 etc.) that corrupt the linear fits
    madf = r.groupby("filter")["resid"].transform(lambda v: 1.4826 * np.median(np.abs(v - v.median())))
    r = r[np.abs(r["resid"]) < 7 * madf]
    td = r["dewpoint"]
    r["evap"] = 6.112 * np.exp(17.62 * td / (243.12 + td))          # hPa
    fin = np.isfinite(r["ambtemp"]) & np.isfinite(r["evap"]) & np.isfinite(r["seeing"]) & np.isfinite(r["resid"])
    r = r[fin]
    for c in ("ambtemp", "evap", "seeing"):
        r[c + "_c"] = r[c] - r.groupby("copy")[c].transform("median")
    print(f"{len(r)} frames with full env")
    print("regressor spreads: T {:.1f}C, e {:.1f}hPa, seeing {:.2f}\"".format(
        r["ambtemp_c"].std(), r["evap_c"].std(), r["seeing_c"].std()))
    corr = r[["ambtemp_c", "evap_c", "seeing_c"]].corr()
    print("regressor correlations:\n", corr.round(2).to_string())

    rows = []
    for f, g in r.groupby("filter"):
        X = np.column_stack([g["ambtemp_c"], g["evap_c"], g["seeing_c"], np.ones(len(g))])
        y = g["resid"].to_numpy()
        beta, res_, rank, _ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ beta
        dof = len(g) - X.shape[1]
        s2 = np.sum((y - yhat) ** 2) / dof
        cov = s2 * np.linalg.inv(X.T @ X)
        err = np.sqrt(np.diag(cov))
        rows.append(dict(filter=f, pivot=int(f[1:4]), n=len(g),
                         bT=beta[0], bT_e=err[0], bE=beta[1], bE_e=err[1], bS=beta[2], bS_e=err[2]))
    T = pd.DataFrame(rows).sort_values("pivot").reset_index(drop=True)
    T.to_csv(FOLDER / "env_regression.csv", index=False)
    show = T.copy()
    for c in ("bT", "bT_e", "bE", "bE_e", "bS", "bS_e"):
        show[c] = (show[c] * 1000).round(2)                          # mmag/mag per unit
    print(show.to_string(index=False))

    fig, ax = plt.subplots(3, 1, figsize=(12.5, 11), sharex=True)
    piv = T["pivot"]
    specs = [("bE", "bE_e", "water-vapour pressure e [mmag/mag per hPa]", "#1f77b4",
              "(A) PWV proxy — prediction: peaks ONLY at water-band filters (shaded)"),
             ("bS", "bS_e", "seeing [mmag/mag per arcsec]", "#2ca02c",
              "(B) seeing — band-independent if aperture/PSF colour effect"),
             ("bT", "bT_e", "temperature [mmag/mag per degC]", "#d62728",
              "(C) temperature (partial, after removing e & seeing shares)")]
    for a_, (col, ecol, lab, c_, title) in zip(ax, specs):
        a_.errorbar(piv, T[col] * 1000, yerr=T[ecol] * 1000, fmt="o-", ms=5, capsize=2, color=c_)
        a_.axhline(0, color="k", lw=0.6)
        for w0, w1 in WATER:
            a_.axvspan(w0, w1, color="#1f77b4", alpha=0.10)
        a_.set_ylabel(lab, fontsize=9)
        a_.set_title(title, fontsize=10)
        a_.grid(alpha=0.3)
    ax[-1].set_xlabel("pivot wavelength [nm]")
    fig.suptitle("Joint regression of the colour-term residual on (T, water vapour, seeing) — per filter", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FOLDER / "F40_env_regression.png", dpi=140)
    print(f"\nsaved -> F40_env_regression.png, env_regression.csv in {FOLDER}")


if __name__ == "__main__":
    main()
