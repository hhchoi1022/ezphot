#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260914_offset_needed.py

채택 색항(colorterm_adopted.csv)을 0으로 만드는 데 필요한 transmission curve 파장 오프셋 계산 (F44).
- 방법 1(주값): 채택 epoch/컷(F43과 동일: m400=2026+, 나머지 2024+, err<3x중앙, 3σ클립)의 δ* 평균
  — δ*는 정의상 "physical 곡선에서 색항을 0으로 만드는 오프셋"이므로 이것이 곧 필요 보정량.
- 방법 2(교차검증): a_adopt ÷ 레버리지. 프레임별 정확 관계 a0 = -m·δ* 에서 m_i = -a/δ* 중앙값.
- 산출: transmission_offset_needed.csv, F44_offset_needed.png
- 주의: m675+는 레버리지 |da/dδ|<0.005로 신뢰도 낮음(빈 원). m500은 rigid shift로 표현 불가한
  shape 오류(레버리지 부호 반전) — 값 자체보다 곡선 재측정 필요 신호.

  MPLCONFIGDIR=/tmp/ezphot-matplotlib /home/hhchoi1022/miniconda3/envs/too/bin/python -u \\
      ezphot/analysis/260914_offset_needed.py
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
MJD_2026 = 61041.0


def main():
    r = pd.read_csv(FOLDER / "offset_timeseries_ezall.csv")
    r = r[np.isfinite(r["a_phys"]) & np.isfinite(r["dstar"]) & np.isfinite(r["dstar_err"])]
    thr = r.groupby("filter")["dstar_err"].transform("median") * 3.0
    r = r[r["dstar_err"] < thr]
    A = pd.read_csv(FOLDER / "colorterm_adopted.csv").set_index("filter")

    rows = []
    for f, g in r.groupby("filter"):
        sel = g[g["mjd"] >= MJD_2026] if f == "m400" else g
        dm, dmed, dstd = sigma_clipped_stats(sel["dstar"].to_numpy(), sigma=3.0, maxiters=10)
        n_eff = int((np.abs(sel["dstar"] - dmed) < 3 * dstd).sum())
        derr = dstd / np.sqrt(max(n_eff, 1))
        gg = sel[np.abs(sel["dstar"]) > 0.5]
        lev = float(np.median(-gg["a_phys"] / gg["dstar"])) if len(gg) > 30 else np.nan
        a_adopt = A.loc[f, "a_mean"]
        check = a_adopt / lev if np.isfinite(lev) and lev != 0 else np.nan
        rows.append(dict(filter=f, pivot=int(f[1:4]), a_adopt=a_adopt,
                         offset_nm=round(float(dm), 3), off_err=round(float(derr), 3),
                         off_std=round(float(dstd), 2), leverage=round(lev, 4),
                         check_a_over_lev=round(float(check), 2), n=n_eff,
                         epoch=("2026+" if f == "m400" else "2024+")))
    T = pd.DataFrame(rows).sort_values("pivot")
    T.to_csv(FOLDER / "transmission_offset_needed.csv", index=False)
    print(T.to_string(index=False))

    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    bl = T[T["pivot"] <= 650]; rd = T[T["pivot"] > 650]
    ax.errorbar(bl["pivot"], bl["offset_nm"], yerr=bl["off_err"], fmt="o", ms=8, capsize=3,
                color="#d62728", label="reliable (m400-m650)")
    ax.errorbar(rd["pivot"], rd["offset_nm"], yerr=rd["off_err"], fmt="o", ms=8, mfc="none",
                capsize=3, color="#d62728", alpha=0.5, label="low leverage (m675+, caution)")
    for _, s in T.iterrows():
        ax.annotate(f"{s['filter']}\n{s['offset_nm']:+.2f}", (s["pivot"], s["offset_nm"]),
                    textcoords="offset points", xytext=(0, 9), ha="center", fontsize=7.5)
    ax.axhline(0, color="k", lw=0.7)
    ax.set(xlabel="pivot wavelength [nm]", ylabel="required curve offset [nm] (redward +)",
           title="Offset needed on the PHYSICAL curve to zero the adopted colour term\n"
                 "(m400: 2026+ copy; others: 2024+; err-cut + 3σ-clipped mean)")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(FOLDER / "F44_offset_needed.png", dpi=150)
    print(f"\nsaved -> F44_offset_needed.png, transmission_offset_needed.csv in {FOLDER}")


if __name__ == "__main__":
    main()
