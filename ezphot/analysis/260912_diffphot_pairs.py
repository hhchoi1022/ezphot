#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260912_diffphot_pairs.py

사용자 제안 검사: 같은 (망원경, 필터, 타일)의 두 이미지 A·B에서 공통 별의 차등측광
    d_i = (m_iA - m_iB) - median_j(m_jA - m_jB)
을 별 색(bp-rp; x축으로만 사용)에 대해 적합 -> 기울기 s_AB.  Gaia 합성등급이 식에 없으므로
별별 고정 XP 오차는 완전히 소거됨.  s_AB가 남으면 = 두 epoch 사이 실제 대기/기기/측광과정 색response 차이.

검증 항목:
  (1) s_AB 분포 vs 자체 오차 -> 초과인자 (реference 없이 잰 epoch간 response 변동)
  (2) KEY: s_AB  vs  (a_A - a_B)  — 참조(XP) 기반 프레임 색항 차와 1:1이면 색항 변동은 실제 기기/대기 변동
  (3) aperture 의존: s(3") vs s(10") 동일 페어
  (4) s_AB vs (dAIRMASS, dSEEING, dT, d(sky)) 상관
Read-only (pipeline cats + headers).  ezall 야간작업과 병행 가능(16워커).

  MPLCONFIGDIR=/tmp/ezphot-matplotlib OMP_NUM_THREADS=1 /home/hhchoi1022/miniconda3/envs/too/bin/python -u \\
      ezphot/analysis/260912_diffphot_pairs.py --pairs-per-filter 200 --n-proc 16
"""
from __future__ import annotations

import argparse
import importlib
import os
import random
import sys
import warnings
from collections import defaultdict
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402
import pandas as pd                 # noqa: E402
from astropy.coordinates import SkyCoord, match_coordinates_sky   # noqa: E402
from astropy.io import fits          # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
OT = importlib.import_module("260911_offset_timeseries")
R, K, C = OT.R, OT.K, OT.C
FOLDER = OT.FOLDER
GRID = OT.GRID


def read_cat_pair(catpath):
    with fits.open(catpath) as h:
        d = h["LDAC_OBJECTS"].data
        cols = {c: np.asarray(d[c], float) for c in
                ["ALPHA_J2000", "DELTA_J2000", "MAG_APER_3", "MAGERR_APER_3", "SNR_APER_3",
                 "MAG_APER_5", "MAGERR_APER_5", "FLAGS", "CLASS_STAR"]}
    df = pd.DataFrame(cols)
    df = df[(df["FLAGS"] <= 0) & (df["CLASS_STAR"] >= 0.8) & (df["MAGERR_APER_3"] < 0.05)
            & (df["SNR_APER_3"] > 20) & (df["SNR_APER_3"] < 300)]
    if len(df) < 30:
        return None
    # isolation 10": nearest neighbour in the same (filtered) list
    cc = SkyCoord(df["ALPHA_J2000"].to_numpy(), df["DELTA_J2000"].to_numpy(), unit="deg")
    _, sep, _ = match_coordinates_sky(cc, cc, nthneighbor=2)
    df = df[np.asarray(sep.arcsec) > 10.0]
    return df.reset_index(drop=True) if len(df) >= 30 else None


def hdr_env(frame):
    try:
        h = fits.getheader(frame)
        return dict(airmass=float(h.get("AIRMASS", np.nan) or np.nan), seeing=float(h.get("SEEING", np.nan) or np.nan),
                    skyval=float(h.get("SKYVAL", np.nan) or np.nan), ambtemp=float(h.get("AMBTEMP", np.nan) or np.nan))
    except Exception:
        return dict(airmass=np.nan, seeing=np.nan, skyval=np.nan, ambtemp=np.nan)


def process(job):
    try:
        (tile, filt, unit, fA, fB, aA, aB, mjdA, mjdB) = job
        cA = read_cat_pair(fA[:-5] + "_cat.fits"); cB = read_cat_pair(fB[:-5] + "_cat.fits")
        if cA is None or cB is None:
            return None
        sA = SkyCoord(cA["ALPHA_J2000"].to_numpy(), cA["DELTA_J2000"].to_numpy(), unit="deg")
        sB = SkyCoord(cB["ALPHA_J2000"].to_numpy(), cB["DELTA_J2000"].to_numpy(), unit="deg")
        idx, sep, _ = match_coordinates_sky(sA, sB)
        ok = np.asarray(sep.arcsec < 1.0)
        if ok.sum() < 30:
            return None
        A = cA[ok].reset_index(drop=True); B = cB.iloc[idx[ok]].reset_index(drop=True)
        # colour (x-axis only): archive bp-rp
        got = K.get_reference("GAIAXP_CORR_LAMOST", tile, float(A["ALPHA_J2000"].median()),
                              float(A["DELTA_J2000"].median()), 1.34, 0.90)
        if got is None:
            return None
        tbl, coords, _ = got
        ic, sc, _ = match_coordinates_sky(SkyCoord(A["ALPHA_J2000"].to_numpy(), A["DELTA_J2000"].to_numpy(), unit="deg"), coords)
        okc = np.asarray(sc.arcsec < 2.0)
        if okc.sum() < 25:
            return None
        bprp = np.full(len(A), np.nan)
        bprp[okc] = np.asarray(tbl["bp-rp"], float)[ic[okc]]
        out = dict(tile=tile, filter=filt, unit=unit, dmjd=abs(mjdA - mjdB), da_ref=aA - aB, n=int(okc.sum()))
        for ap_, tag in (("3", "s3"), ("5", "s10")):
            dm = A[f"MAG_APER_{ap_}"].to_numpy() - B[f"MAG_APER_{ap_}"].to_numpy()
            sig = np.sqrt(A[f"MAGERR_APER_{ap_}"].to_numpy() ** 2 + B[f"MAGERR_APER_{ap_}"].to_numpy() ** 2 + 0.005 ** 2)
            fin = np.isfinite(dm) & np.isfinite(bprp) & np.isfinite(sig)
            if fin.sum() < 25:
                out[tag] = np.nan; out[tag + "_err"] = np.nan; continue
            d_i = dm[fin] - np.median(dm[fin])
            s, se, _, _ = R.weighted_linfit(bprp[fin], d_i, sig[fin])
            out[tag] = s; out[tag + "_err"] = se
        eA = hdr_env(fA); eB = hdr_env(fB)
        for k_ in ("airmass", "seeing", "skyval", "ambtemp"):
            out["d_" + k_] = eA[k_] - eB[k_]
        return out
    except Exception as e:                                       # noqa: BLE001
        return dict(error=str(e)[:150])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-per-filter", type=int, default=200)
    ap.add_argument("--n-proc", type=int, default=16)
    a = ap.parse_args()
    # frames with known paths: frame_env (file,mjd,filter,unit) x offset_timeseries (a_phys, tile, night)
    env = pd.read_csv(FOLDER / "frame_env.csv"); env["mjd"] = env["mjd"].round(6)
    ts = pd.read_csv(FOLDER / "offset_timeseries.csv"); ts["mjd"] = ts["mjd"].round(6)
    m = ts.merge(env[["mjd", "filter", "unit", "file"]], on=["mjd", "filter", "unit"], how="inner")
    m = m[m["file"].notna()]
    print(f"frames with paths: {len(m)}")
    jobs = []
    random.seed(20260912)
    for (filt, unit, tile), g in m.groupby(["filter", "unit", "tile"]):
        nights = g["night"].unique()
        if len(nights) < 2:
            continue
        rows = list(g.itertuples())
        prs = [(r1, r2) for r1, r2 in combinations(rows, 2) if r1.night != r2.night]
        for r1, r2 in prs:
            jobs.append((tile, filt, unit, r1.file, r2.file, r1.a_phys, r2.a_phys, r1.mjd, r2.mjd))
    by_f = defaultdict(list)
    for j in jobs:
        by_f[j[1]].append(j)
    jobs = []
    for f in GRID:
        if by_f[f]:
            random.shuffle(by_f[f])
            jobs += by_f[f][:a.pairs_per_filter]
    print(f"{len(jobs)} pairs ({a.pairs_per_filter}/filter cap), {a.n_proc} workers", flush=True)
    rows = []
    with Pool(a.n_proc, initializer=K._refit_init) as pool:
        for i, r_ in enumerate(pool.imap_unordered(process, jobs, chunksize=4), 1):
            if r_ and "error" not in r_:
                rows.append(r_)
            if i % 200 == 0:
                print(f"  {i}/{len(jobs)}, {len(rows)} ok", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(FOLDER / "diffphot_pairs.csv", index=False)
    print(f"\n{len(res)} pairs measured")

    blue = res[res["filter"].isin(["m400", "m425", "m450", "m475", "m500", "m525", "m550"])].copy()
    print("\n=== per-filter: Gaia-free slope s_AB (3\") 통계 ===")
    g = res.groupby("filter").agg(n=("s3", "size"), s3_med=("s3", "median"),
                                  s3_rob=("s3", lambda v: 1.4826 * np.median(np.abs(v - np.median(v)))),
                                  err_med=("s3_err", "median"))
    g["excess"] = np.sqrt(np.clip(g["s3_rob"] ** 2 - g["err_med"] ** 2, 0, None))
    g = g.reindex([f for f in GRID if f in g.index])
    print(g.round(4).to_string())
    # KEY correlation: s_AB vs da_ref
    fin = np.isfinite(blue["s3"]) & np.isfinite(blue["da_ref"])
    rho = float(blue[fin]["s3"].corr(blue[fin]["da_ref"]))
    sl, sle, _, _ = R.weighted_linfit(blue[fin]["da_ref"].to_numpy(), blue[fin]["s3"].to_numpy(),
                                      np.sqrt(blue[fin]["s3_err"].to_numpy() ** 2 + 0.003 ** 2))
    print(f"\nKEY (blue): corr(s_AB, a_A - a_B) = {rho:+.3f}, slope = {sl:+.3f}+-{sle:.3f} (1이면 완전 일치)")

    # figure
    fig, ax = plt.subplots(2, 2, figsize=(14, 10.5))
    ax[0, 0].errorbar(blue[fin]["da_ref"], blue[fin]["s3"], yerr=blue[fin]["s3_err"], fmt="o", ms=3, alpha=0.4, capsize=0)
    lim = np.nanpercentile(np.r_[blue[fin]["da_ref"], blue[fin]["s3"]], [1, 99])
    ax[0, 0].plot(lim, lim, "k--", lw=1, label="y = x")
    ax[0, 0].plot(lim, sl * lim, "r-", lw=1.2, label=f"fit slope {sl:+.2f}+-{sle:.2f}, corr {rho:+.2f}")
    ax[0, 0].set(xlabel="a_A - a_B  (reference-based colour-term difference)", ylabel="s_AB  (Gaia-free differential slope)",
                 title="(A) KEY: Gaia-free slope vs reference-based slope difference (blue filters)")
    ax[0, 0].legend(fontsize=9)
    # excess per filter
    ax[0, 1].plot(range(len(g)), g["err_med"], "o-", color="grey", label="median pair error")
    ax[0, 1].plot(range(len(g)), g["s3_rob"], "o-", color="#d62728", label="observed robust scatter of s_AB")
    ax[0, 1].plot(range(len(g)), g["excess"], "o-", color="#1f77b4", label="EXCESS = true epoch response variation")
    ax[0, 1].set_xticks(range(len(g))); ax[0, 1].set_xticklabels(g.index, rotation=60, fontsize=7)
    ax[0, 1].set(ylabel="colour slope [mag/mag]", title="(B) s_AB scatter vs error: real epoch-to-epoch response variation")
    ax[0, 1].legend(fontsize=8)
    # aperture dependence
    fin2 = np.isfinite(blue["s3"]) & np.isfinite(blue["s10"])
    ax[1, 0].plot(blue[fin2]["s3"], blue[fin2]["s10"], "o", ms=3, alpha=0.4)
    lim2 = np.nanpercentile(np.r_[blue[fin2]["s3"], blue[fin2]["s10"]], [1, 99])
    ax[1, 0].plot(lim2, lim2, "k--", lw=1)
    rho_ap = float(blue[fin2]["s3"].corr(blue[fin2]["s10"]))
    ax[1, 0].set(xlabel="s_AB (3\")", ylabel="s_AB (10\")", title=f"(C) aperture dependence, corr {rho_ap:+.2f}")
    # env correlations
    labs, vals = [], []
    for k_ in ("d_airmass", "d_seeing", "d_skyval", "d_ambtemp", "dmjd"):
        fin3 = np.isfinite(blue["s3"]) & np.isfinite(blue[k_])
        labs.append(k_); vals.append(float(blue[fin3]["s3"].corr(blue[fin3][k_].abs() if k_ == "dmjd" else blue[fin3][k_],
                                                                 method="spearman")))
    ax[1, 1].bar(labs, vals, color="#9467bd")
    ax[1, 1].axhline(0, color="k", lw=0.6)
    ax[1, 1].set(ylabel="Spearman rho vs s_AB", title="(D) what drives the Gaia-free slope (blue filters)")
    for x in ax.ravel():
        x.grid(alpha=0.3)
    fig.suptitle("Image-pair differential photometry: Gaia-free test of colour-response variation", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FOLDER / "F36_diffphot_pairs.png", dpi=140)
    print(f"\nsaved -> F36_diffphot_pairs.png, diffphot_pairs.csv in {FOLDER}")


if __name__ == "__main__":
    main()
