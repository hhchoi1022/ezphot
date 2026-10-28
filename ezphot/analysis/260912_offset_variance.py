#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260912_offset_variance.py

"날짜별 δ* 차이는 진짜 곡선 변화인가?"의 원인 분해 (§17.3, F33).  No re-photometry.

D1 variance decomposition (per filter, within (filter,unit[,m400 epoch]) copies):
    pairs (same tile, different night)  -> pure TIME variance
    pairs (same night, different tile)  -> pure SAMPLE/SPACE variance
    pairs (same tile, same night)       -> noise cross-check
    each: sigma^2 = 0.5<d^2>_clipped - <err^2>  (excess over measurement noise)
D2 atmosphere: copy residual vs AIRMASS / HUMIDITY (frame headers; PWV proxy)
D3 temperature: copy-centred residual vs AMBTEMP -> beta [nm/deg C] per filter (interference-filter expectation ~+0.02)
D4 shape cross-check: sigma_space per filter vs the std-vs-survey gap (17.2)

Frame identity is reconstructed deterministically (same random seed/sampling as the delta* runs), then headers
are harvested (AIRMASS/AMBTEMP/HUMIDITY) into frame_env.csv (cached).  Pipeline read-only (headers only).

  MPLCONFIGDIR=/tmp/ezphot-matplotlib OMP_NUM_THREADS=1 /home/hhchoi1022/miniconda3/envs/too/bin/python -u \\
      ezphot/analysis/260912_offset_variance.py --n-proc 8
"""
from __future__ import annotations

import argparse
import importlib
import os
import random
import re
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
from astropy.io import fits          # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
OT = importlib.import_module("260911_offset_timeseries")
R = OT.R
GRID = OT.GRID
FOLDER = OT.FOLDER
ENV_CSV = FOLDER / "frame_env.csv"
MJD_2026 = 61041.0


# ------------------------------------------------------------------ env harvest
def _sample_like_run(listfile, per_night=6, seed=20260911):
    frames = [l.strip() for l in Path(listfile).read_text().splitlines() if l.strip()]
    frames = [f for f in frames if f.split("/")[-3] in GRID]
    by_fn = defaultdict(list)
    for f in frames:
        m = re.search(r"/(20\d\d-\d\d-\d\d)/", f)
        by_fn[(f.split("/")[-3], m.group(1) if m else "?")].append(f)
    random.seed(seed)
    out = []
    for _, fl in by_fn.items():
        out += random.sample(fl, min(per_night, len(fl)))
    return out


def _env_one(f):
    try:
        h = fits.getheader(f)
        return dict(file=f, mjd=round(OT.meta(f)[2], 6),
                    filter=f.split("/")[-3], unit=(re.search(r"(7DT\d\d)", f) or [None, "?"])[1] if re.search(r"(7DT\d\d)", f) else "?",
                    airmass=float(h.get("AIRMASS", np.nan) or np.nan),
                    ambtemp=float(h.get("AMBTEMP", np.nan) or np.nan),
                    humidity=float(h.get("HUMIDITY", np.nan) or np.nan))
    except Exception:
        return None


def harvest_env(n_proc):
    if ENV_CSV.exists():
        e = pd.read_csv(ENV_CSV)
        print(f"env cache: {len(e)} rows"); return e
    files = set()
    files |= set(_sample_like_run("/home/hhchoi1022/ezphot/log/20260911_colorterm_timeseries/all_medium_2025.txt"))
    files |= set(_sample_like_run(FOLDER / "all_medium_2024_pipeline.txt"))
    # std2024 campaign frames
    rf = pd.read_csv("/home/hhchoi1022/ezphot/log/20260908_zp_color_check/refit_frames.csv",
                     low_memory=False, usecols=["file", "catalog", "filter"])
    rf = rf[(rf["catalog"] == "GAIAXP_CORR_LAMOST") & rf["filter"].isin(GRID)]
    files |= set(rf["file"].unique())
    files = sorted(files)
    print(f"harvesting headers of {len(files)} frames ...", flush=True)
    rows = []
    with Pool(n_proc) as pool:
        for i, r_ in enumerate(pool.imap_unordered(_env_one, files, chunksize=64), 1):
            if r_:
                rows.append(r_)
            if i % 8000 == 0:
                print(f"  {i}/{len(files)}", flush=True)
    e = pd.DataFrame(rows)
    e.to_csv(ENV_CSV, index=False)
    print(f"env harvested: {len(e)} rows -> {ENV_CSV}")
    return e


def _std_mjd(fr):
    m = re.search(r"_(\d{8})_(\d{6})_", fr)
    if not m:
        return np.nan
    from astropy.time import Time
    s, t = m.group(1), m.group(2)
    return round(Time(f"{s[:4]}-{s[4:6]}-{s[6:]}T{t[:2]}:{t[2:4]}:{t[4:]}").mjd, 6)


# ------------------------------------------------------------------ pair variance
def pair_var(g, cat, max_pairs=6000):
    """0.5<d^2> - <err^2> for a pair category within one copy group."""
    tiles = g["tile"].to_numpy(); nights = g["night"].to_numpy()
    d = g["dstar"].to_numpy(); e2 = g["dstar_err"].to_numpy() ** 2
    pairs = []
    if cat in ("ST_DN", "ST_SN"):
        idx_by = defaultdict(list)
        for i, t in enumerate(tiles):
            idx_by[t].append(i)
    else:                                            # SN_DT
        idx_by = defaultdict(list)
        for i, n_ in enumerate(nights):
            idx_by[n_].append(i)
    for _, idxs in idx_by.items():
        if len(idxs) < 2:
            continue
        for i, j in combinations(idxs, 2):
            if cat == "ST_DN" and nights[i] == nights[j]:
                continue
            if cat == "ST_SN" and nights[i] != nights[j]:
                continue
            if cat == "SN_DT" and tiles[i] == tiles[j]:
                continue
            pairs.append((d[i] - d[j], e2[i] + e2[j]))
    if len(pairs) < 20:
        return np.nan, 0
    if len(pairs) > max_pairs:
        random.seed(1); pairs = random.sample(pairs, max_pairs)
    dd = np.array([p[0] for p in pairs]); ee = np.array([p[1] for p in pairs])
    rob = 1.4826 * np.median(np.abs(dd - np.median(dd)))            # clip 4-sigma outliers
    keep = np.abs(dd - np.median(dd)) < 4 * rob
    var = 0.5 * np.mean(dd[keep] ** 2) - 0.5 * np.mean(ee[keep])
    return var, int(keep.sum())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n-proc", type=int, default=8)
    a = ap.parse_args()
    env = harvest_env(a.n_proc)
    env["mjd"] = env["mjd"].round(6)

    # ---- load delta* rows (survey + 2024pipe + std2024), define copies ----
    parts = []
    for fn, src in [("offset_timeseries.csv", "survey"), ("offset_timeseries_2024pipe.csv", "pipe2024"),
                    ("offset_timeseries_std2024.csv", "std2024")]:
        p = FOLDER / fn
        if p.exists():
            d = pd.read_csv(p); d["src"] = src; parts.append(d)
    r = pd.concat(parts, ignore_index=True)
    r = r[np.isfinite(r["dstar"]) & np.isfinite(r["dstar_err"])]
    r["mjd"] = r["mjd"].round(6)
    r["copy"] = r["filter"] + "|" + r["unit"] + np.where((r["filter"] == "m400") & (r["mjd"] >= MJD_2026), "|26", "")
    r = r.merge(env[["mjd", "filter", "unit", "airmass", "ambtemp", "humidity"]], on=["mjd", "filter", "unit"], how="left")
    print(f"{len(r)} rows; env matched: {np.isfinite(r['ambtemp']).mean() * 100:.0f}%")

    # ---- D1: variance decomposition per filter (pool copies) ----
    rows = []
    for filt, gf in r[r["src"] != "std2024"].groupby("filter"):     # survey-like only (std has 1 field/tile)
        agg = {c: [0.0, 0] for c in ("ST_DN", "SN_DT", "ST_SN")}
        for _, g in gf.groupby("copy"):
            if len(g) < 30:
                continue
            for cat in agg:
                v, n_ = pair_var(g, cat)
                if np.isfinite(v) and n_ > 0:
                    agg[cat][0] += v * n_; agg[cat][1] += n_
        rec = dict(filter=filt, pivot=int(filt[1:4]))
        for cat, (sv, sn) in agg.items():
            rec[f"var_{cat}"] = sv / sn if sn else np.nan
            rec[f"n_{cat}"] = sn
        rows.append(rec)
    D = pd.DataFrame(rows).sort_values("pivot").reset_index(drop=True)
    for cat in ("ST_DN", "SN_DT", "ST_SN"):
        D[f"sig_{cat}"] = np.sqrt(np.clip(D[f"var_{cat}"], 0, None))

    # ---- D2/D3: copy-centred residual vs env ----
    r["resid"] = r["dstar"] - r.groupby("copy")["dstar"].transform("median")
    r["t_c"] = r["ambtemp"] - r.groupby("copy")["ambtemp"].transform("median")
    corr_rows = []
    for filt, g in r.groupby("filter"):
        g = g[np.isfinite(g["resid"])]
        rec = dict(filter=filt, pivot=int(filt[1:4]))
        for col in ("airmass", "humidity"):
            gg = g[np.isfinite(g[col])]
            rec[f"rho_{col}"] = float(gg["resid"].corr(gg[col], method="spearman")) if len(gg) > 100 else np.nan
        gg = g[np.isfinite(g["t_c"])]
        if len(gg) > 100:
            beta, be, _, _ = R.weighted_linfit(gg["t_c"].to_numpy(), gg["resid"].to_numpy(),
                                               np.sqrt(gg["dstar_err"].to_numpy() ** 2 + 0.05 ** 2))
            rec["beta_T"] = beta; rec["beta_T_err"] = be
        else:
            rec["beta_T"] = np.nan; rec["beta_T_err"] = np.nan
        corr_rows.append(rec)
    C = pd.DataFrame(corr_rows).sort_values("pivot").reset_index(drop=True)
    T = D.merge(C, on=["filter", "pivot"])

    # ---- D4: sigma_space vs std-vs-survey gap ----
    S = pd.read_csv(FOLDER / "offset_2024_2026_summary.csv")
    S["gap"] = (S["dstar_survey"] - S["dstar_std2024"]).abs()
    T = T.merge(S[["filter", "gap"]], on="filter", how="left")
    T.to_csv(FOLDER / "variance_decomp.csv", index=False)
    print(T[["filter", "sig_ST_SN", "sig_ST_DN", "sig_SN_DT", "rho_airmass", "rho_humidity", "beta_T", "gap"]]
          .round(3).to_string(index=False))

    # ---- F33 ----
    fig, ax = plt.subplots(2, 2, figsize=(14.5, 10))
    piv = T["pivot"]
    ax[0, 0].plot(piv, T["sig_ST_SN"], "o-", color="grey", label="same tile+night (noise check)")
    ax[0, 0].plot(piv, T["sig_ST_DN"], "o-", color="#d62728", label="TIME: same tile, diff night")
    ax[0, 0].plot(piv, T["sig_SN_DT"], "o-", color="#1f77b4", label="SAMPLE: same night, diff tile")
    ax[0, 0].set(xlabel="pivot [nm]", ylabel="excess sigma [nm]", title="D1 variance decomposition (noise-subtracted)")
    ax[0, 0].legend(fontsize=9); ax[0, 0].set_yscale("log")
    bt = T[T["pivot"] < 660]
    ax[0, 1].errorbar(bt["pivot"], bt["beta_T"], yerr=bt["beta_T_err"], fmt="o", capsize=2, color="#d62728")
    rt = T[T["pivot"] >= 660]
    ax[0, 1].errorbar(rt["pivot"], rt["beta_T"], yerr=rt["beta_T_err"], fmt="o", mfc="none", capsize=2, color="#d62728", alpha=0.5)
    ax[0, 1].axhline(0, color="k", lw=0.6); ax[0, 1].axhline(0.02, color="green", ls="--", lw=1, label="interference-filter ~+0.02 nm/C")
    ax[0, 1].set(xlabel="pivot [nm]", ylabel="beta_T [nm/degC]", title="D3 temperature coefficient (copy-centred)")
    ax[0, 1].legend(fontsize=9)
    ax[1, 0].plot(piv, T["rho_airmass"], "o-", label="Spearman rho vs AIRMASS", color="#9467bd")
    ax[1, 0].plot(piv, T["rho_humidity"], "s-", label="vs HUMIDITY (PWV proxy)", color="#2ca02c")
    ax[1, 0].axhline(0, color="k", lw=0.6)
    ax[1, 0].set(xlabel="pivot [nm]", ylabel="Spearman rho", title="D2 atmosphere correlations (copy-centred residual)")
    ax[1, 0].legend(fontsize=9)
    ax[1, 1].scatter(T["gap"], T["sig_SN_DT"], s=30, color="#1f77b4")
    for _, s_ in T.iterrows():
        if np.isfinite(s_["gap"]) and np.isfinite(s_["sig_SN_DT"]):
            ax[1, 1].annotate(s_["filter"], (s_["gap"], s_["sig_SN_DT"]), textcoords="offset points", xytext=(4, 3), fontsize=7)
    ax[1, 1].set(xlabel="|survey - std2024| gap [nm] (17.2)", ylabel="sigma_SAMPLE [nm]",
                 title="D4 shape hypothesis: sample-variance vs dataset gap")
    for x in ax.ravel():
        x.grid(alpha=0.3)
    fig.suptitle("Why does delta* vary with date? — variance decomposition & environment correlations", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FOLDER / "F33_variance_decomp.png", dpi=140)
    print(f"\nsaved -> F33_variance_decomp.png, variance_decomp.csv in {FOLDER}")


if __name__ == "__main__":
    main()
