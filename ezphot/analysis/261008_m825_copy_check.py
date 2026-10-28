#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
261008_m825_copy_check.py — is the 2024 7DT10 m825 white-dwarf offset a property of that filter copy?

A  copy/unit history of m825: survey colour terms per unit and period (20260911_colorterm_timeseries summary) plus the
   2024 CALSPEC and 2026 SPSS / re-observation units; the colour term's prediction for a hot star (BP-RP -0.22) relative
   to the field median colour, a * (-0.22 - bprp0) with the global-ZP sign convention (Delta = -a * (bp - bp0)).
B  every CALSPEC star observed with 7DT10 m825 in 2024 (11 stars, all frames): Delta with the pooled correction (as 8.1)
   and with the m825 low-signal curve at the noise-free peak (261008_m825_pairs step 2, 2024 "all m825" predicted-peak
   curve), against BP-RP; the same stars with 7DT12 m825 in 2026 where available (step 2 table).
C  field stars in the same frames (step-2 star table): residual to the bright-star ZP versus colour for well-exposed
   stars (predicted peak >= 300 e-, loss < 5 mmag), 2024 7DT10 vs 2026 7DT12 — does either copy treat blue stars
   differently beyond a linear colour term?
Outputs: ~/ezphot/log/20261008_m825_copy/ (history.csv, calspec_7dt10.csv, field_colour.csv, K1_copy.png)
"""
import importlib
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
P = importlib.import_module("261008_m825_pairs")
SO, SS, TW = P.SO, P.SS, P.TW
OUT = Path("/home/hhchoi1022/ezphot/log/20261008_m825_copy")
PAIRS = Path("/home/hhchoi1022/ezphot/log/20261008_m825_pairs")
BP_WD2149 = -0.217


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)
    # ---------------------------------------------------------------- A: copy history
    H = pd.read_csv("/home/hhchoi1022/ezphot/log/20260911_colorterm_timeseries/colorterm_timeseries_summary.csv")
    H = H[H["filter"] == "m825"].copy()
    H["source"] = "survey colour-term time series"
    D = pd.read_csv(PAIRS / "frames_delta.csv")
    extra = []
    for (ep, unit), g in D[(D["filter"] == "m825") & (D.curve == "beam")].groupby(["epoch", "unit"]):
        nights = sorted({int(m - 16 / 24) for m in g.mjd})
        extra.append(dict(filter="m825", unit=unit, n=g.file.nunique(), night_min=str(nights[0]), night_max=str(nights[-1]),
                          a_x_median=g.a_x.median(), source=f"{ep} standards (this study; a_x outlier-prone)"))
    Hh = pd.concat([H, pd.DataFrame(extra)], ignore_index=True)
    bp0 = 1.1
    for c in ("a_nominal", "a_physical"):
        Hh[f"pred_WD_{c}_mmag"] = -1e3 * Hh[c] * (BP_WD2149 - bp0)
    Hh.to_csv(OUT / "history.csv", index=False)
    print("=== A. m825 units / periods / colour terms\n" + Hh[["unit", "night_min", "night_max", "n", "a_nominal", "a_physical", "a_x_median",
          "pred_WD_a_nominal_mmag", "pred_WD_a_physical_mmag", "source"]].round(4).to_string(index=False))

    # ---------------------------------------------------------------- B: all CALSPEC stars with 7DT10 m825 (2024)
    c = pd.read_csv(SO.OUT.parent / "20260914_calspec_ezphot" / "photometry_meta.csv")
    c = c[(c.status == "ok") & (c["filter"] == "m825") & (c.unit == "7DT10")]
    c = c[[P.cat_path(f).exists() for f in c.file]]
    jobs = [(obj, list(zip(g.file, g["filter"], g.unit))) for obj, g in c.groupby("object")]
    with Pool(min(12, len(jobs)), initializer=P._init) as pool:
        rows = [r for rr in pool.map(SO.process_cal, jobs) for r in rr]
    B = pd.DataFrame(rows)
    B = B[(B.variant == "beam") & (B.std_flags == 0) & (B.std_peak <= SS.PEAK_MAX)].copy()
    B["obj"] = B["object"]
    B["bias"] = SS.expected_bias(B.std_peak_e.to_numpy(), "CALSPEC")
    B["d_raw"] = (B.d_gnd - B.zp + B.zp_x) * 1e3
    B["d_pooled"] = B.d_raw - B.bias
    A2 = pd.read_csv(PAIRS / "step2_stars.csv", usecols=["file", "kscale"]).drop_duplicates("file")
    L2 = pd.read_csv(PAIRS / "step2_loss.csv")
    curve24 = L2[(L2.epoch == 2024) & (L2.kind == "predicted_peak") & (L2.field == "all m825")]
    B = B.merge(A2, on="file", how="left")
    m_syn = B.std_mag + B.zp - B.d_gnd
    B["peak_pred"] = B.kscale * 10 ** (-0.4 * (m_syn - B.zp_x))
    B["d_matched"] = B.d_raw - P.interp_loss(curve24, B.peak_pred.fillna(B.std_peak_e).to_numpy())
    P.AN._init(); TW._init(40)
    reg = pd.read_csv(SO.OUT / "spss_registered_bprp.csv").set_index("name").bprp
    info = {}
    for o in B.obj.unique():
        col = float(P.colours(o, *P.wd_pos2016(o))[0])
        if not np.isfinite(col):
            col = float(reg.get(P.ALIAS26.get(o, o), np.nan))
        info[o] = dict(bprp=col, is_da=TW._INFO[o].get("is_da"))
    S = B.groupby("obj").agg(n=("file", "size"), nights=("mjd", lambda m: ",".join(sorted({str(int(x - 16 / 24)) for x in m}))),
                             exptime=("exptime", "median"), snr=("std_snr", "median"), peak=("std_peak_e", "median"), peak_pred=("peak_pred", "median"),
                             d_raw=("d_raw", "median"), d_pooled=("d_pooled", "median"), d_matched=("d_matched", "median"),
                             err=("d_raw", lambda v: P.boot_median(v))).reset_index()
    S["bprp"] = S.obj.map(lambda o: info[o]["bprp"]); S["is_da"] = S.obj.map(lambda o: info[o]["is_da"])
    T26 = pd.read_csv(PAIRS / "step2_wd.csv"); T26 = T26[T26.epoch == 2026].set_index("obj")
    S["d_matched_2026_7DT12"] = S.obj.map(T26.d_m825_pred); S["err_2026"] = S.obj.map(T26.err)
    S = S.sort_values("bprp")
    S.to_csv(OUT / "calspec_7dt10.csv", index=False)
    print("=== B. CALSPEC stars, 2024 7DT10 m825 (mmag; matched = m825 curve at the noise-free peak)\n" + S.round(2).to_string(index=False))

    # ---------------------------------------------------------------- C: field stars vs colour, two copies
    A = pd.read_csv(PAIRS / "step2_stars.csv")
    A = A[(A.peak_pred >= 300) & np.isfinite(A.col)]
    edges = [-1, 0.5, 0.7, 0.9, 1.1, 1.3, 1.6, 2.0, 4]
    rows = []
    for ep, g in A.groupby("epoch"):
        g = g.assign(cb=pd.cut(g.col, edges))
        for cb, q in g.groupby("cb", observed=True):
            rows.append(dict(epoch=ep, colour_bin=str(cb), col_med=q.col.median(), n=len(q), frames=q.file.nunique(), resid=q.loss.median(),
                             err=1.2533 * 1.4826 * np.median(np.abs(q.loss - q.loss.median())) / np.sqrt(len(q))))
    Cc = pd.DataFrame(rows); Cc.to_csv(OUT / "field_colour.csv", index=False)
    print("=== C. well-exposed field stars: residual to the bright-star ZP vs colour (mmag)\n" + Cc.round(2).to_string(index=False))

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    for r in S.itertuples():
        mk = "o" if r.is_da else "s"
        ax.errorbar(r.bprp, r.d_matched, r.err, fmt=mk, color="#d6604d", ms=7, capsize=2)
        ax.annotate(r.obj, (r.bprp, r.d_matched), fontsize=7, xytext=(4, 4), textcoords="offset points")
        if np.isfinite(r.d_matched_2026_7DT12):
            ax.errorbar(r.bprp + 0.01, r.d_matched_2026_7DT12, r.err_2026, fmt=mk, color="#1b7837", mfc="w", ms=7, capsize=2)
    ax.axhline(0, color="0.5", lw=0.8); ax.grid(alpha=0.25)
    ax.set_xlabel("BP − RP"); ax.set_ylabel("Δ m825, m825 low-signal curve [mmag]")
    ax.set_title("CALSPEC stars: 2024 7DT10 (red) vs 2026 7DT12 (green, open)\ncircle = DA, square = other", fontsize=10)
    ax = axes[1]
    for ep, col in ((2024, "#d6604d"), (2026, "#1b7837")):
        q = Cc[Cc.epoch == ep]
        ax.errorbar(q.col_med, q.resid, q.err, fmt="o-", color=col, label=f"{ep} {'7DT10' if ep == 2024 else '7DT12'} field stars (peak ≥ 300 e⁻)")
    for r in S.itertuples():
        ax.plot(r.bprp, r.d_matched, "*", color="#d6604d", ms=9, mec="k", mew=0.3)
    ax.axhline(0, color="0.5", lw=0.8); ax.grid(alpha=0.25); ax.legend(fontsize=8)
    ax.set_xlabel("BP − RP"); ax.set_ylabel("residual to bright-star ZP [mmag]")
    ax.set_title("Field stars vs colour in the same m825 frames (stars: 2024 CALSPEC)", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "K1_copy.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
