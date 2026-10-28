#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260926_delta_star_v2_summary.py — Step 1 summary: new per-frame delta* (260926_delta_star_v2.py) vs the previous run
(260924_colorterm_versions.py, old star selection, air curves) on the same frames.

  selection effect : new(manufacturer_*_air) - old(manufacturer_*)       paired per frame (same curve, air)
  vacuum effect    : new(*_vac) - new(*_air)                             paired per frame
  per copy         : median delta* per (filter, unit) and per (filter, unit, quarter) for the vacuum variants
Outputs (~/ezphot/log/20260926_delta_star_v2/): dstar_v2_filter_summary.csv, dstar_v2_copy_table.csv,
dstar_v2_copy_quarter.csv, D1_dstar_v2.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time

D = Path("/home/hhchoi1022/ezphot/log/20260926_delta_star_v2")
OLD = Path("/home/hhchoi1022/ezphot/log/20260924_colorterm_versions/colorterm_versions_frames.csv")


def mad(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return 1.4826 * np.median(np.abs(v - np.median(v))) if len(v) else np.nan


def med_err(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return (np.median(v), 1.2533 * mad(v) / np.sqrt(len(v))) if len(v) > 2 else (np.nan, np.nan)


def air_to_vac_shift(lam):
    s2 = (1e3 / lam) ** 2
    return lam * (0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s2) + 0.0001599740894897 / (38.92568793293 - s2))


if __name__ == "__main__":
    N = pd.read_csv(D / "delta_star_v2_frames.csv")
    O = pd.read_csv(OLD)
    O = O[O.combo.isin(["manufacturer_beam", "manufacturer_nobeam"])]
    # keep robust frames: finite delta*, error < 3 nm, colour span >= 0.5 mag
    N = N[np.isfinite(N.dstar) & (N.dstar_err < 3) & ((N.bprp_hi - N.bprp_lo) >= 0.5)]
    O = O[np.isfinite(O.dstar) & (O.dstar_err < 3)]
    Nw = N.pivot_table(index=["filter", "file"], columns="variant", values="dstar", aggfunc="first")
    Ow = O.pivot_table(index=["filter", "file"], columns="combo", values="dstar", aggfunc="first")
    J = Nw.join(Ow, how="inner")
    rows = []
    for filt, g in J.groupby(level="filter"):
        lam = float(filt[1:4]) if filt[1:4].isdigit() else np.nan
        r = dict(filter=filt, n_paired=len(g))
        for curve in ("beam", "nobeam"):
            sel = g[f"manufacturer_{curve}_air"] - g[f"manufacturer_{curve}"]
            vac = g[f"manufacturer_{curve}_vac"] - g[f"manufacturer_{curve}_air"]
            r[f"old_{curve}"] = g[f"manufacturer_{curve}"].median()
            r[f"new_{curve}_vac"] = g[f"manufacturer_{curve}_vac"].median()
            r[f"sel_{curve}"], r[f"sel_{curve}_err"] = med_err(sel)
            r[f"vac_{curve}"] = np.nanmedian(vac)
        r["vac_expected"] = -air_to_vac_shift(lam) if np.isfinite(lam) else np.nan
        nn = N[N["filter"] == filt]
        r["n_frames_new"] = nn.file.nunique()
        r["n_stars_new"] = nn.drop_duplicates("file").n.median()
        r["mad_frame_new"] = mad(nn[nn.variant == "manufacturer_nobeam_vac"].dstar)
        oo = O[(O["filter"] == filt) & (O.combo == "manufacturer_nobeam")]
        r["mad_frame_old"] = mad(oo.dstar)
        rows.append(r)
    S = pd.DataFrame(rows)
    S["pivot"] = S["filter"].str[1:4].astype(int)
    S = S.sort_values("pivot"); S.to_csv(D / "dstar_v2_filter_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print("per filter [nm]: old = previous run (air, old selection); new = this run (vacuum, new selection);")
    print("sel = paired new(air) - old (star-selection effect); vac = new(vac) - new(air); mad = frame-to-frame scatter")
    print(S[["filter", "n_paired", "n_frames_new", "n_stars_new", "old_beam", "new_beam_vac", "old_nobeam", "new_nobeam_vac",
             "sel_beam", "sel_beam_err", "sel_nobeam", "sel_nobeam_err", "vac_nobeam", "vac_expected",
             "mad_frame_old", "mad_frame_new"]].round(2).to_string(index=False))
    # per copy (filter, unit) and per quarter, vacuum variants
    V = N[N.variant.isin(["manufacturer_beam_vac", "manufacturer_nobeam_vac"])].copy()
    V["quarter"] = [f"{t.datetime.year}Q{(t.datetime.month - 1) // 3 + 1}" for t in Time(V.mjd.to_numpy(), format="mjd")]
    CT = (V.groupby(["filter", "unit", "variant"])
            .agg(n=("dstar", "size"), dstar=("dstar", "median"), mad=("dstar", mad),
                 first=("quarter", "min"), last=("quarter", "max")).reset_index())
    CT["err"] = 1.2533 * CT["mad"] / np.sqrt(CT.n)
    CT.to_csv(D / "dstar_v2_copy_table.csv", index=False)
    CQ = (V.groupby(["filter", "unit", "quarter", "variant"]).agg(n=("dstar", "size"), dstar=("dstar", "median"),
                                                                  mad=("dstar", mad)).reset_index())
    CQ.to_csv(D / "dstar_v2_copy_quarter.csv", index=False)
    print("\nper copy (filter x unit), nobeam_vac: median delta* [nm] (n frames)")
    P = CT[CT.variant == "manufacturer_nobeam_vac"]
    print(P.assign(s=lambda d: d.dstar.round(2).astype(str) + " (" + d.n.astype(str) + ")")
           .pivot_table(index="filter", columns="unit", values="s", aggfunc="first").to_string())
    # figure
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
    x = S["pivot"].to_numpy()
    for curve, mk in (("nobeam", "o"), ("beam", "s")):
        ax[0].plot(x, S[f"old_{curve}"], mk + "--", color="0.6", ms=4, label=f"previous ({curve}, air, old stars)")
        ax[0].plot(x, S[f"new_{curve}_vac"], mk + "-", ms=4, label=f"new ({curve}, vacuum, SNR≥100 stars)")
        ax[1].errorbar(x, S[f"sel_{curve}"], S[f"sel_{curve}_err"], marker=mk, ms=4, capsize=0, label=f"{curve}")
    ax[0].axhline(0, color="k", lw=0.6); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=7)
    ax[0].set(xlabel="filter [nm]", ylabel="median δ* [nm]", title="(a) per-filter δ*: previous vs new")
    ax[1].axhline(0, color="k", lw=0.6); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=7)
    ax[1].set(xlabel="filter [nm]", ylabel="paired Δδ* new − old, same air curve [nm]",
              title="(b) star-selection effect (SNR≥100, no shape cuts, peak≤50k)")
    ax[2].plot(x, S.vac_nobeam, "o-", label="measured new(vac) − new(air)")
    ax[2].plot(x, S.vac_expected, "k--", label="−λ(n_air − 1)")
    ax[2].grid(alpha=0.3); ax[2].legend(fontsize=7)
    ax[2].set(xlabel="filter [nm]", ylabel="Δδ* [nm]", title="(c) air → vacuum correction")
    fig.suptitle("D1  Step 1: per-frame δ* re-solved on the 2024–2026 archive (corrected XP field colour term)", fontsize=10)
    fig.tight_layout(); fig.savefig(D / "D1_dstar_v2.png", dpi=125)
    print("saved")
