#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260926_colorterm_versions_v2_summary.py — section 8 of 260921_transmission_curves.md redone with the Step 1 star
selection: per-filter median field colour term for four curve versions (vacuum), compared with the previous run
(260924_colorterm_versions.py: old star selection, air curves), plus the per-unit table.

Inputs : ~/ezphot/log/20260926_colorterm_time/colorterm_time_frames.csv (+ CT_full_suspect.csv from the time plot)
         ~/ezphot/log/20260924_colorterm_versions/colorterm_versions_frames.csv (old)
         ~/ezphot/log/20260926_delta_star_v2/delta_star_v2_frames.csv (new selection, air vs vacuum a0 for manufacturer)
         ~/ezphot/log/20260926_delta_star_v2/adelta_all_summary.csv (direct sensitivity, beam)
Outputs (~/ezphot/log/20260926_colorterm_time/): ct_versions_v2_table.csv, ct_versions_v2_units.csv, C2_colorterm_versions_v2.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time

D = Path("/home/hhchoi1022/ezphot/log/20260926_colorterm_time")
OLD = Path("/home/hhchoi1022/ezphot/log/20260924_colorterm_versions/colorterm_versions_frames.csv")
V2 = Path("/home/hhchoi1022/ezphot/log/20260926_delta_star_v2")
VERS = ["manufacturer_nobeam", "manufacturer_beam", "mhj_nobeam", "mhj_beam"]
OLDMAP = {"manufacturer_nobeam": "manufacturer_nobeam", "manufacturer_beam": "manufacturer_beam",
          "version_mhj_nobeam": "mhj_nobeam", "version_mhj_beam": "mhj_beam"}


def mad(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return 1.4826 * np.median(np.abs(v - np.median(v))) if len(v) else np.nan


if __name__ == "__main__":
    A = pd.read_csv(D / "colorterm_time_frames.csv")
    sus = pd.read_csv(D / "CT_full_suspect.csv")
    skey = set(zip(sus["filter"], sus.unit, sus.night))
    bad = np.array([k in skey for k in zip(A["filter"], A.unit, A.night)])
    print(f"images {A.file.nunique()}, suspect-night images removed {A[bad].file.nunique()}")
    A = A[~bad]
    O = pd.read_csv(OLD)
    O = O[O.combo.isin(OLDMAP)].assign(version=lambda d: d.combo.map(OLDMAP))
    # selection effect on a0 with the same (air) curve: new-selection air a0 (delta_star_v2) - old, paired images
    N2 = pd.read_csv(V2 / "delta_star_v2_frames.csv", usecols=["filter", "file", "variant", "a0"])
    Nw = N2.pivot_table(index=["filter", "file"], columns="variant", values="a0", aggfunc="first")
    Ow = O.pivot_table(index=["filter", "file"], columns="version", values="a0", aggfunc="first")
    J = Nw.join(Ow, how="inner")
    sens = pd.read_csv(V2 / "adelta_all_summary.csv").set_index("filter").slope6
    rows = []
    for f, g in A.groupby("filter"):
        r = dict(filter=f, pivot=int(f[1:]), n_images=g.file.nunique())
        for v in VERS:
            x = 1e3 * g[g.variant == v].a0
            r[f"new_{v}"], r[f"new_{v}_err"] = x.median(), 1.2533 * mad(x) / np.sqrt(len(x))
            r[f"old_{v}"] = 1e3 * O[(O["filter"] == f) & (O.version == v)].a0.median()
        jf = J.loc[f] if f in J.index.get_level_values(0) else None
        if jf is not None:
            for c in ("nobeam", "beam"):
                r[f"sel_{c}"] = 1e3 * (jf[f"manufacturer_{c}_air"] - jf[f"manufacturer_{c}"]).median()
                r[f"vac_{c}"] = 1e3 * (jf[f"manufacturer_{c}_vac"] - jf[f"manufacturer_{c}_air"]).median()
        r["closest_new"] = min(VERS, key=lambda v: abs(r[f"new_{v}"]))
        r["closest_old"] = min(VERS, key=lambda v: abs(r[f"old_{v}"]) if np.isfinite(r[f"old_{v}"]) else 1e9)
        r["sens_beam"] = sens.get(f, np.nan)
        rows.append(r)
    T = pd.DataFrame(rows).sort_values("pivot")
    T.to_csv(D / "ct_versions_v2_table.csv", index=False)
    pd.set_option("display.width", 250)
    print("median colour term [mmag/mag]; new = SNR>=100 selection, vacuum; old = section 8 (old selection, air)")
    print(T[["filter", "n_images"] + [f"old_{v}" for v in VERS] + [f"new_{v}" for v in VERS] +
            ["sel_nobeam", "sel_beam", "vac_nobeam", "vac_beam", "closest_old", "closest_new"]].round(1).to_string(index=False))
    # per unit (manufacturer_beam): level and offset from the filter median in nm (direct sensitivity)
    B = A[A.variant == "manufacturer_beam"].copy()
    B["ym"] = [t.strftime("%Y-%m") for t in Time(B.mjd.to_numpy(), format="mjd").datetime]
    U = (B.groupby(["filter", "unit"]).agg(n=("a0", "size"), a=("a0", lambda v: 1e3 * np.median(v)),
                                           first=("ym", "min"), last=("ym", "max")).reset_index())
    U = U[U.n >= 20]
    U["a_minus_filter"] = U.a - U["filter"].map(T.set_index("filter").new_manufacturer_beam)
    U["shift_nm_equiv"] = -U.a_minus_filter / U["filter"].map(sens)      # delta* offset of the unit (+ = redder)
    U.to_csv(D / "ct_versions_v2_units.csv", index=False)
    print("\nper unit (manufacturer_beam, units with >= 20 images): median a [mmag/mag], dates, equivalent shift [nm]")
    print(U.round(2).to_string(index=False))
    # figure C2
    fig, ax = plt.subplots(figsize=(13, 5.8))
    x = T["pivot"].to_numpy()
    col = dict(manufacturer_nobeam="tab:orange", manufacturer_beam="tab:blue", mhj_nobeam="tab:red", mhj_beam="tab:purple")
    off = dict(manufacturer_nobeam=-4.5, manufacturer_beam=-1.5, mhj_nobeam=1.5, mhj_beam=4.5)
    for v in VERS:
        ax.plot(x + off[v], T[f"new_{v}"], "o-", ms=6, lw=1.2, color=col[v], label=v)
    ax.axhspan(-5, 5, color="tab:green", alpha=0.10)
    ax.axhline(0, color="k", lw=0.7); ax.grid(alpha=0.3); ax.set_ylim(-40, 80)
    for k, v in enumerate(VERS):
        y = T[f"new_{v}"].to_numpy()
        for xi, yi in zip(x, y):
            if yi > 80:
                ax.annotate(f"{yi:.0f}", (xi + 6, 77 - 6 * k), fontsize=7, ha="left", va="top", color=col[v])
    ax.set_xticks(x); ax.set_xticklabels(T["filter"], rotation=45, fontsize=8)
    ax.set(ylabel="median field colour term a [mmag / mag BP−RP]",
           title="C2  Field colour term per curve version (SNR ≥ 100 stars, corrected Gaia XP, vacuum curves)")
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    fig.tight_layout(); fig.savefig(D / "C2_colorterm_versions_v2.png", dpi=125)
    print("saved", D / "C2_colorterm_versions_v2.png")
