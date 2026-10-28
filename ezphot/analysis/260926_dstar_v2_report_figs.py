#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260926_dstar_v2_report_figs.py — numbers table + figure D3 for the Step 1 report (260926_step1_delta_star.md).
Reads delta_star_v2_frames.csv (all images, vacuum variants) and dstar_v2_filter_summary.csv (paired new vs old).

  per filter: n images, fraction with dstar_err < 3, median dstar_err, spread of the median over cuts (all, err<3, err<1,
  colour span >= 0.8, n >= 50), sensitivity da/d(delta) (median of -a0/dstar), unit spread (units with >= 50 images),
  median dstar (beam / nobeam, all images), median a0 at delta = 0 (beam / nobeam), lab offset (doc 260921 section 2),
  paired selection effect and frame MAD (old / new)
  stable = frac_err3 >= 0.8 and cut_range <= 0.35 and |sensitivity| >= 3.4 and unit_spread <= 1.2 (m525: cut range 0.61)
Output (~/ezphot/log/20260926_delta_star_v2/): dstar_v2_report_table.csv, D3_dstar_filters.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

D = Path("/home/hhchoi1022/ezphot/log/20260926_delta_star_v2")
LAB = dict(m400=1.3, m425=1.5, m450=0.0, m475=-1.6, m500=1.0, m525=1.2, m550=0.0, m575=0.0, m600=0.5, m625=0.0,
           m650=-0.3, m675=-0.3, m700=1.0, m725=2.3, m750=0.0, m775=0.0, m800=-1.0, m825=1.0, m850=0.0, m875=2.2)
STABLE = ["m400", "m425", "m450", "m475", "m525", "m575", "m650", "m725", "m750"]


def cut_medians(g):
    span = g.bprp_hi - g.bprp_lo
    cuts = {"all": g.dstar, "err<3": g.dstar[g.dstar_err < 3], "err<1": g.dstar[g.dstar_err < 1],
            "span>=0.8": g.dstar[span >= 0.8], "n>=50": g.dstar[g.n >= 50]}
    return {k: v.median() for k, v in cuts.items() if len(v) >= 30}


if __name__ == "__main__":
    N = pd.read_csv(D / "delta_star_v2_frames.csv")
    N = N[np.isfinite(N.dstar)]
    S = pd.read_csv(D / "dstar_v2_filter_summary.csv").set_index("filter")
    rows = []
    for f in LAB:
        r = dict(filter=f, pivot=int(f[1:]), lab=LAB[f])
        for v in ("beam", "nobeam"):
            g = N[(N.variant == f"manufacturer_{v}_vac") & (N["filter"] == f)]
            r[f"dstar_{v}"] = g.dstar.median()
            r[f"a0_{v}"] = 1e3 * g.a0.median()
            if v == "beam":
                meds = cut_medians(g)
                u = g.groupby("unit").dstar.agg(["median", "size"]); u = u[u["size"] >= 50]["median"]
                r.update(n_images=len(g), frac_err3=(g.dstar_err < 3).mean(), med_err=g.dstar_err.median(),
                         cut_range=max(meds.values()) - min(meds.values()),
                         sens=1e3 * (-g.a0 / g.dstar)[np.abs(g.dstar) > 0.3].median(),
                         unit_min=u.min(), unit_max=u.max(), unit_spread=u.max() - u.min(), n_units=len(u),
                         frac_outside6=(np.abs(g.dstar) > 6).mean())
        r["beam_minus_lab"] = r["dstar_beam"] - r["lab"]
        r["nobeam_minus_lab"] = r["dstar_nobeam"] - r["lab"]
        for c in ("sel_nobeam", "sel_nobeam_err", "sel_beam", "sel_beam_err", "mad_frame_old", "mad_frame_new", "n_paired",
                  "old_nobeam", "new_nobeam_vac", "old_beam", "new_beam_vac"):
            r[c] = S.loc[f, c]
        r["stable"] = f in STABLE
        rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(D / "dstar_v2_report_table.csv", index=False)
    pd.set_option("display.width", 260)
    print(T.drop(columns=["pivot"]).round(2).to_string(index=False))
    x = T[T.stable & (T["filter"] != "m400")]
    for v in ("beam", "nobeam"):
        dd = x[f"{v}_minus_lab"]
        print(f"{v}: sky - lab over {len(x)} stable filters (no m400): mean {dd.mean():+.2f}, rms {np.sqrt((dd ** 2).mean()):.2f}, "
              f"r {np.corrcoef(x[f'dstar_{v}'], x.lab)[0, 1]:.2f}")

    # figure D3
    fig, ax = plt.subplots(2, 1, figsize=(12, 9), sharex=True, gridspec_kw=dict(height_ratios=[1.35, 1]))
    xs = T["pivot"].to_numpy()
    YL = (-8, 6)
    for v, mk, off, lab in (("beam", "o", -2.5, "sky δ* with beam (f/3 converging beam)"), ("nobeam", "s", 2.5, "sky δ* without beam")):
        y = T[f"dstar_{v}"].to_numpy()
        for st, alpha in ((True, 1.0), (False, 0.35)):
            m = (T.stable == st).to_numpy()
            ax[0].plot(xs[m] + off, np.clip(y[m], *YL), mk, ms=7, color="tab:blue" if v == "beam" else "tab:orange",
                       alpha=alpha, mfc=None if st else "none", label=f"{lab}" + ("" if st else " — not determined / unstable")
                       if (st or v == "beam") else None)
        low = y < YL[0]
        for xi in xs[low]:
            ax[0].annotate("", xy=(xi + off, YL[0] + 0.1), xytext=(xi + off, YL[0] + 0.9),
                           arrowprops=dict(arrowstyle="->", color="tab:blue" if v == "beam" else "tab:orange", alpha=0.5))
    um = T.stable.to_numpy()
    ax[0].vlines(xs - 2.5, T.unit_min, T.unit_max, color="tab:blue", lw=1.2, alpha=0.6)
    ax[0].plot(xs, T.lab, "kx", ms=9, mew=2, label="lab offset δ_lab (one per filter)")
    ax[0].axhline(0, color="k", lw=0.6); ax[0].grid(alpha=0.3); ax[0].set_ylim(*YL)
    for xi, f in zip(xs[~um], T["filter"][~um]):
        ax[0].axvspan(xi - 10, xi + 10, color="0.9", zorder=0)
    ax[0].text(675, YL[0] + 1.2, "m675 −13.7", fontsize=7, ha="center"); ax[0].text(800, YL[0] + 1.2, "m800 −12.5", fontsize=7, ha="center")
    ax[0].set_ylabel("median δ* over all images [nm]  (+ = real curve redder)")
    ax[0].set_title("(a) per-filter δ* (vacuum, all images; vertical bar = range of unit medians, beam) vs lab offset — grey band = not determined / unstable", fontsize=9)
    ax[0].legend(fontsize=8, loc="upper right", ncol=2)
    for v, mk, off, c in (("beam", "o", -2.5, "tab:blue"), ("nobeam", "s", 2.5, "tab:orange")):
        ax[1].plot(xs + off, T[f"a0_{v}"], mk + "-", ms=6, color=c, lw=1, label=f"{v}")
    ax[1].axhspan(-5, 5, color="tab:green", alpha=0.12, label="±5 mmag/mag")
    ax[1].axhline(0, color="k", lw=0.6); ax[1].grid(alpha=0.3); ax[1].set_ylim(-35, 70)
    ax[1].text(400, 66, f"m400 beam {T.a0_beam.iloc[0]:.0f}", fontsize=7, va="top");
    ax[1].set(xlabel="filter [nm]", ylabel="colour term at δ = 0 [mmag / mag]")
    ax[1].set_title("(b) colour term of the unshifted manufacturer curve: the 'common −13' offset of the no-beam curve mostly disappears with the beam; red filters keep −7…−13", fontsize=9)
    ax[1].legend(fontsize=8, loc="upper right")
    ax[1].set_xticks(xs); ax[1].set_xticklabels(T["filter"], rotation=45, fontsize=8)
    fig.suptitle("D3  Step 1: per-filter δ* from the field colour term (58,423 images, SNR ≥ 100 stars, corrected XP)", fontsize=10)
    fig.tight_layout(); fig.savefig(D / "D3_dstar_filters.png", dpi=125)
    print("saved", D / "D3_dstar_filters.png")
