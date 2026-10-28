#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260910_zpcolor_c1_colorterm.py

Variant of figure C1 (CALSPEC observed - synthetic per star and filter): the lower panel shows the residual after the
header colour term (K_COLOR_APER_3_g-r, C_COLOR) has been applied to the standards instead of the MAG_AUTO residual.
Both panels come from colorterm_test/colorterm_stars.csv (260908_zpcolor_colorterm_test.py): per-star medians over the
cleaned frames, columns `none` (pipeline, APER_3) and `ct_gr` (+ colour term g-r), errors `none_err` / `ct_gr_err`.

  MPLCONFIGDIR=/tmp/ezphot-matplotlib /home/hhchoi1022/anaconda3/envs/py311/bin/python \\
      ezphot/analysis/260910_zpcolor_c1_colorterm.py [--out-dir ...]
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402
import pandas as pd                 # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
C = importlib.import_module("260910_effcurve_common")

FLAG_CALSPEC = 0.03


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(C.PREV_OUT))
    args = ap.parse_args()
    out = Path(args.out_dir)
    T = pd.read_csv(out / "colorterm_test" / "colorterm_stars.csv")
    T = T[T["filter"].isin(C.MEDIUM)]
    piv = {f: C.pivot_nm(C.FINE, C.load_nominal(f)) for f in T["filter"].unique()}
    stars = sorted(T["object"].unique(), key=lambda o: T[T.object == o]["gr_syn"].iloc[0])
    grs = {o: float(T[T.object == o]["gr_syn"].iloc[0]) for o in stars}
    cmap = plt.get_cmap("coolwarm"); norm = plt.Normalize(min(grs.values()), max(grs.values()))
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for ax, col, ecol, lab in zip(axes, ("none", "ct_gr"), ("none_err", "ct_gr_err"), ("pipeline (APER_3)", "+ header colour term (g−r)")):
        for f in sorted(piv, key=piv.get):
            d = T[(T["filter"] == f) & np.isfinite(T[col])]
            if not len(d): continue
            ax.plot(piv[f], float(np.median(d[col])), "_", color="k", ms=14, mew=2)
            for k, o in enumerate(stars):
                v = d[d.object == o]
                if not len(v): continue
                ax.errorbar(piv[f] + (k - len(stars) / 2) * 0.9, float(v[col].iloc[0]), yerr=float(max(v[ecol].iloc[0], 0.001)), fmt="o",
                            color=cmap(norm(grs[o])), ms=3.5, capsize=1.5, alpha=.9)
        ax.axhline(0, color="k", lw=.6); ax.axhspan(-FLAG_CALSPEC, FLAG_CALSPEC, color="grey", alpha=.12); ax.grid(alpha=.3)
        ax.set_ylabel(f"m_obs − m_syn(CALSPEC) [mag]\n{lab}"); ax.set_ylim(-0.15, 0.15)
    for o in stars: axes[0].plot([], [], "o", color=cmap(norm(grs[o])), label=f"{o} (g−r={grs[o]:+.2f})")
    axes[0].legend(fontsize=7, ncol=3, loc="upper right"); axes[1].set_xlabel("pivot wavelength [nm]")
    axes[0].set_title("CALSPEC standards: observed − synthetic per star (median ± error; black = all stars); top = pipeline ZP, bottom = + header colour term")
    fig.tight_layout(); fig.savefig(out / "figures" / "C1b_calspec_residual_colorterm.png", dpi=130); plt.close(fig)
    # numbers for the caption
    rows = []
    for f, g in T.groupby("filter"):
        da = g[g.is_da]; ot = g[~g.is_da]
        rows.append(dict(filter=f, n=len(g), med_none=float(g["none"].median()), med_ct=float(g["ct_gr"].median()), mad_none=C.R.mad(g["none"]), mad_ct=C.R.mad(g["ct_gr"]),
                         da_none=float(da["none"].median() - ot["none"].median()) if len(da) and len(ot) else np.nan, da_ct=float(da["ct_gr"].median() - ot["ct_gr"].median()) if len(da) and len(ot) else np.nan))
    S = pd.DataFrame(rows).sort_values("filter", key=lambda s: s.map(piv)); S.to_csv(out / "colorterm_test" / "c1b_summary.csv", index=False)
    print(S.round(3).to_string(index=False))
    C.log(f"figure written: {out / 'figures' / 'C1b_calspec_residual_colorterm.png'}")


if __name__ == "__main__":
    main()
