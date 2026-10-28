#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260910_zpcolor_c2_routes.py

Corrected version of figure C2: the colour dependence of the calibration measured by two independent routes, put on the
SAME sign convention.  CALSPEC route: slope of (m_obs - m_syn) across the 13 standards vs their synthetic g-r.
Field-star route: the refit colour term a_gr is d(ref - obs)/d(g-r) (zero-point residual), i.e. minus the slope of
(obs - ref); so -a_gr is the quantity comparable with the CALSPEC slope.  The original C2 plotted +a_gr, which made
the two routes look opposite although they agree (2026-09-10).

Reads summary_by_filter.csv of the 2026-09-08 analysis; writes figures/C2b_colour_routes_same_sign.png and
colorterm_test/c2b_summary.csv.
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(C.PREV_OUT))
    args = ap.parse_args(); out = Path(args.out_dir)
    S = pd.read_csv(out / "summary_by_filter.csv"); S = S[S["filter"].str.match(r"^m\d{3}w?$")].copy()
    S["field_minus_a_gr"] = -S["corr_a_gr_median"]; S["diff"] = S["calspec_slope_gr"] - S["field_minus_a_gr"]
    grid = S["filter"].str.match(r"^m\d{3}$")
    fig, ax = plt.subplots(figsize=(12, 5.5))
    m = np.isfinite(S["calspec_slope_gr"])
    ax.errorbar(S.pivot_nm[m] - 2, S.calspec_slope_gr[m], yerr=S.calspec_slope_gr_err[m], fmt="D", color="tab:red", ms=5, capsize=2, label="CALSPEC route: d(m_obs − m_syn)/d(g−r) across the 13 standards")
    ax.plot(S.pivot_nm[grid] + 2, S.field_minus_a_gr[grid], "o", color="tab:blue", ms=5, label="field-star route: −a_gr = d(obs − ref)/d(g−r) of the XP field stars (CORR_LAMOST)")
    ax.plot(S.pivot_nm[~grid] + 2, S.field_minus_a_gr[~grid], "o", mfc="none", color="tab:blue", ms=5, label="same, non-grid filters (tiles only)")
    for f, x, y in zip(S["filter"][m], S.pivot_nm[m], S.calspec_slope_gr[m]):
        if f in ("m400", "m675"): ax.annotate(f, (x - 2, y), xytext=(0, -14), textcoords="offset points", fontsize=8, ha="center", color="tab:red")
    ax.axhline(0, color="k", lw=.6); ax.axhspan(-0.02, 0.02, color="grey", alpha=.12); ax.grid(alpha=.3); ax.set_ylim(-0.08, 0.08)
    ax.set_xlabel("pivot wavelength [nm]"); ax.set_ylabel("colour dependence of (observed − reference) [mag per mag of g−r]")
    g = S[grid & m & ~S["filter"].isin(["m400", "m675"])]
    ax.set_title(f"Two independent routes, same sign convention: median difference {g['diff'].median():+.3f}, rms {np.sqrt((g['diff']**2).mean()):.3f} mag/mag ({len(g)} filters; m400, m675 excluded)", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(out / "figures" / "C2b_colour_routes_same_sign.png", dpi=130); plt.close(fig)
    S[["filter", "pivot_nm", "calspec_slope_gr", "calspec_slope_gr_err", "corr_a_gr_median", "field_minus_a_gr", "diff"]].to_csv(out / "colorterm_test" / "c2b_summary.csv", index=False)
    C.log(f"C2b written; median diff {g['diff'].median():+.3f}, rms {np.sqrt((g['diff']**2).mean()):.3f} over {len(g)} filters")


if __name__ == "__main__":
    main()
