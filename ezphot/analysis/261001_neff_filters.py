#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
261001_neff_filters.py — per-filter effective index n_eff and the resulting DR500 f/3.03 converging-beam shift.

2026-10-07: measured n_eff for 15 filters (NEFF_MEASURED from 261001_calspec_manuf_beam) replace the linear relation, which
turned out to be wrong; the rest keep it provisionally until measured.  Previous outputs: ~/ezphot/log/20261001_neff_pre_neffmeas_20261007/.
n_eff(lam0) = 1.962 - 0.147 (lam0 - 480) / 100   [lam0 in nm]   (user's determination, 2026-10-01, second version;
the first, 2.2898 - 0.002651 (lam0 - 480), is discarded; its outputs are in ~/ezphot/log/20261001_neff_v1_2p2898/)
lam0 = centroid of the vendor normal-incidence curve.  The pupil-averaged beam (dr500_transmission, f/3.03,
obscuration 0.59) is applied to the vendor medium-band curves (Original_sources) with that n_eff, and compared with
the earlier fixed n_eff = 1.5 (260910_filter_beamshift.py).  Broad bands are left out: the relation is for the medium
bands, and at z (lam0 ~ 1014 nm) it gives n_eff < 1.
Outputs (~/ezphot/log/20261001_neff/):
  neff_filters.csv               lam0, n_eff, centroid / edge shifts (new and n_eff 1.5), relative shift
  filters_beamshift_neff/<f>.csv beam-shifted curves (lam, trans) on the vendor grid
  F11_neff_filters.png           n_eff per filter, beam shift per filter (new vs 1.5), m400 / m875 examples
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402
import pandas as pd                 # noqa: E402

SRC = Path("~/ezphot/config/common/transmission/Original_sources").expanduser()
OUT = Path("~/ezphot/log/20261001_neff").expanduser()
sys.path.insert(0, str(Path("~/ezphot/log/20260910_transmission_sources").expanduser()))
from dr500_transmission import dr500_transmission   # noqa: E402

NEFF0, NEFF1, LREF = 1.962, -0.00147, 480.0
# measured n_eff (2026-10-07, user) override the linear relation; same table as 261001_calspec_manuf_beam.NEFF_MEASURED
sys.path.insert(0, str(Path(__file__).parent))
import importlib                                     # noqa: E402
NEFF_MEASURED = {k: v[0] for k, v in importlib.import_module("261001_calspec_manuf_beam").NEFF_MEASURED.items()}
N_OLD, FNUM, EPS = 1.5, 3.03, 0.59


def n_eff(lam0):
    return NEFF0 + NEFF1 * (lam0 - LREF)


def centroid(l, t):
    return np.trapezoid(l * t, l) / np.trapezoid(t, l)


def half_edges(l, t):
    h = t.max() / 2.0
    above = np.where(t > h)[0]
    i, j = above[0], above[-1]
    lo = np.interp(h, [t[i - 1], t[i]], [l[i - 1], l[i]]) if i > 0 else l[0]
    hi = np.interp(h, [t[j + 1], t[j]], [l[j + 1], l[j]]) if j < len(l) - 1 else l[-1]
    return lo, hi


def main():
    (OUT / "filters_beamshift_neff").mkdir(parents=True, exist_ok=True)
    rows, curves = [], {}
    for p in sorted(SRC.glob("*.csv")):
        if not re.fullmatch(r"m\d{3}w?", p.stem):
            continue
        d = pd.read_csv(p)
        lam, tr = d.iloc[:, 0].to_numpy(float), d.iloc[:, 1].to_numpy(float)
        o = np.argsort(lam); lam, tr = lam[o], tr[o]
        l0 = centroid(lam, tr); n = NEFF_MEASURED.get(p.stem, n_eff(l0))
        te = dr500_transmission(lam, tr, n_eff=n, f_number=FNUM, obscuration=EPS)
        to = dr500_transmission(lam, tr, n_eff=N_OLD, f_number=FNUM, obscuration=EPS)
        pd.DataFrame(dict(lam=lam, trans=te)).to_csv(OUT / "filters_beamshift_neff" / f"{p.stem}.csv", index=False)
        curves[p.stem] = (lam, tr, te, to)
        b0, r0 = half_edges(lam, tr); b1, r1 = half_edges(lam, te)
        dc, dco = centroid(lam, te) - l0, centroid(lam, to) - l0
        rows.append(dict(filter=p.stem, lam0_nm=round(l0, 3), n_eff=round(n, 5),
                         n_eff_source="measured" if p.stem in NEFF_MEASURED else "provisional formula",
                         dcentroid_nm=round(dc, 4), dcentroid_pct=round(100 * dc / l0, 5),
                         dblue_edge_nm=round(b1 - b0, 4), dred_edge_nm=round(r1 - r0, 4),
                         dcentroid_neff1p5_nm=round(dco, 4), change_vs_1p5_nm=round(dc - dco, 4),
                         area_ratio=round(np.trapezoid(te, lam) / np.trapezoid(tr, lam), 5)))
    S = pd.DataFrame(rows).sort_values("lam0_nm")
    S.to_csv(OUT / "neff_filters.csv", index=False)
    pd.set_option("display.width", 200)
    print(S.to_string(index=False))
    l_cross = LREF + (N_OLD - NEFF0) / NEFF1
    print(f"n_eff = 1.5 at lam0 = {l_cross:.1f} nm; range {S.n_eff.min():.3f}-{S.n_eff.max():.3f}")

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.32, wspace=0.22)
    lg = np.linspace(360, 890, 200)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(lg, n_eff(lg), "-", color="0.6", lw=1.2,
            label="previous linear relation (provisional, unmeasured filters)")
    me = S.n_eff_source == "measured"
    ax.plot(S.lam0_nm[me], S.n_eff[me], "o", color="#C25B4E", ms=6, mec="k", mew=0.4, label="measured (15 filters, 2026-10-07)")
    ax.plot(S.lam0_nm[~me], S.n_eff[~me], "o", mfc="white", color="0.45", ms=4.5, label="not measured yet: provisional formula")
    for r in S.itertuples(index=False):
        ax.annotate(r.filter, (r.lam0_nm, r.n_eff), xytext=(3, 3), textcoords="offset points", fontsize=6.3, rotation=40,
                    color="#8B2E1F" if r.n_eff_source == "measured" else "0.45")
    ax.axhline(N_OLD, color="0.5", ls="--", lw=1.2, label="previous fixed n_eff = 1.5")
    ax.set(xlabel="λ0 = centroid of the vendor curve [nm]", ylabel="n_eff", xlim=(360, 890), ylim=(1.35, 2.18),
           title=f"(a) effective index per filter: measured {S.n_eff[me].min():.2f}–{S.n_eff[me].max():.2f}, no trend with λ0")
    ax.legend(fontsize=7, loc="lower left"); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(S.lam0_nm, S.dcentroid_neff1p5_nm, "s", mfc="none", color="0.45", ms=5, label="n_eff = 1.5 (−0.40 % λ)")
    ax.plot(S.lam0_nm[me], S.dcentroid_nm[me], "o", color="#C25B4E", ms=5.5, mec="k", mew=0.4, label="centroid, measured n_eff")
    ax.plot(S.lam0_nm[~me], S.dcentroid_nm[~me], "o", mfc="white", color="#C25B4E", ms=5, label="centroid, provisional n_eff")
    ax.plot(S.lam0_nm, S.dblue_edge_nm, "v", mfc="none", color="#1f77b4", ms=4, label="blue edge")
    ax.plot(S.lam0_nm, S.dred_edge_nm, "^", mfc="none", color="#9467bd", ms=4, label="red edge")
    ax2 = ax.twinx()
    ax2.plot(S.lam0_nm, S.dcentroid_pct, "-", color="#C25B4E", lw=0.8, alpha=0.6)
    ax2.set_ylabel("centroid shift [% of λ0] (red line)", color="#C25B4E"); ax2.tick_params(colors="#C25B4E")
    ax.set(xlabel="λ0 [nm]", ylabel="beam shift [nm]", xlim=(360, 890),
           title="(b) cone-averaged beam shift per filter (f/3.03, ε 0.59)")
    ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=0.3)

    for k, name in enumerate(["m386", "m769w"]):
        ax = fig.add_subplot(gs[1, k])
        lam, tr, te, to = curves[name]
        r = S[S["filter"] == name].iloc[0]
        ax.plot(lam, tr, "-", color="#888", lw=1.4, label="vendor (normal incidence)")
        ax.plot(lam, to, "--", color="0.3", lw=1.1, label=f"beam, n_eff 1.5 ({r.dcentroid_neff1p5_nm:+.2f} nm)")
        ax.plot(lam, te, "-", color="#C25B4E", lw=1.5, label=f"beam, measured n_eff {r.n_eff:.2f} ({r.dcentroid_nm:+.2f} nm)")
        lo = dr500_transmission(lam, tr, n_eff=n_eff(r.lam0_nm), f_number=FNUM, obscuration=EPS)
        ax.plot(lam, lo, ":", color="#2F5D9E", lw=1.3, label=f"beam, previous formula n_eff {n_eff(r.lam0_nm):.2f} ({centroid(lam, lo) - r.lam0_nm:+.2f} nm)")
        ax.set(xlim=(r.lam0_nm - 24, r.lam0_nm + 24), ylim=(0, 1.05), xlabel="wavelength [nm]", ylabel="transmission",
               title=f"({'cd'[k]}) {name}: λ0 {r.lam0_nm:.1f} nm, measured n_eff {r.n_eff:.2f}")
        ax.legend(loc="lower center", fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("F11  Per-filter n_eff (measured for 15 filters, 2026-10-07) and the converging-beam shift of the vendor curves", fontsize=12)
    fig.savefig(OUT / "F11_neff_filters.png", dpi=140, bbox_inches="tight")
    print("saved", OUT / "F11_neff_filters.png")


if __name__ == "__main__":
    main()
