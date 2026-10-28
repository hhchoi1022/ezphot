#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260910_optics_external_check.py

Check the components of Original_sources/optics.efficiency.dr350.dr500.csv (reverse-engineered as
total = bulk(H-K9LA x H-LaK4L x H-FK95N, 10 mm each) x VIS^5 x Al^2) against external data:
  - CDGM datasheet internal transmittance tau(10mm) for the three glasses (cdgmgd.com/webapp/pdf/<glass>.pdf),
    which also extends below 400 nm where the file has no data;
  - bare-aluminium reflectivity computed from Rakic 1998 (BB model) n,k (refractiveindex.info database,
    copy at <log>/al_rakic_bb.yml);
  - typical enhanced-aluminium curves (pgo SEA-VIS / SEA-NIR, digitized from their published 45-deg plot);
  - PlaneWave's stated "Enhanced aluminium - 96%" for the DR500 mirrors.

Outputs figures/F8_optics_external.png and optics_external_summary.csv in the 20260910_transmission_sources log dir.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402
import pandas as pd                 # noqa: E402

SRC = Path("~/ezphot/config/common/transmission/Original_sources").expanduser()
OUT = Path("~/ezphot/log/20260910_transmission_sources").expanduser()

# CDGM datasheet tau(10 mm); the <400 nm rows are what the optics file lacks
CDGM = {
    "H-K9LA": [(340, .923), (350, .961), (360, .980), (370, .990), (380, .991), (390, .997), (400, .997),
               (450, .997), (500, .997), (550, .997), (600, .998), (700, .998), (800, .998), (900, .998), (1000, .998)],
    "H-LaK4L": [(340, .630), (350, .760), (360, .847), (370, .908), (380, .943), (390, .963), (400, .974),
                (420, .984), (440, .988), (460, .991), (480, .994), (500, .995), (550, .996), (600, .997),
                (700, .997), (800, .998), (900, .998), (1000, .998)],
    "H-FK95N": [(340, .961), (350, .983), (360, .988), (370, .994), (380, .995), (390, .996), (400, .996),
                (450, .996), (500, .998), (600, .998), (700, .998), (800, .998), (900, .998), (1000, .998)],
}
# pgo SEA enhanced-aluminium "principle curves" (45 deg), digitized from their plot
SEA_VIS = [(400, .948), (450, .968), (500, .973), (550, .972), (600, .965), (650, .952), (700, .935)]
SEA_NIR = [(600, .927), (650, .941), (700, .945), (750, .947), (800, .948), (850, .951), (900, .953),
           (1000, .956), (1100, .958)]


def bare_al():
    lam, n, k = [], [], []
    txt = (OUT / "al_rakic_bb.yml").read_text().splitlines()
    for line in txt[txt.index("    data: |") + 1:]:
        p = line.split()
        if len(p) != 3:
            break
        lam.append(float(p[0]) * 1000); n.append(float(p[1])); k.append(float(p[2]))
    lam, n, k = map(np.array, (lam, n, k))
    R = ((n - 1) ** 2 + k ** 2) / ((n + 1) ** 2 + k ** 2)
    sel = (lam > 300) & (lam < 1200)
    return lam[sel], R[sel]


def main():
    opt = pd.read_csv(SRC / "optics.efficiency.dr350.dr500.csv", sep=r"\s+", comment="#", skiprows=11)
    opt.columns = ["nm", "H-K9LA", "H-LaK4L", "H-FK95N", "VIS", "Al", "Total_pct", "total"]
    al_lam, al_R = bare_al()

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    # P1 mirror coating
    ax = axes[0, 0]
    ax.plot(opt.nm, opt.Al, "ks-", ms=6, lw=2, label="file: 'Al coating'")
    ax.plot(al_lam, al_R, "-", color="#888", lw=1.5, label="bare Al (Rakic 1998 n,k)")
    for pts, c, lab in [(SEA_VIS, "#1f77b4", "pgo SEA-VIS enhanced Al (typ., 45\N{DEGREE SIGN})"),
                        (SEA_NIR, "#d62728", "pgo SEA-NIR enhanced Al (typ., 45\N{DEGREE SIGN})")]:
        p = np.array(pts)
        ax.plot(p[:, 0], p[:, 1], "o--", color=c, ms=5, lw=1, label=lab)
    ax.axhline(0.96, color="#2ca02c", ls=":", lw=1.5, label='PlaneWave claim "enhanced Al 96%"')
    ax.set(xlim=(340, 1100), ylim=(0.75, 1.0), xlabel="wavelength [nm]", ylabel="reflectivity per mirror",
           title="mirror coating: file vs external")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # P2 glass bulk
    ax = axes[0, 1]
    cols = {"H-K9LA": "#1f77b4", "H-LaK4L": "#d62728", "H-FK95N": "#2ca02c"}
    for g, c in cols.items():
        p = np.array(CDGM[g])
        ax.plot(p[:, 0], p[:, 1], "-", color=c, lw=1.2, label=f"CDGM datasheet {g} tau(10mm)")
        ax.plot(opt.nm, opt[g], "o", color=c, ms=7, mfc="none")
    ax.plot([], [], "ko", mfc="none", label="file values (7 nodes)")
    ax.set(xlim=(340, 1050), ylim=(0.6, 1.005), xlabel="wavelength [nm]", ylabel="internal transmittance (10 mm)",
           title="lens glasses: file = CDGM tau(10mm) exactly; datasheet extends < 400 nm")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)

    # P3 AR coating
    ax = axes[1, 0]
    ax.plot(opt.nm, opt.VIS, "ks-", ms=6, lw=2, label="file: 'VIS' AR per surface")
    ax.axvspan(400, 700, color="#2ca02c", alpha=0.12, label="PlaneWave spec: avg R < 0.5% (400-700)")
    ax.axhline(0.995, color="#2ca02c", ls=":", lw=1.5)
    ax.set(xlim=(340, 1050), ylim=(0.85, 1.005), xlabel="wavelength [nm]", ylabel="transmission per surface",
           title="broadband AR: in-band consistent with spec; > 700 nm unverified")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # P4 impact on total
    ax = axes[1, 1]
    ax.plot(opt.nm, opt.total, "ks-", ms=6, lw=2, label="file total")
    sea = np.interp(opt.nm, *np.array(SEA_VIS + SEA_NIR[3:]).T)
    alt = opt.total / opt.Al ** 2 * sea ** 2
    ax.plot(opt.nm, alt, "o--", color="#d62728", ms=6, label="total if mirrors = pgo enhanced Al")
    bare = np.interp(opt.nm, al_lam, al_R)
    ax.plot(opt.nm, opt.total / opt.Al ** 2 * bare ** 2, "o--", color="#888", ms=5, label="total if mirrors = bare Al")
    for x, y0, y1 in zip(opt.nm, opt.total, alt):
        ax.annotate(f"{(y1/y0-1)*100:+.0f}%", (x, y1 + 0.015), ha="center", fontsize=8, color="#d62728")
    ax.set(xlim=(360, 1040), ylim=(0.35, 1.0), xlabel="wavelength [nm]", ylabel="optics total",
           title="throughput impact of the mirror-coating assumption")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle("F8  optics components vs external data (glasses: CDGM datasheets; mirrors: Rakic/pgo/PlaneWave)")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "F8_optics_external.png", dpi=150)

    rows = []
    for r in opt.itertuples():
        rows.append(dict(nm=r.nm, file_Al=r.Al, bare_Al=round(float(np.interp(r.nm, al_lam, al_R)), 3),
                         sea_enhanced=round(float(np.interp(r.nm, *np.array(SEA_VIS + SEA_NIR[3:]).T)), 3),
                         file_total=r.total, total_enhanced=round(float(alt[r.Index]), 3)))
    pd.DataFrame(rows).to_csv(OUT / "optics_external_summary.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print("saved ->", OUT / "figures" / "F8_optics_external.png")


if __name__ == "__main__":
    main()
