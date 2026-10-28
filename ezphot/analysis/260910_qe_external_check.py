#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260910_qe_external_check.py

Compare our IMX455 camera QE (Original_sources/QE.csv) with published external measurements:
  - Gill et al. 2022 (SPIE 12191, arXiv:2207.13052): low-cost absolute QE system, peak 93% @ 480 nm, 42% @ 700, 10% @ 900
  - Alarcon et al. 2023 (PASP 135, arXiv:2302.03700): QHY600M monochromator scan 350-1100 nm,
    peak 80% @ 475 nm, 40% @ 700, 10% @ 900, systematic ~2%; QHYCCD curve is ~9% higher overall
  - QHYCCD published curve: peak 92% @ 450 nm (as quoted by Alarcon et al.)
  - Moravian C3-61000 (7DT camera) spec: peak QE 91% (no public curve)

Outputs figures/F7_qe_external.png and qe_external_anchors.csv in the 20260910_transmission_sources log dir.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402
import pandas as pd                 # noqa: E402

SRC = Path("~/ezphot/config/common/transmission/Original_sources/QE.csv").expanduser()
OUT = Path("~/ezphot/log/20260910_transmission_sources").expanduser()

EXT = {  # (marker, color, [(lam, qe), ...])
    "Gill+2022 (abs. QE bench)": ("D", "#d62728", [(480, 0.93), (700, 0.42), (900, 0.10)]),
    "Alarcon+2023 (QHY600M)": ("s", "#2ca02c", [(475, 0.80), (700, 0.40), (900, 0.10)]),
    "QHYCCD published": ("^", "#9467bd", [(450, 0.92)]),
    "Moravian C3-61000 spec": ("v", "#ff7f0e", [(481, 0.91)]),   # peak value only, wavelength unspecified
}


def main():
    d = pd.read_csv(SRC)
    lam, qe = d.iloc[:, 0].to_numpy(float), d.iloc[:, 1].to_numpy(float)

    anchors = [450, 475, 480, 500, 600, 700, 800, 900]
    ours = {a: float(np.interp(a, lam, qe)) for a in anchors}
    peak_l, peak_q = lam[np.argmax(qe)], qe.max()
    rows = [dict(source="ours (QE.csv)", lam_nm=peak_l, qe=peak_q, note="peak")]
    rows += [dict(source="ours (QE.csv)", lam_nm=a, qe=round(ours[a], 4), note="") for a in anchors]
    print("ours: peak %.3f @ %.0f nm |" % (peak_q, peak_l),
          " ".join(f"{a}:{ours[a]:.3f}" for a in anchors))
    for name, (_, _, pts) in EXT.items():
        for l0, q0 in pts:
            rel = q0 / (0.93 if "Gill" in name else 0.80 if "Alarcon" in name else q0)
            rows.append(dict(source=name, lam_nm=l0, qe=q0, note=f"norm {rel:.3f}"))
            print(f"{name}: {q0:.2f} @ {l0} nm  (ours there {np.interp(l0, lam, qe):.3f})")
    pd.DataFrame(rows).to_csv(OUT / "qe_external_anchors.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, norm in zip(axes, [False, True]):
        scale = 1.0 / peak_q if norm else 1.0
        ax.plot(lam, qe * scale, "-", color="#1f77b4", lw=1.2,
                label=f"ours: QE.csv (peak {peak_q:.3f} @ {peak_l:.0f} nm)")
        ax.plot(lam, qe * scale, "o", color="#1f77b4", ms=2.5)
        for name, (mk, c, pts) in EXT.items():
            p = np.array(pts, float)
            pk = {"Gill+2022 (abs. QE bench)": 0.93, "Alarcon+2023 (QHY600M)": 0.80,
                  "QHYCCD published": 0.92, "Moravian C3-61000 spec": 0.91}[name]
            ax.plot(p[:, 0], p[:, 1] * (1.0 / pk if norm else 1.0), mk, color=c, ms=9,
                    mec="k", mew=0.5, ls="none", label=name)
        ax.set(xlim=(340, 1000), xlabel="wavelength [nm]",
               ylabel="QE / peak" if norm else "absolute QE",
               title="normalized to each peak (shape comparison)" if norm else "absolute scale")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=9)
    fig.suptitle("F7  our IMX455 QE vs published measurements")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "F7_qe_external.png", dpi=150)
    print("saved ->", OUT / "figures" / "F7_qe_external.png")


if __name__ == "__main__":
    main()
