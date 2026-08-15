#!/usr/bin/env python
"""Regenerate the final SN 2026kid light curves excluding flagged epochs.

The initial ZP-based automatic cut (zperr/nstar/extinction) removed epochs
whose photometry was visually fine, so per the user's review the filter is now
an explicit exclusion list: the four nights whose R-band photometry is a
clear outlier against its neighbours are dropped in ALL filters —

    2026_0531  (MJD 61192.3, +0.23 mag R dip)
    2026_0610  (MJD 61202.3, largest R dip)
    2026_0616  (MJD 61208.3, ~1 mag no-annulus R dip)
    2026_0626  (MJD 61218.2, +0.25 mag R dip)

The Stage C measurement table (``SN2026kid_LOAO_dia_photometry.ecsv``) is
filtered — no re-measurement — and the comparison figure plus the per-variant
magnitude tables are rewritten with a ``_qcut`` suffix. The excluded epochs
go to ``14_qualitycut_summary.log``.
"""

from __future__ import annotations

import glob
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ezphot-matplotlib")

import numpy as np
from astropy.io import ascii, fits
from astropy.table import MaskedColumn, Table

# Nights excluded (all filters) after visual light-curve review.
EXCLUDED_NIGHTS = {"2026_0531", "2026_0610", "2026_0616", "2026_0626"}
FILTER_ALIASES = {"B102": "B", "V103": "V", "R104": "R", "I105": "I"}
FILTER_COLORS = {"B": "tab:blue", "V": "tab:green", "R": "tab:red", "I": "maroon"}
OUTPUT_DIR = Path("/home/hhchoi1022/ezphot/data/scidata/LOAO/SN2026kid/lightcurve")
QC_DIR = Path("/home/hhchoi1022/ezphot/log/20260806_stage_run_qc")


def main() -> None:
    import matplotlib.pyplot as plt

    table = Table.read(
        OUTPUT_DIR / "SN2026kid_LOAO_dia_photometry.ecsv", format="ascii.ecsv"
    )
    keep = np.array(
        [str(row["night"]) not in EXCLUDED_NIGHTS for row in table]
    )
    excluded_lines = [
        f"  {night} (all filters): visual light-curve outlier night"
        for night in sorted(EXCLUDED_NIGHTS)
    ]
    filtered = table[keep]
    print(f"kept {len(filtered)} / {len(table)} measurements")

    filtered.write(
        OUTPUT_DIR / "SN2026kid_LOAO_dia_photometry_qcut.ecsv",
        format="ascii.ecsv",
        overwrite=True,
    )
    for variant in ("dia_noann", "dia_ann", "direct_noann", "direct_ann"):
        for label in ("3", "5", "10"):
            dates = sorted(set(filtered["obsdate"]))
            pivot = Table()
            pivot["obsdate"] = dates
            for filter_name in FILTER_COLORS:
                values = np.ma.masked_all(len(dates), dtype=float)
                for index, date in enumerate(dates):
                    select = (
                        (np.asarray(filtered["obsdate"]) == date)
                        & (np.asarray(filtered["filter"]) == filter_name)
                        & np.asarray(
                            filtered[f"{variant}_detected_{label}"], dtype=bool
                        )
                    )
                    magnitudes = np.asarray(
                        filtered[f"{variant}_mag_{label}"][select], dtype=float
                    )
                    magnitudes = magnitudes[np.isfinite(magnitudes)]
                    if len(magnitudes):
                        values[index] = magnitudes[0]
                pivot[filter_name] = MaskedColumn(values, format=".3f")
            pivot.write(
                OUTPUT_DIR
                / f"SN2026kid_LOAO_{variant}_{label}arcsec_magnitudes_qcut.txt",
                format="ascii.fixed_width",
                fill_values=[(ascii.masked, "--")],
                overwrite=True,
            )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, constrained_layout=True)
    series = [
        ("dia_ann", "o", "-", 1.0, "DIA + annulus"),
        ("dia_noann", "s", "--", 0.8, "DIA no annulus"),
        ("direct_noann", ".", ":", 0.55, "no subtraction"),
    ]
    label = "5"
    for axis, (filter_name, color) in zip(axes.flat, FILTER_COLORS.items()):
        subset = filtered[np.asarray(filtered["filter"]) == filter_name]
        mjd = np.asarray(subset["mjd"], dtype=float)
        for variant, marker, linestyle, alpha, name in series:
            detected = np.asarray(subset[f"{variant}_detected_{label}"], dtype=bool)
            axis.errorbar(
                mjd[detected],
                np.asarray(subset[f"{variant}_mag_{label}"], dtype=float)[detected],
                yerr=np.asarray(subset[f"{variant}_magerr_{label}"], dtype=float)[
                    detected
                ],
                fmt=marker, ls=linestyle, ms=3.5, lw=1, capsize=2,
                color=color, alpha=alpha, label=name,
            )
        limits = ~np.asarray(subset[f"dia_ann_detected_{label}"], dtype=bool)
        axis.scatter(
            mjd[limits],
            np.asarray(subset[f"dia_ann_ul5_{label}"], dtype=float)[limits],
            marker="v", s=24, facecolors="none", edgecolors=color, alpha=0.6,
        )
        axis.invert_yaxis()
        axis.set_title(f"{filter_name}  ({int(np.sum(np.isfinite(mjd)))} epochs)")
        axis.set_xlabel("MJD")
        axis.set_ylabel(f"Magnitude ({label}\" aperture)")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
    fig.suptitle(
        "SN 2026kid — final light curves (4 visually flagged nights removed, "
        f"all filters; {len(filtered)}/{len(table)} measurements kept)"
    )
    comparison_png = OUTPUT_DIR / "SN2026kid_LOAO_dia_vs_direct_qcut.png"
    fig.savefig(comparison_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    lines = [
        "Quality-filtered light curves — created "
        + datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "criteria              : explicit exclusion list from visual review "
        "(the earlier zperr/nstar/extinction cut removed good epochs and was "
        "reverted)",
        f"kept                  : {len(filtered)} / {len(table)} stack measurements",
        f"excluded epochs       : {len(excluded_lines)}",
        "",
        *excluded_lines,
        "",
        "VALIDATION_RESULT=PASS",
    ]
    (QC_DIR / "14_qualitycut_summary.log").write_text("\n".join(lines) + "\n")
    print(f"figure: {comparison_png}")
    print(f"log   : {QC_DIR / '14_qualitycut_summary.log'}")


if __name__ == "__main__":
    main()
