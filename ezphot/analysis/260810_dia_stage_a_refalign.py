#!/usr/bin/env python
"""DIA Stage A — verify PS1 reference alignment against the LOAO stack grid.

Checks two things per reference band (g/r/i):

1. Grid agreement: map stack pixel positions to sky with the stack WCS and
   back to reference pixels with the reference WCS; the offset field should
   be far below a pixel everywhere if the two grids are the same.
2. Empirical star agreement: detect stars on the reference with SEP and match
   them by sky position to a deep LOAO stack SExtractor catalog; report the
   median offset in stack pixels.

Writes ``11_refalign_qc.png`` and ``11_refalign_summary.log`` into the shared
QC directory. Verdict is PASS when every band's median star offset and grid
offset are below 0.3 stack pixels — then the references can be used directly
by HOTPANTS without reprojection.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ezphot-matplotlib")

import numpy as np
import sep
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

REF_DIR = Path("/home/hhchoi1022/ezphot/data/scidata/LOAO/SN2026kid/reference_images")
QC_DIR = Path("/home/hhchoi1022/ezphot/log/20260806_stage_run_qc")
# Deep stacks used for the star comparison, one per reference band.
COMPARISON_STACKS = {
    "g": "B102/stack.SN2026kid.2026_0610.B102.fits",
    "r": "R104/stack.SN2026kid.2026_0610.R104.fits",
    "i": "I105/stack.SN2026kid.2026_0610.I105.fits",
}
STACK_BASE = Path(
    "/home/hhchoi1022/ezphot/data/scidata/LOAO/LOAO_E2V_2x2/SN2026kid/1.0-m KASI"
)
PIXEL_TOLERANCE = 0.3
MATCH_RADIUS_ARCSEC = 2.0


def grid_offsets(stack_wcs: WCS, reference_wcs: WCS, size: int = 2048) -> np.ndarray:
    grid = np.linspace(64, size - 64, 9)
    xs, ys = np.meshgrid(grid, grid)
    sky = stack_wcs.pixel_to_world(xs.ravel(), ys.ravel())
    ref_x, ref_y = reference_wcs.world_to_pixel(sky)
    return np.hypot(ref_x - xs.ravel(), ref_y - ys.ravel())


def star_offsets(reference_path: Path, stack_path: Path) -> np.ndarray:
    data = fits.getdata(reference_path, memmap=False).astype(np.float32)
    data = np.ascontiguousarray(np.nan_to_num(data, nan=0.0))
    background = sep.Background(data)
    detections = sep.extract(
        data - background.back(), 10.0, err=background.globalrms, minarea=5
    )
    detections = detections[np.argsort(detections["flux"])[::-1][:600]]
    reference_wcs = WCS(fits.getheader(reference_path, memmap=False))
    ref_sky = reference_wcs.pixel_to_world(detections["x"], detections["y"])

    catalog = Table.read(Path(str(stack_path) + ".cat"), format="ascii")
    quality = (
        (np.asarray(catalog["FLAGS"], dtype=float) == 0)
        & (np.asarray(catalog["CLASS_STAR"], dtype=float) > 0.5)
    )
    catalog = catalog[quality]
    stack_sky = SkyCoord(
        np.asarray(catalog["X_WORLD"], dtype=float),
        np.asarray(catalog["Y_WORLD"], dtype=float),
        unit="deg",
    )
    index, separation, _ = ref_sky.match_to_catalog_sky(stack_sky)
    matched = separation.arcsec < MATCH_RADIUS_ARCSEC
    # Offsets in stack pixels (0.794 "/pix), as signed dx/dy on the shared grid.
    stack_wcs = WCS(fits.getheader(stack_path, memmap=False))
    ref_x, ref_y = stack_wcs.world_to_pixel(ref_sky[matched])
    cat_x = np.asarray(catalog["X_IMAGE"], dtype=float)[index[matched]] - 1
    cat_y = np.asarray(catalog["Y_IMAGE"], dtype=float)[index[matched]] - 1
    return np.column_stack([ref_x - cat_x, ref_y - cat_y])


def main() -> None:
    import argparse

    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ref-suffix",
        default="",
        help='Reference filename infix, e.g. ".aligned" to verify aligned files',
    )
    args = parser.parse_args()

    QC_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    lines = [
        "DIA Stage A (reference alignment) QC summary — created "
        + datetime.now(timezone.utc).isoformat(timespec="seconds"),
        f"references: {REF_DIR}",
        f"tolerance : {PIXEL_TOLERANCE} stack pixels (0.794\"/pix)",
        "",
    ]
    all_ok = True
    for axis, (band, stack_rel) in zip(axes, COMPARISON_STACKS.items()):
        reference_path = REF_DIR / f"ref.PS1.{band}{args.ref_suffix}.fits"
        stack_path = STACK_BASE / stack_rel
        stack_wcs = WCS(fits.getheader(stack_path, memmap=False))
        reference_wcs = WCS(fits.getheader(reference_path, memmap=False))
        grid = grid_offsets(stack_wcs, reference_wcs)
        offsets = star_offsets(reference_path, stack_path)
        dx_median = float(np.median(offsets[:, 0]))
        dy_median = float(np.median(offsets[:, 1]))
        radial_median = float(np.median(np.hypot(*offsets.T)))
        ok = grid.max() < PIXEL_TOLERANCE and radial_median < PIXEL_TOLERANCE
        all_ok &= ok

        axis.scatter(offsets[:, 0], offsets[:, 1], s=6, alpha=0.4, color="tab:blue")
        axis.axhline(0, color="k", lw=0.6)
        axis.axvline(0, color="k", lw=0.6)
        limit = max(1.0, np.percentile(np.abs(offsets), 99))
        axis.set_xlim(-limit, limit)
        axis.set_ylim(-limit, limit)
        axis.set_title(
            f"PS1 {band} vs {Path(stack_rel).name}\n"
            f"stars={len(offsets)}  median dx={dx_median:+.3f} dy={dy_median:+.3f} px",
            fontsize=9,
        )
        axis.set_xlabel("dx [stack pixel]")
        axis.set_ylabel("dy [stack pixel]")
        axis.grid(alpha=0.2)

        lines += [
            f"[{band}] grid offset max {grid.max():.4f} px | star matches "
            f"{len(offsets)} | median dx {dx_median:+.3f} dy {dy_median:+.3f} "
            f"radial {radial_median:.3f} px -> {'OK' if ok else 'REPROJECT NEEDED'}",
        ]

    verdict = "PASS" if all_ok else "FAIL"
    lines += ["", f"VALIDATION_RESULT={verdict}"]
    fig.suptitle(
        "DIA Stage A — PS1 reference vs LOAO stack grid alignment (star offsets)"
    )
    fig.savefig(QC_DIR / "11_refalign_qc.png", dpi=140)
    plt.close(fig)
    (QC_DIR / "11_refalign_summary.log").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
