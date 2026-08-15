#!/usr/bin/env python
"""Query DIA reference images of NGC 5907 matched to the LOAO stack grid.

Uses ``ezphot.utils.ImageQuerier`` (HiPS2FITS) to download survey images on
exactly the LOAO nightly-stack grid: 2048 x 2048 pixels at 0.794 arcsec/pixel
(FoV 0.4517 deg) centered on the stack center STKCRA/STKCDEC (NGC 5907).

The full ``ImageQuerier.query()`` pipeline reassembles tiles through
``Stack.stack_swarp``, which currently calls ``prepare_images`` with keyword
arguments it does not accept, so that path raises TypeError. A 2048 x 2048
request is far below the 45-Mpixel single-tile limit, so this script uses the
single-tile ``_query`` call directly — no reassembly is needed and the
returned FITS already carries the requested TAN WCS.

LOAO filter -> Pan-STARRS DR1 band mapping (PS1 is the deepest survey with
coverage at Dec +56):

    B102 -> g, V103 -> g, R104 -> r, I105 -> i

Outputs to ``scidata/LOAO/SN2026kid/reference_images/``:
``ref.PS1.<band>.fits`` (+ provenance header keys), ``reference_mapping.json``,
and a QC figure comparing each reference with a deep LOAO stack.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ezphot-matplotlib")

import numpy as np
from astropy.io import fits

from ezphot.utils import ImageQuerier

WIDTH = HEIGHT = 2048
PIXEL_SCALE = 0.794  # arcsec / pixel, LOAO stack grid
FOV_DEG = WIDTH * PIXEL_SCALE / 3600.0
CENTER_RA = 228.9736958333333  # STKCRA (NGC 5907, SIMBAD)
CENTER_DEC = 56.32885          # STKCDEC
FILTER_TO_SURVEY_BAND = {
    "B102": "g",
    "V103": "g",
    "R104": "r",
    "I105": "i",
}
SURVEY_PREFIX = "PanSTARRS/PS1"
OUTPUT_DIR = Path(
    "/home/hhchoi1022/ezphot/data/scidata/LOAO/SN2026kid/reference_images"
)
COMPARISON_STACK = Path(
    "/home/hhchoi1022/ezphot/data/scidata/LOAO/LOAO_E2V_2x2/SN2026kid/"
    "1.0-m KASI/I105/stack.SN2026kid.2026_0610.I105.fits"
)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    survey_bands = sorted(set(FILTER_TO_SURVEY_BAND.values()))

    results = {}
    for band in survey_bands:
        catalog_key = f"{SURVEY_PREFIX}/{band}"
        querier = ImageQuerier(catalog_key)
        coverage = querier.check_coverage(
            CENTER_RA, CENTER_DEC, radius_deg=FOV_DEG / 2, verbose=True
        )
        if not coverage.get(catalog_key, False):
            raise RuntimeError(f"No {catalog_key} coverage at NGC 5907")
        save_path = OUTPUT_DIR / f"ref.PS1.{band}.fits"
        querier._query(
            wcs=None,
            width=WIDTH,
            height=HEIGHT,
            ra=CENTER_RA,
            dec=CENTER_DEC,
            fov=FOV_DEG,
            rotation_angle=0.0,
            save_path=str(save_path),
            verbose=True,
        )
        with fits.open(save_path, mode="update", memmap=False) as hdul:
            header = hdul[0].header
            header["SURVEY"] = ("PanSTARRS DR1", "Reference survey")
            header["HIPSID"] = (querier.catalog_ids[catalog_key], "HiPS identifier")
            header["FILTER"] = (band, "Survey band")
            header["OBJECT"] = ("NGC5907", "Field of SN 2026kid")
            header["REFPXSCL"] = (PIXEL_SCALE, "Requested pixel scale [arcsec/pix]")
            header["REFQUTC"] = (
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "UTC reference query time",
            )
            hdul[0].add_checksum()
            hdul.flush()
        data = fits.getdata(save_path, memmap=False)
        results[band] = {
            "path": str(save_path),
            "finite_fraction": float(np.mean(np.isfinite(data))),
            "median": float(np.nanmedian(data)),
        }
        print(f"{band}: {results[band]}")

    mapping = {
        "center_ra_deg": CENTER_RA,
        "center_dec_deg": CENTER_DEC,
        "width": WIDTH,
        "height": HEIGHT,
        "pixel_scale_arcsec": PIXEL_SCALE,
        "fov_deg": FOV_DEG,
        "survey": "PanSTARRS DR1 (HiPS2FITS)",
        "filter_to_band": FILTER_TO_SURVEY_BAND,
        "references": results,
    }
    (OUTPUT_DIR / "reference_mapping.json").write_text(
        json.dumps(mapping, indent=2) + "\n"
    )

    # QC figure: LOAO deep stack vs the three reference images.
    import matplotlib.pyplot as plt
    from astropy.visualization import ZScaleInterval

    def zscale(data):
        finite = data[np.isfinite(data)]
        return ZScaleInterval().get_limits(finite)

    fig, axes = plt.subplots(1, 1 + len(survey_bands), figsize=(5 * (1 + len(survey_bands)), 5.4),
                             constrained_layout=True)
    stack_data = fits.getdata(COMPARISON_STACK, memmap=False).astype(float)[::4, ::4]
    vmin, vmax = zscale(stack_data)
    axes[0].imshow(stack_data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"LOAO stack {COMPARISON_STACK.name}", fontsize=9)
    axes[0].axis("off")
    for axis, band in zip(axes[1:], survey_bands):
        data = fits.getdata(results[band]["path"], memmap=False).astype(float)[::4, ::4]
        vmin, vmax = zscale(data)
        axis.imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
        axis.set_title(f"PS1 {band} reference", fontsize=9)
        axis.axis("off")
    fig.suptitle(
        "NGC 5907 — LOAO stack grid vs PS1 references "
        f"({WIDTH}px, {PIXEL_SCALE}\"/px, same center)"
    )
    qc_path = OUTPUT_DIR / "reference_query_qc.png"
    fig.savefig(qc_path, dpi=120)
    plt.close(fig)
    print(f"QC: {qc_path}")


if __name__ == "__main__":
    main()
