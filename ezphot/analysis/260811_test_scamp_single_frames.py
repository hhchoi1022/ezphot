#!/usr/bin/env python
"""Test SCAMP astrometric refinement on a sample of LOAO single frames.

The single-frame astrometry.net solutions carry 0.3-1.6 arcsec offsets
relative to APASS/Gaia. This test refines copies (the archive is untouched)
of six representative frames with ``Platesolve.solve_scamp`` (GAIA-DR2) and
compares three WCS variants against APASS star positions:

    before  — the current astrometry.net solution
    scamp   — SCAMP TPV refinement
    refit   — plain ``fit_wcs_from_points`` TAN refit on half of the matched
              APASS stars, evaluated on the held-out half (the no-external-
              tool alternative)

The measurement reuses each frame's existing SExtractor pixel coordinates,
so differences come purely from the WCS. Prints one comparison table and
writes it next to the QC files as ``12_scamp_test_summary.log``.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ezphot-matplotlib")

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import fit_wcs_from_points

MODULE_PATH = Path(__file__).with_name("260805_photometry_loao_sn2026kid.py")
spec = importlib.util.spec_from_file_location("loao_photometry", MODULE_PATH)
phot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phot)

from ezphot.methods import Platesolve  # noqa: E402

BASE = Path("/home/hhchoi1022/ezphot/data/scidata/LOAO")
TEST_FRAMES = [
    "LOAO_E2V_2x2/SN2026kid/1.0-m KASI/I105/obj.SN2026kid.20260611.0094.fits",
    "LOAO_E2V_2x2/SN2026kid/1.0-m KASI/B102/obj.SN2026kid.20260611.0085.fits",
    "LOAO_E2V_2x2/SN2026kid/1.0-m KASI/R104/obj.SN2026kid.20260611.0091.fits",
    "LOAO_E2V_2x2/SN2026kid/1.0-m KASI/V103/obj.SN2026kid.20260626.0096.fits",
    "LOAO_iKon_1x1/SN2026kid/1.0-m KASI/I105/obj.SN2026kid.20260506.0030.fits",
    "LOAO_iKon_1x1/SN2026kid/1.0-m KASI/V103/obj.SN2026kid.20260506.0024.fits",
]
WORKDIR = Path(
    "/tmp/claude-10000/-lyman-data1-factory-hhchoi-code-ezphot/"
    "69f0c3d8-4bfc-4426-9d75-082a81cc10d8/scratchpad/scamp_test"
)
QC_DIR = Path("/home/hhchoi1022/ezphot/log/20260806_stage_run_qc")
APASS_PATH = Path(
    "/home/hhchoi1022/ezphot/data/scidata/LOAO/SN2026kid/reference.APASS_DR9_BVRI.ecsv"
)


def star_pixels(cat_path: Path) -> np.ndarray:
    catalog = Table.read(cat_path, format="ascii")
    good = (
        (np.asarray(catalog["FLAGS"], dtype=float) == 0)
        & (np.asarray(catalog["CLASS_STAR"], dtype=float) > 0.5)
    )
    return np.column_stack(
        [
            np.asarray(catalog["X_IMAGE"], dtype=float)[good] - 1,
            np.asarray(catalog["Y_IMAGE"], dtype=float)[good] - 1,
        ]
    )


def offsets_vs_apass(pixels: np.ndarray, wcs: WCS, apass: SkyCoord, select=None):
    """Median dRA/dDec/radial [arcsec] of matched stars; returns match mask."""
    sky = wcs.pixel_to_world(pixels[:, 0], pixels[:, 1])
    index, separation, _ = sky.match_to_catalog_sky(apass)
    matched = separation.arcsec < 3.0
    if select is not None:
        matched &= select
    dra = (sky[matched].ra - apass[index[matched]].ra).arcsec * np.cos(
        sky[matched].dec.radian
    )
    ddec = (sky[matched].dec - apass[index[matched]].dec).arcsec
    radial = np.hypot(dra, ddec)
    return (
        float(np.median(dra)),
        float(np.median(ddec)),
        float(np.median(radial)),
        matched,
        index,
    )


def build_apass_ldac(apass_table: Table, output_path: Path) -> Path:
    """Write the APASS reference as a FITS-LDAC file for ASTREF_CATALOG FILE."""
    ra = np.asarray(apass_table["ra"], dtype=float)
    dec = np.asarray(apass_table["dec"], dtype=float)
    magnitude = np.asarray(apass_table["r_mag"], dtype=float)
    fallback = np.asarray(apass_table["V_mag"], dtype=float)
    magnitude = np.where(np.isfinite(magnitude), magnitude, fallback)
    keep = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(magnitude)
    position_error_deg = 0.2 / 3600.0

    columns = fits.ColDefs(
        [
            fits.Column(name="X_WORLD", format="1D", unit="deg", array=ra[keep]),
            fits.Column(name="Y_WORLD", format="1D", unit="deg", array=dec[keep]),
            fits.Column(
                name="ERRA_WORLD", format="1E", unit="deg",
                array=np.full(keep.sum(), position_error_deg, dtype=np.float32),
            ),
            fits.Column(
                name="ERRB_WORLD", format="1E", unit="deg",
                array=np.full(keep.sum(), position_error_deg, dtype=np.float32),
            ),
            fits.Column(
                name="ERRTHETA_WORLD", format="1E", unit="deg",
                array=np.zeros(keep.sum(), dtype=np.float32),
            ),
            fits.Column(name="MAG", format="1E", unit="mag",
                        array=magnitude[keep].astype(np.float32)),
            fits.Column(
                name="MAGERR", format="1E", unit="mag",
                array=np.full(keep.sum(), 0.05, dtype=np.float32),
            ),
            fits.Column(
                name="OBSDATE", format="1D", unit="yr",
                array=np.full(keep.sum(), 2012.0),
            ),
        ]
    )
    objects_hdu = fits.BinTableHDU.from_columns(columns, name="LDAC_OBJECTS")
    dummy_header = fits.Header()
    dummy_header["NAXIS"] = 2
    header_cards = [card.image.ljust(80) for card in dummy_header.cards] + [
        "END".ljust(80)
    ]
    imhead_column = fits.Column(
        name="Field Header Card",
        format=f"{80 * len(header_cards)}A",
        dim=f"(80, {len(header_cards)})",
        array=np.array([header_cards]),
    )
    imhead_hdu = fits.BinTableHDU.from_columns(
        fits.ColDefs([imhead_column]), name="LDAC_IMHEAD"
    )
    fits.HDUList([fits.PrimaryHDU(), imhead_hdu, objects_hdu]).writeto(
        output_path, overwrite=True
    )
    return output_path


def main() -> None:
    if WORKDIR.exists():
        shutil.rmtree(WORKDIR)
    WORKDIR.mkdir(parents=True)
    apass_table = Table.read(APASS_PATH, format="ascii.ecsv")
    apass = SkyCoord(
        np.asarray(apass_table["ra"], dtype=float),
        np.asarray(apass_table["dec"], dtype=float),
        unit="deg",
    )
    apass_ldac = build_apass_ldac(apass_table, WORKDIR / "apass_astref.ldac.fits")

    rows = []
    solver = Platesolve()
    rng = np.random.default_rng(20260811)
    scamp_variants = [
        ("gaia", "GAIA-DR2", None),
        ("apass", "FILE", {"ASTREFCAT_NAME": str(apass_ldac)}),
    ]
    for relative in TEST_FRAMES:
        source = BASE / relative
        pixels = star_pixels(Path(str(source) + ".cat"))
        original_wcs = WCS(fits.getheader(source, memmap=False))
        before = offsets_vs_apass(pixels, original_wcs, apass)

        telescope_info = phot.telinfo(source)
        scamp_results = {}
        for label, catalog_type, scamp_params in scamp_variants:
            staged = WORKDIR / f"{label}.{source.name}"
            shutil.copy2(source, staged)
            image = phot.image_instance(staged, telescope_info)
            try:
                solver.solve_scamp(
                    target_img=image,
                    catalog_type=catalog_type,
                    scamp_params=scamp_params,
                    overwrite=True,
                    verbose=False,
                )
                scamp_wcs = WCS(fits.getheader(staged, memmap=False))
                scamp_results[label] = (
                    offsets_vs_apass(pixels, scamp_wcs, apass),
                    None,
                )
            except Exception as error:
                scamp_results[label] = (
                    (np.nan, np.nan, np.nan, None, None),
                    f"{type(error).__name__}: {error}",
                )

        # TAN refit baseline with a 50/50 holdout split.
        _, _, _, matched, index = before
        matched_indices = np.flatnonzero(matched)
        if len(matched_indices) >= 10:
            rng.shuffle(matched_indices)
            half = len(matched_indices) // 2
            fit_set = matched_indices[:half]
            evaluate = np.zeros(len(pixels), dtype=bool)
            evaluate[matched_indices[half:]] = True
            refit_wcs = fit_wcs_from_points(
                (pixels[fit_set, 0], pixels[fit_set, 1]),
                apass[index[fit_set]],
                projection="TAN",
            )
            refit = offsets_vs_apass(pixels, refit_wcs, apass, select=evaluate)
        else:
            refit = (np.nan, np.nan, np.nan, None, None)

        name = source.name.replace("obj.SN2026kid.", "")
        detector = "iKon" if "iKon" in str(source) else "E2V"
        parts = [
            f"{name:<22s} {detector:<5s} n={matched.sum():>3d}",
            f"before |r|={before[2]:.2f}",
        ]
        for label, _, _ in scamp_variants:
            result, error = scamp_results[label]
            parts.append(
                f"scamp-{label} |r|={result[2]:.2f}"
                + (f" [FAIL {error}]" if error else "")
            )
        parts.append(f"refit(holdout) |r|={refit[2]:.2f}")
        rows.append(" | ".join(parts))
        print(rows[-1], flush=True)

    lines = [
        "SCAMP single-frame test — created "
        + datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "columns: median dRA dDec |r| [arcsec] vs APASS; refit = TAN "
        "fit_wcs_from_points, holdout half",
        "",
        *rows,
    ]
    (QC_DIR / "12_scamp_test_summary.log").write_text("\n".join(lines) + "\n")
    print(f"\nlog: {QC_DIR / '12_scamp_test_summary.log'}")


if __name__ == "__main__":
    main()
