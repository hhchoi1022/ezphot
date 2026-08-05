#!/usr/bin/env python3
"""Plot the LOAO SN 2026kid multi-band light curve from stacked catalogs.

The catalog paths are discovered with :class:`~ezphot.utils.DataBrowser`,
collected in a :class:`~ezphot.dataobjects.CatalogSet`, and rendered with
:class:`~ezphot.dataobjects.LightCurve`.  The fourth configured aperture
(``MAGSKY_APER_3``) is the 10 arcsec aperture used by the LOAO reduction.

Some LOAO E2V stack headers currently cannot reconstruct ``ScienceImage``
because their CCD/binning metadata do not match the telescope registry.  A
light curve only needs catalog data and catalog metadata, so this script loads
the catalogs in photometry-only mode and deliberately skips target-image
construction.  It does not modify any catalog or FITS file.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ezphot-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import ascii
from astropy.table import MaskedColumn, Table

from ezphot.dataobjects import Catalog, CatalogSet, LightCurve
from ezphot.dataobjects.catalog import Info
from ezphot.utils import DataBrowser


TARGET = "SN2026kid"
TARGET_RA_DEG = 228.9884428
TARGET_DEC_DEG = 56.3089141
FILTERS = ("B102", "V103", "R104", "I105")
FILTER_ALIASES = {"B102": "B", "V103": "V", "R104": "R", "I105": "I"}
MAG_KEY = "MAGSKY_APER_3"
MAGERR_KEY = "MAGERR_APER_3"
ZPERR_KEY = "ZPERR_APER_3"
DEPTH_KEY = "UL5SKY_APER_3"
DEFAULT_OUTPUT_DIR = Path(
    "/home/hhchoi1022/ezphot/data/scidata/LOAO/SN2026kid/lightcurve"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ra", type=float, default=TARGET_RA_DEG)
    parser.add_argument("--dec", type=float, default=TARGET_DEC_DEG)
    parser.add_argument("--matching-radius", type=float, default=3.0)
    parser.add_argument("--obs-start", default="2026-04-20")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_catalog_without_image(path: str | Path) -> Catalog:
    """Load a Catalog and its Info without constructing the target image."""
    path = Path(path)
    info_path = Path(f"{path}.info")
    if not info_path.exists():
        raise FileNotFoundError(f"Missing catalog metadata: {info_path}")

    info_dict = json.loads(info_path.read_text(encoding="utf-8"))
    target_img = info_dict.get("target_img")
    info_dict["target_img"] = None

    # Catalog.__init__ normally searches for and loads the corresponding FITS
    # image.  It is unnecessary for light-curve extraction and fails for E2V
    # headers whose detector is presently registered as iKon/binning=2.
    with patch.object(Catalog, "_load_target_img", lambda self: None):
        catalog = Catalog(path, info=Info.from_dict(info_dict), load=False)
    catalog.info.target_img = target_img
    return catalog


def has_valid_target_photometry(catalog: Catalog) -> bool:
    """Return whether a selected target row has calibrated 10 arcsec data."""
    table = catalog.target_data
    required = (MAG_KEY, MAGERR_KEY, ZPERR_KEY)
    if len(table) == 0 or any(key not in table.colnames for key in required):
        return False
    valid = np.ones(len(table), dtype=bool)
    for key in required:
        values = np.ma.asarray(table[key], dtype=float).filled(np.nan)
        valid &= np.isfinite(values)
    return bool(np.any(valid))


def make_nightly_magnitude_table(lightcurve_table: Table) -> Table:
    """Pivot measurements to one UTC-date row and one column per filter."""
    utc_dates = np.asarray([str(value)[:10] for value in lightcurve_table["obsdate"]])
    unique_dates = sorted(set(utc_dates))
    output = Table()
    output["obsdate"] = unique_dates

    for filter_name in FILTER_ALIASES.values():
        values = np.ma.masked_all(len(unique_dates), dtype=float)
        for index, obsdate in enumerate(unique_dates):
            selected = (
                (utc_dates == obsdate)
                & (np.asarray(lightcurve_table["filter"]) == filter_name)
            )
            magnitudes = np.ma.asarray(
                lightcurve_table[MAG_KEY][selected], dtype=float
            ).compressed()
            if len(magnitudes) > 1:
                raise RuntimeError(
                    f"Multiple {filter_name}-band stacks found on UTC {obsdate}"
                )
            if len(magnitudes) == 1:
                values[index] = magnitudes[0]
        output[filter_name] = MaskedColumn(values, format=".3f")
    return output


def main() -> None:
    args = parse_args()

    browser = DataBrowser("scidata")
    browser.observatory = "LOAO"
    browser.objname = TARGET
    catalog_paths = sorted(
        browser.search(
            pattern=f"stack.{TARGET}.*.fits.cat",
            return_type="path",
        )
    )
    if not catalog_paths:
        raise RuntimeError("DataBrowser found no stacked SN2026kid catalogs")

    catalogs = []
    failures = []
    for path in catalog_paths:
        try:
            catalogs.append(load_catalog_without_image(path))
        except Exception as exc:
            failures.append((str(path), str(exc)))

    catalogset = CatalogSet(catalogs)
    catalogset.select_catalogs(filter=FILTERS, obs_start=args.obs_start)
    catalogset.select_sources(
        args.ra,
        args.dec,
        matching_radius=args.matching_radius,
    )

    valid_catalogs = [
        catalog
        for catalog in catalogset.target_catalogs
        if has_valid_target_photometry(catalog)
    ]
    if not valid_catalogs:
        raise RuntimeError("No valid calibrated 10 arcsec target measurements found")

    # LightCurve's native broadband colors and offsets use B/V/R/I names.
    # Normalize the LOAO hardware filter codes in memory; no .info file is
    # modified.
    for catalog in valid_catalogs:
        catalog.info.filter = FILTER_ALIASES[catalog.info.filter]

    # Use a fresh CatalogSet so LightCurve sees only valid target measurements.
    lightcurve_catalogset = CatalogSet(valid_catalogs)
    lightcurve = LightCurve(lightcurve_catalogset)
    lightcurve.OBSERVATORY_MARKER = {
        **lightcurve.OBSERVATORY_MARKER,
        "LOAO": "o",
    }
    lightcurve.plt_params.figure_figsize = (11, 7)
    lightcurve.plt_params.non_detection_enabled = False

    figures, _, axes, table = lightcurve.plot(
        ra=args.ra,
        dec=args.dec,
        matching_radius_arcsec=args.matching_radius,
        flux_key=MAG_KEY,
        fluxerr_key=MAGERR_KEY,
        zperr_key=ZPERR_KEY,
        depth_key=DEPTH_KEY,
        apply_offset=False,
        title="SN 2026kid — LOAO stacked photometry (10 arcsec aperture)",
    )
    if not figures or table is None or len(table) == 0:
        raise RuntimeError("LightCurve did not produce a plot table")

    table.sort("mjd")
    figure = next(iter(figures.values()))
    axis = next(iter(axes.values()))
    axis.set_xlabel("UTC date")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = "SN2026kid_LOAO_stack_multiband_10arcsec"
    png_path = args.output_dir / f"{stem}.png"
    ecsv_path = args.output_dir / f"{stem}.ecsv"
    fixed_width_path = args.output_dir / f"{stem}_magnitudes.txt"
    summary_path = args.output_dir / f"{stem}.json"

    figure.savefig(png_path, dpi=200, bbox_inches="tight")
    table.write(ecsv_path, format="ascii.ecsv", overwrite=True)
    nightly_table = make_nightly_magnitude_table(table)
    nightly_table.write(
        fixed_width_path,
        format="ascii.fixed_width",
        fill_values=[(ascii.masked, "--")],
        overwrite=True,
    )

    counts = {
        filter_name: int(np.count_nonzero(table["filter"] == filter_name))
        for filter_name in FILTER_ALIASES.values()
    }
    filter_offsets = {filter_name: 0.0 for filter_name in FILTER_ALIASES.values()}
    summary = {
        "target": TARGET,
        "ra_deg": args.ra,
        "dec_deg": args.dec,
        "matching_radius_arcsec": args.matching_radius,
        "aperture_diameter_arcsec": 10.0,
        "instrument_filter_map": FILTER_ALIASES,
        "filter_offsets_mag": filter_offsets,
        "magnitude_key": MAG_KEY,
        "catalog_pattern": f"stack.{TARGET}.*.fits.cat",
        "catalogs_found": len(catalog_paths),
        "catalogs_loaded": len(catalogs),
        "valid_measurements": len(table),
        "measurements_by_filter": counts,
        "load_failures": failures,
        "plot": str(png_path),
        "table": str(ecsv_path),
        "fixed_width_magnitude_table": str(fixed_width_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    plt.close("all")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
