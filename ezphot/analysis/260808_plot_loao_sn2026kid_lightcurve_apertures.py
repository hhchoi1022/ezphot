#!/usr/bin/env python3
"""SN 2026kid LOAO light curves for 3, 5, and 10 arcsec apertures.

Reuses the catalog-loading and pivot helpers of
``260805_plot_loao_sn2026kid_lightcurve.py`` and produces, per aperture, the
same product set as the original 10 arcsec run (PNG + ECSV + fixed-width
magnitude table + JSON summary) with ``_<n>arcsec`` stems, plus one comparison
figure overlaying the three apertures per filter.

Aperture-to-column mapping (APERTURES_ARCSEC = [3, 5, 7, 10]):
3" = MAGSKY_APER, 5" = MAGSKY_APER_1, 10" = MAGSKY_APER_3.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ezphot-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import ascii
from astropy.table import Table

MODULE_PATH = Path(__file__).with_name("260805_plot_loao_sn2026kid_lightcurve.py")
spec = importlib.util.spec_from_file_location("loao_lightcurve", MODULE_PATH)
lc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lc)

from ezphot.dataobjects import CatalogSet, LightCurve  # noqa: E402
from ezphot.utils import DataBrowser  # noqa: E402

APERTURES = {
    "3": ("MAGSKY_APER", "MAGERR_APER", "ZPERR_APER", "UL5SKY_APER"),
    "5": ("MAGSKY_APER_1", "MAGERR_APER_1", "ZPERR_APER_1", "UL5SKY_APER_1"),
    "10": ("MAGSKY_APER_3", "MAGERR_APER_3", "ZPERR_APER_3", "UL5SKY_APER_3"),
}
OUTPUT_DIR = lc.DEFAULT_OUTPUT_DIR
MATCHING_RADIUS = 3.0
OBS_START = "2026-04-20"


def main() -> None:
    browser = DataBrowser("scidata")
    browser.observatory = "LOAO"
    browser.objname = lc.TARGET
    catalog_paths = sorted(
        browser.search(pattern=f"stack.{lc.TARGET}.*.fits.cat", return_type="path")
    )
    if not catalog_paths:
        raise RuntimeError("DataBrowser found no stacked SN2026kid catalogs")

    catalogs = [lc.load_catalog_without_image(path) for path in catalog_paths]
    catalogset = CatalogSet(catalogs)
    catalogset.select_catalogs(filter=lc.FILTERS, obs_start=OBS_START)
    catalogset.select_sources(
        lc.TARGET_RA_DEG, lc.TARGET_DEC_DEG, matching_radius=MATCHING_RADIUS
    )
    for catalog in catalogset.target_catalogs:
        catalog.info.filter = lc.FILTER_ALIASES.get(
            catalog.info.filter, catalog.info.filter
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tables = {}
    for label, (mag_key, magerr_key, zperr_key, depth_key) in APERTURES.items():
        lc.MAG_KEY, lc.MAGERR_KEY, lc.ZPERR_KEY, lc.DEPTH_KEY = (
            mag_key, magerr_key, zperr_key, depth_key,
        )
        valid_catalogs = [
            catalog
            for catalog in catalogset.target_catalogs
            if lc.has_valid_target_photometry(catalog)
        ]
        if not valid_catalogs:
            raise RuntimeError(f"No valid {label}\" measurements found")

        lightcurve = LightCurve(CatalogSet(valid_catalogs))
        lightcurve.OBSERVATORY_MARKER = {
            **lightcurve.OBSERVATORY_MARKER,
            "LOAO": "o",
        }
        lightcurve.plt_params.figure_figsize = (11, 7)
        lightcurve.plt_params.non_detection_enabled = False
        figures, _, axes, table = lightcurve.plot(
            ra=lc.TARGET_RA_DEG,
            dec=lc.TARGET_DEC_DEG,
            matching_radius_arcsec=MATCHING_RADIUS,
            flux_key=mag_key,
            fluxerr_key=magerr_key,
            zperr_key=zperr_key,
            depth_key=depth_key,
            apply_offset=False,
            title=f"SN 2026kid — LOAO stacked photometry ({label} arcsec aperture)",
        )
        if not figures or table is None or len(table) == 0:
            raise RuntimeError(f"LightCurve produced no table for {label}\"")
        table.sort("mjd")
        tables[label] = table

        figure = next(iter(figures.values()))
        next(iter(axes.values())).set_xlabel("UTC date")
        stem = f"SN2026kid_LOAO_stack_multiband_{label}arcsec"
        figure.savefig(OUTPUT_DIR / f"{stem}.png", dpi=200, bbox_inches="tight")
        table.write(OUTPUT_DIR / f"{stem}.ecsv", format="ascii.ecsv", overwrite=True)
        lc.make_nightly_magnitude_table(table).write(
            OUTPUT_DIR / f"{stem}_magnitudes.txt",
            format="ascii.fixed_width",
            fill_values=[(ascii.masked, "--")],
            overwrite=True,
        )
        counts = {
            filter_name: int(np.count_nonzero(table["filter"] == filter_name))
            for filter_name in lc.FILTER_ALIASES.values()
        }
        summary = {
            "target": lc.TARGET,
            "aperture_diameter_arcsec": float(label),
            "magnitude_key": mag_key,
            "valid_measurements": len(table),
            "measurements_by_filter": counts,
            "plot": str(OUTPUT_DIR / f"{stem}.png"),
            "table": str(OUTPUT_DIR / f"{stem}.ecsv"),
            "fixed_width_magnitude_table": str(OUTPUT_DIR / f"{stem}_magnitudes.txt"),
        }
        (OUTPUT_DIR / f"{stem}.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(f"{label}\": {len(table)} measurements {counts}")
        plt.close("all")

    # Comparison figure: one panel per filter, one line per aperture.
    aperture_styles = {"3": (":", 0.55), "5": ("--", 0.75), "10": ("-", 1.0)}
    filter_colors = {"B": "tab:blue", "V": "tab:green", "R": "tab:red", "I": "maroon"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, constrained_layout=True)
    for axis, (filter_name, color) in zip(axes.flat, filter_colors.items()):
        for label, (linestyle, alpha) in aperture_styles.items():
            table = tables[label]
            mag_key, magerr_key = APERTURES[label][0], APERTURES[label][1]
            select = np.asarray(table["filter"]) == filter_name
            mjd = np.asarray(table["mjd"][select], dtype=float)
            mags = np.ma.asarray(table[mag_key][select]).astype(float).filled(np.nan)
            errors = np.ma.asarray(table[magerr_key][select]).astype(float).filled(np.nan)
            finite = np.isfinite(mags)
            axis.errorbar(
                mjd[finite], mags[finite], yerr=errors[finite],
                fmt="o", ls=linestyle, ms=3.5, lw=1, capsize=2,
                color=color, alpha=alpha, label=f'{label}"',
            )
        axis.invert_yaxis()
        axis.set_title(filter_name)
        axis.set_xlabel("MJD")
        axis.set_ylabel("Magnitude")
        axis.legend(fontsize=9, title="aperture")
        axis.grid(alpha=0.2)
    fig.suptitle("SN 2026kid — aperture comparison (3/5/10 arcsec, no offsets)")
    comparison_path = OUTPUT_DIR / "SN2026kid_LOAO_stack_aperture_comparison.png"
    fig.savefig(comparison_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"comparison: {comparison_path}")


if __name__ == "__main__":
    main()
