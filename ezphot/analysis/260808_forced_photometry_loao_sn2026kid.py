#!/usr/bin/env python
"""Forced aperture photometry of SN 2026kid on every LOAO nightly stack.

Runs ``AperturePhotometry.circular_photometry`` at the fixed SN position on
all 283 stacks with 3/5/10 arcsec apertures and NO annulus (the stacks are
already background-subtracted; upper limits come from the stack RMS map).
Instrumental magnitudes are calibrated with the per-aperture zero points the
stacks carry from PhotometricCalibration:

    forced MAG_APER   (3")  + header ZP_APER
    forced MAG_APER_1 (5")  + header ZP_APER_1
    forced MAG_APER_2 (10") + header ZP_APER_3   (catalog aperture list [3,5,7,10])

Outputs (in the ``lightcurve`` directory):
- ``SN2026kid_LOAO_forced_photometry.ecsv``      full per-stack table
- ``SN2026kid_LOAO_forced_<n>arcsec_magnitudes.txt``  obsdate x filter pivots
- ``SN2026kid_LOAO_forced_multiband.png``        forced light curves + limits
- ``SN2026kid_LOAO_forced_vs_catalog.png``       forced minus catalog offsets
Each stack also gains a ``.circ.cat`` sidecar with the raw forced measurement.
A detection requires flux signal-to-noise >= 3; otherwise the 5-sigma limit
is reported.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ezphot-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import ascii, fits
from astropy.table import MaskedColumn, Table

MODULE_PATH = Path(__file__).with_name("260805_photometry_loao_sn2026kid.py")
spec = importlib.util.spec_from_file_location("loao_photometry", MODULE_PATH)
phot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phot)

from ezphot.imageobjects import Errormap  # noqa: E402
from ezphot.methods import AperturePhotometry  # noqa: E402

TARGET_RA_DEG = 228.9884428
TARGET_DEC_DEG = 56.3089141
APERTURES_ARCSEC = [3, 5, 10]
# forced-catalog column suffix -> (label, stack-header ZP key)
APERTURE_MAP = {
    "": ("3", "ZP_APER"),
    "_1": ("5", "ZP_APER_1"),
    "_2": ("10", "ZP_APER_3"),
}
FILTER_ALIASES = {"B102": "B", "V103": "V", "R104": "R", "I105": "I"}
FILTER_COLORS = {"B": "tab:blue", "V": "tab:green", "R": "tab:red", "I": "maroon"}
ROOTS = [
    Path("/home/hhchoi1022/ezphot/data/scidata/LOAO/LOAO_E2V_2x2/SN2026kid/1.0-m KASI"),
    Path("/home/hhchoi1022/ezphot/data/scidata/LOAO/LOAO_iKon_1x1/SN2026kid/1.0-m KASI"),
]
OUTPUT_DIR = Path("/home/hhchoi1022/ezphot/data/scidata/LOAO/SN2026kid/lightcurve")
SNR_DETECTION = 3.0


def discover_stacks() -> list[Path]:
    stacks = []
    for root in ROOTS:
        for filter_name in FILTER_ALIASES:
            stacks.extend(
                path
                for path in sorted((root / filter_name).glob("stack.SN2026kid.*.fits"))
                if path.suffix == ".fits"
            )
    if not stacks:
        raise RuntimeError("No stacks found")
    return stacks


def force_one(photometry: AperturePhotometry, path: Path) -> dict:
    telescope_info = phot.telinfo(path)
    image = phot.image_instance(path, telescope_info)
    rms = Errormap(image.savepath.bkgrmspath, emaptype="bkgrms", load=True)

    module = importlib.import_module("ezphot.methods.aperturephotometry")
    original_catalog_class = module.Catalog
    try:
        module.Catalog = phot.SafeCatalog
        catalog = photometry.circular_photometry(
            target_img=image,
            x_arr=TARGET_RA_DEG,
            y_arr=TARGET_DEC_DEG,
            aperture_diameter_arcsec=APERTURES_ARCSEC,
            aperture_diameter_seeing=None,
            annulus_width_arcsec=None,
            unit="coord",
            target_bkg=None,
            target_bkgrms=rms,
            save=True,
            verbose=False,
            visualize=False,
            save_fig=False,
        )
    finally:
        module.Catalog = original_catalog_class

    header = image.header
    row = {
        "path": str(path),
        "night": str(header["NIGHTDIR"]),
        "filter": FILTER_ALIASES[str(header["FILTER"])],
        "mjd": float(header["MJD"]),
        "obsdate": str(header["DATE-OBS"])[:10],
    }
    measurement = catalog.data[0]
    for suffix, (label, zp_key) in APERTURE_MAP.items():
        zero_point = float(header[zp_key])
        flux = float(measurement[f"FLUX_APER{suffix}"])
        flux_error = float(measurement[f"FLUXERR_APER{suffix}"])
        limit5 = float(measurement[f"UL5_APER{suffix}"])
        detected = (
            np.isfinite(flux)
            and np.isfinite(flux_error)
            and flux_error > 0
            and flux / flux_error >= SNR_DETECTION
        )
        row[f"mag_{label}"] = (
            zero_point - 2.5 * np.log10(flux) if detected else np.nan
        )
        row[f"magerr_{label}"] = (
            2.5 / np.log(10) * flux_error / flux if detected else np.nan
        )
        row[f"ul5_{label}"] = zero_point + limit5 if np.isfinite(limit5) else np.nan
        row[f"detected_{label}"] = bool(detected)
    return row


def pivot(table: Table, label: str) -> Table:
    dates = sorted(set(table["obsdate"]))
    output = Table()
    output["obsdate"] = dates
    for filter_name in FILTER_COLORS:
        values = np.ma.masked_all(len(dates), dtype=float)
        for index, date in enumerate(dates):
            select = (
                (np.asarray(table["obsdate"]) == date)
                & (np.asarray(table["filter"]) == filter_name)
                & np.asarray(table[f"detected_{label}"], dtype=bool)
            )
            magnitudes = np.asarray(table[f"mag_{label}"][select], dtype=float)
            magnitudes = magnitudes[np.isfinite(magnitudes)]
            if len(magnitudes):
                values[index] = magnitudes[0]
        output[filter_name] = MaskedColumn(values, format=".3f")
    return output


def main() -> None:
    stacks = discover_stacks()
    photometry = AperturePhotometry()
    rows, failures = [], []
    for index, path in enumerate(stacks, start=1):
        try:
            rows.append(force_one(photometry, path))
        except Exception as error:
            failures.append({"path": str(path), "error": f"{type(error).__name__}: {error}"})
        if index % 50 == 0:
            print(f"[{index}/{len(stacks)}]", flush=True)
    if failures:
        for failure in failures:
            print("FAIL", failure["path"], failure["error"], flush=True)
    table = Table(rows)
    table.sort("mjd")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.write(
        OUTPUT_DIR / "SN2026kid_LOAO_forced_photometry.ecsv",
        format="ascii.ecsv",
        overwrite=True,
    )
    for _, (label, _) in APERTURE_MAP.items():
        pivot(table, label).write(
            OUTPUT_DIR / f"SN2026kid_LOAO_forced_{label}arcsec_magnitudes.txt",
            format="ascii.fixed_width",
            fill_values=[(ascii.masked, "--")],
            overwrite=True,
        )

    # ---- forced multiband figure (detections + 5-sigma limits) ------------
    aperture_styles = {"3": (":", 0.55), "5": ("--", 0.75), "10": ("-", 1.0)}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, constrained_layout=True)
    for axis, (filter_name, color) in zip(axes.flat, FILTER_COLORS.items()):
        subset = table[np.asarray(table["filter"]) == filter_name]
        mjd = np.asarray(subset["mjd"], dtype=float)
        for label, (linestyle, alpha) in aperture_styles.items():
            detected = np.asarray(subset[f"detected_{label}"], dtype=bool)
            axis.errorbar(
                mjd[detected],
                np.asarray(subset[f"mag_{label}"], dtype=float)[detected],
                yerr=np.asarray(subset[f"magerr_{label}"], dtype=float)[detected],
                fmt="o", ls=linestyle, ms=3.5, lw=1, capsize=2,
                color=color, alpha=alpha, label=f'{label}"',
            )
        limits = ~np.asarray(subset["detected_10"], dtype=bool)
        axis.scatter(
            mjd[limits],
            np.asarray(subset["ul5_10"], dtype=float)[limits],
            marker="v", s=26, facecolors="none", edgecolors=color, alpha=0.6,
            label='10" 5$\\sigma$ limit',
        )
        axis.invert_yaxis()
        axis.set_title(filter_name)
        axis.set_xlabel("MJD")
        axis.set_ylabel("Magnitude")
        axis.legend(fontsize=8, title="forced aperture")
        axis.grid(alpha=0.2)
    fig.suptitle(
        "SN 2026kid — forced photometry on nightly stacks "
        "(no annulus; open triangles = 5$\\sigma$ limits, 10\")"
    )
    forced_png = OUTPUT_DIR / "SN2026kid_LOAO_forced_multiband.png"
    fig.savefig(forced_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # ---- forced minus catalog comparison ----------------------------------
    catalog_tables = {
        label: Table.read(
            OUTPUT_DIR / f"SN2026kid_LOAO_stack_multiband_{label}arcsec.ecsv",
            format="ascii.ecsv",
        )
        for label in ("3", "5", "10")
    }
    catalog_keys = {"3": "MAGSKY_APER", "5": "MAGSKY_APER_1", "10": "MAGSKY_APER_3"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True, constrained_layout=True)
    for axis, label in zip(axes, ("3", "5", "10")):
        reference = catalog_tables[label]
        ref_mjd = np.asarray(reference["mjd"], dtype=float)
        ref_mag = np.ma.asarray(reference[catalog_keys[label]]).astype(float).filled(np.nan)
        ref_filter = np.asarray(reference["filter"])
        differences = []
        for row in table:
            if not row[f"detected_{label}"]:
                continue
            match = (np.abs(ref_mjd - row["mjd"]) < 0.2) & (ref_filter == row["filter"])
            if not np.any(match):
                continue
            catalog_mag = ref_mag[match][0]
            if np.isfinite(catalog_mag):
                differences.append(
                    (row["mjd"], row[f"mag_{label}"] - catalog_mag, row["filter"])
                )
        for filter_name, color in FILTER_COLORS.items():
            points = [(m, d) for m, d, flt in differences if flt == filter_name]
            if points:
                axis.plot(*zip(*points), ".", ms=5, color=color, alpha=0.7,
                          label=filter_name)
        offsets = np.array([d for _, d, _ in differences])
        axis.axhline(0, color="k", lw=0.8)
        axis.set_title(
            f'{label}"  (median {np.median(offsets):+.3f}, '
            f"n={len(offsets)})"
        )
        axis.set_xlabel("MJD")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("forced − catalog [mag]")
    axes[0].legend(fontsize=8)
    fig.suptitle("SN 2026kid — forced photometry minus SExtractor catalog magnitudes")
    compare_png = OUTPUT_DIR / "SN2026kid_LOAO_forced_vs_catalog.png"
    fig.savefig(compare_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    detections = {
        label: int(np.sum(np.asarray(table[f"detected_{label}"], dtype=bool)))
        for _, (label, _) in APERTURE_MAP.items()
    }
    summary = {
        "target_ra_deg": TARGET_RA_DEG,
        "target_dec_deg": TARGET_DEC_DEG,
        "stacks": len(stacks),
        "measured": len(table),
        "failures": failures,
        "annulus": None,
        "snr_detection_threshold": SNR_DETECTION,
        "detections_by_aperture": detections,
        "outputs": [str(forced_png), str(compare_png)],
    }
    (OUTPUT_DIR / "SN2026kid_LOAO_forced_photometry.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
