#!/usr/bin/env python
"""DIA Stage C — fixed-coordinate forced photometry on difference images.

Steps:

1. Refine the SN coordinate: detect the SN with SEP on bright-epoch
   (<= 2026_0615, V/R/I) difference images near the predicted position,
   convert each centroid to sky through that stack's APASS-refit TAN WCS, and
   adopt the median as THE single SN coordinate for every measurement.
2. Forced photometry (3/5/10 arcsec apertures) at that fixed coordinate, in
   pixel space through each stack's refit WCS, on four variants:
       dia_noann    difference image, no annulus
       dia_ann      difference image, local annulus (width 10 arcsec)
       direct_noann science stack, no annulus (matches the earlier forced run)
       direct_ann   science stack, local annulus
   All calibrated with the stack's per-aperture zero points
   (3"->ZP_APER, 5"->ZP_APER_1, 10"->ZP_APER_3).
3. Outputs: full ECSV, per-variant fixed-width magnitude tables, a
   subtracted-vs-unsubtracted light-curve comparison figure, and QC/log files
   (``13_diaphot``).
"""

from __future__ import annotations

import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ezphot-matplotlib")

import numpy as np
import sep
from astropy.coordinates import SkyCoord
from astropy.io import ascii, fits
from astropy.table import MaskedColumn, Table

PHOT_MODULE = Path(__file__).with_name("260805_photometry_loao_sn2026kid.py")
STAGEB_MODULE = Path(__file__).with_name("260811_dia_stage_b_subtract.py")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


phot = load_module(PHOT_MODULE, "loao_photometry")
stage_b = load_module(STAGEB_MODULE, "loao_dia_stage_b")

from ezphot.imageobjects import Errormap  # noqa: E402
from ezphot.methods import AperturePhotometry  # noqa: E402

APERTURE_MAP = {"": ("3", "ZP_APER"), "_1": ("5", "ZP_APER_1"), "_2": ("10", "ZP_APER_3")}
APERTURES_ARCSEC = [3, 5, 10]
FILTER_ALIASES = {"B102": "B", "V103": "V", "R104": "R", "I105": "I"}
FILTER_COLORS = {"B": "tab:blue", "V": "tab:green", "R": "tab:red", "I": "maroon"}
OUTPUT_DIR = Path("/home/hhchoi1022/ezphot/data/scidata/LOAO/SN2026kid/lightcurve")
QC_DIR = Path("/home/hhchoi1022/ezphot/log/20260806_stage_run_qc")
BRIGHT_NIGHT_MAX = "2026_0615"
CENTROID_FILTERS = {"V103", "R104", "I105"}
ANNULUS_WIDTH_ARCSEC = 10.0
SNR_DETECTION = 3.0


def discover_differences() -> list[Path]:
    subs = []
    for root in stage_b.ROOTS:
        for filter_name in stage_b.FILTER_TO_BAND:
            subs.extend(
                sorted((root / filter_name).glob("sub_stack.SN2026kid.*.fits"))
            )
    return [p for p in subs if p.suffix == ".fits"]


def refine_coordinate(differences: list[Path], apass) -> tuple[float, float, dict]:
    positions = []
    used = []
    for sub_path in differences:
        header = fits.getheader(sub_path, memmap=False)
        night = str(header["NIGHTDIR"])
        filter_name = str(header["FILTER"])
        if night > BRIGHT_NIGHT_MAX or filter_name not in CENTROID_FILTERS:
            continue
        data = fits.getdata(sub_path, memmap=False).astype(np.float32)
        x_predicted = float(header["SNXPIX"])
        y_predicted = float(header["SNYPIX"])
        half = 30
        x0, y0 = int(round(x_predicted)) - half, int(round(y_predicted)) - half
        cut = np.ascontiguousarray(
            np.nan_to_num(data[y0 : y0 + 2 * half, x0 : x0 + 2 * half], nan=0.0)
        )
        rms = float(
            np.nanmedian(
                fits.getdata(
                    Path(str(sub_path).replace("sub_", "", 1) + ".bkgrms"),
                    memmap=False,
                )
            )
        )
        try:
            detections = sep.extract(cut, 5.0, err=rms, minarea=5)
        except Exception:
            continue
        if len(detections) == 0:
            continue
        distance = np.hypot(
            detections["x"] + x0 - x_predicted, detections["y"] + y0 - y_predicted
        )
        best = int(np.argmin(distance))
        if distance[best] > 5.0:
            continue
        stack_path = Path(str(sub_path).replace("sub_", "", 1))
        fitted_wcs, n_stars, _ = stage_b.refit_wcs(stack_path, apass)
        if fitted_wcs is None:
            continue
        sky = fitted_wcs.pixel_to_world(
            detections["x"][best] + x0, detections["y"][best] + y0
        )
        positions.append([sky.ra.deg, sky.dec.deg])
        used.append(f"{night} {filter_name} (refit stars {n_stars})")
    positions = np.array(positions)
    if len(positions) < 5:
        raise RuntimeError(f"Only {len(positions)} SN centroids found")
    ra, dec = float(np.median(positions[:, 0])), float(np.median(positions[:, 1]))
    scatter = SkyCoord(positions[:, 0], positions[:, 1], unit="deg").separation(
        SkyCoord(ra, dec, unit="deg")
    )
    info = {
        "n_centroids": len(positions),
        "median_ra_deg": ra,
        "median_dec_deg": dec,
        "scatter_median_arcsec": float(np.median(scatter.arcsec)),
        "offset_from_discovery_arcsec": float(
            SkyCoord(ra, dec, unit="deg")
            .separation(SkyCoord(stage_b.SN_RA, stage_b.SN_DEC, unit="deg"))
            .arcsec
        ),
    }
    return ra, dec, info


def measure(
    photometry: AperturePhotometry,
    image_path: Path,
    stack_path: Path,
    x_pix: float,
    y_pix: float,
    annulus: bool,
) -> dict:
    telescope_info = phot.telinfo(stack_path)
    image = phot.image_instance(image_path, telescope_info)
    rms = Errormap(
        Path(str(stack_path) + ".bkgrms"), emaptype="bkgrms", load=True
    )
    import importlib as importlib_module

    module = importlib_module.import_module("ezphot.methods.aperturephotometry")
    original_catalog_class = module.Catalog
    try:
        module.Catalog = phot.SafeCatalog
        catalog = photometry.circular_photometry(
            target_img=image,
            x_arr=x_pix,
            y_arr=y_pix,
            aperture_diameter_arcsec=APERTURES_ARCSEC,
            aperture_diameter_seeing=None,
            annulus_width_arcsec=ANNULUS_WIDTH_ARCSEC if annulus else None,
            unit="pixel",
            target_bkg=None,
            target_bkgrms=rms,
            save=True,
            verbose=False,
            visualize=False,
            save_fig=False,
        )
    finally:
        module.Catalog = original_catalog_class

    header = fits.getheader(stack_path, memmap=False)
    measurement = catalog.data[0]
    row = {}
    for suffix, (label, zp_key) in APERTURE_MAP.items():
        zero_point = float(header[zp_key])
        flux = float(measurement[f"FLUX_APER{suffix}"])
        flux_error = float(measurement[f"FLUXERR_APER{suffix}"])
        limit_key = f"UL5_APER{suffix}"
        limit5 = (
            float(measurement[limit_key]) if limit_key in catalog.data.colnames else np.nan
        )
        detected = (
            np.isfinite(flux)
            and np.isfinite(flux_error)
            and flux_error > 0
            and flux / flux_error >= SNR_DETECTION
        )
        row[f"mag_{label}"] = zero_point - 2.5 * np.log10(flux) if detected else np.nan
        row[f"magerr_{label}"] = (
            2.5 / np.log(10) * flux_error / flux if detected else np.nan
        )
        row[f"ul5_{label}"] = zero_point + limit5 if np.isfinite(limit5) else np.nan
        row[f"detected_{label}"] = bool(detected)
    return row


def main() -> None:
    import matplotlib.pyplot as plt

    differences = discover_differences()
    if not differences:
        raise RuntimeError("No sub_stack images found; run Stage B first")
    apass = stage_b.load_apass()
    ra, dec, coordinate_info = refine_coordinate(differences, apass)
    print(f"Updated SN coordinate: {ra:.7f} {dec:+.7f}")
    print(json.dumps(coordinate_info, indent=2))

    photometry = AperturePhotometry()
    sn_sky = SkyCoord(ra, dec, unit="deg")
    rows, failures = [], []
    for index, sub_path in enumerate(differences, start=1):
        stack_path = Path(str(sub_path).replace("sub_", "", 1))
        try:
            header = fits.getheader(sub_path, memmap=False)
            fitted_wcs, _, _ = stage_b.refit_wcs(stack_path, apass)
            wcs = fitted_wcs if fitted_wcs is not None else None
            if wcs is None:
                from astropy.wcs import WCS as WCS_class

                wcs = WCS_class(header)
            x_pix, y_pix = (float(v) for v in wcs.world_to_pixel(sn_sky))
            row = {
                "night": str(header["NIGHTDIR"]),
                "filter": FILTER_ALIASES[str(header["FILTER"])],
                "mjd": float(header["MJD"]),
                "obsdate": str(header["DATE-OBS"])[:10],
                "x_pix": x_pix,
                "y_pix": y_pix,
            }
            for variant, image_path, annulus in (
                ("dia_noann", sub_path, False),
                ("dia_ann", sub_path, True),
                ("direct_noann", stack_path, False),
                ("direct_ann", stack_path, True),
            ):
                result = measure(photometry, image_path, stack_path, x_pix, y_pix, annulus)
                row.update({f"{variant}_{k}": v for k, v in result.items()})
            rows.append(row)
            if index % 40 == 0:
                print(f"[{index}/{len(differences)}]", flush=True)
        except Exception as error:
            failures.append(
                {"path": str(sub_path), "error": f"{type(error).__name__}: {error}"}
            )
    table = Table(rows)
    table.sort("mjd")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.write(
        OUTPUT_DIR / "SN2026kid_LOAO_dia_photometry.ecsv",
        format="ascii.ecsv",
        overwrite=True,
    )

    # Fixed-width magnitude tables per variant (10" primary aperture).
    for variant in ("dia_noann", "dia_ann", "direct_noann", "direct_ann"):
        for label in ("3", "5", "10"):
            dates = sorted(set(table["obsdate"]))
            pivot = Table()
            pivot["obsdate"] = dates
            for filter_name in FILTER_COLORS:
                values = np.ma.masked_all(len(dates), dtype=float)
                for date_index, date in enumerate(dates):
                    select = (
                        (np.asarray(table["obsdate"]) == date)
                        & (np.asarray(table["filter"]) == filter_name)
                        & np.asarray(table[f"{variant}_detected_{label}"], dtype=bool)
                    )
                    magnitudes = np.asarray(
                        table[f"{variant}_mag_{label}"][select], dtype=float
                    )
                    magnitudes = magnitudes[np.isfinite(magnitudes)]
                    if len(magnitudes):
                        values[date_index] = magnitudes[0]
                pivot[filter_name] = MaskedColumn(values, format=".3f")
            pivot.write(
                OUTPUT_DIR / f"SN2026kid_LOAO_{variant}_{label}arcsec_magnitudes.txt",
                format="ascii.fixed_width",
                fill_values=[(ascii.masked, "--")],
                overwrite=True,
            )

    # Comparison figure: subtracted vs unsubtracted, 5" and 10".
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, constrained_layout=True)
    series = [
        ("dia_ann", "o", "-", 1.0, "DIA + annulus"),
        ("dia_noann", "s", "--", 0.8, "DIA no annulus"),
        ("direct_noann", ".", ":", 0.55, "no subtraction"),
    ]
    label = "5"
    for axis, (filter_name, color) in zip(axes.flat, FILTER_COLORS.items()):
        subset = table[np.asarray(table["filter"]) == filter_name]
        mjd = np.asarray(subset["mjd"], dtype=float)
        for variant, marker, linestyle, alpha, name in series:
            detected = np.asarray(subset[f"{variant}_detected_{label}"], dtype=bool)
            axis.errorbar(
                mjd[detected],
                np.asarray(subset[f"{variant}_mag_{label}"], dtype=float)[detected],
                yerr=np.asarray(subset[f"{variant}_magerr_{label}"], dtype=float)[detected],
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
        axis.set_title(filter_name)
        axis.set_xlabel("MJD")
        axis.set_ylabel(f"Magnitude ({label}\" aperture)")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
    fig.suptitle(
        "SN 2026kid — DIA vs direct forced photometry at the refined coordinate "
        f"(RA {ra:.6f}, Dec {dec:+.6f}; open triangles = DIA 5-sigma limits)"
    )
    comparison_png = OUTPUT_DIR / "SN2026kid_LOAO_dia_vs_direct.png"
    fig.savefig(comparison_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    detections = {
        variant: int(
            np.sum(np.asarray(table[f"{variant}_detected_5"], dtype=bool))
        )
        for variant, *_ in series
    }
    verdict = "PASS" if not failures and len(table) else "FAIL"
    lines = [
        "DIA Stage C (fixed-coordinate photometry) QC summary — created "
        + datetime.now(timezone.utc).isoformat(timespec="seconds"),
        f"refined coordinate    : RA {ra:.7f}  Dec {dec:+.7f}",
        f"  from {coordinate_info['n_centroids']} difference-image centroids, "
        f"scatter {coordinate_info['scatter_median_arcsec']:.2f}\", offset from "
        f"discovery coordinate {coordinate_info['offset_from_discovery_arcsec']:.2f}\"",
        f"measurements          : {len(table)} stacks x 4 variants x 3 apertures",
        f"failures              : {len(failures)}",
        f"detections (5\")       : " + ", ".join(f"{k}={v}" for k, v in detections.items()),
        "variants              : dia_ann / dia_noann / direct_ann / direct_noann",
        f"annulus               : width {ANNULUS_WIDTH_ARCSEC}\" beyond max aperture",
        "",
        *[f"  FAIL {f['path']}: {f['error']}" for f in failures],
        f"VALIDATION_RESULT={verdict}",
    ]
    (QC_DIR / "13_diaphot_summary.log").write_text("\n".join(lines) + "\n")
    summary = {
        "coordinate": coordinate_info,
        "detections_5arcsec": detections,
        "failures": failures,
        "outputs": [str(comparison_png)],
    }
    (OUTPUT_DIR / "SN2026kid_LOAO_dia_photometry.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    print(f"VALIDATION_RESULT={verdict}")


if __name__ == "__main__":
    main()
