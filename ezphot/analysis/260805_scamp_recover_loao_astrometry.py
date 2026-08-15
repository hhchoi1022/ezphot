#!/usr/bin/env python
"""Recover only failed LOAO astrometry frames with ezphot's SCAMP method."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import tempfile
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from ezphot.imageobjects import ScienceImage
from ezphot.methods import Platesolve


SCRIPT_DIR = Path(__file__).resolve().parent
PHOTOMETRY_SCRIPT = SCRIPT_DIR / "260805_photometry_loao_sn2026kid.py"
REPORT_DEFAULT = Path(
    "/home/hhchoi1022/ezphot/data/scidata/LOAO/SN2026kid/"
    "astrometry_all_obsdates_report.json"
)


def load_photometry_module():
    spec = importlib.util.spec_from_file_location("loao_photometry", PHOTOMETRY_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(PHOTOMETRY_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PHOTOMETRY = load_photometry_module()


def pointing_from_header(header: fits.Header) -> SkyCoord:
    ra = header.get("RA", header.get("CRVAL1"))
    dec = header.get("DEC", header.get("CRVAL2"))
    try:
        return SkyCoord(float(ra), float(dec), unit="deg")
    except (TypeError, ValueError):
        return SkyCoord(str(ra), str(dec), unit=(u.hourangle, u.deg))


def add_rough_wcs(path: Path) -> None:
    header = fits.getheader(path, memmap=False)
    center = pointing_from_header(header)
    detector_scale = 0.356 if "LOAO_iKon_1x1" in str(path) else 0.794
    scale_deg = detector_scale / 3600.0
    header["CRVAL1"] = (center.ra.deg, "Approximate pointing RA for SCAMP seed")
    header["CRVAL2"] = (center.dec.deg, "Approximate pointing Dec for SCAMP seed")
    header["CRPIX1"] = (float(header["NAXIS1"]) / 2.0 + 0.5, "SCAMP seed reference pixel")
    header["CRPIX2"] = (float(header["NAXIS2"]) / 2.0 + 0.5, "SCAMP seed reference pixel")
    header["CD1_1"] = -scale_deg
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = scale_deg
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    header["EQUINOX"] = 2000.0
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".seed.tmp", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        fits.writeto(temporary, fits.getdata(path, memmap=False), header, overwrite=True, checksum=True)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def solve_scamp(path: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="loao_scamp_", dir="/tmp") as work:
        work_path = Path(work) / "image.fits"
        shutil.copy2(path, work_path)
        add_rough_wcs(work_path)
        telescope_info = PHOTOMETRY.telinfo(path)
        image = PHOTOMETRY.image_instance(work_path, telescope_info)
        solver = Platesolve()
        solved = solver.solve_scamp(
            target_img=image,
            catalog_type="GAIA-DR2",
            scamp_sexparams={
                "DETECT_MINAREA": 4,
                "DETECT_THRESH": 3.0,
                "ANALYSIS_THRESH": 3.0,
            },
            overwrite=True,
            verbose=False,
        )[0]
        solved.header["ASTRMCOR"] = (True, "Astrometric WCS solved by SCAMP")
        solved.header["ASTRMMET"] = ("SCAMP", "Astrometry method")
        solved.header["ASTRMUTC"] = (PHOTOMETRY.now_utc(), "UTC astrometry completion time")
        solved.update_status("ASTROMETRY")
        PHOTOMETRY.persist_image_header(solved)
        solved_header = fits.getheader(work_path, memmap=False)
    with fits.open(path, mode="update", memmap=False) as hdul:
        hdul[0].header.update(solved_header)
        hdul[0].add_checksum()
        hdul.flush()
    PHOTOMETRY.verify_checksum(path)
    metrics = PHOTOMETRY.wcs_metrics(path, pointing_from_header(fits.getheader(path)))
    metrics.update({"status": "solved", "method": "SCAMP", "path": str(path)})
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPORT_DEFAULT)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    failures = list(report["failures"])
    failures.extend({"path": row["path"], "reason": "replace legacy fallback"}
                    for row in report["results"]
                    if row.get("method") not in ("astrometry.net", "SCAMP"))
    failures = failures[: args.limit] if args.limit else failures
    remaining = []
    recovered = []
    for failure in failures:
        path = Path(failure["path"])
        try:
            metrics = solve_scamp(path)
            recovered.append(metrics)
            print(
                f"SCAMP OK {path.name} | scale={metrics['pixel_scale_arcsec']:.4f} "
                f"arcsec/pix | offset={metrics['pointing_separation_arcsec']:.1f} arcsec",
                flush=True,
            )
        except Exception as error:
            row = {"path": str(path), "error": f"{type(error).__name__}: {error}"}
            remaining.append(row)
            print(f"SCAMP FAIL {path.name} | {row['error']}", flush=True)
    if args.limit:
        print(f"SCAMP dry subset complete: {len(recovered)} success, {len(remaining)} failed")
        return 1 if remaining else 0
    replaced = {r["path"] for r in recovered}
    report["results"] = [r for r in report["results"] if r["path"] not in replaced]
    report["results"].extend(recovered)
    report["failures"] = remaining
    report["success_count"] = len(report["results"])
    report["failure_count"] = len(remaining)
    temporary = args.report.with_name(f".{args.report.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, args.report)
    print(f"SCAMP recovery: {len(recovered)} success, {len(remaining)} failed")
    return 1 if remaining else 0


if __name__ == "__main__":
    raise SystemExit(main())
