#!/usr/bin/env python
"""Run only astrometric calibration for all preprocessed LOAO SN2026kid images."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table


SCRIPT_DIR = Path(__file__).resolve().parent
PHOTOMETRY_SCRIPT = SCRIPT_DIR / "260805_photometry_loao_sn2026kid.py"
PREPROCESS_REPORT = Path(
    "/home/hhchoi1022/ezphot/data/scidata/LOAO/SN2026kid/"
    "preprocess_all_obsdates_report.json"
)
ASTROMETRY_REPORT = Path(
    "/home/hhchoi1022/ezphot/data/scidata/LOAO/SN2026kid/"
    "astrometry_all_obsdates_report.json"
)
REFERENCE_CATALOG = Path(
    "/home/hhchoi1022/ezphot/data/scidata/LOAO/SN2026kid/"
    "reference.APASS_DR9_BVRI.ecsv"
)


def load_photometry_module():
    spec = importlib.util.spec_from_file_location("loao_photometry", PHOTOMETRY_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(PHOTOMETRY_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PHOTOMETRY = load_photometry_module()
WORKER_REFERENCE: Table | None = None


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, path)


def load_paths(preprocess_report: Path, limit: int | None) -> list[Path]:
    batch = json.loads(preprocess_report.read_text())
    paths: list[Path] = []
    for row in batch["nights"]:
        nightly = json.loads(Path(row["report"]).read_text())
        paths.extend(Path(product["output"]) for product in nightly["products"])
    paths = sorted(paths)
    return paths[:limit] if limit is not None else paths


def prepare_reference(paths: list[Path], reference_path: Path) -> Table:
    telescope_info = PHOTOMETRY.telinfo(paths[0])
    first = PHOTOMETRY.image_instance(paths[0], telescope_info)
    center = SkyCoord(first.ra, first.dec, unit="deg")
    reference = PHOTOMETRY.query_reference_catalog(
        center, reference_path, overwrite=False
    )
    first.clear(clear_data=True, clear_header=True, verbose=False)
    return reference


def initialize_worker(reference_path: str) -> None:
    global WORKER_REFERENCE
    WORKER_REFERENCE = Table.read(reference_path, format="ascii.ecsv")


def solve_one(path_string: str, overwrite: bool) -> dict:
    if WORKER_REFERENCE is None:
        raise RuntimeError("Astrometry worker reference catalog was not initialized")
    path = Path(path_string)
    telescope_info = PHOTOMETRY.telinfo(path)
    probe = PHOTOMETRY.image_instance(path, telescope_info)
    pointing = SkyCoord(probe.ra, probe.dec, unit="deg")
    probe.clear(clear_data=True, clear_header=True, verbose=False)
    _, result = PHOTOMETRY.solve_astrometry_safe(
        path,
        telescope_info,
        WORKER_REFERENCE,
        overwrite=overwrite,
        verbose=False,
    )
    PHOTOMETRY.verify_checksum(path)
    metrics = PHOTOMETRY.wcs_metrics(path, pointing)
    header_after = fits.getheader(path, memmap=False)
    if header_after.get("ASTRMCOR") is not True:
        raise ValueError(f"ASTRMCOR is not true: {path}")
    detector = (
        "LOAO_iKon_1x1"
        if "LOAO_iKon_1x1" in str(path)
        else "LOAO_E2V_2x2"
    )
    return {
        "path": str(path),
        "night": str(header_after.get("RAWNIGHT", "")),
        "filter": str(header_after.get("FILTER", "")),
        "detector": detector,
        "status": result["status"],
        "method": result.get("method", "existing WCS"),
        "matches": result.get("matches"),
        **metrics,
    }


def run(args: argparse.Namespace) -> tuple[dict, int]:
    paths = load_paths(args.preprocess_report, args.limit)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} preprocessed images")
    reference = prepare_reference(paths, args.reference_catalog)
    report = {
        "stage": "2_astrometry",
        "started_utc": now_utc(),
        "preprocess_report": str(args.preprocess_report),
        "reference_catalog": str(args.reference_catalog),
        "reference_source": "SkyCatalog APASS DR9 (fallback only)",
        "image_count": len(paths),
        "workers": args.workers,
        "results": [],
        "failures": [],
    }
    write_report(report, args.report_path)
    print(f"Astrometry inputs: {len(paths)}", flush=True)
    print(f"APASS fallback stars: {len(reference)}", flush=True)
    print(f"Workers: {args.workers}", flush=True)

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=initialize_worker,
        initargs=(str(args.reference_catalog),),
    ) as executor:
        futures = {
            executor.submit(solve_one, str(path), args.overwrite): path
            for path in paths
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            path = futures[future]
            try:
                result = future.result()
                report["results"].append(result)
                print(
                    f"[{completed:04d}/{len(paths):04d}] OK "
                    f"{path.name} | {result['method']} | "
                    f"scale={result['pixel_scale_arcsec']:.4f} arcsec/pix | "
                    f"offset={result['pointing_separation_arcsec']:.1f} arcsec",
                    flush=True,
                )
            except Exception as error:
                failure = {
                    "path": str(path),
                    "error": f"{type(error).__name__}: {error}",
                }
                report["failures"].append(failure)
                print(
                    f"[{completed:04d}/{len(paths):04d}] FAIL "
                    f"{path.name} | {failure['error']}",
                    flush=True,
                )
            write_report(report, args.report_path)

    report["finished_utc"] = now_utc()
    report["success_count"] = len(report["results"])
    report["failure_count"] = len(report["failures"])
    scales = [row["pixel_scale_arcsec"] for row in report["results"]]
    offsets = [row["pointing_separation_arcsec"] for row in report["results"]]
    report["summary"] = {
        "pixel_scale_min": float(np.min(scales)) if scales else None,
        "pixel_scale_median": float(np.median(scales)) if scales else None,
        "pixel_scale_max": float(np.max(scales)) if scales else None,
        "pointing_offset_min_arcsec": float(np.min(offsets)) if offsets else None,
        "pointing_offset_median_arcsec": float(np.median(offsets)) if offsets else None,
        "pointing_offset_max_arcsec": float(np.max(offsets)) if offsets else None,
    }
    write_report(report, args.report_path)
    print(f"Report: {args.report_path}", flush=True)
    print(
        f"Astrometry complete: {report['success_count']}/{len(paths)}; "
        f"failures={report['failure_count']}",
        flush=True,
    )
    return report, 1 if report["failures"] else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocess-report", type=Path, default=PREPROCESS_REPORT)
    parser.add_argument("--reference-catalog", type=Path, default=REFERENCE_CATALOG)
    parser.add_argument("--report-path", type=Path, default=ASTROMETRY_REPORT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    _, returncode = run(build_parser().parse_args())
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
