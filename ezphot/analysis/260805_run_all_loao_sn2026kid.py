#!/usr/bin/env python
"""Preprocess and photometer every available LOAO NGC5907/SN2026kid night.

The driver is deliberately sequential because SWarp and the figure-generation
steps are already multithreaded and memory intensive.  Each component workflow
is resumable: existing calibrated images, photometry products, stacks, and
figures are reused after their provenance checks pass.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from astropy.io import fits


RAW_BASE = Path("/qso/data6/obsdata/LOAO")
SCIENCE_ROOT = Path("/home/hhchoi1022/ezphot/data/scidata")
TARGET = "SN2026kid"
TELESCOPE = "1.0-m KASI"
SOURCE_NAMES = ("NGC5907", "SN2026kid", "ngc5907", "sn2026kid")
SCRIPT_DIR = Path(__file__).resolve().parent
PREPROCESS_SCRIPT = SCRIPT_DIR / "260805_process_loao_sn2026kid.py"
PHOTOMETRY_SCRIPT = SCRIPT_DIR / "260805_photometry_loao_sn2026kid.py"
MANIFEST = Path(
    "/home/hhchoi1022/ezphot/data/mcalibdata/LOAO/loao_master_manifest.csv"
)
REPORT_PATH = SCIENCE_ROOT / "LOAO" / TARGET / "all_obsdates_report.json"
BACKGROUND_MASK_VERSION = "NGC5907_SIMBAD_1.25x_v1"
STACK_CENTER_VERSION = "NGC5907_SIMBAD_v1"


def parse_night(value: str) -> datetime:
    return datetime.strptime(value, "%Y_%m%d")


def target_nights(raw_base: Path, start: str, end: str) -> list[str]:
    start_date = parse_night(start)
    end_date = parse_night(end)
    nights = []
    for directory in sorted(raw_base.glob("20??_????")):
        try:
            date = parse_night(directory.name)
        except ValueError:
            continue
        if not start_date <= date <= end_date:
            continue
        candidates = [
            path
            for path in directory.glob("*.fits")
            if any(name.lower() in path.name.lower() for name in SOURCE_NAMES)
        ]
        # Some calibration files have misleading NGC5907 filenames (notably
        # the 2026-05-23 B flats).  Match the same OBJECT-header criterion as
        # the one-night preprocessor so those dates are not false positives.
        found = any(
            str(fits.getheader(path, memmap=False).get("OBJECT", "")).lower()
            in {name.lower() for name in SOURCE_NAMES}
            for path in candidates
        )
        if found:
            nights.append(directory.name)
    return nights


def atomic_report(report: dict, report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name(f".{report_path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, report_path)


def run_command(command: list[str]) -> int:
    print("COMMAND:", " ".join(command), flush=True)
    return subprocess.run(command, check=False).returncode


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def complete_photometry_report(science_root: Path, night: str) -> tuple[Path, dict] | None:
    """Return a previously completed nightly report for safe batch resume."""
    candidates = sorted(
        (science_root / "LOAO").glob(
            f"*/{TARGET}/{TELESCOPE}/photometry_{night}_report.json"
        )
    )
    for path in candidates:
        current = load_json(path)
        if (
            not current.get("failures")
            and current.get("background_mask_version") == BACKGROUND_MASK_VERSION
            and current.get("stack_center", {}).get("version")
            == STACK_CENTER_VERSION
            and len(current.get("images", [])) > 0
            and len(current.get("stacks", [])) == 4
            and len(current.get("stack_photometry", [])) == 4
        ):
            return path, current
    return None


def run(args: argparse.Namespace) -> dict:
    nights = target_nights(args.raw_base, args.start, args.end)
    report = {
        "target": TARGET,
        "start": args.start,
        "end": args.end,
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "night_count": len(nights),
        "nights": [],
    }
    atomic_report(report, args.report_path)

    for index, night in enumerate(nights, start=1):
        print(f"\n===== [{index:02d}/{len(nights):02d}] {night} =====", flush=True)
        completed = complete_photometry_report(args.science_root, night)
        if completed is not None:
            photometry_path, current = completed
            input_root = Path(current["input_root"])
            result = {
                "night": night,
                "status": "complete",
                "reused": True,
                "telkey": input_root.parents[1].name,
                "photometry_report": str(photometry_path),
                "image_results": len(current["images"]),
                "stack_results": len(current["stacks"]),
                "stack_photometry_results": len(current["stack_photometry"]),
                "failures": [],
            }
            report["nights"].append(result)
            atomic_report(report, args.report_path)
            print(f"REUSE complete report: {photometry_path}", flush=True)
            continue
        result = {"night": night, "status": "running"}
        report["nights"].append(result)
        atomic_report(report, args.report_path)

        preprocess_command = [
            sys.executable,
            str(PREPROCESS_SCRIPT),
            "--night",
            night,
            "--raw-base",
            str(args.raw_base),
            "--manifest",
            str(args.manifest),
            "--output-root",
            str(args.science_root),
        ]
        preprocess_code = run_command(preprocess_command)
        result["preprocess_returncode"] = preprocess_code
        preprocess_reports = sorted(
            (args.science_root / "LOAO").glob(
                f"*/{TARGET}/preprocess_{night}_report.json"
            )
        )
        if preprocess_code != 0 or not preprocess_reports:
            result["status"] = "skipped_or_failed_preprocess"
            atomic_report(report, args.report_path)
            continue

        preprocess_report = load_json(preprocess_reports[-1])
        telkey = preprocess_report["detector"]["telkey"]
        result["telkey"] = telkey
        result["preprocess_report"] = str(preprocess_reports[-1])
        result["science_count"] = len(preprocess_report["products"])
        input_root = args.science_root / "LOAO" / telkey / TARGET / TELESCOPE

        photometry_command = [
            sys.executable,
            str(PHOTOMETRY_SCRIPT),
            "--night",
            night,
            "--input-root",
            str(input_root),
        ]
        if args.skip_png:
            photometry_command.append("--skip-png")
        photometry_code = run_command(photometry_command)
        result["photometry_returncode"] = photometry_code
        photometry_report = input_root / f"photometry_{night}_report.json"
        if photometry_report.exists():
            result["photometry_report"] = str(photometry_report)
            current = load_json(photometry_report)
            result["image_results"] = len(current.get("images", []))
            result["stack_results"] = len(current.get("stacks", []))
            result["stack_photometry_results"] = len(
                current.get("stack_photometry", [])
            )
            result["failures"] = current.get("failures", [])
        result["status"] = "complete" if photometry_code == 0 else "photometry_failed"
        atomic_report(report, args.report_path)

    report["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report["complete_nights"] = sum(
        row["status"] == "complete" for row in report["nights"]
    )
    report["incomplete_nights"] = len(report["nights"]) - report["complete_nights"]
    atomic_report(report, args.report_path)
    print(f"Batch report: {args.report_path}", flush=True)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2026_0420")
    parser.add_argument("--end", default="2026_0805")
    parser.add_argument("--raw-base", type=Path, default=RAW_BASE)
    parser.add_argument("--science-root", type=Path, default=SCIENCE_ROOT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--report-path", type=Path, default=REPORT_PATH)
    parser.add_argument("--skip-png", action="store_true")
    return parser


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
