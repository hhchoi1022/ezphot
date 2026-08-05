#!/usr/bin/env python
"""Run only the preprocessing stage for all LOAO SN2026kid nights."""

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
MANIFEST = Path(
    "/home/hhchoi1022/ezphot/data/mcalibdata/LOAO/loao_master_manifest.csv"
)
TARGET = "SN2026kid"
SOURCE_NAMES = {"ngc5907", "sn2026kid"}
ONE_NIGHT_SCRIPT = Path(__file__).with_name("260805_process_loao_sn2026kid.py")
REPORT_DEFAULT = (
    SCIENCE_ROOT / "LOAO" / TARGET / "preprocess_all_obsdates_report.json"
)


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
            if any(name in path.name.lower() for name in SOURCE_NAMES)
        ]
        if any(
            str(fits.getheader(path, memmap=False).get("OBJECT", "")).lower()
            in SOURCE_NAMES
            for path in candidates
        ):
            nights.append(directory.name)
    return nights


def write_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> tuple[dict, int]:
    nights = target_nights(args.raw_base, args.start, args.end)
    report = {
        "stage": "1_preprocess",
        "target": TARGET,
        "start": args.start,
        "end": args.end,
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "night_count": len(nights),
        "nights": [],
    }
    write_report(report, args.report_path)

    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = "/tmp/ezphot-matplotlib"
    Path(environment["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    for index, night in enumerate(nights, start=1):
        print(f"\n===== PREPROCESS [{index:02d}/{len(nights):02d}] {night} =====", flush=True)
        command = [
            sys.executable,
            str(ONE_NIGHT_SCRIPT),
            "--night",
            night,
            "--raw-base",
            str(args.raw_base),
            "--manifest",
            str(args.manifest),
            "--output-root",
            str(args.science_root),
            "--max-flat-days",
            str(args.max_flat_days),
        ]
        if args.overwrite:
            command.append("--overwrite")
        returncode = subprocess.run(command, check=False, env=environment).returncode
        candidates = sorted(
            (args.science_root / "LOAO").glob(
                f"*/{TARGET}/preprocess_{night}_report.json"
            )
        )
        row = {"night": night, "returncode": returncode}
        if returncode == 0 and candidates:
            nightly_report = json.loads(candidates[-1].read_text())
            products = nightly_report.get("products", [])
            row.update(
                {
                    "status": "complete",
                    "detector": nightly_report["detector"]["telkey"],
                    "product_count": len(products),
                    "raw_files_unchanged": nightly_report.get(
                        "raw_files_unchanged", False
                    ),
                    "report": str(candidates[-1]),
                }
            )
        else:
            row["status"] = "failed"
        report["nights"].append(row)
        write_report(report, args.report_path)

    report["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report["complete_nights"] = sum(
        row["status"] == "complete" for row in report["nights"]
    )
    report["failed_nights"] = len(nights) - report["complete_nights"]
    report["product_count"] = sum(
        row.get("product_count", 0) for row in report["nights"]
    )
    write_report(report, args.report_path)
    print(f"\nBatch report: {args.report_path}", flush=True)
    print(
        f"Completed nights: {report['complete_nights']}/{len(nights)}; "
        f"products: {report['product_count']}; failed nights: {report['failed_nights']}",
        flush=True,
    )
    return report, 1 if report["failed_nights"] else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2026_0420")
    parser.add_argument("--end", default="2026_0805")
    parser.add_argument("--raw-base", type=Path, default=RAW_BASE)
    parser.add_argument("--science-root", type=Path, default=SCIENCE_ROOT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--report-path", type=Path, default=REPORT_DEFAULT)
    parser.add_argument("--max-flat-days", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    _, returncode = run(build_parser().parse_args())
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
