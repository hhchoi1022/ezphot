#!/usr/bin/env python
"""Wait for the two LOAO batch workers and merge their atomic reports."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


REPORT_DIR = Path("/home/hhchoi1022/ezphot/data/scidata/LOAO/SN2026kid")
EARLY = REPORT_DIR / "all_obsdates_early_report.json"
LATE = REPORT_DIR / "all_obsdates_late_report.json"
OUTPUT = REPORT_DIR / "all_obsdates_report.json"


def load_finished(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        report = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return report if report.get("finished_utc") else None


def atomic_write(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict:
    while True:
        early = load_finished(args.early)
        late = load_finished(args.late)
        if early is not None and late is not None:
            break
        time.sleep(args.poll_seconds)

    nights = sorted(
        [*early.get("nights", []), *late.get("nights", [])],
        key=lambda row: row["night"],
    )
    report = {
        "target": "SN2026kid",
        "start": early.get("start"),
        "end": late.get("end"),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_reports": [str(args.early), str(args.late)],
        "night_count": len(nights),
        "nights": nights,
        "complete_nights": sum(row.get("status") == "complete" for row in nights),
        "incomplete_nights": sum(row.get("status") != "complete" for row in nights),
    }
    atomic_write(args.output, report)
    print(f"Merged batch report: {args.output}", flush=True)
    print(
        f"Complete: {report['complete_nights']}; "
        f"incomplete/skipped: {report['incomplete_nights']}",
        flush=True,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--early", type=Path, default=EARLY)
    parser.add_argument("--late", type=Path, default=LATE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--poll-seconds", type=float, default=30)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
