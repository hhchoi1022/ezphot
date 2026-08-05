#!/usr/bin/env python
"""Validate LOAO preprocessing products and render representative QC panels."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


BATCH_REPORT = Path(
    "/home/hhchoi1022/ezphot/data/scidata/LOAO/SN2026kid/"
    "preprocess_all_obsdates_report.json"
)
QC_OUTPUT = Path(
    "/home/hhchoi1022/ezphot/log/20260805_223146_codex_run/"
    "1_preprocess_qc.png"
)
FILTERS = ("B102", "V103", "R104", "I105")
QC_NIGHTS = ("2026_0501", "2026_0623")


def validate(batch: dict) -> tuple[list[dict], list[str], dict]:
    errors: list[str] = []
    products: list[dict] = []
    flatages: Counter = Counter()
    filters: Counter = Counter()
    detectors: Counter = Counter()
    statuses: Counter = Counter()

    for row in batch["nights"]:
        statuses[row["status"]] += 1
        if row["status"] != "complete":
            errors.append(f"night_failed:{row['night']}")
        if not row.get("raw_files_unchanged"):
            errors.append(f"raw_changed:{row['night']}")
        report_path = Path(row["report"])
        if not report_path.is_file():
            errors.append(f"missing_report:{report_path}")
            continue
        nightly = json.loads(report_path.read_text())
        detector = nightly["detector"]["telkey"]
        for product in nightly["products"]:
            products.append(product)
            detectors[detector] += 1
            filters[product["filter"]] += 1
            output = Path(product["output"])
            mask = Path(product["bpmask"])
            if not output.is_file():
                errors.append(f"missing_output:{output}")
                continue
            if not mask.is_file():
                errors.append(f"missing_mask:{mask}")
                continue
            header = fits.getheader(output, memmap=False)
            mask_header = fits.getheader(mask, memmap=False)
            for key in ("BIASCOR", "DARKCOR", "FLATCOR", "BPMCOR"):
                if header.get(key) is not True:
                    errors.append(f"{output}:{key}")
            for key in (
                "CHECKSUM",
                "DATASUM",
                "RAWPATH",
                "BIASPATH",
                "DARKPATH",
                "FLATPATH",
                "FLATNITE",
                "FLATAGE",
            ):
                if key not in header:
                    errors.append(f"{output}:missing_{key}")
            if header.get("OBJECT") != "SN2026kid":
                errors.append(f"{output}:OBJECT")
            if int(header.get("NBADPIX", -1)) != int(product["bad_pixels"]):
                errors.append(f"{output}:NBADPIX")
            if mask_header.get("TARGET") != output.name:
                errors.append(f"{mask}:TARGET")
            if "CHECKSUM" not in mask_header or "DATASUM" not in mask_header:
                errors.append(f"{mask}:checksum_header")
            flatages[int(header.get("FLATAGE", -1))] += 1

    summary = {
        "night_status_counts": dict(statuses),
        "product_count": len(products),
        "detector_counts": dict(detectors),
        "filter_counts": dict(filters),
        "flat_age_days_counts": dict(sorted(flatages.items())),
        "science_files_present": sum(
            Path(product["output"]).is_file() for product in products
        ),
        "bpmask_files_present": sum(
            Path(product["bpmask"]).is_file() for product in products
        ),
        "raw_files_unchanged_nights": sum(
            bool(row.get("raw_files_unchanged")) for row in batch["nights"]
        ),
        "night_count": len(batch["nights"]),
        "header_provenance_errors": len(errors),
    }
    return products, errors, summary


def display_image(ax, path: Path, title: str) -> None:
    data = np.asarray(fits.getdata(path, memmap=False), dtype=np.float32)
    finite = data[np.isfinite(data)]
    vmin, vmax = np.percentile(finite, (1, 99))
    step = max(1, int(np.ceil(max(data.shape) / 1200)))
    ax.imshow(
        data[::step, ::step],
        origin="lower",
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def make_qc(batch: dict, output: Path) -> None:
    rows_by_night = {row["night"]: row for row in batch["nights"]}
    figure, axes = plt.subplots(4, 4, figsize=(15, 14), constrained_layout=True)
    for detector_index, night in enumerate(QC_NIGHTS):
        nightly = json.loads(Path(rows_by_night[night]["report"]).read_text())
        detector = nightly["detector"]["telkey"]
        by_filter: dict[str, dict] = {}
        for product in nightly["products"]:
            by_filter.setdefault(product["filter"], product)
        raw_row = 2 * detector_index
        calibrated_row = raw_row + 1
        for column, filter_name in enumerate(FILTERS):
            product = by_filter[filter_name]
            raw_path = Path(product["raw"])
            output_path = Path(product["output"])
            raw_data = fits.getdata(raw_path, memmap=False)
            raw_median = float(np.nanmedian(raw_data))
            display_image(
                axes[raw_row, column],
                raw_path,
                f"{filter_name} raw\nmedian={raw_median:.3g}",
            )
            display_image(
                axes[calibrated_row, column],
                output_path,
                f"{filter_name} preprocessed\n"
                f"median={product['median']:.3g}, bad={product['bad_pixels']}",
            )
        axes[raw_row, 0].set_ylabel(f"{night}\n{detector}\nRAW", fontsize=10)
        axes[calibrated_row, 0].set_ylabel(
            f"{night}\n{detector}\nBDF + BPM", fontsize=10
        )
    figure.suptitle(
        "LOAO SN2026kid preprocessing QC — raw vs. BIAS/DARK/FLAT corrected\n"
        "Each panel uses an independent 1st–99th percentile stretch",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-report", type=Path, default=BATCH_REPORT)
    parser.add_argument("--qc-output", type=Path, default=QC_OUTPUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    batch = json.loads(args.batch_report.read_text())
    _, errors, summary = validate(batch)
    print("===== POST-RUN PREPROCESS VALIDATION =====")
    for key, value in summary.items():
        print(f"{key}={value}")
    for error in errors[:20]:
        print(f"ERROR {error}")
    passed = not errors and summary["product_count"] == 1020
    print(f"VALIDATION_RESULT={'PASS' if passed else 'FAIL'}")
    make_qc(batch, args.qc_output)
    print(f"QC_PNG={args.qc_output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
