#!/usr/bin/env python
"""Preprocess one LOAO night containing NGC5907 / SN2026kid images.

The script consumes the nightly masters produced by
``260805_make_loao_masterframes.py``.  It deliberately resolves exact-night
masters from that run's CSV manifest instead of using the generic master-frame
search: LOAO dark masters are normally longer than the science exposure and are
scaled by ``ccdproc`` during subtraction.

Raw files are read only.  Calibrated images are organized under the transient
name ``SN2026kid`` while the original ``NGC5907`` name and raw path are retained
in FITS provenance keywords.  A bad-pixel mask is also written beside every
calibrated image; masked pixels in the calibrated image are set to NaN.

Examples
--------
Preview the selected files and masters::

    python 260805_process_loao_sn2026kid.py --dry-run

Process the selected night::

    python 260805_process_loao_sn2026kid.py --night 2026_0714

Test safely in a temporary output tree::

    python 260805_process_loao_sn2026kid.py --output-root /tmp/loao-science
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
from astropy.io import fits

from ezphot.helper import Helper
from ezphot.imageobjects import MasterImage, ScienceImage
from ezphot.methods import Preprocess


RAW_BASE_DEFAULT = Path("/qso/data6/obsdata/LOAO")
MANIFEST_DEFAULT = Path(
    "/home/hhchoi1022/ezphot/data/mcalibdata/LOAO/loao_master_manifest.csv"
)
OUTPUT_ROOT_DEFAULT = Path("/home/hhchoi1022/ezphot/data/scidata")
NIGHT_DEFAULT = "2026_0714"
TARGET_NAME_DEFAULT = "SN2026kid"
SOURCE_NAMES = {"ngc5907", "sn2026kid"}
TELESCOPE_NAME = "1.0-m KASI"

# Conservative E2V defect criteria.  Most flagged pixels are severe master-flat
# outliers; the calibration-deviation threshold catches saturated bias/dark
# defects that happen to have a plausible flat response.
FLAT_LOW = 0.5
FLAT_HIGH = 1.5
CALIBRATION_DEVIATION_ADU = 100.0

CAMERA_PROFILES = {
    "1x1": {"telkey": "LOAO_iKon_1x1", "ccd": "iKon", "binning": 1},
    "2x2": {"telkey": "LOAO_E2V_2x2", "ccd": "E2V", "binning": 2},
}


def filename_mode(path: Path) -> str | None:
    for mode in CAMERA_PROFILES:
        if f".{mode}." in path.name:
            return mode
    instrument = str(fits.getheader(path, memmap=False).get("INSTRUME", "")).lower()
    if "ikon" in instrument:
        return "1x1"
    if "arc camera" in instrument or "e2v" in instrument:
        return "2x2"
    return None


def discover_science(night_dir: Path, limit: int | None = None) -> list[Path]:
    """Return target files based on both filename and OBJECT metadata."""
    candidates = [
        path
        for path in sorted(night_dir.glob("*.fits"))
        if any(name in path.name.lower() for name in SOURCE_NAMES)
    ]
    selected = []
    for path in candidates:
        object_name = str(fits.getheader(path, memmap=False).get("OBJECT", "")).lower()
        if object_name in SOURCE_NAMES:
            selected.append(path)
    return selected[:limit] if limit is not None else selected


def infer_profile(paths: Iterable[Path]) -> dict:
    modes = {filename_mode(path) for path in paths}
    if None in modes or len(modes) != 1:
        raise ValueError(f"Expected one recognized detector mode, found {sorted(modes)}")
    return CAMERA_PROFILES[modes.pop()].copy()


def infer_profile_from_manifest(path: Path, night: str) -> dict:
    """Use the nightly master inventory when old science headers lack INSTRUME."""
    with path.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["night"] == night]
    telkeys = {row["telkey"] for row in rows}
    matching = [
        profile.copy()
        for profile in CAMERA_PROFILES.values()
        if profile["telkey"] in telkeys
    ]
    if len(matching) != 1:
        raise ValueError(
            f"Cannot infer one detector for {night} from manifest telkeys {sorted(telkeys)}"
        )
    return matching[0]


def read_manifest(path: Path, night: str, telkey: str) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if row["telkey"] == telkey]
    if not any(row["night"] == night for row in selected):
        raise RuntimeError(f"No {night} / {telkey} masters found in {path}")
    return selected


def resolve_masters(
    rows: list[dict[str, str]],
    filters: Iterable[str],
    night: str,
    max_flat_days: int,
) -> tuple[Path, Path, dict[str, Path], dict[str, dict[str, int | str]]]:
    same_night_rows = [row for row in rows if row["night"] == night]
    by_kind: dict[str, list[dict[str, str]]] = {}
    for row in same_night_rows:
        by_kind.setdefault(row["kind"].upper(), []).append(row)

    if len(by_kind.get("BIAS", [])) != 1:
        raise RuntimeError("Expected exactly one same-night master bias")
    if not by_kind.get("DARK"):
        raise RuntimeError("No same-night master dark found")

    # Longest master maximizes the dark-current signal-to-noise ratio.  The
    # ezphot/ccdproc subtraction scales it to each science EXPTIME.
    dark_row = max(by_kind["DARK"], key=lambda row: float(row["exptime"]))
    bias_path = Path(by_kind["BIAS"][0]["path"])
    dark_path = Path(dark_row["path"])
    flat_rows = {row["filter"]: row for row in by_kind.get("FLAT", [])}
    flat_paths: dict[str, Path] = {}
    flat_provenance: dict[str, dict[str, int | str]] = {}
    science_date = datetime.strptime(night, "%Y_%m%d").date()
    for filter_name in sorted(set(filters)):
        flat_row = flat_rows.get(filter_name)
        if flat_row is None:
            candidates = [
                row
                for row in rows
                if row["kind"].upper() == "FLAT" and row["filter"] == filter_name
            ]
            if not candidates:
                raise RuntimeError(f"No compatible master flat for {filter_name}")
            flat_row = min(
                candidates,
                key=lambda row: (
                    abs(
                        (
                            datetime.strptime(row["night"], "%Y_%m%d").date()
                            - science_date
                        ).days
                    ),
                    row["night"],
                ),
            )
        flat_date = datetime.strptime(flat_row["night"], "%Y_%m%d").date()
        age_days = abs((flat_date - science_date).days)
        if age_days > max_flat_days:
            raise RuntimeError(
                f"Nearest {filter_name} master flat is {age_days} days from {night}; "
                f"limit is {max_flat_days} days"
            )
        flat_paths[filter_name] = Path(flat_row["path"])
        flat_provenance[filter_name] = {
            "night": flat_row["night"],
            "age_days": age_days,
        }

    for path in [bias_path, dark_path, *flat_paths.values()]:
        if not path.is_file():
            raise FileNotFoundError(path)
    return bias_path, dark_path, flat_paths, flat_provenance


def raw_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def load_master(path: Path, telinfo) -> MasterImage:
    image = MasterImage(path, telinfo=telinfo, load=False)
    image.header
    return image


def robust_center(data: np.ndarray) -> float:
    return float(np.nanmedian(np.asarray(data, dtype=np.float64)))


def make_bad_pixel_mask(
    bias: np.ndarray, dark: np.ndarray, flat: np.ndarray
) -> np.ndarray:
    bias_center = robust_center(bias)
    dark_center = robust_center(dark)
    return (
        ~np.isfinite(bias)
        | ~np.isfinite(dark)
        | ~np.isfinite(flat)
        | (np.abs(bias - bias_center) > CALIBRATION_DEVIATION_ADU)
        | (np.abs(dark - dark_center) > CALIBRATION_DEVIATION_ADU)
        | (flat <= FLAT_LOW)
        | (flat >= FLAT_HIGH)
    )


def target_filename(raw_name: str, target_name: str) -> str:
    result = raw_name
    for source in ("NGC5907", "ngc5907", "SN2026kid", "sn2026kid"):
        result = result.replace(source, target_name)
    return result


def output_directory(output_root: Path, telkey: str, target: str, filter_name: str) -> Path:
    return output_root / "LOAO" / telkey / target / TELESCOPE_NAME / filter_name


def add_fits_checksum(path: Path) -> None:
    with fits.open(path, mode="update", memmap=False) as hdul:
        hdul[0].add_checksum()
        hdul.flush()


def validate_product(path: Path, bpmask_path: Path, expected_shape: tuple[int, int]) -> None:
    with fits.open(path, checksum=True, memmap=False) as hdul:
        hdul.verify("exception")
        if hdul[0].verify_checksum() != 1 or hdul[0].verify_datasum() != 1:
            raise ValueError(f"Invalid FITS checksum: {path}")
        if hdul[0].data.shape != expected_shape:
            raise ValueError(f"Unexpected calibrated shape: {hdul[0].data.shape}")
        for key in ("BIASCOR", "DARKCOR", "FLATCOR", "BPMCOR"):
            if hdul[0].header.get(key) is not True:
                raise ValueError(f"{path}: {key} is not true")
    with fits.open(bpmask_path, checksum=True, memmap=False) as hdul:
        hdul.verify("exception")
        if hdul[0].verify_checksum() != 1 or hdul[0].verify_datasum() != 1:
            raise ValueError(f"Invalid FITS checksum: {bpmask_path}")
        if hdul[0].data.shape != expected_shape:
            raise ValueError(f"Unexpected mask shape: {hdul[0].data.shape}")


def image_statistics(data: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(data)
    values = data[finite]
    if not values.size:
        raise ValueError("Calibrated image contains no finite pixels")
    return {
        "finite_fraction": float(np.mean(finite)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def process(args: argparse.Namespace) -> dict:
    night_dir = args.raw_base / args.night
    if not night_dir.is_dir():
        raise FileNotFoundError(night_dir)
    science_paths = discover_science(night_dir, args.limit)
    if not science_paths:
        raise RuntimeError(f"No NGC5907/SN2026kid images found in {night_dir}")

    try:
        profile = infer_profile(science_paths)
    except ValueError:
        profile = infer_profile_from_manifest(args.manifest, args.night)
    headers = {path: fits.getheader(path, memmap=False) for path in science_paths}
    filters = [str(headers[path]["FILTER"]).strip() for path in science_paths]
    rows = read_manifest(args.manifest, args.night, profile["telkey"])
    bias_path, dark_path, flat_paths, flat_provenance = resolve_masters(
        rows, filters, args.night, args.max_flat_days
    )

    print(f"Night: {args.night}")
    print(f"Science files: {len(science_paths)}")
    print(f"Detector: {profile['telkey']}")
    print(f"Bias: {bias_path}")
    print(f"Dark: {dark_path}")
    for filter_name, path in sorted(flat_paths.items()):
        count = sum(value == filter_name for value in filters)
        provenance = flat_provenance[filter_name]
        source_note = (
            "same night"
            if provenance["age_days"] == 0
            else f"fallback {provenance['night']}, {provenance['age_days']} day"
        )
        print(f"Flat {filter_name}: {path} ({count} science files; {source_note})")
    if args.dry_run:
        return {"status": "dry-run", "science_count": len(science_paths)}

    helper = Helper()
    telinfo = helper.get_telinfo(
        telescope="LOAO", ccd=profile["ccd"], binning=profile["binning"]
    )
    header_fix = {
        "XBINNING": profile["binning"],
        "YBINNING": profile["binning"],
        "GAIN": float(telinfo["gain"]),
    }
    preprocess = Preprocess()
    bias_image = load_master(bias_path, telinfo)
    dark_image = load_master(dark_path, telinfo)
    flat_images = {
        filter_name: load_master(path, telinfo)
        for filter_name, path in flat_paths.items()
    }
    bias_data = np.asarray(bias_image.data)
    dark_data = np.asarray(dark_image.data)
    masks = {
        filter_name: make_bad_pixel_mask(
            bias_data, dark_data, np.asarray(flat_image.data)
        )
        for filter_name, flat_image in flat_images.items()
    }

    before = {path: raw_signature(path) for path in science_paths}
    products = []
    for index, raw_path in enumerate(science_paths, start=1):
        raw_header = headers[raw_path]
        filter_name = str(raw_header["FILTER"]).strip()
        savedir = output_directory(
            args.output_root, profile["telkey"], args.target_name, filter_name
        )
        output_name = target_filename(raw_path.name, args.target_name)
        output_path = savedir / output_name
        bpmask_path = savedir / f"{output_name}.bpmask"

        if output_path.exists() and not args.overwrite:
            print(f"[{index:02d}/{len(science_paths):02d}] SKIP {output_path}")
            products.append(
                {"raw": str(raw_path), "output": str(output_path), "status": "existing"}
            )
            continue

        science = ScienceImage(raw_path, telinfo=telinfo, load=False)
        science.header
        original_object = str(science.header.get("OBJECT", ""))
        for key, value in header_fix.items():
            science.header[key] = value
        science.header["ORIGOBJ"] = (original_object, "Original raw OBJECT value")
        science.header["OBJECT"] = (args.target_name, "Canonical transient target")
        science.header["RAWFILE"] = (raw_path.name, "Original LOAO raw filename")
        science.header["RAWNIGHT"] = (args.night, "LOAO raw night directory")

        calibrated = preprocess.correct_bdf(
            target_img=science,
            bias_image=bias_image,
            dark_image=dark_image,
            flat_image=flat_images[filter_name],
            save=False,
            verbose=False,
        )
        calibrated.filename = output_name
        calibrated.savedir = savedir
        mask = masks[filter_name]
        if calibrated.data.shape != mask.shape:
            raise ValueError(
                f"Shape mismatch for {raw_path}: science {calibrated.data.shape}, "
                f"mask {mask.shape}"
            )
        calibrated.data = np.asarray(calibrated.data, dtype=np.float32)
        calibrated.data[mask] = np.nan
        calibrated.header["OBJECT"] = args.target_name
        calibrated.header["ORIGOBJ"] = original_object
        calibrated.header["RAWPATH"] = str(raw_path)
        calibrated.header["BIASPATH"] = str(bias_path)
        calibrated.header["DARKPATH"] = str(dark_path)
        calibrated.header["FLATPATH"] = str(flat_paths[filter_name])
        calibrated.header["FLATNITE"] = (
            flat_provenance[filter_name]["night"],
            "Master-flat observation night",
        )
        calibrated.header["FLATAGE"] = (
            flat_provenance[filter_name]["age_days"],
            "Absolute master-flat age [day]",
        )
        calibrated.header["BPMCOR"] = (True, "Bad master-calibration pixels set to NaN")
        calibrated.header["NBADPIX"] = (int(np.count_nonzero(mask)), "Bad pixels masked")
        calibrated.header["BPMFLO"] = (FLAT_LOW, "Minimum accepted master-flat response")
        calibrated.header["BPMFHI"] = (FLAT_HIGH, "Maximum accepted master-flat response")
        calibrated.header["BPMDEV"] = (
            CALIBRATION_DEVIATION_ADU,
            "Maximum bias/dark deviation from median [ADU]",
        )
        calibrated.header["PROCUTC"] = (
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "UTC preprocessing time",
        )
        calibrated.write(verbose=False)
        add_fits_checksum(output_path)

        mask_header = fits.Header()
        mask_header["MASKTYPE"] = "badpixel"
        mask_header["OBJECT"] = args.target_name
        mask_header["FILTER"] = filter_name
        mask_header["TARGET"] = output_name
        mask_header["BIASPATH"] = str(bias_path)
        mask_header["DARKPATH"] = str(dark_path)
        mask_header["FLATPATH"] = str(flat_paths[filter_name])
        mask_header["FLATNITE"] = flat_provenance[filter_name]["night"]
        mask_header["FLATAGE"] = flat_provenance[filter_name]["age_days"]
        mask_header["NBADPIX"] = int(np.count_nonzero(mask))
        fits.writeto(
            bpmask_path,
            mask.astype(np.uint8),
            mask_header,
            overwrite=True,
            checksum=True,
        )
        validate_product(output_path, bpmask_path, mask.shape)
        stats = image_statistics(calibrated.data)
        product = {
            "raw": str(raw_path),
            "output": str(output_path),
            "bpmask": str(bpmask_path),
            "status": "processed",
            "filter": filter_name,
            "exptime": float(calibrated.header["EXPTIME"]),
            "bad_pixels": int(np.count_nonzero(mask)),
            **stats,
        }
        products.append(product)
        print(
            f"[{index:02d}/{len(science_paths):02d}] OK {output_name} "
            f"({filter_name}, median={stats['median']:.3f}, bad={product['bad_pixels']})"
        )

    raw_unchanged = all(raw_signature(path) == before[path] for path in science_paths)
    if not raw_unchanged:
        raise RuntimeError("At least one raw file size or modification time changed")

    report = {
        "night": args.night,
        "target": args.target_name,
        "source_objects": sorted(SOURCE_NAMES),
        "detector": profile,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw_files_unchanged": raw_unchanged,
        "masters": {
            "bias": str(bias_path),
            "dark": str(dark_path),
            "flats": {key: str(value) for key, value in flat_paths.items()},
            "flat_provenance": flat_provenance,
        },
        "products": products,
    }
    report_dir = args.output_root / "LOAO" / profile["telkey"] / args.target_name
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"preprocess_{args.night}_report.json"
    temporary = report_path.with_name(f".{report_path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, report_path)
    print(f"Report: {report_path}")
    print(f"Raw files unchanged: {raw_unchanged}")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--night", default=NIGHT_DEFAULT, help="Night as YYYY_MMDD")
    parser.add_argument("--raw-base", type=Path, default=RAW_BASE_DEFAULT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DEFAULT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT_DEFAULT)
    parser.add_argument("--target-name", default=TARGET_NAME_DEFAULT)
    parser.add_argument("--limit", type=int, help="Optional number of science files")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--max-flat-days",
        type=int,
        default=7,
        help="Maximum date separation for a same-detector fallback flat",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    process(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
