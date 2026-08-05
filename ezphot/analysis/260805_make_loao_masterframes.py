#!/usr/bin/env python
"""Build nightly LOAO master bias, dark, and flat frames.

This script is intentionally independent of the ezphot object graph.  The current
LOAO analysis drafts depend on import-time state and a combiner path that is not
safe for a long batch.  Only NumPy and Astropy are required here.

Raw observations are never modified.  Output follows the directory hierarchy
used by ``MasterImage``::

    <output-root>/LOAO/<telkey>/<BIAS|DARK|FLAT>/1.0-m KASI/

The requested 2026 interval spans a detector change:

* ``*.1x1.*`` files are assigned to ``LOAO_iKon_1x1``.
* ``*.2x2.*`` files are assigned to ``LOAO_E2V_2x2``.

Examples
--------
Inventory without writing::

    python 260805_make_loao_masterframes.py --dry-run

Build a single night in a temporary directory::

    python 260805_make_loao_masterframes.py \
        --start 2026_0424 --end 2026_0424 --output-root /tmp/loao-masters

Build the requested interval with four night-level workers::

    python 260805_make_loao_masterframes.py --workers 4 --register
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from astropy.io import fits


RAW_BASE_DEFAULT = Path("/qso/data6/obsdata/LOAO")
OUTPUT_ROOT_DEFAULT = Path("/home/hhchoi1022/ezphot/data/mcalibdata")
OBSERVATORY = "LOAO"
TELESCOPE_NAME = "1.0-m KASI"
GAIN_E_PER_ADU = 2.68

CAMERA_PROFILES = {
    "1x1": {
        "telkey": "LOAO_iKon_1x1",
        "ccd": "iKon",
        "binning": 1,
        "instrument": "iKon-L 936",
    },
    "2x2": {
        "telkey": "LOAO_E2V_2x2",
        "ccd": "E2V",
        "binning": 2,
        "instrument": "ARC Camera",
    },
}


@dataclass
class MasterRecord:
    night: str
    kind: str
    telkey: str
    ccd: str
    binning: int
    filter: str
    exptime: float
    ncombine: int
    path: str
    status: str
    median: float
    std: float
    minimum: float
    maximum: float
    finite_fraction: float


def parse_night(value: str) -> date:
    """Parse either YYYY_MMDD or ISO YYYY-MM-DD."""
    for fmt in ("%Y_%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass
    raise argparse.ArgumentTypeError(
        f"Invalid date {value!r}; expected YYYY_MMDD or YYYY-MM-DD"
    )


def night_date(path: Path) -> date | None:
    try:
        return datetime.strptime(path.name, "%Y_%m%d").date()
    except ValueError:
        return None


def find_nights(raw_base: Path, start: date, end: date) -> list[Path]:
    nights = []
    for path in sorted(raw_base.glob("20??_????")):
        parsed = night_date(path)
        if path.is_dir() and parsed is not None and start <= parsed <= end:
            nights.append(path)
    return nights


def calibration_paths(night_dir: Path) -> dict[str, list[Path]]:
    return {
        "BIAS": sorted(night_dir.glob("zero*.fits")),
        "DARK": sorted(night_dir.glob("dark*.fits")),
        "FLAT": sorted(
            set(night_dir.glob("ef*.fits")) | set(night_dir.glob("mf*.fits"))
        ),
    }


def filename_mode(path: Path) -> str | None:
    for mode in CAMERA_PROFILES:
        if f".{mode}." in path.name:
            return mode
    return None


def infer_profile(paths: Iterable[Path]) -> dict:
    """Infer one detector profile and reject mixed or unknown raw groups."""
    modes = {filename_mode(path) for path in paths}
    modes.discard(None)
    if len(modes) != 1:
        raise ValueError(f"Expected one LOAO detector mode, found {sorted(modes)}")
    return CAMERA_PROFILES[modes.pop()].copy()


def load_header(path: Path) -> fits.Header:
    return fits.getheader(path, memmap=False)


def load_data(path: Path) -> np.ndarray:
    data = fits.getdata(path, memmap=False)
    if data.ndim != 2:
        raise ValueError(f"Expected a 2-D FITS image: {path} has shape {data.shape}")
    return np.asarray(data, dtype=np.float32)


def median_combine(arrays: Sequence[np.ndarray]) -> np.ndarray:
    if not arrays:
        raise ValueError("Cannot combine an empty image list")
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        raise ValueError("All calibration inputs in a master must have the same shape")
    stack = np.stack(arrays, axis=0)
    combined = np.nanmedian(stack, axis=0).astype(np.float32)
    del stack
    return combined


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in value)


def master_path(
    output_root: Path,
    profile: dict,
    kind: str,
    night: str,
    filter_name: str | None = None,
    exptime: float | None = None,
) -> Path:
    folder = output_root / OBSERVATORY / profile["telkey"] / kind / TELESCOPE_NAME
    stem = f"master_{kind.lower()}_{night}"
    if kind == "DARK":
        stem += f"_{exptime:g}s"
    elif kind == "FLAT":
        stem += f"_{safe_name(str(filter_name))}"
    return folder / f"{stem}.fits"


def update_common_header(
    header: fits.Header,
    profile: dict,
    kind: str,
    night: str,
    source_paths: Sequence[Path],
) -> fits.Header:
    header = header.copy()
    header["TELESCOP"] = (TELESCOPE_NAME, "Telescope name")
    header["INSTRUME"] = (profile["instrument"], "Detector used for this master")
    header["CCD"] = (profile["ccd"], "ezphot detector identifier")
    header["TELKEY"] = (profile["telkey"], "ezphot telescope key")
    header["XBINNING"] = (profile["binning"], "Detector binning in X")
    header["YBINNING"] = (profile["binning"], "Detector binning in Y")
    header["GAIN"] = (GAIN_E_PER_ADU, "Detector gain [electron/ADU]")
    header["IMAGETYP"] = (kind, "Master calibration image type")
    header["IMGTYPE"] = (kind, "Master calibration image type")
    header["MASTER"] = (True, "This is a combined master frame")
    header["MSTRTYPE"] = (kind, "Master frame type")
    header["NCOMBINE"] = (len(source_paths), "Number of combined raw frames")
    header["COMBMETH"] = ("MEDIAN", "Combination method")
    header["NIGHTDIR"] = (night, "LOAO raw observation directory")
    header["DATE-MST"] = (
        datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "UTC master creation time",
    )
    for index, source in enumerate(source_paths, start=1):
        header.add_history(f"SOURCE {index:03d}: {source.name}")
    return header


def image_statistics(data: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(data)
    if not np.any(finite):
        raise ValueError("Master contains no finite pixels")
    values = data[finite]
    return {
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "finite_fraction": float(np.mean(finite)),
    }


def validate_existing(path: Path, expected: dict) -> dict[str, float]:
    header = fits.getheader(path, checksum=True)
    for key, value in expected.items():
        if str(header.get(key)) != str(value):
            raise ValueError(
                f"Existing output {path} has {key}={header.get(key)!r}, expected {value!r}"
            )
    return image_statistics(load_data(path))


def write_master(
    path: Path,
    data: np.ndarray,
    header: fits.Header,
    overwrite: bool,
) -> tuple[str, dict[str, float]]:
    expected = {
        "NIGHTDIR": header["NIGHTDIR"],
        "TELKEY": header["TELKEY"],
        "MSTRTYPE": header["MSTRTYPE"],
    }
    if path.exists() and not overwrite:
        return "existing", validate_existing(path, expected)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        fits.writeto(temporary, data, header, overwrite=True, checksum=True)
        fits.open(temporary, checksum=True).close()
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return "created" if not overwrite else "overwritten", image_statistics(data)


def make_record(
    night: str,
    kind: str,
    profile: dict,
    filter_name: str,
    exptime: float,
    ncombine: int,
    path: Path,
    status: str,
    statistics: dict[str, float],
) -> MasterRecord:
    return MasterRecord(
        night=night,
        kind=kind,
        telkey=profile["telkey"],
        ccd=profile["ccd"],
        binning=profile["binning"],
        filter=filter_name,
        exptime=exptime,
        ncombine=ncombine,
        path=str(path),
        status=status,
        **statistics,
    )


def build_night(night_dir: Path, output_root: Path, overwrite: bool) -> dict:
    night = night_dir.name
    paths = calibration_paths(night_dir)
    all_paths = paths["BIAS"] + paths["DARK"] + paths["FLAT"]
    if not all_paths:
        return {"night": night, "records": [], "warnings": ["no calibration frames"]}

    profile = infer_profile(all_paths)
    records: list[MasterRecord] = []
    warnings: list[str] = []

    if not paths["BIAS"]:
        raise RuntimeError(f"{night}: dark/flat data exist but no bias frames were found")

    bias_header = update_common_header(
        load_header(paths["BIAS"][0]), profile, "BIAS", night, paths["BIAS"]
    )
    bias_data = median_combine([load_data(path) for path in paths["BIAS"]])
    bias_output = master_path(output_root, profile, "BIAS", night)
    bias_status, bias_stats = write_master(
        bias_output, bias_data, bias_header, overwrite=overwrite
    )
    records.append(
        make_record(
            night,
            "BIAS",
            profile,
            str(bias_header.get("FILTER", "")),
            0.0,
            len(paths["BIAS"]),
            bias_output,
            bias_status,
            bias_stats,
        )
    )

    dark_groups: dict[float, list[Path]] = {}
    for path in paths["DARK"]:
        exptime = float(load_header(path).get("EXPTIME", -1))
        if exptime <= 0:
            raise ValueError(f"{path}: dark EXPTIME must be positive")
        dark_groups.setdefault(exptime, []).append(path)

    dark_products: dict[float, tuple[Path, np.ndarray]] = {}
    for exptime, group in sorted(dark_groups.items()):
        corrected = [load_data(path) - bias_data for path in group]
        dark_data = median_combine(corrected)
        del corrected
        dark_header = update_common_header(
            load_header(group[0]), profile, "DARK", night, group
        )
        dark_header["EXPTIME"] = (exptime, "Master dark exposure time [s]")
        dark_header["BIASCOR"] = (True, "Bias subtracted before combination")
        dark_header["BIASFILE"] = (bias_output.name, "Master bias used")
        dark_output = master_path(
            output_root, profile, "DARK", night, exptime=exptime
        )
        dark_status, dark_stats = write_master(
            dark_output, dark_data, dark_header, overwrite=overwrite
        )
        dark_products[exptime] = (dark_output, dark_data)
        records.append(
            make_record(
                night,
                "DARK",
                profile,
                str(dark_header.get("FILTER", "")),
                exptime,
                len(group),
                dark_output,
                dark_status,
                dark_stats,
            )
        )

    if paths["FLAT"] and not dark_products:
        raise RuntimeError(f"{night}: flat data exist but no usable dark frames were found")

    flat_groups: dict[str, list[Path]] = {}
    for path in paths["FLAT"]:
        filter_name = str(load_header(path).get("FILTER", "")).strip()
        if not filter_name:
            raise ValueError(f"{path}: flat FILTER is missing")
        flat_groups.setdefault(filter_name, []).append(path)

    for filter_name, group in sorted(flat_groups.items()):
        normalized_flats = []
        normalization_values = []
        dark_exposures_used = []
        for path in group:
            header = load_header(path)
            flat_exptime = float(header.get("EXPTIME", -1))
            if flat_exptime <= 0:
                raise ValueError(f"{path}: flat EXPTIME must be positive")
            dark_exptime = min(dark_products, key=lambda value: abs(value - flat_exptime))
            _, dark_data = dark_products[dark_exptime]
            corrected = (
                load_data(path)
                - bias_data
                - dark_data * np.float32(flat_exptime / dark_exptime)
            )
            normalization = float(np.nanmedian(corrected))
            if not np.isfinite(normalization) or normalization <= 0:
                warnings.append(
                    f"{path.name}: rejected flat with invalid median {normalization}"
                )
                continue
            normalized_flats.append((corrected / normalization).astype(np.float32))
            normalization_values.append(normalization)
            dark_exposures_used.append(dark_exptime)

        if not normalized_flats:
            raise RuntimeError(f"{night}: no usable {filter_name} flats after normalization")

        flat_data = median_combine(normalized_flats)
        final_normalization = float(np.nanmedian(flat_data))
        if not np.isfinite(final_normalization) or final_normalization <= 0:
            raise RuntimeError(f"{night}: combined {filter_name} flat has invalid median")
        flat_data /= np.float32(final_normalization)
        del normalized_flats

        flat_header = update_common_header(
            load_header(group[0]), profile, "FLAT", night, group
        )
        flat_header["FILTER"] = (filter_name, "Master flat filter")
        flat_header["EXPTIME"] = (
            float(np.median([load_header(path)["EXPTIME"] for path in group])),
            "Median raw flat exposure [s]",
        )
        flat_header["BIASCOR"] = (True, "Bias subtracted before normalization")
        flat_header["DARKCOR"] = (True, "Exposure-scaled dark subtracted")
        flat_header["FLATNORM"] = (1.0, "Final master flat median")
        flat_header["BIASFILE"] = (bias_output.name, "Master bias used")
        flat_header["DARKFILE"] = (
            ",".join(f"{value:g}s" for value in sorted(set(dark_exposures_used))),
            "Master dark exposure(s) used",
        )
        flat_header["RAWFMIN"] = (
            float(min(normalization_values)),
            "Minimum corrected raw-flat median [ADU]",
        )
        flat_header["RAWFMAX"] = (
            float(max(normalization_values)),
            "Maximum corrected raw-flat median [ADU]",
        )
        flat_output = master_path(
            output_root, profile, "FLAT", night, filter_name=filter_name
        )
        flat_status, flat_stats = write_master(
            flat_output, flat_data, flat_header, overwrite=overwrite
        )
        records.append(
            make_record(
                night,
                "FLAT",
                profile,
                filter_name,
                float(flat_header["EXPTIME"]),
                len(normalization_values),
                flat_output,
                flat_status,
                flat_stats,
            )
        )

    return {
        "night": night,
        "profile": profile,
        "records": [asdict(record) for record in records],
        "warnings": warnings,
    }


def inventory_line(night_dir: Path) -> str:
    paths = calibration_paths(night_dir)
    all_paths = paths["BIAS"] + paths["DARK"] + paths["FLAT"]
    profile = infer_profile(all_paths) if all_paths else {"telkey": "unknown"}
    return (
        f"{night_dir.name}: telkey={profile['telkey']} "
        f"bias={len(paths['BIAS'])} dark={len(paths['DARK'])} flat={len(paths['FLAT'])}"
    )


def write_manifest(output_root: Path, records: list[dict]) -> Path:
    manifest = output_root / OBSERVATORY / "loao_master_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict] = {}
    if manifest.exists():
        with manifest.open(newline="") as file:
            existing = {row["path"]: row for row in csv.DictReader(file)}
    for record in records:
        existing[record["path"]] = record
    rows = sorted(
        existing.values(),
        key=lambda row: (
            row["night"],
            row["telkey"],
            row["kind"],
            row["filter"],
            float(row["exptime"]),
        ),
    )
    fields = list(asdict(MasterRecord("", "", "", "", 0, "", 0, 0, "", "", 0, 0, 0, 0, 0)).keys())
    temporary = manifest.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, manifest)
    return manifest


def register_summary(output_root: Path, records: list[dict]) -> Path:
    """Register generated masters in ezphot's fixed-width summary in one update."""
    import portalocker
    from astropy.table import Table, vstack

    summary = output_root / "summary.ascii_fixed_width"
    summary.parent.mkdir(parents=True, exist_ok=True)
    lock_path = summary.with_suffix(summary.suffix + ".lock")
    with portalocker.Lock(str(lock_path), timeout=120):
        if summary.exists():
            existing = Table.read(summary, format="ascii.fixed_width")
        else:
            existing = Table(
                names=(
                    "file",
                    "observatory",
                    "telkey",
                    "telname",
                    "imagetyp",
                    "file_size_bytes",
                    "modified_time",
                    "exptime",
                    "obsdate",
                    "filtername",
                    "group_id",
                ),
                dtype=("U512", "U32", "U64", "U64", "U16", "f8", "U32", "f8", "U32", "U32", "f8"),
            )

        existing_paths = {str(value) for value in existing["file"]}
        rows = []
        for record in records:
            path = Path(record["path"])
            relative = str(path.relative_to(output_root))
            if relative in existing_paths:
                continue
            header = fits.getheader(path)
            stat = path.stat()
            rows.append(
                {
                    "file": relative,
                    "observatory": OBSERVATORY,
                    "telkey": record["telkey"],
                    "telname": TELESCOPE_NAME,
                    "imagetyp": record["kind"],
                    "file_size_bytes": float(stat.st_size),
                    "modified_time": datetime.fromtimestamp(
                        stat.st_mtime, tz=timezone.utc
                    ).isoformat(timespec="seconds"),
                    "exptime": float(header.get("EXPTIME", record["exptime"])),
                    "obsdate": str(header.get("DATE-OBS", "")),
                    "filtername": str(header.get("FILTER", "")),
                    "group_id": -1.0,
                }
            )

        if rows:
            updated = vstack([existing, Table(rows=rows)], join_type="outer")
            updated.sort(["observatory", "telname", "imagetyp", "obsdate"])
            temporary = summary.with_suffix(summary.suffix + ".tmp")
            updated.write(temporary, format="ascii.fixed_width", overwrite=True)
            os.replace(temporary, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-base", type=Path, default=RAW_BASE_DEFAULT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT_DEFAULT)
    parser.add_argument("--start", type=parse_night, default=date(2026, 4, 20))
    parser.add_argument("--end", type=parse_night, default=date.today())
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--register",
        action="store_true",
        help="Batch-register outputs in <output-root>/summary.ascii_fixed_width",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.start > args.end:
        raise ValueError("--start must not be after --end")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if not args.raw_base.is_dir():
        raise FileNotFoundError(args.raw_base)

    nights = find_nights(args.raw_base, args.start, args.end)
    print(
        f"Found {len(nights)} LOAO night directories from {args.start} through {args.end}",
        flush=True,
    )
    for night in nights:
        print(inventory_line(night), flush=True)
    if args.dry_run:
        return 0

    results = []
    failures = []
    if args.workers == 1:
        for index, night in enumerate(nights, start=1):
            print(f"[{index}/{len(nights)}] Building {night.name}", flush=True)
            try:
                result = build_night(night, args.output_root, args.overwrite)
                results.append(result)
                print(
                    f"[{index}/{len(nights)}] Finished {night.name}: "
                    f"{len(result['records'])} masters",
                    flush=True,
                )
            except Exception as error:
                failures.append(
                    {
                        "night": night.name,
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(),
                    }
                )
                print(f"[FAIL] {night.name}: {failures[-1]['error']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(build_night, night, args.output_root, args.overwrite): night
                for night in nights
            }
            for index, future in enumerate(as_completed(futures), start=1):
                night = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(
                        f"[{index}/{len(nights)}] Finished {night.name}: "
                        f"{len(result['records'])} masters",
                        flush=True,
                    )
                except Exception as error:
                    failures.append(
                        {
                            "night": night.name,
                            "error": f"{type(error).__name__}: {error}",
                            "traceback": traceback.format_exc(),
                        }
                    )
                    print(f"[FAIL] {night.name}: {failures[-1]['error']}", flush=True)

    records = [record for result in results for record in result["records"]]
    manifest = write_manifest(args.output_root, records) if records else None
    summary = register_summary(args.output_root, records) if records and args.register else None
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw_base": str(args.raw_base),
        "output_root": str(args.output_root),
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "night_count": len(nights),
        "master_count": len(records),
        "manifest": str(manifest) if manifest else None,
        "summary": str(summary) if summary else None,
        "results": sorted(results, key=lambda result: result["night"]),
        "failures": sorted(failures, key=lambda failure: failure["night"]),
    }
    report_path = args.output_root / OBSERVATORY / "loao_master_run_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, report_path)

    print(f"Master records: {len(records)}", flush=True)
    print(f"Manifest: {manifest}", flush=True)
    print(f"Report: {report_path}", flush=True)
    if summary:
        print(f"Registered summary: {summary}", flush=True)
    if failures:
        print(f"Failed nights: {len(failures)}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
