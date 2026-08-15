#!/usr/bin/env python
"""Run the LOAO SN2026kid photometry workflow one stage at a time.

``260805_photometry_loao_sn2026kid.py`` performs masking, background
estimation, photometry, calibration, and stacking in a single pass per night.
This runner loads that script as a module and exposes each step as a separate
``--stage`` so intermediate products can be inspected before continuing:

    mask       invalid mask + source mask + NGC 5907 host-galaxy mask
    bkg        SEP 2-D background map from each science image
    bkgrms     SEP background RMS map
    phot       Source Extractor aperture photometry (PNGs saved)
    calib      PhotometricCalibration against SkyCatalog/APASS (figures saved)
    stack      per-filter background-subtracted, ZP-scaled stacks
    stackphot  aperture photometry + calibration on the stacks

Without ``--night`` it discovers every night under both detector trees and
runs itself per night through a subprocess, capturing each night's output in
``<log-root>/<timestamp>_stage_run/<stage>_<night>.log`` and writing one JSON
report per night plus an aggregate report per stage. Products, parameters, and
header bookkeeping are identical to the single-pass script: each stage calls
the same functions with the same arguments.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("260805_photometry_loao_sn2026kid.py")
SCIDATA_LOAO = Path("/home/hhchoi1022/ezphot/data/scidata/LOAO")
ROOTS_DEFAULT = [
    SCIDATA_LOAO / "LOAO_E2V_2x2/SN2026kid/1.0-m KASI",
    SCIDATA_LOAO / "LOAO_iKon_1x1/SN2026kid/1.0-m KASI",
]
REFERENCE_DEFAULT = SCIDATA_LOAO / "SN2026kid/reference.APASS_DR9_BVRI.ecsv"
REPORT_DIR_DEFAULT = SCIDATA_LOAO / "SN2026kid/stage_reports"
REJECTS_DEFAULT = SCIDATA_LOAO / "SN2026kid/rejected_frames.json"
# Frames calibrated on fewer APASS stars than this keep their single-image
# photometry but are excluded from stack inputs (user decision 2026-08-07).
STACK_MIN_ZP_STARS = 5
LOG_ROOT_DEFAULT = Path("/home/hhchoi1022/ezphot/log")
STAGES = ("mask", "bkg", "bkgrms", "phot", "calib", "stack", "stackphot")


def load_photometry_module():
    spec = importlib.util.spec_from_file_location("loao_photometry", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def rejected_paths(rejects_file: Path) -> set[str]:
    """Frames excluded from every stage (e.g. cloudy frames with no stars)."""
    if not rejects_file.exists():
        return set()
    payload = json.loads(rejects_file.read_text())
    return {frame["path"] for frame in payload["frames"]}


def discover_by_night(
    roots: list[Path], rejects_file: Path = REJECTS_DEFAULT
) -> dict[str, list[Path]]:
    """Group every calibrated SN2026kid frame by its RAWNIGHT header."""
    from astropy.io import fits

    rejected = rejected_paths(rejects_file)
    filters = ("B102", "V103", "R104", "I105")
    nights: dict[str, list[Path]] = defaultdict(list)
    for root in roots:
        for filter_name in filters:
            for path in sorted((root / filter_name).glob("obj.SN2026kid.*.fits")):
                if ".com." in path.name or str(path) in rejected:
                    continue
                night = str(fits.getheader(path, memmap=False).get("RAWNIGHT", ""))
                if not night:
                    raise RuntimeError(f"Missing RAWNIGHT header: {path}")
                nights[night].append(path)
    if not nights:
        raise RuntimeError(f"No calibrated SN2026kid images under {roots}")
    return dict(sorted(nights.items()))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sidecar(path: Path, suffix: str) -> Path:
    return Path(str(path) + suffix)


# ---------------------------------------------------------------------------
# Per-image stages (mask/bkg/bkgrms/phot/calib)


def run_mask(phot, path: Path, overwrite: bool) -> dict:
    import numpy as np
    from astropy.io import fits

    phot.verify_checksum(path)
    metrics = phot.wcs_metrics(path)
    invalid_path = sidecar(path, ".invalidmask")
    source_path = sidecar(path, ".srcmask")
    if (
        not overwrite
        and invalid_path.exists()
        and source_path.exists()
        and sidecar(invalid_path, ".png").exists()
        and sidecar(source_path, ".png").exists()
        and str(fits.getheader(source_path, memmap=False).get("BKGMASK", ""))
        == phot.BACKGROUND_MASK_VERSION
    ):
        return {"status": "existing"}

    telescope_info = phot.telinfo(path)
    image = phot.image_instance(path, telescope_info)
    invalid_mask = image.calculate_invalidmask(
        save=True, verbose=False, visualize=False, save_fig=True
    )
    source_mask = image.calculate_sourcemask(
        target_srcmask=None,
        sigma=5.0,
        mask_radius_factor=3,
        saturation_level=60000,
        save=True,
        verbose=False,
        visualize=False,
        save_fig=True,
    )
    source_mask = phot.add_host_galaxy_mask(
        image=image,
        source_mask=source_mask,
        save=True,
        save_fig=True,
        verbose=False,
    )
    for product in (invalid_mask.path, source_mask.path):
        phot.add_checksum(Path(product))
        phot.verify_checksum(Path(product))
    return {
        "status": "processed",
        "pixel_scale_arcsec": metrics["pixel_scale_arcsec"],
        "invalid_pixels": int(np.count_nonzero(invalid_mask.data)),
        "source_mask_pixels": int(np.count_nonzero(source_mask.data)),
        "source_mask_fraction": float(np.mean(np.asarray(source_mask.data, dtype=bool))),
    }


def run_bkg(phot, path: Path, overwrite: bool) -> dict:
    import numpy as np
    from astropy.io import fits
    from ezphot.imageobjects import Mask

    background_path = sidecar(path, ".bkgmap")
    if (
        not overwrite
        and background_path.exists()
        and sidecar(background_path, ".png").exists()
    ):
        data = fits.getdata(background_path, memmap=False)
        return {"status": "existing", "background_median": float(np.nanmedian(data))}

    invalid_path = sidecar(path, ".invalidmask")
    source_path = sidecar(path, ".srcmask")
    require(
        invalid_path.exists() and source_path.exists(),
        f"Masks missing for {path.name}; run --stage mask first",
    )
    require(
        str(fits.getheader(source_path, memmap=False).get("BKGMASK", ""))
        == phot.BACKGROUND_MASK_VERSION,
        f"Stale source mask for {path.name}; rerun --stage mask",
    )
    telescope_info = phot.telinfo(path)
    image = phot.image_instance(path, telescope_info)
    invalid_mask = Mask(image.savepath.invalidmaskpath, masktype="invalid", load=True)
    source_mask = Mask(image.savepath.srcmaskpath, masktype="source", load=True)
    background = image.calculate_bkg(
        target_srcmask=source_mask,
        target_ivpmask=invalid_mask,
        is_2D_bkg=True,
        box_size=64,
        filter_size=3,
        correct_global_offset=True,
        save=True,
        verbose=False,
        visualize=False,
        save_fig=True,
    )
    phot.add_checksum(Path(background.path))
    phot.verify_checksum(Path(background.path))
    data = np.asarray(background.data, dtype=float)
    return {"status": "processed", "background_median": float(np.nanmedian(data))}


def run_bkgrms(phot, path: Path, overwrite: bool) -> dict:
    import numpy as np
    from astropy.io import fits
    from ezphot.imageobjects import Mask

    rms_path = sidecar(path, ".bkgrms")
    if not overwrite and rms_path.exists() and sidecar(rms_path, ".png").exists():
        data = fits.getdata(rms_path, memmap=False)
        return {"status": "existing", "background_rms_median": float(np.nanmedian(data))}

    invalid_path = sidecar(path, ".invalidmask")
    source_path = sidecar(path, ".srcmask")
    require(
        invalid_path.exists() and source_path.exists(),
        f"Masks missing for {path.name}; run --stage mask first",
    )
    telescope_info = phot.telinfo(path)
    image = phot.image_instance(path, telescope_info)
    invalid_mask = Mask(image.savepath.invalidmaskpath, masktype="invalid", load=True)
    source_mask = Mask(image.savepath.srcmaskpath, masktype="source", load=True)
    bkgrms = image.calculate_bkgrms(
        target_srcmask=source_mask,
        target_ivpmask=invalid_mask,
        box_size=64,
        filter_size=3,
        save=True,
        verbose=False,
        visualize=False,
        save_fig=True,
    )
    phot.add_checksum(Path(bkgrms.path))
    phot.verify_checksum(Path(bkgrms.path))
    data = np.asarray(bkgrms.data, dtype=float)
    return {"status": "processed", "background_rms_median": float(np.nanmedian(data))}


def staged_photometry(phot, image, background, bkgrms, invalid_mask, telescope_info):
    """Aperture photometry on a pre-subtracted temp copy in a no-space path.

    ``photometry_in_safe_temp`` passes a ``Background`` object, which makes
    ``photometry_sex`` write a ``subbkg_`` image into the organized archive
    directory (containing ``1.0-m KASI``) and run SExtractor on that spaced
    path, which SExtractor 2.25 truncates. Subtract the background before
    staging and pass ``target_bkg=0.0`` so no subtracted sibling is created;
    everything else matches ``photometry_in_safe_temp``.
    """
    import importlib
    import json as json_module
    import shutil
    import tempfile

    import numpy as np
    from astropy.io import fits
    from ezphot.imageobjects import Errormap, Mask
    from ezphot.methods import BackgroundGenerator

    output_path = image.savepath.catalogpath
    photometry_figure = Path(str(output_path) + ".png")
    subtracted_figure = Path(str(image.path) + ".photometry_subbkg.png")
    background_data = np.asarray(background.data, dtype=np.float32)
    background_value = float(np.nanmean(background_data))
    with tempfile.TemporaryDirectory(prefix="sn2026kid-phot-", dir="/tmp") as directory:
        directory = Path(directory)
        image_path = directory / "image.fits"
        staged_header = image.header.copy()
        staged_header["BKGVALU"] = (0.0, "Background after subtraction")
        fits.writeto(
            image_path,
            np.asarray(image.data, dtype=np.float32) - background_data,
            staged_header,
            overwrite=True,
            checksum=True,
        )
        temporary_image = phot.image_instance(image_path, telescope_info)
        temporary_image.savedir = directory
        rms_path = temporary_image.savepath.bkgrmspath
        mask_path = temporary_image.savepath.invalidmaskpath
        fits.writeto(
            rms_path,
            np.asarray(bkgrms.data, dtype=np.float32),
            bkgrms.header,
            overwrite=True,
            checksum=True,
        )
        fits.writeto(
            mask_path,
            np.asarray(invalid_mask.data, dtype=np.uint8),
            invalid_mask.header,
            overwrite=True,
            checksum=True,
        )
        temporary_rms = Errormap(rms_path, emaptype="bkgrms", load=True)
        temporary_mask = Mask(mask_path, masktype="invalid", load=True)

        photometry_module = importlib.import_module("ezphot.methods.aperturephotometry")
        original_catalog_class = photometry_module.Catalog
        try:
            photometry_module.Catalog = phot.SafeCatalog
            temporary_catalog = temporary_image.photometry_sex(
                target_bkg=0.0,
                target_bkgrms=temporary_rms,
                target_mask=temporary_mask,
                detection_sigma=1.5,
                aperture_diameter_arcsec=phot.APERTURES_ARCSEC,
                aperture_diameter_seeing=phot.APERTURES_SEEING,
                saturation_level=60000 - background_value,
                kron_factor=2.5,
                save=True,
                verbose=False,
                visualize=True,
                save_fig=True,
            )
        finally:
            photometry_module.Catalog = original_catalog_class
        catalog = temporary_catalog.data.copy()
        seeing_values = np.asarray(catalog["FWHM_WORLD"], dtype=float) * 3600.0
        seeing_values = seeing_values[
            np.isfinite(seeing_values) & (seeing_values > 0.5) & (seeing_values < 10)
        ]
        seeing = float(np.nanmedian(seeing_values)) if len(seeing_values) else 1.2

        temporary_photometry_figure = Path(str(temporary_catalog.path) + ".png")
        if not temporary_photometry_figure.exists():
            raise RuntimeError("photometry_sex did not create its catalog PNG")
        shutil.copy2(temporary_photometry_figure, photometry_figure)
        BackgroundGenerator()._visualize(
            target_img=temporary_image,
            mask_data=None,
            bkg_map=background_data,
            save_path=str(subtracted_figure),
            show=False,
        )
        if not subtracted_figure.exists():
            raise RuntimeError("Background-subtraction review PNG was not created")

    catalog.write(output_path, format="ascii", overwrite=True)
    catalog_info = {
        "path": str(output_path),
        "target_img": str(image.path),
        "obsdate": str(image.obsdate),
        "filter": image.filter,
        "exptime": image.exptime,
        "seeing": seeing,
        "catalog_type": "all",
        "ra": image.ra,
        "dec": image.dec,
        "fov_ra": image.fovx,
        "fov_dec": image.fovy,
        "objname": image.objname,
        "observatory": image.observatory,
        "telname": image.telname,
    }
    Path(str(output_path) + ".info").write_text(
        json_module.dumps(catalog_info, indent=2) + "\n"
    )
    image.header["PHOTSFG"] = (True, "photometry_sex PNGs saved")
    image.header["PHOTPNG"] = (photometry_figure.name, "Photometry source-review PNG")
    image.header["PHOTSUBP"] = (
        subtracted_figure.name,
        "Photometry background-subtraction PNG",
    )
    return catalog


def run_phot(phot, path: Path, overwrite: bool) -> dict:
    from astropy.io import fits
    from ezphot.imageobjects import Background, Errormap, Mask

    header = fits.getheader(path, memmap=False)
    catalog_path = sidecar(path, ".cat")
    if (
        not overwrite
        and catalog_path.exists()
        and bool(header.get("PHOTDONE", False))
        and bool(header.get("PHOTSFG", False))
        and all(figure.exists() for figure in phot.photometry_png_paths(path))
    ):
        from astropy.table import Table

        return {
            "status": "existing",
            "sources": len(Table.read(catalog_path, format="ascii")),
        }

    for suffix, stage in ((".invalidmask", "mask"), (".bkgmap", "bkg"), (".bkgrms", "bkgrms")):
        require(
            sidecar(path, suffix).exists(),
            f"{suffix} missing for {path.name}; run --stage {stage} first",
        )
    telescope_info = phot.telinfo(path)
    image = phot.image_instance(path, telescope_info)
    background = Background(image.savepath.bkgpath, load=True)
    bkgrms = Errormap(image.savepath.bkgrmspath, emaptype="bkgrms", load=True)
    invalid_mask = Mask(image.savepath.invalidmaskpath, masktype="invalid", load=True)
    catalog = staged_photometry(
        phot, image, background, bkgrms, invalid_mask, telescope_info
    )
    image.header["BKGFILE"] = (background.path.name, "Image-derived SEP background map")
    image.header["RMSFILE"] = (bkgrms.path.name, "Image-derived SEP background RMS")
    image.header["CATFILE"] = (image.savepath.catalogpath.name, "Source Extractor catalog")
    image.header["BKG2D"] = (True, "Background estimated directly from science image")
    image.header["BKGMASK"] = (
        phot.BACKGROUND_MASK_VERSION,
        "Host-aware source-mask version",
    )
    image.header["HSTMAJ"] = (
        phot.HOST_MAJOR_ARCMIN * phot.HOST_MASK_SCALE,
        "Masked host major diameter [arcmin]",
    )
    image.header["HSTMIN"] = (
        phot.HOST_MINOR_ARCMIN * phot.HOST_MASK_SCALE,
        "Masked host minor diameter [arcmin]",
    )
    image.header["HSTPA"] = (phot.HOST_POSITION_ANGLE_DEG, "Host PA [deg E of N]")
    image.header["PHOTDONE"] = (True, "Aperture photometry completed")
    image.header["PHOTUTC"] = (phot.now_utc(), "UTC photometry completion time")
    image.update_status("PHOTOMETRY")
    phot.persist_image_header(image)
    for product in (
        path,
        image.savepath.invalidmaskpath,
        image.savepath.srcmaskpath,
        image.savepath.bkgpath,
        image.savepath.bkgrmspath,
    ):
        phot.add_checksum(Path(product))
        phot.verify_checksum(Path(product))
    return {"status": "processed", "sources": int(len(catalog))}


def run_calib(phot, path: Path, reference, overwrite: bool) -> dict:
    from astropy.table import Table

    if not overwrite and phot.products_complete(path):
        from astropy.io import fits

        header = fits.getheader(path, memmap=False)
        return {
            "status": "existing",
            "zp": float(header["ZP_APER_2"]),
            "zp_scatter": float(header["ZPERR_APER_2"]),
            "zp_stars": int(header["NZPSTAR"]),
        }

    catalog_path = sidecar(path, ".cat")
    require(
        catalog_path.exists(),
        f".cat missing for {path.name}; run --stage phot first",
    )
    telescope_info = phot.telinfo(path)
    image = phot.image_instance(path, telescope_info)
    catalog = Table.read(catalog_path, format="ascii")
    zero_point = phot.calibrate_with_ezphot(
        image, catalog, reference, telescope_info, verbose=False
    )
    phot.verify_checksum(path)
    require(
        phot.products_complete(path),
        f"Products incomplete after calibration for {path.name}",
    )
    return {
        "status": "processed",
        "zp": zero_point["primary_zp"],
        "zp_scatter": zero_point["primary_zp_scatter"],
        "zp_stars": zero_point["matched_stars"],
        "reference_band": zero_point["reference_band"],
    }


# ---------------------------------------------------------------------------
# Per-night stages (stack/stackphot)


def run_stack_night(phot, night: str, paths: list[Path], overwrite: bool) -> list[dict]:
    from astropy.io import fits

    roots = {path.parents[1] for path in paths}
    require(
        len(roots) == 1,
        f"Night {night} spans multiple detector trees: {sorted(map(str, roots))}",
    )
    telescope_info = phot.telinfo(paths[0])
    grouped: dict[str, list[Path]] = defaultdict(list)
    results = []
    for path in paths:
        header = fits.getheader(path, memmap=False)
        require(
            bool(header.get("ZPCALC", False)),
            f"{path.name} has no zero point; run --stage calib first",
        )
        if int(header.get("NZPSTAR", 0)) < STACK_MIN_ZP_STARS:
            results.append(
                {
                    "filter": str(header["FILTER"]),
                    "status": "stack_excluded",
                    "path": str(path),
                    "reason": f"NZPSTAR {int(header.get('NZPSTAR', 0))} < "
                    f"{STACK_MIN_ZP_STARS}",
                }
            )
            continue
        grouped[str(header["FILTER"])].append(path)
    for filter_name, filter_paths in sorted(grouped.items()):
        if len(filter_paths) < 2:
            results.append(
                {
                    "filter": filter_name,
                    "status": "skipped",
                    "reason": f"only {len(filter_paths)} image(s)",
                }
            )
            continue
        results.append(
            phot.stack_filter(
                filter_paths,
                filter_name,
                night,
                telescope_info,
                overwrite=overwrite,
                verbose=False,
            )
        )
    return results


def run_stackphot_night(
    phot, night: str, paths: list[Path], reference, overwrite: bool
) -> list[dict]:
    telescope_info = phot.telinfo(paths[0])
    stack_paths = sorted(
        {path.parent for path in paths},
        key=lambda directory: directory.name,
    )
    results = []
    for directory in stack_paths:
        for stack_path in sorted(directory.glob(f"stack.SN2026kid.{night}.*.fits")):
            if any(
                stack_path.name.endswith(suffix)
                for suffix in (".bkgrms", ".cat", ".refcat")
            ):
                continue
            results.append(
                phot.process_stack_catalog(
                    {"output": str(stack_path), "filter": stack_path.stem.split(".")[-1]},
                    reference,
                    telescope_info,
                    overwrite=overwrite,
                    verbose=False,
                )
            )
    require(bool(results), f"No stacks found for night {night}; run --stage stack first")
    return results


# ---------------------------------------------------------------------------
# Night runner (child mode) and orchestrator (parent mode)


def night_image_list(args) -> list[Path]:
    if args.image_list:
        rejected = rejected_paths(args.rejects)
        paths = [
            Path(line)
            for line in args.image_list.read_text().splitlines()
            if line.strip() and line.strip() not in rejected
        ]
        require(bool(paths), f"Empty image list {args.image_list}")
        return paths
    return discover_by_night(args.roots, args.rejects)[args.night]


def run_night(args) -> dict:
    phot = load_photometry_module()
    # process_stack_catalog photometers through the module-level
    # photometry_in_safe_temp, which hits the spaced-archive-path SExtractor
    # failure described in staged_photometry. Redirect it on this private
    # module instance so stacks use the same staged path as --stage phot.
    phot.photometry_in_safe_temp = (
        lambda image, background, bkgrms, invalid_mask, telescope_info, verbose: (
            staged_photometry(
                phot, image, background, bkgrms, invalid_mask, telescope_info
            )
        )
    )
    paths = night_image_list(args)
    if args.limit:
        paths = paths[: args.limit]
    reference = None
    if args.stage in ("calib", "stackphot"):
        from astropy.coordinates import SkyCoord

        telescope_info = phot.telinfo(paths[0])
        first = phot.image_instance(paths[0], telescope_info)
        center = SkyCoord(first.ra, first.dec, unit="deg")
        reference = phot.query_reference_catalog(center, args.reference_catalog, False)

    results, failures = [], []
    if args.stage in ("stack", "stackphot"):
        try:
            if args.stage == "stack":
                results = run_stack_night(phot, args.night, paths, args.overwrite)
            else:
                results = run_stackphot_night(
                    phot, args.night, paths, reference, args.overwrite
                )
        except Exception as error:
            failures.append({"night": args.night, "error": f"{type(error).__name__}: {error}"})
    else:
        per_image = {
            "mask": lambda path: run_mask(phot, path, args.overwrite),
            "bkg": lambda path: run_bkg(phot, path, args.overwrite),
            "bkgrms": lambda path: run_bkgrms(phot, path, args.overwrite),
            "phot": lambda path: run_phot(phot, path, args.overwrite),
            "calib": lambda path: run_calib(phot, path, reference, args.overwrite),
        }[args.stage]
        for index, path in enumerate(paths, start=1):
            print(f"[{index:02d}/{len(paths):02d}] {path.name}", flush=True)
            try:
                result = per_image(path)
                result["path"] = str(path)
                results.append(result)
                print(f"  {result['status']}", flush=True)
            except Exception as error:
                failure = {"path": str(path), "error": f"{type(error).__name__}: {error}"}
                failures.append(failure)
                print(f"  FAIL {failure['error']}", flush=True)

    report = {
        "stage": args.stage,
        "night": args.night,
        "created_utc": now_utc(),
        "image_count": len(paths),
        "results": results,
        "failures": failures,
    }
    write_json(args.report_dir / f"stage_{args.stage}_{args.night}.json", report)
    if failures:
        raise SystemExit(f"{len(failures)} failure(s) in stage {args.stage} {args.night}")
    return report


def orchestrate(args) -> None:
    nights = discover_by_night(args.roots, args.rejects)
    if args.limit:
        nights = dict(list(nights.items())[: args.limit])
    run_dir = args.run_dir
    if run_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = args.log_root / f"{stamp}_stage_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Stage {args.stage}: {len(nights)} nights, logs in {run_dir}", flush=True)

    environment = dict(os.environ)
    environment.setdefault("MPLCONFIGDIR", "/tmp/ezphot-matplotlib")
    aggregate = {
        "stage": args.stage,
        "started_utc": now_utc(),
        "run_dir": str(run_dir),
        "nights": {},
        "failures": [],
    }
    aggregate_path = args.report_dir / f"stage_{args.stage}_report.json"
    for index, (night, paths) in enumerate(nights.items(), start=1):
        manifest = run_dir / f"manifest_{night}.txt"
        manifest.write_text("\n".join(str(path) for path in paths) + "\n")
        log_path = run_dir / f"{args.stage}_{night}.log"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--stage",
            args.stage,
            "--night",
            night,
            "--image-list",
            str(manifest),
            "--reference-catalog",
            str(args.reference_catalog),
            "--report-dir",
            str(args.report_dir),
            "--rejects",
            str(args.rejects),
        ]
        if args.overwrite:
            command.append("--overwrite")
        with open(log_path, "w") as log_file:
            status = subprocess.run(
                command, stdout=log_file, stderr=subprocess.STDOUT, env=environment
            ).returncode
        night_report_path = args.report_dir / f"stage_{args.stage}_{night}.json"
        night_summary = {"images": len(paths), "returncode": status}
        if night_report_path.exists():
            night_report = json.loads(night_report_path.read_text())
            counts = defaultdict(int)
            for result in night_report["results"]:
                counts[result.get("status", "unknown")] += 1
            night_summary.update(counts)
            night_summary["failed"] = len(night_report["failures"])
            aggregate["failures"].extend(
                {**failure, "night": night} for failure in night_report["failures"]
            )
        else:
            night_summary["failed"] = len(paths)
            aggregate["failures"].append(
                {"night": night, "error": f"no report; see {log_path}"}
            )
        aggregate["nights"][night] = night_summary
        aggregate["finished_utc"] = now_utc()
        write_json(aggregate_path, aggregate)
        print(
            f"[{index:02d}/{len(nights):02d}] {night}: {night_summary}",
            flush=True,
        )
    total_failed = sum(summary["failed"] for summary in aggregate["nights"].values())
    print(f"Aggregate report: {aggregate_path}", flush=True)
    print(f"STAGE_RESULT={'PASS' if total_failed == 0 else 'FAIL'}", flush=True)
    if total_failed:
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--night", help="Run a single night in-process")
    parser.add_argument("--image-list", type=Path, help="Manifest of image paths")
    parser.add_argument("--roots", type=Path, nargs="+", default=ROOTS_DEFAULT)
    parser.add_argument("--reference-catalog", type=Path, default=REFERENCE_DEFAULT)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR_DEFAULT)
    parser.add_argument("--rejects", type=Path, default=REJECTS_DEFAULT)
    parser.add_argument("--log-root", type=Path, default=LOG_ROOT_DEFAULT)
    parser.add_argument("--run-dir", type=Path, help="Reuse an existing log directory")
    parser.add_argument(
        "--limit",
        type=int,
        help="Cap images per night (child) or number of nights (orchestrator)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.night:
        run_night(args)
    else:
        orchestrate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
