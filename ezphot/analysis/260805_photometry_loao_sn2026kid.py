#!/usr/bin/env python
"""Run the single-night LOAO SN2026kid photometry workflow.

The default input is the calibrated 2026_0714 data tree produced by
``260805_process_loao_sn2026kid.py``.  Processing order is:

1. astrometry.net WCS solution;
2. invalid/source masks;
3. SEP background and background-RMS maps estimated directly from each image;
4. Source Extractor aperture photometry;
5. catalog zero points;
6. background-subtracted, zero-point-scaled, WCS-reprojected stacks by filter.

The astrometry step never asks ezphot to overwrite its input directly.  It
solves to a temporary sibling, validates the WCS, and only then atomically
replaces the image.  This avoids an in-place failure path in the current helper
that can remove the input image when ``solve-field`` fails.

Reference stars are loaded with ezphot's ``SkyCatalog`` class from APASS DR9.
The LOAO wheel labels map as B102->B, V103->V, R104->R, and I105->I.  APASS B
and V are used directly; Cousins R and I are derived from APASS Sloan r/i with
the transformations implemented in ``ezphot.skycatalog.conversion``.  Derived
passbands are recorded in the report and FITS headers.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.table import Table
from astropy.time import Time
from astropy.wcs import WCS
from astropy.wcs.utils import fit_wcs_from_points, proj_plane_pixel_scales

from ezphot.dataobjects.catalog import Catalog, Info as CatalogInfo
from ezphot.helper import Helper
from ezphot.imageobjects import Background, Errormap, Mask, ScienceImage
from ezphot.methods import PhotometricCalibration, Stack
from ezphot.skycatalog import SkyCatalog


NIGHT_DEFAULT = "2026_0714"
TARGET_DEFAULT = "SN2026kid"
INPUT_ROOT_DEFAULT = Path(
    "/home/hhchoi1022/ezphot/data/scidata/LOAO/LOAO_E2V_2x2/"
    "SN2026kid/1.0-m KASI"
)
FILTERS = ("B102", "V103", "R104", "I105")
REFERENCE_BANDS = {
    "B102": ("B_mag", "B", False),
    "V103": ("V_mag", "V", False),
    "R104": ("R_mag", "R", True),
    "I105": ("I_mag", "I", True),
}
REFERENCE_LABEL = "SkyCatalog APASS DR9"
CALIBRATION_METHOD = "PhotometricCalibration"
PIXEL_SCALE = 0.794
IKON_PIXEL_SCALE = 0.356
PRIMARY_MAG_KEY = "MAG_APER_2"  # 7 arcsec diameter with the aperture list below
APERTURES_ARCSEC = [3, 5, 7, 10]
APERTURES_SEEING = [3.5, 4.5]
BACKGROUND_MASK_VERSION = "NGC5907_SIMBAD_1.25x_v1"
HOST_CENTER = SkyCoord("15h15m53.687s", "+56d19m43.86s", frame="icrs")
HOST_MAJOR_ARCMIN = 15.4703
HOST_MINOR_ARCMIN = 1.68317
HOST_POSITION_ANGLE_DEG = 155.0
HOST_MASK_SCALE = 1.25
STACK_CENTER_VERSION = "NGC5907_SIMBAD_v1"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def add_host_galaxy_mask(
    image: ScienceImage,
    source_mask: Mask,
    save: bool = True,
    save_fig: bool = True,
    verbose: bool = True,
) -> Mask:
    """Add a padded, WCS-aligned NGC 5907 optical ellipse to a source mask."""
    from photutils.aperture import EllipticalAperture
    from ezphot.methods import MaskGenerator

    center_x, center_y = image.wcs.world_to_pixel(HOST_CENTER)
    major_endpoint = HOST_CENTER.directional_offset_by(
        HOST_POSITION_ANGLE_DEG * u.deg,
        60.0 * u.arcsec,
    )
    endpoint_x, endpoint_y = image.wcs.world_to_pixel(major_endpoint)
    theta = float(np.arctan2(endpoint_y - center_y, endpoint_x - center_x))
    pixel_scale = float(np.mean(proj_plane_pixel_scales(image.wcs)) * 3600.0)
    semimajor_pixel = (
        0.5 * HOST_MAJOR_ARCMIN * 60.0 * HOST_MASK_SCALE / pixel_scale
    )
    semiminor_pixel = (
        0.5 * HOST_MINOR_ARCMIN * 60.0 * HOST_MASK_SCALE / pixel_scale
    )

    aperture = EllipticalAperture(
        (center_x, center_y),
        a=semimajor_pixel,
        b=semiminor_pixel,
        theta=theta,
    )
    aperture_mask = aperture.to_mask(method="center")
    host_mask = aperture_mask.to_image(image.data.shape)
    if host_mask is None:
        raise RuntimeError("NGC 5907 mask does not overlap the image")
    host_mask = np.asarray(host_mask > 0, dtype=bool)
    previous_mask = np.asarray(source_mask.data, dtype=bool).copy()
    source_mask.combine_mask(host_mask, "or")
    source_mask.header.update(
        {
            "BKGMASK": (BACKGROUND_MASK_VERSION, "Background source-mask version"),
            "HSTMAJ": (
                HOST_MAJOR_ARCMIN * HOST_MASK_SCALE,
                "Masked host major diameter [arcmin]",
            ),
            "HSTMIN": (
                HOST_MINOR_ARCMIN * HOST_MASK_SCALE,
                "Masked host minor diameter [arcmin]",
            ),
            "HSTPA": (HOST_POSITION_ANGLE_DEG, "Host position angle [deg E of N]"),
            "HSTRA": (HOST_CENTER.ra.deg, "Host-mask center RA [deg]"),
            "HSTDEC": (HOST_CENTER.dec.deg, "Host-mask center Dec [deg]"),
        }
    )
    source_mask.add_status(
        "host_galaxy_mask",
        name="NGC5907",
        major_arcmin=str(HOST_MAJOR_ARCMIN * HOST_MASK_SCALE),
        minor_arcmin=str(HOST_MINOR_ARCMIN * HOST_MASK_SCALE),
        position_angle_deg=str(HOST_POSITION_ANGLE_DEG),
    )
    if save:
        source_mask.write(verbose=verbose)
    if save_fig:
        MaskGenerator()._visualize(
            target_img=image,
            final_mask=source_mask,
            previous_mask=previous_mask,
            save_path=str(source_mask.path) + ".png",
            show=False,
        )
    return source_mask


def discover_images(
    root: Path, night: str | None = None, limit: int | None = None
) -> list[Path]:
    paths = []
    for filter_name in FILTERS:
        paths.extend(sorted((root / filter_name).glob("obj.SN2026kid.*.fits")))
    paths = [path for path in paths if ".com." not in path.name]
    if night is not None:
        paths = [
            path
            for path in paths
            if str(fits.getheader(path, memmap=False).get("RAWNIGHT", "")) == night
        ]
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise RuntimeError(f"No calibrated SN2026kid images found under {root}")
    return paths


def stage_images(paths: list[Path], test_root: Path | None) -> list[Path]:
    if test_root is None:
        return paths
    staged = []
    for source in paths:
        destination = test_root / source.parent.name / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        for suffix in (".bpmask", ".status", ".info"):
            sidecar = Path(str(source) + suffix)
            if sidecar.exists():
                shutil.copy2(sidecar, Path(str(destination) + suffix))
        staged.append(destination)
    return staged


def telinfo(path: Path):
    header = fits.getheader(path, memmap=False)
    telkey = str(header.get("TELKEY", ""))
    instrument = str(header.get("INSTRUME", ""))
    if (
        "ikon" in telkey.lower()
        or str(header.get("CCD", "")).lower() == "ikon"
        or "ikon" in instrument.lower()
        or "loao_ikon_1x1" in str(path).lower()
    ):
        info = Helper().get_telinfo(telescope="LOAO", ccd="iKon", binning=1)
        # The registry currently duplicates the E2V 2x2 scale (0.794 arcsec)
        # for iKon.  APASS pattern matching on these 2048-pixel frames gives
        # 0.356 arcsec/pixel, which is also the scale required by solve-field.
        info["pixelscale"] = IKON_PIXEL_SCALE
        return info
    return Helper().get_telinfo(telescope="LOAO", ccd="E2V", binning=2)


def image_instance(path: Path, telescope_info) -> ScienceImage:
    header = fits.getheader(path, memmap=False)
    if "IMGTYPE" not in header:
        # Several legacy iKon science headers omit this otherwise standard
        # keyword.  ScienceImage repeatedly warns while resolving the image
        # type, so normalize the derived product before constructing it.
        with fits.open(path, mode="update", memmap=False) as hdul:
            hdul[0].header["IMGTYPE"] = ("LIGHT", "Normalized science image type")
            hdul[0].add_checksum()
            hdul.flush()
    image = ScienceImage(path, telinfo=telescope_info, load=True)
    image.savedir = path.parent
    return image


def add_checksum(path: Path) -> None:
    with fits.open(path, mode="update", memmap=False) as hdul:
        hdul[0].add_checksum()
        hdul.flush()


def verify_checksum(path: Path) -> None:
    with fits.open(path, checksum=True, memmap=False) as hdul:
        hdul.verify("exception")
        if hdul[0].verify_checksum() != 1 or hdul[0].verify_datasum() != 1:
            raise ValueError(f"Invalid FITS checksum: {path}")


def wcs_metrics(path: Path, pointing: SkyCoord | None = None) -> dict[str, float]:
    header = fits.getheader(path, memmap=False)
    wcs = WCS(header)
    if not wcs.has_celestial:
        raise ValueError(f"No celestial WCS in {path}")
    shape = (int(header["NAXIS2"]), int(header["NAXIS1"]))
    center = wcs.pixel_to_world((shape[1] - 1) / 2, (shape[0] - 1) / 2)
    scale = float(np.mean(proj_plane_pixel_scales(wcs.celestial)) * 3600)
    if not 0.30 <= scale <= 0.95:
        raise ValueError(f"Implausible LOAO pixel scale {scale:.5f} arcsec/pixel")
    separation = float(center.separation(pointing).arcsec) if pointing else np.nan
    if pointing and separation > 7200:
        raise ValueError(f"Solved center is {separation:.1f} arcsec from pointing")
    return {
        "center_ra": float(center.ra.deg),
        "center_dec": float(center.dec.deg),
        "pixel_scale_arcsec": scale,
        "pointing_separation_arcsec": separation,
    }


def persist_image_header(image: ScienceImage) -> None:
    header = image.header.copy()
    image.clear(clear_data=True, clear_header=False, verbose=False)
    with fits.open(image.path, mode="update", memmap=False) as hdul:
        hdul[0].header.update(header)
        hdul[0].add_checksum()
        hdul.flush()
    image.header = fits.getheader(image.path, memmap=False)
    image.save_status()
    image.save_info()


def persist_image_data_and_header(image: ScienceImage) -> None:
    """Atomically persist a derived image after editing its data and header."""
    with tempfile.NamedTemporaryFile(
        prefix=f".{image.path.name}.", suffix=".tmp", dir=image.path.parent, delete=False
    ) as handle:
        temporary_path = Path(handle.name)
    try:
        fits.writeto(
            temporary_path,
            image.data,
            image.header,
            overwrite=True,
            checksum=True,
        )
        os.replace(temporary_path, image.path)
    finally:
        temporary_path.unlink(missing_ok=True)
    image.header = fits.getheader(image.path, memmap=False)
    image.save_status()
    image.save_info()


def remove_stale_sip(header: fits.Header) -> None:
    """Remove input-image SIP terms after SWarp has made a rectified TAN grid."""
    sip_order_keys = {"A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER"}
    sip_coefficient = re.compile(r"^(?:A|B|AP|BP)_\d+_\d+$")
    for key in list(header):
        if key in sip_order_keys or sip_coefficient.fullmatch(key):
            del header[key]


def relaxed_astrometry_config(source: Path, destination: Path) -> Path:
    """Write a more sensitive copy of ezphot's astrometry SExtractor config.

    The common ezphot configuration requires 15 pixels at 10 sigma.  That is
    appropriate for crowded/deep frames, but the older LOAO iKon exposures
    yield fewer than ten detections at that setting.  Astrometry.net needs a
    substantially longer source list, so use a 1.5-sigma, 5-pixel
    extraction for plate solving only.  Science photometry retains its own
    independent settings below.
    """
    text = source.read_text()
    replacements = {
        "DETECT_MINAREA": "5",
        "DETECT_THRESH": "1.5",
        "ANALYSIS_THRESH": "1.5",
    }
    for key, value in replacements.items():
        text, count = re.subn(
            rf"(?m)^(\s*{key}\s+)\S+",
            rf"\g<1>{value}",
            text,
            count=1,
        )
        if count != 1:
            raise RuntimeError(f"Could not update {key} in {source}")
    destination.write_text(text)
    return destination


def suppress_astrometry_edge_artifacts(path: Path, border: int = 32) -> None:
    """Replace the detector rim in the temporary solve image with sky.

    The iKon frames contain a strong first-row structure which SExtractor
    deblends into dozens of bright, elongated detections.  Those detections
    otherwise dominate astrometry.net's magnitude-sorted input list.  The
    science image itself is never changed: only the temporary plate-solving
    copy is masked, and the solved WCS is later attached to the original data.
    """
    data, header = fits.getdata(path, header=True, memmap=False)
    data = np.asarray(data, dtype=np.float32).copy()
    sky = float(np.nanmedian(data[border:-border, border:-border]))
    data[:border, :] = sky
    data[-border:, :] = sky
    data[:, :border] = sky
    data[:, -border:] = sky
    fits.writeto(path, data, header, overwrite=True, checksum=True)


def solve_astrometry_from_apass(
    image_path: Path,
    output_path: Path,
    sex_config: Path,
    reference: Table,
    pointing: SkyCoord,
    filter_name: str,
    pixel_scale: float,
    verbose: bool,
) -> dict:
    """Fit a TAN WCS by pattern-matching image detections to APASS stars."""
    import astroalign

    helper = Helper()
    result, sources, _, _ = helper.run_sextractor(
        target_path=image_path,
        sex_configfile=sex_config,
        sex_params={
            "DETECT_THRESH": 1.5,
            "ANALYSIS_THRESH": 1.5,
            "DETECT_MINAREA": 5,
        },
        target_outpath=image_path.with_suffix(".astrometry.cat"),
        return_result=True,
        verbose=verbose,
    )
    if not result:
        raise RuntimeError("APASS-assisted astrometry source extraction failed")
    quality = (
        (np.asarray(sources["X_IMAGE"], dtype=float) > 32)
        & (np.asarray(sources["X_IMAGE"], dtype=float) < 2016)
        & (np.asarray(sources["Y_IMAGE"], dtype=float) > 32)
        & (np.asarray(sources["Y_IMAGE"], dtype=float) < 2016)
        & (np.asarray(sources["FLAGS"], dtype=float) == 0)
        & (np.asarray(sources["FWHM_IMAGE"], dtype=float) > 2)
        & (np.asarray(sources["FWHM_IMAGE"], dtype=float) < 20)
        & (np.asarray(sources["ELONGATION"], dtype=float) < 2.5)
    )
    sources = sources[quality]
    sources.sort("MAG_AUTO")
    source_points = np.column_stack(
        [sources["X_IMAGE"], sources["Y_IMAGE"]]
    ).astype(float)[:40]
    if len(source_points) < 10:
        raise RuntimeError(
            f"Only {len(source_points)} stellar detections for APASS astrometry"
        )

    magnitude_key = f"{filter_name}_mag"
    reference_quality = (
        np.isfinite(np.asarray(reference["ra"], dtype=float))
        & np.isfinite(np.asarray(reference["dec"], dtype=float))
        & np.isfinite(np.asarray(reference[magnitude_key], dtype=float))
    )
    reference_stars = reference[reference_quality]
    reference_stars.sort(magnitude_key)
    offsets = SkyCoord(
        reference_stars["ra"], reference_stars["dec"], unit="deg"
    ).transform_to(pointing.skyoffset_frame())
    target_points_all = np.column_stack(
        [
            offsets.lon.to_value(u.arcsec) / pixel_scale + 1024,
            offsets.lat.to_value(u.arcsec) / pixel_scale + 1024,
        ]
    )

    last_error = None
    matched_source = matched_target = target_points = target_rows = None
    for count in (100, len(target_points_all)):
        target_points = target_points_all[:count]
        target_rows = reference_stars[:count]
        try:
            _, (matched_source, matched_target) = astroalign.find_transform(
                source_points,
                target_points,
                max_control_points=max(len(source_points), len(target_points)),
            )
            if len(matched_source) >= 8:
                break
        except Exception as error:
            last_error = error
            matched_source = None
    if matched_source is None or len(matched_source) < 8:
        raise RuntimeError(f"APASS pattern match failed: {last_error}")

    reference_indices = np.array(
        [
            int(np.argmin(np.sum((target_points - position) ** 2, axis=1)))
            for position in matched_target
        ]
    )
    sky_matches = SkyCoord(
        target_rows["ra"][reference_indices],
        target_rows["dec"][reference_indices],
        unit="deg",
    )
    fitted_wcs = fit_wcs_from_points(
        (matched_source[:, 0] - 1, matched_source[:, 1] - 1),
        sky_matches,
        projection="TAN",
    )
    data, header = fits.getdata(image_path, header=True, memmap=False)
    header.update(fitted_wcs.to_header(relax=True))
    fits.writeto(output_path, data, header, overwrite=True, checksum=True)
    return {
        "method": "APASS pattern match + fit_wcs_from_points",
        "matches": int(len(matched_source)),
    }


def solve_astrometry_safe(
    path: Path,
    telescope_info,
    reference: Table,
    overwrite: bool,
    verbose: bool,
) -> tuple[ScienceImage, dict]:
    image = image_instance(path, telescope_info)
    pointing = SkyCoord(image.ra, image.dec, unit="deg")
    try:
        existing = wcs_metrics(path, pointing)
    except Exception:
        existing = None
    if existing is not None and not overwrite:
        image.update_status("ASTROMETRY")
        image.header["EGAIN"] = (
            float(image.header.get("EGAIN", image.header.get("GAIN", 2.68))),
            "Effective gain [e-/ADU]",
        )
        image.header["ASTRMCOR"] = (True, "Astrometric WCS is present and validated")
        image.header["ASTRMSCL"] = (existing["pixel_scale_arcsec"], "WCS scale [arcsec/pix]")
        persist_image_header(image)
        return image_instance(path, telescope_info), {"status": "existing", **existing}

    temporary = path.with_name(f".{path.name}.astrometry.{os.getpid()}.tmp.fits")
    if temporary.exists():
        temporary.unlink()
    helper = Helper()
    astrometry_method = "astrometry.net"
    fallback_info = {}
    try:
        with tempfile.TemporaryDirectory(
            prefix="sn2026kid-astrometry-", dir="/tmp"
        ) as safe_directory:
            safe_directory = Path(safe_directory)
            safe_input = safe_directory / "input.fits"
            safe_output = safe_directory / "solved.fits"
            shutil.copy2(path, safe_input)
            suppress_astrometry_edge_artifacts(safe_input)
            astrometry_config = relaxed_astrometry_config(
                Path(image.config["ASTROMETRY_SEXCONFIG"]),
                safe_directory / "astrometry.sexconfig",
            )
            try:
                result, output = helper.run_astrometry(
                    target_path=safe_input,
                    astrometry_sexconfigfile=astrometry_config,
                    ra=float(pointing.ra.deg),
                    dec=float(pointing.dec.deg),
                    radius=2,
                    pixelscale=float(telescope_info["pixelscale"]),
                    target_outpath=safe_output,
                    verbose=verbose,
                )
            except Exception:
                fallback_info = solve_astrometry_from_apass(
                    image_path=safe_input,
                    output_path=safe_output,
                    sex_config=astrometry_config,
                    reference=reference,
                    pointing=pointing,
                    filter_name=image.filter,
                    pixel_scale=float(telescope_info["pixelscale"]),
                    verbose=verbose,
                )
                result, output = True, safe_output
                astrometry_method = fallback_info["method"]
            if not result or Path(output) != safe_output:
                raise RuntimeError("Astrometry helper did not return the requested output")
            metrics = wcs_metrics(safe_output, pointing)
            solved_header = fits.getheader(safe_output, memmap=False)
            fits.writeto(
                temporary,
                np.asarray(image.data),
                solved_header,
                overwrite=True,
                checksum=True,
            )
        verify_checksum(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()

    solved = image_instance(path, telescope_info)
    solved.header["EGAIN"] = (float(solved.header.get("GAIN", 2.68)), "Effective gain [e-/ADU]")
    solved.header["ASTRMCOR"] = (True, f"Solved with {astrometry_method}")
    solved.header["ASTRMSCL"] = (metrics["pixel_scale_arcsec"], "WCS scale [arcsec/pix]")
    solved.header["ASTRMSEP"] = (
        metrics["pointing_separation_arcsec"],
        "Solved center separation from pointing [arcsec]",
    )
    solved.header["ASTRMUTC"] = (now_utc(), "UTC astrometry completion time")
    solved.update_status("ASTROMETRY")
    persist_image_header(solved)
    verify_checksum(path)
    return image_instance(path, telescope_info), {
        "status": "solved",
        "method": astrometry_method,
        **fallback_info,
        **metrics,
    }


def query_reference_catalog(center: SkyCoord, path: Path, overwrite: bool) -> Table:
    if path.exists() and not overwrite:
        reference = Table.read(path, format="ascii.ecsv")
        for wheel_filter, (_, standard_band, _) in REFERENCE_BANDS.items():
            reference[f"{standard_band}_mag"] = reference[f"{wheel_filter}_mag"]
            reference[f"e_{standard_band}_mag"] = reference[f"{wheel_filter}_err"]
        return reference

    sky_catalog = SkyCatalog(
        objname="NGC5907",
        catalog_type="APASS",
        catalog_version="v1",
        fov_ra=0.65,
        fov_dec=0.65,
        verbose=True,
    )
    source = sky_catalog.data
    if source is None or len(source) == 0:
        raise RuntimeError("SkyCatalog returned no APASS reference sources")

    def values(key: str) -> np.ndarray:
        return np.asarray(np.ma.filled(np.ma.asarray(source[key]), np.nan), dtype=float)

    # These are the same APASS -> Johnson-Cousins equations used by
    # ezphot.skycatalog.conversion.Conversion.APASS_to_JH.
    r_mag = values("r_mag")
    i_mag = values("i_mag")
    e_r_mag = values("e_r_mag")
    e_i_mag = values("e_i_mag")
    r_minus_i = r_mag - i_mag
    e_r_minus_i = np.hypot(e_r_mag, e_i_mag)
    transformed_r = r_mag - 0.153 * r_minus_i - 0.117
    transformed_i = transformed_r - 0.930 * r_minus_i - 0.259
    e_transformed_r = np.sqrt(e_r_mag**2 + 0.003**2 + (0.153 * e_r_minus_i) ** 2)
    e_transformed_i = np.sqrt(e_r_mag**2 + 0.002**2 + (0.930 * e_r_minus_i) ** 2)

    reference = Table()
    reference["ra"] = values("ra")
    reference["dec"] = values("dec")
    reference["B102_mag"] = values("B_mag")
    reference["B102_err"] = values("e_B_mag")
    reference["V103_mag"] = values("V_mag")
    reference["V103_err"] = values("e_V_mag")
    reference["R104_mag"] = transformed_r
    reference["R104_err"] = e_transformed_r
    reference["I105_mag"] = transformed_i
    reference["I105_err"] = e_transformed_i
    reference["g_mag"] = values("g_mag")
    reference["r_mag"] = r_mag
    reference["i_mag"] = i_mag
    for wheel_filter, (_, standard_band, _) in REFERENCE_BANDS.items():
        reference[f"{standard_band}_mag"] = reference[f"{wheel_filter}_mag"]
        reference[f"e_{standard_band}_mag"] = reference[f"{wheel_filter}_err"]
    path.parent.mkdir(parents=True, exist_ok=True)
    reference.meta["source"] = REFERENCE_LABEL
    reference.meta["skycatalog_type"] = sky_catalog.catalog_type
    reference.meta["skycatalog_file"] = str(sky_catalog.filepath)
    reference.meta["query_center"] = center.to_string("decimal")
    reference.meta["query_width_deg"] = 0.65
    reference.meta["R_transform"] = "R = r - 0.153*(r-i) - 0.117"
    reference.meta["I_transform"] = "I = R - 0.930*(r-i) - 0.259"
    reference.write(path, format="ascii.ecsv", overwrite=True)
    return reference


def finite_column(table: Table, name: str, default: float = np.nan) -> np.ndarray:
    if name not in table.colnames:
        return np.full(len(table), default, dtype=float)
    try:
        return np.asarray(table[name], dtype=float)
    except Exception:
        return np.full(len(table), default, dtype=float)


class SafeCatalog(Catalog):
    """Catalog variant that never auto-loads a ScienceImage without telinfo."""

    def __init__(
        self,
        path: Path | str,
        catalog_type: str = "all",
        info: CatalogInfo | None = None,
        load: bool = False,
        data: Table | None = None,
    ):
        self.helper = Helper()
        self.is_loaded = False
        self.path = Path(path)
        self.catalog_type = catalog_type
        self.target_img = None
        self._data = data
        self._target_data = None if data is None else data.copy()
        self.info = info.copy() if info is not None else CatalogInfo()
        self.info.path = str(self.path)
        self.info.catalog_type = catalog_type
        self._savedir = self.path.parent
        if load and data is None and self.path.exists():
            self._data = Table.read(self.path, format="ascii")
            self._target_data = self._data.copy()

    def copy(self) -> "SafeCatalog":
        return SafeCatalog(
            self.path,
            catalog_type=self.catalog_type,
            info=self.info,
            data=None if self.data is None else self.data.copy(),
        )


class PreparedReferenceUtility:
    """Feed the SkyCatalog/APASS BVRI table to PhotometricCalibration."""

    def __init__(self, reference: Table):
        self.reference = reference

    def get_catalogs(self, **kwargs):
        reference_catalog = type("PreparedSkyCatalog", (), {})()
        reference_catalog.data = self.reference
        reference_catalog.catalog_type = "APASS"
        reference_catalog.objname = "NGC5907"
        return [reference_catalog]

    def select_reference_sources(
        self,
        catalog,
        mag_lower: float | None = None,
        mag_upper: float | None = None,
        **kwargs,
    ):
        data = catalog.data
        finite_position = np.isfinite(finite_column(data, "ra")) & np.isfinite(
            finite_column(data, "dec")
        )
        return data[finite_position], []


def catalog_info_for(image: ScienceImage, path: Path) -> CatalogInfo:
    return CatalogInfo(
        path=str(path),
        target_img=str(image.path),
        obsdate=str(image.obsdate),
        filter=image.filter,
        exptime=image.exptime,
        catalog_type="all",
        ra=image.ra,
        dec=image.dec,
        fov_ra=image.fovx,
        fov_dec=image.fovy,
        objname=image.objname,
        observatory=image.observatory,
        telname=image.telname,
    )


def photometry_png_paths(image_path: Path) -> list[Path]:
    catalog_path = Path(str(image_path) + ".cat")
    return [
        Path(str(catalog_path) + ".png"),
        Path(str(image_path) + ".photometry_subbkg.png"),
    ]


def calibration_png_paths(image_path: Path) -> list[Path]:
    catalog_path = Path(str(image_path) + ".cat")
    return [
        catalog_path.with_suffix(".zp_2d.png"),
        catalog_path.with_suffix(".zp_mag.png"),
        catalog_path.with_suffix(".zp_color.png"),
        Path(str(image_path) + ".refcat.png"),
    ]


def calibrate_with_ezphot(
    image: ScienceImage,
    catalog_table: Table,
    reference: Table,
    telescope_info,
    verbose: bool,
) -> dict:
    """Run ezphot PhotometricCalibration with figures and safe filter aliases."""
    import importlib

    filter_name = image.filter
    _, standard_band, transformed = REFERENCE_BANDS[filter_name]
    catalog_path = image.savepath.catalogpath
    catalog = SafeCatalog(
        catalog_path,
        catalog_type="all",
        info=catalog_info_for(image, catalog_path),
        data=catalog_table,
    )
    calibration = PhotometricCalibration()
    calibration.catalogutils = PreparedReferenceUtility(reference)

    calibration_module = importlib.import_module("ezphot.methods.photometriccalibration")
    original_catalog_class = calibration_module.Catalog
    try:
        calibration_module.Catalog = SafeCatalog
        with tempfile.TemporaryDirectory(
            prefix=f"sn2026kid-zp-{filter_name}-", dir="/tmp"
        ) as directory:
            temporary_path = Path(directory) / image.path.name
            fits.writeto(
                temporary_path,
                np.asarray(image.data),
                image.header,
                overwrite=True,
                checksum=True,
            )
            calibration_image = image_instance(temporary_path, telescope_info)
            calibration_image.header["FILTER"] = standard_band
            calibration_image.savedir = Path(directory)
            (
                calibration_image,
                calibrated_catalog,
                filtered_catalog,
                update_header,
            ) = calibration.photometric_calibration(
                target_img=calibration_image,
                target_catalog=catalog,
                catalog_type="APASS",
                catalog_version="v1",
                max_distance_second=2.5,
                min_number_of_sources=10,
                calculate_color_terms=True,
                calculate_mag_terms=True,
                mag_lower=11,
                mag_upper=18,
                dynamic_mag_range=False,
                classstar_lower=0.35,
                elongation_upper=2.0,
                elongation_sigma=5,
                fwhm_lower_arcsec=0.7,
                fwhm_upper_arcsec=10,
                fwhm_sigma=5,
                flag_upper=1,
                maskflag_upper=1,
                inner_fraction=0.9,
                isolation_radius_arcsec=10,
                magnitude_key=PRIMARY_MAG_KEY,
                magnitudeerr_key=PRIMARY_MAG_KEY.replace("MAG_", "MAGERR_"),
                fwhm_key="FWHM_WORLD",
                ra_key="X_WORLD",
                dec_key="Y_WORLD",
                classstar_key="CLASS_STAR",
                elongation_key="ELONGATION",
                flag_key="FLAGS",
                maskflag_key="IMAFLAGS_ISO",
                update_header=True,
                save=True,
                verbose=verbose,
                visualize=False,
                save_fig=True,
                save_refcat=True,
            )
    finally:
        calibration_module.Catalog = original_catalog_class

    for key, value in update_header.items():
        image.header[key] = value
    image.header["ZPCALC"] = (True, "Catalog zero point calculated")
    image.header["ZPREF"] = (REFERENCE_LABEL, "Photometric reference")
    image.header["ZPBAND"] = (standard_band, "Reference passband")
    image.header["ZPAPPROX"] = (transformed, "Reference passband is transformed")
    image.header["ZPMETH"] = (CALIBRATION_METHOD, "Zeropoint calibration method")
    image.header["ZPSAVEFG"] = (True, "Photometric calibration figures saved")
    image.header["NZPSTAR"] = (len(filtered_catalog.data), "Selected zero-point stars")
    image.header["ZPMATCH"] = (2.5, "Reference match radius [arcsec]")
    image.header["ZPUTC"] = (now_utc(), "UTC zero-point completion time")
    image.update_status("ZPCALC")
    persist_image_header(image)

    calibrated_catalog.info = catalog_info_for(image, catalog_path)
    calibrated_catalog.save_info()
    expected_figures = calibration_png_paths(image.path)
    missing_figures = [path for path in expected_figures if not path.exists()]
    if missing_figures:
        raise RuntimeError(
            "PhotometricCalibration did not create: "
            + ", ".join(str(path) for path in missing_figures)
        )
    figure_paths = [str(path) for path in expected_figures]
    return {
        "reference_band": standard_band,
        "transformed_band": transformed,
        "matched_stars": int(len(filtered_catalog.data)),
        "primary_key": "ZP_APER_2",
        "primary_zp": float(image.header["ZP_APER_2"]),
        "primary_zp_scatter": float(image.header["ZPERR_APER_2"]),
        "method": CALIBRATION_METHOD,
        "save_fig": True,
        "figures": figure_paths,
    }


def photometry_in_safe_temp(
    image: ScienceImage,
    background: Background,
    bkgrms: Errormap,
    invalid_mask: Mask,
    telescope_info,
    verbose: bool,
) -> Table:
    """Run ezphot ``photometry_sex`` from a path without spaces.

    Source Extractor 2.25 truncates the input image name at the space in the
    telescope directory ``1.0-m KASI``.  The data passed here are the requested
    image-derived background subtraction; no master-frame noise model is used.
    Every PNG emitted by ``photometry_sex`` is copied back beside the science
    image before the temporary directory is removed.
    """
    output_path = image.savepath.catalogpath
    photometry_figure = Path(str(output_path) + ".png")
    subtracted_figure = Path(str(image.path) + ".photometry_subbkg.png")
    with tempfile.TemporaryDirectory(prefix="sn2026kid-phot-", dir="/tmp") as directory:
        directory = Path(directory)
        image_path = directory / "image.fits"
        fits.writeto(
            image_path,
            np.asarray(image.data, dtype=np.float32),
            image.header,
            overwrite=True,
            checksum=True,
        )
        temporary_image = image_instance(image_path, telescope_info)
        temporary_image.savedir = directory
        rms_path = temporary_image.savepath.bkgrmspath
        mask_path = temporary_image.savepath.invalidmaskpath
        background_path = temporary_image.savepath.bkgpath
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
        fits.writeto(
            background_path,
            np.asarray(background.data, dtype=np.float32),
            background.header,
            overwrite=True,
            checksum=True,
        )

        temporary_background = Background(background_path, load=True)
        temporary_rms = Errormap(rms_path, emaptype="bkgrms", load=True)
        temporary_mask = Mask(mask_path, masktype="invalid", load=True)
        import importlib

        photometry_module = importlib.import_module("ezphot.methods.aperturephotometry")
        original_catalog_class = photometry_module.Catalog
        try:
            photometry_module.Catalog = SafeCatalog
            temporary_catalog = temporary_image.photometry_sex(
                target_bkg=temporary_background,
                target_bkgrms=temporary_rms,
                target_mask=temporary_mask,
                detection_sigma=1.5,
                aperture_diameter_arcsec=APERTURES_ARCSEC,
                aperture_diameter_seeing=APERTURES_SEEING,
                saturation_level=60000,
                kron_factor=2.5,
                save=True,
                verbose=verbose,
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
        subtracted_figures = sorted(directory.glob("subbkg_*.fits.subbkg.png"))
        if not temporary_photometry_figure.exists():
            raise RuntimeError("photometry_sex did not create its catalog PNG")
        if len(subtracted_figures) != 1:
            raise RuntimeError(
                f"photometry_sex created {len(subtracted_figures)} background-subtraction PNGs"
            )
        shutil.copy2(temporary_photometry_figure, photometry_figure)
        shutil.copy2(subtracted_figures[0], subtracted_figure)

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
    Path(str(output_path) + ".info").write_text(json.dumps(catalog_info, indent=2) + "\n")
    image.header["PHOTSFG"] = (True, "photometry_sex PNGs saved")
    image.header["PHOTPNG"] = (photometry_figure.name, "Photometry source-review PNG")
    image.header["PHOTSUBP"] = (
        subtracted_figure.name,
        "Photometry background-subtraction PNG",
    )
    if verbose:
        print(f"Saved: {output_path}")
    return catalog


def calibrate_zero_points(
    image: ScienceImage,
    catalog: Table,
    reference: Table,
    min_stars: int = 10,
) -> dict:
    data = catalog
    filter_name = image.filter
    reference_key = f"{filter_name}_mag"
    if reference_key not in reference.colnames:
        raise KeyError(f"No reference passband for {filter_name}")
    if PRIMARY_MAG_KEY not in data.colnames:
        raise KeyError(f"Photometry catalog lacks {PRIMARY_MAG_KEY}")

    observed_coord = SkyCoord(data["X_WORLD"], data["Y_WORLD"], unit="deg")
    reference_coord = SkyCoord(reference["ra"], reference["dec"], unit="deg")
    ref_indices, separations, _ = match_coordinates_sky(observed_coord, reference_coord)
    separations_arcsec = separations.arcsec
    obs_indices = np.arange(len(data))
    ref_mag = np.asarray(reference[reference_key][ref_indices], dtype=float)
    reference_error_key = f"{filter_name}_err"
    ref_magerr = (
        np.asarray(reference[reference_error_key][ref_indices], dtype=float)
        if reference_error_key in reference.colnames
        else np.full(len(data), np.nan)
    )
    inst_mag = finite_column(data, PRIMARY_MAG_KEY)
    magerr = finite_column(data, PRIMARY_MAG_KEY.replace("MAG_", "MAGERR_"))

    quality = (
        (separations_arcsec < 2.5)
        & np.isfinite(ref_mag)
        & np.isfinite(ref_magerr)
        & np.isfinite(inst_mag)
        & np.isfinite(magerr)
        & (ref_mag > 11)
        & (ref_mag < 19)
        & (ref_magerr > 0)
        & (ref_magerr < 0.2)
        & (magerr > 0)
        & (magerr < 0.2)
    )
    if "FLAGS" in data.colnames:
        quality &= finite_column(data, "FLAGS", 99) <= 1
    if "IMAFLAGS_ISO" in data.colnames:
        quality &= finite_column(data, "IMAFLAGS_ISO", 99) <= 1
    if "CLASS_STAR" in data.colnames:
        quality &= finite_column(data, "CLASS_STAR", 0) >= 0.35
    if "ELONGATION" in data.colnames:
        quality &= finite_column(data, "ELONGATION", 99) <= 2.0
    if "FWHM_WORLD" in data.colnames:
        fwhm_arcsec = finite_column(data, "FWHM_WORLD") * 3600
        quality &= (fwhm_arcsec > 0.7) & (fwhm_arcsec < 10)

    selected_obs = obs_indices[quality]
    selected_ref = ref_indices[quality]
    selected_sep = separations_arcsec[quality]
    primary_zp = ref_mag[quality] - inst_mag[quality]
    clipped = sigma_clip(primary_zp, sigma=3, maxiters=5, masked=True)
    keep = ~np.ma.getmaskarray(clipped) & np.isfinite(primary_zp)
    selected_obs = selected_obs[keep]
    selected_ref = selected_ref[keep]
    selected_sep = selected_sep[keep]
    if len(selected_obs) < min_stars:
        raise RuntimeError(
            f"Only {len(selected_obs)} clean {filter_name} zero-point stars; need {min_stars}"
        )

    update_header: dict[str, tuple[float, str]] = {}
    calibrated_keys = []
    for mag_key in [
        name
        for name in data.colnames
        if name.startswith("MAG_") and not name.startswith("MAG_REF")
    ]:
        mag_values = finite_column(data, mag_key)
        zp_values = np.asarray(reference[reference_key][selected_ref], dtype=float) - mag_values[selected_obs]
        clipped_zp = sigma_clip(zp_values, sigma=3, maxiters=5, masked=True)
        valid = ~np.ma.getmaskarray(clipped_zp) & np.isfinite(zp_values)
        if np.count_nonzero(valid) < min_stars:
            continue
        zp = float(np.nanmedian(zp_values[valid]))
        zperr = float(np.nanstd(zp_values[valid]))
        zp_key = mag_key.replace("MAG_", "ZP_")
        zperr_key = mag_key.replace("MAG_", "ZPERR_")
        sky_key = mag_key.replace("MAG_", "MAGSKY_")
        data[sky_key] = mag_values + zp
        data[zp_key] = np.full(len(data), zp)
        data[zperr_key] = np.full(len(data), zperr)
        update_header[zp_key] = (zp, f"Zeropoint for {mag_key}")
        update_header[zperr_key] = (zperr, f"Zeropoint scatter for {mag_key}")
        calibrated_keys.append(mag_key)

    primary_zp_key = PRIMARY_MAG_KEY.replace("MAG_", "ZP_")
    if primary_zp_key not in update_header:
        raise RuntimeError(f"Failed to determine {primary_zp_key}")

    reference_rows = Table()
    reference_rows["OBS_INDEX"] = selected_obs
    reference_rows["REF_INDEX"] = selected_ref
    reference_rows["SEP_ARCSEC"] = selected_sep
    reference_rows["RA"] = np.asarray(reference["ra"][selected_ref], dtype=float)
    reference_rows["DEC"] = np.asarray(reference["dec"][selected_ref], dtype=float)
    reference_rows["REF_MAG"] = np.asarray(reference[reference_key][selected_ref], dtype=float)
    reference_rows["INST_MAG"] = finite_column(data, PRIMARY_MAG_KEY)[selected_obs]
    reference_rows["ZP"] = reference_rows["REF_MAG"] - reference_rows["INST_MAG"]
    reference_rows.write(image.savepath.refcatalogpath, format="ascii.ecsv", overwrite=True)

    data.write(image.savepath.catalogpath, format="ascii", overwrite=True)
    _, band_name, transformed = REFERENCE_BANDS[filter_name]
    image.header.update(update_header)
    image.header["ZPCALC"] = (True, "Catalog zero point calculated")
    image.header["ZPREF"] = (REFERENCE_LABEL, "Photometric reference")
    image.header["ZPBAND"] = (band_name, "Reference passband")
    image.header["ZPAPPROX"] = (transformed, "Reference passband is transformed")
    image.header["NZPSTAR"] = (len(selected_obs), "Clean zero-point stars")
    image.header["ZPMATCH"] = (2.5, "Reference match radius [arcsec]")
    image.header["ZPUTC"] = (now_utc(), "UTC zero-point completion time")
    if "FWHM_WORLD" in data.colnames:
        seeing = float(np.nanmedian(finite_column(data, "FWHM_WORLD")[selected_obs]) * 3600)
        image.header["SEEING"] = (seeing, "Median stellar FWHM [arcsec]")
        image.header["PEEING"] = (seeing / PIXEL_SCALE, "Median stellar FWHM [pixel]")
    image.update_status("ZPCALC")
    return {
        "reference_band": band_name,
        "transformed_band": transformed,
        "matched_stars": int(len(selected_obs)),
        "primary_key": primary_zp_key,
        "primary_zp": float(update_header[primary_zp_key][0]),
        "primary_zp_scatter": float(
            update_header[PRIMARY_MAG_KEY.replace("MAG_", "ZPERR_")][0]
        ),
        "calibrated_magnitude_keys": calibrated_keys,
    }


def products_complete(path: Path, require_current_reference: bool = True) -> bool:
    header = fits.getheader(path, memmap=False)
    required = [
        path.with_name(path.name + suffix)
        for suffix in (".invalidmask", ".srcmask", ".bkgmap", ".bkgrms", ".cat", ".refcat")
    ]
    required_pngs = [
        path.with_name(path.name + suffix)
        for suffix in (
            ".invalidmask.png",
            ".srcmask.png",
            ".bkgmap.png",
            ".bkgrms.png",
        )
    ] + photometry_png_paths(path) + calibration_png_paths(path)
    try:
        wcs_ok = WCS(header).has_celestial
    except Exception:
        wcs_ok = False
    return (
        wcs_ok
        and bool(header.get("ZPCALC"))
        and str(header.get("BKGMASK", "")) == BACKGROUND_MASK_VERSION
        and bool(header.get("PHOTSFG", False))
        and np.isfinite(float(header.get("ZP_APER_2", np.nan)))
        and (
            not require_current_reference
            or (
                str(header.get("ZPREF", "")) == REFERENCE_LABEL
                and str(header.get("ZPMETH", "")) == CALIBRATION_METHOD
                and bool(header.get("ZPSAVEFG", False))
            )
        )
        and all(product.exists() for product in required + required_pngs)
    )


def process_image(
    path: Path,
    telescope_info,
    reference: Table,
    overwrite: bool,
    verbose: bool,
) -> dict:
    if products_complete(path) and not overwrite:
        header = fits.getheader(path, memmap=False)
        catalog = Table.read(Path(str(path) + ".cat"), format="ascii")
        return {
            "path": str(path),
            "filter": str(header["FILTER"]),
            "status": "existing",
            "sources": len(catalog),
            "zp": float(header["ZP_APER_2"]),
            "zp_scatter": float(header["ZPERR_APER_2"]),
            "zp_stars": int(header["NZPSTAR"]),
        }

    # Reuse valid astrometry/background/photometry products when only the
    # reference catalog has changed.  This is the expected path when replacing
    # the earlier synthetic-Gaia calibration with SkyCatalog/APASS.
    if products_complete(path, require_current_reference=False) and not overwrite:
        image = image_instance(path, telescope_info)
        catalog = Table.read(image.savepath.catalogpath, format="ascii")
        zero_point = calibrate_with_ezphot(
            image, catalog, reference, telescope_info, verbose
        )
        verify_checksum(path)
        return {
            "path": str(path),
            "filter": image.filter,
            "status": "recalibrated",
            "sources": int(len(catalog)),
            "zero_point": zero_point,
        }

    image, astrometry = solve_astrometry_safe(
        path, telescope_info, reference, overwrite, verbose
    )
    invalid_mask = image.calculate_invalidmask(
        save=True, verbose=verbose, visualize=False, save_fig=True
    )
    source_mask = image.calculate_sourcemask(
        target_srcmask=None,
        sigma=5.0,
        mask_radius_factor=3,
        saturation_level=60000,
        save=True,
        verbose=verbose,
        visualize=False,
        save_fig=True,
    )
    source_mask = add_host_galaxy_mask(
        image=image,
        source_mask=source_mask,
        save=True,
        save_fig=True,
        verbose=verbose,
    )
    background = image.calculate_bkg(
        target_srcmask=source_mask,
        target_ivpmask=invalid_mask,
        is_2D_bkg=True,
        box_size=64,
        filter_size=3,
        correct_global_offset=True,
        save=True,
        verbose=verbose,
        visualize=False,
        save_fig=True,
    )
    bkgrms = image.calculate_bkgrms(
        target_srcmask=source_mask,
        target_ivpmask=invalid_mask,
        box_size=64,
        filter_size=3,
        save=True,
        verbose=verbose,
        visualize=False,
        save_fig=True,
    )
    catalog = photometry_in_safe_temp(
        image=image,
        background=background,
        bkgrms=bkgrms,
        invalid_mask=invalid_mask,
        telescope_info=telescope_info,
        verbose=verbose,
    )
    zero_point = calibrate_with_ezphot(
        image, catalog, reference, telescope_info, verbose
    )

    image.header["BKGFILE"] = (background.path.name, "Image-derived SEP background map")
    image.header["RMSFILE"] = (bkgrms.path.name, "Image-derived SEP background RMS")
    image.header["CATFILE"] = (image.savepath.catalogpath.name, "Source Extractor catalog")
    image.header["BKG2D"] = (True, "Background estimated directly from science image")
    image.header["BKGMASK"] = (
        BACKGROUND_MASK_VERSION,
        "Host-aware source-mask version",
    )
    image.header["HSTMAJ"] = (
        HOST_MAJOR_ARCMIN * HOST_MASK_SCALE,
        "Masked host major diameter [arcmin]",
    )
    image.header["HSTMIN"] = (
        HOST_MINOR_ARCMIN * HOST_MASK_SCALE,
        "Masked host minor diameter [arcmin]",
    )
    image.header["HSTPA"] = (HOST_POSITION_ANGLE_DEG, "Host PA [deg E of N]")
    image.header["PHOTDONE"] = (True, "Aperture photometry completed")
    image.header["PHOTUTC"] = (now_utc(), "UTC photometry completion time")
    image.update_status("PHOTOMETRY")
    persist_image_header(image)

    for fits_product in (
        path,
        invalid_mask.path,
        source_mask.path,
        background.path,
        bkgrms.path,
    ):
        add_checksum(Path(fits_product))
        verify_checksum(Path(fits_product))

    bkg_data = np.asarray(background.data, dtype=float)
    rms_data = np.asarray(bkgrms.data, dtype=float)
    return {
        "path": str(path),
        "filter": image.filter,
        "status": "processed",
        "astrometry": astrometry,
        "invalid_pixels": int(np.count_nonzero(invalid_mask.data)),
        "source_mask_pixels": int(np.count_nonzero(source_mask.data)),
        "background_median": float(np.nanmedian(bkg_data)),
        "background_rms_median": float(np.nanmedian(rms_data)),
        "sources": int(len(catalog)),
        "zero_point": zero_point,
    }


def stack_filter(
    paths: list[Path],
    filter_name: str,
    night: str,
    telescope_info,
    overwrite: bool,
    verbose: bool,
) -> dict:
    output = paths[0].parent / f"stack.SN2026kid.{night}.{filter_name}.fits"
    rms_output = Path(str(output) + ".bkgrms")
    if output.exists() and rms_output.exists() and not overwrite:
        header = fits.getheader(output, memmap=False)
        if (
            bool(header.get("STACKMSK", False))
            and str(header.get("BKGMASK", "")) == BACKGROUND_MASK_VERSION
            and str(header.get("STKCEN", "")) == STACK_CENTER_VERSION
            and np.isclose(
                float(header.get("STKCRA", np.nan)), HOST_CENTER.ra.deg, atol=1e-8
            )
            and np.isclose(
                float(header.get("STKCDEC", np.nan)), HOST_CENTER.dec.deg, atol=1e-8
            )
            and str(header.get("ZPREF", "")) == REFERENCE_LABEL
            and str(header.get("ZPMETH", "")) == CALIBRATION_METHOD
            and int(header.get("NCOMBINE", 0)) == len(paths)
        ):
            return {
                "filter": filter_name,
                "status": "existing",
                "output": str(output),
                "bkgrms": str(rms_output),
                "ncombine": int(header["NCOMBINE"]),
                "total_exposure": float(header["TOTEXP"]),
            }

    source_images = [image_instance(path, telescope_info) for path in paths]
    # Use a stable, scientifically meaningful output grid.  CRVAL is merely
    # the WCS reference coordinate and is not guaranteed to be the geometric
    # image center when CRPIX differs between astrometric solutions.
    stack_center_ra = float(HOST_CENTER.ra.deg)
    stack_center_dec = float(HOST_CENTER.dec.deg)

    # Both solve-field and SWarp parse command-line paths poorly when a parent
    # directory contains a space (the production tree contains ``1.0-m KASI``).
    # Stage only SWarp's inputs/intermediates in a no-space temporary directory;
    # stack_multiprocess writes the validated final products to the archive.
    with tempfile.TemporaryDirectory(
        prefix=f"sn2026kid-stack-{filter_name}-", dir="/tmp"
    ) as temp_name:
        temp_dir = Path(temp_name)
        images = []
        rms_maps = []
        reference_zp = min(float(image.header["ZP_APER_2"]) for image in source_images)
        for index, source_image in enumerate(source_images):
            temp_path = temp_dir / f"image-{index:02d}.fits"
            science_data = np.asarray(source_image.data, dtype=np.float32)
            background_data = np.asarray(
                fits.getdata(source_image.savepath.bkgpath, memmap=False), dtype=np.float32
            )
            rms_data = np.asarray(
                fits.getdata(source_image.savepath.bkgrmspath, memmap=False), dtype=np.float32
            )
            image_zp = float(source_image.header["ZP_APER_2"])
            delta_zp = reference_zp - image_zp
            scale_factor = 10 ** (0.4 * delta_zp)

            staged_header = source_image.header.copy()
            staged_header["BKGVALU"] = (0.0, "Background after subtraction")
            staged_header["SCLEKEY"] = "ZP_APER_2"
            staged_header["SCLEREF"] = reference_zp
            staged_header["SCLEZP"] = delta_zp
            staged_header["SCLEFACT"] = scale_factor
            for key in list(staged_header):
                if key.startswith("ZP_"):
                    staged_header[key] = float(staged_header[key]) + delta_zp
            fits.writeto(
                temp_path,
                (science_data - background_data) * scale_factor,
                staged_header,
                overwrite=True,
                checksum=True,
            )
            temp_image = image_instance(temp_path, telescope_info)
            staged_rms_path = temp_image.savepath.bkgrmspath
            rms_header = fits.getheader(source_image.savepath.bkgrmspath, memmap=False)
            fits.writeto(
                staged_rms_path,
                rms_data * scale_factor,
                rms_header,
                overwrite=True,
                checksum=True,
            )
            images.append(temp_image)
            staged_rms = Errormap(staged_rms_path, emaptype="bkgrms", load=True)
            staged_rms.savedir = temp_dir
            rms_maps.append(staged_rms)

        stacker = Stack()
        original_directory = Path.cwd()
        try:
            # SWarp writes its XML diagnostics in the process working directory.
            # Keep those disposable files inside the temporary staging tree too.
            os.chdir(temp_dir)
            prepared_images, prepared_rms = stacker.prepare_images(
                target_imglist=images,
                target_bkglist=None,
                target_bkgrmslist=rms_maps,
                n_proc=1,
                scale=False,
                zp_key="ZP_APER_2",
                convolve=False,
                reproject=True,
                reproject_type="LANCZOS3",
                center_ra=stack_center_ra,
                center_dec=stack_center_dec,
                pixel_scale=PIXEL_SCALE,
                x_size=2048,
                y_size=2048,
                keep_header_keys=["GAIN", "EGAIN", "FILTER", "OBJECT"],
                verbose=verbose,
                save=False,
                clear=False,
            )
        finally:
            os.chdir(original_directory)
        stack_image, stack_rms = stacker.stack_multiprocess(
            target_imglist=list(prepared_images),
            target_bkgrmslist=list(prepared_rms),
            target_outpath=output,
            bkgrms_outpath=rms_output,
            n_proc=1,
            combine_type="weighted",
            clip_type=None,
            verbose=verbose,
            save=True,
            remove_intermediate=False,
        )
    stack_image.header["NIGHTDIR"] = night
    stack_image.header["STACKZP"] = "ZP_APER_2"
    stack_image.header["STACKUTC"] = now_utc()
    stack_image.header["STKCEN"] = (
        STACK_CENTER_VERSION,
        "Stack center definition",
    )
    stack_image.header["STKCRA"] = (
        stack_center_ra,
        "Stack center RA [deg, ICRS]",
    )
    stack_image.header["STKCDEC"] = (
        stack_center_dec,
        "Stack center Dec [deg, ICRS]",
    )
    stack_image.header["BKG2D"] = True
    stack_image.header["BKGMASK"] = (
        BACKGROUND_MASK_VERSION,
        "Input background host-mask version",
    )
    for key in ("ZPREF", "ZPBAND", "ZPAPPROX", "ZPMETH", "ZPSAVEFG"):
        stack_image.header[key] = source_images[0].header[key]
    remove_stale_sip(stack_image.header)
    observation_time = Time(stack_image.header["DATE-OBS"], format="isot", scale="utc")
    stack_image.header["JD"] = (float(observation_time.jd), "Julian Date at stack midpoint")
    stack_image.header["MJD"] = (float(observation_time.mjd), "Modified JD at stack midpoint")
    stack_image.header["MJD-OBS"] = (
        float(observation_time.mjd),
        "Modified JD at stack midpoint",
    )
    valid_coverage = np.isfinite(stack_rms.data) & (stack_rms.data > 0)
    stack_image.data[~valid_coverage] = np.nan
    stack_image.header["STACKMSK"] = (True, "Pixels without weight set to NaN")
    persist_image_data_and_header(stack_image)
    add_checksum(stack_rms.path)
    verify_checksum(stack_image.path)
    verify_checksum(stack_rms.path)

    data = fits.getdata(stack_image.path, memmap=False)
    header = fits.getheader(stack_image.path, memmap=False)
    if int(header["NCOMBINE"]) != len(paths):
        raise ValueError(f"Stack {output} has incorrect NCOMBINE")
    return {
        "filter": filter_name,
        "status": "stacked",
        "output": str(stack_image.path),
        "bkgrms": str(stack_rms.path),
        "ncombine": int(header["NCOMBINE"]),
        "total_exposure": float(header["TOTEXP"]),
        "finite_fraction": float(np.mean(np.isfinite(data))),
        "median": float(np.nanmedian(data)),
        "std": float(np.nanstd(data)),
    }


def process_stack_catalog(
    stack_result: dict,
    reference: Table,
    telescope_info,
    overwrite: bool,
    verbose: bool,
) -> dict:
    """Run aperture photometry and PhotometricCalibration on one stack."""
    path = Path(stack_result["output"])
    image = image_instance(path, telescope_info)
    catalog_path = image.savepath.catalogpath
    if (
        not overwrite
        and bool(image.header.get("STKPHOT", False))
        and str(image.header.get("BKGMASK", "")) == BACKGROUND_MASK_VERSION
        and bool(image.header.get("PHOTSFG", False))
        and str(image.header.get("ZPMETH", "")) == CALIBRATION_METHOD
        and catalog_path.exists()
        and image.savepath.refcatalogpath.exists()
        and all(
            figure_path.exists()
            for figure_path in photometry_png_paths(path) + calibration_png_paths(path)
        )
    ):
        catalog = Table.read(catalog_path, format="ascii")
        return {
            "path": str(path),
            "filter": image.filter,
            "status": "existing",
            "sources": len(catalog),
            "zp": float(image.header["ZP_APER_2"]),
            "zp_scatter": float(image.header["ZPERR_APER_2"]),
            "zp_stars": int(image.header["NZPSTAR"]),
        }

    rms = Errormap(image.savepath.bkgrmspath, emaptype="bkgrms", load=True)
    zero_background = Background(image.savepath.bkgpath, load=False)
    zero_background.data = np.zeros_like(image.data, dtype=np.float32)
    zero_background.header = image.header.copy()
    invalid_mask = Mask(image.savepath.invalidmaskpath, masktype="invalid", load=False)
    invalid_mask.data = ~np.isfinite(image.data) | ~np.isfinite(rms.data) | (rms.data <= 0)
    invalid_mask.header = image.header.copy()
    invalid_mask.header["MASKTYPE"] = "invalid"

    catalog = photometry_in_safe_temp(
        image=image,
        background=zero_background,
        bkgrms=rms,
        invalid_mask=invalid_mask,
        telescope_info=telescope_info,
        verbose=verbose,
    )
    zero_point = calibrate_with_ezphot(
        image, catalog, reference, telescope_info, verbose
    )
    image.header["STKPHOT"] = (True, "Stack aperture photometry completed")
    image.header["PHOTDONE"] = (True, "Aperture photometry completed")
    image.header["PHOTUTC"] = (now_utc(), "UTC stack photometry completion time")
    image.header["CATFILE"] = (catalog_path.name, "Stack Source Extractor catalog")
    image.update_status("PHOTOMETRY")
    persist_image_header(image)
    verify_checksum(path)
    return {
        "path": str(path),
        "filter": image.filter,
        "status": "processed",
        "sources": len(catalog),
        "zero_point": zero_point,
    }


def make_qc_pngs(
    image_paths: list[Path],
    stack_results: list[dict],
    output_directory: Path,
) -> list[str]:
    """Create image/background/RMS, stack, and APASS-ZP review figures."""
    import matplotlib.pyplot as plt
    from astropy.visualization import ZScaleInterval

    plt.switch_backend("Agg")
    output_directory.mkdir(parents=True, exist_ok=True)
    created: list[str] = []

    def zscale(data: np.ndarray) -> tuple[float, float]:
        sample = np.asarray(data)[::4, ::4]
        finite = sample[np.isfinite(sample)]
        if finite.size == 0:
            return 0.0, 1.0
        return tuple(float(value) for value in ZScaleInterval().get_limits(finite))

    def percentile_limits(data: np.ndarray) -> tuple[float, float]:
        finite = np.asarray(data)[np.isfinite(data)]
        if finite.size == 0:
            return 0.0, 1.0
        low, high = np.nanpercentile(finite, [1, 99])
        return float(low), float(high)

    for path in image_paths:
        science = fits.getdata(path, memmap=False)
        background = fits.getdata(Path(str(path) + ".bkgmap"), memmap=False)
        rms = fits.getdata(Path(str(path) + ".bkgrms"), memmap=False)
        catalog = Table.read(Path(str(path) + ".cat"), format="ascii")
        header = fits.getheader(path, memmap=False)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        panels = [
            (science, "Science + detections", "gray", zscale(science)),
            (background, "Direct SEP background", "viridis", percentile_limits(background)),
            (rms, "Direct-image background RMS", "magma", percentile_limits(rms)),
        ]
        for axis, (data, title, cmap, limits) in zip(axes, panels):
            shown = axis.imshow(
                data,
                origin="lower",
                cmap=cmap,
                vmin=limits[0],
                vmax=limits[1],
                interpolation="nearest",
            )
            axis.set_title(title)
            axis.set_xlabel("X [pixel]")
            axis.set_ylabel("Y [pixel]")
            if axis is not axes[0]:
                fig.colorbar(shown, ax=axis, shrink=0.78)
        axes[0].scatter(
            finite_column(catalog, "X_IMAGE"),
            finite_column(catalog, "Y_IMAGE"),
            s=9,
            facecolors="none",
            edgecolors="tab:red",
            linewidths=0.35,
        )
        fig.suptitle(
            f"{path.name} | {header['FILTER']} | APASS ZP={header['ZP_APER_2']:.3f}",
            fontsize=13,
        )
        output = output_directory / f"{path.name}.qc.png"
        fig.savefig(output, dpi=150)
        plt.close(fig)
        created.append(str(output))

    stack_paths = [Path(result["output"]) for result in stack_results]
    for path in stack_paths:
        data = fits.getdata(path, memmap=False)
        header = fits.getheader(path, memmap=False)
        fig, axis = plt.subplots(figsize=(8, 8), constrained_layout=True)
        limits = zscale(data)
        axis.imshow(
            data,
            origin="lower",
            cmap="gray",
            vmin=limits[0],
            vmax=limits[1],
            interpolation="nearest",
        )
        axis.set_title(
            f"SN2026kid {header['FILTER']} stack | N={header['NCOMBINE']} | "
            f"{header['TOTEXP']:.0f} s"
        )
        axis.set_xlabel("X [pixel]")
        axis.set_ylabel("Y [pixel]")
        output = output_directory / f"{path.name}.png"
        fig.savefig(output, dpi=180)
        plt.close(fig)
        created.append(str(output))

    if stack_paths:
        fig, axes = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)
        for axis, path in zip(axes.flat, sorted(stack_paths)):
            data = fits.getdata(path, memmap=False)
            header = fits.getheader(path, memmap=False)
            limits = zscale(data)
            axis.imshow(
                data,
                origin="lower",
                cmap="gray",
                vmin=limits[0],
                vmax=limits[1],
                interpolation="nearest",
            )
            axis.set_title(f"{header['FILTER']} | 3 x 60 s")
            axis.set_xlabel("X [pixel]")
            axis.set_ylabel("Y [pixel]")
        summary_output = output_directory / "stack_summary_BVRI.png"
        fig.suptitle("SN2026kid / NGC5907 — 2026_0714")
        fig.savefig(summary_output, dpi=180)
        plt.close(fig)
        created.append(str(summary_output))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for axis, filter_name in zip(axes.flat, FILTERS):
        filter_paths = [
            path
            for path in image_paths
            if str(fits.getheader(path, memmap=False)["FILTER"]) == filter_name
        ]
        headers = [fits.getheader(path, memmap=False) for path in filter_paths]
        frame_labels = [path.stem.split(".")[-1] for path in filter_paths]
        zeropoints = [float(header["ZP_APER_2"]) for header in headers]
        scatters = [float(header["ZPERR_APER_2"]) for header in headers]
        axis.errorbar(
            np.arange(len(filter_paths)),
            zeropoints,
            yerr=scatters,
            fmt="o",
            capsize=4,
            color="tab:blue",
        )
        _, band, transformed = REFERENCE_BANDS[filter_name]
        qualifier = " (transformed)" if transformed else " (direct)"
        axis.set_title(f"{filter_name} → {band}{qualifier}")
        axis.set_xlabel("Frame")
        axis.set_ylabel("Zeropoint [mag]")
        axis.set_xticks(np.arange(len(frame_labels)), frame_labels)
        axis.grid(alpha=0.2)
    zp_output = output_directory / "zeropoint_APASS_BVRI.png"
    fig.suptitle("PhotometricCalibration / APASS DR9 — MAG_APER_2")
    fig.savefig(zp_output, dpi=180)
    plt.close(fig)
    created.append(str(zp_output))
    return created


def run(args: argparse.Namespace) -> dict:
    source_paths = discover_images(args.input_root, args.night, args.limit)
    paths = stage_images(source_paths, args.test_root)
    telescope_info = telinfo(paths[0])
    first = image_instance(paths[0], telescope_info)
    center = SkyCoord(first.ra, first.dec, unit="deg")
    processing_root = args.test_root if args.test_root else args.input_root
    reference_path = (
        args.reference_catalog
        if args.reference_catalog
        else processing_root / "reference.APASS_DR9_BVRI.ecsv"
    )
    reference = query_reference_catalog(center, reference_path, args.overwrite_reference)

    image_results = []
    failures = []
    for index, path in enumerate(paths, start=1):
        print(f"[{index:02d}/{len(paths):02d}] {path.name}")
        try:
            result = process_image(
                path,
                telescope_info,
                reference,
                overwrite=args.overwrite,
                verbose=args.verbose,
            )
            image_results.append(result)
            zp = result.get("zero_point", {}).get("primary_zp", result.get("zp", np.nan))
            print(
                f"  OK {result['filter']}: sources={result['sources']}, "
                f"ZP_APER_2={float(zp):.4f}"
            )
        except Exception as error:
            failure = {
                "path": str(path),
                "error": f"{type(error).__name__}: {error}",
            }
            failures.append(failure)
            print(f"  FAIL {failure['error']}")

    stack_results = []
    if not args.skip_stack:
        successful_paths = [Path(result["path"]) for result in image_results]
        grouped: dict[str, list[Path]] = defaultdict(list)
        for path in successful_paths:
            grouped[str(fits.getheader(path, memmap=False)["FILTER"])].append(path)
        for filter_name, filter_paths in sorted(grouped.items()):
            if len(filter_paths) < 2:
                continue
            try:
                result = stack_filter(
                    filter_paths,
                    filter_name,
                    args.night,
                    telescope_info,
                    overwrite=args.overwrite,
                    verbose=args.verbose,
                )
                stack_results.append(result)
                print(
                    f"STACK {filter_name}: {result['ncombine']} images, "
                    f"{result['total_exposure']:.0f} s"
                )
            except Exception as error:
                failures.append(
                    {
                        "filter": filter_name,
                        "stage": "stack",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                print(f"STACK FAIL {filter_name}: {type(error).__name__}: {error}")

    stack_photometry_results = []
    for stack_result in stack_results:
        try:
            result = process_stack_catalog(
                stack_result,
                reference,
                telescope_info,
                overwrite=args.overwrite,
                verbose=args.verbose,
            )
            stack_photometry_results.append(result)
            zp = result.get("zero_point", {}).get("primary_zp", result.get("zp", np.nan))
            print(
                f"STACK PHOT {result['filter']}: sources={result['sources']}, "
                f"ZP_APER_2={float(zp):.4f}"
            )
        except Exception as error:
            failures.append(
                {
                    "filter": stack_result["filter"],
                    "stage": "stack_photometry",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(
                f"STACK PHOT FAIL {stack_result['filter']}: "
                f"{type(error).__name__}: {error}"
            )

    qc_pngs = []
    if not args.skip_png:
        try:
            qc_pngs = make_qc_pngs(
                [Path(result["path"]) for result in image_results],
                stack_results,
                processing_root / f"qc_{args.night}",
            )
            print(f"QC PNGs: {processing_root / f'qc_{args.night}'}")
        except Exception as error:
            failures.append(
                {
                    "stage": "qc_png",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(f"QC PNG FAIL: {type(error).__name__}: {error}")

    report = {
        "night": args.night,
        "target": TARGET_DEFAULT,
        "created_utc": now_utc(),
        "input_root": str(args.input_root),
        "processing_root": str(processing_root),
        "reference_catalog": str(reference_path),
        "reference_source": REFERENCE_LABEL,
        "reference_rows": len(reference),
        "filter_mapping": {
            key: {"band": value[1], "transformed": value[2]}
            for key, value in REFERENCE_BANDS.items()
        },
        "background_source": "direct image estimate (SEP), not master frames",
        "background_mask_version": BACKGROUND_MASK_VERSION,
        "host_mask": {
            "name": "NGC 5907",
            "center_ra_deg": HOST_CENTER.ra.deg,
            "center_dec_deg": HOST_CENTER.dec.deg,
            "catalog_major_arcmin": HOST_MAJOR_ARCMIN,
            "catalog_minor_arcmin": HOST_MINOR_ARCMIN,
            "position_angle_deg": HOST_POSITION_ANGLE_DEG,
            "scale_factor": HOST_MASK_SCALE,
            "masked_major_arcmin": HOST_MAJOR_ARCMIN * HOST_MASK_SCALE,
            "masked_minor_arcmin": HOST_MINOR_ARCMIN * HOST_MASK_SCALE,
        },
        "stack_center": {
            "version": STACK_CENTER_VERSION,
            "name": "NGC 5907",
            "frame": "ICRS",
            "ra_deg": HOST_CENTER.ra.deg,
            "dec_deg": HOST_CENTER.dec.deg,
        },
        "function_pngs": {
            "photometry_per_image": 2,
            "photometric_calibration_per_image": 4,
            "save_fig": True,
        },
        "images": image_results,
        "stacks": stack_results,
        "stack_photometry": stack_photometry_results,
        "qc_pngs": qc_pngs,
        "failures": failures,
    }
    report_path = processing_root / f"photometry_{args.night}_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name(f".{report_path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, report_path)
    print(f"Report: {report_path}")
    if failures:
        raise RuntimeError(f"Workflow completed with {len(failures)} failure(s)")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--night", default=NIGHT_DEFAULT)
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT_DEFAULT)
    parser.add_argument("--test-root", type=Path, help="Stage inputs and products here")
    parser.add_argument("--reference-catalog", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-stack", action="store_true")
    parser.add_argument("--skip-png", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-reference", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
