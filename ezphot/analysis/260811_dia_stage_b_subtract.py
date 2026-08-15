#!/usr/bin/env python
"""DIA Stage B — HOTPANTS subtraction of PS1 references from LOAO stacks.

Per stack:

1. astroalign registers the aligned PS1 reference (band mapping B,V->g,
   R->r, I->i) onto the stack's actual image content — this absorbs the
   per-stack astrometric offsets found in Stage A.
2. Both images are staged into a no-space temp directory (NaN filled with 0;
   both are background-subtracted) and HOTPANTS subtracts the template with
   the library defaults (template convolved, normalized to the science image).
3. The difference is written next to the stack as
   ``sub_stack.SN2026kid.<night>.<filter>.fits`` with provenance keys, NaN
   restored outside the common coverage.
4. A per-stack TAN WCS refit against APASS (``fit_wcs_from_points`` on the
   existing stack catalog stars) provides the corrected SN pixel position,
   stored in the difference header (``SNXPIX``/``SNYPIX``) for Stage C.

Without ``--nights`` it processes every night; the QC figure shows the
science/template/difference triplet around the SN for each processed stack
(capped at the 8 stacks of a 2-night test for readability).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ezphot-matplotlib")

import astroalign
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import fit_wcs_from_points

MODULE_PATH = Path(__file__).with_name("260805_photometry_loao_sn2026kid.py")
spec = importlib.util.spec_from_file_location("loao_photometry", MODULE_PATH)
phot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phot)

from ezphot.helper import Helper  # noqa: E402

SN_RA, SN_DEC = 228.9884428, 56.3089141
ROOTS = [
    Path("/home/hhchoi1022/ezphot/data/scidata/LOAO/LOAO_E2V_2x2/SN2026kid/1.0-m KASI"),
    Path("/home/hhchoi1022/ezphot/data/scidata/LOAO/LOAO_iKon_1x1/SN2026kid/1.0-m KASI"),
]
REF_DIR = Path("/home/hhchoi1022/ezphot/data/scidata/LOAO/SN2026kid/reference_images")
FILTER_TO_BAND = {"B102": "g", "V103": "g", "R104": "r", "I105": "i"}
APASS_PATH = Path(
    "/home/hhchoi1022/ezphot/data/scidata/LOAO/SN2026kid/reference.APASS_DR9_BVRI.ecsv"
)
REPORT_DIR = Path("/home/hhchoi1022/ezphot/data/scidata/LOAO/SN2026kid/stage_reports")
QC_DIR = Path("/home/hhchoi1022/ezphot/log/20260806_stage_run_qc")


def load_apass() -> SkyCoord:
    table = Table.read(APASS_PATH, format="ascii.ecsv")
    return SkyCoord(
        np.asarray(table["ra"], dtype=float),
        np.asarray(table["dec"], dtype=float),
        unit="deg",
    )


def refit_wcs(stack_path: Path, apass: SkyCoord):
    """TAN refit of the stack WCS from catalog stars matched to APASS."""
    catalog = Table.read(Path(str(stack_path) + ".cat"), format="ascii")
    for classstar_limit, min_matches in ((0.5, 10), (0.3, 10), (0.2, 6)):
        good = (
            (np.asarray(catalog["FLAGS"], dtype=float) == 0)
            & (np.asarray(catalog["CLASS_STAR"], dtype=float) > classstar_limit)
        )
        pixels = np.column_stack(
            [
                np.asarray(catalog["X_IMAGE"], dtype=float)[good] - 1,
                np.asarray(catalog["Y_IMAGE"], dtype=float)[good] - 1,
            ]
        )
        wcs = WCS(fits.getheader(stack_path, memmap=False))
        sky = wcs.pixel_to_world(pixels[:, 0], pixels[:, 1])
        index, separation, _ = sky.match_to_catalog_sky(apass)
        matched = separation.arcsec < 3.0
        if matched.sum() >= min_matches:
            fitted = fit_wcs_from_points(
                (pixels[matched, 0], pixels[matched, 1]),
                apass[index[matched]],
                projection="TAN",
            )
            residual = fitted.world_to_pixel(apass[index[matched]])
            radial = np.hypot(
                residual[0] - pixels[matched, 0], residual[1] - pixels[matched, 1]
            )
            return fitted, int(matched.sum()), float(np.median(radial))
    return None, 0, np.nan


def valid_transform(transform) -> bool:
    return (
        abs(transform.translation[0]) <= 30
        and abs(transform.translation[1]) <= 30
        and abs(np.rad2deg(transform.rotation)) <= 0.5
        and abs(transform.scale - 1) <= 0.005
    )


def subtract_one(stack_path: Path, helper: Helper, apass: SkyCoord) -> dict:
    header = fits.getheader(stack_path, memmap=False)
    night = str(header["NIGHTDIR"])
    filter_name = str(header["FILTER"])
    band = FILTER_TO_BAND[filter_name]
    science = fits.getdata(stack_path, memmap=False).astype(np.float32)
    reference = fits.getdata(
        REF_DIR / f"ref.PS1.{band}.aligned.fits", memmap=False
    ).astype(np.float32)

    fitted_wcs, refit_stars, refit_residual = refit_wcs(stack_path, apass)

    science_filled = np.nan_to_num(science, nan=0.0)
    reference_filled = np.nan_to_num(reference, nan=0.0)
    transform = None
    try:
        candidate, _ = astroalign.find_transform(
            reference_filled, science_filled, detection_sigma=8
        )
        if valid_transform(candidate):
            transform = candidate
    except Exception:
        transform = None

    if transform is not None:
        registered, footprint = astroalign.apply_transform(
            transform, reference_filled, science_filled, fill_value=0.0
        )
        align_method = "astroalign"
        align_dx = float(transform.translation[0])
        align_dy = float(transform.translation[1])
        align_rot = float(np.rad2deg(transform.rotation))
        align_scale = float(transform.scale)
    else:
        # Fallback: iKon-era stacks fill only the central ~900 px of the
        # grid and astroalign's triangle matching is unreliable there.
        # The APASS-refit WCS tracks the stack's image content to ~0.4 px,
        # so reprojecting the original PS1 reference (accurate HiPS WCS)
        # onto it registers the template without pattern matching.
        if fitted_wcs is None:
            raise RuntimeError("No astroalign transform and no WCS refit")
        from astropy.wcs import WCS as WCS_class
        from reproject import reproject_interp

        with fits.open(REF_DIR / f"ref.PS1.{band}.fits", memmap=False) as hdul:
            registered, coverage = reproject_interp(
                hdul[0], fitted_wcs, shape_out=science.shape
            )
        registered = np.nan_to_num(
            np.asarray(registered, dtype=np.float32), nan=0.0
        )
        footprint = coverage < 0.5
        align_method = "wcs-reproject"
        align_dx = align_dy = align_rot = 0.0
        align_scale = 1.0
    invalid = ~np.isfinite(science) | footprint | ~np.isfinite(reference)

    with tempfile.TemporaryDirectory(prefix="sn2026kid-dia-", dir="/tmp") as temp_name:
        temp_dir = Path(temp_name)
        science_path = temp_dir / "science.fits"
        template_path = temp_dir / "template.fits"
        difference_path = temp_dir / "difference.fits"
        fits.writeto(science_path, np.where(invalid, 0.0, science_filled), header)
        fits.writeto(template_path, np.where(invalid, 0.0, registered), header)
        helper.run_hotpants(
            target_path=science_path,
            reference_path=template_path,
            target_outpath=difference_path,
            convim="t",
            normim="i",
            iu=60000,
            il=-1000,
            tu=60000,
            tl=-1000,
            # Absorb the large-scale color-mismatch and HiPS-halo residuals:
            # quadratic differential background, more spatial kernel freedom.
            bgo=2,
            nrx=4,
            nry=4,
            verbose=False,
        )
        difference = fits.getdata(difference_path, memmap=False).astype(np.float32)

    difference[invalid] = np.nan
    if fitted_wcs is not None:
        sn_x, sn_y = fitted_wcs.world_to_pixel(SkyCoord(SN_RA, SN_DEC, unit="deg"))
    else:
        sn_x, sn_y = WCS(header).world_to_pixel(SkyCoord(SN_RA, SN_DEC, unit="deg"))

    output_header = header.copy()
    output_header["DIATMPL"] = (f"ref.PS1.{band}.aligned.fits", "DIA template")
    output_header["DIAMETH"] = (align_method, "Template registration method")
    output_header["DIAALGN"] = (
        f"dx={align_dx:+.2f} dy={align_dy:+.2f} rot={align_rot:+.4f}deg",
        "Template registration",
    )
    output_header["DIACONV"] = ("template", "HOTPANTS convolution direction")
    output_header["WCSRFITN"] = (refit_stars, "APASS stars in TAN WCS refit")
    output_header["WCSRFITR"] = (
        refit_residual if np.isfinite(refit_residual) else -1.0,
        "Refit median residual [pix]",
    )
    output_header["SNXPIX"] = (float(sn_x), "SN x (0-based, refit WCS)")
    output_header["SNYPIX"] = (float(sn_y), "SN y (0-based, refit WCS)")
    output_header["DIAUTC"] = (
        datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "UTC subtraction time",
    )
    output_path = stack_path.parent / f"sub_{stack_path.name}"
    fits.writeto(output_path, difference, output_header, overwrite=True, checksum=True)

    # Subtraction quality: sigma-clipped residual RMS in a 12-40 px annulus
    # around the SN (the region that matters for Stage C photometry) versus
    # the stack RMS there. Star and edge residuals elsewhere are irrelevant.
    from astropy.stats import sigma_clipped_stats

    stack_rms = fits.getdata(Path(str(stack_path) + ".bkgrms"), memmap=False)
    yy, xx = np.ogrid[: difference.shape[0], : difference.shape[1]]
    radius = np.hypot(xx - sn_x, yy - sn_y)
    annulus = (
        (radius > 12)
        & (radius < 40)
        & np.isfinite(difference)
        & np.isfinite(stack_rms)
    )
    _, _, local_std = sigma_clipped_stats(difference[annulus], sigma=3.0)
    ratio = float(local_std / np.nanmedian(stack_rms[annulus]))
    return {
        "night": night,
        "filter": filter_name,
        "band": band,
        "stack": str(stack_path),
        "difference": str(output_path),
        "align_method": align_method,
        "align_dx": align_dx,
        "align_dy": align_dy,
        "align_rot_deg": align_rot,
        "align_scale": align_scale,
        "refit_stars": refit_stars,
        "refit_residual_pix": refit_residual,
        "sn_x": float(sn_x),
        "sn_y": float(sn_y),
        "residual_over_stackrms": ratio,
        "science_data": science,
        "template_data": registered,
        "difference_data": difference,
    }


def discover_stacks(nights: list[str] | None) -> list[Path]:
    stacks = []
    for root in ROOTS:
        for filter_name in FILTER_TO_BAND:
            stacks.extend(sorted((root / filter_name).glob("stack.SN2026kid.*.fits")))
    stacks = [p for p in stacks if p.suffix == ".fits" and not p.name.startswith("sub_")]
    if nights:
        stacks = [
            p for p in stacks if p.name.split(".")[2] in nights
        ]
    return stacks


def qc_figure(results: list[dict], output_path: Path) -> None:
    import matplotlib.pyplot as plt
    from astropy.visualization import ZScaleInterval

    shown = results[:8]
    fig, axes = plt.subplots(
        len(shown), 3, figsize=(12, 3.6 * len(shown)), constrained_layout=True
    )
    axes = np.atleast_2d(axes)
    half = 75
    for row_index, result in enumerate(shown):
        x, y = int(round(result["sn_x"])), int(round(result["sn_y"]))
        panels = [
            (result["science_data"], "science"),
            (result["template_data"], "template (registered)"),
            (result["difference_data"], "difference"),
        ]
        for column, (data, label) in enumerate(panels):
            cut = data[y - half : y + half, x - half : x + half]
            finite = cut[np.isfinite(cut)]
            vmin, vmax = (
                ZScaleInterval().get_limits(finite) if finite.size else (0, 1)
            )
            axis = axes[row_index, column]
            axis.imshow(cut, origin="lower", cmap="gray", vmin=vmin, vmax=vmax,
                        interpolation="nearest")
            axis.plot(half, half, marker="+", ms=16, mew=1.2, color="red")
            if column == 0:
                axis.set_ylabel(
                    f"{result['night']} {result['filter']}", fontsize=9
                )
            axis.set_title(
                label if row_index == 0 else "", fontsize=9
            )
            axis.set_xticks([])
            axis.set_yticks([])
    fig.suptitle(
        "DIA Stage B — science / registered PS1 template / HOTPANTS difference "
        "(SN position marked, 120\" cutouts)"
    )
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nights", nargs="+", help="Restrict to these nights")
    parser.add_argument("--qc-name", default="12_subtract")
    parser.add_argument(
        "--stacks-from",
        type=Path,
        help="File listing stack paths to (re)process, one per line",
    )
    args = parser.parse_args()

    if args.stacks_from:
        stacks = [
            Path(line)
            for line in args.stacks_from.read_text().splitlines()
            if line.strip()
        ]
    else:
        stacks = discover_stacks(args.nights)
    if not stacks:
        raise RuntimeError("No stacks selected")
    helper = Helper()
    apass = load_apass()
    results, failures = [], []
    for index, stack_path in enumerate(stacks, start=1):
        try:
            result = subtract_one(stack_path, helper, apass)
            results.append(result)
            print(
                f"[{index:03d}/{len(stacks)}] {stack_path.name}: "
                f"shift=({result['align_dx']:+.2f},{result['align_dy']:+.2f})px "
                f"refit={result['refit_residual_pix']:.2f}px "
                f"resid/rms={result['residual_over_stackrms']:.2f}",
                flush=True,
            )
        except Exception as error:
            failures.append(
                {"stack": str(stack_path), "error": f"{type(error).__name__}: {error}"}
            )
            print(f"[{index:03d}/{len(stacks)}] {stack_path.name}: FAIL {error}",
                  flush=True)

    QC_DIR.mkdir(parents=True, exist_ok=True)
    qc_figure(results, QC_DIR / f"{args.qc_name}_qc.png")

    ratios = np.array([r["residual_over_stackrms"] for r in results])
    shifts = np.array([[r["align_dx"], r["align_dy"]] for r in results])
    verdict = "PASS" if not failures and ratios.size and np.median(ratios) < 2.0 else "FAIL"
    lines = [
        "DIA Stage B (HOTPANTS subtraction) QC summary — created "
        + datetime.now(timezone.utc).isoformat(timespec="seconds"),
        f"stacks processed      : {len(results)}  (failures {len(failures)})",
        "template              : PS1 g/r/i aligned + astroalign per stack",
        f"astroalign shift [px] : median dx {np.median(shifts[:,0]):+.2f} "
        f"dy {np.median(shifts[:,1]):+.2f}, max |shift| "
        f"{np.abs(shifts).max():.2f}",
        f"WCS refit residual    : median "
        f"{np.median([r['refit_residual_pix'] for r in results]):.2f} px",
        f"local resid/stack RMS : median {np.median(ratios):.2f} "
        "(sigma-clipped, 12-40 px annulus at SN; 1 = clean at sky level)",
        "note                  : B102 uses a g-band template; its disk "
        "residual is color-mismatch limited — treat DIA-B as secondary",
        "",
        *[
            f"  {r['night']} {r['filter']}: resid/rms={r['residual_over_stackrms']:.2f} "
            f"shift=({r['align_dx']:+.2f},{r['align_dy']:+.2f})px "
            f"refit={r['refit_residual_pix']:.2f}px stars={r['refit_stars']}"
            for r in results
        ],
        *[f"  FAIL {f['stack']}: {f['error']}" for f in failures],
        "",
        f"VALIDATION_RESULT={verdict}",
    ]
    (QC_DIR / f"{args.qc_name}_summary.log").write_text("\n".join(lines) + "\n")

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "results": [
            {k: v for k, v in r.items() if not k.endswith("_data")} for r in results
        ],
        "failures": failures,
    }
    suffix = "_".join(args.nights) if args.nights else "all"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"stage_dia_subtract_{suffix}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(f"QC: {QC_DIR / f'{args.qc_name}_qc.png'}")
    print(f"VALIDATION_RESULT={verdict}")


if __name__ == "__main__":
    main()
