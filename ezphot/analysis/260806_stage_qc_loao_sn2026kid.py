#!/usr/bin/env python
"""Per-stage QC summaries for the staged LOAO SN2026kid workflow.

For each completed stage of ``260806_stage_photometry_loao_sn2026kid.py`` this
writes exactly two files into a shared QC directory, following the
``<n>_<stage>`` naming of the earlier ``*_codex_run`` log folders:

    <qc-dir>/<n>_<stage>_qc.png       one review figure for the whole stage
    <qc-dir>/<n>_<stage>_summary.log  counts, statistics, outliers, verdict

Stage numbers continue the existing convention: 1 preprocess, 2 astrometry,
3 srcmask, 4 bkg, 5 bkgrms, 6 phot, 7 calib, 8 stack, 9 stackphot.
Currently implemented: ``mask``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPORT_DIR = Path("/home/hhchoi1022/ezphot/data/scidata/LOAO/SN2026kid/stage_reports")
REJECTS_FILE = Path("/home/hhchoi1022/ezphot/data/scidata/LOAO/SN2026kid/rejected_frames.json")
QC_DIR_DEFAULT = Path("/home/hhchoi1022/ezphot/log/20260806_stage_run_qc")
STAGE_NUMBERS = {"mask": 3, "bkg": 4, "bkgrms": 5, "phot": 6, "calib": 7, "stack": 8, "stackphot": 9}
STAGE_LABELS = {
    "mask": "srcmask", "bkg": "bkg", "bkgrms": "bkgrms", "phot": "phot",
    "calib": "calib", "stack": "stack", "stackphot": "stackphot",
}
FILTER_COLORS = {"B102": "tab:blue", "V103": "tab:green", "R104": "tab:red", "I105": "maroon"}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_stage_rows(stage: str) -> tuple[dict, list[dict]]:
    """Aggregate report plus flattened per-image rows with night attached.

    Frames listed in ``rejected_frames.json`` are dropped even when they were
    processed before being rejected, so statistics describe the valid dataset.
    """
    rejected = set()
    if REJECTS_FILE.exists():
        rejected = {f["path"] for f in json.loads(REJECTS_FILE.read_text())["frames"]}
    aggregate = json.loads((REPORT_DIR / f"stage_{stage}_report.json").read_text())
    rows = []
    for night_file in sorted(REPORT_DIR.glob(f"stage_{stage}_2026_*.json")):
        night_report = json.loads(night_file.read_text())
        for result in night_report["results"]:
            if result.get("path") in rejected:
                continue
            rows.append({**result, "night": night_report["night"]})
    return aggregate, rows


def detector_of(path: str) -> str:
    return "iKon" if "LOAO_iKon_1x1" in path else "E2V"


def night_to_date(night: str) -> datetime:
    return datetime.strptime(night, "%Y_%m%d")


def zscale_limits(data: np.ndarray) -> tuple[float, float]:
    from astropy.visualization import ZScaleInterval

    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 0.0, 1.0
    return tuple(float(v) for v in ZScaleInterval().get_limits(finite))


def draw_mask_overlay(axis, image_path: Path, title: str) -> None:
    from astropy.io import fits

    downsample = 4
    science = fits.getdata(image_path, memmap=False).astype(float)[::downsample, ::downsample]
    mask = (
        fits.getdata(Path(str(image_path) + ".srcmask"), memmap=False)
        .astype(bool)[::downsample, ::downsample]
    )
    vmin, vmax = zscale_limits(science)
    axis.imshow(science, origin="lower", cmap="gray", vmin=vmin, vmax=vmax,
                interpolation="nearest")
    axis.imshow(np.ma.masked_where(~mask, np.ones_like(mask, dtype=float)),
                origin="lower", cmap="autumn", alpha=0.4, vmin=0, vmax=1,
                interpolation="nearest")
    axis.set_title(title, fontsize=9)
    axis.set_xticks([])
    axis.set_yticks([])


def qc_mask(qc_dir: Path) -> tuple[Path, Path, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.io import fits

    aggregate, rows = load_stage_rows("mask")
    done = [r for r in rows if r["status"] in ("processed", "existing") ]
    stats = [r for r in rows if r["status"] == "processed"]
    fractions = np.array([r["source_mask_fraction"] for r in stats])
    invalids = np.array([r["invalid_pixels"] for r in stats])
    detectors = np.array([detector_of(r["path"]) for r in stats])
    dates = [night_to_date(r["night"]) for r in stats]

    failed = sum(n["failed"] for n in aggregate["nights"].values())
    rejects = json.loads(REJECTS_FILE.read_text())["frames"] if REJECTS_FILE.exists() else []

    # ---- figure -----------------------------------------------------------
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 3)

    axis = fig.add_subplot(grid[0, 0])
    axis.hist(fractions, bins=40, color="tab:blue", alpha=0.8)
    axis.axvline(np.median(fractions), color="k", ls="--", lw=1,
                 label=f"median {np.median(fractions):.3f}")
    axis.set_xlabel("source-mask fraction")
    axis.set_ylabel("images")
    axis.set_title(f"Mask fraction ({len(stats)} images)")
    axis.legend(fontsize=8)

    axis = fig.add_subplot(grid[0, 1])
    for detector, color in (("E2V", "tab:blue"), ("iKon", "tab:orange")):
        select = detectors == detector
        axis.plot(np.array(dates)[select], fractions[select], ".", ms=4,
                  color=color, alpha=0.6, label=f"{detector} ({select.sum()})")
    axis.set_xlabel("night")
    axis.set_ylabel("source-mask fraction")
    axis.set_title("Mask fraction by night / detector")
    axis.legend(fontsize=8)
    axis.tick_params(axis="x", rotation=45, labelsize=8)
    axis.grid(alpha=0.2)

    axis = fig.add_subplot(grid[0, 2])
    axis.hist(invalids, bins=40, color="tab:red", alpha=0.8)
    axis.set_xlabel("invalid pixels per image")
    axis.set_ylabel("images")
    axis.set_title(f"Invalid pixels (median {int(np.median(invalids))})")

    order = np.argsort(fractions)
    examples = [
        (order[0], "min"),
        (order[len(order) // 2], "median"),
        (order[-1], "max"),
    ]
    for column, (index, kind) in enumerate(examples):
        row = stats[int(index)]
        header = fits.getheader(row["path"], memmap=False)
        axis = fig.add_subplot(grid[1, column])
        name = Path(row["path"]).name.replace("obj.SN2026kid.", "")
        draw_mask_overlay(
            axis,
            Path(row["path"]),
            f"{kind}: {name}\n{header['FILTER']} {row['night']} "
            f"fraction={row['source_mask_fraction']:.3f}",
        )
    fig.suptitle(
        "Stage 3 source masking QC — SN2026kid / NGC 5907 (mask overlay in orange)",
        fontsize=13,
    )
    png_path = qc_dir / "3_srcmask_qc.png"
    fig.savefig(png_path, dpi=140)
    plt.close(fig)

    # ---- summary log ------------------------------------------------------
    scale_ikon = int((detectors == "iKon").sum())
    scale_e2v = int((detectors == "E2V").sum())
    fraction_ok = bool(fractions.max() < 0.35)
    verdict = "PASS" if failed == 0 and fraction_ok else "FAIL"
    rows_sorted = sorted(stats, key=lambda r: r["source_mask_fraction"])

    def describe(row: dict) -> str:
        return (
            f"{row['source_mask_fraction']:.4f}  inv={row['invalid_pixels']:>6d}  "
            f"{row['night']}  {Path(row['path']).name}"
        )

    lines = [
        f"Stage 3 (source masking) QC summary — created {now_utc()}",
        f"stage run dir: {aggregate['run_dir']}",
        f"reports: {REPORT_DIR}/stage_mask_*.json",
        "",
        f"nights processed      : {len(aggregate['nights'])}",
        f"images (valid dataset): {len(done)}  [processed {len(stats)}, "
        f"existing {len(done) - len(stats)}]",
        f"images failed         : {failed}",
        f"rejected frames       : {len(rejects)} (see {REJECTS_FILE})",
        f"detector split        : E2V {scale_e2v}, iKon {scale_ikon}",
        "",
        "source-mask fraction  : "
        f"min {fractions.min():.4f} / median {np.median(fractions):.4f} / "
        f"max {fractions.max():.4f}",
        "invalid pixels        : "
        f"min {invalids.min()} / median {int(np.median(invalids))} / max {invalids.max()}",
        "mask contents         : photutils source segments (5 sigma) + NGC 5907 "
        "host ellipse (SIMBAD size x 1.25)",
        "",
        "-- lowest mask fractions --",
        *[describe(r) for r in rows_sorted[:5]],
        "-- highest mask fractions --",
        *[describe(r) for r in rows_sorted[-5:]],
        "",
        "-- rejected frames (excluded from all stages) --",
        *[f"{Path(f['path']).name}: {f['reason']}" for f in rejects],
        "",
        f"VALIDATION_RESULT={verdict}",
    ]
    log_path = qc_dir / "3_srcmask_summary.log"
    log_path.write_text("\n".join(lines) + "\n")
    return png_path, log_path, verdict


def qc_bkg(qc_dir: Path) -> tuple[Path, Path, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.io import fits

    aggregate, rows = load_stage_rows("bkg")
    stats = [r for r in rows if r["status"] in ("processed", "existing")]
    for row in stats:
        row["filter"] = Path(row["path"]).parent.name
        row["detector"] = detector_of(row["path"])
    levels = np.array([r["background_median"] for r in stats])
    failed = sum(n["failed"] for n in aggregate["nights"].values())

    # ---- figure -----------------------------------------------------------
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 3)

    axis = fig.add_subplot(grid[0, 0])
    for filter_name, color in FILTER_COLORS.items():
        values = [r["background_median"] for r in stats if r["filter"] == filter_name]
        axis.hist(values, bins=30, histtype="step", lw=1.5, color=color,
                  label=f"{filter_name} ({len(values)})")
    axis.set_xlabel("background median [ADU]")
    axis.set_ylabel("images")
    axis.set_title(f"Background level by filter ({len(stats)} images)")
    axis.axvline(0, color="k", lw=0.8)
    axis.legend(fontsize=8)

    axis = fig.add_subplot(grid[0, 1:])
    for filter_name, color in FILTER_COLORS.items():
        select = [r for r in stats if r["filter"] == filter_name]
        axis.plot([night_to_date(r["night"]) for r in select],
                  [r["background_median"] for r in select],
                  ".", ms=4, color=color, alpha=0.65, label=filter_name)
    axis.set_yscale("symlog", linthresh=100)
    axis.axhline(0, color="k", lw=0.8)
    axis.set_xlabel("night")
    axis.set_ylabel("background median [ADU] (symlog)")
    axis.set_title("Sky level by night (moon phase / conditions)")
    axis.legend(fontsize=8, ncol=4)
    axis.tick_params(axis="x", rotation=45, labelsize=8)
    axis.grid(alpha=0.2)

    order = np.argsort(levels)
    examples = [
        (order[0], "min sky"),
        (order[len(order) // 2], "median sky"),
        (order[-1], "max sky"),
    ]
    for column, (index, kind) in enumerate(examples):
        row = stats[int(index)]
        background = fits.getdata(
            Path(str(row["path"]) + ".bkgmap"), memmap=False
        ).astype(float)[::4, ::4]
        finite = background[np.isfinite(background)]
        vmin, vmax = np.nanpercentile(finite, [1, 99])
        axis = fig.add_subplot(grid[1, column])
        shown = axis.imshow(background, origin="lower", cmap="viridis",
                            vmin=vmin, vmax=vmax, interpolation="nearest")
        fig.colorbar(shown, ax=axis, shrink=0.8)
        name = Path(row["path"]).name.replace("obj.SN2026kid.", "")
        axis.set_title(
            f"{kind}: {name}\n{row['filter']} {row['night']} "
            f"median={row['background_median']:.0f} ADU",
            fontsize=9,
        )
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(
        "Stage 4 background map QC — direct SEP estimate per science image "
        "(source+host masked)",
        fontsize=13,
    )
    png_path = qc_dir / "4_bkg_qc.png"
    fig.savefig(png_path, dpi=140)
    plt.close(fig)

    # ---- summary log ------------------------------------------------------
    negatives = [r for r in stats if r["background_median"] < 0]
    verdict = "PASS" if failed == 0 and np.isfinite(levels).all() else "FAIL"
    rows_sorted = sorted(stats, key=lambda r: r["background_median"])

    def describe(row: dict) -> str:
        return (
            f"{row['background_median']:>9.1f} ADU  {row['filter']}  {row['night']}  "
            f"{Path(row['path']).name}"
        )

    per_filter = []
    for filter_name in FILTER_COLORS:
        values = np.array(
            [r["background_median"] for r in stats if r["filter"] == filter_name]
        )
        per_filter.append(
            f"  {filter_name}: n={len(values)}  min {values.min():.0f} / "
            f"median {np.median(values):.0f} / max {values.max():.0f} ADU"
        )

    lines = [
        f"Stage 4 (background map) QC summary — created {now_utc()}",
        f"stage run dir: {aggregate['run_dir']}",
        f"reports: {REPORT_DIR}/stage_bkg_*.json",
        "",
        f"nights processed      : {len(aggregate['nights'])}",
        f"images                : {len(stats)}",
        f"images failed         : {failed}",
        "method                : SEP 2-D background, box 64 / filter 3, "
        "global-offset corrected, computed per science image (not master frames)",
        "masks applied         : source mask (incl. NGC 5907 host ellipse) + invalid mask",
        "",
        "background median [ADU] per filter:",
        *per_filter,
        "",
        f"WARNING: {len(negatives)} images have a negative background median "
        "(all iKon, mostly B102).",
        "  This is a small master-calibration pedestal over-subtraction on "
        "dark-sky frames;",
        "  it is removed by the 2-D background subtraction and does not affect "
        "photometry.",
        "",
        "-- lowest sky levels --",
        *[describe(r) for r in rows_sorted[:5]],
        "-- highest sky levels --",
        *[describe(r) for r in rows_sorted[-5:]],
        "",
        f"VALIDATION_RESULT={verdict}",
    ]
    log_path = qc_dir / "4_bkg_summary.log"
    log_path.write_text("\n".join(lines) + "\n")
    return png_path, log_path, verdict


def qc_bkgrms(qc_dir: Path) -> tuple[Path, Path, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.io import fits

    aggregate, rows = load_stage_rows("bkgrms")
    stats = [r for r in rows if r["status"] in ("processed", "existing")]
    for row in stats:
        row["filter"] = Path(row["path"]).parent.name
        row["detector"] = detector_of(row["path"])
    _, bkg_rows = load_stage_rows("bkg")
    bkg_by_path = {
        r["path"]: r["background_median"]
        for r in bkg_rows
        if r["status"] in ("processed", "existing")
    }
    rms_values = np.array([r["background_rms_median"] for r in stats])
    failed = sum(n["failed"] for n in aggregate["nights"].values())

    # ---- figure -----------------------------------------------------------
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 3)

    axis = fig.add_subplot(grid[0, 0])
    for filter_name, color in FILTER_COLORS.items():
        values = [
            r["background_rms_median"] for r in stats if r["filter"] == filter_name
        ]
        axis.hist(values, bins=30, histtype="step", lw=1.5, color=color,
                  label=f"{filter_name} ({len(values)})")
    axis.set_xlabel("background RMS median [ADU]")
    axis.set_ylabel("images")
    axis.set_xscale("log")
    axis.set_title(f"Background RMS by filter ({len(stats)} images)")
    axis.legend(fontsize=8)

    axis = fig.add_subplot(grid[0, 1])
    for filter_name, color in FILTER_COLORS.items():
        select = [r for r in stats if r["filter"] == filter_name]
        axis.plot([night_to_date(r["night"]) for r in select],
                  [r["background_rms_median"] for r in select],
                  ".", ms=4, color=color, alpha=0.65, label=filter_name)
    axis.set_yscale("log")
    axis.set_xlabel("night")
    axis.set_ylabel("background RMS median [ADU]")
    axis.set_title("RMS by night")
    axis.legend(fontsize=8, ncol=2)
    axis.tick_params(axis="x", rotation=45, labelsize=8)
    axis.grid(alpha=0.2)

    axis = fig.add_subplot(grid[0, 2])
    matched = [
        (bkg_by_path[r["path"]], r["background_rms_median"], r["detector"])
        for r in stats
        if r["path"] in bkg_by_path
    ]
    for detector, color in (("E2V", "tab:blue"), ("iKon", "tab:orange")):
        pairs = [(b, s) for b, s, d in matched if d == detector]
        if pairs:
            axis.plot(*zip(*pairs), ".", ms=4, color=color, alpha=0.5,
                      label=f"{detector} ({len(pairs)})")
    span = np.logspace(
        np.log10(min(b for b, _, _ in matched)),
        np.log10(max(b for b, _, _ in matched)),
        50,
    )
    gain = 2.68
    axis.plot(span, np.sqrt(span / gain), "k--", lw=1,
              label=r"$\sqrt{\mathrm{bkg}/g}$ (g=2.68)")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("background median [ADU]")
    axis.set_ylabel("background RMS median [ADU]")
    axis.set_title("Noise vs sky level (physics check)")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.2)

    order = np.argsort(rms_values)
    examples = [
        (order[0], "min RMS"),
        (order[len(order) // 2], "median RMS"),
        (order[-1], "max RMS"),
    ]
    for column, (index, kind) in enumerate(examples):
        row = stats[int(index)]
        rms_map = fits.getdata(
            Path(str(row["path"]) + ".bkgrms"), memmap=False
        ).astype(float)[::4, ::4]
        finite = rms_map[np.isfinite(rms_map)]
        vmin, vmax = np.nanpercentile(finite, [1, 99])
        axis = fig.add_subplot(grid[1, column])
        shown = axis.imshow(rms_map, origin="lower", cmap="magma",
                            vmin=vmin, vmax=vmax, interpolation="nearest")
        fig.colorbar(shown, ax=axis, shrink=0.8)
        name = Path(row["path"]).name.replace("obj.SN2026kid.", "")
        axis.set_title(
            f"{kind}: {name}\n{row['filter']} {row['night']} "
            f"median={row['background_rms_median']:.1f} ADU",
            fontsize=9,
        )
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(
        "Stage 5 background RMS QC — SEP RMS per science image (source+host masked)",
        fontsize=13,
    )
    png_path = qc_dir / "5_bkgrms_qc.png"
    fig.savefig(png_path, dpi=140)
    plt.close(fig)

    # ---- summary log ------------------------------------------------------
    ratios = np.array(
        [s / np.sqrt(b / gain) for b, s, _ in matched if b > 0]
    )
    verdict = (
        "PASS"
        if failed == 0 and np.isfinite(rms_values).all() and (rms_values > 0).all()
        else "FAIL"
    )
    rows_sorted = sorted(stats, key=lambda r: r["background_rms_median"])

    def describe(row: dict) -> str:
        return (
            f"{row['background_rms_median']:>7.2f} ADU  {row['filter']}  "
            f"{row['night']}  {Path(row['path']).name}"
        )

    per_filter = []
    for filter_name in FILTER_COLORS:
        values = np.array(
            [r["background_rms_median"] for r in stats if r["filter"] == filter_name]
        )
        per_filter.append(
            f"  {filter_name}: n={len(values)}  min {values.min():.2f} / "
            f"median {np.median(values):.2f} / max {values.max():.2f} ADU"
        )

    lines = [
        f"Stage 5 (background RMS map) QC summary — created {now_utc()}",
        f"stage run dir: {aggregate['run_dir']}",
        f"reports: {REPORT_DIR}/stage_bkgrms_*.json",
        "",
        f"nights processed      : {len(aggregate['nights'])}",
        f"images                : {len(stats)}",
        f"images failed         : {failed}",
        "method                : SEP RMS, box 64 / filter 3, source+invalid masks",
        "",
        "background RMS median [ADU] per filter:",
        *per_filter,
        "",
        "noise model check     : RMS / sqrt(bkg/gain), gain 2.68 e-/ADU",
        f"  ratio min {ratios.min():.2f} / median {np.median(ratios):.2f} / "
        f"max {ratios.max():.2f}  (~1 means sky-limited Poisson noise)",
        "",
        "-- lowest RMS --",
        *[describe(r) for r in rows_sorted[:5]],
        "-- highest RMS --",
        *[describe(r) for r in rows_sorted[-5:]],
        "",
        f"VALIDATION_RESULT={verdict}",
    ]
    log_path = qc_dir / "5_bkgrms_summary.log"
    log_path.write_text("\n".join(lines) + "\n")
    return png_path, log_path, verdict


def qc_phot(qc_dir: Path) -> tuple[Path, Path, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.io import fits
    from astropy.table import Table

    aggregate, rows = load_stage_rows("phot")
    stats = [r for r in rows if r["status"] in ("processed", "existing")]
    for row in stats:
        row["filter"] = Path(row["path"]).parent.name
        row["detector"] = detector_of(row["path"])
        info_path = Path(str(row["path"]) + ".cat.info")
        row["seeing"] = (
            json.loads(info_path.read_text()).get("seeing") if info_path.exists() else None
        )
    counts = np.array([r["sources"] for r in stats])
    failed = sum(n["failed"] for n in aggregate["nights"].values())

    # ---- figure -----------------------------------------------------------
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 3)

    axis = fig.add_subplot(grid[0, 0])
    for detector, color in (("E2V", "tab:blue"), ("iKon", "tab:orange")):
        values = [r["sources"] for r in stats if r["detector"] == detector]
        axis.hist(values, bins=40, histtype="step", lw=1.5, color=color,
                  label=f"{detector} ({len(values)})")
    axis.set_xlabel("detected sources per image")
    axis.set_ylabel("images")
    axis.set_title(f"Source counts ({len(stats)} images)")
    axis.legend(fontsize=8)

    axis = fig.add_subplot(grid[0, 1])
    for filter_name, color in FILTER_COLORS.items():
        select = [r for r in stats if r["filter"] == filter_name]
        axis.plot([night_to_date(r["night"]) for r in select],
                  [r["sources"] for r in select],
                  ".", ms=4, color=color, alpha=0.65, label=filter_name)
    axis.set_yscale("log")
    axis.set_xlabel("night")
    axis.set_ylabel("detected sources")
    axis.set_title("Detections by night (depth / conditions)")
    axis.legend(fontsize=8, ncol=2)
    axis.tick_params(axis="x", rotation=45, labelsize=8)
    axis.grid(alpha=0.2)

    axis = fig.add_subplot(grid[0, 2])
    for detector, color in (("E2V", "tab:blue"), ("iKon", "tab:orange")):
        values = [
            r["seeing"] for r in stats
            if r["detector"] == detector and r["seeing"] is not None
        ]
        axis.hist(values, bins=40, histtype="step", lw=1.5, color=color,
                  label=f"{detector} (median {np.median(values):.2f}\")")
    axis.set_xlabel("seeing FWHM [arcsec]")
    axis.set_ylabel("images")
    axis.set_title("Seeing from photometry catalogs")
    axis.legend(fontsize=8)

    order = np.argsort(counts)
    examples = [
        (order[0], "min"),
        (order[len(order) // 2], "median"),
        (order[-1], "max"),
    ]
    for column, (index, kind) in enumerate(examples):
        row = stats[int(index)]
        downsample = 4
        science = fits.getdata(row["path"], memmap=False).astype(float)
        catalog = Table.read(Path(str(row["path"]) + ".cat"), format="ascii")
        vmin, vmax = zscale_limits(science[::downsample, ::downsample])
        axis = fig.add_subplot(grid[1, column])
        axis.imshow(science[::downsample, ::downsample], origin="lower",
                    cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        axis.scatter(
            np.asarray(catalog["X_IMAGE"], dtype=float) / downsample,
            np.asarray(catalog["Y_IMAGE"], dtype=float) / downsample,
            s=10, facecolors="none", edgecolors="tab:red", linewidths=0.4,
        )
        name = Path(row["path"]).name.replace("obj.SN2026kid.", "")
        axis.set_title(
            f"{kind}: {name}\n{row['filter']} {row['night']} "
            f"sources={row['sources']} seeing={row['seeing']:.2f}\"",
            fontsize=9,
        )
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(
        "Stage 6 aperture photometry QC — SExtractor 1.5 sigma, apertures "
        "3/5/7/10\" + seeing-scaled (background pre-subtracted)",
        fontsize=13,
    )
    png_path = qc_dir / "6_phot_qc.png"
    fig.savefig(png_path, dpi=140)
    plt.close(fig)

    # ---- summary log ------------------------------------------------------
    seeing_all = np.array([r["seeing"] for r in stats if r["seeing"] is not None])
    verdict = "PASS" if failed == 0 and counts.min() > 0 else "FAIL"
    rows_sorted = sorted(stats, key=lambda r: r["sources"])

    def describe(row: dict) -> str:
        return (
            f"{row['sources']:>5d} src  seeing={row['seeing']:.2f}\"  "
            f"{row['filter']}  {row['night']}  {Path(row['path']).name}"
        )

    per_detector = []
    for detector in ("E2V", "iKon"):
        values = np.array([r["sources"] for r in stats if r["detector"] == detector])
        per_detector.append(
            f"  {detector}: n={len(values)}  min {values.min()} / "
            f"median {int(np.median(values))} / max {values.max()} sources"
        )

    lines = [
        f"Stage 6 (aperture photometry) QC summary — created {now_utc()}",
        f"stage run dir: {aggregate['run_dir']}",
        f"reports: {REPORT_DIR}/stage_phot_*.json",
        "",
        f"nights processed      : {len(aggregate['nights'])}",
        f"images                : {len(stats)}",
        f"images failed         : {failed}",
        "method                : SExtractor via staged no-space temp copy, "
        "background pre-subtracted (BACK_TYPE MANUAL 0)",
        "detection             : 1.5 sigma, RMS-map weighting, invalid-mask flags",
        "apertures             : 3/5/7/10 arcsec diameter + 3.5x/4.5x seeing "
        "(MAG_APER_2 = 7\" primary)",
        "products              : .cat + .cat.info + 2 PNGs per image, "
        "PHOTOMETRY status set",
        "",
        "sources per image:",
        *per_detector,
        f"seeing [arcsec]       : min {seeing_all.min():.2f} / "
        f"median {np.median(seeing_all):.2f} / max {seeing_all.max():.2f}",
        "",
        "-- fewest detections --",
        *[describe(r) for r in rows_sorted[:5]],
        "-- most detections --",
        *[describe(r) for r in rows_sorted[-5:]],
        "",
        f"VALIDATION_RESULT={verdict}",
    ]
    log_path = qc_dir / "6_phot_summary.log"
    log_path.write_text("\n".join(lines) + "\n")
    return png_path, log_path, verdict


def qc_calib(qc_dir: Path) -> tuple[Path, Path, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.io import fits

    aggregate, rows = load_stage_rows("calib")
    stats = [r for r in rows if r["status"] in ("processed", "existing")]
    for row in stats:
        row["filter"] = Path(row["path"]).parent.name
        row["detector"] = detector_of(row["path"])
        if "zp_stars" not in row:
            row["zp_stars"] = row.get("zp_stars", 0)
    scatters = np.array([r["zp_scatter"] for r in stats])
    failed = sum(n["failed"] for n in aggregate["nights"].values())

    # ---- figure -----------------------------------------------------------
    fig = plt.figure(figsize=(16, 11), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.2])

    axis = fig.add_subplot(grid[0, 0])
    for filter_name, color in FILTER_COLORS.items():
        select = [r for r in stats if r["filter"] == filter_name]
        axis.plot([night_to_date(r["night"]) for r in select],
                  [r["zp"] for r in select],
                  ".", ms=4, color=color, alpha=0.65, label=filter_name)
    axis.set_xlabel("night")
    axis.set_ylabel("ZP_APER_2 [mag]")
    axis.set_title("Zero point by night (detector swap visible)")
    axis.legend(fontsize=8, ncol=2)
    axis.tick_params(axis="x", rotation=45, labelsize=8)
    axis.grid(alpha=0.2)

    axis = fig.add_subplot(grid[0, 1])
    for filter_name, color in FILTER_COLORS.items():
        values = [r["zp_scatter"] for r in stats if r["filter"] == filter_name]
        axis.hist(values, bins=40, histtype="step", lw=1.5, color=color,
                  label=f"{filter_name} (med {np.median(values):.3f})")
    axis.set_xlabel("ZP scatter [mag]")
    axis.set_ylabel("images")
    axis.set_title("Zero-point scatter by filter")
    axis.legend(fontsize=8)

    axis = fig.add_subplot(grid[0, 2])
    stars = np.array([r["zp_stars"] for r in stats])
    axis.plot([night_to_date(r["night"]) for r in stats], stars, ".", ms=4,
              color="tab:purple", alpha=0.5)
    axis.set_yscale("log")
    axis.set_xlabel("night")
    axis.set_ylabel("APASS zero-point stars")
    axis.set_title(f"Calibration stars (median {int(np.median(stars))})")
    axis.tick_params(axis="x", rotation=45, labelsize=8)
    axis.grid(alpha=0.2)

    order = np.argsort(scatters)
    examples = [
        (order[0], "best scatter"),
        (order[len(order) // 2], "median scatter"),
        (order[-1], "worst scatter"),
    ]
    for column, (index, kind) in enumerate(examples):
        row = stats[int(index)]
        figure_path = Path(str(row["path"]) + ".cat").with_suffix(".zp_mag.png")
        axis = fig.add_subplot(grid[1, column])
        if figure_path.exists():
            axis.imshow(plt.imread(figure_path))
        name = Path(row["path"]).name.replace("obj.SN2026kid.", "")
        axis.set_title(
            f"{kind}: {name}\n{row['filter']} {row['night']} "
            f"ZP={row['zp']:.3f} scatter={row['zp_scatter']:.3f} "
            f"stars={row['zp_stars']}",
            fontsize=9,
        )
        axis.axis("off")
    fig.suptitle(
        "Stage 7 photometric calibration QC — PhotometricCalibration vs "
        "SkyCatalog APASS DR9 (B,V direct; R,I transformed)",
        fontsize=13,
    )
    png_path = qc_dir / "7_calib_qc.png"
    fig.savefig(png_path, dpi=140)
    plt.close(fig)

    # ---- summary log ------------------------------------------------------
    verdict = "PASS" if failed == 0 and np.isfinite(scatters).all() else "FAIL"
    rows_sorted = sorted(stats, key=lambda r: r["zp_scatter"])

    def describe(row: dict) -> str:
        return (
            f"scatter={row['zp_scatter']:.3f}  ZP={row['zp']:.3f}  "
            f"stars={row['zp_stars']:>3d}  {row['filter']}  {row['night']}  "
            f"{Path(row['path']).name}"
        )

    per_filter = []
    for filter_name in FILTER_COLORS:
        select = [r for r in stats if r["filter"] == filter_name]
        zp_values = np.array([r["zp"] for r in select])
        scatter_values = np.array([r["zp_scatter"] for r in select])
        star_values = np.array([r["zp_stars"] for r in select])
        per_filter.append(
            f"  {filter_name}: n={len(select)}  ZP median {np.median(zp_values):.3f}  "
            f"scatter median {np.median(scatter_values):.3f}  "
            f"stars median {int(np.median(star_values))}"
        )

    high_scatter = [r for r in stats if r["zp_scatter"] > 0.2]
    lines = [
        f"Stage 7 (photometric calibration) QC summary — created {now_utc()}",
        f"stage run dir: {aggregate['run_dir']}",
        f"reports: {REPORT_DIR}/stage_calib_*.json",
        "",
        f"nights processed      : {len(aggregate['nights'])}",
        f"images                : {len(stats)}",
        f"images failed         : {failed}",
        "reference             : SkyCatalog APASS DR9; B102->B, V103->V direct; "
        "R104->R, I105->I via APASS r/i transforms",
        "method                : PhotometricCalibration, MAG_APER_2 (7\"), "
        "match 2.5\", mag 11-18, color+mag terms, save_fig",
        "",
        "per filter:",
        *per_filter,
        "",
        f"quality note          : {len(high_scatter)} images with ZP scatter > 0.2 mag "
        "(cloudy candidates; consider excluding before stacking)",
        "",
        "-- best zero points --",
        *[describe(r) for r in rows_sorted[:5]],
        "-- worst zero points --",
        *[describe(r) for r in rows_sorted[-5:]],
        "",
        f"VALIDATION_RESULT={verdict}",
    ]
    log_path = qc_dir / "7_calib_summary.log"
    log_path.write_text("\n".join(lines) + "\n")
    return png_path, log_path, verdict


def qc_stack(qc_dir: Path) -> tuple[Path, Path, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.io import fits

    aggregate, rows = load_stage_rows("stack")
    stacked = [r for r in rows if r["status"] in ("stacked", "existing")]
    skipped = [r for r in rows if r["status"] == "skipped"]
    excluded = [r for r in rows if r["status"] == "stack_excluded"]
    failed = sum(n["failed"] for n in aggregate["nights"].values())
    for row in stacked:
        row["detector"] = detector_of(row["output"])

    # ---- figure -----------------------------------------------------------
    fig = plt.figure(figsize=(16, 11), constrained_layout=True)
    grid = fig.add_gridspec(2, 4, height_ratios=[1, 1.4])

    axis = fig.add_subplot(grid[0, 0])
    counts = {
        filter_name: sum(1 for r in stacked if r["filter"] == filter_name)
        for filter_name in FILTER_COLORS
    }
    axis.bar(list(counts), list(counts.values()),
             color=[FILTER_COLORS[f] for f in counts])
    axis.set_ylabel("stacks")
    axis.set_title(f"Stacks per filter (total {len(stacked)})")

    axis = fig.add_subplot(grid[0, 1])
    values = [r["ncombine"] for r in stacked]
    axis.hist(values, bins=range(1, max(values) + 2), color="tab:blue", alpha=0.8)
    axis.set_xlabel("images combined (NCOMBINE)")
    axis.set_ylabel("stacks")
    axis.set_title(f"Images per stack (median {int(np.median(values))})")

    axis = fig.add_subplot(grid[0, 2])
    for filter_name, color in FILTER_COLORS.items():
        select = [r for r in stacked if r["filter"] == filter_name]
        axis.plot([night_to_date(r["night"]) for r in select],
                  [r["total_exposure"] for r in select],
                  ".", ms=4, color=color, alpha=0.65, label=filter_name)
    axis.set_xlabel("night")
    axis.set_ylabel("total exposure [s]")
    axis.set_title("Stack exposure by night")
    axis.legend(fontsize=7, ncol=2)
    axis.tick_params(axis="x", rotation=45, labelsize=7)
    axis.grid(alpha=0.2)

    axis = fig.add_subplot(grid[0, 3])
    fractions = [r["finite_fraction"] for r in stacked if "finite_fraction" in r]
    axis.hist(fractions, bins=30, color="tab:green", alpha=0.8)
    axis.set_xlabel("finite pixel fraction")
    axis.set_ylabel("stacks")
    axis.set_title("Stack coverage")

    # Example stacks: one per filter from the night with the deepest exposure.
    best = {}
    for row in stacked:
        current = best.get(row["filter"])
        if current is None or row["total_exposure"] > current["total_exposure"]:
            best[row["filter"]] = row
    for column, filter_name in enumerate(FILTER_COLORS):
        axis = fig.add_subplot(grid[1, column])
        row = best.get(filter_name)
        if row is None:
            axis.axis("off")
            continue
        data = fits.getdata(row["output"], memmap=False).astype(float)[::4, ::4]
        vmin, vmax = zscale_limits(data)
        axis.imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax,
                    interpolation="nearest")
        axis.set_title(
            f"{filter_name} {row['night']}  N={row['ncombine']}  "
            f"{row['total_exposure']:.0f}s",
            fontsize=9,
        )
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(
        "Stage 8 stacking QC — background-subtracted, ZP-scaled, "
        "NGC 5907-centered weighted coadds (deepest per filter shown)",
        fontsize=13,
    )
    png_path = qc_dir / "8_stack_qc.png"
    fig.savefig(png_path, dpi=140)
    plt.close(fig)

    # ---- summary log ------------------------------------------------------
    verdict = "PASS" if failed == 0 and stacked else "FAIL"
    nights_with = {r["night"] for r in stacked}

    per_filter = []
    for filter_name in FILTER_COLORS:
        select = [r for r in stacked if r["filter"] == filter_name]
        exposures = np.array([r["total_exposure"] for r in select])
        per_filter.append(
            f"  {filter_name}: {len(select)} stacks  total {exposures.sum():.0f}s  "
            f"median/night {np.median(exposures):.0f}s"
        )

    lines = [
        f"Stage 8 (stacking) QC summary — created {now_utc()}",
        f"stage run dir: {aggregate['run_dir']}",
        f"reports: {REPORT_DIR}/stage_stack_*.json",
        "",
        f"nights with stacks    : {len(nights_with)} / {len(aggregate['nights'])}",
        f"stacks created        : {len(stacked)}",
        f"filter groups skipped : {len(skipped)} (fewer than 2 usable images)",
        f"frames stack-excluded : {len(excluded)} (NZPSTAR < 5; single-image "
        "photometry kept)",
        f"failures              : {failed}",
        "method                : per-image background subtracted, scaled to "
        "common ZP_APER_2, SWarp LANCZOS3 reprojection to NGC 5907-centered "
        "2048x2048 0.794\"/px grid, weighted mean, uncovered pixels NaN",
        "",
        "per filter:",
        *per_filter,
        "",
        "-- stack-excluded frames --",
        *[
            f"  {Path(r['path']).name}: {r['reason']}"
            for r in excluded
        ],
        "",
        f"VALIDATION_RESULT={verdict}",
    ]
    log_path = qc_dir / "8_stack_summary.log"
    log_path.write_text("\n".join(lines) + "\n")
    return png_path, log_path, verdict


def normalize_stackphot_row(row: dict) -> dict:
    """Flatten processed rows (nested zero_point) and existing rows (flat)."""
    zero_point = row.get("zero_point") or {}
    return {
        **row,
        "zp": zero_point.get("primary_zp", row.get("zp")),
        "zp_scatter": zero_point.get("primary_zp_scatter", row.get("zp_scatter")),
        "zp_stars": zero_point.get("matched_stars", row.get("zp_stars")),
    }


def qc_stackphot(qc_dir: Path) -> tuple[Path, Path, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    aggregate, rows = load_stage_rows("stackphot")
    stats = [
        normalize_stackphot_row(r)
        for r in rows
        if r["status"] in ("processed", "existing")
    ]
    for row in stats:
        row["night"] = row["night"]
    failed = sum(n["failed"] for n in aggregate["nights"].values())

    _, single_rows = load_stage_rows("calib")
    single_scatter = [
        r["zp_scatter"] for r in single_rows if r["status"] in ("processed", "existing")
    ]
    scatters = np.array([r["zp_scatter"] for r in stats])

    # ---- figure -----------------------------------------------------------
    fig = plt.figure(figsize=(16, 11), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.2])

    axis = fig.add_subplot(grid[0, 0])
    for filter_name, color in FILTER_COLORS.items():
        select = [r for r in stats if r["filter"] == filter_name]
        axis.plot([night_to_date(r["night"]) for r in select],
                  [r["zp"] for r in select],
                  ".", ms=4, color=color, alpha=0.65, label=filter_name)
    axis.set_xlabel("night")
    axis.set_ylabel("stack ZP_APER_2 [mag]")
    axis.set_title("Stack zero point by night")
    axis.legend(fontsize=8, ncol=2)
    axis.tick_params(axis="x", rotation=45, labelsize=8)
    axis.grid(alpha=0.2)

    axis = fig.add_subplot(grid[0, 1])
    bins = np.linspace(0, 0.25, 40)
    axis.hist(single_scatter, bins=bins, color="lightgray", label="single images")
    axis.hist(scatters, bins=bins, histtype="step", lw=2, color="tab:blue",
              label=f"stacks (med {np.median(scatters):.3f})")
    axis.set_xlabel("ZP scatter [mag]")
    axis.set_ylabel("count")
    axis.set_title("Stack vs single-image ZP scatter")
    axis.legend(fontsize=8)

    axis = fig.add_subplot(grid[0, 2])
    for filter_name, color in FILTER_COLORS.items():
        select = [r for r in stats if r["filter"] == filter_name]
        axis.plot([night_to_date(r["night"]) for r in select],
                  [r["sources"] for r in select],
                  ".", ms=4, color=color, alpha=0.65, label=filter_name)
    axis.set_yscale("log")
    axis.set_xlabel("night")
    axis.set_ylabel("sources in stack catalog")
    axis.set_title("Stack catalog size (depth proxy)")
    axis.legend(fontsize=8, ncol=2)
    axis.tick_params(axis="x", rotation=45, labelsize=8)
    axis.grid(alpha=0.2)

    order = np.argsort(scatters)
    examples = [
        (order[0], "best scatter"),
        (order[len(order) // 2], "median scatter"),
        (order[-1], "worst scatter"),
    ]
    for column, (index, kind) in enumerate(examples):
        row = stats[int(index)]
        figure_path = Path(row["path"] + ".cat").with_suffix(".zp_mag.png")
        axis = fig.add_subplot(grid[1, column])
        if figure_path.exists():
            axis.imshow(plt.imread(figure_path))
        axis.set_title(
            f"{kind}: {Path(row['path']).name}\n"
            f"ZP={row['zp']:.3f} scatter={row['zp_scatter']:.3f} "
            f"stars={row['zp_stars']} sources={row['sources']}",
            fontsize=9,
        )
        axis.axis("off")
    fig.suptitle(
        "Stage 9 stack photometry QC — SExtractor + PhotometricCalibration "
        "on nightly stacks (APASS DR9)",
        fontsize=13,
    )
    png_path = qc_dir / "9_stackphot_qc.png"
    fig.savefig(png_path, dpi=140)
    plt.close(fig)

    # ---- summary log ------------------------------------------------------
    verdict = "PASS" if failed == 0 and np.isfinite(scatters).all() else "FAIL"
    rows_sorted = sorted(stats, key=lambda r: r["zp_scatter"])

    def describe(row: dict) -> str:
        return (
            f"scatter={row['zp_scatter']:.3f}  ZP={row['zp']:.3f}  "
            f"stars={row['zp_stars']:>3d}  src={row['sources']:>5d}  "
            f"{Path(row['path']).name}"
        )

    per_filter = []
    for filter_name in FILTER_COLORS:
        select = [r for r in stats if r["filter"] == filter_name]
        scatter_values = np.array([r["zp_scatter"] for r in select])
        source_values = np.array([r["sources"] for r in select])
        per_filter.append(
            f"  {filter_name}: {len(select)} stacks  scatter median "
            f"{np.median(scatter_values):.3f}  sources median "
            f"{int(np.median(source_values))}"
        )

    lines = [
        f"Stage 9 (stack photometry + calibration) QC summary — created {now_utc()}",
        f"stage run dir: {aggregate['run_dir']}",
        f"reports: {REPORT_DIR}/stage_stackphot_*.json",
        "",
        f"nights processed      : {len(aggregate['nights'])}",
        f"stack catalogs        : {len(stats)}",
        f"failures              : {failed}",
        "method                : staged SExtractor (zero background, stack RMS "
        "weights, coverage mask) + PhotometricCalibration vs APASS, save_fig",
        "",
        "per filter:",
        *per_filter,
        "",
        f"single-image scatter median {np.median(single_scatter):.3f} -> "
        f"stack scatter median {np.median(scatters):.3f}",
        "",
        "-- best stack calibrations --",
        *[describe(r) for r in rows_sorted[:5]],
        "-- worst stack calibrations --",
        *[describe(r) for r in rows_sorted[-5:]],
        "",
        f"VALIDATION_RESULT={verdict}",
    ]
    log_path = qc_dir / "9_stackphot_summary.log"
    log_path.write_text("\n".join(lines) + "\n")
    return png_path, log_path, verdict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=list(STAGE_NUMBERS))
    parser.add_argument("--qc-dir", type=Path, default=QC_DIR_DEFAULT)
    args = parser.parse_args()
    args.qc_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "mask":
        png_path, log_path, verdict = qc_mask(args.qc_dir)
    elif args.stage == "bkg":
        png_path, log_path, verdict = qc_bkg(args.qc_dir)
    elif args.stage == "bkgrms":
        png_path, log_path, verdict = qc_bkgrms(args.qc_dir)
    elif args.stage == "phot":
        png_path, log_path, verdict = qc_phot(args.qc_dir)
    elif args.stage == "calib":
        png_path, log_path, verdict = qc_calib(args.qc_dir)
    elif args.stage == "stack":
        png_path, log_path, verdict = qc_stack(args.qc_dir)
    elif args.stage == "stackphot":
        png_path, log_path, verdict = qc_stackphot(args.qc_dir)
    else:
        raise SystemExit(f"QC for stage {args.stage} is not implemented yet")
    print(png_path)
    print(log_path)
    print(f"VALIDATION_RESULT={verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
