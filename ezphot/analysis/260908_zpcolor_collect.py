#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260908_zpcolor_collect.py

Collect everything needed to test whether the 7DT medium-band zero points (ZP)
depend on stellar colour, i.e. whether the synthetic reference photometry
(GaiaXP x our transmission curves) matches the real filter response.

Stages (``--stage``):

  curves  : compare the transmission curves that exist on disk
            (ezphot config / repository ``transmission_old`` / doabdi ``Final_transmission``)
            -> curves_comparison.csv, figures/curves_comparison.png
  harvest : read the FITS header and sidecars of every medium-band single frame
            -> frames.csv (one row per frame), colorterms_header.csv (long: one row per
               K_COLOR/C_COLOR pair)
  refit   : re-match the stars of each ``.refcat`` to the GAIAXP and GAIAXP_CORR_LAMOST
            archives, recompute the ZP the same way the library does and fit colour terms
            -> refit_frames.csv (frame x catalogue x aperture), refit_stars.csv.gz (per star)
  calspec : absolute check on the CALSPEC standard-star fields
            -> calspec_synphot.csv (synthetic mags of the standards), calspec_frames.csv
  all     : curves, harvest, refit, calspec in that order

Strictly read-only with respect to the ezphot library, its configuration and the
sky-catalogue archives.  All products go to ``--out-dir``.

Typical use::

  MPLCONFIGDIR=/tmp/ezphot-matplotlib /home/hhchoi1022/anaconda3/envs/py311/bin/python \\
      ezphot/analysis/260908_zpcolor_collect.py --stage all --n-proc 32

  # smoke test on a random subset
  ... --stage harvest --limit 300
  ... --stage refit --limit 60 --filters m500,m425,m438,m525
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.io import ascii, fits
from astropy.stats import SigmaClip
from astropy.table import Table, vstack

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
SCIDATA_DIR = Path("/home/hhchoi1022/ezphot/data/scidata")
DEFAULT_TELKEY = "7DT_C361K_HIGH_1x1"
DEFAULT_OUT_DIR = Path("/home/hhchoi1022/ezphot/log/20260908_zp_color_check")
REPO_ROOT = Path(__file__).resolve().parents[2]

TRANSMISSION_DIRS = {
    "config": Path("/home/hhchoi1022/ezphot/config/common/transmission"),
    "repo_old": REPO_ROOT / "ezphot" / "configuration" / "common" / "transmission_old",
    "doabdi_final": Path("/lyman/data1/factory_doabdi/Final_transmission"),
}

CALSPEC_DIR = Path("/qso/data3/doabdi/intern/synthetic/calspec_data")
CALSPEC_INFO = Path(__file__).resolve().parent / "calspec_info.csv"

# Header OBJECT of the standard-star fields -> name used in calspec_info.csv ('star_name_1').
# BD+02 3375 is not in that file: its ICRS J2000 position and proper motion (mas/yr)
# come from SIMBAD (queried 2026-09-08) and the CALSPEC model spectrum is given explicitly.
STANDARD_STARS = {
    "BD+02_3375": dict(csv_name=None, spec="bd02d3375_mod_005.fits",
                       ra_j2000=264.9399786, dec_j2000=2.4165575, pmra=-365.856, pmdec=75.156),
    "BD-11_3759": dict(csv_name="BD-11 3759"),
    "FEIGE110": dict(csv_name="FEIGE110"),
    "GD71": dict(csv_name="GD71"),
    "GJ754.1A": dict(csv_name="GJ754.1A"),
    "HZ4": dict(csv_name="HZ4"),
    "LDS749B": dict(csv_name="LDS749B"),
    "WD0227+050": dict(csv_name="WD0227+050"),
    "WD0308-565": dict(csv_name="WD0308-565"),
    "WD1544-377": dict(csv_name="WD1544-377"),
    "WD1919+145": dict(csv_name="WD1919+145"),
    "WD2039-682": dict(csv_name="WD2039-682"),
    "WD2149+021": dict(csv_name="WD2149+021"),
}

# Medium bands that exist only in the legacy 12.5 nm grid (GAIAXP v1 archive) ...
LEGACY_GRID = {"m412", "m437", "m462", "m487", "m537", "m562", "m587", "m612", "m637", "m662",
               "m687", "m712", "m737", "m762", "m787", "m812", "m837", "m862", "m887"}
# ... and bands that exist only in the current filter set (GAIAXP_CORR_LAMOST archive).
NEW_ONLY = {"m386", "m438", "m466w", "m483", "m534", "m561", "m586", "m615", "m640", "m661",
            "m692w", "m710w", "m769w", "m832w"}
# Nearest legacy band used as a stand-in when refitting a new band against GAIAXP v1.
LEGACY_PROXY = {"m438": "m437", "m483": "m487", "m534": "m537", "m561": "m562",
                "m586": "m587", "m615": "m612", "m640": "m637", "m661": "m662"}

CATALOGS = ("GAIAXP", "GAIAXP_CORR_LAMOST")
APERTURES = ("APER_3", "AUTO")          # APER_3 is what BaseImage.zp / .depth read
APER_TAGS = ("APER", "APER_1", "APER_2", "APER_3", "APER_4", "APER_5", "APER_6", "AUTO")
SCALAR_KEYS = ("OBJECT", "TELESCOP", "FILTER", "DATE-OBS", "MJD", "EXPTIME", "AIRMASS",
               "ALTITUDE", "AZIMUTH", "CRVAL1", "CRVAL2", "NAXIS1", "NAXIS2", "SEEING",
               "PEEING", "SKYSIG", "SKYVAL", "ELLIP", "SATURATE", "FOCUSPOS", "CCD-TEMP",
               "GAIN", "EGAIN", "NCOMBINE", "TOTEXP")
COLOR_KEY_RE = re.compile(r"^([KC])_COLOR_(APER(?:_\d)?|AUTO)_(.+?)-(.+)$")

MATCH_RADIUS_ARCSEC = 2.5               # same as PhotometricCalibration default
REF_MAG_RANGE = (10, 20)                # same hard-coded cut as PhotometricCalibration L200
BPRP_BINS = [(-0.5, 0.5), (0.5, 0.8), (0.8, 1.1), (1.1, 1.5), (1.5, 3.0)]
MAGERR_FLOOR = 0.01                     # added in quadrature for the weighted fits
DEFAULT_PIXSCALE = 0.505                # arcsec/pix, 7DT C361K 1x1 (fallback only)


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def log(msg: str) -> None:
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def filter_nm(name: str):
    m = re.findall(r"\d+", str(name))
    return int(m[0]) if m else None


def list_single_frames(telkey: str, filters=None, frames: str = "single", bands: str = "medium"):
    """Frames of the medium bands ('medium'), the broad bands u g r i z ('broad') or both ('all').
    'single': prefix-less '7DT##_..._<filter>_....fits'; 'coadd': 'coadd_scaled_subbkg_*.com.fits'."""
    base = SCIDATA_DIR / "7DT" / telkey
    name = "7DT??_*.fits" if frames == "single" else "coadd_scaled_subbkg_*.com.fits"
    dirs = {"medium": ["m*"], "broad": ["u", "g", "r", "i", "z"], "all": ["m*", "u", "g", "r", "i", "z"]}[bands]
    files = sorted(f for d in dirs for f in base.glob(f"*/7DT??/{d}/{name}"))
    if filters:
        files = [f for f in files if f.parent.name in filters]
    return files


def subset(items, limit, seed):
    if limit and limit < len(items):
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(items), size=limit, replace=False))
        return [items[i] for i in idx]
    return list(items)


def weighted_linfit(x, y, sigma, nsig=3.0, maxiter=5):
    """Weighted linear fit y = a x + b with iterative nsig clipping.
    Returns a, a_err, b, n_used. a_err comes from the covariance scaled by chi2/dof."""
    x = np.asarray(x, float); y = np.asarray(y, float); sigma = np.asarray(sigma, float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(sigma) & (sigma > 0)
    a = b = a_err = np.nan
    for _ in range(maxiter):
        if m.sum() < 5 or np.ptp(x[m]) == 0:
            return np.nan, np.nan, np.nan, int(m.sum())
        (a, b), cov = np.polyfit(x[m], y[m], 1, w=1.0 / sigma[m], cov=True)
        a_err = float(np.sqrt(cov[0, 0]))
        res = y - (a * x + b)
        s = max(np.std(res[m]), 1e-4)
        new = m & (np.abs(res) < nsig * s)
        if new.sum() == m.sum():
            break
        m = new
    return float(a), a_err, float(b), int(m.sum())


def unweighted_slope(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.ptp(x[m]) == 0:
        return np.nan, np.nan
    a, b = np.polyfit(x[m], y[m], 1)
    return float(a), float(b)


# --------------------------------------------------------------------------------------
# Stage: curves
# --------------------------------------------------------------------------------------
def read_curve(path: Path):
    d = pd.read_csv(path)
    w = d.iloc[:, 0].to_numpy(float); r = np.clip(d.iloc[:, 1].to_numpy(float), 0, None)
    if np.nanmax(w) < 2000:            # doabdi files are in nm
        w = w * 10.0
    o = np.argsort(w)
    return w[o], r[o]


def curve_stats(w, r):
    piv = np.sqrt(np.trapezoid(r * w, w) / np.trapezoid(r / w, w))
    half = np.where(r >= r.max() / 2)[0]
    return dict(pivot_AA=float(piv), fwhm_AA=float(w[half[-1]] - w[half[0]]), max_response=float(r.max()),
                wmin_AA=float(w[0]), wmax_AA=float(w[-1]), n_points=int(len(w)))


def stage_curves(out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg_dir = TRANSMISSION_DIRS["config"]
    names = sorted(p.stem for p in cfg_dir.glob("*.csv") if re.fullmatch(r"(m\d{3}w?|[ugriz])", p.stem))
    rows = []
    for name in names:
        wc, rc = read_curve(cfg_dir / f"{name}.csv")
        base = curve_stats(wc, rc)
        for src, d in TRANSMISSION_DIRS.items():
            p = d / f"{name}.csv"
            if not p.exists():
                rows.append(dict(filter=name, source=src, exists=False)); continue
            w, r = read_curve(p)
            st = curve_stats(w, r)
            ri = np.interp(wc, w, r, left=0, right=0)
            shape_rms = float(np.sqrt(np.mean((rc / rc.max() - ri / ri.max()) ** 2)))
            rows.append(dict(filter=name, source=src, exists=True, **st,
                             dpivot_vs_config_AA=st["pivot_AA"] - base["pivot_AA"],
                             dfwhm_vs_config_AA=st["fwhm_AA"] - base["fwhm_AA"],
                             shape_rms_vs_config=shape_rms))
    df = pd.DataFrame(rows)
    write_csv_atomic(df, out_dir / "curves_comparison.csv")

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for src, mk in (("repo_old", "o"), ("doabdi_final", "s")):
        d = df[(df.source == src) & (df.exists == True)]
        piv = df[(df.source == "config")].set_index("filter").loc[d["filter"], "pivot_AA"].to_numpy()
        axes[0].plot(piv / 10, d["dpivot_vs_config_AA"], mk, label=src, alpha=.8)
        axes[1].plot(piv / 10, d["shape_rms_vs_config"], mk, label=src, alpha=.8)
        for x, y, f in zip(piv / 10, d["dpivot_vs_config_AA"], d["filter"]):
            if abs(y) > 1: axes[0].annotate(f, (x, y), fontsize=7)
    axes[0].axhline(0, color="k", lw=.5); axes[0].set_ylabel("pivot - pivot(config) [Å]")
    axes[1].set_ylabel("normalised shape RMS vs config"); axes[1].set_xlabel("pivot wavelength (config) [nm]")
    axes[1].set_yscale("log"); axes[0].legend(); axes[0].set_title("Transmission-curve provenance check")
    (out_dir / "figures").mkdir(exist_ok=True)
    fig.tight_layout(); fig.savefig(out_dir / "figures" / "curves_comparison.png", dpi=130); plt.close(fig)
    log(f"curves: {len(names)} filters -> curves_comparison.csv")
    med = df[(df.source == "repo_old") & df["filter"].str.match(r"m\d{3}$") & (df.exists == True)]
    log(f"curves: medium-band |dpivot| repo_old vs config: max {med['dpivot_vs_config_AA'].abs().max():.1f} Å")


# --------------------------------------------------------------------------------------
# Stage: harvest
# --------------------------------------------------------------------------------------
def count_table_rows(path: Path):
    n = 0
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            if line.strip() and not line.startswith("#"):
                n += 1
    return max(n - 1, 0)   # header row


def classify_refcat(partners: set) -> str:
    leg = partners & LEGACY_GRID; new = partners & NEW_ONLY
    if leg and not new: return "LEGACY"
    if new and not leg: return "NEW"
    if leg and new: return "MIXED"
    return "AMBIGUOUS"


def harvest_one(path_str: str):
    path = Path(path_str)
    row = {"file": str(path), "objname_dir": path.parents[2].name,
           "telname_dir": path.parents[1].name, "filter_dir": path.parent.name}
    try:
        h = fits.getheader(path)
    except Exception as e:                      # noqa: BLE001
        row["error"] = repr(e); return row, []
    for k in SCALAR_KEYS:
        row[k] = h.get(k)
    for tag in APER_TAGS:
        for pre in ("ZP_", "ZPERR_", "UL5SKY_", "K_MAG_", "C_MAG_"):
            row[f"{pre}{tag}"] = h.get(f"{pre}{tag}")
    try:
        cd = np.array([[h.get("CD1_1", np.nan), h.get("CD1_2", 0.0)], [h.get("CD2_1", 0.0), h.get("CD2_2", np.nan)]], float)
        row["pixscale"] = float(np.sqrt(abs(np.linalg.det(cd))) * 3600)
    except Exception:                           # noqa: BLE001
        row["pixscale"] = np.nan
    refcat = path.with_name(path.name + ".refcat")
    row["has_refcat"] = refcat.exists()
    row["n_refstar"] = count_table_rows(refcat) if refcat.exists() else np.nan
    row["has_cat"] = path.with_name(path.name + ".cat").exists()
    status = path.with_name(path.name + ".status")
    row["zpcalc_time"] = None
    if status.exists():
        try:
            st = json.load(open(status))
            if st.get("ZPCALC", {}).get("status"):
                row["zpcalc_time"] = st["ZPCALC"].get("update_time")
        except Exception:                       # noqa: BLE001
            pass
    pairs, partners = {}, set()
    for k in h.keys():
        m = COLOR_KEY_RE.match(k)
        if not m: continue
        kind, aper, f1, f2 = m.groups()
        pairs.setdefault((aper, f1, f2), {})[kind] = h[k]
        partners.update([f1, f2])
    long = [dict(file=str(path), filter=row["FILTER"], telescop=row["TELESCOP"], aper=a, f1=f1, f2=f2,
                 K=v.get("K"), C=v.get("C")) for (a, f1, f2), v in pairs.items()]
    row["n_colorterms"] = len(long)
    row["refcat_class"] = classify_refcat(partners)
    obj = str(row.get("OBJECT"))
    row["obj_type"] = "tile" if re.fullmatch(r"T\d{5}", obj) else ("standard" if obj in STANDARD_STARS else "other")
    return row, long


def stage_harvest(out_dir: Path, telkey: str, filters, limit, seed, n_proc, frames="single", bands="medium"):
    files = subset(list_single_frames(telkey, filters, frames, bands), limit, seed)
    log(f"harvest: {len(files)} {frames} frames ({bands} bands) under {telkey}")
    rows, longs = [], []
    with Pool(n_proc) as pool:
        for i, (row, long) in enumerate(pool.imap_unordered(harvest_one, [str(f) for f in files], chunksize=16), 1):
            rows.append(row); longs.extend(long)
            if i % 2000 == 0: log(f"harvest: {i}/{len(files)}")
    df = pd.DataFrame(rows).sort_values("file").reset_index(drop=True)
    write_csv_atomic(df, out_dir / "frames.csv")
    write_csv_atomic(pd.DataFrame(longs), out_dir / "colorterms_header.csv")
    n_zp = df["ZP_APER_3"].notna().sum()
    log(f"harvest: frames.csv {len(df)} rows ({n_zp} with ZP_APER_3, {df['has_refcat'].sum()} with .refcat); "
        f"colorterms_header.csv {len(longs)} rows")
    log("harvest: refcat_class " + json.dumps(df["refcat_class"].value_counts().to_dict()))


# --------------------------------------------------------------------------------------
# Stage: refit
# --------------------------------------------------------------------------------------
_G: dict = {}


def _refit_init():
    warnings.filterwarnings("ignore")
    from ezphot.dataobjects import PhotometricSpectrum
    from ezphot.skycatalog.skycatalogutility import SkyCatalogUtility
    _G["util"] = SkyCatalogUtility()
    _G["pivot"] = PhotometricSpectrum.FILTER_PIVOT_WAVELENGTH_NM
    _G["cache"] = {}


def get_reference(cat_type, objname, ra, dec, fov_ra, fov_dec):
    """Reference stars of one catalogue around a pointing, with the library's default cuts.
    Cached per (catalogue, OBJECT); trimmed to a box around the pointing to keep memory small."""
    key = (cat_type, objname)
    cache = _G["cache"]
    if key in cache:
        return cache[key]
    if len(cache) > 40:
        cache.clear()
    util = _G["util"]
    try:
        cats = util.get_catalogs(ra=ra, dec=dec, objname=objname, fov_ra=fov_ra, fov_dec=fov_dec,
                                 catalog_type=cat_type, catalog_version="v1",
                                 query_when_not_archived=False, verbose=False)
    except Exception:                           # noqa: BLE001
        cats = []
    tables, tiles = [], []
    for c in cats:
        try:
            data, _ = util.select_reference_sources(c, mag_lower=REF_MAG_RANGE[0], mag_upper=REF_MAG_RANGE[1])
        except Exception:                       # noqa: BLE001
            continue
        if data is None or len(data) == 0: continue
        keep = [n for n in data.colnames if n in ("id", "ra", "dec", "bp-rp", "g_mean") or n.endswith("_mag")]
        tables.append(data[keep]); tiles.append(str(c.objname))
    if not tables:
        cache[key] = None; return None
    tbl = vstack(tables, metadata_conflicts="silent")
    cosd = max(np.cos(np.deg2rad(dec)), 0.05)
    dra = np.abs(((np.asarray(tbl["ra"], float) - ra + 180) % 360) - 180) * cosd
    ddec = np.abs(np.asarray(tbl["dec"], float) - dec)
    tbl = tbl[(dra < 0.75 * fov_ra + 0.15) & (ddec < 0.75 * fov_dec + 0.15)]
    coords = SkyCoord(np.asarray(tbl["ra"], float), np.asarray(tbl["dec"], float), unit="deg")
    cache[key] = (tbl, coords, tiles)
    return cache[key]


def resolve_mag_column(filt: str, colnames, cat_type: str):
    if f"{filt}_mag" in colnames:
        return f"{filt}_mag", ""
    proxy = LEGACY_PROXY.get(filt)
    if proxy and f"{proxy}_mag" in colnames:
        return f"{proxy}_mag", proxy
    return None, ""


def neighbour_filter(filt: str, colnames):
    """A medium band 150-350 nm away (same window as the library), preferring |offset| ~ 250 nm.
    Uses the number in the filter name, exactly like PhotometricCalibration."""
    nm0 = filter_nm(filt)
    if nm0 is None: return None
    target = nm0 + 250 if nm0 < 650 else nm0 - 250
    cands = []
    for c in colnames:
        if not c.endswith("_mag"): continue
        f = c[:-4]
        if not re.fullmatch(r"m\d{3}", f) or f == filt: continue
        d = abs(filter_nm(f) - nm0)
        if 150 < d < 350: cands.append(f)
    return min(cands, key=lambda f: abs(filter_nm(f) - target)) if cands else None


def refit_one(row: dict):
    path = Path(row["file"])
    filt = row["FILTER"]; objname = str(row["OBJECT"])
    base = dict(file=str(path), filter=filt, telescop=row["TELESCOP"], objname=objname,
                obj_type=row["obj_type"], mjd=row["MJD"], airmass=row["AIRMASS"],
                zp_header=row["ZP_APER_3"], zperr_header=row["ZPERR_APER_3"],
                refcat_class=row["refcat_class"])
    refcat = path.with_name(path.name + ".refcat")
    if not refcat.exists():
        return [dict(base, catalog=c, aper=a, status="no_refcat") for c in CATALOGS for a in APERTURES], []
    try:
        ref = ascii.read(refcat)
    except Exception as e:                      # noqa: BLE001
        return [dict(base, catalog=c, aper=a, status=f"refcat_read_error:{e!r}"[:80]) for c in CATALOGS for a in APERTURES], []
    need = {"X_WORLD", "Y_WORLD", "X_IMAGE", "Y_IMAGE"} | {f"MAG_{a}" for a in APERTURES} | {f"MAGERR_{a}" for a in APERTURES}
    if not need.issubset(ref.colnames):
        return [dict(base, catalog=c, aper=a, status="refcat_missing_columns") for c in CATALOGS for a in APERTURES], []
    obj_coords = SkyCoord(np.asarray(ref["X_WORLD"], float), np.asarray(ref["Y_WORLD"], float), unit="deg")
    pixscale = row.get("pixscale")
    pixscale = float(pixscale) if pixscale and np.isfinite(pixscale) else DEFAULT_PIXSCALE
    nx = float(row.get("NAXIS1") or 9576); ny = float(row.get("NAXIS2") or 6388)
    r_arcmin_all = np.hypot(np.asarray(ref["X_IMAGE"], float) - nx / 2, np.asarray(ref["Y_IMAGE"], float) - ny / 2) * pixscale / 60
    ra0 = float(row.get("CRVAL1") if np.isfinite(row.get("CRVAL1", np.nan)) else np.nanmedian(obj_coords.ra.deg))
    dec0 = float(row.get("CRVAL2") if np.isfinite(row.get("CRVAL2", np.nan)) else np.nanmedian(obj_coords.dec.deg))
    fov_ra = nx * pixscale / 3600; fov_dec = ny * pixscale / 3600

    results, stars = [], []
    for cat_type in CATALOGS:
        got = get_reference(cat_type, objname, ra0, dec0, fov_ra, fov_dec)
        if got is None:
            results += [dict(base, catalog=cat_type, aper=a, status="no_reference") for a in APERTURES]; continue
        tbl, coords, tiles = got
        magcol, proxy = resolve_mag_column(filt, tbl.colnames, cat_type)
        if magcol is None:
            results += [dict(base, catalog=cat_type, aper=a, status="no_filter_column") for a in APERTURES]; continue
        idx, d2d, _ = match_coordinates_sky(obj_coords, coords)      # same as Helper.cross_match
        ok = d2d.arcsec < MATCH_RADIUS_ARCSEC
        n_matched = int(ok.sum())
        if n_matched < 5:
            results += [dict(base, catalog=cat_type, aper=a, status="too_few_matches", n_matched=n_matched) for a in APERTURES]; continue
        mo = ref[ok]; mr = tbl[idx[ok]]; r_arcmin = r_arcmin_all[ok]
        nbr = neighbour_filter(filt, tbl.colnames)
        ref_mag = np.asarray(mr[magcol], float)
        colors = {"bprp": np.asarray(mr["bp-rp"], float) if "bp-rp" in mr.colnames else None,
                  "gr": (np.asarray(mr["g_mag"], float) - np.asarray(mr["r_mag"], float))
                        if {"g_mag", "r_mag"}.issubset(mr.colnames) else None,
                  "nbr": (ref_mag - np.asarray(mr[f"{nbr}_mag"], float)) if nbr else None}
        for aper in APERTURES:
            inst = np.asarray(mo[f"MAG_{aper}"], float); ierr = np.asarray(mo[f"MAGERR_{aper}"], float)
            zp = ref_mag - inst
            finite = np.isfinite(zp) & np.isfinite(ierr) & (inst < 90)
            rec = dict(base, catalog=cat_type, aper=aper, mag_column=magcol, proxy=proxy, nbr_filter=nbr,
                       ref_tiles=";".join(tiles), n_matched=n_matched, n_finite=int(finite.sum()))
            if finite.sum() < 5:
                rec["status"] = "too_few_finite"; results.append(rec); continue
            # --- zero point exactly as PhotometricCalibration: 3-sigma clip, median, std
            clipped = SigmaClip(sigma=3.0, maxiters=5)(zp[finite])
            keep = np.where(finite)[0][~np.ma.getmaskarray(clipped)]
            zp_k = zp[keep]; resid = zp_k - np.nanmedian(zp_k)
            sig = np.sqrt(ierr[keep] ** 2 + MAGERR_FLOOR ** 2)
            rec.update(status="ok", n_used=int(len(keep)), zp_median=float(np.nanmedian(zp_k)), zp_std=float(np.nanstd(zp_k)),
                       sigma_before=float(np.nanstd(zp_k)))
            # --- colour terms
            for cname, col in colors.items():
                if col is None: rec[f"a_{cname}"] = np.nan; continue
                c = col[keep]
                a_u, b_u = unweighted_slope(c, resid)                    # = library method (curve_fit linear)
                a_w, a_e, b_w, n_w = weighted_linfit(c, resid, sig)
                good = np.isfinite(c)
                rec.update({f"a_{cname}_unw": a_u, f"b_{cname}_unw": b_u, f"a_{cname}": a_w, f"a_{cname}_err": a_e,
                            f"b_{cname}": b_w, f"n_{cname}": n_w,
                            f"color_{cname}_median": float(np.nanmedian(c)) if good.any() else np.nan,
                            f"color_{cname}_span": float(np.nanpercentile(c[good], 90) - np.nanpercentile(c[good], 10)) if good.sum() > 2 else np.nan})
                if cname == "bprp" and np.isfinite(a_w):
                    rec["sigma_after_bprp"] = float(np.nanstd(resid[good] - (a_w * c[good] + b_w)))
                    if good.sum() >= 8 and np.ptp(c[good]) > 0:
                        rec["a2_bprp"] = float(np.polyfit(c[good], resid[good], 2)[0])
                    for i, (lo, hi) in enumerate(BPRP_BINS, 1):
                        inb = good & (c >= lo) & (c < hi)
                        rec[f"bin{i}_median"] = float(np.nanmedian(resid[inb])) if inb.sum() >= 5 else np.nan
                        rec[f"bin{i}_n"] = int(inb.sum())
            # --- field position
            r = r_arcmin[keep]
            rec["a_radius"], _ = unweighted_slope(r, resid)
            rec["r_median_arcmin"] = float(np.nanmedian(r))
            if colors["bprp"] is not None:
                c = colors["bprp"][keep]; inner = r <= np.nanmedian(r)
                rec["a_bprp_inner"] = weighted_linfit(c[inner], resid[inner], sig[inner])[0]
                rec["a_bprp_outer"] = weighted_linfit(c[~inner], resid[~inner], sig[~inner])[0]
            results.append(rec)
            if aper == "APER_3":
                ids = mr["id"] if "id" in mr.colnames else np.arange(len(mr))
                for j, kk in enumerate(keep):
                    stars.append(dict(file=str(path), catalog=cat_type, filter=filt, telescop=row["TELESCOP"],
                                      objname=objname, ref_id=str(ids[kk]),
                                      bprp=colors["bprp"][kk] if colors["bprp"] is not None else np.nan,
                                      gr=colors["gr"][kk] if colors["gr"] is not None else np.nan,
                                      nbr=colors["nbr"][kk] if colors["nbr"] is not None else np.nan,
                                      zp=float(zp_k[j]), resid=float(resid[j]), magerr=float(ierr[kk]),
                                      ref_mag=float(ref_mag[kk]), r_arcmin=float(r_arcmin[kk])))
    return results, stars


def stage_refit(out_dir: Path, filters, limit, seed, n_proc):
    fpath = out_dir / "frames.csv"
    if not fpath.exists():
        sys.exit("refit: frames.csv not found - run --stage harvest first")
    frames = pd.read_csv(fpath, low_memory=False)
    frames = frames[frames["ZP_APER_3"].notna() & frames["has_refcat"].astype(bool)]
    if filters:
        frames = frames[frames["FILTER"].isin(filters)]
    frames = frames.sort_values(["OBJECT", "TELESCOP", "FILTER"])
    rows = subset(frames.to_dict("records"), limit, seed)
    log(f"refit: {len(rows)} frames x {len(CATALOGS)} catalogues")
    stars_path = out_dir / "refit_stars.csv.gz"
    if stars_path.exists(): stars_path.unlink()
    results, buf, first = [], [], True
    with Pool(n_proc, initializer=_refit_init) as pool:
        for i, (res, stars) in enumerate(pool.imap_unordered(refit_one, rows, chunksize=4), 1):
            results += res; buf += stars
            if len(buf) >= 200000 or i == len(rows):
                if buf:
                    pd.DataFrame(buf).to_csv(stars_path, index=False, mode="a", header=first, compression="gzip")
                    first = False; buf = []
            if i % 500 == 0: log(f"refit: {i}/{len(rows)}")
    df = pd.DataFrame(results)
    write_csv_atomic(df, out_dir / "refit_frames.csv")
    log("refit: status counts " + json.dumps(df["status"].value_counts().to_dict()))
    ok = df[(df.status == "ok") & (df.aper == "APER_3")]
    for cat in CATALOGS:
        d = ok[ok.catalog == cat]
        if len(d):
            dz = (d["zp_median"] - d["zp_header"]).abs()
            log(f"refit: {cat}: {len(d)} frames, |zp_refit - zp_header| median {dz.median():.4f}, "
                f"fraction < 0.002: {(dz < 0.002).mean():.2f}")


# --------------------------------------------------------------------------------------
# Stage: calspec
# --------------------------------------------------------------------------------------
def load_standard_table() -> pd.DataFrame:
    info = pd.read_csv(CALSPEC_INFO)
    rows = []
    for obj, d in STANDARD_STARS.items():
        if d.get("csv_name") is None:
            rows.append(dict(object=obj, spec=d["spec"], ra_j2000=d["ra_j2000"], dec_j2000=d["dec_j2000"],
                             pmra=d["pmra"], pmdec=d["pmdec"], spectral_type=None, b_v=np.nan)); continue
        m = info[info["star_name_1"].str.upper() == d["csv_name"].upper()]
        if len(m) != 1:
            raise RuntimeError(f"{obj}: {len(m)} rows in calspec_info.csv for '{d['csv_name']}'")
        m = m.iloc[0]
        rows.append(dict(object=obj, spec=m["spectrum_filename"], ra_j2000=float(m["ra_j2000_deg"]),
                         dec_j2000=float(m["dec_j2000_deg"]), pmra=float(m["pmra_masyr"]) if pd.notna(m["pmra_masyr"]) else 0.0,
                         pmdec=float(m["pmdec_masyr"]) if pd.notna(m["pmdec_masyr"]) else 0.0,
                         spectral_type=m.get("spectral_type"), b_v=m.get("B-V", np.nan)))
    df = pd.DataFrame(rows)
    missing = [s for s in df["spec"] if not (CALSPEC_DIR / s).exists()]
    if missing:
        raise RuntimeError(f"CALSPEC spectra not found in {CALSPEC_DIR}: {missing}")
    return df


def synphot_standards(stars: pd.DataFrame, out_dir: Path, overwrite: bool) -> pd.DataFrame:
    path = out_dir / "calspec_synphot.csv"
    if path.exists() and not overwrite:
        return pd.read_csv(path)
    from ezphot.dataobjects import Spectrum
    filters = {}
    rows = []
    for _, s in stars.iterrows():
        t = Table.read(CALSPEC_DIR / s["spec"])
        wl = np.asarray(t["WAVELENGTH"], float); fl = np.asarray(t["FLUX"], float)
        m = (wl > 3000) & (wl < 11000) & np.isfinite(fl)
        spec = Spectrum(wavelength=wl[m], flux=fl[m], wavelength_unit="AA", flux_unit="flamb", verbose=False)
        phot, filters = spec.synphot(filterset=["medium", "u", "g", "r", "i", "z"], visualize=False,
                                     pyphot_filters=filters)[:2]
        for f, v in phot.items():
            rows.append(dict(object=s["object"], spec=s["spec"], filter=str(f), mag_syn=float(v["mag"]),
                             wl_pivot_nm=float(getattr(v["wl_pivot"], "value", v["wl_pivot"]))))
        log(f"calspec: synphot {s['object']} ({s['spec']}) done")
    df = pd.DataFrame(rows)
    write_csv_atomic(df, path)
    return df


def calspec_one(args):
    row, star, syn = args      # syn: {filter: mag_syn}
    path = Path(row["file"]); filt = row["FILTER"]
    rec = dict(file=str(path), object=row["OBJECT"], filter=filt, telescop=row["TELESCOP"], mjd=row["MJD"],
               airmass=row["AIRMASS"], exptime=row["EXPTIME"], seeing=row["SEEING"], refcat_class=row["refcat_class"],
               n_refstar=row["n_refstar"], mag_syn=syn.get(filt, np.nan), spectral_type=star["spectral_type"], b_v=star["b_v"])
    for a in APERTURES:
        rec[f"zp_{a}"] = row[f"ZP_{a}"]; rec[f"zperr_{a}"] = row[f"ZPERR_{a}"]
    cat = path.with_name(path.name + ".cat")
    if not cat.exists():
        rec["status"] = "no_cat"; return rec
    try:
        t = ascii.read(cat)
    except Exception as e:                      # noqa: BLE001
        rec["status"] = f"cat_read_error:{e!r}"[:80]; return rec
    # proper-motion propagation J2000 -> observation epoch (pmra includes cos(dec))
    dt_yr = (float(row["MJD"]) - 51544.5) / 365.25 if pd.notna(row["MJD"]) else 24.5
    dec = star["dec_j2000"] + star["pmdec"] * dt_yr / 3.6e6
    ra = star["ra_j2000"] + star["pmra"] * dt_yr / 3.6e6 / np.cos(np.deg2rad(star["dec_j2000"]))
    c_star = SkyCoord(ra, dec, unit="deg")
    c_cat = SkyCoord(np.asarray(t["X_WORLD"], float), np.asarray(t["Y_WORLD"], float), unit="deg")
    sep = c_cat.separation(c_star).arcsec
    j = int(np.argmin(sep))
    rec["sep_arcsec"] = float(sep[j])
    if sep[j] > 3.0:
        rec["status"] = "not_detected"; return rec
    src = t[j]
    pixscale = row.get("pixscale"); pixscale = float(pixscale) if pixscale and np.isfinite(pixscale) else DEFAULT_PIXSCALE
    nx = float(row.get("NAXIS1") or 9576); ny = float(row.get("NAXIS2") or 6388)
    rec["r_arcmin"] = float(np.hypot(float(src["X_IMAGE"]) - nx / 2, float(src["Y_IMAGE"]) - ny / 2) * pixscale / 60)
    for k in ("FLAGS", "FLUX_MAX", "CLASS_STAR", "FWHM_WORLD", "IMAFLAGS_ISO"):
        rec[k.lower()] = float(src[k]) if k in t.colnames else np.nan
    sat = row.get("SATURATE")
    rec["saturated"] = bool(pd.notna(sat) and np.isfinite(rec["flux_max"]) and rec["flux_max"] > float(sat))
    for a in APERTURES:
        inst = float(src[f"MAG_{a}"]) if f"MAG_{a}" in t.colnames else np.nan
        if f"MAGSKY_{a}" in t.colnames and np.isfinite(float(src[f"MAGSKY_{a}"])):
            mag = float(src[f"MAGSKY_{a}"])
        else:
            mag = inst + float(row[f"ZP_{a}"]) if pd.notna(row[f"ZP_{a}"]) else np.nan
        rec[f"mag_inst_{a}"] = inst
        rec[f"mag_obs_{a}"] = mag
        rec[f"magerr_{a}"] = float(src[f"MAGERR_{a}"]) if f"MAGERR_{a}" in t.colnames else np.nan
        rec[f"delta_{a}"] = mag - rec["mag_syn"]
    rec["status"] = "ok"
    return rec


def stage_calspec(out_dir: Path, filters, limit, seed, n_proc, overwrite):
    fpath = out_dir / "frames.csv"
    if not fpath.exists():
        sys.exit("calspec: frames.csv not found - run --stage harvest first")
    stars = load_standard_table()
    syn = synphot_standards(stars, out_dir, overwrite)
    syn_map = {o: dict(zip(g["filter"], g["mag_syn"])) for o, g in syn.groupby("object")}
    frames = pd.read_csv(fpath, low_memory=False)
    frames = frames[frames["OBJECT"].isin(STANDARD_STARS) & frames["ZP_APER_3"].notna() & frames["has_cat"].astype(bool)]
    if filters:
        frames = frames[frames["FILTER"].isin(filters)]
    rows = subset(frames.to_dict("records"), limit, seed)
    star_map = {s["object"]: s for s in stars.to_dict("records")}
    jobs = [(r, star_map[r["OBJECT"]], syn_map[r["OBJECT"]]) for r in rows]
    log(f"calspec: {len(jobs)} standard-field frames, {len(stars)} standards")
    recs = []
    with Pool(n_proc) as pool:
        for i, rec in enumerate(pool.imap_unordered(calspec_one, jobs, chunksize=8), 1):
            recs.append(rec)
            if i % 1000 == 0: log(f"calspec: {i}/{len(jobs)}")
    df = pd.DataFrame(recs)
    write_csv_atomic(df, out_dir / "calspec_frames.csv")
    log("calspec: status counts " + json.dumps(df["status"].value_counts().to_dict()))
    ok = df[df.status == "ok"]
    if len(ok):
        log(f"calspec: delta_APER_3 median {ok['delta_APER_3'].median():+.4f}, MAD {1.4826 * (ok['delta_APER_3'] - ok['delta_APER_3'].median()).abs().median():.4f}")


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["curves", "harvest", "refit", "calspec", "all"])
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--telkey", default=DEFAULT_TELKEY)
    ap.add_argument("--filters", default=None, help="comma-separated filter names (default: all medium bands)")
    ap.add_argument("--limit", type=int, default=None, help="random subset size (smoke tests)")
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--n-proc", type=int, default=32)
    ap.add_argument("--overwrite", action="store_true", help="recompute cached products (calspec_synphot.csv)")
    ap.add_argument("--frames", default="single", choices=["single", "coadd"], help="single frames or coadd_scaled_subbkg stacks")
    ap.add_argument("--bands", default="medium", choices=["medium", "broad", "all"], help="medium bands, broad bands u g r i z, or both")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)
    filters = [f.strip() for f in args.filters.split(",")] if args.filters else None
    stages = ["curves", "harvest", "refit", "calspec"] if args.stage == "all" else [args.stage]
    t0 = time.time()
    for st in stages:
        log(f"=== stage {st} ===")
        if st == "curves":
            stage_curves(out_dir)
        elif st == "harvest":
            stage_harvest(out_dir, args.telkey, filters, args.limit, args.seed, args.n_proc, args.frames, args.bands)
        elif st == "refit":
            stage_refit(out_dir, filters, args.limit, args.seed, args.n_proc)
        elif st == "calspec":
            stage_calspec(out_dir, filters, args.limit, args.seed, args.n_proc, args.overwrite)
    log(f"done in {(time.time() - t0) / 60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
