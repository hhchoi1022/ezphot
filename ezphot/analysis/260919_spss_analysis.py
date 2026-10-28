#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260919_spss_analysis.py  (stage A: delta=0)

SPSS-campaign frame-level calibration against a reference built DIRECTLY from GAIA_BEST tiles
(coords + bp-rp + corrected-XP flux; the new SPSS fields are not in the ezphot catalog archive).

Per frame (physical curve, delta=0):
  ZP = clipped median of (syn_XP - inst) over SNR>40 matched field stars (gate zsd<=0.08)
  a_frame = weighted zp-colour slope of the ZP stars, c_ref = their weighted mean colour
  standard star (PM-propagated Gaia position):
    delta_spss = inst + ZP - syn(SPSS V2 flux table)   [ground truth]
    delta_xp   = inst + ZP - syn(star's own corrected XP)  [cross-check axis]

Output: spss_frames.csv — one row per (frame), powering QA2 / colour-offset relation / delta_f fits.

  MPLCONFIGDIR=/tmp/ezphot-matplotlib OMP_NUM_THREADS=1 /home/hhchoi1022/miniconda3/envs/too/bin/python -u \\
      ezphot/analysis/260919_spss_analysis.py --n-proc 14
"""
from __future__ import annotations

import argparse
import glob
import importlib
import os
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
import numpy as np                  # noqa: E402
import pandas as pd                 # noqa: E402

warnings.filterwarnings("ignore")
from astropy.coordinates import SkyCoord, match_coordinates_sky   # noqa: E402
from astropy.io import fits          # noqa: E402
from astropy.stats import SigmaClip  # noqa: E402

sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
OT = importlib.import_module("260911_offset_timeseries")           # select_stars_like, C/K/R
OZ = importlib.import_module("260911_offset_ezphot")               # load_ez
C, K, R = OT.C, OT.K, OT.R
XP_WAVE = OT.XP_WAVE
GRID20 = OT.GRID

OUT = Path("/home/hhchoi1022/ezphot/log/20260918_spss")
CATD = OUT / "catalogs"
FLUXD = Path("/home/hhchoi1022/ezphot/data/spss_fluxtables_v2")
SPSS_CSV = ("/tmp/claude-10000/-lyman-data1-factory-hhchoi-code-ezphot/"
            "b1308704-c229-4e49-b19d-bd7bc334e4b7/scratchpad/spss_table1.csv")
GAIA_EPOCH_MJD = 57389.0            # J2016.0
SNR_ZP = 40.0
MIN_REF = 15
APER = "APER_3"                     # photometry column: APER_3 (10") or AUTO

_PHYS = None


def load_cat(cp):
    """trimmed catalog -> select_stars_like schema, magnitudes from the APER column."""
    from astropy.io import fits as _f
    with _f.open(cp) as h:
        d = h["LDAC_OBJECTS"].data
        names = set(d.columns.names)
        rk = "X_WORLD" if "X_WORLD" in names else "ALPHA_J2000"
        dk = "Y_WORLD" if "Y_WORLD" in names else "DELTA_J2000"
        out = {"ALPHA_J2000": np.asarray(d[rk], float), "DELTA_J2000": np.asarray(d[dk], float),
               "MAG_APER_3": np.asarray(d[f"MAG_{APER}"], float),
               "MAGERR_APER_3": np.asarray(d[f"MAGERR_{APER}"], float),
               "FLAGS": np.asarray(d["FLAGS"], float), "CLASS_STAR": np.asarray(d["CLASS_STAR"], float),
               "FWHM_WORLD": np.asarray(d["FWHM_WORLD"], float),
               "ELONGATION": np.asarray(d["ELONGATION"], float)}
        out["SNR_APER_3"] = (np.asarray(d[f"SNR_{APER}"], float) if f"SNR_{APER}" in names
                             else 1.0857 / np.clip(out["MAGERR_APER_3"], 1e-4, None))
    return pd.DataFrame(out)


def log(m):
    print(time.strftime("[%H:%M:%S] ") + m, flush=True)


def norm(s):
    return str(s).upper().replace(" ", "").replace("_", "").replace("-", "").replace("+", "")


def _init():
    global _PHYS
    warnings.filterwarnings("ignore")
    _PHYS = {}
    for p in sorted((Path("/home/hhchoi1022/ezphot/log/20260910_transmission_sources")
                     / "total_curves_physical").glob("m*.csv")):
        d = pd.read_csv(p)
        _PHYS[p.stem] = (d.lam.to_numpy(), d.trans.to_numpy())


def gather_reference(ra0, dec0, radius=0.9):
    """GAIA_BEST rows near (ra0, dec0): sid, ra, dec, bp-rp, corrected-XP flux (343)."""
    T = C.TILES
    dd = np.abs(T.dec.values - dec0)
    dr = np.abs(((T.ra.values - ra0 + 180) % 360) - 180) * np.cos(np.radians(dec0))
    near = T[(dd < radius + 0.7) & (dr < radius + 0.7)]
    sids, ras, decs, bprp, pmra, pmdec, flux = [], [], [], [], [], [], []
    cosd = np.cos(np.radians(dec0))
    for tn in near.id.values:
        p = C.TILESRC / f"{tn}.csv"
        if not p.exists():
            continue
        with open(p) as fh:
            fh.readline()
            for line in fh:
                cc = line.split(",", 14)
                try:
                    ra, dec = float(cc[1]), float(cc[2])
                except ValueError:
                    continue
                if abs(dec - dec0) > radius or abs(((ra - ra0 + 180) % 360) - 180) * cosd > radius:
                    continue
                sid = cc[0].lstrip("﻿")
                if sid in sids:
                    continue
                vc = np.fromstring(cc[13], sep=";")
                if vc.size != 343:
                    continue
                try:
                    bp, rp = float(cc[6]), float(cc[7])
                    col = bp - rp
                except ValueError:
                    col = np.nan
                try:
                    pma = float(cc[4]) if cc[4] else 0.0
                    pmd = float(cc[5]) if cc[5] else 0.0
                except ValueError:
                    pma, pmd = 0.0, 0.0
                sids.append(sid); ras.append(ra); decs.append(dec); bprp.append(col)
                pmra.append(pma); pmdec.append(pmd); flux.append(vc)
    if not sids:
        return None
    return dict(sid=np.array(sids), ra=np.array(ras), dec=np.array(decs),
                bprp=np.array(bprp), pmra=np.array(pmra), pmdec=np.array(pmdec),
                flux=np.vstack(flux))


def load_spss_spec(spss_id):
    p = FLUXD / f"V2.SPSS{int(spss_id):03d}.ascii"
    if not p.exists():
        return None
    d = np.genfromtxt(p, comments="#", missing_values="NA", filling_values=np.nan, usecols=(0, 1))
    m = np.isfinite(d[:, 0]) & np.isfinite(d[:, 1]) & (d[:, 1] > 0)
    if m.sum() < 100:
        return None
    return d[m, 0], d[m, 1]                                          # nm, W/m2/nm


def process_object(args):
    obj, files, srow = args
    sc = SigmaClip(sigma=3.0, maxiters=5)
    spec = load_spss_spec(srow["SPSS"])
    ref = gather_reference(srow["RA_ICRS"], srow["DE_ICRS"])
    if ref is None:
        return [dict(file=f, object=obj, status="no_reference") for f in files]
    # the standard itself in GAIA_BEST (for PM + own-XP truth)
    sid = str(int(srow["GaiaEDR3"])) if pd.notna(srow["GaiaEDR3"]) else ""
    jstar = np.where(ref["sid"] == sid)[0]
    star_ast = None
    if len(jstar):
        j = int(jstar[0])
        star_ast = (ref["ra"][j], ref["dec"][j], ref["pmra"][j], ref["pmdec"][j],
                    ref["flux"][j], ref["bprp"][j])
    coords = SkyCoord(ref["ra"], ref["dec"], unit="deg")
    rows = []
    for fr in files:
        cp = CATD / (Path(fr).name[:-5] + "_cat.fits")
        row = dict(file=fr, object=obj, unit=Path(fr).name.split("_")[0],
                   filter=Path(fr).name.split("_")[-4], status="ok")
        try:
            if row["filter"] not in _PHYS:
                row["status"] = "no_curve"; rows.append(row); continue
            with fits.open(cp) as h:
                hdr = h[0].header
            df = load_cat(cp)
            mjd = float(hdr.get("MJD", np.nan))
            if not np.isfinite(mjd):
                from astropy.time import Time
                mjd = Time(hdr["DATE-OBS"]).mjd
            row["mjd"] = round(mjd, 5)
            row["airmass"] = float(hdr.get("AIRMASS", np.nan))
            sel = OT.select_stars_like(df)
            if len(sel) < MIN_REF:
                row["status"] = "few_stars"; rows.append(row); continue
            src = SkyCoord(sel.ALPHA_J2000.to_numpy(), sel.DELTA_J2000.to_numpy(), unit="deg")
            idx, d2d, _ = match_coordinates_sky(src, coords)
            ok = np.asarray(d2d.arcsec < 2.0)
            if ok.sum() < MIN_REF:
                row["status"] = "few_matches"; rows.append(row); continue
            mi = idx[ok]
            fc = ref["flux"][mi]
            bp = ref["bprp"][mi]
            inst = sel.MAG_APER_3.to_numpy()[ok]
            ierr = sel.MAGERR_APER_3.to_numpy()[ok]
            snr = sel.SNR_APER_3.to_numpy()[ok]
            sig = np.sqrt(ierr ** 2 + 0.01 ** 2)
            fw, ft = _PHYS[row["filter"]]
            syn = R.syn_mag(XP_WAVE, fc, fw, ft, 0.0)
            zp_arr = syn - inst
            fin = np.isfinite(zp_arr) & np.isfinite(bp)
            zsel = fin & (snr > SNR_ZP)
            if zsel.sum() < 10:
                row["status"] = "few_bright"; rows.append(row); continue
            clm = sc(zp_arr[zsel])
            gz = ~np.ma.getmaskarray(clm)
            zp = float(np.ma.median(clm)); zsd = float(np.ma.std(clm))
            if not np.isfinite(zp) or zsd > 0.08:
                row["status"] = "bad_zp"; rows.append(row); continue
            row.update(zp=round(zp, 4), zp_std=round(zsd, 4), n_zp=int(gz.sum()), n_match=int(fin.sum()))
            aa, ae, bb, nf = R.weighted_linfit(bp[zsel][gz], zp_arr[zsel][gz] - zp, sig[zsel][gz])
            row.update(a_frame=round(float(aa), 4), a_err=round(float(ae), 4),
                       c_ref=round(float(np.average(bp[zsel][gz], weights=1 / sig[zsel][gz] ** 2)), 3))
            # colour term over ALL matched stars too (wider colour range)
            aw, awe, _, _ = R.weighted_linfit(bp[fin], zp_arr[fin] - zp, sig[fin])
            row.update(a_all=round(float(aw), 4), a_all_err=round(float(awe), 4))
            # ---- standard star ----
            if star_ast is not None:
                ra_s, dec_s, pma, pmd, fstar, cstar = star_ast
                dt = (mjd - GAIA_EPOCH_MJD) / 365.25
                pos = SkyCoord(ra_s + pma * dt / 3.6e6 / np.cos(np.deg2rad(dec_s)),
                               dec_s + pmd * dt / 3.6e6, unit="deg")
                call = SkyCoord(df.ALPHA_J2000.to_numpy(), df.DELTA_J2000.to_numpy(), unit="deg")
                sep = call.separation(pos).arcsec
                j = int(np.argmin(sep))
                srow_ = df.iloc[j]
                sat = float(hdr.get("SATURATE", np.nan))
                with fits.open(cp) as h:
                    dd_ = h["LDAC_OBJECTS"].data
                    fm = float(dd_["FLUX_MAX"][j]) if "FLUX_MAX" in dd_.columns.names else np.nan
                if (sep[j] < 2.0 and srow_.FLAGS <= 1 and srow_.MAGERR_APER_3 < 0.05
                        and (not np.isfinite(sat) or fm < 0.9 * sat)):
                    m_xp = float(R.syn_mag(XP_WAVE, fstar[None, :], fw, ft, 0.0)[0])
                    row.update(star_inst=round(float(srow_.MAG_APER_3), 4),
                               star_err=round(float(srow_.MAGERR_APER_3), 4),
                               c_star=round(float(cstar), 3),
                               delta_xp=round(float(srow_.MAG_APER_3) + zp - m_xp, 4))
                    if spec is not None:
                        m_true = float(R.syn_mag(spec[0], spec[1], fw, ft, 0.0))
                        row["delta_spss"] = round(float(srow_.MAG_APER_3) + zp - m_true, 4)
        except Exception as e:                                       # noqa: BLE001
            row["status"] = "error:" + str(e)[:120]
        rows.append(row)
    return rows


def main():
    global APER
    ap = argparse.ArgumentParser(); ap.add_argument("--n-proc", type=int, default=14)
    ap.add_argument("--aper", default="APER_3", choices=["APER_3", "AUTO"])
    a = ap.parse_args()
    APER = a.aper
    sfx = "" if a.aper == "APER_3" else "_auto"
    spss = pd.read_csv(SPSS_CSV)
    srow_of = {norm(s.Name): s for _, s in spss.iterrows()}
    metas = sorted(glob.glob(str(OUT / "process_meta_7DT*.csv")))
    meta = pd.concat([pd.read_csv(m) for m in metas], ignore_index=True)
    meta = meta[meta.status == "ok"]
    jobs = []
    unmatched = []
    for obj, g in meta.groupby("object"):
        s = srow_of.get(norm(obj))
        if s is None:
            unmatched.append(obj)
            continue
        jobs.append((obj, sorted(g.file), s))
    log(f"{len(meta)} ok frames, {len(jobs)} SPSS objects; not in SPSS table: {unmatched}")
    rows = []
    with Pool(min(a.n_proc, len(jobs)), initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(process_object, jobs), 1):
            rows += r
            log(f"  [{i}/{len(jobs)}] +{len(r)} (total {len(rows)})")
            pd.DataFrame(rows).to_csv(OUT / f"spss_frames{sfx}.partial.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"spss_frames{sfx}.csv", index=False)
    okd = df[df.status == "ok"]
    log(f"done: {len(df)} rows, ok {len(okd)}, with delta_spss {okd.get('delta_spss', pd.Series()).notna().sum()}")
    print(df.status.str[:22].value_counts().head(8).to_string())


if __name__ == "__main__":
    main()
