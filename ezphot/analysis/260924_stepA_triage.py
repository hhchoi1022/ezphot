#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260924_stepA_triage.py — Step A of the passband diagnosis: before blaming a curve for an offset shared by all
references, test the non-curve explanations frame by frame.

For every standard-star frame (CALSPEC images 2024, SPSS images 2026), with the manufacturer·nobeam total
curve, it records:
  ZP reference   : ZP from corrected XP (baseline), raw XP, blue field stars (BP-RP<0.8), red field stars
                   (BP-RP>1.2), bright (G<13.5) and faint (G>=13.5) field stars -> an offset that moves with
                   these is a ZP-reference (XP) effect, not a curve/system effect of the standard.
  field profile  : median field residual (syn_cor - inst - ZP) in BP-RP bins, SNR>20 stars incl. M dwarfs
                   -> curvature rising to the red is the red-leak signature.
  standard       : Delta vs its own spectrum (CALSPEC or SPSS), vs its corrected and raw XP; FLUX_MAX,
                   instrumental mag, detector position, colour.
  conditions     : AIRMASS, EXPTIME, HUMIDITY, AMBTEMP, MJD, seeing, unit, night.

Stage 1 caches the reference-star table (raw+corrected XP, G, BP-RP, PM) per object; stage 2 processes
(object, filter) groups in parallel.  Output: stepA_frames.csv

  MPLCONFIGDIR=/tmp/ezphot-matplotlib OMP_NUM_THREADS=1 /home/hhchoi1022/miniconda3/envs/too/bin/python -u \\
      ezphot/analysis/260924_stepA_triage.py --n-proc 24 --cap 40
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
import numpy as np                  # noqa: E402
import pandas as pd                 # noqa: E402

warnings.filterwarnings("ignore")
from astropy.coordinates import SkyCoord, match_coordinates_sky   # noqa: E402
from astropy.io import fits          # noqa: E402
from astropy.stats import SigmaClip  # noqa: E402
from astropy.table import Table      # noqa: E402

sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
TC = importlib.import_module("260921_truthcompare")
OT, OZ, AN, TW, C, R = TC.OT, TC.OZ, TC.AN, TC.TW, TC.C, TC.R
XP_WAVE = TC.XP_WAVE

OUT = Path("/home/hhchoi1022/ezphot/log/20260924_stepA")
OUT.mkdir(exist_ok=True)
CACHE = OUT / "refcache"
CACHE.mkdir(exist_ok=True)
SPSS_CATD = Path("/home/hhchoi1022/ezphot/log/20260918_spss/catalogs")
CAL_CATD = Path("/home/hhchoi1022/ezphot/log/20260914_calspec_ezphot/catalogs")
COMBO = ("manufacturer", "nobeam")
MIN_REF = 10
BINS = np.array([-0.6, 0.4, 0.8, 1.2, 1.6, 2.0, 3.6])
GAIA_EPOCH_MJD = 57389.0            # J2016.0
J2000_MJD = 51544.5

_CURVES = None


def log(m):
    print(time.strftime("[%H:%M:%S] ") + m, flush=True)


# ---------------------------------------------------------------- stage 1: reference cache
def gather_both(ra0, dec0, radius=0.9):
    """GAIA_BEST rows near (ra0, dec0): sid, ra, dec, pm, bp-rp, G, raw XP, corrected XP."""
    T = C.TILES
    dd = np.abs(T.dec.values - dec0)
    dr = np.abs(((T.ra.values - ra0 + 180) % 360) - 180) * np.cos(np.radians(dec0))
    near = T[(dd < radius + 0.7) & (dr < radius + 0.7)]
    cosd = np.cos(np.radians(dec0))
    seen = set()
    cols = {k: [] for k in ("sid", "ra", "dec", "pmra", "pmdec", "bprp", "g", "raw", "cor")}
    for tn in near.id.values:
        p = C.TILESRC / f"{tn}.csv"
        if not p.exists():
            continue
        with open(p) as fh:
            fh.readline()
            for line in fh:
                cc = line.split(",", 15)
                try:
                    ra, dec = float(cc[1]), float(cc[2])
                except ValueError:
                    continue
                if abs(dec - dec0) > radius or abs(((ra - ra0 + 180) % 360) - 180) * cosd > radius:
                    continue
                sid = cc[0].lstrip("﻿")
                if sid in seen:
                    continue
                vo = np.fromstring(cc[11], sep=";"); vc = np.fromstring(cc[13], sep=";")
                if vo.size != 343 or vc.size != 343:
                    continue
                seen.add(sid)
                def f(x):
                    try:
                        return float(x)
                    except ValueError:
                        return np.nan
                bp, rp, g = f(cc[6]), f(cc[7]), f(cc[8])
                cols["sid"].append(sid); cols["ra"].append(ra); cols["dec"].append(dec)
                cols["pmra"].append(f(cc[4]) if cc[4] else 0.0); cols["pmdec"].append(f(cc[5]) if cc[5] else 0.0)
                cols["bprp"].append(bp - rp); cols["g"].append(g)
                cols["raw"].append(vo.astype(np.float32)); cols["cor"].append(vc.astype(np.float32))
    if not cols["sid"]:
        return None
    out = {k: np.array(v) for k, v in cols.items() if k not in ("raw", "cor")}
    out["raw"] = np.vstack(cols["raw"]); out["cor"] = np.vstack(cols["cor"])
    return out


def cache_object(args):
    key, ra0, dec0 = args
    p = CACHE / f"{key}.npz"
    if p.exists():
        return key, "cached"
    ref = gather_both(ra0, dec0)
    if ref is None:
        return key, "empty"
    np.savez(p, **ref)
    return key, f"{len(ref['sid'])} stars"


def load_cache(key):
    p = CACHE / f"{key}.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=False)
    return {k: d[k] for k in d.files}


# ---------------------------------------------------------------- stage 2: per (object, filter) group
def _init():
    global _CURVES
    warnings.filterwarnings("ignore")
    AN._init(); TW._init(40)
    _CURVES = TC.build_totals()[COMBO]


PIX = 0.505
AREA = {d: np.pi * (d / 2.0 / PIX) ** 2 for d in (5.0, 10.0, 15.0)}      # aperture areas [pix]
PEDESTAL = False


def estimate_pedestal(f_in, f_out, a_in, a_out, snr, n_iter=3):
    """per-frame additive pedestal p [flux/pix] from a pair of apertures.
    bright stars (SNR>80) fix the growth ratio g = EE_out/EE_in; faint stars (SNR 20-80) carry the pedestal:
    p = median[(F_out - g F_in) / (A_out - g A_in)], iterated with g re-measured on pedestal-corrected fluxes."""
    ok = np.isfinite(f_in) & np.isfinite(f_out) & (f_in > 0) & (f_out > 0)
    br = ok & (snr > 80); ft = ok & (snr > 20) & (snr <= 80)
    if br.sum() < 5 or ft.sum() < 8:
        return np.nan, np.nan, 0
    p = 0.0
    for _ in range(n_iter):
        g = np.median((f_out[br] - p * a_out) / (f_in[br] - p * a_in))
        est = (f_out[ft] - g * f_in[ft]) / (a_out - g * a_in)
        p = float(np.median(est))
    return p, float(g), int(ft.sum())


def clipped(v):
    v = v[np.isfinite(v)]
    if len(v) < 5:
        return np.nan, np.nan, len(v)
    cl = SigmaClip(sigma=3.0, maxiters=5)(v)
    return float(np.ma.median(cl)), float(np.ma.std(cl)), int((~np.ma.getmaskarray(cl)).sum())


def process_group(job):
    ds, obj, filt, files, star = job
    if filt not in _CURVES:
        return []
    ref = load_cache(star["cache"])
    if ref is None:
        return []
    fw, ft = _CURVES[filt]
    coords = SkyCoord(ref["ra"], ref["dec"], unit="deg")
    # truths for the standard
    t_gnd = star["spec"]
    m_gnd = float(R.syn_mag(t_gnd[0], t_gnd[1], fw, ft, 0.0)) if t_gnd is not None else np.nan
    js = star.get("ref_index")
    m_cor = float(R.syn_mag(XP_WAVE, ref["cor"][js][None, :], fw, ft, 0.0)[0]) if js is not None else np.nan
    m_raw = float(R.syn_mag(XP_WAVE, ref["raw"][js][None, :], fw, ft, 0.0)[0]) if js is not None else np.nan
    # synthetic mags of all reference stars once per group
    syn_cor_all = R.syn_mag(XP_WAVE, ref["cor"].astype(float), fw, ft, 0.0)
    syn_raw_all = R.syn_mag(XP_WAVE, ref["raw"].astype(float), fw, ft, 0.0)
    rows = []
    for fr in files:
        catd = CAL_CATD if ds == "CALSPEC" else SPSS_CATD
        cp = catd / (Path(fr).name[:-5] + "_cat.fits")
        if not cp.exists():
            continue
        try:
            with fits.open(cp) as h:
                hdr = h[0].header
                raw = h["LDAC_OBJECTS"].data
                fmax = np.asarray(raw["FLUX_MAX"], float)
                ximg = np.asarray(raw["X_IMAGE"], float); yimg = np.asarray(raw["Y_IMAGE"], float)
                m5 = np.asarray(raw["MAG_APER_1"], float); m10 = np.asarray(raw["MAG_APER_3"], float)
                m15 = np.asarray(raw["MAG_APER_4"], float)
            df = OZ.load_ez(cp) if ds == "CALSPEC" else AN.load_cat(cp)
            df["FLUX_MAX"] = fmax
            df["XI"] = ximg; df["YI"] = yimg
            df["F5"] = np.where(m5 < 90, 10 ** (-0.4 * m5), np.nan)
            df["F10"] = np.where(m10 < 90, 10 ** (-0.4 * m10), np.nan)
            df["F15"] = np.where(m15 < 90, 10 ** (-0.4 * m15), np.nan)
            sel = OT.select_stars_like(df)
            ped_p, ped_g, ped_n = estimate_pedestal(sel.F5.to_numpy(), sel.F10.to_numpy(), AREA[5.0], AREA[10.0],
                                                    sel.SNR_APER_3.to_numpy())
            ped_p15, _, _ = estimate_pedestal(sel.F10.to_numpy(), sel.F15.to_numpy(), AREA[10.0], AREA[15.0],
                                              sel.SNR_APER_3.to_numpy())
            if PEDESTAL:
                if not np.isfinite(ped_p):
                    continue
                corr = sel.F10.to_numpy() - ped_p * AREA[10.0]
                sel = sel.assign(MAG_APER_3=np.where(corr > 0, -2.5 * np.log10(np.clip(corr, 1e-30, None)), np.nan))
            if len(sel) < MIN_REF:
                continue
            src = SkyCoord(sel.ALPHA_J2000.to_numpy(), sel.DELTA_J2000.to_numpy(), unit="deg")
            idx, d2d, _ = match_coordinates_sky(src, coords)
            ok = np.asarray(d2d.arcsec < 2.0)
            if ok.sum() < MIN_REF:
                continue
            mi = idx[ok]
            inst = sel.MAG_APER_3.to_numpy()[ok]; snr = sel.SNR_APER_3.to_numpy()[ok]
            bp = ref["bprp"][mi]; g = ref["g"][mi]
            zc_all = syn_cor_all[mi] - inst; zr_all = syn_raw_all[mi] - inst
            br = snr > 40.0
            zp_cor, zsd, nz = clipped(zc_all[br])
            if not np.isfinite(zp_cor) or zsd > 0.08 or nz < MIN_REF:
                continue
            zp_raw, _, _ = clipped(zr_all[br])
            zp_blue, _, nb = clipped(zc_all[br & (bp < 0.8)])
            zp_red, _, nr = clipped(zc_all[br & (bp > 1.2)])
            zp_gbr, _, _ = clipped(zc_all[br & (g < 13.5)])
            zp_gft, _, _ = clipped(zc_all[br & (g >= 13.5)])
            row = dict(dataset=ds, object=obj, filter=filt, file=Path(fr).name,
                       unit=Path(fr).name.split("_")[0],
                       mjd=float(hdr.get("MJD", np.nan)), airmass=float(hdr.get("AIRMASS", np.nan)),
                       exptime=float(hdr.get("EXPTIME", np.nan)), humidity=float(hdr.get("HUMIDITY", np.nan) or np.nan),
                       ambtemp=float(hdr.get("AMBTEMP", np.nan) or np.nan),
                       seeing=float(np.median(sel.FWHM_WORLD.to_numpy()) * 3600.0),
                       n_zp=nz, zp_cor=zp_cor, zp_std=zsd, zp_raw=zp_raw, zp_blue=zp_blue, n_blue=nb,
                       zp_red=zp_red, n_red=nr, zp_gbright=zp_gbr, zp_gfaint=zp_gft,
                       field_c_med=float(np.median(bp[br])),
                       ped_p=ped_p, ped_p15=ped_p15, ped_g=ped_g, ped_n=ped_n)
            # field residual colour profile (SNR>20, all colours)
            s20 = (snr > 20.0) & np.isfinite(zc_all) & np.isfinite(bp)
            res = zc_all[s20] - zp_cor
            for k in range(len(BINS) - 1):
                m = (bp[s20] >= BINS[k]) & (bp[s20] < BINS[k + 1])
                row[f"rb{k}"] = float(np.median(res[m])) if m.sum() >= 3 else np.nan
                row[f"nb{k}"] = int(m.sum())
            # star-level sample (0.4<BP-RP<1.6, SNR>20) for the counts-vs-magnitude separation
            pk = sel.FLUX_MAX.to_numpy()[ok]
            xi = sel.XI.to_numpy()[ok]; yi = sel.YI.to_numpy()[ok]
            ss = np.where((snr > 20.0) & np.isfinite(zc_all) & (bp > 0.4) & (bp < 1.6) & np.isfinite(g) & (pk > 0))[0]
            if len(ss) > 80:
                ss = ss[np.linspace(0, len(ss) - 1, 80).astype(int)]
            row["_stars"] = [(float(g[k]), float(bp[k]), float(np.log10(pk[k])), float(snr[k]),
                              float(zc_all[k] - zp_cor), float(xi[k]), float(yi[k])) for k in ss]
            # the standard star
            mjd = row["mjd"]
            if ds == "CALSPEC":
                dt = (mjd - J2000_MJD) / 365.25 if np.isfinite(mjd) else 24.5
            else:
                dt = (mjd - GAIA_EPOCH_MJD) / 365.25 if np.isfinite(mjd) else 10.7
            pos = SkyCoord(star["ra"] + star["pmra"] * dt / 3.6e6 / np.cos(np.deg2rad(star["dec"])),
                           star["dec"] + star["pmdec"] * dt / 3.6e6, unit="deg")
            call = SkyCoord(df.ALPHA_J2000.to_numpy(), df.DELTA_J2000.to_numpy(), unit="deg")
            sep = call.separation(pos).arcsec
            j = int(np.argmin(sep))
            st = df.iloc[j]
            if sep[j] < 2.0 and st.FLAGS <= 1 and st.MAGERR_APER_3 < 0.05:
                m_i = float(st.MAG_APER_3)
                row["star_ped_frac"] = float(ped_p * AREA[10.0] / st.F10) if np.isfinite(ped_p) else np.nan
                if PEDESTAL:
                    fc_ = st.F10 - ped_p * AREA[10.0]
                    if not fc_ > 0:
                        rows.append(row); continue
                    m_i = float(-2.5 * np.log10(fc_))
                row.update(star_inst=m_i, star_err=float(st.MAGERR_APER_3), star_fmax=float(fmax[j]),
                           star_x=float(ximg[j]), star_y=float(yimg[j]), c_star=star["c"],
                           d_gnd=m_i + zp_cor - m_gnd, d_cor=m_i + zp_cor - m_cor, d_raw=m_i + zp_cor - m_raw,
                           d_gnd_zpraw=m_i + zp_raw - m_gnd,
                           d_gnd_zpblue=m_i + zp_blue - m_gnd, d_gnd_zpred=m_i + zp_red - m_gnd)
            rows.append(row)
        except Exception:                                    # noqa: BLE001
            continue
    return rows


def star_colour(wl, fl):
    import pyphot
    lib = pyphot.get_library()

    def vega(bn):
        f = lib[bn]; fx = f.get_flux(wl * 10.0, fl * 100.0)
        fx = fx.magnitude if hasattr(fx, "magnitude") else float(fx)
        return -2.5 * np.log10(fx) - float(f.Vega_zero_mag)
    return float(vega("Gaia_MAW_BP_bright") - vega("Gaia_MAW_RP"))


def cap_even(files, cap):
    files = sorted(files)
    if len(files) <= cap:
        return files
    return [files[i] for i in np.linspace(0, len(files) - 1, cap).astype(int)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=24)
    ap.add_argument("--cap", type=int, default=40)
    ap.add_argument("--only", nargs="*", default=None, help="limit to these filters (smoke tests)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--pedestal", action="store_true", help="subtract the per-frame 10-arcsec pedestal")
    a = ap.parse_args()
    global PEDESTAL
    PEDESTAL = a.pedestal
    TW._init(40)
    # ---- CALSPEC jobs
    cmeta = pd.read_csv(TC.CAL_META); cmeta = cmeta[cmeta.status == "ok"]
    cal_stars, jobs = {}, []
    for obj, g in cmeta.groupby("object"):
        st = TW._INFO[obj]
        if not st["spec"]:
            continue
        t = Table.read(TW.CALSPEC_DIR / st["spec"])
        wl = np.asarray(t["WAVELENGTH"], float) / 10.0; fl = np.asarray(t["FLUX"], float) * 1e-2
        m = (wl > 300) & (wl < 1100) & np.isfinite(fl) & (fl > 0)
        cal_stars[obj] = dict(cache=f"CAL_{obj}", ra=st["ra"], dec=st["dec"], pmra=st["pmra"],
                              pmdec=st["pmdec"], spec=(wl[m], fl[m]), c=star_colour(wl[m], fl[m]))
    # ---- SPSS jobs
    spss = pd.read_csv(AN.SPSS_CSV)
    srow_of = {AN.norm(s.Name): s for _, s in spss.iterrows()}
    smeta = pd.concat([pd.read_csv(p) for p in glob.glob(str(Path(AN.CATD).parent / "process_meta_7DT*.csv"))],
                      ignore_index=True)
    smeta = smeta[(smeta.status == "ok") & ~smeta.unit.isin(TC.EXCLUDE_UNITS) & ~smeta["filter"].isin(TC.BAD_FILTERS)]
    spss_stars = {}
    for obj, g in smeta.groupby("object"):
        s = srow_of.get(AN.norm(obj))
        if s is None or pd.isna(s.GaiaEDR3):
            continue
        spss_stars[obj] = dict(cache=f"SPSS_{AN.norm(obj)}", ra=float(s.RA_ICRS), dec=float(s.DE_ICRS),
                               sid=str(int(s.GaiaEDR3)), spec=AN.load_spss_spec(s.SPSS))
    # ---- stage 1: cache references
    todo = [(v["cache"], v["ra"], v["dec"]) for v in list(cal_stars.values()) + list(spss_stars.values())]
    log(f"stage 1: caching {len(todo)} reference fields")
    with Pool(min(a.n_proc, len(todo))) as pool:
        for key, msg in pool.imap_unordered(cache_object, todo):
            log(f"  {key}: {msg}")
    # attach the standard's own row in the cache (XP truths + Gaia position/PM for SPSS)
    for obj, v in spss_stars.items():
        ref = load_cache(v["cache"])
        j = np.where(ref["sid"] == v["sid"])[0] if ref is not None else []
        if not len(j):
            v["ref_index"] = None; v.update(pmra=0.0, pmdec=0.0, c=np.nan)
            continue
        j = int(j[0])
        v.update(ref_index=j, ra=float(ref["ra"][j]), dec=float(ref["dec"][j]),
                 pmra=float(ref["pmra"][j]), pmdec=float(ref["pmdec"][j]), c=float(ref["bprp"][j]))
    for obj, v in cal_stars.items():
        ref = load_cache(v["cache"])
        v["ref_index"] = None
        if ref is None:
            continue
        dt = (GAIA_EPOCH_MJD - J2000_MJD) / 365.25
        ra16 = v["ra"] + v["pmra"] * dt / 3.6e6 / np.cos(np.deg2rad(v["dec"]))
        de16 = v["dec"] + v["pmdec"] * dt / 3.6e6
        sep = np.hypot((ref["ra"] - ra16) * np.cos(np.deg2rad(v["dec"])), ref["dec"] - de16) * 3600
        if sep.min() < 3.0:
            v["ref_index"] = int(np.argmin(sep))
    for obj, g in cmeta.groupby("object"):
        if obj not in cal_stars:
            continue
        for filt, gf in g.groupby("filter"):
            if a.only and filt not in a.only:
                continue
            jobs.append(("CALSPEC", obj, filt, cap_even(gf.file.tolist(), a.cap), cal_stars[obj]))
    for obj, g in smeta.groupby("object"):
        if obj not in spss_stars:
            continue
        for filt, gf in g.groupby("filter"):
            if a.only and filt not in a.only:
                continue
            jobs.append(("SPSS", obj, filt, cap_even(gf.file.tolist(), a.cap), spss_stars[obj]))
    jobs.sort(key=lambda j: -len(j[3]))
    log(f"stage 2: {len(jobs)} (object, filter) groups, {sum(len(j[3]) for j in jobs)} frames")
    rows = []
    with Pool(a.n_proc, initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(process_group, jobs), 1):
            rows += r
            if i % 50 == 0 or i == len(jobs):
                log(f"  {i}/{len(jobs)} groups, {len(rows)} frames")
    star_rows = []
    for i, r in enumerate(rows):
        for (gg, cc, lp, sn, rs, xx, yy) in r.pop("_stars", []):
            star_rows.append((i, r["dataset"], r["filter"], r["unit"], r["exptime"], gg, cc, lp, sn, rs, xx, yy))
    pd.DataFrame(star_rows, columns=["frame_i", "dataset", "filter", "unit", "exptime", "g", "bprp", "logpeak",
                                     "snr", "res", "x", "y"]).to_csv(OUT / f"stepA_fieldstars{a.tag}.csv", index=False)
    fr = pd.DataFrame(rows)
    fr.to_csv(OUT / f"stepA_frames{a.tag}.csv", index=False)
    log(f"saved -> stepA_frames{a.tag}.csv ({len(fr)} rows, {fr['d_gnd'].notna().sum()} with standard)")


if __name__ == "__main__":
    main()
