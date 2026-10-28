#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260927_std_offset.py — CALSPEC and SPSS magnitude offsets from scratch, on the re-photometered catalogs
(260927_std_rephot.py: fixed background = per-mesh 5-sigma mean after source masking).

For every standard-star frame (all frames, no cap):
  field ZP  : Step 1 star selection (select_stars_v2: FLAGS 0, SNR_10" >= 100, FLUX_MAX <= 50k ADU, isolated 10" against
              all detections, no shape cuts), corrected Gaia XP match < 2"; ZP = 3-sigma-clipped median of
              syn_mag(corrected XP, curve) - MAG_APER_3, kept if clipped scatter <= 0.08 and >= 10 stars; field colour
              term a (ZP vs BP-RP) recorded for screening wrong-filter frames; the standard's image position is recorded
  standard  : nearest detection to the proper-motion-corrected position (CALSPEC < 1.5", SPSS < 2"); its SNR, FLUX_MAX,
              peak above sky in e-, FLAGS are recorded (cuts are applied in the summary, not here)
  offset    : Delta_truth = MAG_APER_3(std) + ZP - syn_mag(truth spectrum, curve) for truth = ground spectrum (CALSPEC /
              SPSS V2 table), corrected XP, raw XP.  Three ZPs: global (all ZP stars; column d_*), local (ZP stars within
              R_LOC px of the standard, >= 5 stars; dl_*) and radial (per-frame fit ZP = c0 + c1 r^2 over the ZP stars,
              r = distance from the image centre, evaluated at the standard; dr_*).  The standards sit near the image
              centre while the field stars fill the frame, and the flat-field leaves a radial ZP pattern that differs
              between epochs (radial_check: centre - median = -2...-15 mmag in 2024, +1...+7 mmag in 2026 frames).
  curves    : manufacturer, version_mhj (old-gen filters), version_hyp, version_ky (new-gen) x (no beam | f/3 beam
              n_eff 1.5) x QE x optics x El Sauce atmosphere, vacuum wavelengths (atmosphere on its lam_air column)
  colour term: a_field = weighted slope of ZP_i vs BP-RP (the standard itself can be one of the ZP stars; a_field, zp,
              zp_loc keep it).  Without the standard (ZP stars > 2 px from it): a_x, zp_loc_x / n_loc_x, bprp0 = median
              BP-RP of the fit stars, zg_c0 / zl_c0 = global / local median of ZP_i - a_x (BP-RP_i - bprp0), i.e. the ZP at
              colour bprp0 (the ZP at the standard's colour is zl_c0 + a_x (BP-RP_std - bprp0)); zp_x = global ZP without the
              standard (median of the clipped ZP stars > 2 px from it)
  --mag AUTO / APER_4 : the same star selection (10" SNR, FLAGS, peak, isolation) but MAG_AUTO / the 15" aperture for
              the field stars and the standard
Output (~/ezphot/log/20260927_std_newbkg/): std_offset_frames.csv (--mag X: std_offset_frames_<x>.csv)
  MPLCONFIGDIR=/tmp/ezphot-matplotlib OMP_NUM_THREADS=1 python -u ezphot/analysis/260927_std_offset.py --n-proc 48
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from astropy.coordinates import SkyCoord, match_coordinates_sky   # noqa: E402
from astropy.io import fits                                      # noqa: E402
from astropy.stats import SigmaClip                              # noqa: E402
from astropy.table import Table                                  # noqa: E402

sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
TC = importlib.import_module("260921_truthcompare")
V2 = importlib.import_module("260926_truthcompare_v2")
DS = importlib.import_module("260926_delta_star_v2")
AN, TW, OZ, K, R = TC.AN, TC.TW, TC.OZ, TC.K, TC.R
XP_WAVE = TC.XP_WAVE
OUT = Path("/home/hhchoi1022/ezphot/log/20260927_std_newbkg")
CATD = OUT / "catalogs"
MIN_REF = 10
R_LOC, MIN_LOC = 1500.0, 5
_TOTALS = None
MAG = "MAG_APER_3"


def log(m):
    print(time.strftime("[%H:%M:%S] ") + m, flush=True)


def _init(mag="MAG_APER_3"):
    global _TOTALS, MAG
    MAG = mag
    warnings.filterwarnings("ignore")
    AN._init()
    TW._init(40)
    _TOTALS = V2.build_totals()


def load_cat(cp):
    df = OZ.load_ez(cp)
    with fits.open(cp) as h:
        d = h["LDAC_OBJECTS"].data
        names = d.columns.names
        df["FLUX_MAX"] = np.asarray(d["FLUX_MAX"], float)
        df["BACKGROUND"] = np.asarray(d["BACKGROUND"], float) if "BACKGROUND" in names else np.nan
        df["X"], df["Y"] = np.asarray(d["X_IMAGE"], float), np.asarray(d["Y_IMAGE"], float)
        for c in ("AUTO", "APER_4"):
            df["MAG_" + c], df["MAGERR_" + c] = np.asarray(d["MAG_" + c], float), np.asarray(d["MAGERR_" + c], float)
        hdr = h[0].header
    meta = dict(mjd=float(hdr.get("MJD", np.nan)), exptime=float(hdr.get("EXPTIME", np.nan)),
                airmass=float(hdr.get("AIRMASS", np.nan)), egain=float(hdr.get("EGAIN", np.nan)),
                bkgmean=float(hdr.get("BKGMEAN", np.nan)), nx=float(hdr.get("NAXIS1", 9576)), ny=float(hdr.get("NAXIS2", 6388)))
    return df, meta


def select_zp_stars(df):
    """select_stars_v2 on an already loaded catalog, keeping X/Y: FLAGS 0, SNR >= 100, FLUX_MAX <= 50k, isolated 10\"."""
    import astropy.units as u
    from astropy.coordinates import search_around_sky
    allc = SkyCoord(df.ALPHA_J2000.to_numpy(), df.DELTA_J2000.to_numpy(), unit="deg")
    q = (df.FLAGS.to_numpy() == 0) & (df.SNR_APER_3.to_numpy() >= DS.SNR_MIN) & (df.FLUX_MAX.to_numpy() <= DS.PEAK_MAX) \
        & (df.MAG_APER_3.to_numpy() < 90)
    if q.sum() < MIN_REF:
        return df.iloc[[]]
    ig, _, s2, _ = search_around_sky(allc[q], allc, DS.ISO_ARCSEC * u.arcsec)
    nbr = np.bincount(ig[s2.arcsec > 0.5], minlength=int(q.sum()))
    return df[q][nbr == 0].reset_index(drop=True)


def frame_rows(filt, fc, inst, bprp, sig, st, truth, xy, meta):
    sc = SigmaClip(sigma=3.0, maxiters=5)
    rows = []
    for (ver, var), curves in _TOTALS.items():
        if filt not in curves:
            continue
        fw, ft = curves[filt]
        zp_arr = R.syn_mag(XP_WAVE, fc, fw, ft, 0.0) - inst
        fin = np.isfinite(zp_arr)
        if fin.sum() < MIN_REF:
            continue
        cl = sc(zp_arr[fin])
        zp, zsd = float(np.ma.median(cl)), float(np.ma.std(cl))
        if not np.isfinite(zp) or zsd > 0.08:
            continue
        keep = np.where(fin)[0][~np.ma.getmaskarray(cl)]
        # local and radial ZPs at the standard's position
        dist = np.hypot(xy[keep, 0] - st.X, xy[keep, 1] - st.Y)
        near = keep[dist < R_LOC]
        zp_loc = float(np.median(zp_arr[near])) if len(near) >= MIN_LOC else np.nan
        r2 = ((xy[keep, 0] - meta["nx"] / 2) ** 2 + (xy[keep, 1] - meta["ny"] / 2) ** 2) / 1e6
        c1, c0 = np.polyfit(r2, zp_arr[keep], 1)
        r2s = ((st.X - meta["nx"] / 2) ** 2 + (st.Y - meta["ny"] / 2) ** 2) / 1e6
        zp_rad = float(c0 + c1 * r2s)
        ok = keep[np.isfinite(bprp[keep])]
        a = float(R.weighted_linfit(bprp[ok], zp_arr[ok] - np.median(zp_arr[ok]), sig[ok])[0]) if len(ok) >= MIN_REF else np.nan
        # the same without the standard itself (a high-leverage point at an extreme colour)
        sep = np.hypot(xy[:, 0] - st.X, xy[:, 1] - st.Y)
        keep_x = keep[sep[keep] > 2.0]
        zp_x = float(np.median(zp_arr[keep_x])) if len(keep_x) >= MIN_REF else np.nan
        near_x = near[sep[near] > 2.0]
        zp_loc_x = float(np.median(zp_arr[near_x])) if len(near_x) >= MIN_LOC else np.nan
        ox = ok[sep[ok] > 2.0]
        a_x = float(R.weighted_linfit(bprp[ox], zp_arr[ox] - np.median(zp_arr[ox]), sig[ox])[0]) if len(ox) >= MIN_REF else np.nan
        b0 = zg_c0 = zl_c0 = np.nan
        if np.isfinite(a_x):
            b0 = float(np.median(bprp[ox]))
            zg_c0 = float(np.median(zp_arr[ox] - a_x * (bprp[ox] - b0)))
            oxn = ox[sep[ox] < R_LOC]
            zl_c0 = float(np.median(zp_arr[oxn] - a_x * (bprp[oxn] - b0))) if len(oxn) >= MIN_LOC else np.nan
        row = dict(version=ver, variant=var, zp=round(zp, 5), zsd=round(zsd, 4), n_zp=int(len(keep)), a_field=round(a, 5),
                   zp_loc=round(zp_loc, 5), n_loc=int(len(near)), zp_rad=round(zp_rad, 5), rad_c1=round(1e3 * c1, 3),
                   n_std_zp=int(len(ok) - len(ox)), a_x=round(a_x, 5), zp_x=round(zp_x, 5), zp_loc_x=round(zp_loc_x, 5),
                   n_loc_x=int(len(near_x)),
                   bprp0=round(b0, 4), zg_c0=round(zg_c0, 5), zl_c0=round(zl_c0, 5))
        for tag, spec in truth.items():
            if spec is None:
                continue
            m_t = (float(R.syn_mag(XP_WAVE, spec[None, :], fw, ft, 0.0)[0]) if tag in ("raw", "cor")
                   else float(R.syn_mag(spec[0], spec[1], fw, ft, 0.0)))
            row[f"d_{tag}"] = round(float(st[MAG]) + zp - m_t, 5)
            row[f"dl_{tag}"] = round(float(st[MAG]) + zp_loc - m_t, 5) if np.isfinite(zp_loc) else np.nan
            row[f"dr_{tag}"] = round(float(st[MAG]) + zp_rad - m_t, 5)
        rows.append(row)
    return rows


def std_info(df, pos, maxsep, meta):
    call = SkyCoord(df.ALPHA_J2000.to_numpy(), df.DELTA_J2000.to_numpy(), unit="deg")
    sep = call.separation(pos).arcsec
    js = int(np.argmin(sep))
    st = df.iloc[js]
    if not (sep[js] < maxsep and st[MAG] < 90 and np.isfinite(st[MAG])):
        return None, None
    info = dict(std_sep=round(float(sep[js]), 3), std_mag=float(st[MAG]), std_snr=float(st.SNR_APER_3),
                std_peak=float(st.FLUX_MAX), std_peak_e=float(st.FLUX_MAX) * meta["egain"], std_flags=int(st.FLAGS),
                std_bkg=float(st.BACKGROUND), std_x=float(st.X), std_y=float(st.Y),
                std_rad=float(np.hypot(st.X - meta["nx"] / 2, st.Y - meta["ny"] / 2)))
    return st, info


def process_spss(args):
    obj, frames, srow = args
    spec = AN.load_spss_spec(srow["SPSS"])
    ref = AN.gather_reference(srow["RA_ICRS"], srow["DE_ICRS"])
    if ref is None or spec is None:
        return []
    sid = str(int(srow["GaiaEDR3"])) if pd.notna(srow["GaiaEDR3"]) else ""
    j = np.where(ref["sid"] == sid)[0]
    if not len(j):
        return []
    j = int(j[0])
    xp = TC.star_xp_by_pos(ref["ra"][j], ref["dec"][j], ref["pmra"][j], ref["pmdec"][j])
    truth = {"gnd": spec, "cor": ref["flux"][j], "raw": xp[0] if xp else None}
    ra_s, dec_s, pma, pmd = ref["ra"][j], ref["dec"][j], ref["pmra"][j], ref["pmdec"][j]
    coords = SkyCoord(ref["ra"], ref["dec"], unit="deg")
    rows = []
    for fr, filt, unit in frames:
        cp = CATD / (Path(fr).name[:-5] + "_cat.fits")
        try:
            df, meta = load_cat(cp)
            sel = select_zp_stars(df)
            if len(sel) < MIN_REF:
                continue
            idx, d2d, _ = match_coordinates_sky(SkyCoord(sel.ALPHA_J2000.to_numpy(), sel.DELTA_J2000.to_numpy(), unit="deg"), coords)
            ok = np.asarray(d2d.arcsec < 2.0)
            if ok.sum() < MIN_REF:
                continue
            mi = idx[ok]
            dt = (meta["mjd"] - AN.GAIA_EPOCH_MJD) / 365.25 if np.isfinite(meta["mjd"]) else 10.7
            pos = SkyCoord(ra_s + pma * dt / 3.6e6 / np.cos(np.deg2rad(dec_s)), dec_s + pmd * dt / 3.6e6, unit="deg")
            st, info = std_info(df, pos, 2.0, meta)
            if st is None:
                continue
            sig = np.sqrt(sel["MAGERR" + MAG[3:]].to_numpy()[ok] ** 2 + 0.01 ** 2)
            xy = np.c_[sel.X.to_numpy()[ok], sel.Y.to_numpy()[ok]]
            for r in frame_rows(filt, ref["flux"][mi], sel[MAG].to_numpy()[ok], ref["bprp"][mi], sig, st, truth, xy, meta):
                rows.append(dict(dataset="SPSS", object=obj, filter=filt, unit=unit, file=Path(fr).name, **meta, **info, **r))
        except Exception:                                    # noqa: BLE001
            continue
    return rows


def process_cal(args):
    obj, frames = args
    st0 = TW._INFO[obj]
    if not st0["spec"]:
        return []
    t = Table.read(TW.CALSPEC_DIR / st0["spec"])
    wl = np.asarray(t["WAVELENGTH"], float) / 10.0
    fl = np.asarray(t["FLUX"], float) * 1e-2
    m = (wl > 300) & (wl < 1100) & np.isfinite(fl) & (fl > 0)
    got = K.get_reference("GAIAXP_CORR_LAMOST", obj, st0["ra"], st0["dec"], 1.34, 0.90)
    if got is None:
        return []
    tbl, coords, _ = got
    xp = TC.star_xp_by_pos(st0["ra"], st0["dec"], st0["pmra"], st0["pmdec"])
    truth = {"gnd": (wl[m], fl[m]), "cor": xp[1] if xp else None, "raw": xp[0] if xp else None}
    prepped, need = [], set()
    for fr, filt, unit in frames:
        cp = CATD / (Path(fr).name[:-5] + "_cat.fits")
        try:
            df, meta = load_cat(cp)
            sel = select_zp_stars(df)
            if len(sel) < MIN_REF:
                continue
            idx, d2d, _ = match_coordinates_sky(SkyCoord(sel.ALPHA_J2000.to_numpy(), sel.DELTA_J2000.to_numpy(), unit="deg"), coords)
            ok = np.asarray(d2d.arcsec < 2.0)
            if ok.sum() < MIN_REF:
                continue
            mr = tbl[idx[ok]]
            dt = (meta["mjd"] - 51544.5) / 365.25 if np.isfinite(meta["mjd"]) else 24.5
            pos = SkyCoord(st0["ra"] + st0["pmra"] * dt / 3.6e6 / np.cos(np.deg2rad(st0["dec"])),
                           st0["dec"] + st0["pmdec"] * dt / 3.6e6, unit="deg")
            st, info = std_info(df, pos, 1.5, meta)
            if st is None:
                continue
            prepped.append((filt, unit, Path(fr).name, meta, info, st, np.asarray(mr["id"]).astype(str),
                            np.asarray(mr["bp-rp"], float), sel[MAG].to_numpy()[ok],
                            np.sqrt(sel["MAGERR" + MAG[3:]].to_numpy()[ok] ** 2 + 0.01 ** 2),
                            np.c_[sel.X.to_numpy()[ok], sel.Y.to_numpy()[ok]]))
            need.update(prepped[-1][6])
        except Exception:                                    # noqa: BLE001
            continue
    if not prepped:
        return []
    xpf = TW.tile_xp_radec(st0["ra"], st0["dec"], need)
    rows = []
    for filt, unit, name, meta, info, st, rid, bprp, inst, sig, xy in prepped:
        keep = np.array([r in xpf for r in rid])
        if keep.sum() < MIN_REF:
            continue
        fc = np.vstack([xpf[r] for r in rid[keep]])
        for r in frame_rows(filt, fc, inst[keep], bprp[keep], sig[keep], st, truth, xy[keep], meta):
            rows.append(dict(dataset="CALSPEC", object=obj, filter=filt, unit=unit, file=name, **meta, **info, **r))
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n-proc", type=int, default=48)
    ap.add_argument("--mag", choices=["APER_3", "APER_4", "AUTO"], default="APER_3")
    a = ap.parse_args()
    M = pd.read_csv(OUT / "rephot_meta.csv")
    M = M[M.status.isin(["ok", "cached"])]
    M = M[[(CATD / (Path(f).name[:-5] + "_cat.fits")).exists() for f in M.file]]
    spss = pd.read_csv(AN.SPSS_CSV)
    srow_of = {AN.norm(s.Name): s for _, s in spss.iterrows()}
    jobs_s, jobs_c = [], []
    for (obj, filt), g in M[M.dataset == "SPSS"].groupby(["object", "filter"]):
        if AN.norm(obj) in srow_of:
            jobs_s.append((obj, list(zip(g.file, g["filter"], g.unit)), srow_of[AN.norm(obj)]))
    for (obj, filt), g in M[M.dataset == "CALSPEC"].groupby(["object", "filter"]):
        jobs_c.append((obj, list(zip(g.file, g["filter"], g.unit))))
    log(f"frames {len(M)} ({M.groupby('dataset').size().to_dict()}); jobs SPSS {len(jobs_s)}, CALSPEC {len(jobs_c)}")
    rows = []
    with Pool(a.n_proc, initializer=_init, initargs=("MAG_" + a.mag,)) as pool:
        for r in pool.imap_unordered(process_spss, jobs_s):
            rows += r
        log(f"  SPSS done, {len(rows)} rows")
        for r in pool.imap_unordered(process_cal, jobs_c):
            rows += r
        log(f"  CALSPEC done, {len(rows)} rows")
    fr = pd.DataFrame(rows)
    fr.to_csv(OUT / ("std_offset_frames.csv" if a.mag == "APER_3" else f"std_offset_frames_{a.mag.lower()}.csv"), index=False)
    log(f"saved {len(fr)} rows, frames {fr.file.nunique()}")


if __name__ == "__main__":
    main()
