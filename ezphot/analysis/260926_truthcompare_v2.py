#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260926_truthcompare_v2.py — T1 (standards vs three truths) redone with the Step 1 conventions.

Changes vs 260921_truthcompare.py (same images, same frame cap, same truths, same curve versions):
  field ZP stars : 260926_delta_star_v2.select_stars_v2 (FLAGS 0, SNR_10" >= 100, FLUX_MAX <= 50k, isolated 10",
                   no shape cuts) instead of select_stars_like + SNR > 40
  curves         : vacuum wavelengths (lambda_vac = n lambda_air); atmosphere interpolated on its lam_air column
  per frame      : also records the standard's SNR and FLUX_MAX, the ZP scatter and star count, and the field colour
                   term a (ZP vs BP-RP, same fit as OT.solve_offset at delta = 0) for screening wrong-filter images
  summary        : per-star values with |Delta| > 0.5 mag dropped (broken truth spectrum); frames kept if the standard has SNR >= 100 and FLUX_MAX <= 50000 and the field colour term is within
                   0.1 mag/mag of its (dataset, filter, version, variant) median; the old-style cut (standard
                   MAGERR < 0.05 only) is kept as a second summary for comparison
Outputs (~/ezphot/log/20260918_spss/): truthcompare_v2_frames.csv, truthcompare_v2_summary.csv,
truthcompare_v2_summary_loose.csv
  MPLCONFIGDIR=/tmp/ezphot-matplotlib OMP_NUM_THREADS=1 python -u ezphot/analysis/260926_truthcompare_v2.py --n-proc 16
"""
from __future__ import annotations

import argparse
import glob
import importlib
import sys
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
DS = importlib.import_module("260926_delta_star_v2")
AN, TW, OZ, K, R = TC.AN, TC.TW, TC.OZ, TC.K, TC.R
XP_WAVE = TC.XP_WAVE
OUT = TC.OUT
MIN_REF = 10
STD_SNR, PEAK_MAX, A_TOL, D_MAX = 100.0, 50000.0, 0.1, 0.5
_TOTALS = None


def build_totals():
    atm = pd.read_csv(TC.TS / "atmosphere_elsauce.csv")
    qe = pd.read_csv(TC.SRC / "QE.csv")
    opt = pd.read_csv(TC.SRC / "optics.efficiency.dr350.dr500.csv", sep=r"\s+", comment="#", skiprows=11)
    out = {}
    for ver in TC.VERSIONS:
        out[(ver, "nobeam")], out[(ver, "beam")] = {}, {}
        for p in sorted((TC.VBASE / ver).glob("m*.csv")):
            f = p.stem
            if f in TC.BAD_FILTERS:
                continue
            d = pd.read_csv(p)
            lam = d.lam_nm.to_numpy(float); tr = d.trans.to_numpy(float)
            env = (np.interp(lam, qe.iloc[:, 0], qe.iloc[:, 1]) * np.interp(lam, opt.iloc[:, 0], opt.iloc[:, -1])
                   * np.interp(lam, atm["lam_air"], atm["trans"]))
            lv = DS.air_to_vac_nm(lam)
            out[(ver, "nobeam")][f] = (lv, tr * env)
            out[(ver, "beam")][f] = (lv, TC.dr500_transmission(lam, tr, n_eff=1.5, f_number=3.03, obscuration=0.59) * env)
    return out


def _init():
    global _TOTALS
    warnings.filterwarnings("ignore")
    AN._init()
    TW._init(40)
    _TOTALS = build_totals()


def load_full(cp):
    df = OZ.load_ez(cp)
    with fits.open(cp) as h:
        d = h["LDAC_OBJECTS"].data
        df["FLUX_MAX"] = np.asarray(d["FLUX_MAX"], float) if "FLUX_MAX" in d.columns.names else np.nan
        mjd = float(h[0].header.get("MJD", np.nan))
    return df, mjd


def frame_deltas(filt, fc, inst, bprp, sig, star_inst, truth_specs):
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
        a = np.nan
        ok = keep[np.isfinite(bprp[keep])]
        if len(ok) >= MIN_REF:
            a = float(R.weighted_linfit(bprp[ok], zp_arr[ok] - np.median(zp_arr[ok]), sig[ok])[0])
        row = dict(version=ver, variant=var, zp=round(zp, 4), zsd=round(zsd, 4), n_zp=int(len(keep)), a_field=round(a, 5))
        for tag, spec in truth_specs.items():
            if spec is None:
                continue
            m_t = (float(R.syn_mag(XP_WAVE, spec[None, :], fw, ft, 0.0)[0]) if tag in ("raw", "cor")
                   else float(R.syn_mag(spec[0], spec[1], fw, ft, 0.0)))
            row[f"d_{tag}"] = round(star_inst + zp - m_t, 4)
        rows.append(row)
    return rows


def _std_row(df, pos, maxsep):
    call = SkyCoord(df.ALPHA_J2000.to_numpy(), df.DELTA_J2000.to_numpy(), unit="deg")
    sep = call.separation(pos).arcsec
    js = int(np.argmin(sep))
    st = df.iloc[js]
    if not (sep[js] < maxsep and st.FLAGS <= 1 and st.MAGERR_APER_3 < 0.05):
        return None
    return st


def process_spss(args):
    obj, files, srow = args
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
    for fr in files:
        cp = AN.CATD / (Path(fr).name[:-5] + "_cat.fits")
        filt = Path(fr).name.split("_")[-4]
        unit = Path(fr).name.split("_")[0]
        try:
            df, mjd = load_full(cp)
            sel = DS.select_stars_v2(cp)
            if len(sel) < MIN_REF:
                continue
            idx, d2d, _ = match_coordinates_sky(SkyCoord(sel.ALPHA_J2000.to_numpy(), sel.DELTA_J2000.to_numpy(), unit="deg"), coords)
            ok = np.asarray(d2d.arcsec < 2.0)
            if ok.sum() < MIN_REF:
                continue
            mi = idx[ok]
            dt = (mjd - AN.GAIA_EPOCH_MJD) / 365.25 if np.isfinite(mjd) else 10.7
            pos = SkyCoord(ra_s + pma * dt / 3.6e6 / np.cos(np.deg2rad(dec_s)), dec_s + pmd * dt / 3.6e6, unit="deg")
            st = _std_row(df, pos, 2.0)
            if st is None:
                continue
            sig = np.sqrt(sel.MAGERR_APER_3.to_numpy()[ok] ** 2 + 0.01 ** 2)
            for r in frame_deltas(filt, ref["flux"][mi], sel.MAG_APER_3.to_numpy()[ok], ref["bprp"][mi], sig,
                                  float(st.MAG_APER_3), truth):
                rows.append(dict(dataset="SPSS", object=obj, filter=filt, unit=unit, file=Path(fr).name, mjd=mjd,
                                 std_snr=float(st.SNR_APER_3), std_peak=float(st.FLUX_MAX), **r))
        except Exception:                                    # noqa: BLE001
            continue
    return rows


def process_cal(args):
    obj, files = args
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
    rows, need, prepped = [], set(), []
    for fr in files:
        cp = TC.CAL_CATD / (Path(fr).name[:-5] + "_cat.fits")
        if not cp.exists():
            continue
        filt, unit = fr.split("/")[-2], fr.split("/")[-3]
        try:
            df, mjd = load_full(cp)
            sel = DS.select_stars_v2(cp)
            if len(sel) < MIN_REF:
                continue
            idx, d2d, _ = match_coordinates_sky(SkyCoord(sel.ALPHA_J2000.to_numpy(), sel.DELTA_J2000.to_numpy(), unit="deg"), coords)
            ok = np.asarray(d2d.arcsec < 2.0)
            if ok.sum() < MIN_REF:
                continue
            mr = tbl[idx[ok]]
            dt = (mjd - 51544.5) / 365.25 if np.isfinite(mjd) else 24.5
            pos = SkyCoord(st0["ra"] + st0["pmra"] * dt / 3.6e6 / np.cos(np.deg2rad(st0["dec"])),
                           st0["dec"] + st0["pmdec"] * dt / 3.6e6, unit="deg")
            st = _std_row(df, pos, 1.5)
            if st is None:
                continue
            prepped.append(dict(filt=filt, unit=unit, file=Path(fr).name, mjd=mjd, rid=np.asarray(mr["id"]).astype(str),
                                bprp=np.asarray(mr["bp-rp"], float), inst=sel.MAG_APER_3.to_numpy()[ok],
                                sig=np.sqrt(sel.MAGERR_APER_3.to_numpy()[ok] ** 2 + 0.01 ** 2),
                                star_inst=float(st.MAG_APER_3), std_snr=float(st.SNR_APER_3), std_peak=float(st.FLUX_MAX)))
            need.update(prepped[-1]["rid"])
        except Exception:                                    # noqa: BLE001
            continue
    if not prepped:
        return []
    xpf = TW.tile_xp_radec(st0["ra"], st0["dec"], need)
    for pr in prepped:
        keep = np.array([r in xpf for r in pr["rid"]])
        if keep.sum() < MIN_REF:
            continue
        fc = np.vstack([xpf[r] for r in pr["rid"][keep]])
        for r in frame_deltas(pr["filt"], fc, pr["inst"][keep], pr["bprp"][keep], pr["sig"][keep], pr["star_inst"], truth):
            rows.append(dict(dataset="CALSPEC", object=obj, filter=pr["filt"], unit=pr["unit"], file=pr["file"], mjd=pr["mjd"],
                             std_snr=pr["std_snr"], std_peak=pr["std_peak"], **r))
    return rows


def summarise(fr):
    res = []
    for (ds, f, ver, var), g in fr.groupby(["dataset", "filter", "version", "variant"]):
        row = dict(dataset=ds, filter=f, version=ver, variant=var, pivot=int(f[1:4]), n_frames=len(g), n_stars=g.object.nunique())
        for tag in ("gnd", "cor", "raw"):
            c = f"d_{tag}"
            if c not in g.columns:
                continue
            gg = g.dropna(subset=[c])
            gg = gg[gg[c].abs() <= D_MAX]                # |Delta| > 0.5 mag = broken truth spectrum (seen in raw XP, red)
            per = gg.groupby("object")[c].median()
            if len(per) < 2:
                continue
            row[f"{tag}_mmag"] = round(1e3 * float(per.median()), 1)
            row[f"{tag}_err"] = round(1e3 * 1.4826 * float((per - per.median()).abs().median()) / np.sqrt(len(per)), 1)
            row[f"{tag}_n"] = len(per)
        res.append(row)
    return pd.DataFrame(res).sort_values(["dataset", "pivot", "version", "variant"])


def write_summaries(fr):
    med = fr.groupby(["dataset", "filter", "version", "variant"]).a_field.transform("median")
    good_a = (fr.a_field - med).abs() <= A_TOL
    strict = fr[good_a & (fr.std_snr >= STD_SNR) & (fr.std_peak <= PEAK_MAX)]
    TC.log(f"frames x versions: all {len(fr)}, colour-term screen {int(good_a.sum())}, + standard SNR>=100 & peak<=50k {len(strict)}")
    S = summarise(strict); S.to_csv(OUT / "truthcompare_v2_summary.csv", index=False)
    L = summarise(fr[good_a]); L.to_csv(OUT / "truthcompare_v2_summary_loose.csv", index=False)
    return S


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n-proc", type=int, default=16)
    ap.add_argument("--summary-only", action="store_true", help="rebuild the summaries from truthcompare_v2_frames.csv")
    a = ap.parse_args()
    if a.summary_only:
        write_summaries(pd.read_csv(OUT / "truthcompare_v2_frames.csv"))
        TC.log("saved")
        return
    spss = pd.read_csv(AN.SPSS_CSV)
    srow_of = {AN.norm(s.Name): s for _, s in spss.iterrows()}
    meta = pd.concat([pd.read_csv(m) for m in glob.glob(str(OUT / "process_meta_7DT*.csv"))], ignore_index=True)
    meta = meta[(meta.status == "ok") & ~meta.unit.isin(TC.EXCLUDE_UNITS) & ~meta["filter"].isin(TC.BAD_FILTERS)]
    jobs_s = [(obj, TC.cap_frames(list(zip(g["filter"], g.file))), srow_of[AN.norm(obj)])
              for obj, g in meta.groupby("object") if AN.norm(obj) in srow_of]
    cmeta = pd.read_csv(TC.CAL_META)
    cmeta = cmeta[cmeta.status == "ok"]
    jobs_c = [(obj, TC.cap_frames([(f.split("/")[-2], f) for f in g.file])) for obj, g in cmeta.groupby("object")]
    TC.log(f"SPSS: {len(jobs_s)} objects; CALSPEC: {len(jobs_c)} objects (cap {TC.FRAME_CAP}/filter)")
    rows = []
    with Pool(a.n_proc, initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(process_spss, jobs_s), 1):
            rows += r
        TC.log(f"  SPSS done, {len(rows)} rows")
        for i, r in enumerate(pool.imap_unordered(process_cal, jobs_c), 1):
            rows += r
        TC.log(f"  CALSPEC done, {len(rows)} rows")
    fr = pd.DataFrame(rows)
    fr.to_csv(OUT / "truthcompare_v2_frames.csv", index=False)
    S = write_summaries(fr)
    pd.set_option("display.width", 220)
    print(S[(S.version == "manufacturer")].to_string(index=False))
    TC.log("saved")


if __name__ == "__main__":
    main()
