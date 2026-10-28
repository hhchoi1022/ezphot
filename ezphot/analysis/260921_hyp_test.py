#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260921_hyp_test.py — version_hyp measured curves (excluding bad m832w): closed-loop test.

For each new-filter x unit group of the SPSS campaign and for BOTH beam-applied and no-beam totals
(hyp filter curve x QE x optics x atm), calibrate with corrected XP and derive:
  - field colour terms (a_all, a_frame medians)
  - SPSS absolute offset: delta-grid fit, truth = SPSS V2 ground flux tables (all campaign stars)
  - CALSPEC absolute offset: same fit, truth = CALSPEC spectra (overlap stars observed in the campaign)

Output: hyp_test_results.csv + V2 figure.

  MPLCONFIGDIR=/tmp/ezphot-matplotlib OMP_NUM_THREADS=1 /home/hhchoi1022/miniconda3/envs/too/bin/python -u \\
      ezphot/analysis/260921_hyp_test.py --n-proc 12
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
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402
import pandas as pd                 # noqa: E402

warnings.filterwarnings("ignore")
from astropy.coordinates import SkyCoord, match_coordinates_sky   # noqa: E402
from astropy.io import fits          # noqa: E402
from astropy.stats import SigmaClip  # noqa: E402
from astropy.table import Table      # noqa: E402

sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
sys.path.insert(0, "/home/hhchoi1022/ezphot/log/20260910_transmission_sources")
from dr500_transmission import dr500_transmission   # noqa: E402
OT = importlib.import_module("260911_offset_timeseries")
AN = importlib.import_module("260919_spss_analysis")
R = OT.R
XP_WAVE = OT.XP_WAVE

OUT = Path("/home/hhchoi1022/ezphot/log/20260918_spss")
HYP = Path("/tmp/claude-10000/-lyman-data1-factory-hhchoi-code-ezphot/"
           "b1308704-c229-4e49-b19d-bd7bc334e4b7/scratchpad/version_hyp")
SRC = Path("/home/hhchoi1022/ezphot/config/common/transmission/Original_sources")
TS = Path("/home/hhchoi1022/ezphot/log/20260910_transmission_sources")
CAL = Path("/qso/data3/doabdi/intern/synthetic/calspec_data")
OVCSV = ("/tmp/claude-10000/-lyman-data1-factory-hhchoi-code-ezphot/"
         "b1308704-c229-4e49-b19d-bd7bc334e4b7/scratchpad/spss_calspec_overlap.csv")
DGRID = np.arange(-6.0, 6.01, 2.0)
EXCLUDE_UNITS = {"7DT11"}
BAND_FILTER = {"386": "m386", "438": "m438", "466": "m466w", "483": "m483", "512": "m512",
               "534": "m534", "586": "m586", "615": "m615", "661": "m661",
               "692": "m692w", "710": "m710w", "769": "m769w"}    # m832w bad, m561/m640 unobserved
MIN_REF = 10

_CURVES = {}          # variant -> {filter: (lam, total)}
_OVSPEC = None


def log(m):
    print(time.strftime("[%H:%M:%S] ") + m, flush=True)


def build_curves():
    atm = pd.read_csv(TS / "atmosphere_elsauce.csv")
    qe = pd.read_csv(SRC / "QE.csv")
    opt = pd.read_csv(SRC / "optics.efficiency.dr350.dr500.csv", sep=r"\s+", comment="#", skiprows=11)
    out = {"beam": {}, "nobeam": {}}
    for band, filt in BAND_FILTER.items():
        d = pd.read_csv(HYP / f"filter_{band}_transmission.csv")
        lam = d.iloc[:, 0].to_numpy(float)
        tr = np.clip(d.iloc[:, 1].to_numpy(float), 0, None) / 100.0
        env = (np.interp(lam, qe.iloc[:, 0], qe.iloc[:, 1])
               * np.interp(lam, opt.iloc[:, 0], opt.iloc[:, -1])
               * np.interp(lam, atm.iloc[:, 0], atm["trans"] if "trans" in atm.columns else atm.iloc[:, 1]))
        te = dr500_transmission(lam, tr, n_eff=1.5, f_number=3.03, obscuration=0.59)
        out["beam"][filt] = (lam, te * env)
        out["nobeam"][filt] = (lam, tr * env)
    return out


def _init():
    global _CURVES, _OVSPEC
    warnings.filterwarnings("ignore")
    AN._init()
    _CURVES = build_curves()
    ov = pd.read_csv(OVCSV)
    _OVSPEC = {}
    for _, s in ov.iterrows():
        p = CAL / s.calspec_spec
        if p.exists():
            t = Table.read(p)
            wl = np.asarray(t["WAVELENGTH"], float) / 10.0
            fl = np.asarray(t["FLUX"], float) * 1e-2
            m = (wl > 300) & (wl < 1100) & np.isfinite(fl) & (fl > 0)
            _OVSPEC[AN.norm(s.spss_name)] = (wl[m], fl[m])


def process_object(args):
    obj, files, srow = args
    sc = SigmaClip(sigma=3.0, maxiters=5)
    spec = AN.load_spss_spec(srow["SPSS"])
    cspec = _OVSPEC.get(AN.norm(obj))
    ref = AN.gather_reference(srow["RA_ICRS"], srow["DE_ICRS"])
    if ref is None or spec is None:
        return []
    sid = str(int(srow["GaiaEDR3"])) if pd.notna(srow["GaiaEDR3"]) else ""
    j = np.where(ref["sid"] == sid)[0]
    if not len(j):
        return []
    j = int(j[0])
    ra_s, dec_s, pma, pmd = ref["ra"][j], ref["dec"][j], ref["pmra"][j], ref["pmdec"][j]
    coords = SkyCoord(ref["ra"], ref["dec"], unit="deg")
    rows = []
    for fr in files:
        cp = AN.CATD / (Path(fr).name[:-5] + "_cat.fits")
        filt = Path(fr).name.split("_")[-4]
        unit = Path(fr).name.split("_")[0]
        if filt not in _CURVES["beam"] or unit in EXCLUDE_UNITS:
            continue
        try:
            with fits.open(cp) as h:
                hdr = h[0].header
                dd_ = h["LDAC_OBJECTS"].data
                fmax = np.asarray(dd_["FLUX_MAX"], float) if "FLUX_MAX" in dd_.columns.names else None
            df = AN.load_cat(cp)
            mjd = float(hdr.get("MJD", np.nan))
            sel = OT.select_stars_like(df)
            if len(sel) < MIN_REF:
                continue
            src = SkyCoord(sel.ALPHA_J2000.to_numpy(), sel.DELTA_J2000.to_numpy(), unit="deg")
            idx, d2d, _ = match_coordinates_sky(src, coords)
            ok = np.asarray(d2d.arcsec < 2.0)
            if ok.sum() < MIN_REF:
                continue
            mi = idx[ok]
            fc = ref["flux"][mi]
            bp = ref["bprp"][mi]
            inst = sel.MAG_APER_3.to_numpy()[ok]
            ierr = sel.MAGERR_APER_3.to_numpy()[ok]
            snr = sel.SNR_APER_3.to_numpy()[ok]
            sig = np.sqrt(ierr ** 2 + 0.01 ** 2)
            bright = snr > 40.0
            if bright.sum() < MIN_REF:
                continue
            dt = (mjd - AN.GAIA_EPOCH_MJD) / 365.25 if np.isfinite(mjd) else 10.7
            pos = SkyCoord(ra_s + pma * dt / 3.6e6 / np.cos(np.deg2rad(dec_s)),
                           dec_s + pmd * dt / 3.6e6, unit="deg")
            call = SkyCoord(df.ALPHA_J2000.to_numpy(), df.DELTA_J2000.to_numpy(), unit="deg")
            sep = call.separation(pos).arcsec
            js = int(np.argmin(sep))
            srow_ = df.iloc[js]
            sat = float(hdr.get("SATURATE", np.nan))
            fm = fmax[js] if fmax is not None else np.nan
            star_ok = (sep[js] < 2.0 and srow_.FLAGS <= 1 and srow_.MAGERR_APER_3 < 0.05
                       and (not np.isfinite(sat) or fm < 0.9 * sat))
            for variant in ("beam", "nobeam"):
                fw, ft = _CURVES[variant][filt]
                zp0a = (R.syn_mag(XP_WAVE, fc, fw, ft, 0.0) - inst)
                fin = np.isfinite(zp0a) & np.isfinite(bp)
                zsel = fin & bright
                if zsel.sum() < MIN_REF:
                    continue
                cl0 = sc(zp0a[zsel])
                zp0 = float(np.ma.median(cl0))
                if not np.isfinite(zp0) or float(np.ma.std(cl0)) > 0.08:
                    continue
                aaa, _, _, _ = R.weighted_linfit(bp[fin], zp0a[fin] - zp0, sig[fin])
                gz = ~np.ma.getmaskarray(cl0)
                aaf, _, _, _ = R.weighted_linfit(bp[zsel][gz], zp0a[zsel][gz] - zp0, sig[zsel][gz])
                row = dict(object=obj, filter=filt, unit=unit, variant=variant,
                           a_all=round(float(aaa), 4), a_frame=round(float(aaf), 4))
                if star_ok:
                    for dd in DGRID:
                        syn = R.syn_mag(XP_WAVE, fc, fw, ft, dd)
                        zp_arr = (syn - inst)[bright]
                        zp = float(np.ma.median(sc(zp_arr[np.isfinite(zp_arr)])))
                        m_sp = float(R.syn_mag(spec[0], spec[1], fw, ft, dd))
                        row[f"sp{dd:+.1f}"] = round(float(srow_.MAG_APER_3) + zp - m_sp, 4)
                        if cspec is not None:
                            m_ca = float(R.syn_mag(cspec[0], cspec[1], fw, ft, dd))
                            row[f"ca{dd:+.1f}"] = round(float(srow_.MAG_APER_3) + zp - m_ca, 4)
                rows.append(row)
        except Exception:                                            # noqa: BLE001
            continue
    return rows


def fit_offset(g, pre):
    dcols = [f"{pre}{dd:+.1f}" for dd in DGRID]
    if not all(c in g.columns for c in dcols):
        return None
    per = g.dropna(subset=[dcols[3]]).groupby("object")[dcols].median()
    sizes = g.dropna(subset=[dcols[3]]).groupby("object").size()
    per = per[sizes >= 2]
    if len(per) < 3:
        return None
    med = per.median(axis=0).to_numpy()
    mad = 1.4826 * (per - per.median(axis=0)).abs().median(axis=0).to_numpy()
    p = np.polyfit(DGRID, med, 2)
    roots = np.roots(p); roots = roots[np.isreal(roots)].real
    if not len(roots):
        return None
    dstar = float(roots[np.argmin(np.abs(roots))])
    slope = float(np.polyval(np.polyder(p), 0.0))
    err = float(np.median(mad) / np.sqrt(len(per)) / max(abs(slope), 1e-6))
    return dict(dstar=round(dstar, 2), err=round(err, 2), slope=round(slope, 5), n_stars=len(per))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n-proc", type=int, default=12)
    a = ap.parse_args()
    spss = pd.read_csv(AN.SPSS_CSV)
    srow_of = {AN.norm(s.Name): s for _, s in spss.iterrows()}
    meta = pd.concat([pd.read_csv(m) for m in glob.glob(str(OUT / "process_meta_7DT*.csv"))],
                     ignore_index=True)
    meta = meta[(meta.status == "ok") & meta["filter"].isin(BAND_FILTER.values())
                & ~meta.unit.isin(EXCLUDE_UNITS)]
    jobs = []
    for obj, g in meta.groupby("object"):
        s = srow_of.get(AN.norm(obj))
        if s is not None:
            jobs.append((obj, sorted(g.file), s))
    log(f"{len(meta)} frames, {len(jobs)} objects, filters: {sorted(meta['filter'].unique())}")
    rows = []
    with Pool(min(a.n_proc, len(jobs)), initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(process_object, jobs), 1):
            rows += r
            log(f"  [{i}/{len(jobs)}] total {len(rows)}")
    fr = pd.DataFrame(rows)
    fr.to_csv(OUT / "hyp_test_frames.csv", index=False)

    res = []
    for (f, u, v), g in fr.groupby(["filter", "unit", "variant"]):
        row = dict(filter=f, unit=u, variant=v, pivot=int(f[1:4]), n_frames=len(g),
                   a_all=round(g.a_all.median(), 4), a_frame=round(g.a_frame.median(), 4))
        for pre, tag in [("sp", "spss"), ("ca", "calspec")]:
            r = fit_offset(g, pre)
            if r:
                row[f"d_{tag}"] = r["dstar"]; row[f"e_{tag}"] = r["err"]
                row[f"slope_{tag}"] = r["slope"]; row[f"n_{tag}"] = r["n_stars"]
        res.append(row)
    T = pd.DataFrame(res).sort_values(["pivot", "variant"])
    T.to_csv(OUT / "hyp_test_results.csv", index=False)
    print(T.to_string(index=False))
    log("saved -> hyp_test_results.csv, hyp_test_frames.csv")


if __name__ == "__main__":
    main()
