#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260927_std_faintcheck.py — are stars with SNR 20-100 unbiased on the re-photometered standard-star catalogs?

For a sample of standard-star frames (260927_std_rephot.py catalogs, new background), field stars with FLAGS 0,
FLUX_MAX <= 50k ADU, isolated 10" against all detections, SNR_10" >= 15 and a corrected-XP match < 2" get
ZP_i = syn_mag(corrected XP, manufacturer curve, no beam, vacuum) - MAG_APER_3.  Per frame the residual
ZP_i - median(ZP of SNR >= 100 stars) is binned in SNR and in peak above sky [e-] (FLUX_MAX x EGAIN).
A positive residual = the star measured too bright relative to the bright stars (pedestal-like), negative = too faint
(low-signal loss).  Output (~/ezphot/log/20260927_std_newbkg/): faintcheck_stars.csv, faintcheck_bins.csv
  python 260927_std_faintcheck.py [n_frames_per_dataset] [AUTO|APER_4]   (MAG_AUTO / 15" in place of MAG_APER_3, same
                                                                          selection, outputs faintcheck_*_<auto|aper_4>.csv)
"""
import importlib
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from astropy.coordinates import SkyCoord, match_coordinates_sky, search_around_sky   # noqa: E402
import astropy.units as u                                                             # noqa: E402

sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
SO = importlib.import_module("260927_std_offset")
TC, AN, TW, K, R, OZ = SO.TC, SO.AN, SO.TW, SO.K, SO.R, SO.OZ
OUT, CATD = SO.OUT, SO.CATD
_C = {}
MAG = "MAG_" + (sys.argv[2] if len(sys.argv) > 2 else "APER_3")
SFX = "" if MAG == "MAG_APER_3" else "_" + MAG[4:].lower()


def _init():
    warnings.filterwarnings("ignore")
    AN._init(); TW._init(40)
    _C["curves"] = SO.V2.build_totals()[("manufacturer", "nobeam")]


def frame_stars(args):
    dataset, fr, obj, filt = args
    cp = CATD / (Path(fr).name[:-5] + "_cat.fits")
    try:
        df, meta = SO.load_cat(cp)
        allc = SkyCoord(df.ALPHA_J2000.to_numpy(), df.DELTA_J2000.to_numpy(), unit="deg")
        q = (df.FLAGS.to_numpy() == 0) & (df.FLUX_MAX.to_numpy() <= 50000) & (df.SNR_APER_3.to_numpy() >= 15) & (df[MAG].to_numpy() < 90)
        sub = allc[q]
        ig, _, s2, _ = search_around_sky(sub, allc, 10 * u.arcsec)
        nbr = np.bincount(ig[s2.arcsec > 0.5], minlength=int(q.sum()))
        d = df[q][nbr == 0].reset_index(drop=True)
        if dataset == "SPSS":
            s = pd.read_csv(AN.SPSS_CSV); s = s[s.Name.map(AN.norm) == AN.norm(obj)].iloc[0]
            ref = AN.gather_reference(s["RA_ICRS"], s["DE_ICRS"])
            coords = SkyCoord(ref["ra"], ref["dec"], unit="deg")
            idx, d2d, _ = match_coordinates_sky(SkyCoord(d.ALPHA_J2000.to_numpy(), d.DELTA_J2000.to_numpy(), unit="deg"), coords)
            ok = np.asarray(d2d.arcsec < 2.0)
            fc = ref["flux"][idx[ok]]
        else:
            st0 = TW._INFO[obj]
            tbl, coords, _ = K.get_reference("GAIAXP_CORR_LAMOST", obj, st0["ra"], st0["dec"], 1.34, 0.90)
            idx, d2d, _ = match_coordinates_sky(SkyCoord(d.ALPHA_J2000.to_numpy(), d.DELTA_J2000.to_numpy(), unit="deg"), coords)
            ok = np.asarray(d2d.arcsec < 2.0)
            rid = np.asarray(tbl[idx[ok]]["id"]).astype(str)
            xpf = TW.tile_xp_radec(st0["ra"], st0["dec"], set(rid))
            keep = np.array([r in xpf for r in rid])
            ok[np.where(ok)[0][~keep]] = False
            fc = np.vstack([xpf[r] for r in rid[keep]]) if keep.any() else np.empty((0, 343))
        dd = d[ok]
        if len(dd) < 15:
            return None
        fw, ft = _C["curves"][filt]
        zp = R.syn_mag(TC.XP_WAVE, fc, fw, ft, 0.0) - dd[MAG].to_numpy()
        snr = dd.SNR_APER_3.to_numpy()
        bright = np.isfinite(zp) & (snr >= 100)
        if bright.sum() < 10:
            return None
        return pd.DataFrame(dict(dataset=dataset, file=Path(fr).name, object=obj, filter=filt, exptime=meta["exptime"],
                                 year=int(str(Path(fr).name).split("_")[1][:4]), snr=snr,
                                 peak_e=dd.FLUX_MAX.to_numpy() * meta["egain"], res=1e3 * (zp - np.median(zp[bright]))))
    except Exception:                                                 # noqa: BLE001
        return None


if __name__ == "__main__":
    nper = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    M = pd.read_csv(OUT / "rephot_meta.csv")
    M = M[M.status.isin(["ok", "cached"])]
    M = M[[(CATD / (Path(f).name[:-5] + "_cat.fits")).exists() for f in M.file]]
    S = pd.concat([g.sample(min(nper, len(g)), random_state=1) for _, g in M.groupby("dataset")])
    items = list(S[["dataset", "file", "object", "filter"]].itertuples(index=False, name=None))
    with Pool(16, initializer=_init) as pool:
        res = [r for r in pool.map(frame_stars, items) if r is not None]
    A = pd.concat(res, ignore_index=True)
    A = A[np.isfinite(A.res) & (A.res.abs() < 500)]
    A.to_csv(OUT / f"faintcheck_stars{SFX}.csv", index=False)
    rows = []
    for ds, g in A.groupby("dataset"):
        for lo, hi in [(15, 20), (20, 30), (30, 50), (50, 100), (100, 200), (200, 1e9)]:
            x = g[(g.snr >= lo) & (g.snr < hi)]
            rows.append(dict(dataset=ds, kind="snr", lo=lo, hi=hi, n=len(x), frames=x.file.nunique(), res=x.res.median(),
                             err=1.2533 * 1.4826 * np.median(np.abs(x.res - x.res.median())) / np.sqrt(max(len(x), 1)),
                             peak_e_med=x.peak_e.median()))
        for lo, hi in [(0, 32), (32, 64), (64, 128), (128, 256), (256, 1e9)]:
            x = g[(g.peak_e >= lo) & (g.peak_e < hi) & (g.snr >= 15)]
            rows.append(dict(dataset=ds, kind="peak_e", lo=lo, hi=hi, n=len(x), frames=x.file.nunique(), res=x.res.median(),
                             err=1.2533 * 1.4826 * np.median(np.abs(x.res - x.res.median())) / np.sqrt(max(len(x), 1)),
                             peak_e_med=x.peak_e.median()))
    B = pd.DataFrame(rows); B.to_csv(OUT / f"faintcheck_bins{SFX}.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"{A.file.nunique()} frames, {len(A)} stars")
    print(B.round(1).to_string(index=False))
