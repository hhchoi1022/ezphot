#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260927_std_radial_check.py — radial ZP pattern of the field stars on the re-photometered standard-star frames.
200 random frames per dataset; field stars FLAGS 0, SNR_10" >= 100, FLUX_MAX <= 50k, isolated 10", corrected-XP match;
ZP_i - frame median vs distance from the image centre.  The standards sit at the centre (median 116-149 px), so a
centre-vs-median ZP difference biases an offset computed with the frame-median ZP.
Output: ~/ezphot/log/20260927_std_newbkg/radial_check.csv (+ printed profile)
"""
import importlib, sys, warnings, numpy as np, pandas as pd
from pathlib import Path
from multiprocessing import Pool
warnings.filterwarnings("ignore")
sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
FC = importlib.import_module("260927_std_faintcheck")
SO = FC.SO; AN, TW, K, R, TC = SO.AN, SO.TW, SO.K, SO.R, SO.TC
from astropy.io import fits
from astropy.coordinates import SkyCoord, match_coordinates_sky, search_around_sky
import astropy.units as u

def run(args):
    dataset, fr, obj, filt = args
    cp = SO.CATD / (Path(fr).name[:-5] + "_cat.fits")
    try:
        df, meta = SO.load_cat(cp)
        with fits.open(cp) as h:
            d0 = h["LDAC_OBJECTS"].data
            df["X"] = np.asarray(d0["X_IMAGE"], float); df["Y"] = np.asarray(d0["Y_IMAGE"], float)
            nx, ny = h[0].header.get("NAXIS1", 9576), h[0].header.get("NAXIS2", 6388)
        allc = SkyCoord(df.ALPHA_J2000.to_numpy(), df.DELTA_J2000.to_numpy(), unit="deg")
        q = (df.FLAGS.to_numpy() == 0) & (df.FLUX_MAX.to_numpy() <= 50000) & (df.SNR_APER_3.to_numpy() >= 100) & (df.MAG_APER_3.to_numpy() < 90)
        ig, _, s2, _ = search_around_sky(allc[q], allc, 10 * u.arcsec)
        nbr = np.bincount(ig[s2.arcsec > 0.5], minlength=int(q.sum()))
        d = df[q][nbr == 0].reset_index(drop=True)
        src = SkyCoord(d.ALPHA_J2000.to_numpy(), d.DELTA_J2000.to_numpy(), unit="deg")
        if dataset == "SPSS":
            s = pd.read_csv(AN.SPSS_CSV); s = s[s.Name.map(AN.norm) == AN.norm(obj)].iloc[0]
            ref = AN.gather_reference(s["RA_ICRS"], s["DE_ICRS"])
            idx, d2d, _ = match_coordinates_sky(src, SkyCoord(ref["ra"], ref["dec"], unit="deg"))
            ok = np.asarray(d2d.arcsec < 2.0); fc = ref["flux"][idx[ok]]
        else:
            st0 = TW._INFO[obj]
            tbl, coords, _ = K.get_reference("GAIAXP_CORR_LAMOST", obj, st0["ra"], st0["dec"], 1.34, 0.90)
            idx, d2d, _ = match_coordinates_sky(src, coords)
            ok = np.asarray(d2d.arcsec < 2.0)
            rid = np.asarray(tbl[idx[ok]]["id"]).astype(str)
            xpf = TW.tile_xp_radec(st0["ra"], st0["dec"], set(rid))
            keep = np.array([r in xpf for r in rid]); ok[np.where(ok)[0][~keep]] = False
            fc = np.vstack([xpf[r] for r in rid[keep]]) if keep.any() else np.empty((0, 343))
        dd = d[ok]
        if len(dd) < 15: return None
        fw, ft = FC._C["curves"][filt]
        zp = R.syn_mag(TC.XP_WAVE, fc, fw, ft, 0.0) - dd.MAG_APER_3.to_numpy()
        m = np.isfinite(zp)
        return pd.DataFrame(dict(dataset=dataset, file=Path(fr).name, filter=filt, res=1e3 * (zp[m] - np.median(zp[m])),
                                 rad=np.hypot(dd.X.to_numpy()[m] - nx / 2, dd.Y.to_numpy()[m] - ny / 2)))
    except Exception as e:
        return None

if __name__ == "__main__":
    M = pd.read_csv(SO.OUT / "rephot_meta.csv"); M = M[M.status.isin(["ok", "cached"])]
    S = pd.concat([g.sample(min(200, len(g)), random_state=3) for _, g in M.groupby("dataset")])
    with Pool(16, initializer=FC._init) as pool:
        res = [x for x in pool.map(run, list(S[["dataset", "file", "object", "filter"]].itertuples(index=False, name=None))) if x is not None]
    A = pd.concat(res); A.to_csv(SO.OUT / "radial_check.csv", index=False)
    edges = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000]
    for ds, g in A.groupby("dataset"):
        out = [f"{lo}-{hi}: {g[(g.rad >= lo) & (g.rad < hi)].res.median():+.1f} ({((g.rad >= lo) & (g.rad < hi)).sum()})" for lo, hi in zip(edges[:-1], edges[1:])]
        print(ds, g.file.nunique(), "frames |", " | ".join(out))
    A["yr"] = A.file.str.split("_").str[1].str[:4]
    for (ds, y, f), g in A.groupby(["dataset", "yr", "filter"]):
        c = g[g.rad < 1000].res
        if len(c) >= 30: print(ds, y, f, "centre(<1000px) - frame median:", round(c.median(), 1), "n", len(c), "frames", g.file.nunique())
