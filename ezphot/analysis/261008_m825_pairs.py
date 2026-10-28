#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
261008_m825_pairs.py — the same white dwarf is ~40-50 mmag fainter (relative to its spectrum) in the 2024 7DT10 m825 frames
than in the 2026 7DT12 m825 frames (WD2149+021 = SPSS G93-48: +70 vs +29 mmag).  Steps 0-2 of the plan
(report 261008_m825_epoch_report.md):

  0  per-frame Delta of every star observed in m800/m825/m850 in both epochs (2024 CALSPEC set; 2026 SPSS campaign
     2026-09-15 and the CALSPEC re-observation 2026-10-02), BOTH epochs against the same CALSPEC spectrum and the same
     curve (manufacturer, beam with the per-filter n_eff and nobeam), exactly as section 8.1:
     Delta = MAG_APER_3(std) + zp_x (global field ZP, standard excluded) - syn - expected_bias(peak_e)
     (expected_bias: CALSPEC model for 2024, SPSS model for 2026, as sections 8.1/8.2).  Per star/epoch/filter: median,
     frame-bootstrap error, the error expected from the SNR, the 2024-2026 difference and its significance.
  1  same-field relative photometry 2024 vs 2026 (no reference catalogue): 2026 reference magnitude of every star = mean
     over the 2026 frames (per-frame offsets from bright stars); per 2024 frame dm = m24 - m26.  Joint robust fit per field
     dm = c_frame + quadratic surface(x, y) + beta (BP-RP) + L(bin of predicted 2024 peak); the predicted (noise-free)
     peak = frame PSF scale x 10^(-0.4 (m26 + c)), so the binning is not coupled to the 2024 noise.  L = 0 for predicted
     peak >= 256 e-.  The white dwarf (removed from the field stars, r > 5 px) is then placed on L:
     excess = dm_WD - model - L(peak_WD).  Sensitivity: the same fit with a linear surface and with no surface; the
     colour-term error times the white dwarf's colour lever arm is added to its excess error; field stars near the white
     dwarf (white dwarf excluded) give the residual pattern at the image centre.
  2  low-signal correction audit: per-star field residuals against corrected Gaia XP (ZP_i - median ZP of SNR >= 100
     stars) in every 2024 7DT10 m825 frame and every 2026 7DT12 m825 frame, binned in the MEASURED peak (as
     260927_std_faintcheck / expected_bias) and in the PREDICTED peak (from the XP magnitude); compared with the pooled
     expected_bias and the step-1 curve; Delta of the white dwarfs re-corrected with the frame-matched curves.
Outputs: ~/ezphot/log/20261008_m825_pairs/ (frames_delta.csv, step0_summary.csv, step0_pairs.csv, step1_stars.csv,
  step1_loss.csv, step1_wd.csv, step2_stars.csv, step2_loss.csv, step2_wd.csv, E1-E4 png)

  MPLCONFIGDIR=/tmp/ezphot-matplotlib python -u ezphot/analysis/261008_m825_pairs.py --n-proc 24 [--rerun]
"""
import argparse
import importlib
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

warnings.filterwarnings("ignore")
from astropy.coordinates import SkyCoord, match_coordinates_sky, search_around_sky   # noqa: E402
import astropy.units as u                                                             # noqa: E402

sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
CM = importlib.import_module("261001_calspec_manuf_beam")
SO, SS, R = CM.SO, CM.SS, CM.R
TC, TW, K, AN = SO.TC, SO.TW, SO.K, SO.AN
OUT = Path("/home/hhchoi1022/ezphot/log/20261008_m825_pairs")
FILTS = ("m800", "m825", "m850")
UNIT24 = {"m800": "7DT10", "m825": "7DT10", "m850": "7DT11"}
# CALSPEC name -> SPSS-campaign name of the same star (2026-09-15); WD0227 / WD0308 come from the 2026-10-02 re-observation
ALIAS26 = {"WD2149+021": "G93-48", "WD2039-682": "WD2039-682", "GJ754.1A": "EG131", "FEIGE110": "Feige110"}
REOBS = ("WD0227+050", "WD0308-565")
CURVE = ("manufacturer", "beam")
PEAK_BINS = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1e5])
PEAK_REF = 256.0                          # loss defined as 0 above this predicted peak [e-]
SNR_BRIGHT = 100.0
ISO = 10.0                                # isolation radius ["]
_C = {}


def log(m):
    print(time.strftime("[%H:%M:%S] ") + m, flush=True)


def cat_path(f):
    return SO.CATD / (Path(f).name[:-5] + "_cat.fits")


# ----------------------------------------------------------------------------------------------------------- frames
def frame_table():
    c = pd.read_csv(SO.OUT.parent / "20260914_calspec_ezphot" / "photometry_meta.csv")
    c = c[(c.status == "ok") & c["filter"].isin(FILTS)]
    c = c[[u == UNIT24[f] for u, f in zip(c.unit, c["filter"])]]
    pair_obj = set(ALIAS26) | set(REOBS)
    a = pd.DataFrame(dict(epoch=2024, ds="CALSPEC", obj=c.object, object=c.object, filter=c["filter"], unit=c.unit, file=c.file))
    r = pd.read_csv(SO.OUT / "rephot_meta.csv")
    r = r[(r.dataset == "SPSS") & r.status.isin(["ok", "cached"]) & r["filter"].isin(FILTS) & (r.unit != "7DT11")]
    inv = {AN.norm(v): k for k, v in ALIAS26.items()}
    r = r[r.object.map(lambda o: AN.norm(o) in inv)]
    b = pd.DataFrame(dict(epoch=2026, ds="SPSS", obj=r.object.map(lambda o: inv[AN.norm(o)]), object=r.object,
                          filter=r["filter"], unit=r.unit, file=r.file))
    parts = [a, b]
    fr = OUT / "frames_reobs.csv"
    if fr.exists():
        q = pd.read_csv(fr)
        q = q[q.status.isin(["ok", "cached"]) & q["filter"].isin(FILTS)]
        parts.append(pd.DataFrame(dict(epoch=2026, ds="SPSS", obj=q.object, object=q.object, filter=q["filter"], unit=q.unit,
                                       file=q.file)))
    F = pd.concat(parts, ignore_index=True)
    F = F[[cat_path(f).exists() for f in F.file]].drop_duplicates("file")
    F["name"] = F.file.map(lambda f: Path(f).name)
    F["pair"] = F.obj.isin(pair_obj)
    return F.reset_index(drop=True)


# ----------------------------------------------------------------------------------------------------------- step 0
def _init():
    CM._init()
    SO._TOTALS = {k: v for k, v in SO._TOTALS.items() if k in (("manufacturer", "beam"), ("manufacturer", "nobeam"))}


def run_delta(F, n_proc):
    P = F[F.pair]
    jobs = [(obj, list(zip(g.file, g["filter"], g.unit))) for (obj, ep, flt), g in P.groupby(["obj", "epoch", "filter"])]
    rows = []
    with Pool(n_proc, initializer=_init) as pool:
        for r in pool.imap_unordered(SO.process_cal, jobs):
            rows += r
    D = pd.DataFrame(rows)
    D["obj"] = D["object"]
    D = D.merge(P[["name", "epoch", "ds", "object"]].rename(columns={"name": "file", "object": "obj_file"}), on="file", how="left")
    D["curve"] = D.variant
    D["bias"] = np.nan
    for ds in ("CALSPEC", "SPSS"):
        m = D.ds == ds
        D.loc[m, "bias"] = SS.expected_bias(D.loc[m, "std_peak_e"].to_numpy(), ds)
    D["d_raw"] = (D.d_gnd - D.zp + D.zp_x) * 1e3
    D["d"] = D.d_raw - D.bias
    D["d_rad"] = D.dr_gnd * 1e3 - D.bias                             # ZP from the frame's radial fit at the standard's radius
    D["err_exp"] = 1e3 * np.hypot(1.0857 / D.std_snr, D.zsd / np.sqrt(D.n_zp))
    D["ok"] = (D.std_flags == 0) & (D.std_peak <= SS.PEAK_MAX) & D.d.notna()
    return D


def boot_median(v, n=4000, seed=1):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 2:
        return np.nan
    rng = np.random.default_rng(seed)
    return float(np.std(np.median(rng.choice(v, (n, len(v))), axis=1)))


def combine(v, e):
    """weighted mean, its error, chi2, dof and the Birge-scaled error (x sqrt(chi2/dof) when chi2/dof > 1)."""
    v, e = np.asarray(v, float), np.asarray(e, float)
    ok = np.isfinite(v) & np.isfinite(e) & (e > 0)
    v, e = v[ok], e[ok]
    if not len(v):
        return dict(n=0, mean=np.nan, err=np.nan, chi2=np.nan, err_scaled=np.nan)
    w = 1 / e ** 2; mu = float((v * w).sum() / w.sum()); err = float(1 / np.sqrt(w.sum()))
    chi2 = float(((v - mu) ** 2 * w).sum()); dof = len(v) - 1
    return dict(n=len(v), mean=mu, err=err, chi2=chi2, err_scaled=err * np.sqrt(max(1.0, chi2 / dof)) if dof > 0 else err)


def step0(D):
    rows = []
    for (obj, flt, curve, ep), g in D[D.ok].groupby(["obj", "filter", "curve", "epoch"]):
        rows.append(dict(obj=obj, filter=flt, curve=curve, epoch=ep, unit=",".join(sorted(set(g.unit))), n=len(g),
                         exptime=g.exptime.median(), snr=g.std_snr.median(), peak_e=g.std_peak_e.median(), bias=g.bias.median(),
                         d_raw=g.d_raw.median(), d=g.d.median(), d_rad=g.d_rad.median(), err_boot=boot_median(g.d), mean=g.d.mean(),
                         sem=g.d.std(ddof=1) / np.sqrt(len(g)), rms=g.d.std(ddof=1), err_exp_frame=g.err_exp.median(),
                         chi2_frame=float(np.mean(((g.d - g.d.mean()) / g.err_exp) ** 2)), n_zp=g.n_zp.median(),
                         zp_loc_minus_glob=1e3 * np.nanmedian(g.zp_loc_x - g.zp_x), nights=",".join(sorted({str(int(m - 16 / 24)) for m in g.mjd}))))
    S = pd.DataFrame(rows)
    a, b = S[S.epoch == 2024].set_index(["obj", "filter", "curve"]), S[S.epoch == 2026].set_index(["obj", "filter", "curve"])
    J = a.join(b, lsuffix="_24", rsuffix="_26", how="inner")
    J["diff"] = J.d_24 - J.d_26
    J["diff_err"] = np.hypot(J.err_boot_24, J.err_boot_26)
    J["z"] = J["diff"] / J.diff_err
    J["diff_raw"] = J.d_raw_24 - J.d_raw_26
    J["diff_rad"] = J.d_rad_24 - J.d_rad_26
    return S, J.reset_index()


# ----------------------------------------------------------------------------------------------------------- step 1
def load_sources(f):
    df, meta = SO.load_cat(cat_path(f))
    df = df[(df.MAG_APER_3 < 90) & np.isfinite(df.MAG_APER_3)].reset_index(drop=True)
    return df, meta


def isolated(c_sel, c_all, rad=ISO):
    ig, _, s2, _ = search_around_sky(c_sel, c_all, rad * u.arcsec)
    return np.bincount(ig[s2.arcsec > 0.5], minlength=len(c_sel)) == 0


def ref26(files):
    """2026 reference magnitudes: anchor = frame with most detections; per-frame offset from stars SNR >= 50."""
    cats = [load_sources(f) for f in files]
    k0 = int(np.argmax([len(c[0]) for c in cats]))
    a = cats[k0][0]
    ca = SkyCoord(a.ALPHA_J2000.to_numpy(), a.DELTA_J2000.to_numpy(), unit="deg")
    M = np.full((len(a), len(cats)), np.nan); E = M.copy(); PK = M.copy(); FL = np.zeros_like(M)
    for j, (df, meta) in enumerate(cats):
        idx, d2d, _ = match_coordinates_sky(ca, SkyCoord(df.ALPHA_J2000.to_numpy(), df.DELTA_J2000.to_numpy(), unit="deg"))
        ok = d2d.arcsec < 1.0
        M[ok, j] = df.MAG_APER_3.to_numpy()[idx[ok]]; E[ok, j] = df.MAGERR_APER_3.to_numpy()[idx[ok]]
        PK[ok, j] = df.FLUX_MAX.to_numpy()[idx[ok]]; FL[ok, j] = df.FLAGS.to_numpy()[idx[ok]]
    snr_a = a.SNR_APER_3.to_numpy()
    offs = {}
    for j in range(len(cats)):
        b = (snr_a >= 50) & np.isfinite(M[:, j]) & (FL[:, j] == 0) & (PK[:, j] < 50000)
        offs[Path(files[j]).name] = float(np.nanmedian(M[b, j] - a.MAG_APER_3.to_numpy()[b]))
        M[:, j] -= offs[Path(files[j]).name]
    good = (FL == 0) & (PK < 50000) & np.isfinite(M)
    Mg = np.where(good, M, np.nan); W = np.where(good, 1 / np.maximum(E, 0.002) ** 2, 0)
    nd = good.sum(1)
    m26 = np.nansum(Mg * W, 1) / np.where(W.sum(1) > 0, W.sum(1), np.nan)
    e26 = 1 / np.sqrt(np.where(W.sum(1) > 0, W.sum(1), np.nan))
    keep = nd >= max(2, int(np.ceil(0.6 * len(cats))))
    iso = isolated(ca, ca)
    ref = pd.DataFrame(dict(ra=a.ALPHA_J2000, dec=a.DELTA_J2000, m26=m26, e26=e26, n26=nd, iso=iso))[keep].reset_index(drop=True)
    return ref, offs, cats


def wd_in(obj, df, meta):
    """the white dwarf in one catalogue (CALSPEC J2000 position moved with the proper motion, 1.5\" match)."""
    st0 = TW._INFO[obj]
    dt = (meta["mjd"] - 51544.5) / 365.25
    pos = SkyCoord(st0["ra"] + st0["pmra"] * dt / 3.6e6 / np.cos(np.deg2rad(st0["dec"])), st0["dec"] + st0["pmdec"] * dt / 3.6e6, unit="deg")
    return SO.std_info(df, pos, 1.5, meta)[1]


def wd_pos2016(obj):
    """CALSPEC J2000 position moved to the Gaia epoch 2016.0 with the proper motion."""
    st0 = TW._INFO[obj]
    dt = 16.0
    return (np.array([st0["ra"] + st0["pmra"] * dt / 3.6e6 / np.cos(np.deg2rad(st0["dec"]))]),
            np.array([st0["dec"] + st0["pmdec"] * dt / 3.6e6]))


def colours(obj, ra, dec):
    st0 = TW._INFO[obj]
    tbl, coords, _ = K.get_reference("GAIAXP_CORR_LAMOST", obj, st0["ra"], st0["dec"], 1.34, 0.90)
    idx, d2d, _ = match_coordinates_sky(SkyCoord(ra, dec, unit="deg"), coords)
    col = np.asarray(tbl["bp-rp"], float)[idx]
    col[d2d.arcsec > 1.5] = np.nan
    return col


def rel_field(args):
    """step 1 for one (pair star, filter): every 2024 frame against the 2026 reference."""
    obj, flt, f24, f26, wd24, wd26 = args
    try:
        AN._init(); TW._init(40)
        ref, offs, cats26 = ref26(f26)
        ref["col"] = colours(obj, ref.ra.to_numpy(), ref.dec.to_numpy())
        cref = SkyCoord(ref.ra.to_numpy(), ref.dec.to_numpy(), unit="deg")
        # the white dwarf: 2026 reference = median over the 2026 frames of its magnitude minus that frame's offset
        w26 = []
        for f, (df, meta) in zip(f26, cats26):
            inf = wd_in(obj, df, meta)
            if inf is not None and inf["std_flags"] == 0 and inf["std_peak"] < 50000:
                w26.append(inf["std_mag"] - offs[Path(f).name])
        m26_wd = float(np.median(w26)) if len(w26) >= 2 else np.nan
        e26_wd = float(1.2533 * np.std(w26, ddof=1) / np.sqrt(len(w26))) if len(w26) >= 2 else np.nan
        col_wd = float(colours(obj, *wd_pos2016(obj))[0])
        if not np.isfinite(col_wd):
            reg = pd.read_csv(SO.OUT / "spss_registered_bprp.csv").set_index("name").bprp
            col_wd = float(reg.get(ALIAS26.get(obj, obj), np.nan))
        rows, wrows = [], []
        for f in f24:
            df, meta = load_sources(f)
            c24 = SkyCoord(df.ALPHA_J2000.to_numpy(), df.DELTA_J2000.to_numpy(), unit="deg")
            idx, d2d, _ = match_coordinates_sky(c24, cref)
            inf = wd_in(obj, df, meta)
            ok = (d2d.arcsec < 1.5) & (df.FLAGS.to_numpy() == 0) & (df.FLUX_MAX.to_numpy() < 50000) & ref.iso.to_numpy()[idx]
            if inf is not None:                                      # the white dwarf is not a field star
                ok &= np.hypot(df.X.to_numpy() - inf["std_x"], df.Y.to_numpy() - inf["std_y"]) > 5
            d = df[ok].reset_index(drop=True); r = ref.iloc[idx[ok]].reset_index(drop=True)
            if len(d) < 30:
                continue
            if inf is not None and inf["std_flags"] == 0 and np.isfinite(m26_wd):
                wrows.append(dict(obj=obj, filter=flt, file=Path(f).name, x=inf["std_x"], y=inf["std_y"], nx=meta["nx"], ny=meta["ny"],
                                  m24=inf["std_mag"], m26=m26_wd, e26=e26_wd, n26=len(w26), col=col_wd, peak_e=inf["std_peak_e"],
                                  snr24=inf["std_snr"], dm=inf["std_mag"] - m26_wd))
            rows.append(pd.DataFrame(dict(obj=obj, filter=flt, file=Path(f).name, sid=idx[ok], x=d.X, y=d.Y, m24=d.MAG_APER_3,
                                          e24=d.MAGERR_APER_3, snr24=d.SNR_APER_3, peak_e=d.FLUX_MAX * meta["egain"], m26=r.m26,
                                          e26=r.e26, col=r.col, nx=meta["nx"], ny=meta["ny"])))
        if not rows:
            return None
        S = pd.concat(rows, ignore_index=True)
        S["dm"] = S.m24 - S.m26
        return S, pd.DataFrame(wrows)
    except Exception as e:                                     # noqa: BLE001
        log(f"rel_field {obj} {flt}: {e}")
        return None


SURF = {"quad": lambda T: [T.xn, T.yn, T.xn ** 2, T.yn ** 2, T.xn * T.yn], "lin": lambda T: [T.xn, T.yn], "none": lambda T: []}


def fit_loss(S, wd=None, ref_peak=PEAK_REF, surf="quad"):
    """robust joint fit dm = c_frame + surface(x,y) + beta*(col-col0) + L(pred-peak bin); returns stars with model/loss,
    the binned L, beta and its error, and the white-dwarf rows with their model."""
    nsp = len(SURF[surf](pd.DataFrame(dict(xn=[0.0], yn=[0.0]))))
    S = S.copy()
    # predicted peak: per frame, PSF scale from stars SNR >= 50, magnitude from m26 + c_frame (c from bright stars)
    for f, g in S.groupby("file"):
        b = g[(g.snr24 >= 50)]
        c0 = np.median(b.dm) if len(b) >= 5 else np.nan
        kk = np.median(b.peak_e / 10 ** (-0.4 * b.m24)) if len(b) >= 5 else np.nan
        S.loc[g.index, "c0"] = c0; S.loc[g.index, "kscale"] = kk
    S = S[np.isfinite(S.c0)]
    S["peak_pred"] = S.kscale * 10 ** (-0.4 * (S.m26 + S.c0))
    col0 = np.nanmedian(S.col[S.snr24 >= 50])
    S["colc"] = np.where(np.isfinite(S.col), S.col - col0, 0.0)
    S["xn"] = (S.x - S.nx / 2) / (S.nx / 2); S["yn"] = (S.y - S.ny / 2) / (S.ny / 2)
    S["bin"] = np.digitize(S.peak_pred, PEAK_BINS) - 1
    S = S[(S.bin >= 0) & (S.bin < len(PEAK_BINS) - 1)]
    files = sorted(S.file.unique()); nb = len(PEAK_BINS) - 1
    ref_bins = [i for i in range(nb) if PEAK_BINS[i] >= ref_peak]

    def design(T):
        X = [np.asarray([T.file.to_numpy() == f for f in files], float).T,
             np.column_stack(SURF[surf](T) + [T.colc])]
        Lb = np.asarray([(T.bin.to_numpy() == i) for i in range(nb) if i not in ref_bins], float).T
        return np.hstack(X + [Lb])
    X = design(S); y = S.dm.to_numpy(); s = np.hypot(np.hypot(S.e24, S.e26), 0.005).to_numpy()
    w = np.ones(len(y), bool)
    for _ in range(5):
        beta, *_ = np.linalg.lstsq(X[w] / s[w, None], y[w] / s[w], rcond=None)
        res = (y - X @ beta) / s
        mad = 1.4826 * np.median(np.abs(res[w]))
        w = np.abs(res) < 4 * max(mad, 1.0)
    nf = len(files); nk = nf + nsp + 1                              # columns before the loss bins
    lbins = [i for i in range(nb) if i not in ref_bins]
    Lval = dict(zip(lbins, beta[nk:]))
    Xw = X[w] / s[w, None]
    beta_err = float(np.sqrt(max(mad, 1.0) ** 2 * np.linalg.pinv(Xw.T @ Xw)[nk - 1, nk - 1]))
    # bin error: bootstrap over stars would be costly; use the robust scatter of the residuals in each bin
    S["model_nl"] = X[:, :nk] @ beta[:nk]
    S["loss"] = S.dm - S.model_nl                                   # star-by-star loss estimate (+ = too faint)
    S["used"] = w
    rows = []
    for i in range(nb):
        g = S[(S.bin == i) & S.used]
        rows.append(dict(lo=PEAK_BINS[i], hi=PEAK_BINS[i + 1], n=len(g), frames=g.file.nunique(),
                         peak_med=g.peak_pred.median() if len(g) else np.nan,
                         L_fit=1e3 * Lval.get(i, 0.0), L_med=1e3 * g.loss.median() if len(g) else np.nan,
                         err=1e3 * 1.2533 * 1.4826 * np.median(np.abs(g.loss - g.loss.median())) / np.sqrt(max(len(g), 1)) if len(g) else np.nan))
    L = pd.DataFrame(rows)
    out_wd = None
    if wd is not None and len(wd):
        Wd = wd.copy()
        Wd = Wd.merge(S.groupby("file")[["c0", "kscale"]].first().reset_index(), on="file", how="inner")
        Wd["xn"] = (Wd.x - Wd.nx / 2) / (Wd.nx / 2); Wd["yn"] = (Wd.y - Wd.ny / 2) / (Wd.ny / 2)
        Wd["colc"] = np.where(np.isfinite(Wd.col), Wd.col - col0, 0.0)
        Wd["peak_pred"] = Wd.kscale * 10 ** (-0.4 * (Wd.m26 + Wd.c0))
        fidx = {f: k for k, f in enumerate(files)}
        mod = []
        for r in Wd.itertuples():
            v = np.zeros(nk); v[fidx[r.file]] = 1
            v[nf:nk] = [float(np.asarray(t)) for t in SURF[surf](pd.DataFrame(dict(xn=[r.xn], yn=[r.yn])))] + [r.colc]
            mod.append(float(v @ beta[:nk]))
        Wd["model_nl"] = mod
        Wd["loss"] = Wd.dm - Wd.model_nl
        lp = np.log10(L.peak_med.to_numpy()); lv = L.L_med.to_numpy() / 1e3
        okb = np.isfinite(lp) & np.isfinite(lv)
        Wd["L_curve"] = np.interp(np.log10(Wd.peak_pred), lp[okb], lv[okb])
        Wd["excess"] = Wd.loss - Wd.L_curve
        Wd["beta"], Wd["beta_err"] = float(beta[nk - 1]), beta_err
        out_wd = Wd
    return S, L, float(beta[nk - 1]), col0, out_wd


def step1(D, F, n_proc):
    jobs = []
    for (obj, flt), g in F[F.pair].groupby(["obj", "filter"]):
        g26 = g[g.epoch == 2026]
        if g26.empty:
            continue
        u26 = g26.unit.value_counts().index[0]                       # one 2026 unit (filter copy) per reference
        f24 = g[g.epoch == 2024].file.tolist(); f26 = g26[g26.unit == u26].file.tolist()
        if len(f24) >= 3 and len(f26) >= 3:
            jobs.append((obj, flt, f24, f26, None, None))
    with Pool(min(n_proc, len(jobs))) as pool:
        res = pool.map(rel_field, jobs)
    stars, losses, wds, ncs = [], [], [], []
    for (obj, flt, f24, f26, _, _), out in zip(jobs, res):
        if out is None:
            continue
        S, wd = out
        if wd.empty:
            log(f"step1 {obj} {flt}: white dwarf not measured"); continue
        log(f"step1 {obj} {flt}: white dwarf in {len(wd)} 2024 frames, 2026 reference from {int(wd.n26.iloc[0])} frames ± {1e3 * wd.e26.iloc[0]:.1f} mmag")
        for surf in ("quad", "lin", "none"):
            S2, L, beta, col0, W = fit_loss(S, wd, surf=surf)
            S2["obj"], S2["filter"] = obj, flt
            L["obj"], L["filter"], L["beta"], L["col0"] = obj, flt, beta, col0
            if W is not None:
                W["obj"], W["filter"], W["surf"] = obj, flt, surf
                wds.append(W)
                nc = near_centre(S2, L, W); nc["obj"], nc["filter"], nc["surf"] = obj, flt, surf
                ncs.append(nc)
            if surf == "quad":
                stars.append(S2); losses.append(L)
            log(f"step1 {obj} {flt} [{surf}]: {S2.file.nunique()} frames, {len(S2)} star-frames, beta {1e3 * beta:.1f} mmag/mag")
    return (pd.concat(stars, ignore_index=True), pd.concat(losses, ignore_index=True),
            (pd.concat(wds, ignore_index=True) if wds else None), pd.concat(ncs, ignore_index=True))


def near_centre(S2, L, W, radii=(400, 800, 1500), seed=2):
    """field stars near the white dwarf (white dwarf already excluded), predicted peak > 16 e-: residual after the
    binned loss, median over stars (each star = median of its frames) and a bootstrap over stars."""
    T = S2[S2.used & (S2.peak_pred > 16)].copy()
    Lm = dict(zip(range(len(L)), L.L_med))
    T["r"] = 1e3 * T.loss - T.bin.map(Lm)
    xw, yw = W.x.median(), W.y.median()
    T["dist"] = np.hypot(T.x - xw, T.y - yw)
    rng = np.random.default_rng(seed)
    rows = []
    for rad in radii:
        per = T[T.dist < rad].groupby("sid").r.median().dropna().to_numpy()
        err = float(np.std([np.median(rng.choice(per, len(per))) for _ in range(2000)])) if len(per) >= 2 else np.nan
        rows.append(dict(rad=rad, n_stars=len(per), r_med=float(np.median(per)) if len(per) else np.nan, err=err,
                         wd_excess=1e3 * W.excess.median()))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------- step 2
def _init_xp():
    warnings.filterwarnings("ignore")
    AN._init(); TW._init(40)
    _C["curves"] = CM.manuf_totals()[CURVE]


def xp_frame(args):
    """field stars of one frame vs corrected Gaia XP: measured and predicted peak, residual to the bright-star ZP."""
    ds, f, obj, flt = args
    try:
        df, meta = SO.load_cat(cat_path(f))
        df = df[(df.MAG_APER_3 < 90) & np.isfinite(df.MAG_APER_3)].reset_index(drop=True)
        allc = SkyCoord(df.ALPHA_J2000.to_numpy(), df.DELTA_J2000.to_numpy(), unit="deg")
        q = (df.FLAGS.to_numpy() == 0) & (df.FLUX_MAX.to_numpy() <= 50000) & (df.SNR_APER_3.to_numpy() >= 3)
        iso = isolated(allc[q], allc)
        d = df[q][iso].reset_index(drop=True)
        cd = SkyCoord(d.ALPHA_J2000.to_numpy(), d.DELTA_J2000.to_numpy(), unit="deg")
        if ds == "SPSS":
            s = pd.read_csv(AN.SPSS_CSV); s = s[s.Name.map(AN.norm) == AN.norm(obj)].iloc[0]
            ref = AN.gather_reference(s["RA_ICRS"], s["DE_ICRS"])
            idx, d2d, _ = match_coordinates_sky(cd, SkyCoord(ref["ra"], ref["dec"], unit="deg"))
            ok = np.asarray(d2d.arcsec < 2.0)
            fc = ref["flux"][idx[ok]]; col = np.asarray(ref["bprp"], float)[idx[ok]]
            spos = SkyCoord(s["RA_ICRS"], s["DE_ICRS"], unit="deg")
        else:
            st0 = TW._INFO[obj]
            tbl, coords, _ = K.get_reference("GAIAXP_CORR_LAMOST", obj, st0["ra"], st0["dec"], 1.34, 0.90)
            idx, d2d, _ = match_coordinates_sky(cd, coords)
            ok = np.asarray(d2d.arcsec < 2.0)
            rid = np.asarray(tbl[idx[ok]]["id"]).astype(str)
            xpf = TW.tile_xp_radec(st0["ra"], st0["dec"], set(rid))
            keep = np.array([r in xpf for r in rid])
            ok[np.where(ok)[0][~keep]] = False
            fc = np.vstack([xpf[r] for r in rid[keep]]) if keep.any() else np.empty((0, len(TC.XP_WAVE)))
            col = np.asarray(tbl[idx[ok]]["bp-rp"], float) if ok.any() else np.array([])
            spos = SkyCoord(st0["ra"], st0["dec"], unit="deg")
        dd = d[ok].reset_index(drop=True)
        if len(dd) < 15:
            return None
        fw, ft = _C["curves"][flt]
        syn = R.syn_mag(TC.XP_WAVE, fc, fw, ft, 0.0)
        zp = syn - dd.MAG_APER_3.to_numpy()
        snr = dd.SNR_APER_3.to_numpy()
        far = cd[ok].separation(spos).arcsec > 20                   # the standard itself is not a field star
        bright = np.isfinite(zp) & (snr >= SNR_BRIGHT) & far
        if bright.sum() < 8:
            return None
        z0 = np.median(zp[bright])
        pk = dd.FLUX_MAX.to_numpy() * meta["egain"]
        b50 = np.isfinite(zp) & (snr >= 50) & far
        kscale = np.median(pk[b50] / 10 ** (-0.4 * dd.MAG_APER_3.to_numpy()[b50]))
        return pd.DataFrame(dict(ds=ds, epoch=int(Path(f).name.split("_")[1][:4]), file=Path(f).name, obj=obj, filter=flt,
                                 unit=Path(f).name.split("_")[0], exptime=meta["exptime"], bkg=meta["bkgmean"], snr=snr,
                                 peak_e=pk, peak_pred=kscale * 10 ** (-0.4 * (syn - z0)), col=col, far=far,
                                 loss=-1e3 * (zp - z0), n_bright=int(bright.sum()), kscale=kscale, z0=z0))[far]
    except Exception:                                              # noqa: BLE001
        return None


def binned(A, xcol, sel=None, bins=PEAK_BINS):
    A = A if sel is None else A[sel]
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        g = A[(A[xcol] >= lo) & (A[xcol] < hi)]
        rows.append(dict(lo=lo, hi=hi, n=len(g), frames=g.file.nunique(), x_med=g[xcol].median() if len(g) else np.nan,
                         snr_med=g.snr.median() if len(g) else np.nan, loss=g.loss.median() if len(g) else np.nan,
                         err=1.2533 * 1.4826 * np.median(np.abs(g.loss - g.loss.median())) / np.sqrt(max(len(g), 1)) if len(g) else np.nan))
    return pd.DataFrame(rows)


def step2(F, n_proc):
    # every 2024 7DT10 m825 frame (all CALSPEC fields) and every 2026 7DT12 m825 frame (SPSS campaign + re-observation)
    a = F[(F.epoch == 2024) & (F["filter"] == "m825")]
    r = pd.read_csv(SO.OUT / "rephot_meta.csv")
    r = r[(r.dataset == "SPSS") & r.status.isin(["ok", "cached"]) & (r["filter"] == "m825") & (r.unit == "7DT12")]
    b = pd.DataFrame(dict(ds="SPSS", file=r.file, obj=r.object, filter=r["filter"]))
    c = F[(F.epoch == 2026) & (F["filter"] == "m825") & F.obj.isin(REOBS)]
    jobs = [("CALSPEC", f, o, "m825") for f, o in zip(a.file, a.obj)] + \
           [("SPSS", f, o, "m825") for f, o in zip(b.file, b.obj) if cat_path(f).exists()] + \
           [("CALSPEC", f, o, "m825") for f, o in zip(c.file, c.obj)]
    with Pool(n_proc, initializer=_init_xp) as pool:
        res = [x for x in pool.map(xp_frame, jobs) if x is not None]
    A = pd.concat(res, ignore_index=True)
    A = A[np.isfinite(A.loss) & (A.loss.abs() < 1000)]
    rows = []
    groups = [(ep, "all m825", A[A.epoch == ep]) for ep in (2024, 2026)]
    for (ep, t), E in A.groupby(["epoch", "exptime"]):
        groups.append((ep, f"exp {int(t)} s", E))
    E = A[(A.epoch == 2026) & (A.exptime == 100)].copy()
    E["sky_e"] = E.bkg * 0.2562                                    # [e-/px], EGAIN of the 7DT cameras
    q = E.drop_duplicates("file").sky_e.quantile([1 / 3, 2 / 3]).to_numpy()
    for lab, sel in ((f"2026 100 s sky < {q[0]:.1f}", E.sky_e < q[0]), (f"2026 100 s sky {q[0]:.1f}-{q[1]:.1f}", (E.sky_e >= q[0]) & (E.sky_e < q[1])),
                     (f"2026 100 s sky > {q[1]:.1f}", E.sky_e >= q[1])):
        groups.append((2026, lab, E[sel]))
    pair_names = {**{k: k for k in ALIAS26}, **{v: k for k, v in ALIAS26.items()}, **{k: k for k in REOBS}}
    for (ep, o), E in A[A.obj.isin(pair_names)].groupby(["epoch", "obj"]):
        groups.append((ep, pair_names[o], E))
    for ep, field, E in groups:
        for kind, xcol, sel in (("measured_peak_snr15", "peak_e", E.snr >= 15), ("measured_peak_all", "peak_e", None),
                                ("predicted_peak", "peak_pred", None)):
            B = binned(E, xcol, sel); B["epoch"], B["field"], B["kind"] = ep, field, kind
            rows.append(B)
    return A, pd.concat(rows, ignore_index=True)


SNR_COMPLETE = 8.0                        # a predicted-peak bin whose median measured SNR is lower is detection-limited


def reliable(B):
    return B[np.isfinite(B.x_med) & np.isfinite(B.loss) & (B.n >= 20) & (B.snr_med >= SNR_COMPLETE)]


def interp_loss(B, x):
    B = reliable(B).sort_values("x_med")
    return np.interp(np.log10(np.clip(x, 1, None)), np.log10(B.x_med), B.loss)


# ----------------------------------------------------------------------------------------------------------- figures
COL = {2024: "#B5523B", 2026: "#2F6F8F"}


def fig_e1(D, J):
    stars = sorted(D[D.ok & (D["filter"] == "m825")].obj.unique())
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for k, obj in enumerate(stars):
        for ep, dx in ((2024, -0.17), (2026, 0.17)):
            g = D[D.ok & (D.obj == obj) & (D["filter"] == "m825") & (D.curve == "beam") & (D.epoch == ep)]
            if g.empty:
                continue
            xs = k + dx + np.linspace(-0.07, 0.07, len(g))
            ax.errorbar(xs, g.d, g.err_exp, fmt="o", ms=3, color=COL[ep], alpha=0.35, lw=0.6)
            ax.errorbar(k + dx, g.d.median(), boot_median(g.d), fmt="s", ms=8, color=COL[ep], mec="k", lw=1.8, capsize=3,
                        label=f"{ep} ({'7DT10 2024-08' if ep == 2024 else '7DT12 2026-09/10'})" if k == 0 else None)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xticks(range(len(stars))); ax.set_xticklabels(stars)
    ax.set_ylabel("Δ = obs − syn [mmag] (bias-corrected)"); ax.set_title("m825, manuf·beam: each frame (small) and the frame median ± bootstrap error (square)")
    ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "E1_step0_frames.png", dpi=150); plt.close(fig)


def fig_e2(S, L, W):
    objs = [o for o in L.obj.unique() if ((L.obj == o) & (L["filter"] == "m825")).any()]
    fig, axes = plt.subplots(1, len(objs), figsize=(3.3 * len(objs), 4.0), sharey=True, squeeze=False)
    for ax, obj in zip(axes[0], objs):
        s = S[(S.obj == obj) & (S["filter"] == "m825") & S.used]
        ax.plot(s.peak_pred, 1e3 * s.loss, ".", ms=1.2, color="0.6", alpha=0.35, rasterized=True)
        l = L[(L.obj == obj) & (L["filter"] == "m825") & (L.n >= 10)]
        ax.errorbar(l.peak_med, l.L_med, l.err, fmt="o-", color="k", ms=4, lw=1.2, label="field stars (bin median)")
        if W is not None:
            w = W[(W.obj == obj) & (W["filter"] == "m825")]
            ax.plot(w.peak_pred, 1e3 * w.loss, "*", ms=7, color=COL[2024], mec="k", mew=0.3, label="white dwarf (frames)")
            ax.errorbar(w.peak_pred.median(), 1e3 * w.loss.median(), 1e3 * boot_median(w.loss), fmt="*", ms=15,
                        color=COL[2024], mec="k", capsize=3, label="white dwarf (median)")
        ax.axhline(0, color="0.5", lw=0.8); ax.set_xscale("log"); ax.set_ylim(-150, 250)
        ax.set_title(obj, fontsize=10); ax.set_xlabel("predicted 2024 peak [e⁻]"); ax.grid(alpha=0.25)
    axes[0][0].set_ylabel("m(2024) − m(2026) − model [mmag]\n(+ = fainter in 2024)")
    axes[0][0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Step 1 — same stars, 2024 (7DT10, 20–30 s) vs 2026 (7DT12, 30–100 s), m825; no reference catalogue", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "E2_step1_relative.png", dpi=150); plt.close(fig)


def fig_e3(B2, L1, W2):
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    pk = np.logspace(0.7, 3.2, 100)
    ax.plot(pk, SS.expected_bias(pk, "CALSPEC"), "-", color=COL[2024], lw=2, alpha=0.5, label="expected_bias, CALSPEC model (pooled, measured peak) — used for 2024")
    ax.plot(pk, SS.expected_bias(pk, "SPSS"), "-", color=COL[2026], lw=2, alpha=0.5, label="expected_bias, SPSS model — used for 2026")
    sty = {"measured_peak_snr15": ("s", "--", "XP, binned in measured peak (SNR ≥ 15, as expected_bias)"),
           "predicted_peak": ("o", "-", "XP, binned in predicted peak")}
    for ep in (2024, 2026):
        for kind, (mk, ls, lab) in sty.items():
            b = reliable(B2[(B2.epoch == ep) & (B2.kind == kind) & (B2.field == "all m825")])
            ax.errorbar(b.x_med, b.loss, b.err, fmt=mk + ls, color=COL[ep], ms=5, lw=1, mfc="w" if kind.startswith("meas") else COL[ep],
                        label=f"{ep} m825 {'7DT10' if ep == 2024 else '7DT12'}: {lab}")
    l = L1[(L1["filter"] == "m825") & (L1.n >= 30)].groupby("lo").apply(
        lambda g: pd.Series(dict(x=np.average(g.peak_med, weights=g.n), y=np.average(g.L_med, weights=g.n)))).reset_index()
    ax.plot(l.x, l.y, "D:", color="0.2", ms=5, label="step 1: 2024 − 2026, same stars (pooled fields; 0 at ≥ 256 e⁻)")
    if W2 is not None:
        for ep in (2024, 2026):
            w = W2[W2.epoch == ep]
            ax.plot(w.peak_pred, w.d_raw, "*", ms=7, color=COL[ep], mec="k", mew=0.3, alpha=0.6,
                    label=f"{ep} white-dwarf frames: Δ before correction, at noise-free peak")
    ax.axhline(0, color="0.5", lw=0.8); ax.set_xscale("log"); ax.set_xlim(4, 1600); ax.set_ylim(-40, 160)
    ax.set_xlabel("peak above sky [e⁻]"); ax.set_ylabel("low-signal loss [mmag] (+ = measured too faint)")
    ax.set_title("Step 2 — m825 low-signal loss: pooled correction vs curves from the m825 frames themselves", fontsize=10)
    ax.legend(fontsize=6.6, loc="upper right"); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "E3_step2_loss.png", dpi=150); plt.close(fig)


def fig_e4(T):
    objs = T.obj.unique().tolist()
    vers = [c for c in T.columns if c.startswith("d_")]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    w = 0.8 / (2 * len(vers))
    for k, obj in enumerate(objs):
        for j, v in enumerate(vers):
            for e, ep in enumerate((2024, 2026)):
                r = T[(T.obj == obj) & (T.epoch == ep)]
                if r.empty:
                    continue
                x = k - 0.4 + (2 * j + e + 0.5) * w
                ax.errorbar(x, r[v].iloc[0], r.err.iloc[0], fmt="o", color=COL[ep], mfc=["w", COL[ep], "0.5"][j % 3], ms=5, capsize=2)
    ax.axhline(0, color="0.5", lw=0.8); ax.set_xticks(range(len(objs))); ax.set_xticklabels(objs); ax.grid(alpha=0.25)
    ax.set_ylabel("Δ m825 [mmag]")
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], marker="s", ls="", color=COL[2024], label="2024 (7DT10, 20–30 s)"), Line2D([], [], marker="s", ls="", color=COL[2026], label="2026 (7DT12, 30–100 s)"),
          Line2D([], [], marker="o", ls="", color="k", mfc="w", label="pooled expected_bias (as 8.1/8.2)"),
          Line2D([], [], marker="o", ls="", color="k", label="m825 curve (all frames, noise-free peak)"),
          Line2D([], [], marker="o", ls="", color="k", mfc="0.5", label="same-field curve (noise-free peak)")]
    ax.legend(handles=hs, fontsize=8, ncol=2, loc="upper right")
    ax.set_title("Step 2 — white-dwarf Δ m825 with each low-signal correction", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "E4_step2_wd.png", dpi=150); plt.close(fig)


# ----------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n-proc", type=int, default=24); ap.add_argument("--rerun", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    F = frame_table()
    F.to_csv(OUT / "frames_all.csv", index=False)
    log("frames: " + repr(F.groupby(["epoch", "filter"]).size().to_dict()) + f"; pair-star frames {int(F.pair.sum())}")
    fd = OUT / "frames_delta.csv"
    if a.rerun or not fd.exists():
        D = run_delta(F, a.n_proc); D.to_csv(fd, index=False)
    else:
        D = pd.read_csv(fd)
    S0, J0 = step0(D)
    S0.round(3).to_csv(OUT / "step0_summary.csv", index=False); J0.round(3).to_csv(OUT / "step0_pairs.csv", index=False)
    print("=== step 0 (beam)\n" + J0[J0.curve == "beam"][["obj", "filter", "unit_24", "n_24", "snr_24", "peak_e_24", "bias_24", "d_24",
          "err_boot_24", "unit_26", "n_26", "snr_26", "peak_e_26", "bias_26", "d_26", "err_boot_26", "diff", "diff_err", "z",
          "diff_raw"]].round(1).to_string(index=False))
    rows = []
    for c in ("diff", "diff_raw", "diff_rad"):
        m = J0[(J0.curve == "beam") & (J0["filter"] == "m825")]
        rows.append(dict(quantity=c, **combine(m[c].to_numpy(), m.diff_err.to_numpy())))
    C0 = pd.DataFrame(rows); C0.round(2).to_csv(OUT / "step0_combined.csv", index=False)
    print("=== step 0 m825 weighted 2024 - 2026\n" + C0.round(1).to_string(index=False))
    fig_e1(D, J0)

    AN._init(); TW._init(40)
    S1, L1, W1, NC = step1(D, F, a.n_proc)
    S1.to_csv(OUT / "step1_stars.csv", index=False); L1.round(3).to_csv(OUT / "step1_loss.csv", index=False)
    NC.round(2).to_csv(OUT / "step1_near_centre.csv", index=False)
    print("=== step 1 field stars near the white dwarf (white dwarf excluded)\n" + NC.round(1).to_string(index=False))
    if W1 is not None:
        W1.to_csv(OUT / "step1_wd.csv", index=False)
        print("=== step 1 white dwarfs (median over 2024 frames, mmag); excess_err includes the 2026 reference and beta_err x colour lever")
        W1s = W1.groupby(["obj", "filter", "surf"]).apply(lambda g: pd.Series(dict(n=len(g), n26=g.n26.iloc[0], peak_pred=g.peak_pred.median(),
              peak_meas=g.peak_e.median(), snr24=g.snr24.median(), beta=1e3 * g.beta.iloc[0], beta_err=1e3 * g.beta_err.iloc[0],
              colc=g.colc.iloc[0], loss=1e3 * g.loss.median(),
              loss_err=1e3 * np.hypot(boot_median(g.loss), g.e26.iloc[0]), L_curve=1e3 * g.L_curve.median(),
              excess=1e3 * g.excess.median(),
              excess_err=1e3 * np.sqrt(boot_median(g.excess) ** 2 + g.e26.iloc[0] ** 2 + (g.beta_err.iloc[0] * g.colc.iloc[0]) ** 2)))).reset_index()
        W1s["z"] = W1s.excess / W1s.excess_err
        W1s.round(2).to_csv(OUT / "step1_wd_summary.csv", index=False)
        print(W1s.round(1).to_string(index=False))
        rows = []
        for (flt, surf), g in W1s.groupby(["filter", "surf"]):
            for tag, gg in (("all", g), ("no FEIGE110", g[g.obj != "FEIGE110"])):
                rows.append(dict(filter=flt, surf=surf, sample=tag, **combine(gg.excess.to_numpy(), gg.excess_err.to_numpy())))
        C1 = pd.DataFrame(rows); C1.round(2).to_csv(OUT / "step1_combined.csv", index=False)
        print("=== step 1 weighted means\n" + C1.round(1).to_string(index=False))
    print("=== step 1 loss bins (m825)\n" + L1[L1["filter"] == "m825"].round(1).to_string(index=False))
    fig_e2(S1, L1, W1[W1.surf == "quad"] if W1 is not None else None)

    A2, B2 = step2(F, a.n_proc)
    A2.to_csv(OUT / "step2_stars.csv", index=False); B2.round(2).to_csv(OUT / "step2_loss.csv", index=False)
    print("=== step 2 loss bins\n" + B2.round(1).to_string(index=False))
    # white dwarfs (m825, beam) re-corrected: pooled (as 8.1/8.2), XP frame-matched in predicted peak, step-1 curve
    W2 = D[D.ok & (D["filter"] == "m825") & (D.curve == "beam")].copy()
    W2["d_pooled"] = W2.d
    # the white dwarf's noise-free peak: its synthetic magnitude through the frame's ZP and PSF scale
    ks = A2.groupby("file")[["kscale"]].first()
    W2 = W2.merge(ks, left_on="file", right_index=True, how="left")
    m_syn = W2.std_mag + W2.zp - W2.d_gnd
    W2["peak_pred"] = W2.kscale * 10 ** (-0.4 * (m_syn - W2.zp_x))
    for ep in (2024, 2026):
        m = W2.epoch == ep
        sel = lambda kind, field: B2[(B2.epoch == ep) & (B2.kind == kind) & (B2.field == field)]
        W2.loc[m, "d_m825_meas"] = W2.loc[m, "d_raw"] - interp_loss(sel("measured_peak_snr15", "all m825"), W2.loc[m, "std_peak_e"].to_numpy())
        W2.loc[m, "d_m825_pred"] = W2.loc[m, "d_raw"] - interp_loss(sel("predicted_peak", "all m825"), W2.loc[m, "peak_pred"].to_numpy())
        for t in W2.loc[m, "exptime"].unique():
            mt = m & (W2.exptime == t)
            b = sel("predicted_peak", f"exp {int(t)} s")
            if len(reliable(b)) >= 3:
                W2.loc[mt, "d_exp_pred"] = W2.loc[mt, "d_raw"] - interp_loss(b, W2.loc[mt, "peak_pred"].to_numpy())
        for obj in W2.loc[m, "obj"].unique():
            mm = m & (W2.obj == obj)
            b = sel("predicted_peak", obj)
            if len(reliable(b)) >= 3:
                W2.loc[mm, "d_field_pred"] = W2.loc[mm, "d_raw"] - interp_loss(b, W2.loc[mm, "peak_pred"].to_numpy())
    W2.to_csv(OUT / "step2_wd_frames.csv", index=False)
    rows = []
    for (obj, ep), g in W2.groupby(["obj", "epoch"]):
        r = dict(obj=obj, epoch=ep, n=len(g), peak_e=g.std_peak_e.median(), peak_pred=g.peak_pred.median(), d_raw=g.d_raw.median(),
                 d_pooled=g.d_pooled.median(), d_m825_meas=g.d_m825_meas.median(), d_m825_pred=g.d_m825_pred.median(),
                 d_field_pred=g.d_field_pred.median() if "d_field_pred" in g else np.nan,
                 d_exp_pred=g.d_exp_pred.median() if "d_exp_pred" in g else np.nan, err=boot_median(g.d_raw))
        if W1 is not None and ep == 2024:
            w = W1[(W1.obj == obj) & (W1["filter"] == "m825") & (W1.surf == "quad")]
            r["d_step1"] = g.d_raw.median() - 1e3 * w.L_curve.median() if len(w) else np.nan
        rows.append(r)
    T = pd.DataFrame(rows)
    T.round(2).to_csv(OUT / "step2_wd.csv", index=False)
    print("=== step 2 white dwarfs\n" + T.round(1).to_string(index=False))
    a24, a26 = T[T.epoch == 2024].set_index("obj"), T[T.epoch == 2026].set_index("obj")
    rows = []
    for c in ("d_raw", "d_pooled", "d_m825_meas", "d_m825_pred", "d_exp_pred", "d_field_pred"):
        d = (a24[c] - a26[c]).dropna(); e = np.hypot(a24.err, a26.err).reindex(d.index)
        rows.append(dict(correction=c, **combine(d.to_numpy(), e.to_numpy()), **{o: round(v, 1) for o, v in d.items()}))
    C2 = pd.DataFrame(rows); C2.round(2).to_csv(OUT / "step2_combined.csv", index=False)
    print("=== step 2: 2024 - 2026 per star and weighted (err_scaled = Birge)\n" + C2.round(1).to_string(index=False))
    fig_e3(B2, L1, W2)
    fig_e4(T[["obj", "epoch", "err", "d_pooled", "d_m825_pred", "d_field_pred"]])
    log("done")


if __name__ == "__main__":
    main()
