#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260921_truthcompare.py — calibrated-magnitude difference of the standards against THREE truths
(raw XP / corrected XP / ground spectrum), for each transmission-curve version.

ZP is always from corrected-XP field stars (SNR>40, clipped median, zsd<=0.08), so for a
standard star:  Delta_truth = m_inst(std) + ZP_version - syn_truth,version(std).

  - CALSPEC images (20260914_calspec_ezphot): truths raw XP / corrected XP / CALSPEC spectrum;
    curve versions manufacturer, version_mhj (old-gen GRID filters).
  - SPSS images (20260918_spss): truths raw XP / corrected XP / SPSS V2 flux table;
    curve versions manufacturer, version_mhj (old-gen bands), version_hyp, version_ky (new bands).

Totals = version filter curve x QE x optics x El Sauce atm (no beam). m832w excluded (defective).

  MPLCONFIGDIR=/tmp/ezphot-matplotlib OMP_NUM_THREADS=1 /home/hhchoi1022/miniconda3/envs/too/bin/python -u \
      ezphot/analysis/260921_truthcompare.py --n-proc 12
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
OZ = importlib.import_module("260911_offset_ezphot")
AN = importlib.import_module("260919_spss_analysis")
TW = importlib.import_module("260914_calspec_twoway_ez")
C, K, R = OT.C, OT.K, OT.R
XP_WAVE = OT.XP_WAVE

VBASE = Path("/home/hhchoi1022/ezphot/data/transmission_versions")
SRC = Path("/home/hhchoi1022/ezphot/config/common/transmission/Original_sources")
TS = Path("/home/hhchoi1022/ezphot/log/20260910_transmission_sources")
OUT = Path("/home/hhchoi1022/ezphot/log/20260918_spss")
CAL_CATD = Path("/home/hhchoi1022/ezphot/log/20260914_calspec_ezphot/catalogs")
CAL_META = Path("/home/hhchoi1022/ezphot/log/20260914_calspec_ezphot/photometry_meta.csv")
VERSIONS = ["manufacturer", "version_mhj", "version_ky", "version_hyp"]
EXCLUDE_UNITS = {"7DT11"}
BAD_FILTERS = {"m832w"}
MIN_REF = 10
FRAME_CAP = 6                      # per (object, filter)

_TOTALS = None                     # version -> {filter: (lam, total)}


def log(m):
    print(time.strftime("[%H:%M:%S] ") + m, flush=True)


def build_totals():
    atm = pd.read_csv(TS / "atmosphere_elsauce.csv")
    qe = pd.read_csv(SRC / "QE.csv")
    opt = pd.read_csv(SRC / "optics.efficiency.dr350.dr500.csv", sep=r"\s+", comment="#", skiprows=11)
    out = {}
    for ver in VERSIONS:
        out[(ver, "nobeam")] = {}
        out[(ver, "beam")] = {}
        for p in sorted((VBASE / ver).glob("m*.csv")):
            f = p.stem
            if f in BAD_FILTERS:
                continue
            d = pd.read_csv(p)
            lam = d.lam_nm.to_numpy(float)
            tr = d.trans.to_numpy(float)
            env = (np.interp(lam, qe.iloc[:, 0], qe.iloc[:, 1])
                   * np.interp(lam, opt.iloc[:, 0], opt.iloc[:, -1])
                   * np.interp(lam, atm.iloc[:, 0], atm["trans"] if "trans" in atm.columns
                               else atm.iloc[:, 1]))
            out[(ver, "nobeam")][f] = (lam, tr * env)
            te = dr500_transmission(lam, tr, n_eff=1.5, f_number=3.03, obscuration=0.59)
            out[(ver, "beam")][f] = (lam, te * env)
    return out


def _init():
    global _TOTALS
    warnings.filterwarnings("ignore")
    AN._init()
    TW._init(40)
    _TOTALS = build_totals()


def star_xp_by_pos(ra0, dec0, pmra, pmdec):
    """(raw343, cor343) of the star nearest the PM-corrected position in GAIA_BEST tiles."""
    T = C.TILES
    dd = np.abs(T.dec.values - dec0)
    dr = np.abs(((T.ra.values - ra0 + 180) % 360) - 180) * np.cos(np.radians(dec0))
    for dt in (25.0, 0.0):
        ra_s = ra0 + pmra * dt / 3.6e6 / np.cos(np.deg2rad(dec0))
        dec_s = dec0 + pmdec * dt / 3.6e6
        for tn in T[(dd < 0.75) & (dr < 0.75)].id.values:
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
                    sep = np.hypot((ra - ra_s) * np.cos(np.deg2rad(dec0)), dec - dec_s) * 3600
                    if sep < 3.0:
                        vo = np.fromstring(cc[11], sep=";")
                        vc = np.fromstring(cc[13], sep=";")
                        if vo.size == 343 and vc.size == 343:
                            return vo, vc
    return None


def frame_deltas(filt, fc, inst, snr, star_inst, truth_specs):
    """rows over versions: ZP(corrected XP) + standard Delta per truth."""
    sc = SigmaClip(sigma=3.0, maxiters=5)
    rows = []
    for (ver, var), curves in _TOTALS.items():
        if filt not in curves:
            continue
        fw, ft = curves[filt]
        zp_arr = R.syn_mag(XP_WAVE, fc, fw, ft, 0.0) - inst
        zsel = np.isfinite(zp_arr) & (snr > 40.0)
        if zsel.sum() < MIN_REF:
            continue
        cl = sc(zp_arr[zsel])
        zp = float(np.ma.median(cl))
        if not np.isfinite(zp) or float(np.ma.std(cl)) > 0.08:
            continue
        row = dict(version=ver, variant=var, zp=round(zp, 4))
        for tag, spec in truth_specs.items():
            if spec is None:
                continue
            if tag in ("raw", "cor"):
                m_t = float(R.syn_mag(XP_WAVE, spec[None, :], fw, ft, 0.0)[0])
            else:
                m_t = float(R.syn_mag(spec[0], spec[1], fw, ft, 0.0))
            row[f"d_{tag}"] = round(star_inst + zp - m_t, 4)
        rows.append(row)
    return rows


# ---------------- SPSS images ----------------
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
    xp = star_xp_by_pos(ref["ra"][j], ref["dec"][j], ref["pmra"][j], ref["pmdec"][j])
    truth = {"gnd": spec, "cor": ref["flux"][j], "raw": xp[0] if xp else None}
    ra_s, dec_s, pma, pmd = ref["ra"][j], ref["dec"][j], ref["pmra"][j], ref["pmdec"][j]
    coords = SkyCoord(ref["ra"], ref["dec"], unit="deg")
    rows = []
    for fr in files:
        cp = AN.CATD / (Path(fr).name[:-5] + "_cat.fits")
        filt = Path(fr).name.split("_")[-4]
        unit = Path(fr).name.split("_")[0]
        try:
            with fits.open(cp) as h:
                hdr = h[0].header
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
            dt = (mjd - AN.GAIA_EPOCH_MJD) / 365.25 if np.isfinite(mjd) else 10.7
            pos = SkyCoord(ra_s + pma * dt / 3.6e6 / np.cos(np.deg2rad(dec_s)),
                           dec_s + pmd * dt / 3.6e6, unit="deg")
            call = SkyCoord(df.ALPHA_J2000.to_numpy(), df.DELTA_J2000.to_numpy(), unit="deg")
            sep = call.separation(pos).arcsec
            js = int(np.argmin(sep))
            st = df.iloc[js]
            if not (sep[js] < 2.0 and st.FLAGS <= 1 and st.MAGERR_APER_3 < 0.05):
                continue
            for r in frame_deltas(filt, ref["flux"][mi],
                                  sel.MAG_APER_3.to_numpy()[ok], sel.SNR_APER_3.to_numpy()[ok],
                                  float(st.MAG_APER_3), truth):
                rows.append(dict(dataset="SPSS", object=obj, filter=filt, unit=unit, **r))
        except Exception:                                    # noqa: BLE001
            continue
    return rows


# ---------------- CALSPEC images ----------------
def process_cal(args):
    obj, files = args
    st = TW._INFO[obj]
    if not st["spec"]:
        return []
    t = Table.read(TW.CALSPEC_DIR / st["spec"])
    wl = np.asarray(t["WAVELENGTH"], float) / 10.0
    fl = np.asarray(t["FLUX"], float) * 1e-2
    m = (wl > 300) & (wl < 1100) & np.isfinite(fl) & (fl > 0)
    cal_spec = (wl[m], fl[m])
    got = K.get_reference("GAIAXP_CORR_LAMOST", obj, st["ra"], st["dec"], 1.34, 0.90)
    if got is None:
        return []
    tbl, coords, _ = got
    xp = star_xp_by_pos(st["ra"], st["dec"], st["pmra"], st["pmdec"])
    truth = {"gnd": cal_spec, "cor": xp[1] if xp else None, "raw": xp[0] if xp else None}
    rows, need, prepped = [], set(), []
    for fr in files:
        cp = CAL_CATD / (Path(fr).name[:-5] + "_cat.fits")
        if not cp.exists():
            continue
        filt = fr.split("/")[-2]
        unit = fr.split("/")[-3]
        try:
            with fits.open(cp) as h:
                hdr = h[0].header
            df = OZ.load_ez(cp)
            sel = OT.select_stars_like(df)
            if len(sel) < MIN_REF:
                continue
            src = SkyCoord(sel.ALPHA_J2000.to_numpy(), sel.DELTA_J2000.to_numpy(), unit="deg")
            idx, d2d, _ = match_coordinates_sky(src, coords)
            ok = np.asarray(d2d.arcsec < 2.0)
            if ok.sum() < MIN_REF:
                continue
            mr = tbl[idx[ok]]
            rid = np.asarray(mr["id"]).astype(str)
            mjd = float(hdr.get("MJD", np.nan))
            dt = (mjd - 51544.5) / 365.25 if np.isfinite(mjd) else 24.5
            pos = SkyCoord(st["ra"] + st["pmra"] * dt / 3.6e6 / np.cos(np.deg2rad(st["dec"])),
                           st["dec"] + st["pmdec"] * dt / 3.6e6, unit="deg")
            call = SkyCoord(df.ALPHA_J2000.to_numpy(), df.DELTA_J2000.to_numpy(), unit="deg")
            sep = call.separation(pos).arcsec
            js = int(np.argmin(sep))
            srow_ = df.iloc[js]
            if not (sep[js] < 1.5 and srow_.FLAGS <= 1 and srow_.MAGERR_APER_3 < 0.05):
                continue
            prepped.append(dict(filt=filt, unit=unit, rid=rid,
                                inst=sel.MAG_APER_3.to_numpy()[ok],
                                snr=sel.SNR_APER_3.to_numpy()[ok],
                                star_inst=float(srow_.MAG_APER_3)))
            need.update(rid)
        except Exception:                                    # noqa: BLE001
            continue
    if not prepped:
        return []
    xpf = TW.tile_xp_radec(st["ra"], st["dec"], need)
    for pr in prepped:
        keep = np.array([r in xpf for r in pr["rid"]])
        if keep.sum() < MIN_REF:
            continue
        fc = np.vstack([xpf[r] for r in pr["rid"][keep]])
        for r in frame_deltas(pr["filt"], fc, pr["inst"][keep], pr["snr"][keep],
                              pr["star_inst"], truth):
            rows.append(dict(dataset="CALSPEC", object=obj, filter=pr["filt"],
                             unit=pr["unit"], **r))
    return rows


def cap_frames(pairs):
    """pairs: list of (key, file). Keep at most FRAME_CAP per key, spread evenly."""
    out = {}
    for k, f in pairs:
        out.setdefault(k, []).append(f)
    kept = []
    for k, fl in out.items():
        fl = sorted(fl)
        step = max(1, len(fl) // FRAME_CAP)
        kept += fl[::step][:FRAME_CAP]
    return kept


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n-proc", type=int, default=12)
    a = ap.parse_args()
    # SPSS jobs
    spss = pd.read_csv(AN.SPSS_CSV)
    srow_of = {AN.norm(s.Name): s for _, s in spss.iterrows()}
    meta = pd.concat([pd.read_csv(m) for m in glob.glob(str(OUT / "process_meta_7DT*.csv"))],
                     ignore_index=True)
    meta = meta[(meta.status == "ok") & ~meta.unit.isin(EXCLUDE_UNITS)
                & ~meta["filter"].isin(BAD_FILTERS)]
    jobs_s = []
    for obj, g in meta.groupby("object"):
        s = srow_of.get(AN.norm(obj))
        if s is None:
            continue
        files = cap_frames(list(zip(g["filter"], g.file)))
        jobs_s.append((obj, files, s))
    # CALSPEC jobs
    cmeta = pd.read_csv(CAL_META)
    cmeta = cmeta[cmeta.status == "ok"]
    jobs_c = []
    for obj, g in cmeta.groupby("object"):
        files = cap_frames([(f.split("/")[-2], f) for f in g.file])
        jobs_c.append((obj, files))
    log(f"SPSS: {len(jobs_s)} objects; CALSPEC: {len(jobs_c)} objects (cap {FRAME_CAP}/filter)")
    rows = []
    with Pool(a.n_proc, initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(process_spss, jobs_s), 1):
            rows += r
            log(f"  SPSS [{i}/{len(jobs_s)}] total {len(rows)}")
        for i, r in enumerate(pool.imap_unordered(process_cal, jobs_c), 1):
            rows += r
            log(f"  CAL  [{i}/{len(jobs_c)}] total {len(rows)}")
    fr = pd.DataFrame(rows)
    fr.to_csv(OUT / "truthcompare_frames.csv", index=False)

    # star-level then filter-level medians
    res = []
    for (ds, f, ver, var), g in fr.groupby(["dataset", "filter", "version", "variant"]):
        row = dict(dataset=ds, filter=f, version=ver, variant=var, pivot=int(f[1:4]),
                   n_frames=len(g), n_stars=g.object.nunique())
        for tag in ("gnd", "cor", "raw"):
            c = f"d_{tag}"
            if c not in g.columns:
                continue
            per = g.dropna(subset=[c]).groupby("object")[c].median()
            if len(per) < 2:
                continue
            row[f"{tag}_mmag"] = round(1e3 * float(per.median()), 1)
            row[f"{tag}_err"] = round(1e3 * 1.4826 * float((per - per.median()).abs().median())
                                      / np.sqrt(len(per)), 1)
            row[f"{tag}_n"] = len(per)
        res.append(row)
    T = pd.DataFrame(res).sort_values(["dataset", "pivot", "version", "variant"])
    T.to_csv(OUT / "truthcompare_summary.csv", index=False)
    print(T.to_string(index=False))

    # ---- figure T1: three truths, manufacturer curve (version effect drawn separately later) ----
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 10.5), sharex=True)
    tru = [("gnd", "#1f4e8c"), ("cor", "#C25B4E"), ("raw", "#E09A3E")]
    for ax, ds, gl in [(axes[0], "CALSPEC", "CALSPEC spectrum"),
                       (axes[1], "SPSS", "SPSS flux table")]:
        man = T[(T.dataset == ds) & (T.version == "manufacturer") & (T.variant == "nobeam")]
        for tag, col in tru:
            lab = {"gnd": gl, "cor": "corrected XP", "raw": "raw XP"}[tag]
            g = man.dropna(subset=[f"{tag}_mmag"]).sort_values("pivot")
            ax.errorbar(g["pivot"], g[f"{tag}_mmag"], yerr=g[f"{tag}_err"], ls="-", marker="o",
                        ms=4.5, lw=1.5, color=col, label=lab)
        ax.axhline(0, color="k", lw=0.7); ax.grid(alpha=0.3)
        ax.set_ylabel("m_cal - syn_truth [mmag]")
        ax.set_title(f"{ds} images - manufacturer curve, ZP from corrected-XP field stars",
                     fontsize=11.5)
        ax.legend(fontsize=9.5)
    axes[0].set_ylim(-35, 55)
    axes[1].set_ylim(-50, 50)          # m375w (+118) clips off; annotated instead
    axes[1].annotate("m375w +118 (off scale)", (378, 47), fontsize=8.5, ha="left",
                     va="top", color="0.35")
    axes[1].set_xlabel("pivot wavelength [nm]")
    fig.suptitle("T1  Standards vs three truths (manufacturer curve)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "T1_truthcompare.png", dpi=140)
    log("saved -> truthcompare_summary.csv, truthcompare_frames.csv, T1_truthcompare.png")


if __name__ == "__main__":
    main()
