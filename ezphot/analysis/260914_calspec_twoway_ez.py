#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260914_calspec_twoway_ez.py  (stage 2 of the CALSPEC redo)

Calibrate the re-photometered CALSPEC frames (stage 1: 260914_calspec_photometry.py, ezphot 10" =
MAG_APER_3) against GAIAXP_CORR_LAMOST corrected-XP field stars synthesized live through TWO curves:

  P0_phys : physical total curve (vendor x f/3 beam x QE x optics x atm), delta_f = 0
  PL_lab  : the same curve rigidly shifted by the lab-measured delta_f (labdelta_vs_fitted.csv 'lab')

and compare each frame's calibrated CALSPEC magnitude to the CALSPEC spectrum synthesized through the
SAME curve -> Delta, DA-minus-other, between-star MAD, plus field colour term and ZP scatter.
Reproduces the F12 panels with the two methods (old pipeline-3" M2/M4 overlaid for reference).

  MPLCONFIGDIR=/tmp/ezphot-matplotlib OMP_NUM_THREADS=1 /home/hhchoi1022/miniconda3/envs/too/bin/python -u \\
      ezphot/analysis/260914_calspec_twoway_ez.py --n-proc 13
"""
from __future__ import annotations

import argparse
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
OT = importlib.import_module("260911_offset_timeseries")           # select_stars_like, C/K/R
OZ = importlib.import_module("260911_offset_ezphot")               # load_ez
C, K, R = OT.C, OT.K, OT.R
XP_WAVE = OT.XP_WAVE
GRID = OT.GRID

OUT = Path("/home/hhchoi1022/ezphot/log/20260914_calspec_ezphot")
CATD = OUT / "catalogs"
ZPC = Path("/home/hhchoi1022/ezphot/log/20260908_zp_color_check")
CALSPEC_DIR = Path("/qso/data3/doabdi/intern/synthetic/calspec_data")
LABCSV = Path("/home/hhchoi1022/ezphot/log/20260910_transmission_sources/labdelta_vs_fitted.csv")
OLD = Path("/home/hhchoi1022/ezphot/log/20260910_transmission_sources")

METHODS = ["P0_phys", "PL_lab", "PV_onsky"]
MIN_FRAMES = 10
MIN_REF = 15
SNR_ZP = 0                      # >0: ZP from stars with SNR>SNR_ZP only (19.2 faint-bias mitigation)
OFFCSV = Path("/home/hhchoi1022/ezphot/log/20260911_offset_timeseries/transmission_offset_needed.csv")

_PHYS, _LAB, _ONSKY, _INFO = None, None, None, None


def log(m):
    print(time.strftime("[%H:%M:%S] ") + m, flush=True)


def star_info():
    """object -> (ra_j2000, dec_j2000, pmra, pmdec, spec_file, is_da)."""
    info = pd.read_csv(Path(__file__).resolve().parent / "calspec_info.csv")
    syn = pd.read_csv(ZPC / "calspec_synphot.csv").drop_duplicates("object")[["object", "spec"]]
    spec_of = dict(zip(syn.object, syn.spec))
    cf = pd.read_csv(ZPC / "calspec_frames.csv")
    sptype = cf.groupby("object")["spectral_type"].first().to_dict()
    out = {}
    for obj, d in K.STANDARD_STARS.items():
        if d.get("csv_name") is None:
            ra, dec, pmra, pmdec = d["ra_j2000"], d["dec_j2000"], d["pmra"], d["pmdec"]
        else:
            m = info[info.star_name_1 == d["csv_name"]].iloc[0]
            ra, dec = float(m["ra_j2000_deg"]), float(m["dec_j2000_deg"])
            pmra = float(m["pmra_masyr"]) if pd.notna(m["pmra_masyr"]) else 0.0
            pmdec = float(m["pmdec_masyr"]) if pd.notna(m["pmdec_masyr"]) else 0.0
        out[obj] = dict(ra=ra, dec=dec, pmra=pmra, pmdec=pmdec, spec=spec_of.get(obj),
                        is_da=str(sptype.get(obj, "")).startswith("DA"))
    return out


def tile_xp_radec(ra0, dec0, want):
    """corrected XP flux for the wanted ids, scanning final_result tiles near (ra0, dec0)."""
    T = C.TILES
    dd = np.abs(T.dec.values - dec0)
    dr = np.abs(((T.ra.values - ra0 + 180) % 360) - 180) * np.cos(np.radians(dec0))
    near = T[(dd < 1.2) & (dr < 1.2)]
    want = set(str(i) for i in want); out = {}
    for tn in near.id.values:
        p = C.TILESRC / f"{tn}.csv"
        if not p.exists() or not want - set(out):
            continue
        with open(p) as fh:
            fh.readline()
            for line in fh:
                cc = line.split(",", 14)
                sid = cc[0].lstrip("﻿")
                if sid in want and sid not in out:
                    vc = np.fromstring(cc[13], sep=";")
                    if vc.size == 343:
                        out[sid] = vc
    return out


def _init(snr_zp=0):
    global _PHYS, _LAB, _ONSKY, _INFO, SNR_ZP
    SNR_ZP = snr_zp
    warnings.filterwarnings("ignore")
    K._refit_init()
    _PHYS = {f: C.load_new_curve(f) for f in GRID}
    lab = pd.read_csv(LABCSV).set_index("filter")["lab"]
    _LAB = {f: float(lab.get(f, 0.0)) for f in GRID}
    ons = pd.read_csv(OFFCSV).set_index("filter")["offset_nm"]
    _ONSKY = {f: float(ons.get(f, 0.0)) for f in GRID}
    _ONSKY["m400"] = 3.65          # std frames are 2024-08: use the 2024-copy delta*, not the 2026+ value
    _INFO = star_info()


def process_object(args):
    """One CALSPEC star: all its frames, both methods."""
    obj, files = args
    st = _INFO[obj]
    cal_spec = None
    if st["spec"]:
        t = Table.read(CALSPEC_DIR / st["spec"])
        wl = np.asarray(t["WAVELENGTH"], float) / 10.0
        fl = np.asarray(t["FLUX"], float) * 1e-2
        m = (wl > 300) & (wl < 1000) & np.isfinite(fl)
        cal_spec = (wl[m], fl[m])
    got = K.get_reference("GAIAXP_CORR_LAMOST", obj, st["ra"], st["dec"], 1.34, 0.90)
    if got is None or cal_spec is None:
        return [], None
    tbl, coords, _ = got
    # synthetic Gaia BP-RP of the standard from its CALSPEC spectrum (MAW passbands reproduce the
    # catalog bp-rp to <0.01 where a match exists; most standards are absent from GAIA_BEST)
    bprp_syn = np.nan
    try:
        import pyphot
        _plib = pyphot.get_library()

        def _vega(bn):
            f = _plib[bn]
            fx = f.get_flux(cal_spec[0] * 10.0, cal_spec[1] * 100.0)   # nm->AA, W/m2/nm->flam
            fx = fx.magnitude if hasattr(fx, "magnitude") else float(fx)
            return -2.5 * np.log10(fx) - float(f.Vega_zero_mag)

        bprp_syn = float(_vega("Gaia_MAW_BP_bright") - _vega("Gaia_MAW_RP"))
    except Exception:
        pass

    # pass 1: match every frame, accumulate the ids we need XP flux for
    prepped, need = [], set()
    for fr in files:
        cp = CATD / (Path(fr).name[:-5] + "_cat.fits")
        if not cp.exists():
            continue
        try:
            with fits.open(cp) as h:
                hdr = h[0].header
                filt = fr.split("/")[-2]
            df = OZ.load_ez(cp)
            with fits.open(cp) as h:
                fx = h["LDAC_OBJECTS"].data
                flux_max = np.asarray(fx["FLUX_MAX"], float) if "FLUX_MAX" in fx.columns.names \
                    else np.full(len(df), np.nan)
        except Exception:
            continue
        if filt not in GRID:
            continue
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
        bprp = np.asarray(mr["bp-rp"], float)
        di = sel[ok]
        # CALSPEC star position at this epoch (pmra includes cos dec)
        mjd = float(hdr.get("MJD", np.nan))
        dt = (mjd - 51544.5) / 365.25 if np.isfinite(mjd) else 24.5
        ra_s = st["ra"] + st["pmra"] * dt / 3.6e6 / np.cos(np.deg2rad(st["dec"]))
        dec_s = st["dec"] + st["pmdec"] * dt / 3.6e6
        cstar = SkyCoord(ra_s, dec_s, unit="deg")
        star_bprp = bprp_syn
        call = SkyCoord(df.ALPHA_J2000.to_numpy(), df.DELTA_J2000.to_numpy(), unit="deg")
        sep = call.separation(cstar).arcsec
        j = int(np.argmin(sep))
        srow = df.iloc[j]
        sat = float(hdr.get("SATURATE", np.nan))
        star_ok = (sep[j] < 1.5 and srow.FLAGS <= 1 and srow.MAGERR_APER_3 < 0.05
                   and np.isfinite(flux_max[j]) and flux_max[j] < 45000.0
                   and (not np.isfinite(sat) or flux_max[j] < sat))
        prepped.append(dict(file=fr, filt=filt, mjd=mjd, unit=fr.split("/")[-3],
                            rid=rid, bprp=bprp, snr=di.SNR_APER_3.to_numpy(),
                            inst=di.MAG_APER_3.to_numpy(), ierr=di.MAGERR_APER_3.to_numpy(),
                            star_bprp=star_bprp,
                            star_ok=bool(star_ok), star_inst=float(srow.MAG_APER_3),
                            star_err=float(srow.MAGERR_APER_3), sep=float(sep[j])))
        need.update(rid)
    if not prepped:
        return [], None

    xp = tile_xp_radec(st["ra"], st["dec"], need)
    rows, pool_res = [], []
    sc = SigmaClip(sigma=3.0, maxiters=5)
    for pr in prepped:
        keep = np.array([r in xp for r in pr["rid"]])
        if keep.sum() < MIN_REF:
            continue
        fc = np.vstack([xp[r] for r in pr["rid"][keep]])
        bp = pr["bprp"][keep]
        inst = pr["inst"][keep]
        snrk = pr["snr"][keep]
        sig = np.sqrt(pr["ierr"][keep] ** 2 + 0.01 ** 2)
        fw, ft = _PHYS[pr["filt"]]
        for method in METHODS:
            shift = {"P0_phys": 0.0, "PL_lab": _LAB[pr["filt"]], "PV_onsky": _ONSKY[pr["filt"]]}[method]
            syn = R.syn_mag(XP_WAVE, fc, fw, ft, shift)
            zp_arr = syn - inst
            fin = np.isfinite(zp_arr)
            zsel = (fin & (snrk > SNR_ZP)) if SNR_ZP > 0 else fin
            if fin.sum() < MIN_REF or zsel.sum() < 10:
                continue
            clm = sc(zp_arr[zsel])
            zp = float(np.ma.median(clm))
            zsd = float(np.ma.std(clm))
            if not np.isfinite(zp) or zsd > 0.08:
                continue
            cla = sc(zp_arr[fin])
            good = ~np.ma.getmaskarray(cla)
            resid = zp_arr[fin][good] - zp
            row = dict(file=pr["file"], object=obj, filter=pr["filt"], unit=pr["unit"],
                       mjd=pr["mjd"], method=method, n_ref=int(good.sum()),
                       n_zp=int((~np.ma.getmaskarray(clm)).sum()),
                       zp=round(zp, 4), zp_std=round(zsd, 4))
            # per-frame colour term of the ZP stars (clipped, weighted) -> colour-matched ZP for the standard
            gz = ~np.ma.getmaskarray(clm)
            bp_z, zp_z, sg_z = bp[zsel][gz], zp_arr[zsel][gz], sig[zsel][gz]
            aa, ae, bb, nf = R.weighted_linfit(bp_z, zp_z - zp, sg_z)
            c_ref = float(np.average(bp_z, weights=1.0 / sg_z ** 2))
            row["a_frame"] = round(float(aa), 4)
            row["c_ref"] = round(c_ref, 3)
            if pr["star_ok"]:
                m_true = float(R.syn_mag(cal_spec[0], cal_spec[1], fw, ft, shift))
                row["delta"] = round(pr["star_inst"] + zp - m_true, 4)
                row["star_err"] = pr["star_err"]
                if np.isfinite(pr["star_bprp"]) and np.isfinite(aa):
                    row["c_star"] = round(pr["star_bprp"], 3)
                    row["delta_cc"] = round(row["delta"] + float(aa) * (pr["star_bprp"] - c_ref), 4)
            rows.append(row)
            k = min(len(resid), 200)
            pool_res.append(pd.DataFrame(dict(
                filter=pr["filt"], method=method,
                bprp=bp[fin][good][:k], resid=resid[:k], sig=sig[fin][good][:k])))
    pool_df = pd.concat(pool_res, ignore_index=True) if pool_res else None
    return rows, pool_df


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n-proc", type=int, default=13)
    ap.add_argument("--snrmin", type=float, default=0, help="ZP from stars with SNR>this only (0 = all)")
    a = ap.parse_args()
    sfx = f"_snr{a.snrmin:.0f}" if a.snrmin > 0 else ""
    meta = pd.read_csv(OUT / "photometry_meta.csv")
    meta = meta[meta.status == "ok"]
    byobj = [(o, sorted(g.file)) for o, g in meta.groupby("object")]
    log(f"{len(meta)} frames, {len(byobj)} objects, {a.n_proc} workers, ZP SNR cut {a.snrmin}")
    all_rows, pools = [], []
    with Pool(min(a.n_proc, len(byobj)), initializer=_init, initargs=(a.snrmin,)) as pool:
        for rows, pdf in pool.imap_unordered(process_object, byobj):
            all_rows += rows
            if pdf is not None:
                pools.append(pdf)
            log(f"  +{len(rows)} rows (total {len(all_rows)})")
    fr = pd.DataFrame(all_rows)
    fr.to_csv(OUT / f"calspec_ez_frames{sfx}.csv", index=False)
    pool_df = pd.concat(pools, ignore_index=True)

    _init(a.snrmin)                                           # is_da map in main process
    is_da = {o: v["is_da"] for o, v in _INFO.items()}

    # ---- field summary: pooled colour term + ZP scatter ----
    field_rows = []
    for (f, method), g in fr.groupby(["filter", "method"]):
        pg = pool_df[(pool_df["filter"] == f) & (pool_df["method"] == method)]
        a_, a_err, b, nfit = R.weighted_linfit(pg.bprp.to_numpy(), pg.resid.to_numpy(), pg.sig.to_numpy())
        field_rows.append(dict(filter=f, method=method, n_frames=int(g.file.nunique()),
                               zp_std=round(float(g.zp_std.median()), 4),
                               a_bprp=round(a_, 4), a_err=round(a_err, 4)))
    field = pd.DataFrame(field_rows)
    field.to_csv(OUT / f"field_ez_summary{sfx}.csv", index=False)

    # ---- CALSPEC summary ----
    cal_rows, star_rows = [], []
    cf = fr.dropna(subset=["delta"])
    for (f, method), g in cf.groupby(["filter", "method"]):
        per = g.groupby("object").agg(n=("delta", "size"), d=("delta", "median"),
                                      scat=("delta", R.mad))
        per = per[per.n >= MIN_FRAMES]
        if not len(per):
            continue
        per["is_da"] = [is_da.get(o, False) for o in per.index]
        for o, r_ in per.iterrows():
            star_rows.append(dict(filter=f, method=method, object=o, is_da=bool(r_.is_da),
                                  n=int(r_.n), delta=round(float(r_.d), 4)))
        da = per[per.is_da].d; ot_ = per[~per.is_da].d
        cal_rows.append(dict(filter=f, method=method, n_stars=len(per),
                             median_all=round(float(per.d.median()), 4),
                             da_minus_other=round(float(da.median() - ot_.median()), 4)
                             if len(da) and len(ot_) else np.nan,
                             between_star_mad=round(float(R.mad(per.d.values)), 4) if len(per) >= 3 else np.nan))
    cal = pd.DataFrame(cal_rows)
    cal.to_csv(OUT / f"calspec_ez_summary{sfx}.csv", index=False)
    pd.DataFrame(star_rows).to_csv(OUT / f"calspec_ez_stars{sfx}.csv", index=False)
    log("CALSPEC summary:")
    print(cal.pivot_table(index="filter", columns="method",
                          values=["median_all", "da_minus_other", "between_star_mad"]).round(4).to_string())

    # ---- colour-term-corrected summary (delta_cc = delta + a_frame*(c_star - c_ref)) ----
    if "delta_cc" in fr.columns:
        cc_rows, ccstar_rows = [], []
        cfc = fr.dropna(subset=["delta_cc"])
        for (f, method), g in cfc.groupby(["filter", "method"]):
            per = g.groupby("object").agg(n=("delta_cc", "size"), d=("delta_cc", "median"))
            per = per[per.n >= MIN_FRAMES]
            if not len(per):
                continue
            per["is_da"] = [is_da.get(o, False) for o in per.index]
            for o, r_ in per.iterrows():
                ccstar_rows.append(dict(filter=f, method=method, object=o, is_da=bool(r_.is_da),
                                        n=int(r_.n), delta=round(float(r_.d), 4)))
            da = per[per.is_da].d; ot_ = per[~per.is_da].d
            cc_rows.append(dict(filter=f, method=method, n_stars=len(per),
                                median_all=round(float(per.d.median()), 4),
                                da_minus_other=round(float(da.median() - ot_.median()), 4)
                                if len(da) and len(ot_) else np.nan,
                                between_star_mad=round(float(R.mad(per.d.values)), 4) if len(per) >= 3 else np.nan))
        calcc = pd.DataFrame(cc_rows)
        calcc.to_csv(OUT / f"calspec_ez_summary_cc{sfx}.csv", index=False)
        pd.DataFrame(ccstar_rows).to_csv(OUT / f"calspec_ez_stars_cc{sfx}.csv", index=False)
        log("CALSPEC colour-term-corrected summary:")
        print(calcc.pivot_table(index="filter", columns="method",
                                values=["median_all", "da_minus_other", "between_star_mad"]).round(4).to_string())

    # ---- F48: F12-style panels (P0 vs PL, old pipeline-3" M2/M4 as faded reference) ----
    old_f = pd.read_csv(OLD / "fourway_field.csv")
    old_c = pd.read_csv(OLD / "fourway_calspec.csv")
    for d in (field, cal, old_f, old_c):
        d["pivot"] = pd.to_numeric(d["filter"].str[1:], errors="coerce")
    colors = {"P0_phys": "#d62728", "PL_lab": "#ff7f0e", "PV_onsky": "#2ca02c"}
    oldc = {"M2_new_corr": "#999999", "M4_old_corr": "#bbbbbb"}
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for method in METHODS:
        fm = field[field.method == method].sort_values("pivot")
        axes[0, 0].plot(fm["pivot"], fm.a_bprp, "o-", ms=4, color=colors[method], label=method)
        axes[0, 1].plot(fm["pivot"], fm.zp_std * 1000, "o-", ms=4, color=colors[method], label=method)
    for method in oldc:
        fm = old_f[old_f.method == method].sort_values("pivot")
        axes[0, 0].plot(fm["pivot"], fm.a_bprp, "--", lw=1, color=oldc[method], label=method + ' (old ezphot 10\", bright)')
        axes[0, 1].plot(fm["pivot"], fm.zp_std * 1000, "--", lw=1, color=oldc[method], label=method + ' (old ezphot 10\", bright)')
    axes[0, 0].axhline(0, color="k", lw=0.5)
    axes[0, 0].set(xlabel="pivot [nm]", ylabel="colour term a(BP-RP)", title="field-star colour term")
    axes[0, 1].set(xlabel="pivot [nm]", ylabel="ZP scatter [mmag]", title="field-star ZP scatter")
    for method in METHODS:
        cm = cal[cal.method == method].sort_values("pivot")
        axes[1, 0].plot(cm["pivot"], cm.da_minus_other, "o-", ms=4, color=colors[method], label=method)
        axes[1, 1].plot(cm["pivot"], cm.between_star_mad * 1000, "o-", ms=4, color=colors[method], label=method)
    for method in oldc:
        cm = old_c[old_c.method == method].sort_values("pivot")
        axes[1, 0].plot(cm["pivot"], cm.da_minus_other, "--", lw=1, color=oldc[method], label=method + ' (old ezphot 10\", bright)')
        axes[1, 1].plot(cm["pivot"], cm.between_star_mad * 1000, "--", lw=1, color=oldc[method],
                        label=method + ' (old ezphot 10\", bright)')
    axes[1, 0].axhline(0, color="k", lw=0.5)
    axes[1, 0].set(xlabel="pivot [nm]", ylabel="DA - other [mag]", title="CALSPEC DA white dwarfs minus other standards")
    axes[1, 1].set(xlabel="pivot [nm]", ylabel="between-star MAD [mmag]", title="CALSPEC between-star scatter")
    for ax in axes.ravel():
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    zptag = f"ZP from SNR>{a.snrmin:.0f} stars" if a.snrmin > 0 else "ZP from all stars (SNR 10-150)"
    fig.suptitle(f"F48  CALSPEC redo with ezphot 10\" photometry, {zptag}: physical vs physical+lab "
                 "(dashed grey = 2026-09-10 fourway, also ezphot 10\", bright-star sample)")
    fig.tight_layout(); fig.savefig(OUT / "F48_calspec_twoway.png", dpi=145)

    # ---- F49: per-star Delta vs pivot ----
    stars = pd.DataFrame(star_rows)
    stars["pivot"] = pd.to_numeric(stars["filter"].str[1:], errors="coerce")
    fig, ax = plt.subplots(1, len(METHODS), figsize=(7 * len(METHODS) + 1, 6), sharey=True)
    for k, method in enumerate(METHODS):
        sm = stars[stars.method == method]
        for o, g in sm.groupby("object"):
            g = g.sort_values("pivot")
            ax[k].plot(g["pivot"], g.delta, "o-" if is_da.get(o) else "s--", ms=4, lw=0.9,
                       alpha=0.85, label=o)
        ax[k].axhline(0, color="k", lw=0.6)
        ax[k].set(xlabel="pivot [nm]", title=method)
        ax[k].grid(alpha=0.3)
    ax[0].set_ylabel("Delta = calibrated obs - CALSPEC synthetic [mag]")
    ax[1].legend(fontsize=7, ncol=2)
    fig.suptitle(f"F49  per-star CALSPEC magnitude differences ({zptag}; circles/solid = DA white dwarfs)")
    fig.tight_layout(); fig.savefig(OUT / "F49_calspec_perstar.png", dpi=145)
    log(f"saved -> calspec_ez_{{frames,summary,stars}}{sfx}.csv, field_ez_summary{sfx}.csv, F48/F49 in {OUT}")


if __name__ == "__main__":
    main()
