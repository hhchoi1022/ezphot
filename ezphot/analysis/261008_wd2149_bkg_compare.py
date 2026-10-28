#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
261008_wd2149_bkg_compare.py — WD2149+021 photometry with the two background methods, same frames, same SExtractor run
settings and the same 10" aperture (MAG_APER_3):
  old : SExtractor's own background (ezphot photometry_sex default; 2024 catalogues of 260914_calspec_photometry.py in
        20260914_calspec_ezphot/catalogs, 2026 SPSS catalogues of 260918_spss_process.py in 20260918_spss/catalogs)
  new : photutils per-mesh 5-sigma-clipped mean with a source mask (260927_std_rephot.py, 20260927_std_newbkg/catalogs)
Frames: WD2149+021 / G93-48, 2024 (7DT10 m800 m825, 7DT11 m850) and 2026 (7DT07 m800, 7DT12 m825, 7DT09 m850).
Per method:
  1. Delta of the white dwarf per frame (SO.process_cal: global field ZP from corrected XP, manufacturer beam curve), no
     low-signal correction (Delta_raw);
  2. the low-signal loss of field stars in the same frames versus the predicted (noise-free) peak (as 261008_m825_pairs
     step 2), and the white dwarf's excess over that curve at its own noise-free peak;
  3. same stars in both catalogues: dm = MAG_old - MAG_new, and the implied background difference per pixel
     dB = (F_old - F_new) / area  (= B_new - B_old), field stars versus the white dwarf.
Outputs: ~/ezphot/log/20261008_wd2149_bkg/ (frames.csv, field_stars.csv, pairs.csv, summary.csv, B1_bkg.png)
"""
import importlib
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402
from astropy.coordinates import SkyCoord, match_coordinates_sky   # noqa: E402

sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
P = importlib.import_module("261008_m825_pairs")
SO, SS = P.SO, P.SS
OUT = Path("/home/hhchoi1022/ezphot/log/20261008_wd2149_bkg")
LOG = Path("/home/hhchoi1022/ezphot/log")
CATS = {("old", 2024): LOG / "20260914_calspec_ezphot/catalogs", ("old", 2026): LOG / "20260918_spss/catalogs",
        ("new", 2024): LOG / "20260927_std_newbkg/catalogs", ("new", 2026): LOG / "20260927_std_newbkg/catalogs"}
AREA = np.pi * 9.9 ** 2


def _init():
    P._init()
    SO._TOTALS = {k: v for k, v in SO._TOTALS.items() if k == ("manufacturer", "beam")}
    P._init_xp()


def job(args):
    method, epoch, f, flt, unit = args
    SO.CATD = CATS[(method, epoch)]
    if not (SO.CATD / (Path(f).name[:-5] + "_cat.fits")).exists():
        return [], None
    rows = SO.process_cal(("WD2149+021", [(f, flt, unit)]))
    xs = P.xp_frame(("CALSPEC", f, "WD2149+021", flt))
    if xs is not None:
        xs = xs.assign(method=method, filter=flt)
    return [dict(r, method=method, epoch=epoch) for r in rows], xs


def pair_frame(args):
    epoch, f = args
    try:
        a, _ = SO.load_cat(CATS[("old", epoch)] / (Path(f).name[:-5] + "_cat.fits"))
        b, meta = SO.load_cat(CATS[("new", epoch)] / (Path(f).name[:-5] + "_cat.fits"))
    except Exception:                                      # noqa: BLE001
        return None
    a = a[(a.FLAGS == 0) & (a.MAG_APER_3 < 90)]; b = b[(b.FLAGS == 0) & (b.MAG_APER_3 < 90)]
    idx, d2d, _ = match_coordinates_sky(SkyCoord(b.ALPHA_J2000.to_numpy(), b.DELTA_J2000.to_numpy(), unit="deg"),
                                        SkyCoord(a.ALPHA_J2000.to_numpy(), a.DELTA_J2000.to_numpy(), unit="deg"))
    ok = d2d.arcsec < 1.0
    bb, aa = b[ok].reset_index(drop=True), a.iloc[idx[ok]].reset_index(drop=True)
    f_old, f_new = 10 ** (-0.4 * aa.MAG_APER_3.to_numpy()), 10 ** (-0.4 * bb.MAG_APER_3.to_numpy())
    return pd.DataFrame(dict(file=Path(f).name, x=bb.X, y=bb.Y, ra=bb.ALPHA_J2000, dec=bb.DELTA_J2000, snr_new=bb.SNR_APER_3,
                             peak_e=bb.FLUX_MAX * meta["egain"], mag_new=bb.MAG_APER_3, dm=1e3 * (aa.MAG_APER_3.to_numpy() - bb.MAG_APER_3.to_numpy()),
                             dB=(f_old - f_new) / AREA))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    D = pd.read_csv(P.OUT / "frames_delta.csv")
    d = D[(D.obj == "WD2149+021") & (D.curve == "beam")].drop_duplicates("file")
    jobs = [(m, int(r.epoch), r.file, r["filter"], r.unit) for _, r in d.iterrows() for m in ("old", "new")]
    rows, xs = [], []
    with Pool(16, initializer=_init) as pool:
        for rr, x in pool.imap_unordered(job, jobs):
            rows += rr
            if x is not None:
                xs.append(x)
    F = pd.DataFrame(rows)
    F["d_raw"] = (F.d_gnd - F.zp + F.zp_x) * 1e3
    F["ok"] = (F.std_flags == 0) & (F.std_peak <= SS.PEAK_MAX)
    X = pd.concat(xs, ignore_index=True)
    X = X[np.isfinite(X.loss) & (X.loss.abs() < 1000)]
    X.to_csv(OUT / "field_stars.csv", index=False)
    # white dwarf noise-free peak per frame and method (frame PSF scale from the same method's field stars)
    ks = X.groupby(["method", "file"]).kscale.first()
    F["kscale"] = [ks.get((m, f), np.nan) for m, f in zip(F.method, F.file)]
    m_syn = F.std_mag + F.zp - F.d_gnd
    F["peak_pred"] = F.kscale * 10 ** (-0.4 * (m_syn - F.zp_x))
    out = []
    for (m, ep, flt), g in F[F.ok].groupby(["method", "epoch", "filter"]):
        x = X[(X.method == m) & (X.epoch == ep) & (X["filter"] == flt)]
        B = P.binned(x, "peak_pred")
        L_wd = P.interp_loss(B, g.peak_pred.to_numpy()) if len(P.reliable(B)) >= 3 else np.full(len(g), np.nan)
        F.loc[g.index, "L_field"] = L_wd
        F.loc[g.index, "excess"] = g.d_raw.to_numpy() - L_wd
        b = B.set_index("lo")
        out.append(dict(method=m, epoch=ep, filter=flt, unit=",".join(sorted(set(g.unit))), n=len(g), peak_pred=g.peak_pred.median(),
                        d_raw=g.d_raw.median(), err=P.boot_median(g.d_raw), L_field_at_wd=float(np.median(L_wd)),
                        excess=float(np.median(g.d_raw.to_numpy() - L_wd)), err_excess=P.boot_median(g.d_raw.to_numpy() - L_wd),
                        **{f"L_{int(lo)}": b.loss.get(lo, np.nan) for lo in (8, 16, 32, 64, 128, 256)}))
    S = pd.DataFrame(out).sort_values(["epoch", "filter", "method"])
    F.to_csv(OUT / "frames.csv", index=False)
    # 3. same stars, old vs new
    with Pool(12) as pool:
        Pp = [x for x in pool.map(pair_frame, [(int(r.epoch), r.file) for _, r in d.iterrows()]) if x is not None]
    PR = pd.concat(Pp, ignore_index=True).merge(d[["file", "epoch", "filter", "unit", "std_x", "std_y"]], on="file")
    PR["is_wd"] = np.hypot(PR.x - PR.std_x, PR.y - PR.std_y) < 3
    PR.to_csv(OUT / "pairs.csv", index=False)
    rows = []
    for (ep, flt), g in PR.groupby(["epoch", "filter"]):
        w = g[g.is_wd]; fs = g[~g.is_wd]
        near = fs[np.hypot(fs.x - fs.std_x, fs.y - fs.std_y) < 800]
        rows.append(dict(epoch=ep, filter=flt, dB_field_med=fs.dB.median(), dB_near_med=near.dB.median(), dB_wd_med=w.dB.median(),
                         dB_wd_err=P.boot_median(w.dB), dm_wd=w.dm.median(), dm_wd_err=P.boot_median(w.dm),
                         dm_field_same_peak=fs[(fs.peak_e > w.peak_e.min()) & (fs.peak_e < w.peak_e.max())].dm.median()))
    PS = pd.DataFrame(rows)
    S = S.merge(PS, on=["epoch", "filter"], how="left")
    S.to_csv(OUT / "summary.csv", index=False)
    print("=== per method (mmag; L_xx = field-star loss in the predicted-peak bin starting at xx e-)")
    print(S[["epoch", "filter", "unit", "method", "n", "peak_pred", "d_raw", "err", "L_field_at_wd", "excess", "err_excess",
             "L_8", "L_16", "L_32", "L_64", "L_128", "L_256"]].round(1).to_string(index=False))
    print("=== same stars old vs new: dB = B_new - B_old [ADU/px], dm = m_old - m_new [mmag]")
    print(PS.round(3).to_string(index=False))

    # figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    cols = {"old": "#2166ac", "new": "#b2182b"}
    for k, (ep, flt) in enumerate(((2024, "m825"), (2024, "m800"), (2026, "m825"))):
        ax = axes[k]
        for m in ("old", "new"):
            x = X[(X.method == m) & (X.epoch == ep) & (X["filter"] == flt)]
            B = P.reliable(P.binned(x, "peak_pred"))
            ax.errorbar(B.x_med, B.loss, B.err, fmt="o-", color=cols[m], ms=4, label=f"field stars, {m} background")
            g = F[(F.method == m) & (F.epoch == ep) & (F["filter"] == flt) & F.ok]
            ax.plot(g.peak_pred, g.d_raw, "*", color=cols[m], ms=7, mec="k", mew=0.3, alpha=0.6, label=f"WD2149 frames, {m}")
            ax.errorbar(g.peak_pred.median(), g.d_raw.median(), P.boot_median(g.d_raw), fmt="*", color=cols[m], ms=16, mec="k", capsize=3)
        ax.axhline(0, color="0.5", lw=0.8); ax.set_xscale("log"); ax.grid(alpha=0.25); ax.set_ylim(-150, 200)
        ax.set_xlabel("noise-free peak [e⁻]"); ax.set_ylabel("Δ (WD) / loss (field) [mmag], + = too faint")
        ax.set_title(f"{ep} {flt}: old vs new background", fontsize=10)
        if k == 0:
            ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "B1_bkg.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
