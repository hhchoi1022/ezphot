"""
CALSPEC 2024 7DT10 m825: the DA white dwarfs sit at Delta +70...+84 mmag while m800 / m600 taken just before / after on
the same unit are ~0.  Two hypotheses:

A  wrong / shifted filter.  The manufacturer·beam m825 (and m800) curve shifted by s = -30...+30 nm is put through the
   section 8.1 pipeline (SO.process_cal: real field-star corrected XP spectra, CALSPEC spectrum, global ZP without the
   standard).  Per frame: field colour term a_x(s) and Delta(s).  The field stars fix s (a_x ~ 0 for the right curve);
   Delta(s at that a_x) tells what a mis-labelled / shifted filter can do to the WDs.
B  partial clouds (or a flat / ghost pattern).  Per frame, field stars (FLAGS 0, peak <= 50k ADU, isolated 10",
   SNR >= 20, corrected XP) give ZP_i - median(ZP of SNR >= 100 stars) with detector X, Y.  Clouds over the standard
   would make the stars near it (r < 1500 px) fainter than the rest, varying frame to frame; a flat / pupil-ghost problem
   would be a fixed pattern in detector coordinates for the unit-filter.
Outputs: ~/ezphot/log/20261004_m825_check/
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
import astropy.units as u         # noqa: E402
from astropy.coordinates import SkyCoord, match_coordinates_sky, search_around_sky   # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
CM = importlib.import_module("261001_calspec_manuf_beam")
SO, SS, R = CM.SO, CM.SS, CM.R
OUT = Path("/home/hhchoi1022/ezphot/log/20261004_m825_check")
CAL = Path("/home/hhchoi1022/ezphot/log/20261001_calspec_manuf")
SHIFTS = [-30, -20, -15, -10, -5, -2, 0, 2, 5, 10, 15, 20, 30]
SHIFT_FILTERS = ["m800", "m825"]
SPATIAL = {"7DT10": ["m600", "m800", "m825"], "7DT11": ["m850", "m875"], "7DT06": ["m775"], "7DT09": ["m775"]}
R_NEAR = 1500.0
_C = {}


# ---------------------------------------------------------------- A: shifted curves
def shift_totals():
    base = CM.manuf_totals()[("manufacturer", "beam")]
    out = {}
    for s in SHIFTS:
        out[("shift", f"{s:+d}")] = {f: (base[f][0] + s, base[f][1]) for f in SHIFT_FILTERS}
    return out


def _init_shift():
    warnings.filterwarnings("ignore")
    SO.AN._init(); SO.TW._init(40)
    SO._TOTALS = shift_totals()


def run_shift(n_proc):
    M = pd.read_csv(SO.OUT / "rephot_meta.csv")
    M = M[M.status.isin(["ok", "cached"]) & (M.dataset == "CALSPEC") & (M.unit == "7DT10") & M["filter"].isin(SHIFT_FILTERS)]
    M = M[[(SO.CATD / (Path(f).name[:-5] + "_cat.fits")).exists() for f in M.file]]
    jobs = [(obj, list(zip(g.file, g["filter"], g.unit))) for (obj, _), g in M.groupby(["object", "filter"])]
    print(time.strftime("[%H:%M:%S]"), f"A: 7DT10 m800/m825 frames {len(M)}, jobs {len(jobs)}", flush=True)
    rows = []
    with Pool(n_proc, initializer=_init_shift) as pool:
        for r in pool.imap_unordered(SO.process_cal, jobs):
            rows += r
    F = pd.DataFrame(rows)
    F.to_csv(OUT / "shift_frames.csv", index=False)
    return F


# ---------------------------------------------------------------- B: per-star ZP with detector position
def _init_star():
    warnings.filterwarnings("ignore")
    SO.AN._init(); SO.TW._init(40)
    _C["curves"] = CM.manuf_totals()[("manufacturer", "beam")]


def frame_stars_xy(args):
    fr, obj, filt, unit, sx, sy = args
    cp = SO.CATD / (Path(fr).name[:-5] + "_cat.fits")
    try:
        df, meta = SO.load_cat(cp)
        allc = SkyCoord(df.ALPHA_J2000.to_numpy(), df.DELTA_J2000.to_numpy(), unit="deg")
        q = (df.FLAGS.to_numpy() == 0) & (df.FLUX_MAX.to_numpy() <= 50000) & (df.SNR_APER_3.to_numpy() >= 20) \
            & (df.MAG_APER_3.to_numpy() < 90)
        ig, _, s2, _ = search_around_sky(allc[q], allc, 10 * u.arcsec)
        nbr = np.bincount(ig[s2.arcsec > 0.5], minlength=int(q.sum()))
        d = df[q][nbr == 0].reset_index(drop=True)
        st0 = SO.TW._INFO[obj]
        tbl, coords, _ = SO.K.get_reference("GAIAXP_CORR_LAMOST", obj, st0["ra"], st0["dec"], 1.34, 0.90)
        idx, d2d, _ = match_coordinates_sky(SkyCoord(d.ALPHA_J2000.to_numpy(), d.DELTA_J2000.to_numpy(), unit="deg"), coords)
        ok = np.asarray(d2d.arcsec < 2.0)
        rid = np.asarray(tbl[idx[ok]]["id"]).astype(str)
        xpf = SO.TW.tile_xp_radec(st0["ra"], st0["dec"], set(rid))
        keep = np.array([r in xpf for r in rid])
        ok[np.where(ok)[0][~keep]] = False
        dd = d[ok]
        if len(dd) < 15:
            return None
        fc = np.vstack([xpf[r] for r in rid[keep]])
        fw, ft = _C["curves"][filt]
        zp = R.syn_mag(SO.TC.XP_WAVE, fc, fw, ft, 0.0) - dd.MAG_APER_3.to_numpy()
        snr = dd.SNR_APER_3.to_numpy()
        x, y = dd.X.to_numpy(), dd.Y.to_numpy()
        sep_std = np.hypot(x - sx, y - sy)
        bright = np.isfinite(zp) & (snr >= 100) & (sep_std > 3)
        if bright.sum() < 8:
            return None
        return pd.DataFrame(dict(file=Path(fr).name, object=obj, filter=filt, unit=unit, mjd=meta["mjd"], snr=snr,
                                 peak_e=dd.FLUX_MAX.to_numpy() * meta["egain"], x=x, y=y, r_std=sep_std,
                                 r_cen=np.hypot(x - meta["nx"] / 2, y - meta["ny"] / 2), bprp=np.asarray(tbl[idx[ok]]["bp-rp"], float)[keep],
                                 res=1e3 * (zp - np.median(zp[bright]))))[lambda t: sep_std > 3]
    except Exception:                                                 # noqa: BLE001
        return None


def run_stars(n_proc):
    F = pd.read_csv(CAL / "calspec_manuf_frames.csv", usecols=["file", "object", "filter", "unit", "std_x", "std_y", "variant", "version"])
    F = F[(F.version == "manufacturer") & (F.variant == "beam")].drop_duplicates("file")
    F = F[[f in SPATIAL.get(un, []) for un, f in zip(F.unit, F["filter"])]]
    M = pd.read_csv(SO.OUT / "rephot_meta.csv")[["file"]].assign(base=lambda t: t.file.map(lambda s: Path(s).name))
    F = F.merge(M.rename(columns={"file": "path"}), left_on="file", right_on="base", how="inner")
    items = list(F[["path", "object", "filter", "unit", "std_x", "std_y"]].itertuples(index=False, name=None))
    print(time.strftime("[%H:%M:%S]"), f"B: frames {len(items)}", flush=True)
    with Pool(n_proc, initializer=_init_star) as pool:
        res = [r for r in pool.imap_unordered(frame_stars_xy, items, chunksize=2) if r is not None]
    A = pd.concat(res, ignore_index=True)
    A = A[np.isfinite(A.res) & (A.res.abs() < 500)]
    A.to_csv(OUT / "stars_xy.csv", index=False)
    return A


# ---------------------------------------------------------------- analysis
def std_frames():
    F = pd.read_csv(CAL / "calspec_manuf_frames.csv")
    F = F[(F.version == "manufacturer") & (F.variant == "beam")]
    K = pd.read_csv(CAL / "calspec_frames_flags.csv", usecols=["file", "curve", "keep", "flag_filter"])
    K = K[K.curve == "beam"][["file", "keep", "flag_filter"]].drop_duplicates("file")
    F = F.merge(K, on="file", how="inner")
    F["night"] = np.floor(F.mjd - 16 / 24).astype(int)
    F["bias"] = SS.expected_bias(F.std_peak_e.to_numpy(), "CALSPEC")
    F["d"] = (F.d_gnd - F.zp + F.zp_x) * 1e3 - F.bias
    return F


def analyse_shift(S, F):
    S = S.copy()
    S["d_nb"] = (S.d_gnd - S.zp + S.zp_x) * 1e3
    S["s"] = S.variant.astype(int)
    S = S.merge(F[["file", "keep", "night", "std_snr"]].rename(columns={"std_snr": "snr_f"}), on="file", how="left")
    S = S[S.keep.fillna(False).astype(bool) & (S.std_snr > 10)]
    G = S.groupby(["object", "filter", "night", "s"]).agg(a_x=("a_x", "median"), d=("d_nb", "median"), zsd=("zsd", "median"),
                                                          n=("file", "nunique")).reset_index()
    rows = []
    for (o, f, n), g in G.groupby(["object", "filter", "night"]):
        g = g.sort_values("s")
        a0 = float(g.loc[g.s == 0, "a_x"].iloc[0]) * 1e3; d0 = float(g.loc[g.s == 0, "d"].iloc[0])
        # shift at which the field colour term vanishes (linear interpolation in a_x over s)
        ax = g.a_x.to_numpy() * 1e3; ss = g.s.to_numpy()
        sz = np.interp(0.0, ax[::-1], ss[::-1]) if (ax.max() > 0 > ax.min()) else np.nan
        dz = np.interp(sz, ss, g.d.to_numpy()) if np.isfinite(sz) else np.nan
        k = np.polyfit(ss[np.abs(ss) <= 10], ax[np.abs(ss) <= 10], 1)[0]
        kd = np.polyfit(ss[np.abs(ss) <= 10], g.d.to_numpy()[np.abs(ss) <= 10], 1)[0]
        rows.append(dict(object=o, filter=f, night=n, n=int(g.n.iloc[0]), a_x0=a0, d0=d0, dax_ds=k, dd_ds=kd, s_ax0=sz,
                         d_at_s_ax0=dz, zsd0=float(g.loc[g.s == 0, "zsd"].iloc[0]) * 1e3,
                         zsd_min_s=int(g.loc[g.zsd.idxmin(), "s"])))
    T = pd.DataFrame(rows)
    T.round(2).to_csv(OUT / "shift_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print("A: per star/filter/night — a_x and Delta(no bias) at s=0, slopes per nm, shift where a_x=0 and Delta there:")
    print(T.round(1).to_string(index=False))
    return S, G, T


def analyse_spatial(A, F):
    A = A.copy()
    rows = []
    for fl, g in A.groupby("file"):
        for lab, lo, hi in (("snr50", 50, 1e9), ("snr20_50", 20, 50)):
            q = g[(g.snr >= lo) & (g.snr < hi)]
            nr, fa = q[q.r_std < R_NEAR], q[q.r_std >= R_NEAR]
            rows.append(dict(file=fl, sel=lab, n_near=len(nr), n_far=len(fa),
                             near_minus_far=(nr.res.median() - fa.res.median()) if len(nr) >= 3 and len(fa) >= 5 else np.nan,
                             err=1.2533 * np.sqrt((nr.res.std() ** 2 / max(len(nr), 1)) + (fa.res.std() ** 2 / max(len(fa), 1)))
                             if len(nr) >= 3 and len(fa) >= 5 else np.nan))
    P = pd.DataFrame(rows).merge(F[["file", "object", "filter", "unit", "night", "mjd", "d", "std_snr", "keep"]], on="file", how="left")
    P.round(2).to_csv(OUT / "near_far_frames.csv", index=False)
    W = P[P.sel == "snr50"]
    T = W.groupby(["unit", "filter", "object", "night"]).agg(n=("file", "size"), near_minus_far=("near_minus_far", "median"),
                                                          err=("err", lambda e: np.sqrt(np.nansum(e ** 2)) / max(e.notna().sum(), 1)),
                                                          n_near=("n_near", "median"), d=("d", "median")).reset_index()
    print("B1: stars near the standard (r < 1500 px) minus the rest, SNR >= 50, per sequence [mmag]  (negative = centre fainter):")
    print(T.round(1).to_string(index=False))
    # B2: stacked detector-position pattern per unit-filter (SNR >= 50): radial profile from the optical centre
    A["rb"] = pd.cut(A.r_cen, [0, 800, 1500, 2200, 3000, 4000, 6000])
    Z = A[A.snr >= 50].groupby(["unit", "filter", "rb"], observed=True).res.agg(["median", "size"]).reset_index()
    print("B2: field-star residual vs distance from the detector centre, SNR >= 50 [mmag]:")
    print(Z.pivot_table(index=["unit", "filter"], columns="rb", values="median", observed=True).round(1).to_string())
    return P, T, A


def plot(S, G, T, P, A):
    fig, axs = plt.subplots(2, 2, figsize=(13, 9.5))
    cols = {"WD2149+021": "#1b9e77", "WD1919+145": "#d95f02", "WD2039-682": "#7570b3", "GJ754.1A": "#e7298a",
            "LDS749B": "#66a61e", "HZ4": "#e6ab02", "WD0308-565": "#a6761d", "BD+02_3375": "#666666"}
    for f, ax in zip(SHIFT_FILTERS, axs[0]):
        for (o, n), g in G[G["filter"] == f].groupby(["object", "night"]):
            g = g.sort_values("s")
            ax.plot(g.a_x * 1e3, g.d, "o-", ms=3, lw=1, color=cols.get(o, "k"), label=f"{o} ({n})")
            z = g[g.s == 0]
            ax.plot(z.a_x * 1e3, z.d, "s", ms=8, mfc="none", mec=cols.get(o, "k"))
        ax.axvline(0, color="k", lw=0.6); ax.axhline(0, color="k", lw=0.6); ax.grid(alpha=0.3)
        ax.set_xlabel("field colour term a_x [mmag/mag]  (curve shifted by -30…+30 nm; square = nominal)")
        ax.set_ylabel("Δ, no faint bias [mmag]"); ax.set_xlim(-250, 250)
        ax.set_title(f"A  7DT10 {f}: what a shifted / wrong {f} curve does", fontsize=10, loc="left")
        ax.legend(fontsize=6.5, ncol=2)
    ax = axs[1, 0]
    W = P[(P.sel == "snr50") & (P.unit == "7DT10")]
    for f, mk in zip(["m800", "m825", "m600"], ["o", "s", "^"]):
        for o, g in W[W["filter"] == f].groupby("object"):
            ax.errorbar(g.d, g.near_minus_far, yerr=g.err, fmt=mk, ms=4, color=cols.get(o, "k"), alpha=0.7, lw=0.5,
                        label=f"{o} {f}" if f == "m825" else None)
    ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6); ax.grid(alpha=0.3); ax.set_ylim(-80, 80); ax.set_xlim(-150, 250)
    ax.set_xlabel("standard Δ (bias corrected) [mmag]"); ax.set_ylabel("field ZP near std (r<1500 px) − rest [mmag]")
    ax.set_title("B1  7DT10 per frame (o m800, □ m825, △ m600): is the area around the standard dimmer?", fontsize=10, loc="left")
    ax.legend(fontsize=6.5)
    ax = axs[1, 1]
    q = A[(A.snr >= 50)]
    for (un, f), g in q.groupby(["unit", "filter"]):
        b = g.groupby(pd.cut(g.r_cen, np.arange(0, 6001, 600)), observed=True).res.median()
        ax.plot([iv.mid for iv in b.index], b.values, "o-", ms=3, label=f"{un} {f}", lw=2 if (un, f) == ("7DT10", "m825") else 1)
    ax.axhline(0, color="k", lw=0.6); ax.grid(alpha=0.3); ax.legend(fontsize=7); ax.set_ylim(-40, 40)
    ax.axvspan(0, 600, color="0.9", zorder=0)
    ax.set_xlabel("distance from detector centre [px] (standards at r ≈ 150 px, shaded)"); ax.set_ylabel("field-star residual, SNR ≥ 50 [mmag]")
    ax.set_title("B2  stacked radial pattern per unit-filter (all CALSPEC 2024 frames)", fontsize=10, loc="left")
    fig.suptitle("7DT10 m825 white-dwarf excess: wrong filter (A) or clouds / flat pattern (B)?", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(OUT / "C1_m825_filter_cloud.png", dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n-proc", type=int, default=24); ap.add_argument("--rerun", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore")
    F = std_frames()
    S = run_shift(a.n_proc) if a.rerun or not (OUT / "shift_frames.csv").exists() else pd.read_csv(OUT / "shift_frames.csv")
    A = run_stars(a.n_proc) if a.rerun or not (OUT / "stars_xy.csv").exists() else pd.read_csv(OUT / "stars_xy.csv")
    S, G, T = analyse_shift(S, F)
    P, T2, A = analyse_spatial(A, F)
    plot(S, G, T, P, A)


if __name__ == "__main__":
    main()
