"""
Why a transmission-curve error shows up as a field colour term — figures for a stand-alone explainer report.

The field ZP of a frame is ZP_i = m_syn,i(T_model) - m_inst,i with m_inst,i = m_i(T_true) - Z, so
    ZP_i - Z = m_syn,i(T_model) - m_i(T_true) = dm_i .
If the model filter curve is the true one shifted by d (nm), dm_i ~ d * s_i with s_i = dm_AB/dlambda inside the band (the
star's in-band spectral slope in AB mag per nm).  s_i runs with the star's colour, so dm_i lines up along BP-RP; the slope of
that line is the field colour term a = d * ds/d(BP-RP).  A correct curve gives dm_i = 0 for every star and a = 0.

Two parts (v2, 2026-10-05, after verification: the first version used G < 16 stars of three fields, which are not the
pipeline ZP stars; v1 kept in ~/ezphot/log/20261005_colorterm_mechanism/v1_fieldG16/):
  A  pipeline-exact: every kept CALSPEC frame of section 8.1 is re-processed with SO.process_cal (same ZP stars, same
     weighted colour-term fit) with the FILTER curve shifted by d = -3..+3 nm (QE, optics and atmosphere kept fixed) and
     with the real curve versions.  Per frame K = da_x/dd; per filter the median over frames.  Closure: per frame
     a_x(version) - a_x(manuf nobeam) vs K x centroid difference.
  B  illustration: the unique ZP stars of those frames (SNR >= 100, FLAGS 0, isolated, corrected XP, the standard
     excluded) per filter, for the scatter plots, the in-band slopes and the response of the CALSPEC standards.
Outputs (~/ezphot/log/20261005_colorterm_mechanism/): CT0_slope.png, CT1_mechanism.png, CT2_filters.png, CT3_closure.png,
CT4_nonlinear.png, pipe_frames.csv (part A), zpstars.npz (part B), sensitivity.csv, closure.csv, ct0_dots.csv, ct4_devs.csv;
part C (2026-10-05): the same filter-shift sensitivity on the SPSS 2026 kept frames (section 8.2) for every medium-band filter
-> spss_pipe_frames.csv, sensitivity_spss.csv, CT5_sensitivity_all.png
  python 261005_colorterm_mechanism.py [--rerun] [--n-proc 24]
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
from matplotlib.ticker import FixedLocator, ScalarFormatter, NullFormatter   # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
CM = importlib.import_module("261001_calspec_manuf_beam")
SO, R, SF = CM.SO, CM.R, CM.SF
OUT = Path("/home/hhchoi1022/ezphot/log/20261005_colorterm_mechanism")
CAL = Path("/home/hhchoi1022/ezphot/log/20261001_calspec_manuf")
FILTERS = [f"m{k}" for k in range(400, 876, 25)]
SHIFTS = [-3, -2, -1, 0, 1, 2, 3]
C_DA, C_HOT, C_COOL = "#1E3A8A", "#7FB3E0", "#C0502E"
PAIRS = {"beam": ("manufacturer", "beam"), "mhj_nobeam": ("version_mhj", "nobeam"), "mhj_beam": ("version_mhj", "beam"),
         "beam_neff1.5": ("manufacturer", "beam_neff1.5")}
PAIR_STYLE = {"beam": ("o", "#C25B4E", "manuf·beam − manuf·nobeam"), "mhj_nobeam": ("d", "#C48FDB", "mhj·nobeam − manuf·nobeam"),
              "mhj_beam": ("D", "#7B3FA0", "mhj·beam − manuf·nobeam"), "beam_neff1.5": ("s", "#7F9CC9", "manuf·beam n_eff 1.5 − manuf·nobeam")}


# ---------------------------------------------------------------- curves
def pipe_totals():
    """the section 8.1 curves plus the manufacturer FILTER curve shifted by d (QE, optics, atmosphere fixed), no beam."""
    TC, DS = SO.TC, SO.DS
    atm = pd.read_csv(TC.TS / "atmosphere_elsauce.csv"); qe = pd.read_csv(TC.SRC / "QE.csv")
    opt = pd.read_csv(TC.SRC / "optics.efficiency.dr350.dr500.csv", sep=r"\s+", comment="#", skiprows=11)
    out = dict(CM.manuf_totals())
    for d in SHIFTS:
        if d:
            out[("shift", f"{d:+d}")] = {}
    for p in sorted((TC.VBASE / "manufacturer").glob("m*.csv")):
        if p.stem not in FILTERS:
            continue
        dd = pd.read_csv(p); lam = dd.lam_nm.to_numpy(float); tr = dd.trans.to_numpy(float)
        env = (np.interp(lam, qe.iloc[:, 0], qe.iloc[:, 1]) * np.interp(lam, opt.iloc[:, 0], opt.iloc[:, -1])
               * np.interp(lam, atm["lam_air"], atm["trans"]))
        for d in SHIFTS:
            if d:
                out[("shift", f"{d:+d}")][p.stem] = (DS.air_to_vac_nm(lam), np.interp(lam - d, lam, tr, left=0, right=0) * env)
    return out


def curve(tot, d, flt):
    return tot[("manufacturer", "nobeam")][flt] if d == 0 else tot[("shift", f"{d:+d}")][flt]


def centroid(c):
    lam, T = c
    return np.trapz(lam * T, lam) / np.trapz(T, lam)


# ---------------------------------------------------------------- part A: pipeline-exact colour terms
def _init_a():
    warnings.filterwarnings("ignore")
    SO.AN._init(); SO.TW._init(40)
    tot = pipe_totals()
    keep = [("manufacturer", "nobeam")] + [("shift", f"{d:+d}") for d in SHIFTS if d] + list(PAIRS.values())
    SO._TOTALS = {k: {f: v for f, v in tot[k].items() if f in FILTERS} for k in keep}


def kept_frames():
    K = pd.read_csv(CAL / "calspec_frames_flags.csv", usecols=["file", "object", "filter", "unit", "curve", "keep"])
    K = K[(K.curve == "nobeam") & K.keep & K["filter"].isin(FILTERS)].drop_duplicates("file")
    M = pd.read_csv(SO.OUT / "rephot_meta.csv")[["file"]].assign(base=lambda t: t.file.map(lambda s: Path(s).name))
    return K.merge(M.rename(columns={"file": "path"}), left_on="file", right_on="base", how="inner")


def run_a(n_proc):
    K = kept_frames()
    jobs = [(obj, list(zip(g.path, g["filter"], g.unit))) for (obj, _), g in K.groupby(["object", "filter"])]
    print(time.strftime("[%H:%M:%S]"), f"A: kept CALSPEC frames {len(K)}, jobs {len(jobs)}", flush=True)
    rows = []
    with Pool(n_proc, initializer=_init_a) as pool:
        for r in pool.imap_unordered(SO.process_cal, jobs):
            rows += r
    P = pd.DataFrame(rows)[["file", "object", "filter", "unit", "version", "variant", "a_x", "zsd", "n_zp", "bprp0"]]
    P.to_csv(OUT / "pipe_frames.csv", index=False)
    print(time.strftime("[%H:%M:%S]"), f"A: {len(P)} rows", flush=True)
    return P


# ---------------------------------------------------------------- part B: the ZP stars themselves
def _init_b():
    warnings.filterwarnings("ignore")
    SO.AN._init(); SO.TW._init(40)


def zp_stars(args):
    """unique ZP stars (pipeline selection) of a star's kept frames, per filter, with their corrected XP spectra."""
    from astropy.coordinates import SkyCoord, match_coordinates_sky
    obj, frames = args
    st0 = SO.TW._INFO[obj]
    got = SO.K.get_reference("GAIAXP_CORR_LAMOST", obj, st0["ra"], st0["dec"], 1.34, 0.90)
    if got is None:
        return []
    tbl, coords, _ = got
    ids_all = np.asarray(tbl["id"]).astype(str); bp_all = np.asarray(tbl["bp-rp"], float); g_all = np.asarray(tbl["g_mean"], float)
    per = {}
    for path, filt in frames:
        cp = SO.CATD / (Path(path).name[:-5] + "_cat.fits")
        try:
            df, meta = SO.load_cat(cp)
            sel = SO.select_zp_stars(df)
            if len(sel) < SO.MIN_REF:
                continue
            c = SkyCoord(sel.ALPHA_J2000.to_numpy(), sel.DELTA_J2000.to_numpy(), unit="deg")
            idx, d2d, _ = match_coordinates_sky(c, coords)
            dt = (meta["mjd"] - 51544.5) / 365.25 if np.isfinite(meta["mjd"]) else 24.5
            pos = SkyCoord(st0["ra"] + st0["pmra"] * dt / 3.6e6 / np.cos(np.deg2rad(st0["dec"])),
                           st0["dec"] + st0["pmdec"] * dt / 3.6e6, unit="deg")
            ok = np.asarray(d2d.arcsec < 2.0) & (c.separation(pos).arcsec > 5.0)
            per.setdefault(filt, set()).update(idx[ok].tolist())
        except Exception:                                      # noqa: BLE001
            continue
    need = set(ids_all[j] for s in per.values() for j in s)
    if not need:
        return []
    xp = SO.TW.tile_xp_radec(st0["ra"], st0["dec"], need)
    out = []
    for filt, js in per.items():
        for j in js:
            if ids_all[j] in xp and np.isfinite(bp_all[j]) and np.all(np.isfinite(xp[ids_all[j]])):
                out.append((filt, obj, ids_all[j], bp_all[j], g_all[j], xp[ids_all[j]]))
    return out


def run_b(n_proc):
    K = kept_frames()
    jobs = [(obj, list(zip(g.path, g["filter"]))) for obj, g in K.groupby("object")]
    rows = []
    with Pool(min(n_proc, len(jobs)), initializer=_init_b) as pool:
        for r in pool.imap_unordered(zp_stars, jobs):
            rows += r
    flt, obj, ids, bp, g, fx = zip(*rows)
    np.savez_compressed(OUT / "zpstars.npz", filter=np.array(flt), object=np.array(obj), id=np.array(ids), bprp=np.array(bp),
                        g=np.array(g), fx=np.vstack(fx).astype(np.float32))
    print(time.strftime("[%H:%M:%S]"), f"B: {len(rows)} (filter, star) pairs", flush=True)


def load_b():
    z = np.load(OUT / "zpstars.npz")
    S = pd.DataFrame({k: z[k] for k in ("filter", "object", "id", "bprp", "g")})
    return S, z["fx"].astype(float)


# ---------------------------------------------------------------- helpers
def lsfit(x, y, nclip=3):
    """unweighted line through y vs x, 3-sigma clipped; returns slope, intercept, rms of the kept points, kept mask."""
    m = np.isfinite(x) & np.isfinite(y)
    for _ in range(nclip):
        b, a = np.polyfit(x[m], y[m], 1)
        r = y - (a + b * x); s = 1.4826 * np.median(np.abs(r[m]))
        m = m & (np.abs(r) < 3 * max(s, 1e-6))
    return b, a, float(np.std(r[m])), m


def mabs(x):
    return f"{x:+.0f}".replace("-", "−")


# ---------------------------------------------------------------- part C: SPSS 2026 frames, every filter
SPSS_DIR = Path("/home/hhchoi1022/ezphot/log/20261001_spss_manuf")


def shift_totals_all(filters):
    """manufacturer nobeam total curve and the filter-only shifted versions for every filter in `filters`."""
    TC, DS = SO.TC, SO.DS
    atm = pd.read_csv(TC.TS / "atmosphere_elsauce.csv"); qe = pd.read_csv(TC.SRC / "QE.csv")
    opt = pd.read_csv(TC.SRC / "optics.efficiency.dr350.dr500.csv", sep=r"\s+", comment="#", skiprows=11)
    out = {("manufacturer", "nobeam"): {}}
    out.update({("shift", f"{d:+d}"): {} for d in SHIFTS if d})
    for f in filters:
        p = TC.VBASE / "manufacturer" / f"{f}.csv"
        if not p.exists():
            continue
        dd = pd.read_csv(p).sort_values("lam_nm"); lam = dd.lam_nm.to_numpy(float); tr = dd.trans.to_numpy(float)
        env = (np.interp(lam, qe.iloc[:, 0], qe.iloc[:, 1]) * np.interp(lam, opt.iloc[:, 0], opt.iloc[:, -1])
               * np.interp(lam, atm["lam_air"], atm["trans"]))
        for d in SHIFTS:
            key = ("manufacturer", "nobeam") if d == 0 else ("shift", f"{d:+d}")
            out[key][f] = (DS.air_to_vac_nm(lam), np.interp(lam - d, lam, tr, left=0, right=0) * env)
    return out


def spss_kept():
    K = pd.read_csv(SPSS_DIR / "spss_frames_flags.csv", usecols=["file", "object", "filter", "unit", "curve", "keep"])
    K = K[(K.curve == "nobeam") & K.keep].drop_duplicates("file")
    M = pd.read_csv(SO.OUT / "rephot_meta.csv")[["file"]].assign(base=lambda t: t.file.map(lambda s: Path(s).name))
    return K.merge(M.rename(columns={"file": "path"}), left_on="file", right_on="base", how="inner")


def _init_c(filters):
    warnings.filterwarnings("ignore")
    SO.AN._init(); SO.TW._init(40)
    SO._TOTALS = shift_totals_all(filters)


def run_c(n_proc):
    K = spss_kept()
    sp = pd.read_csv(SO.AN.SPSS_CSV)
    srow_of = {SO.AN.norm(s.Name): s for _, s in sp.iterrows()}
    jobs = [(obj, list(zip(g.path, g["filter"], g.unit)), srow_of[SO.AN.norm(obj)])
            for (obj, _), g in K.groupby(["object", "filter"]) if SO.AN.norm(obj) in srow_of]
    filters = sorted(K["filter"].unique())
    print(time.strftime("[%H:%M:%S]"), f"C: SPSS kept frames {len(K)}, filters {len(filters)}, jobs {len(jobs)}", flush=True)
    rows = []
    with Pool(n_proc, initializer=_init_c, initargs=(filters,)) as pool:
        for r in pool.imap_unordered(SO.process_spss, jobs):
            rows += r
    P = pd.DataFrame(rows)[["file", "object", "filter", "unit", "version", "variant", "a_x", "zsd", "n_zp", "bprp0"]]
    P.to_csv(OUT / "spss_pipe_frames.csv", index=False)
    print(time.strftime("[%H:%M:%S]"), f"C: {len(P)} rows", flush=True)


def sensitivity_from(P):
    """per-frame K = d a_x / d shift (filter-only shift), per filter the median and 25-75%."""
    P = P.copy(); P["key"] = P.version + "/" + P.variant.astype(str)
    A = P.pivot_table(index=["file", "filter"], columns="key", values="a_x").reset_index()
    a0 = A["manufacturer/nobeam"] * 1e3
    D = np.vstack([(A[f"shift/{d:+d}"] * 1e3 - a0).to_numpy() if d else np.zeros(len(A)) for d in SHIFTS]).T
    ok = np.all(np.isfinite(D), axis=1)
    A = A[ok].reset_index(drop=True); D = D[ok]
    A["K"] = np.polyfit(SHIFTS, D.T, 1)[0]
    rows = []
    for flt, g in A.groupby("filter"):
        Dg = D[(A["filter"] == flt).to_numpy()]
        amed = np.median(Dg, axis=0)
        rows.append(dict(filter=flt, frames=len(g), K=g.K.median(), K_p25=g.K.quantile(.25), K_p75=g.K.quantile(.75),
                         a_nominal=float(a0[ok][(A["filter"] == flt).to_numpy()].median()),
                         linear_dev=float(np.max(np.abs(amed - g.K.median() * np.array(SHIFTS)))),
                         **{f"a_{d:+d}": amed[i] for i, d in enumerate(SHIFTS)}))
    return pd.DataFrame(rows)


def lam_key(f):
    return int(f[1:4])


def plot_ct5(Ks, Kc, tot_all):
    """CT5: sensitivity per filter, all medium bands (SPSS 2026) and the CALSPEC 2024 set."""
    Ks = Ks.assign(lc=[centroid(tot_all[("manufacturer", "nobeam")][f]) for f in Ks["filter"]]).sort_values("lc").reset_index(drop=True)
    x = {f: i for i, f in enumerate(Ks["filter"])}
    wide = Ks["filter"].str.endswith("w").to_numpy()
    fig, ax = plt.subplots(figsize=(15, 6))
    cols = plt.get_cmap("turbo")((Ks.lc - 370) / (890 - 370))
    ax.bar(Ks.index, Ks.K, width=0.62, color=cols, edgecolor="k", lw=0.4, alpha=0.85, label="SPSS 2026 frames (median)")
    ax.errorbar(Ks.index, Ks.K, yerr=[Ks.K - Ks.K_p25, Ks.K_p75 - Ks.K], fmt="none", ecolor="k", lw=0.9, capsize=2.5,
                label="SPSS 2026: 25–75% range over frames")
    q = Kc[Kc["filter"].isin(x)]
    ax.errorbar([x[f] + 0.22 for f in q["filter"]], q.K, yerr=[q.K - q.K_p25, q.K_p75 - q.K], fmt="D", color="white", mec="k",
                ecolor="0.35", ms=6, mew=1.1, lw=0.8, capsize=2, label="CALSPEC 2024 frames (median, 25–75%)")
    for i, r in Ks.iterrows():
        ax.text(i, 5.6, f"{r.K:.1f}".replace("-", "−"), ha="center", va="center", fontsize=8, color="0.2")
    ax.axhline(0, color="k", lw=0.6); ax.grid(alpha=0.3, axis="y")
    ax.set_xticks(Ks.index); ax.set_xticklabels(Ks["filter"], rotation=60, fontsize=9.5)
    for t, w in zip(ax.get_xticklabels(), wide):
        if w:
            t.set_fontstyle("italic"); t.set_color("#7B3FA0")
    ax.set_xlim(-0.7, len(Ks) - 0.3); ax.set_ylim(min(Ks.K_p25.min(), Kc.K_p25.min()) - 4, 7.5)
    ax.axhspan(4.4, 7.5, color="0.95", zorder=0)
    ax.set_ylabel("sensitivity da/dδ [mmag/mag per nm of filter shift]")
    ax.set_xlabel("filter (ordered by centre wavelength; wide filters in purple italics)")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("CT5  Field colour-term change per nm of filter-curve shift (model − true), every filter. "
                 "Pipeline ZP stars and colour-term fit, per frame; numbers on top = SPSS 2026 median", fontsize=10.5, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "CT5_sensitivity_all.png", dpi=115); plt.close(fig)


# ---------------------------------------------------------------- CT0
def plot_ct0(tot, spec):
    """CT0: the same spectrum seen through two curves; the magnitude difference follows the local spectral slope."""
    from astropy.io import fits
    D = "/qso/data3/doabdi/intern/synthetic/calspec_data/"
    stars = [("WD2149+021 (DA2.8 white dwarf)", spec["WD2149+021"], "#2B6CB0"),
             ("P330E (G2V)", None, "#2F855A"), ("KF06T2 (K1.5 III)", None, "#C0502E"),
             ("BD−11 3759 (M3.5 V)", spec["BD-11_3759"], "#7B3FA0")]
    files = {"P330E (G2V)": "p330e_stis_001", "KF06T2 (K1.5 III)": "kf06t2_stiswfcnic_006"}
    S = []
    for name, sp, col in stars:
        if sp is None:
            t = fits.getdata(D + files[name] + ".fits", 1)
            sp = (np.asarray(t["WAVELENGTH"], float) / 10.0, np.asarray(t["FLUX"], float))
        S.append((name, sp, col))
    d = 3
    g = np.arange(355.0, 915.0, 0.25)

    def mab(w, f, lo, hi):          # AB magnitude through a top-hat band [lo, hi) on the grid g
        m = (g >= lo) & (g < hi)
        fl = np.interp(g[m], w, f)
        return -2.5 * np.log10(np.sum(fl * g[m]) / np.sum(1.0 / g[m]))

    fig = plt.figure(figsize=(15, 13))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 0.95, 1.1], hspace=0.42, wspace=0.18)
    ax = fig.add_subplot(gs[0, :])
    for name, (w, f), col in S:
        mm = (w > 365) & (w < 905)
        mag = -2.5 * np.log10(np.clip(f[mm], 1e-40, None) * w[mm] ** 2)
        mag -= np.interp(550.0, w[mm], mag)
        ax.plot(w[mm], mag, "-", color=col, lw=1.0, label=name)
    ax.set_xlim(365, 905); ax.set_ylim(4.0, -3.3); ax.grid(alpha=0.3)
    ax.set_ylabel("AB magnitude − m(550 nm)\n(up = brighter)"); ax.legend(fontsize=9, loc="upper left", ncol=2)
    ax2 = ax.twinx()
    for flt in FILTERS:
        lam, T = tot[("manufacturer", "nobeam")][flt]
        ax2.fill_between(lam, T / T.max(), color="0.5", alpha=0.10, lw=0)
    ax2.set_ylim(0, 4.5); ax2.set_yticks([])
    for flt in ("m400", "m600"):
        lam, T = tot[("manufacturer", "nobeam")][flt]
        ax2.fill_between(lam, T / T.max(), color="#E3A008", alpha=0.35, lw=0)
        ax.text(centroid((lam, T)), 3.85, f"{flt} (zoom below)", ha="center", va="bottom", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    ax.set_title("(a) four CALSPEC spectra, normalised at 550 nm; grey = the 7DT medium-band curves (m400 and m600 highlighted)",
                 fontsize=10, loc="left")

    name, (w, f), col = S[2]
    for k, (flt, title) in enumerate([("m400", "(b) steep part (4000 Å break): m400"), ("m600", "(c) flat part: m600")]):
        ax = fig.add_subplot(gs[1, k])
        lam, T = tot[("manufacturer", "nobeam")][flt]
        lc = centroid((lam, T))
        xs = np.linspace(lc - 30, lc + 30, 600)
        mag = -2.5 * np.log10(np.interp(xs, w, f) * xs ** 2); mag -= np.interp(lc, xs, mag)
        ax.plot(xs, mag * 1e3, "-", color=col, lw=1.3, label=name)
        ax.invert_yaxis(); ax.set_xlim(xs[0], xs[-1]); ax.grid(alpha=0.3)
        ax.set_ylabel("AB mag − m(band centre) [mmag]"); ax.set_xlabel("wavelength [nm]")
        m0 = float(R.syn_mag(w, f, lam, T)); m1 = float(R.syn_mag(w, f, lam + d, T))
        a2 = ax.twinx()
        a2.fill_between(lam, T / T.max(), color="0.45", alpha=0.20, lw=0, label="curve A")
        a2.plot(lam + d, T / T.max(), "--", color="k", lw=1.3, label=f"curve B = A shifted +{d} nm")
        a2.set_ylim(0, 2.9); a2.set_yticks([])
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
        a2.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left", framealpha=0.95)
        ax.set_title(f"{title}:  m(B) − m(A) = {mabs((m1 - m0) * 1e3)} mmag", fontsize=10.5, loc="left")

    ax = fig.add_subplot(gs[2, :])
    cs = np.arange(380.0, 890.0, 0.5)
    dots = []
    for name, (w, f), col in S:
        line = [(mab(w, f, c - 12.5 + d, c + 12.5 + d) - mab(w, f, c - 12.5, c + 12.5)) * 1e3 for c in cs]
        ax.plot(cs, line, "-", color=col, lw=1.0, alpha=0.75)
        xs_, ys_ = [], []
        for flt in FILTERS:
            lam, T = tot[("manufacturer", "nobeam")][flt]
            xs_.append(centroid((lam, T)))
            ys_.append((float(R.syn_mag(w, f, lam + d, T)) - float(R.syn_mag(w, f, lam, T))) * 1e3)
            dots.append(dict(star=name, filter=flt, dm_mmag=ys_[-1]))
        ax.plot(xs_, ys_, "o", color=col, ms=6, mec="k", mew=0.4, label=name)
    ax.axhline(0, color="k", lw=0.6); ax.grid(alpha=0.3); ax.set_xlim(365, 905); ax.set_ylim(-140, 80)
    ax.set_xlabel("filter centre [nm]"); ax.set_ylabel(f"m(B) − m(A) for a +{d} nm shift [mmag]")
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4, frameon=False)
    ax.set_title(f"(d) dots: the 7DT filters (curve shifted +{d} nm); lines: a 25-nm top-hat shifted +{d} nm at every wavelength. "
                 "Large where the spectrum is steep\n     (4000 Å break, Balmer lines, TiO bands), small where it is flat; "
                 "the sign follows the slope (fainter toward the red → positive).", fontsize=9.5, loc="left")
    fig.suptitle("CT0  Two curves that sample different parts of a spectrum measure different magnitudes, most of all where "
                 "the spectrum is steep", fontsize=12, x=0.01, ha="left")
    fig.savefig(OUT / "CT0_slope.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    pd.DataFrame(dots).round(2).to_csv(OUT / "ct0_dots.csv", index=False)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n-proc", type=int, default=24); ap.add_argument("--rerun", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.rerun or not (OUT / "pipe_frames.csv").exists():
        run_a(a.n_proc)
    if a.rerun or not (OUT / "zpstars.npz").exists():
        run_b(a.n_proc)
    if a.rerun or not (OUT / "spss_pipe_frames.csv").exists():
        run_c(a.n_proc)
    P = pd.read_csv(OUT / "pipe_frames.csv")
    S, FX = load_b()
    tot = pipe_totals()
    W = SO.TC.XP_WAVE
    cb = pd.read_csv("/home/hhchoi1022/ezphot/log/20260930_std_ct_applied/cal_bprp_gaia.csv").set_index("object").bprp_gaia
    L = pd.read_csv(CAL / "calspec_sed_long.csv")
    stars = L.drop_duplicates("object")[["object", "sptype"]].set_index("object").sptype.to_dict()
    spec = {o: v for (_, o), v in SF.spectra([], list(stars)).items()}
    plot_ct0(tot, spec)
    is_da = {o: str(t).startswith("DA") for o, t in stars.items()}

    # ---- part A: per-frame sensitivity and closure; part B summaries per filter --------------------------------------------
    P["key"] = P.version + "/" + P.variant.astype(str)
    A = P.pivot_table(index=["file", "filter", "object"], columns="key", values="a_x").reset_index()
    a0 = A["manufacturer/nobeam"] * 1e3
    D = np.vstack([(A[f"shift/{d:+d}"] * 1e3 - a0).to_numpy() if d else np.zeros(len(A)) for d in SHIFTS]).T
    ok = np.all(np.isfinite(D), axis=1)
    A = A[ok].reset_index(drop=True); D = D[ok]
    A["K"] = np.polyfit(SHIFTS, D.T, 1)[0]
    zsd = P[(P.version == "manufacturer") & (P.variant == "nobeam")].groupby("filter").zsd.median() * 1e3
    zsd_d = {d: (P[P.key == (f"shift/{d:+d}" if d else "manufacturer/nobeam")].groupby("filter").zsd.median() * 1e3) for d in SHIFTS}
    srow, crow, B2 = [], [], {}
    for flt in FILTERS:
        sel = (A["filter"] == flt).to_numpy()
        g = A[sel]; Dg = D[sel]
        K = g.K.median(); amed = np.median(Dg, axis=0)
        sm = (S["filter"] == flt).to_numpy(); x = S.bprp.to_numpy()[sm]; fx = FX[sm]
        m0 = R.syn_mag(W, fx, *curve(tot, 0, flt), 0.0)
        dm2 = (R.syn_mag(W, fx, *curve(tot, 2, flt), 0.0) - m0) * 1e3
        b2, i2, rms2, keep2 = lsfit(x, dm2)
        frac = np.var(b2 * x[keep2]) / np.var(dm2[keep2])
        dmc2 = {o: (float(R.syn_mag(*spec[o], *curve(tot, 2, flt))) - float(R.syn_mag(*spec[o], *curve(tot, 0, flt)))) * 1e3
                for o in spec}
        xm, ym = np.median(x), np.median(dm2)
        r_da = [(v - ym) / (b2 * (cb[o] - xm)) for o, v in dmc2.items() if is_da[o]]
        B2[flt] = (x, dm2, b2, i2, dmc2, fx)
        gcen = (centroid(curve(tot, 1, flt)) - centroid(curve(tot, -1, flt))) / 2      # total-curve centroid per nm of filter shift
        srow.append(dict(filter=flt, lam_c=centroid(curve(tot, 0, flt)), frames=len(g), objects=g.object.nunique(), cen_per_nm=gcen,
                         **{f"zsd_{d:+d}": zsd_d[d].get(flt, np.nan) for d in SHIFTS},
                         K=K, K_p25=g.K.quantile(.25), K_p75=g.K.quantile(.75), a_plus2=amed[SHIFTS.index(2)],
                         linear_dev=float(np.max(np.abs(amed - K * np.array(SHIFTS)))),
                         zp_stars=int(sm.sum()), bprp_med=float(xm), bprp_sd=float(np.std(x[keep2])),
                         G_med=float(np.median(S.g.to_numpy()[sm])), pool_a_plus2=b2,
                         slope_rms_plus2=abs(b2) * float(np.std(x[keep2])), resid_rms_plus2=rms2, slope_var_frac=frac,
                         zsd_nobeam=zsd.get(flt, np.nan),
                         r_DA_median=float(np.median(r_da)), r_DA_min=float(np.min(r_da)), r_DA_max=float(np.max(r_da))))
        for lab, key in PAIRS.items():
            col = "/".join(key)
            if col not in g:
                continue
            dlc = centroid(tot[key][flt]) - centroid(curve(tot, 0, flt))
            obs = g[col] * 1e3 - g["manufacturer/nobeam"] * 1e3
            crow.append(dict(filter=flt, pair=lab, dlam_c=dlc, observed=float(obs.median()), predicted=float((g.K * dlc / gcen).median()),
                             trivial=bool(abs(dlc) < 1e-6)))
    Kt = pd.DataFrame(srow); Kt.round(4).to_csv(OUT / "sensitivity.csv", index=False)
    Ps = pd.read_csv(OUT / "spss_pipe_frames.csv")
    Ks = sensitivity_from(Ps); Ks = Ks.iloc[np.argsort([lam_key(f) for f in Ks["filter"]])]
    Ks.round(3).to_csv(OUT / "sensitivity_spss.csv", index=False)
    print("SPSS 2026 sensitivity:\n", Ks.round(2).to_string(index=False))
    plot_ct5(Ks, Kt, shift_totals_all(list(Ks["filter"])))
    C = pd.DataFrame(crow); C.round(3).to_csv(OUT / "closure.csv", index=False)
    pd.set_option("display.width", 250)
    print(Kt.round(2).to_string(index=False))
    print(C.round(1).to_string(index=False))
    for lab, gq in C.groupby("pair"):
        nt = gq[~gq.trivial]
        print(f"{lab}: rms(observed - predicted) all {np.sqrt(np.mean((gq.observed - gq.predicted) ** 2)):.2f}, "
              f"non-trivial {np.sqrt(np.mean((nt.observed - nt.predicted) ** 2)):.2f} ({len(nt)} filters)")
    Kf = Kt.set_index("filter")
    xx = np.linspace(-0.6, 3.0, 10)

    def stds(ax, vals):
        for o, v in vals.items():
            c = C_DA if is_da[o] else (C_HOT if cb[o] < 0.3 else C_COOL)
            ax.plot(cb[o], v, "*", ms=12, color=c, mec="k", mew=0.5, zorder=4)

    # ---- CT1: the mechanism in m550 ----------------------------------------------------------------------------------------
    flt = "m550"
    lam, T = curve(tot, 0, flt); lc = Kf.lam_c[flt]
    x, dm2, b2, i2, dmc2, fx = B2[flt]
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.3)
    ax = fig.add_subplot(gs[0, :2])
    kst = int(np.argmin(np.abs(x - 1.6)))
    xs = np.linspace(lc - 32, lc + 32, 300)
    for name, (w, f), col in [("WD2149+021 (DA white dwarf)", spec["WD2149+021"], C_DA),
                              ("BD+02 3375 (F subdwarf)", spec["BD+02_3375"], C_COOL),
                              (f"field K star (ZP star, BP−RP {x[kst]:.2f})", (W, fx[kst]), "0.35")]:
        mab = -2.5 * np.log10(np.interp(xs, w, f) * xs ** 2); mab -= np.interp(lc, xs, mab)
        ax.plot(xs, mab * 1e3, "-", color=col, lw=2.2, label=name)
    ax.invert_yaxis(); ax.set_ylabel("AB magnitude relative to band centre [mmag]\n(up = brighter)")
    ax.set_xlabel("wavelength [nm]"); ax.grid(alpha=0.3); ax.set_xlim(xs[0], xs[-1])
    ax2 = ax.twinx()
    ax2.fill_between(lam, T / T.max(), color="0.55", alpha=0.18, lw=0, label="model curve")
    ax2.plot(lam - 5, T / T.max(), "--", color="k", lw=1.2, label="true curve = model − 5 nm (exaggerated)")
    ax2.set_ylim(0, 3.2); ax2.set_yticks([]); ax2.legend(loc="lower right", fontsize=8.5)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_title(f"(a) {flt}: inside the band a hot star gets fainter toward the red, a cool star brighter", fontsize=10, loc="left")

    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(x, dm2, s=3, c="0.55", alpha=0.35, lw=0, label=f"ZP stars ({len(x)}, corrected XP)")
    ax.plot(xx, i2 + b2 * xx, "-", color="k", lw=1.5, label=f"fit: colour term {b2:+.1f} mmag/mag")
    stds(ax, dmc2)
    for c, lab in [(C_DA, "CALSPEC DA"), (C_HOT, "CALSPEC DB, sdO, sdB"), (C_COOL, "CALSPEC F sdF, M3.5V")]:
        ax.plot([], [], "*", ms=10, color=c, mec="k", mew=0.5, label=lab)
    ax.axhline(0, color="k", lw=0.5); ax.grid(alpha=0.3); ax.set_xlim(-0.7, 3.1)
    ax.set_xlabel("BP − RP"); ax.set_ylabel("ZP_i − Z = m_syn(model) − m(true) [mmag]")
    ax.set_title(f"(b) {flt}, model curve +2 nm off: the ZP stars\n    line up along colour", fontsize=10, loc="left")
    ax.legend(fontsize=8.5, loc="lower left")

    ax = fig.add_subplot(gs[1, 0])
    for d in SHIFTS:
        dmd = (R.syn_mag(W, fx, *curve(tot, d, flt), 0.0) - R.syn_mag(W, fx, *curve(tot, 0, flt), 0.0)) * 1e3
        bb, aa, _, _ = lsfit(x, dmd)
        ax.plot(xx, aa + bb * xx, "-", lw=1.4, color=plt.get_cmap("coolwarm")((d + 3) / 6), label=f"{d:+d} nm")
    ax.axhline(0, color="k", lw=0.5); ax.grid(alpha=0.3); ax.set_xlim(-0.7, 3.1)
    ax.set_xlabel("BP − RP"); ax.set_ylabel("ZP_i − Z [mmag]"); ax.legend(fontsize=8.5, ncol=2, title="model − true", title_fontsize=8.5)
    ax.set_title("(c) the line tilts in proportion to the shift", fontsize=10, loc="left")

    ax = fig.add_subplot(gs[1, 1])
    for fl, col in [("m400", "#5B3A8C"), ("m475", "#2B6CB0"), ("m550", "#2F855A"), ("m650", "#B7791F"), ("m825", "#C0502E")]:
        Dg = D[(A["filter"] == fl).to_numpy()]
        ax.plot(SHIFTS, np.median(Dg, axis=0), "o-", color=col, ms=4, label=f"{fl}: {Kf.K[fl]:+.1f} per nm")
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5); ax.grid(alpha=0.3)
    ax.set_xlabel("model filter curve − true [nm]"); ax.set_ylabel("change of the field colour term from δ = 0 [mmag/mag]")
    ax.legend(fontsize=7.5)
    ax.set_title("(d) colour term vs shift (median over the CALSPEC\n    frames, pipeline fit); slope = sensitivity", fontsize=10, loc="left")

    ax = fig.add_subplot(gs[1, 2])
    s_in = (R.syn_mag(W, fx, *curve(tot, 1, flt), 0.0) - R.syn_mag(W, fx, *curve(tot, -1, flt), 0.0)) * 1e3 / 2
    ax.scatter(x, s_in, s=3, c="0.55", alpha=0.35, lw=0)
    stds(ax, {o: (float(R.syn_mag(*spec[o], *curve(tot, 1, flt))) - float(R.syn_mag(*spec[o], *curve(tot, -1, flt)))) * 1e3 / 2
              for o in spec})
    ax.axhline(0, color="k", lw=0.5); ax.grid(alpha=0.3); ax.set_xlim(-0.7, 3.1)
    ax.set_xlabel("BP − RP"); ax.set_ylabel("in-band slope s = dm_AB/dλ [mmag/nm]")
    ax.set_title("(e) why: each star's in-band spectral slope\n    runs with its colour", fontsize=10, loc="left")
    fig.suptitle("CT1  How a curve error becomes a colour term (m550 example)", fontsize=12, x=0.01, ha="left")
    fig.savefig(OUT / "CT1_mechanism.png", dpi=115, bbox_inches="tight"); plt.close(fig)

    # ---- CT2: all filters --------------------------------------------------------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.0), gridspec_kw=dict(width_ratios=[1.3, 1]))
    ax = axs[0]
    cols = plt.get_cmap("turbo")((Kt.lam_c - 380) / (900 - 380))
    ax.bar(Kt.lam_c, Kt.K, width=16, color=cols, edgecolor="k", lw=0.4)
    ax.errorbar(Kt.lam_c, Kt.K, yerr=[Kt.K - Kt.K_p25, Kt.K_p75 - Kt.K], fmt="none", ecolor="k", lw=0.8, capsize=2)
    ax.set_xticks(Kt.lam_c); ax.set_xticklabels(Kt["filter"], rotation=60, fontsize=9)
    ax.axhline(0, color="k", lw=0.5); ax.grid(alpha=0.3, axis="y")
    ax.set_ylabel("sensitivity da/dδ [mmag/mag per nm]")
    ax.set_title("(a) change of the field colour term per nm of filter-curve shift\n    (median over CALSPEC frames, error bars = 25–75% range)",
                 fontsize=10, loc="left")
    ax = axs[1]
    for d, c, mk in [(-2, "#2B6CB0", "v"), (0, "k", "o"), (2, "#C0502E", "^")]:
        ax.plot(Kt.lam_c, Kt[f"zsd_{d:+d}"], mk + "-", color=c, ms=5, lw=1.4, label=f"model filter curve {d:+d} nm")
    ax.grid(alpha=0.3); ax.legend(fontsize=8.5); ax.set_xlabel("filter centre [nm]")
    ax.set_ylabel("frame ZP scatter zsd [mmag] (median over frames)"); ax.set_ylim(10, 60)
    ax.set_title("(b) measured zsd when the filter curve is shifted:\n    it grows as the colour term moves away from zero", fontsize=10, loc="left")
    fig.suptitle("CT2  Sensitivity of the field colour term to a filter-curve shift, per filter (CALSPEC frames, pipeline ZP stars)",
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(OUT / "CT2_filters.png", dpi=115); plt.close(fig)

    # ---- CT3: closure ------------------------------------------------------------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    ax = axs[0]
    q = C[~C.trivial]
    for lab, gq in q.groupby("pair"):
        m_, c_, t_ = PAIR_STYLE[lab]
        ax.plot(gq.dlam_c, gq.observed, m_, color=c_, ms=7, mec="k", mew=0.3, label=t_)
    for (fl, pr, dx, dy) in [("m400", "beam_neff1.5", 8, 0), ("m425", "beam_neff1.5", 8, 0), ("m400", "mhj_nobeam", -34, -4),
                             ("m425", "mhj_nobeam", 8, 0), ("m475", "mhj_beam", 8, 0), ("m400", "beam", 8, 0)]:
        r = q[(q["filter"] == fl) & (q.pair == pr)].iloc[0]
        ax.annotate(fl, (r.dlam_c, r.observed), fontsize=9.5, xytext=(dx, dy), textcoords="offset points", va="center")
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_xlabel("centroid difference between the two curves [nm]"); ax.set_ylabel("colour-term difference [mmag/mag]")
    ax.set_title("(a) colour-term change between curve versions\n    (paired per frame, median over frames)", fontsize=10, loc="left")
    ax = axs[1]
    lim = np.nanmax(np.abs(np.r_[q.observed, q.predicted])) * 1.08
    ax.plot([-lim, lim], [-lim, lim], "-", color="0.6", lw=1)
    for lab, gq in q.groupby("pair"):
        m_, c_, t_ = PAIR_STYLE[lab]
        ax.plot(gq.predicted, gq.observed, m_, color=c_, ms=7, mec="k", mew=0.3, label=t_)
    rms = np.sqrt(np.mean((q.observed - q.predicted) ** 2))
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.grid(alpha=0.3); ax.set_aspect("equal")
    ax.set_xlabel("predicted = sensitivity × equivalent filter shift [mmag/mag]"); ax.set_ylabel("colour-term difference [mmag/mag]")
    ax.set_title(f"(b) prediction from the centroid shift alone\n    (rms {rms:.1f} mmag/mag, {len(q)} points pooled)", fontsize=10, loc="left")
    fig.suptitle("CT3  Between curve versions, the field colour term moves as the centroid shift predicts", fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(OUT / "CT3_closure.png", dpi=115); plt.close(fig)

    # ---- CT4: features at the band edge ------------------------------------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(14.5, 5.4))
    devrows = []
    titles = {"m475": "(a) m475: red edge on Hβ (486 nm)",
              "m400": "(b) m400: DA Balmer lines at the blue edge;\n     ZP-star line steepened by the 4000 Å break",
              "m725": "(c) m725: TiO band in the M dwarf\n     (redder than every ZP star)"}
    boxpos = {"m475": (0.98, 0.98, "right", "top"), "m400": (0.98, 0.98, "right", "top"), "m725": (0.03, 0.04, "left", "bottom")}
    for ax, fl in zip(axs, ["m475", "m400", "m725"]):
        x, dm2, b2, i2, dmc2, _ = B2[fl]
        ax.scatter(x, dm2, s=4, c="0.6", alpha=0.4, lw=0)
        ax.plot(xx, i2 + b2 * xx, "-", color="k", lw=1.3, label=f"ZP-star line ({b2:+.1f} mmag/mag)")
        stds(ax, dmc2)
        devs = {o: v - (i2 + b2 * cb[o]) for o, v in dmc2.items()}
        for o, dv in devs.items():
            devrows.append(dict(filter=fl, object=o, sptype=stars[o], bprp=cb[o], dm=dmc2[o], dev=dv))
        dav = [dv for o, dv in devs.items() if is_da[o]]
        lines = [f"DA ({len(dav)}): {mabs(min(dav))} … {mabs(max(dav))}"]
        lines += [f"{o.replace('_', ' ')}: {mabs(dv)}" for o, dv in devs.items() if not is_da[o] and abs(dv) > 4]
        bx, by, ha, va = boxpos[fl]
        ax.text(bx, by, "distance from the line [mmag]\n" + "\n".join(lines), transform=ax.transAxes, ha=ha, va=va, fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.92))
        ax.axhline(0, color="k", lw=0.5); ax.grid(alpha=0.3); ax.set_xlim(-0.7, 3.1)
        ax.set_xlabel("BP − RP")
        ax.set_title(titles[fl], fontsize=10, loc="left")
    for ax, fl in zip(axs, ["m475", "m400", "m725"]):
        h, l = ax.get_legend_handles_labels()
        if fl == "m475":
            for c, lab in [(C_DA, "DA"), (C_HOT, "DB, sdO, sdB"), (C_COOL, "F sdF, M3.5V")]:
                h.append(plt.Line2D([], [], marker="*", ls="", ms=10, color=c, mec="k", mew=0.5)); l.append(lab)
        ax.legend(h, l, fontsize=8.5, loc="lower left" if fl != "m725" else "upper right")
    axs[0].set_ylabel("ZP_i − Z = m_syn(model) − m(true) [mmag], model +2 nm")
    fig.suptitle("CT4  When the band edge sits on a spectral feature, stars leave the line: the field colour term does not "
                 "predict the standards", fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(OUT / "CT4_nonlinear.png", dpi=115); plt.close(fig)
    pd.DataFrame(devrows).round(2).to_csv(OUT / "ct4_devs.csv", index=False)


if __name__ == "__main__":
    main()
