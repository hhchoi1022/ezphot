"""
Is the standard-star offset (Delta = obs - syn, section 8.1) seeing dependent?

Per frame (CALSPEC 2024 and the faint-check frames, catalogs of the new-background re-photometry, MAG_APER_3 = 10"):
  - seeing = median FWHM_IMAGE of the bright field stars (FLAGS 0, SNR >= 100, peak <= 50k), also the header SEEING;
  - aperture loss 10" -> 15" (median MAG_APER_3 - MAG_APER_4 of the same stars), centre (r < 1500 px) and edge (r > 3000 px).
Tests:
  S1  standard: Delta minus its (star, filter, unit) median vs the frame seeing minus its group median (within-group slope),
      by SNR class; and the group medians against the group seeing (between groups)
  S2  field stars (faint check): residual vs SNR in seeing terciles -> does the low-signal loss depend on seeing?
  S3  aperture loss vs seeing, centre vs edge -> a seeing-dependent aperture loss that differs between the standard (centre)
      and the field ZP stars (whole field) would make Delta seeing dependent
Outputs: ~/ezphot/log/20261004_calspec_seeing/
"""
import argparse
import importlib
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402
from astropy.io import fits       # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
CM = importlib.import_module("261001_calspec_manuf_beam")
SO, SS = CM.SO, CM.SS
OUT = Path("/home/hhchoi1022/ezphot/log/20261004_calspec_seeing")
CAL = Path("/home/hhchoi1022/ezphot/log/20261001_calspec_manuf")
NEWB = Path("/home/hhchoi1022/ezphot/log/20260927_std_newbkg")
PIX = 0.5056
R_CEN, R_EDGE = 1500, 3000
SNR_CLASSES = [(10, 30), (30, 100), (100, 1e9)]


def frame_seeing(name):
    cp = SO.CATD / (name[:-5] + "_cat.fits")
    try:
        with fits.open(cp) as h:
            d = h["LDAC_OBJECTS"].data; hdr = h[0].header
            m3, e3, m4 = (np.asarray(d[c], float) for c in ("MAG_APER_3", "MAGERR_APER_3", "MAG_APER_4"))
            fw, fl, pk = np.asarray(d["FWHM_IMAGE"], float), np.asarray(d["FLAGS"]), np.asarray(d["FLUX_MAX"], float)
            x, y = np.asarray(d["X_IMAGE"], float), np.asarray(d["Y_IMAGE"], float)
            nx, ny = hdr.get("NAXIS1", 9576), hdr.get("NAXIS2", 6388)
            row = dict(file=name, seeing_hdr=float(hdr.get("SEEING", np.nan)), ellip_hdr=float(hdr.get("ELLIP", np.nan)))
    except Exception:                                                # noqa: BLE001
        return None
    q = (fl == 0) & (1.0857 / np.clip(e3, 1e-9, None) >= 100) & (pk <= 50000) & (m3 < 90) & (m4 < 90) & (fw > 0.5)
    r = np.hypot(x - nx / 2, y - ny / 2)
    ap = m3 - m4
    row.update(n_bright=int(q.sum()), seeing=np.median(fw[q]) * PIX if q.sum() >= 5 else np.nan,
               ap34=np.median(ap[q]) * 1e3 if q.sum() >= 5 else np.nan)
    for lab, s in (("cen", q & (r < R_CEN)), ("edge", q & (r > R_EDGE))):
        row[f"seeing_{lab}"] = np.median(fw[s]) * PIX if s.sum() >= 5 else np.nan
        row[f"ap34_{lab}"] = np.median(ap[s]) * 1e3 if s.sum() >= 5 else np.nan
        row[f"n_{lab}"] = int(s.sum())
    return row


def harvest(names, n_proc):
    with Pool(n_proc) as pool:
        rows = [r for r in pool.imap_unordered(frame_seeing, names, chunksize=8) if r]
    return pd.DataFrame(rows)


def std_frames():
    """section 8.1 frames (manuf·beam, keep, std SNR > 10), Delta with and without the faint bias"""
    F = pd.read_csv(CAL / "calspec_manuf_frames.csv")
    F = F[(F.version == "manufacturer") & (F.variant == "beam")]
    K = pd.read_csv(CAL / "calspec_frames_flags.csv", usecols=["file", "curve", "keep"])
    K = K[K.curve == "beam"][["file", "keep"]].drop_duplicates("file")
    F = F.merge(K, on="file", how="inner")
    F = F[F.keep & (F.std_snr > 10)].copy()
    bias = SS.expected_bias(F.std_peak_e.to_numpy(), "CALSPEC")
    F["d_nobias"] = (F.d_gnd - F.zp + F.zp_x) * 1e3
    F["bias"] = bias
    F["d"] = F.d_nobias - bias
    return F[F.d.notna() & (F.d.abs() <= 1e3 * SS.D_MAX)].copy()


def wfit(x, y, w):
    """weighted LS slope and its error (scaled by sqrt(chi2/dof) when > 1)"""
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    x, y, w = x[ok], y[ok], w[ok]
    if len(x) < 5 or np.ptp(x) == 0:
        return np.nan, np.nan, len(x)
    A = np.vstack([np.ones_like(x), x]).T
    cov = np.linalg.inv(A.T @ (A * w[:, None]))
    b = cov @ (A.T @ (w * y))
    chi = np.sum(w * (y - A @ b) ** 2) / max(len(x) - 2, 1)
    return b[1], np.sqrt(cov[1, 1] * max(chi, 1.0)), len(x)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n-proc", type=int, default=24); ap.add_argument("--rerun", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore")
    F = std_frames()
    FC = pd.read_csv(NEWB / "faintcheck_stars.csv")
    sc = OUT / "frame_seeing.csv"
    if a.rerun or not sc.exists():
        S = harvest(sorted(set(F.file) | set(FC.file)), a.n_proc); S.to_csv(sc, index=False)
    S = pd.read_csv(sc)
    print(f"seeing for {len(S)} frames; header SEEING vs star FWHM: median ratio {np.nanmedian(S.seeing_hdr / S.seeing):.3f}")
    F = F.merge(S, on="file", how="left")
    F["err"] = 1.0857e3 / F.std_snr
    F["pivot"] = F["filter"].str[1:4].astype(int)
    F["red"] = F["pivot"] >= 600

    # S1 within (star, filter, unit) groups
    g = F.groupby(["object", "filter", "unit"])
    F["n_g"] = g.d.transform("size")
    F = F[F.n_g >= 3].copy()
    F["dd"] = F.d - g.d.transform("median")
    F["dd_nb"] = F.d_nobias - g.d_nobias.transform("median")
    F["ds"] = F.seeing - g.seeing.transform("median")
    F["dsnr"] = np.log10(F.std_snr) - np.log10(F.groupby(["object", "filter", "unit"]).std_snr.transform("median"))
    rows = []
    for lab, s in [("all", F.index == F.index)] + [(f"snr{lo}-{hi:g}", (F.std_snr >= lo) & (F.std_snr < hi)) for lo, hi in SNR_CLASSES] \
            + [("blue<600", ~F.red), ("red>=600", F.red), ("7DT10 m825", (F.unit == "7DT10") & (F["filter"] == "m825"))]:
        q = F[s]
        for col in ("dd", "dd_nb"):
            b, e, n = wfit(q.ds.to_numpy(), q[col].to_numpy(), 1 / q.err.to_numpy() ** 2)
            rows.append(dict(sample=lab, delta=col, slope_mmag_per_arcsec=b, err=e, n=n, seeing_spread=q.ds.std(),
                             rho=q[["ds", col]].corr(method="spearman").iloc[0, 1]))
    T1 = pd.DataFrame(rows); T1.round(3).to_csv(OUT / "S1_within_slopes.csv", index=False)
    pd.set_option("display.width", 220)
    print("S1 within-group slope of Delta vs seeing [mmag/arcsec]:\n", T1.round(2).to_string(index=False))
    # seeing vs SNR inside groups (worse seeing -> lower SNR in a fixed aperture?) -- a confounder check
    print("within-group Spearman(seeing, log SNR):", round(F[["ds", "dsnr"]].corr(method="spearman").iloc[0, 1], 3))

    # S1b between groups: group-median Delta vs group-median seeing, partial on SNR via the bias already removed
    G = F.groupby(["object", "filter", "unit"]).agg(d=("d", "median"), seeing=("seeing", "median"), snr=("std_snr", "median"),
                                                  n=("d", "size"), red=("red", "first")).reset_index()
    G["err"] = 1.2533 * 1.0857e3 / G.snr / np.sqrt(G.n)
    b, e, n = wfit(G.seeing.to_numpy(), G.d.to_numpy(), 1 / (G.err.to_numpy() ** 2 + 10.0 ** 2))
    print(f"S1b between groups: slope {b:.1f} +- {e:.1f} mmag/arcsec over {n} groups; Spearman {G[['seeing', 'd']].corr(method='spearman').iloc[0, 1]:.2f}")
    G.round(3).to_csv(OUT / "S1_groups.csv", index=False)

    # S2 field stars: residual vs SNR in seeing terciles
    FC = FC.merge(S[["file", "seeing"]], on="file", how="left").dropna(subset=["seeing"])
    bins = [15, 20, 30, 50, 100, 200, 1e9]
    FC["sbin"] = pd.cut(FC.snr, bins, right=False)
    out2 = []
    for ds, q in FC.groupby("dataset"):
        fr = q.drop_duplicates("file")[["file", "seeing"]]
        t = np.nanquantile(fr.seeing, [1 / 3, 2 / 3])
        q = q.assign(stert=np.digitize(q.seeing, t))
        for (sb, st), r in q.groupby(["sbin", "stert"], observed=True):
            out2.append(dict(dataset=ds, snr_bin=str(sb), seeing_tercile=int(st), seeing_med=r.seeing.median(), n=len(r),
                             frames=r.file.nunique(), res=r.res.median(), err=1.2533 * 1.4826 * np.median(np.abs(r.res - r.res.median())) / np.sqrt(len(r))))
    T2 = pd.DataFrame(out2); T2.round(2).to_csv(OUT / "S2_field_faint_vs_seeing.csv", index=False)
    print("S2 field-star residual [mmag] by SNR bin and seeing tercile (0 good, 2 bad):")
    print(T2.pivot_table(index=["dataset", "snr_bin"], columns="seeing_tercile", values="res", sort=False).round(1).to_string())
    print(T2.pivot_table(index=["dataset"], columns="seeing_tercile", values="seeing_med").round(2).to_string())
    # per-frame faint loss (SNR 15-30) vs seeing, frame-level slope
    pf = FC[(FC.snr >= 15) & (FC.snr < 30)].groupby(["dataset", "file"]).agg(res=("res", "median"), n=("res", "size"), seeing=("seeing", "first")).reset_index()
    pf = pf[pf.n >= 10]
    for ds, q in pf.groupby("dataset"):
        b, e, n = wfit(q.seeing.to_numpy(), q.res.to_numpy(), q.n.to_numpy().astype(float))
        print(f"S2 {ds}: per-frame SNR15-30 field loss vs seeing slope {b:.1f} +- {e:.1f} mmag/arcsec ({n} frames), "
              f"Spearman {q[['seeing', 'res']].corr(method='spearman').iloc[0, 1]:.2f}")

    # S3 aperture loss vs seeing, centre vs edge
    S3 = S.dropna(subset=["seeing", "ap34"])
    for lab in ("ap34", "ap34_cen", "ap34_edge"):
        q = S3.dropna(subset=[lab])
        b, e, n = wfit(q.seeing.to_numpy(), q[lab].to_numpy(), np.ones(len(q)))
        print(f"S3 {lab} (10\"->15\" loss, mmag) vs seeing: slope {b:.1f} +- {e:.1f} mmag/arcsec, median {q[lab].median():.1f} ({n} frames)")
    q = S3.dropna(subset=["ap34_cen", "ap34_edge"])
    q = q.assign(ce=q.ap34_cen - q.ap34_edge, fe=q.seeing_edge - q.seeing_cen)
    b, e, n = wfit(q.seeing.to_numpy(), q.ce.to_numpy(), np.ones(len(q)))
    print(f"S3 centre - edge aperture loss: median {q.ce.median():.1f} mmag, slope vs seeing {b:.1f} +- {e:.1f}; "
          f"edge - centre FWHM median {q.fe.median():.2f}\"")

    # figure
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    ax = axs[0, 0]
    for (lab, s), c in zip([(f"SNR {lo}-{hi:g}", (F.std_snr >= lo) & (F.std_snr < hi)) for lo, hi in SNR_CLASSES], ["#d7301f", "#fc8d59", "#4575b4"]):
        q = F[s]; ax.plot(q.ds, q.dd, ".", ms=3, alpha=0.4, color=c, label=f"{lab} (n {len(q)})")
        xb = np.arange(-1.5, 1.6, 0.25); idx = np.digitize(q.ds, xb)
        med = [(xb[i - 1] + 0.125, q.dd[idx == i].median()) for i in range(1, len(xb)) if (idx == i).sum() >= 8]
        if med:
            ax.plot(*zip(*med), "-", color=c, lw=2)
    ax.axhline(0, color="k", lw=0.6); ax.set_xlim(-1.6, 1.6); ax.set_ylim(-150, 150); ax.grid(alpha=0.3)
    ax.set_xlabel("frame seeing − (star, filter, unit) median [arcsec]"); ax.set_ylabel("Δ − group median [mmag]")
    ax.set_title("S1  standard: within-group Δ vs seeing", fontsize=10, loc="left"); ax.legend(fontsize=7)
    ax = axs[0, 1]
    ax.errorbar(G.seeing, G.d, yerr=G.err, fmt="none", ecolor="0.7", lw=0.6)
    sc_ = ax.scatter(G.seeing, G.d, c=np.log10(G.snr), cmap="viridis", s=16, zorder=3)
    plt.colorbar(sc_, ax=ax, label="log10 SNR (group median)")
    ax.axhline(0, color="k", lw=0.6); ax.set_ylim(-150, 200); ax.grid(alpha=0.3)
    ax.set_xlabel("group median seeing [arcsec]"); ax.set_ylabel("group median Δ [mmag]")
    ax.set_title("S1b  standard: between groups (star, filter, unit)", fontsize=10, loc="left")
    ax = axs[1, 0]
    for (ds, st), q in T2.groupby(["dataset", "seeing_tercile"]):
        x = [float(s.split(",")[0].strip("[")) for s in q.snr_bin]
        ax.errorbar(np.array(x) * (1 + 0.04 * (st - 1)), q.res, yerr=q.err, fmt="o-" if ds == "CALSPEC" else "s--", ms=4,
                    color=["#1a9850", "#fdae61", "#d73027"][st], label=f"{ds} seeing tercile {st} ({q.seeing_med.median():.1f}\")")
    ax.set_xscale("log"); ax.axhline(0, color="k", lw=0.6); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    ax.set_xlabel("field-star SNR (bin lower edge)"); ax.set_ylabel("field-star residual vs bright stars [mmag]")
    ax.set_title("S2  field stars: low-signal loss by seeing tercile", fontsize=10, loc="left")
    ax = axs[1, 1]
    ax.plot(S3.seeing, S3.ap34_cen, ".", ms=3, alpha=0.4, color="#2b8cbe", label="centre r<1500 px")
    ax.plot(S3.seeing, S3.ap34_edge, ".", ms=3, alpha=0.4, color="#e34a33", label="edge r>3000 px")
    ax.set_xlabel("frame seeing [arcsec]"); ax.set_ylabel("MAG_APER_3 − MAG_APER_4 (10\"→15\") [mmag]"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    ax.set_title("S3  aperture loss 10\"→15\" vs seeing", fontsize=10, loc="left")
    fig.suptitle("Seeing dependence of the CALSPEC 2024 offsets (manuf·beam, MAG_APER_3 10\")", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(OUT / "SE1_seeing.png", dpi=110); plt.close(fig)
    F.to_csv(OUT / "std_frames_seeing.csv", index=False)


if __name__ == "__main__":
    main()
