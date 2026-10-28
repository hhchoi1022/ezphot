#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260918_spss_qa.py — standard photometry-level QA figures for the SPSS campaign frames of one unit.

Q panels: (1) seeing per filter/night, (2) detections per frame, (3) instrumental mag-err diagram +
SNR=5 limit, (4) sky background per frame, (5) SPSS standard star: SNR & FLUX_MAX (saturation check)
per filter, (6) astrometry/status summary. Uses the trimmed catalogs in 20260918_spss/catalogs.

  MPLCONFIGDIR=/tmp/ezphot-matplotlib /home/hhchoi1022/miniconda3/envs/too/bin/python -u \\
      ezphot/analysis/260918_spss_qa.py --unit 7DT01
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402
import pandas as pd                 # noqa: E402

warnings.filterwarnings("ignore")
from astropy.coordinates import SkyCoord   # noqa: E402
from astropy.io import fits          # noqa: E402

plt.rcParams["axes.unicode_minus"] = False
OUT = Path("/home/hhchoi1022/ezphot/log/20260918_spss")
CATD = OUT / "catalogs"
SPSS_CSV = ("/tmp/claude-10000/-lyman-data1-factory-hhchoi-code-ezphot/"
            "b1308704-c229-4e49-b19d-bd7bc334e4b7/scratchpad/spss_table1.csv")


def norm(s):
    return str(s).upper().replace(" ", "").replace("_", "").replace("-", "").replace("+", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", default="7DT01")
    a = ap.parse_args()
    meta = pd.read_csv(OUT / f"process_meta_{a.unit}.csv")
    ok = meta[meta.status == "ok"].copy()
    print(f"{a.unit}: {len(meta)} frames, ok {len(ok)}; status:", meta.status.str[:22].value_counts().to_dict())

    spss = pd.read_csv(SPSS_CSV)
    coord_of = {norm(s.Name): (float(s.RA_ICRS), float(s.DE_ICRS)) for _, s in spss.iterrows()}
    # Gaia position + proper motion (epoch 2016.0) from GAIA_BEST tiles -> propagate to obs epoch.
    # LTT/G-catalog SPSS are high-PM stars: J2000-ish coords miss by up to tens of arcsec in 2026.
    import importlib, sys as _sys
    _sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
    C = importlib.import_module("260911_colorterm_2026")
    sid_of = {norm(s.Name): str(int(s.GaiaEDR3)) for _, s in spss.iterrows() if pd.notna(s.GaiaEDR3)}
    gaia_astrom = {}
    T = C.TILES
    for key, (ra0, dec0) in coord_of.items():
        sid = sid_of.get(key)
        if sid is None:
            continue
        dd = np.abs(T.dec.values - dec0)
        dr = np.abs(((T.ra.values - ra0 + 180) % 360) - 180) * np.cos(np.radians(dec0))
        for tn in T[(dd < 0.8) & (dr < 0.8)].id.values:
            p = C.TILESRC / f"{tn}.csv"
            if not p.exists() or key in gaia_astrom:
                continue
            with open(p) as fh:
                fh.readline()
                for line in fh:
                    if line.split(",", 1)[0].lstrip("﻿") == sid:
                        cc = line.split(",", 7)
                        try:
                            gaia_astrom[key] = (float(cc[1]), float(cc[2]),
                                                float(cc[4]) if cc[4] else 0.0,
                                                float(cc[5]) if cc[5] else 0.0)
                        except ValueError:
                            pass
                        break
            if key in gaia_astrom:
                break

    def star_pos(key, mjd):
        """(ra, dec) at obs epoch; Gaia 2016.0 + PM if available, else catalog J2000 coords."""
        if key in gaia_astrom:
            ra, dec, pmra, pmdec = gaia_astrom[key]
            dt = ((mjd if np.isfinite(mjd) else 61300) - 57389.0) / 365.25
            return (ra + pmra * dt / 3.6e6 / np.cos(np.deg2rad(dec)), dec + pmdec * dt / 3.6e6)
        return coord_of.get(key)

    filts = sorted(ok["filter"].unique(), key=lambda f: (f[0] != "m", f))
    fcol = {f: plt.get_cmap("turbo")(i / max(len(filts) - 1, 1)) for i, f in enumerate(filts)}
    nights = sorted(ok["night"].unique())
    nmark = {n: m for n, m in zip(nights, ["o", "s", "^"])}

    # star-level harvest: match SPSS star in each catalog
    star_rows, magerr_sample = [], []
    for _, r in ok.iterrows():
        cp = CATD / (Path(r.file).name[:-5] + "_cat.fits")
        if not cp.exists():
            continue
        try:
            with fits.open(cp) as h:
                hdr = h[0].header
                d = h["LDAC_OBJECTS"].data
        except Exception:
            continue
        names = set(d.columns.names)
        rk = "X_WORLD" if "X_WORLD" in names else "ALPHA_J2000"
        dk = "Y_WORLD" if "Y_WORLD" in names else "DELTA_J2000"
        ra = np.asarray(d[rk], float); dec = np.asarray(d[dk], float)
        m3 = np.asarray(d["MAG_APER_3"], float); e3 = np.asarray(d["MAGERR_APER_3"], float)
        if len(magerr_sample) < 60000:
            sel = np.isfinite(m3) & np.isfinite(e3) & (m3 < 50)
            k = min(sel.sum(), 400)
            idxs = np.random.default_rng(1).choice(np.where(sel)[0], k, replace=False)
            for i in idxs:
                magerr_sample.append((r["filter"], m3[i], e3[i]))
        mjd_ = float(hdr.get("MJD", np.nan)) if "MJD" in hdr else np.nan
        tgt = star_pos(norm(r.object), mjd_)
        if tgt is None:
            continue
        sep = SkyCoord(ra, dec, unit="deg").separation(SkyCoord(*tgt, unit="deg")).arcsec
        j = int(np.argmin(sep))
        sat = float(hdr.get("SATURATE", np.nan))
        fm = float(d["FLUX_MAX"][j]) if "FLUX_MAX" in names else np.nan
        star_rows.append(dict(object=r.object, filter=r["filter"], night=r.night, sep=float(sep[j]),
                              detected=bool(sep[j] < 2.0), mag=m3[j], magerr=e3[j],
                              snr=1.0857 / max(e3[j], 1e-4), flux_max=fm,
                              satfrac=fm / sat if np.isfinite(sat) and sat > 0 else np.nan,
                              exptime=float(Path(r.file).name.split("_")[-2][:-1])))
    st = pd.DataFrame(star_rows)
    st.to_csv(OUT / f"qa_star_{a.unit}.csv", index=False)
    me = pd.DataFrame(magerr_sample, columns=["filter", "mag", "magerr"])

    fig, axes = plt.subplots(2, 3, figsize=(19, 10))
    ax = axes.ravel()
    # (1) seeing
    for n in nights:
        g = ok[ok.night == n]
        ax[0].scatter(g["filter"].map(lambda f: filts.index(f)) + np.random.uniform(-.15, .15, len(g)),
                      g.seeing, s=8, alpha=0.5, marker=nmark[n], label=n[:10])
    ax[0].set_xticks(range(len(filts))); ax[0].set_xticklabels(filts, rotation=45)
    ax[0].set_ylabel('seeing FWHM ["]'); ax[0].set_title("(1) per-frame seeing"); ax[0].legend(fontsize=8)
    # (2) detections
    for n in nights:
        g = ok[ok.night == n]
        ax[1].scatter(g["filter"].map(lambda f: filts.index(f)) + np.random.uniform(-.15, .15, len(g)),
                      g.n_det, s=8, alpha=0.5, marker=nmark[n])
    ax[1].set_xticks(range(len(filts))); ax[1].set_xticklabels(filts, rotation=45)
    ax[1].set_ylabel("N detections"); ax[1].set_yscale("log"); ax[1].set_title("(2) detections per frame")
    # (3) mag-err
    for f in filts:
        g = me[me["filter"] == f]
        if len(g):
            ax[2].scatter(g.mag, g.magerr, s=2, alpha=0.25, color=fcol[f], label=f)
    ax[2].axhline(0.2171, color="k", ls="--", lw=0.8)
    ax[2].text(0.02, 0.93, "SNR=5", transform=ax[2].transAxes, fontsize=8)
    ax[2].set(xlabel="instrumental MAG_APER_3 (10\")", ylabel="MAGERR", ylim=(0, 0.5),
              title="(3) mag - error (all frames, sampled)")
    ax[2].legend(fontsize=6, ncol=3, markerscale=4)
    # (4) sky background
    for n in nights:
        g = ok[ok.night == n]
        ax[3].scatter(g["filter"].map(lambda f: filts.index(f)) + np.random.uniform(-.15, .15, len(g)),
                      g.skyval, s=8, alpha=0.5, marker=nmark[n])
    ax[3].set_xticks(range(len(filts))); ax[3].set_xticklabels(filts, rotation=45)
    ax[3].set_ylabel("median BACKGROUND [ADU]"); ax[3].set_title("(4) sky level per frame")
    # (5) standard star SNR + saturation
    dt = st[st.detected]
    for f in filts:
        g = dt[dt["filter"] == f]
        if len(g):
            ax[4].scatter([filts.index(f)] * len(g) + np.random.uniform(-.15, .15, len(g)),
                          g.snr, s=10, alpha=0.6, color=fcol[f])
    ax[4].axhline(40, color="k", ls="--", lw=0.8)
    ax[4].set_xticks(range(len(filts))); ax[4].set_xticklabels(filts, rotation=45)
    ax[4].set_yscale("log"); ax[4].set_ylabel("standard-star SNR (10\")")
    ax[4].set_title(f"(5) SPSS star SNR — detected {dt.shape[0]}/{st.shape[0]} "
                    f"({100 * st.detected.mean():.0f}%), sat>0.8: {(dt.satfrac > 0.8).sum()}")
    # (6) per-object detection/saturation table (text)
    ax[5].axis("off")
    lines = [f"{'object':<12} {'n':>3} {'det%':>5} {'medSNR':>7} {'sat>0.8':>7}"]
    for o, g in st.groupby("object"):
        gd = g[g.detected]
        lines.append(f"{o:<12} {len(g):>3} {100 * g.detected.mean():>4.0f}% "
                     f"{gd.snr.median() if len(gd) else 0:>7.0f} {(gd.satfrac > 0.8).sum():>7}")
    ax[5].text(0.02, 0.98, "\n".join(lines), transform=ax[5].transAxes, va="top",
               family="monospace", fontsize=8.5)
    ax[5].set_title("(6) standard-star summary")
    for a_ in ax[:5]:
        a_.grid(alpha=0.3)
    fig.suptitle(f"Q1  SPSS campaign photometry QA — {a.unit}, nights 2026-09-14..16 "
                 f"({len(ok)} frames processed to photometry)")
    fig.tight_layout()
    fig.savefig(OUT / f"Q1_qa_{a.unit}.png", dpi=140)
    print(f"saved -> Q1_qa_{a.unit}.png, qa_star_{a.unit}.csv in {OUT}")


if __name__ == "__main__":
    main()
