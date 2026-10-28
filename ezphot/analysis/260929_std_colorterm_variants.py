#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
260929_std_colorterm_variants.py — sensitivity of the standard-star colour term (260927_std_colorterm.py) to the star
selection, after the 2026-09-27 red SPSS stars were added.

Standard colour term ct = -dDelta/d(BP-RP) [mmag/mag], ground truth, manufacturer curve, SPSS only.
Variants (all on 260927_std_offset_summary.screened frames):
  A  default      : local ZP (dl_gnd), SPSS std SNR >= 100                                     (= 260927_std_colorterm)
  B  fallback     : local ZP, or the image ZP (d_gnd) when fewer than 5 local ZP stars; std SNR >= 50 with the
                    faint-star bias of 260927_std_offset_summary.expected_bias subtracted
  C  B, FGK/M only: main-sequence FGK/M standards only (field-like stars)
  D  B, no WD     : everything except WD/sdOB
Per variant: OLS ct with star bootstrap (1000), number of stars, BP-RP range.
Output (~/ezphot/log/20260927_std_newbkg/): std_colorterm_variants.csv, std_colorterm_variant_stars.csv (per-star Delta)
"""
import importlib
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "/lyman/data1/factory_hhchoi/code/ezphot/ezphot/analysis")
SS = importlib.import_module("260927_std_offset_summary")
CT = importlib.import_module("260927_std_colorterm")
AN = CT.AN
D = CT.D


def per_star(F, col):
    return F.dropna(subset=[col]).groupby("object").agg(d=(col, lambda v: 1e3 * np.median(v)), bprp=("bprp", "first"),
                                                        cls=("cls", "first"), n=("bprp", "size"))


def main():
    S = pd.read_csv(D / "std_colours.csv")
    key = dict(zip(zip(S.dataset, S.key), zip(S.bprp, S.cls)))
    F = SS.screened(pd.read_csv(D / "std_offset_frames.csv"))
    F = F[(F.dataset == "SPSS") & (F.version == "manufacturer")].copy()
    F["k"] = F.object.map(AN.norm)
    F["bprp"] = [key.get(("SPSS", k), (np.nan,))[0] for k in F.k]
    F["cls"] = [key.get(("SPSS", k), (np.nan, "?"))[1] for k in F.k]
    F = F[F.bprp.notna()]
    bias = SS.expected_bias(F.std_peak_e.to_numpy(), "SPSS") / 1e3
    F["A"] = np.where(F.std_snr >= 100, F.dl_gnd, np.nan)
    zp_fb = np.where(F.n_loc >= 5, F.dl_gnd, F.d_gnd)
    F["B"] = np.where(F.std_snr >= 50, zp_fb - bias, np.nan)
    rows, stars = [], []
    for (filt, var), g in F.groupby(["filter", "variant"]):
        base = dict(filter=filt, pivot=int(filt[1:4]), variant=var, a_field=round(1e3 * g.a_field.median(), 1))
        for v, col, keep in (("A", "A", None), ("B", "B", None), ("C", "B", ["FGK", "M"]), ("D", "B", "noWD")):
            p = per_star(g, col)
            p = p[p.d.abs() <= 1e3 * SS.D_MAX]
            if keep == "noWD":
                p = p[~p.cls.isin(["WD", "sdOB"])]
            elif keep:
                p = p[p.cls.isin(keep)]
            s, e = CT.fit(p)
            rows.append(dict(base, var=v, ct=round(s, 1), err=round(e, 1), n=len(p),
                             bprp_max=round(p.bprp.max(), 2) if len(p) else np.nan))
            if v == "B":
                stars += [dict(filter=filt, variant=var, object=o, d=round(r.d, 1), bprp=round(r.bprp, 3), cls=r.cls, n=r.n)
                          for o, r in p.iterrows()]
    R = pd.DataFrame(rows).sort_values(["variant", "pivot", "var"])
    R.to_csv(D / "std_colorterm_variants.csv", index=False)
    pd.DataFrame(stars).to_csv(D / "std_colorterm_variant_stars.csv", index=False)
    W = R.pivot_table(index=["variant", "pivot", "filter", "a_field"], columns="var", values=["ct", "err", "n"]).reset_index()
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
    print(W.to_string(index=False))
    for var in ("beam", "nobeam"):
        for v in "ABCD":
            t = R[(R.variant == var) & (R["var"] == v) & (R["pivot"] >= 475) & (R.n >= 6) & np.isfinite(R.err) & (R.err > 0)]
            print(f"{var:7s} {v}  >=475 N={len(t):2d}  ct median {t.ct.median():+5.1f}  wmean {np.average(t.ct, weights=1 / t.err ** 2):+5.1f}"
                  f"  chi2/N {np.mean((t.ct / t.err) ** 2):4.2f}  ct - a_field median {(t.ct - t.a_field).median():+5.1f}")


if __name__ == "__main__":
    main()
