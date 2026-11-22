#!/usr/bin/env python
"""
target_photometry: pull one target's photometry out of calibrated ezphot catalogs.

Uses CatalogSet + LightCurve.extract_source_info: one row per catalog (epoch/filter/unit)
with the matched source's MAGSKY_*/MAGERR_*/ZPERR_*/UL5SKY_* columns. 10" aperture = *_APER_3.
Adds SATDIST (distance to the nearest saturated star / its mask radius) and warns for rows with
SATDIST < 1.2: there the bright-star halo is masked out of the background map, so the sky is under-subtracted
(about +0.8 ADU/pix at 0.6) and the target too bright; use DIA or a local-annulus measurement for those rows.

dedupe=True / --dedupe keeps one catalog per (observatory, telname, filter, obsdate), the one with the deepest
UL5 (median UL5SKY_APER_3 of the catalog, else its info depth), e.g. when the same epoch was coadded twice;
off by default.

Python:
    from ezphot.recipes import target_photometry
    tbl = target_photometry(catalogset_or_list, ra, dec)

CLI (catalogs are found with DataBrowser('scidata') unless --paths is given; that search skips derived-image prefixes
and, unless --pattern itself ends with .circ.cat or .psf.cat, forced catalogs *.circ.cat and *.psf.cat):
    python -m ezphot.recipes.target_photometry --objname EG21 --observatory 7DT --ra 76.3 --dec -68.0 \
        [--filter m400] [--radius 5] [--out table.csv] [--plot] [--dedupe]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Union

import numpy as np

from ezphot.dataobjects import Catalog, CatalogSet, LightCurve
from ezphot.recipes import defaults as D
from ezphot.recipes._common import NothingToDo

DERIVED_PREFIXES = ('align_', 'com_', 'coadd_', 'scale_', 'conv_', 'sub_', 'inv_', 'subbkg_')
SATDIST_WARN = 1.2
FORCED_SUFFIXES = ('.circ.cat', '.psf.cat')   # forced aperture / PSF catalogs, not single-image catalogs


def _target_satdist(catalog: Catalog, ra: float, dec: float, matching_radius_arcsec: float) -> float:
    """SATDIST of the source nearest (ra, dec) in this catalog; NaN if no column or no source in the radius."""
    d = catalog.data
    if d is None or 'SATDIST' not in d.colnames or len(d) == 0:
        return np.nan
    cosd = np.cos(np.deg2rad(dec))
    sep = np.hypot((np.asarray(d['X_WORLD']) - ra) * cosd, np.asarray(d['Y_WORLD']) - dec) * 3600
    i = int(np.argmin(sep))
    return float(d['SATDIST'][i]) if sep[i] < matching_radius_arcsec else np.nan


def _deepest_per_epoch(catalogs: List[Catalog]):
    """(kept, dropped), both in input order: one catalog per (observatory, telname, filter, obsdate), the deepest UL5 =
    median of the finite UL5SKY_APER_3 values of the catalog, else info.depth (neither: shallowest; ties keep the first).
    Catalogs missing one of the keys are kept."""
    ul5_key = f'UL5SKY_{D.APER10_KEY}'
    best = {}
    for c in catalogs:
        key = (c.info.observatory, c.info.telname, c.info.filter, c.info.obsdate)
        if None in key:
            key = id(c)
        d = c.data
        ul5 = []
        if d is not None and ul5_key in d.colnames:
            ul5 = np.ma.masked_invalid(np.ma.asarray(d[ul5_key], dtype=float)).compressed()
        depth = float(np.median(ul5)) if len(ul5) else c.info.depth
        if depth is None or not np.isfinite(depth):
            depth = -np.inf
        if key not in best or depth > best[key][0]:
            best[key] = (depth, c)
    kept = {id(c) for _, c in best.values()}
    return [c for c in catalogs if id(c) in kept], [c for c in catalogs if id(c) not in kept]


def target_photometry(catalogs: Union[CatalogSet, List[Catalog]],
                      ra: float,
                      dec: float,
                      matching_radius_arcsec: float = 5.0,
                      out: Union[str, Path] = None,
                      plot: bool = False,
                      verbose: bool = True,
                      dedupe: bool = False):
    """Return an astropy Table (one row per catalog) for the source nearest (ra, dec); save CSV.
    dedupe=True first keeps one catalog per (observatory, telname, filter, obsdate), the deepest UL5 (UL5SKY_APER_3)."""
    catset = catalogs if isinstance(catalogs, CatalogSet) else CatalogSet(list(catalogs))
    if len(catset.catalogs) == 0:
        raise NothingToDo("no catalogs given")
    if dedupe:
        kept, dropped = _deepest_per_epoch(catset.target_catalogs)
        if verbose:
            print(f"[target_photometry] dedupe: kept {len(kept)}/{len(kept) + len(dropped)} catalogs (deepest UL5 per "
                  f"observatory/telname/filter/obsdate), dropped: "
                  + (', '.join(Path(str(c.path)).name for c in dropped[:10]) or 'none'))
        catset = CatalogSet(kept)
    lc = LightCurve(catset)
    tbl = lc.extract_source_info(ra, dec, matching_radius_arcsec=matching_radius_arcsec, verbose=verbose)
    if tbl is None:
        if verbose:
            print(f"[target_photometry] no source within {matching_radius_arcsec}\" of ({ra}, {dec})")
        return None

    if 'path' in tbl.colnames:
        satdist = {str(c.path): _target_satdist(c, ra, dec, matching_radius_arcsec) for c in catset.catalogs}
        tbl['SATDIST'] = [satdist.get(str(p), np.nan) for p in tbl['path']]
        near = tbl[tbl['SATDIST'] < SATDIST_WARN]
        if len(near) and verbose:
            print(f"[target_photometry] WARNING: {len(near)}/{len(tbl)} rows have the target inside a bright-star halo "
                  f"(SATDIST < {SATDIST_WARN}); background under-subtracted there, magnitudes too bright -> use DIA "
                  f"or a local annulus for: " + ', '.join(Path(str(p)).name for p in near['path'][:10]))

    if out is None:
        objname = catset.catalogs[0].info.objname or 'unknown'
        out = Path.home() / 'ezphot' / 'log' / 'recipes' / objname / f"target_photometry_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    out = Path(out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    tbl.write(out, format='csv', overwrite=True)
    if verbose:
        print(f"[target_photometry] {len(tbl)} rows -> {out}")

    if plot:
        k = D.APER10_KEY
        figures, _, _, _ = lc.plot(ra=ra, dec=dec, matching_radius_arcsec=matching_radius_arcsec,
                                   flux_key=f'MAGSKY_{k}', fluxerr_key=f'MAGERR_{k}',
                                   zperr_key=f'ZPERR_{k}', depth_key=f'UL5SKY_{k}', verbose=verbose)
        figs = list(figures.values()) if isinstance(figures, dict) else list(figures or [])
        for i, fig in enumerate(figs):
            if isinstance(fig, (str, Path)):          # LightCurve.plot already saved it
                png = fig
            else:
                png = out.with_suffix(f'.{i}.png' if len(figs) > 1 else '.png')
                fig.savefig(png, dpi=150, bbox_inches='tight')
            if verbose:
                print(f"[target_photometry] figure -> {png}")
    return tbl


def load_catalogs(observatory: str = None, telkey: str = None, telname: str = None,
                  objname: str = None, filter: str = None, obsdate: str = None,
                  pattern: str = '*.cat', paths: List[str] = None) -> CatalogSet:
    """CatalogSet of single-image catalogs, from explicit paths or DataBrowser('scidata') (without derived-image
    prefixes, and without forced *.circ.cat / *.psf.cat catalogs unless pattern ends with .circ.cat or .psf.cat;
    explicit paths are taken as given)."""
    if paths:
        return CatalogSet([Catalog(path=p, load=True) for p in paths])
    from ezphot.utils import DataBrowser
    db = DataBrowser('scidata')
    for key, val in dict(observatory=observatory, telkey=telkey, telname=telname,
                         objname=objname, filter=filter, obsdate=obsdate).items():
        if val is not None:
            setattr(db, key, val)
    catset = db.search(pattern=pattern, return_type='catalog')
    drop_forced = not pattern.endswith(FORCED_SUFFIXES)   # a pattern for forced catalogs keeps them
    cats = [c for c in catset.catalogs if not Path(c.path).name.startswith(DERIVED_PREFIXES)
            and not (drop_forced and Path(c.path).name.endswith(FORCED_SUFFIXES))]
    return CatalogSet(cats)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--paths', nargs='+', help='explicit .cat paths (skips DataBrowser search)')
    ap.add_argument('--observatory'); ap.add_argument('--telkey'); ap.add_argument('--telname')
    ap.add_argument('--objname'); ap.add_argument('--filter'); ap.add_argument('--obsdate')
    ap.add_argument('--pattern', default='*.cat')
    ap.add_argument('--ra', type=float, required=True); ap.add_argument('--dec', type=float, required=True)
    ap.add_argument('--radius', type=float, default=5.0, help='matching radius [arcsec]')
    ap.add_argument('--plot', action='store_true')
    ap.add_argument('--dedupe', action='store_true',
                    help='keep one catalog per (observatory, telname, filter, obsdate): the deepest UL5 (UL5SKY_APER_3)')
    ap.add_argument('--out')
    a = ap.parse_args(argv)

    catset = load_catalogs(a.observatory, a.telkey, a.telname, a.objname, a.filter, a.obsdate, a.pattern, a.paths)
    print(f"[target_photometry] {len(catset.catalogs)} catalogs")
    tbl = target_photometry(catset, a.ra, a.dec, matching_radius_arcsec=a.radius, out=a.out, plot=a.plot, dedupe=a.dedupe)
    return 0 if tbl is not None else 1


if __name__ == '__main__':
    raise SystemExit(main())
