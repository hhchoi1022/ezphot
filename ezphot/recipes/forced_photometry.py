#!/usr/bin/env python
"""
forced_photometry: forced circular-aperture photometry at given sky positions on calibrated images
(single frames, stacks or difference images), all targets of an image in one call.

    calibrate='header' (default): photometry_forced_circular at (ra, dec) -> apply_zp with the image header
        ZP_APER_k / ZPERR_APER_k (UL3SKY/UL5SKY = the catalog's UL3/UL5_APER_k + ZP), as dia does on difference images
    calibrate='reference_stars': photometry_forced_circular at (ra, dec) and at the reference stars of the image
        (X_WORLD/Y_WORLD of <img>.fits.refcat, written by process_images/stack) -> photometric_calibration on that
        catalog (process_images photcal settings; reference stars picked on MAG_APER_3, no region cut: the refcat
        stars already passed the region cut of process_images). The forced catalog has no FWHM/CLASS_STAR/
        ELONGATION/FLAGS columns, so those star cuts and SATDIST are skipped, and the isolation cut only sees the
        measured positions.
        The recipe fixes update_header=False, save_image=False, save=True, mag_lower/upper (defaults.MAG_RANGE) and
        the region (ra_deg=dec_deg=None, inner_fraction=1); config cannot change these.

Background: difference images (sub_*) have no background map (circular_photometry computes the bkgrms from the
image). The forced_photometry_sub section (defaults.FORCED_PHOTOMETRY_SUB, on top of forced_photometry for them)
measures the flux over a local annulus mean (bkg_source 'annulus'), scales SKYSIG/UL3/UL5 and the background part of
FLUXERR by the noise ratio of random empty apertures of the same image (noise_ratio, NOISE_RATIO_APER_k) and adds the
Poisson noise of the host, the aperture flux of the convolved reference conv_<name> next to it (hotpants, CONVOL00
TEMPLATE), to FLUXERR/MAGERR only (host_source 'conv'; 'missing' without conv_, 'not template' when conv_ is the convolved science
image, 'off' with host_poisson=False, 'not finite' for a target whose conv_ aperture or annulus is not finite: no host
term, HOST_FLUX_APER_k NaN). Other images use their .bkgmap/.bkgrms sidecars, i.e. the background of the
process_images/stack photometry that set the header ZP (stacks: the propagated bkgrms). A missing sidecar is computed
from the image in memory, without a source mask (sources only sigma-clipped, so it may differ from process_images);
bkg_source / bkgrms_source say which.

PSF (config forced_photometry_psf, defaults.FORCED_PHOTOMETRY_PSF; off by default, CLI --psf): after the aperture
values, on images that are not difference images, PSFPhotometry.build_epsf (grid, model, poly_order, max_stars_total,
snr_lower, snr_lower_fallback, recenter_iter; default model 'poly' of order <= 3 (order ladder) on the automatic tiling
of grid 1; stars of the photcal section's catalog)
-> psf_photometry at (ra, dec) and the reference stars of <img>.fits.refcat (same background map/bkgrms as the
aperture photometry, then the local annulus by default; fit_radius_fwhm, background, weighting, recenter (default
'offset': one smooth position offset of the image, from the position fits at S/N >= recenter_snr, added to the targets
and the reference stars alike), recenter_snr, recenter_max_shift and the other psf_photometry arguments) -> calibrate_zp
(ZP_PSF from those reference stars on MAG_PSF with the reference_stars rules above, also with calibrate='header').
Writes <img>.fits.psf.cat (+ .info) next to the image only with a finite ZP_PSF; the header and the aperture catalog
do not change. Summary columns only then: psf_catalog, psf_source (epsf | moffat | mixed), psf_model (build_epsf
info 'psf_model', one per image: poly3 | poly2 | poly1 | grid | moffat), psf_nstars (fewest ePSF
stars of a tile; model 'poly': stars of the fit, or with no usable polynomial order the usable stars of its
grid-1 fallback), n_ref_psf, then
per target MAGSKY_PSF, MAGERR_PSF, ZP_PSF, ZPERR_PSF, UL5SKY_PSF, CHI2_PSF, FLAG_PSF, DX_PSF, DY_PSF (px; recenter
'offset': the offset applied at the target, 'each': its fitted move, NaN when not fitted; empty with recenter False)
(psf_catalog only once the
.psf.cat is written). A PSF failure fails the image's rows at step psf (a sub_ image), psf_refcat (no .refcat),
psf_epsf, psf_photometry or psf_zp (no finite ZP_PSF or n_ref_psf 0); the aperture values stay in them.

Products next to each image (its own folder, not the config archive layout): <img>.fits.circ.cat (+ .info), an ASCII
table with X_WORLD/Y_WORLD and the per-aperture MAGSKY/MAGERR/ZP/ZPERR/UL5SKY_APER[_k] columns; with
'reference_stars' also <img>.fits.circ.refcat. A second run on the same image overwrites them (a 'header' run leaves
an older .circ.refcat as it is). No figures. The input FITS is not rewritten.
One summary row per image x target (target = index in ra/dec): success/step/error (the same for all targets of an
image), n_ref (reference stars of 'reference_stars'), bkg_source, bkgrms_source, catalog and the target's
MAGSKY/MAGERR/ZP/ZPERR/UL5SKY in the 10" aperture (*_APER_3), then noise_ratio (r of the 10" aperture) and
host_source (difference images, else empty). An image without a finite ZP_APER_3 fails (header: step
apply_zp, no ZP_APER_3 in the header; reference_stars: step photometric_calibration, also when n_ref is 0); an image
that cannot be opened fails at step load; an empty UL5SKY_APER_3 means no depth was available.
While it runs, <out>.progress.json next to the summary counts the finished images (read by recipes.status).

Python:
    from ezphot.recipes import forced_photometry
    summary = forced_photometry(['coadd_....fits', 'stack_....csv'], ra=[76.3, 76.4], dec=[-68.0, -68.1])

CLI (images: FITS paths and/or process_images/stack summary CSVs, whose successful rows are used):
    python -m ezphot.recipes.forced_photometry <image.fits | summary.csv> [...] --ra 76.3 [76.4 ...] \
        --dec -68.0 [-68.1 ...] [--calibrate header|reference_stars] [--psf] [--n-proc 4] [--out forced.csv]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd

from ezphot.dataobjects import Catalog
from ezphot.imageobjects import ScienceImage
from ezphot.methods import PSFPhotometry
from ezphot.recipes import defaults as D
from ezphot.recipes._common import NothingToDo
from ezphot.recipes._progress import run_tasks
from ezphot.recipes.stack import load_summary

FORCED_CONFIG_KEYS = ('forced_photometry', 'forced_photometry_sub', 'photcal', 'forced_photometry_psf')
CALIBRATE = ('header', 'reference_stars')
FORCED_KEYS = [f'{p}_{D.APER10_KEY}' for p in ('MAGSKY', 'MAGERR', 'ZP', 'ZPERR', 'UL5SKY')]
ROW_KEYS = ('path', 'target', 'ra', 'dec', 'telname', 'filter', 'obsdate', 'exptime', 'calibrate', 'success', 'step',
            'error', 'n_ref', 'bkg_source', 'bkgrms_source', 'catalog')   # summary columns before FORCED_KEYS
NOISE_KEYS = ['noise_ratio', 'host_source']   # summary columns after FORCED_KEYS (difference images), per target
# Summary columns after NOISE_KEYS, only with forced_photometry_psf enabled: per image, then per target
PSF_ROW_KEYS = ['psf_catalog', 'psf_source', 'psf_model', 'psf_nstars', 'n_ref_psf']
PSF_KEYS = ['MAGSKY_PSF', 'MAGERR_PSF', 'ZP_PSF', 'ZPERR_PSF', 'UL5SKY_PSF', 'CHI2_PSF', 'FLAG_PSF', 'DX_PSF', 'DY_PSF']
# Processing recipes make no figures. Forced catalogs have no MAG_AUTO (the photcal section's key): reference
# stars are picked on the 10" aperture; photcal still fits the ZP of every aperture.
RECIPE_DEFAULTS = {'forced_photometry': dict(visualize=False, save_fig=False), 'forced_photometry_sub': {},
                   'photcal': dict(visualize=False, save_fig=False, magnitude_key=f'MAG_{D.APER10_KEY}',
                                   magnitudeerr_key=f'MAGERR_{D.APER10_KEY}'),
                   'forced_photometry_psf': {}}
# photcal keys that PSFPhotometry.calibrate_zp sets itself
PSF_ZP_FIXED = ('magnitude_key', 'magnitudeerr_key', 'update_header', 'save_image', 'save_refcat', 'save')
# forced_photometry_psf keys for PSFPhotometry.build_epsf (the others go to psf_photometry)
PSF_EPSF_KEYS = ('grid', 'model', 'poly_order', 'max_stars_total', 'snr_lower', 'snr_lower_fallback', 'recenter_iter')


def _config(overrides: dict = None) -> dict:
    cfg = {k: dict(getattr(D, k.upper()), **RECIPE_DEFAULTS[k]) for k in FORCED_CONFIG_KEYS}
    for k, v in (overrides or {}).items():
        if k not in cfg:
            raise KeyError(f"unknown config section {k!r}; expected one of {FORCED_CONFIG_KEYS}")
        cfg[k].update(v)
    return cfg


def _targets(ra, dec):
    ra, dec = np.atleast_1d(np.asarray(ra, dtype=float)), np.atleast_1d(np.asarray(dec, dtype=float))
    if ra.ndim != 1 or ra.shape != dec.shape or len(ra) == 0:
        raise ValueError(f"ra and dec need the same number (>= 1) of values, got shapes {ra.shape} and {dec.shape}")
    return ra, dec


def _background(target_img: ScienceImage):
    """(bkg, bkgrms, bkg_source, bkgrms_source) for the forced photometry; target_img.savedir is its own folder."""
    if target_img.path.name.startswith('sub_'):   # no map; a local annulus with forced_photometry_sub (forced_image)
        return None, None, 'none', 'image'
    bkg, bkgrms = target_img.bkgmap, target_img.bkgrms
    sources = ('sidecar' if bkg is not None else 'image', 'sidecar' if bkgrms is not None else 'image')
    if bkg is None or bkgrms is None:
        # In memory and without a source mask: mask_sources/mask_circle delete an existing <img>.srcmask here.
        quiet = dict(save=False, verbose=False, visualize=False, save_fig=False)
        ivpmask = target_img.calculate_invalidmask(**dict(D.INVALIDMASK, **quiet))
        if bkg is None:
            bkg = target_img.calculate_bkg(target_ivpmask=ivpmask, **dict(D.BKGMAP, **quiet))
        if bkgrms is None:
            bkgrms = target_img.calculate_bkgrms(target_ivpmask=ivpmask, **dict(D.BKGRMS, **quiet))
    return (bkg, bkgrms) + sources


def _host(target_img: ScienceImage):
    """(host image or None, host_source) for the host Poisson term of a difference image sub_<name>: conv_<name> in its
    folder (hotpants output) when it is the convolved reference (header CONVOL00 TEMPLATE). Opened read-only."""
    conv = target_img.path.with_name('conv_' + target_img.path.name[len('sub_'):])
    if not conv.exists():
        return None, 'missing'
    if str(target_img.header.get('CONVOL00', '')).strip().upper() != 'TEMPLATE':
        return None, 'not template'
    return ScienceImage(path=conv, telinfo=target_img.telinfo, load=True), 'conv'


def forced_image(target_img: ScienceImage,
                 ra,
                 dec,
                 calibrate: str = 'header',
                 config: dict = None,
                 clear: bool = True) -> List[dict]:
    """Forced photometry at every (ra, dec) on one image, in one call; one summary row per target.
    Processing errors do not raise (rows with success=False); bad arguments (config section, calibrate, ra/dec) do.
    Sets target_img.savedir to the image's folder, where the .circ.cat (and with forced_photometry_psf the .psf.cat)
    is written."""
    cfg = _config(config)
    if calibrate not in CALIBRATE:
        raise ValueError(f"calibrate must be one of {CALIBRATE}, got {calibrate!r}")
    ra, dec = _targets(ra, dec)
    psf_cfg = dict(cfg['forced_photometry_psf'])
    psf_on = bool(psf_cfg.pop('enabled', False))
    row = dict(dict.fromkeys(ROW_KEYS), path=str(target_img.path), calibrate=calibrate, success=False)
    values = _empty_values(ra, psf_on)
    bkg = bkgrms = refcat = catalog = used = host_img = psf_catalog = None
    try:
        row['step'] = 'load'   # header values that do not parse (DATE-OBS, EXPTIME) fail this image only
        row.update(telname=target_img.telname, filter=target_img.filter, obsdate=target_img.obsdate,
                   exptime=target_img.exptime)
        target_img.savedir = target_img.path.parent   # products next to the image, not the config archive layout
        row['step'] = 'background'
        bkg, bkgrms, row['bkg_source'], row['bkgrms_source'] = _background(target_img)
        phot = dict(cfg['forced_photometry'])
        if target_img.path.name.startswith('sub_'):   # difference image: forced_photometry_sub on top
            sub = dict(cfg['forced_photometry_sub'])
            host_source = 'off'
            if sub.pop('host_poisson', False):
                host_img, host_source = _host(target_img)
            for v in values:
                v['host_source'] = host_source
            phot.update(sub, host_img=host_img)
            if phot.get('annulus_width_arcsec') is not None:
                row['bkg_source'] = 'annulus'
        x, y = ra, dec
        if calibrate == 'reference_stars':
            row['step'] = 'refcat'
            refcat = Catalog(target_img.savepath.refcatalogpath, catalog_type='reference', load=False, target_img=target_img)
            if not refcat.is_exists:
                raise FileNotFoundError(f"no reference-star catalog {refcat.path}")
            x = np.concatenate([ra, np.asarray(refcat.data['X_WORLD'], dtype=float)])
            y = np.concatenate([dec, np.asarray(refcat.data['Y_WORLD'], dtype=float)])
        row['step'] = 'forced_photometry'
        catalog = target_img.photometry_forced_circular(
            x_arr=x, y_arr=y, unit='coord', target_bkg=bkg, target_bkgrms=bkgrms,
            **dict(phot, save=True))   # save=False would remove the .circ.cat
        row['catalog'] = str(catalog.path)
        ratio_key, host_key = f'NOISE_RATIO_{D.APER10_KEY}', f'HOST_FLUX_{D.APER10_KEY}'
        for i, v in enumerate(values):   # the targets are the first len(ra) rows
            if ratio_key in catalog.data.colnames:
                v['noise_ratio'] = float(catalog.data[ratio_key][i])
            if host_key in catalog.data.colnames and not np.isfinite(float(catalog.data[host_key][i])):
                v['host_source'] = 'not finite'   # conv_ aperture or annulus not finite: no host term
        if calibrate == 'header':
            row['step'] = 'apply_zp'
            catalog = target_img.apply_zp(catalog, verbose=False, save=True)
        else:
            row['step'] = 'photometric_calibration'
            # The input FITS stays as it is; no region cut, the refcat stars already passed the process_images one.
            photcal = dict(cfg['photcal'], update_header=False, save_image=False, save=True,
                           ra_deg=None, dec_deg=None, inner_fraction=1.0)
            photcal['mag_lower'], photcal['mag_upper'] = D.MAG_RANGE.get(target_img.filter, D.MAG_RANGE_DEFAULT)
            _, catalog, used, _ = target_img.photometric_calibration(target_catalog=catalog, **photcal)
            row['n_ref'] = used.nsources
        data = catalog.data   # the targets are its first len(ra) rows
        zp_key = FORCED_KEYS[2]   # photcal writes a rough MAGSKY before selecting stars: judge on the ZP column
        if zp_key not in data.colnames or not np.isfinite(float(data[zp_key][0])) or row['n_ref'] == 0:
            why = ('missing or not finite in the image header' if calibrate == 'header'
                   else f"{row['n_ref']} reference stars")
            raise RuntimeError(f"no finite {zp_key}: {why}")
        for i, v in enumerate(values):
            for k in FORCED_KEYS:
                if k in data.colnames:
                    v[k] = float(data[k][i])
        if psf_on:   # after the aperture values: a failure here keeps them in the (failed) rows
            row['step'] = 'psf'
            if target_img.path.name.startswith('sub_'):
                raise NotImplementedError("PSF photometry of a difference image needs the PSF and ZP_PSF of its science "
                                          "image: PSFPhotometry.psf_photometry(psf=..., zp_psf=...)")
            row['step'] = 'psf_refcat'
            if refcat is None:
                refcat = Catalog(target_img.savepath.refcatalogpath, catalog_type='reference', load=False,
                                 target_img=target_img)
            if not refcat.is_exists:
                raise FileNotFoundError(f"no reference-star catalog {refcat.path} for ZP_PSF")
            psfphot = PSFPhotometry()
            row['step'] = 'psf_epsf'
            epsf_kwargs = {k: psf_cfg.pop(k) for k in PSF_EPSF_KEYS if k in psf_cfg}
            psf, psf_info = psfphot.build_epsf(target_img, **epsf_kwargs,
                                               catalog_type=cfg['photcal']['catalog_type'],
                                               catalog_version=cfg['photcal']['catalog_version'], verbose=False)
            sources = set(np.asarray(psf_info['source']).ravel().tolist())
            psf_row = dict(psf_source=sources.pop() if len(sources) == 1 else 'mixed', psf_model=psf_info['psf_model'],
                           psf_nstars=int(np.min(psf_info['nstars'])))
            row['step'] = 'psf_photometry'   # targets first, then the reference stars of ZP_PSF
            psf_catalog = psfphot.psf_photometry(
                target_img, np.concatenate([ra, np.asarray(refcat.data['X_WORLD'], dtype=float)]),
                np.concatenate([dec, np.asarray(refcat.data['Y_WORLD'], dtype=float)]), psf=psf, psf_info=psf_info,
                unit='coord', target_bkg=bkg, target_bkgrms=bkgrms, save=False, verbose=False, **psf_cfg)
            row['step'] = 'psf_zp'
            # ZP_PSF: the reference_stars rules (photcal section, no region cut, defaults.MAG_RANGE) on MAG_PSF
            zp_kwargs = {k: v for k, v in cfg['photcal'].items() if k not in PSF_ZP_FIXED}
            zp_kwargs.update(ra_deg=None, dec_deg=None, inner_fraction=1.0)
            zp_kwargs['mag_lower'], zp_kwargs['mag_upper'] = D.MAG_RANGE.get(target_img.filter, D.MAG_RANGE_DEFAULT)
            psf_catalog, zp_psf, _, psf_row['n_ref_psf'] = psfphot.calibrate_zp(target_img, psf_catalog, save=False,
                                                                                **zp_kwargs)
            for v in values:
                v.update(psf_row)
            if not np.isfinite(zp_psf) or psf_row['n_ref_psf'] == 0:   # no .psf.cat is written then
                raise RuntimeError(f"no finite ZP_PSF: {psf_row['n_ref_psf']} reference stars")
            psf_catalog.write(verbose=False)
            data = psf_catalog.data
            for i, v in enumerate(values):
                v['psf_catalog'] = str(psf_catalog.path)   # only once written: empty in failed rows
                for k in PSF_KEYS:
                    if k in data.colnames:
                        v[k] = int(data[k][i]) if k == 'FLAG_PSF' else float(data[k][i])
        row['success'] = True
        row['step'] = 'done'
    except Exception as e:
        row['error'] = f"{type(e).__name__}: {e}"
    finally:
        if clear:
            for obj in (target_img, bkg, bkgrms, refcat, catalog, used, host_img, psf_catalog):
                if obj is not None:
                    try:
                        obj.clear(verbose=False)
                    except Exception:
                        pass
    return _target_rows(row, ra, dec, values)


def _empty_values(ra, psf: bool = False) -> List[dict]:
    return [dict.fromkeys(FORCED_KEYS + NOISE_KEYS + (PSF_ROW_KEYS + PSF_KEYS if psf else [])) for _ in ra]


def _target_rows(row: dict, ra, dec, values: List[dict] = None, psf: bool = False) -> List[dict]:
    values = values or _empty_values(ra, psf)
    return [dict(row, target=i, ra=float(r), dec=float(d), **v) for i, (r, d, v) in enumerate(zip(ra, dec, values))]


def _image_paths(images: Union[str, Path, List]) -> List[str]:
    """Resolved image paths, duplicates dropped: FITS paths, and for *.csv (process_images/stack summaries) the paths
    of their successful rows."""
    paths = []
    for f in ([images] if isinstance(images, (str, Path)) else images):
        paths += [str(Path(p).expanduser().resolve()) for p in (load_summary(f) if str(f).endswith('.csv') else [f])]
    return list(dict.fromkeys(paths))


def _worker(args):
    path, ra, dec, calibrate, config = args
    try:
        target_img = ScienceImage(path=path, load=True)
    except Exception as e:   # missing file, unreadable header, unknown telescope: failed rows, the run goes on
        row = dict(dict.fromkeys(ROW_KEYS), path=path, calibrate=calibrate, success=False, step='load',
                   error=f"{type(e).__name__}: {e}")
        rows = _target_rows(row, ra, dec, psf=bool(_config(config)['forced_photometry_psf'].get('enabled')))
    else:
        rows = forced_image(target_img, ra, dec, calibrate, config, True)
    return dict(success=rows[0]['success'], rows=rows)   # one progress step per image


def forced_photometry(images: Union[str, Path, List],
                      ra,
                      dec,
                      calibrate: str = 'header',
                      config: dict = None,
                      n_proc: int = 4,
                      out: Union[str, Path] = None,
                      verbose: bool = True) -> pd.DataFrame:
    """Run forced_image over FITS paths and/or the successful images of process_images/stack summary CSVs;
    return (and save) one row per image x target."""
    paths = _image_paths(images)
    if len(paths) == 0:
        raise NothingToDo("no images given")
    ra, dec = _targets(ra, dec)
    _config(config)   # unknown sections fail here, not once per image
    if calibrate not in CALIBRATE:
        raise ValueError(f"calibrate must be one of {CALIBRATE}, got {calibrate!r}")
    if out is None:   # decided at the start: <out>.progress.json is written next to it while running
        objname = ScienceImage(path=paths[0], load=True).objname or 'unknown'
        out = Path.home() / 'ezphot' / 'log' / 'recipes' / objname / f"forced_photometry_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    out = Path(out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tasks = [(p, ra, dec, calibrate, config) for p in paths]
    rows = [row for result in run_tasks(_worker, tasks, n_proc, out) for row in result['rows']]
    summary = pd.DataFrame(rows)
    summary['n_ref'] = summary['n_ref'].astype('Int64')   # integer also when failed rows (empty) are mixed in
    for k in ('psf_nstars', 'n_ref_psf', 'FLAG_PSF'):   # forced_photometry_psf enabled
        if k in summary.columns:
            summary[k] = summary[k].astype('Int64')
    summary.to_csv(out, index=False)
    if verbose:
        n_ok = int(summary['success'].sum())
        print(f"[forced_photometry] {n_ok}/{len(summary)} rows ({len(paths)} images x {len(ra)} targets) "
              f"succeeded in {time.time() - t0:.0f}s")
        if n_ok < len(summary):
            failed = summary.loc[~summary['success']].drop_duplicates('path')
            print(failed[['path', 'step', 'error']].to_string(index=False))
        print(f"[forced_photometry] summary -> {out}")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('images', nargs='+', help='FITS image(s) and/or process_images or stack summary CSV(s)')
    ap.add_argument('--ra', type=float, nargs='+', required=True)
    ap.add_argument('--dec', type=float, nargs='+', required=True)
    ap.add_argument('--calibrate', choices=CALIBRATE, default='header')
    ap.add_argument('--n-proc', type=int, default=4)
    ap.add_argument('--out', help='summary CSV path')
    ap.add_argument('--psf', action='store_true',
                    help='also forced PSF photometry (config forced_photometry_psf enabled; not for difference images)')
    a = ap.parse_args(argv)
    config = {'forced_photometry_psf': dict(enabled=True)} if a.psf else None
    summary = forced_photometry(a.images, a.ra, a.dec, calibrate=a.calibrate, config=config, n_proc=a.n_proc,
                                out=a.out)
    return 0 if summary['success'].all() else 1


if __name__ == '__main__':
    raise SystemExit(main())
