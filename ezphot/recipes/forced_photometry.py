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
values, PSFPhotometry.build_epsf (grid, model, poly_order, max_stars_total,
snr_lower, snr_lower_fallback, recenter_iter; default model 'poly' of order <= 3 (order ladder) on the automatic tiling
of grid 1; stars of the photcal section's catalog)
-> psf_photometry at (ra, dec) and the reference stars of <img>.fits.refcat (same background map/bkgrms as the
aperture photometry, then the local annulus by default; fit_radius_fwhm, background, weighting, recenter (default
'offset': one smooth position offset of the image, from the position fits at S/N >= recenter_snr, added to the targets
and the reference stars alike), recenter_snr, recenter_max_shift and the other psf_photometry arguments) -> calibrate_zp
(ZP_PSF from those reference stars on MAG_PSF with the reference_stars rules above, also with calibrate='header').
A difference image sub_<name> (hotpants output of dia) takes these three steps on the image of its PSF and flux scale
in its folder, by header CONVOL00: TEMPLATE -> sci_<name>, IMAGE -> conv_<name> with SEEING of ref_<name> (in memory),
with sci_<name>.fits.refcat (the science image's .refcat, copied by dia) and that image's background as above; then
psf_photometry on sub_ at the targets only: that ePSF, ZP_PSF/ZPERR_PSF, positions (sub_ WCS) + the DX_PSF/DY_PSF of
that image at the target (NaN: none) fixed (recenter False), the annulus radii of the aperture photometry of sub_
(forced_photometry_sub), the PSF noise ratio of up to n_empty_apertures random empty positions (noise_ratio,
_psf_noise_ratio) and the host Poisson term of conv_ (host_poisson, CONVOL00 TEMPLATE); with IMAGE sub_ is measured
through a second object with that SEEING (the given target_img does not change). Correlated noise enters through
the empirical ratio only (no decorrelation).
Neighbours (forced_photometry_psf neighbors, off by default): the sources of the measured image's SExtractor catalog
<img>.fits.cat next to it (_neighbors; sub_: its own .cat, for the targets and the empty positions of the noise ratio;
the sci_/conv_ reference stars without) are fitted with each target and reference star (psf_photometry neighbor_x/y:
fixed 0-based X_IMAGE/Y_IMAGE of FLUX_AUTO > 0, the max_neighbors brightest within the fit radius + 2 FWHM but the
nearest within 1 FWHM (the source's own detection) and those within 0.5 FWHM (not separable); FWHM_IMAGE >
neighbor_extended_seeing x seeing (px): extended, not fitted, FLAG_PSF 32); without the .cat a warning and no neighbours
(psf_neighbors 'missing').
Writes <img>.fits.psf.cat (+ .info) next to the image only with a finite ZP_PSF (sub_: the targets, with DX_PSF/DY_PSF
and FLAG_PSF bit 16 of the PSF image); the header and the aperture catalog
do not change. Summary columns only then: psf_catalog, psf_source (epsf | moffat | mixed), psf_model (build_epsf
info 'psf_model', one per image: poly3 | poly2 | poly1 | grid | moffat), psf_nstars (fewest ePSF
stars of a tile; model 'poly': stars of the fit, or with no usable polynomial order the usable stars of its
grid-1 fallback), n_ref_psf, then
per target MAGSKY_PSF, MAGERR_PSF, ZP_PSF, ZPERR_PSF, UL5SKY_PSF, CHI2_PSF, FLAG_PSF, DX_PSF, DY_PSF (px; recenter
'offset': the offset applied at the target, 'each': its fitted move, NaN when not fitted; empty with recenter False;
sub_: those of the PSF image), then per image psf_image (sci | conv) and noise_ratio_psf (empty when too few empty
positions: no scaling) for difference images, else empty, then psf_neighbors (cat | missing) and NNEIGHBOR_PSF
(neighbours fitted with the target) with neighbors on, else empty (psf_catalog only once the
.psf.cat is written). A PSF failure fails the image's rows at step psf (sub_: no sci_/conv_, CONVOL00, ref_ SEEING,
background of that image), psf_refcat (no .refcat; sub_: no sci_<name>.fits.refcat, e.g. a dia before 0.12.1),
psf_epsf, psf_photometry, psf_zp (no finite ZP_PSF or n_ref_psf 0) or psf_sub (the measurement on sub_); the aperture
values stay in them.

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
from astropy.coordinates import SkyCoord
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.wcs.utils import skycoord_to_pixel
from photutils.aperture import ApertureStats, CircularAnnulus, CircularAperture, aperture_photometry

from ezphot.dataobjects import Catalog
from ezphot.imageobjects import ScienceImage
from ezphot.methods import PSFPhotometry
from ezphot.methods.psfphotometry import FLAG_RECENTER, _bad_pixels
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
# Summary columns after PSF_KEYS, with forced_photometry_psf enabled (difference images, else empty), per image: the
# image of the ePSF, ZP_PSF and recenter offset (sci | conv, _psf_image) and the PSF noise ratio (_psf_noise_ratio)
PSF_SUB_KEYS = ['psf_image', 'noise_ratio_psf']
# Summary columns after PSF_SUB_KEYS, with forced_photometry_psf enabled (neighbors on, else empty): per image the
# neighbour list (cat | missing, _neighbors), per target the neighbours fitted with it
PSF_NB_KEYS = ['psf_neighbors', 'NNEIGHBOR_PSF']
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
# forced_photometry_psf keys of the recipe's neighbour list (_neighbors), not psf_photometry arguments
PSF_NEIGHBOR_KEYS = ('neighbors', 'neighbor_extended_seeing')


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


def _psf_image(target_img: ScienceImage):
    """(image, 'sci' | 'conv', seeing) for the ePSF, ZP_PSF and recenter offset of a difference image sub_<name>, by its
    header CONVOL00 (hotpants): TEMPLATE (reference convolved; sub_ has the PSF and flux scale of the science) ->
    sci_<name>, seeing None; IMAGE (science convolved; sub_ has those of conv_) -> conv_<name> with its in-memory header
    SEEING set to that of ref_<name> (the PSF it was matched to), seeing = that value. Same folder and pixel grid as
    sub_; opened read-only, no file is written."""
    name = target_img.path.name[len('sub_'):]
    convol = str(target_img.header.get('CONVOL00', '')).strip().upper()
    if convol not in ('TEMPLATE', 'IMAGE'):
        raise ValueError(f"header CONVOL00 is {convol!r}, not TEMPLATE or IMAGE: no image for the PSF of the difference")
    path = target_img.path.with_name(('sci_' if convol == 'TEMPLATE' else 'conv_') + name)
    if not path.exists():
        raise FileNotFoundError(f"no {path.name} for the PSF of the difference image")
    img = ScienceImage(path=path, telinfo=target_img.telinfo, load=True)
    img.savedir = path.parent
    seeing = None
    if convol == 'IMAGE':
        seeing = ScienceImage(path=target_img.path.with_name('ref_' + name), telinfo=target_img.telinfo,
                              load=True).seeing
        if seeing is None:
            raise ValueError(f"no SEEING in the header of ref_{name} for the PSF of {path.name}")
        img.header['SEEING'] = seeing
    return img, ('sci' if convol == 'TEMPLATE' else 'conv'), seeing


def _neighbors(img: ScienceImage, extended_seeing: float):
    """(psf_photometry neighbour arguments, 'cat') from the SExtractor catalog of img next to it
    (img.savepath.catalogpath, <img>.fits.cat; savedir = its folder): finite 0-based X_IMAGE/Y_IMAGE of the sources with
    FLUX_AUTO > 0, neighbor_flux FLUX_AUTO, neighbor_extended FWHM_IMAGE > extended_seeing x the header seeing in
    pixels. Without the catalog ({}, 'missing') and a warning: no neighbours."""
    cat = Catalog(img.savepath.catalogpath, catalog_type='all', load=False, target_img=img)
    if not cat.is_exists:
        print(f"[WARNING] no {cat.path.name}: PSF photometry of {img.path.name} without neighbours", flush=True)
        return {}, 'missing'
    x, y, flux, fwhm = (np.ma.filled(np.ma.asarray(cat.data[k], dtype=float), np.nan)
                        for k in ('X_IMAGE', 'Y_IMAGE', 'FLUX_AUTO', 'FWHM_IMAGE'))
    keep = np.isfinite(x) & np.isfinite(y) & (flux > 0)
    seeing_pixel = img.seeing / float(np.mean(img.pixelscale))
    return dict(neighbor_x=x[keep], neighbor_y=y[keep], neighbor_flux=flux[keep],
                neighbor_extended=fwhm[keep] > extended_seeing * seeing_pixel), 'cat'


def _sub_annulus_arcsec(target_img: ScienceImage, phot: dict):
    """(inner, outer) radius (arcsec) of the local annulus of the aperture photometry of a difference image with the
    settings phot (forced_photometry + forced_photometry_sub): the rule of AperturePhotometry.circular_photometry
    (inner diameter = largest aperture + 1.5 seeing, 1.5 x largest aperture without seeing; outer = inner +
    annulus_width_arcsec); None without annulus_width_arcsec."""
    width = phot.get('annulus_width_arcsec')
    if width is None:
        return None
    seeing = target_img.seeing
    apertures = list(np.atleast_1d(phot.get('aperture_diameter_arcsec', [3, 5, 7, 10, 15])))   # its defaults
    ratios = phot.get('aperture_diameter_seeing', [2.5, 3.5])
    if ratios is not None and seeing is not None:
        apertures += [seeing * k for k in np.atleast_1d(ratios)]
    inner = max(apertures) * 1.5 if seeing is None else max(apertures) + seeing * 1.5
    return inner / 2, (inner + width) / 2


def _psf_noise_ratio(psfphot: PSFPhotometry, sub_img: ScienceImage, psf, psf_info, bkgrms, host_img, kwargs: dict,
                     n: int, seed: int = 0) -> float:
    """Measured / predicted background noise of the PSF flux on a difference image (the PSF version of
    AperturePhotometry._empty_aperture_ratios, same position rules): up to n random empty positions (fixed seed),
    PSF fit radius r (kwargs fit_radius_fwhm x FWHM capped at the PSF array radius, or bkg_radius_fwhm x FWHM, as
    psf_photometry) and the annulus (kwargs annulus_radii_arcsec, if given) inside the image; no invalid pixel
    (psfphotometry._bad_pixels: not finite or nonzero |value| < 1e-20) within r; with host_img (first 2n such
    positions) its sum within r minus its own annulus mean x area not more than 3 x sqrt(sum of bkgrms^2 within r)
    above the median of those positions. psf_photometry there (unit pixel, recenter False, noise_ratio and host_img
    None, the other kwargs as for the targets); z = FLUX_PSF / FLUXERR_BKG, the background-only error that the ratio
    scales (FLUXERR_BKG = 10**(-0.4 UL5_PSF) / 5); ratio = 4-sigma-clipped (5 iterations) std of the finite z.
    NaN with fewer than max(n // 4, 10) usable positions or finite z."""
    n_min = max(n // 4, 10)
    native = psfphot.helper.to_native
    data = native(sub_img.data)
    ny, nx = data.shape
    fwhm, scale = psfphot._fwhm_pixel(sub_img)
    if kwargs.get('bkg_radius_fwhm') is None:
        r = min(kwargs.get('fit_radius_fwhm', 2.5) * fwhm, np.asarray(psf).shape[-1] // 2)   # psf_photometry default 2.5
    else:
        r = kwargs['bkg_radius_fwhm'] * fwhm
    annulus = kwargs.get('annulus_radii_arcsec')
    annulus = None if annulus is None else (float(annulus[0]) / scale, float(annulus[1]) / scale)
    edge = int(np.ceil(max(r, annulus[1] if annulus is not None else 0))) + 1
    if nx <= 2 * edge + 1 or ny <= 2 * edge + 1:
        return np.nan
    rng = np.random.default_rng(seed)
    pos = np.column_stack([rng.uniform(edge, nx - 1 - edge, 4 * n), rng.uniform(edge, ny - 1 - edge, 4 * n)])
    bad = _bad_pixels(data)
    pos = pos[np.asarray(aperture_photometry(bad.astype(np.uint8), CircularAperture(pos, r=r),
                                             method='center')['aperture_sum']) == 0]
    if host_img is not None and len(pos):
        pos = pos[:2 * n]
        aperture = CircularAperture(pos, r=r)
        pred = np.sqrt(np.asarray(aperture_photometry(native(bkgrms.data) ** 2, aperture)['aperture_sum']))
        host = native(host_img.data)
        host_sum = np.asarray(aperture_photometry(host, aperture)['aperture_sum'])
        if annulus is not None:
            host_sum = host_sum - aperture.area * np.array(ApertureStats(
                host, CircularAnnulus(pos, r_in=annulus[0], r_out=annulus[1]),
                sigma_clip=SigmaClip(sigma=5.0, maxiters=5)).mean)
        pos = pos[host_sum - np.nanmedian(host_sum) <= 3 * pred]
    pos = pos[:n]
    if len(pos) < n_min:
        return np.nan
    cat = psfphot.psf_photometry(sub_img, pos[:, 0], pos[:, 1], psf=psf, psf_info=psf_info, unit='pixel',
                                 target_bkgrms=bkgrms, save=False, verbose=False,
                                 **dict(kwargs, noise_ratio=None, host_img=None, recenter=False)).data
    z = np.asarray(cat['FLUX_PSF'], dtype=float) / (10 ** (-0.4 * np.asarray(cat['UL5_PSF'], dtype=float)) / 5)
    z = z[np.isfinite(z)]
    return float(sigma_clipped_stats(z, sigma=4.0, maxiters=5)[2]) if len(z) >= n_min else np.nan


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
    nb_cfg = {k: psf_cfg.pop(k) for k in PSF_NEIGHBOR_KEYS if k in psf_cfg}
    row = dict(dict.fromkeys(ROW_KEYS), path=str(target_img.path), calibrate=calibrate, success=False)
    values = _empty_values(ra, psf_on)
    bkg = bkgrms = refcat = catalog = used = host_img = psf_catalog = None
    psf_img = psf_bkg = psf_bkgrms = psf_refcat = psf_catalog_img = sub_img = sub_bkgrms = None   # PSF of sub_
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
            is_sub = target_img.path.name.startswith('sub_')
            # The image of the ePSF, ZP_PSF and recenter offset: the image itself, for a difference image sci_ or conv_
            psf_img, psf_bkg, psf_bkgrms, seeing = target_img, bkg, bkgrms, None
            if is_sub:
                psf_img, psf_image, seeing = _psf_image(target_img)
                for v in values:
                    v['psf_image'] = psf_image
                psf_bkg, psf_bkgrms, _, _ = _background(psf_img)   # as for other images: sidecars, else in memory
            row['step'] = 'psf_refcat'
            if is_sub:   # the science image's ZP stars next to sci_<name> (dia copies them), also for conv_ (same grid)
                psf_refcat = Catalog(target_img.path.with_name('sci_' + target_img.path.name[len('sub_'):] + '.refcat'),
                                     catalog_type='reference', load=False, target_img=psf_img)
            else:
                if refcat is None:
                    refcat = Catalog(target_img.savepath.refcatalogpath, catalog_type='reference', load=False,
                                     target_img=target_img)
                psf_refcat = refcat
            if not psf_refcat.is_exists:
                raise FileNotFoundError(f"no reference-star catalog {psf_refcat.path} for ZP_PSF")
            psfphot = PSFPhotometry()
            row['step'] = 'psf_epsf'
            epsf_kwargs = {k: psf_cfg.pop(k) for k in PSF_EPSF_KEYS if k in psf_cfg}
            psf, psf_info = psfphot.build_epsf(psf_img, **epsf_kwargs,
                                               catalog_type=cfg['photcal']['catalog_type'],
                                               catalog_version=cfg['photcal']['catalog_version'], verbose=False)
            sources = set(np.asarray(psf_info['source']).ravel().tolist())
            psf_row = dict(psf_source=sources.pop() if len(sources) == 1 else 'mixed', psf_model=psf_info['psf_model'],
                           psf_nstars=int(np.min(psf_info['nstars'])))
            row['step'] = 'psf_photometry'   # targets first, then the reference stars of ZP_PSF
            nb_kwargs = {}   # neighbours of the image's own catalog (sub_: on sub_ below; sci_/conv_ stars without)
            if nb_cfg.get('neighbors') and not is_sub:
                nb_kwargs, nb_source = _neighbors(target_img, nb_cfg['neighbor_extended_seeing'])
                for v in values:
                    v['psf_neighbors'] = nb_source
            psf_catalog = psfphot.psf_photometry(
                psf_img, np.concatenate([ra, np.asarray(psf_refcat.data['X_WORLD'], dtype=float)]),
                np.concatenate([dec, np.asarray(psf_refcat.data['Y_WORLD'], dtype=float)]), psf=psf, psf_info=psf_info,
                unit='coord', target_bkg=psf_bkg, target_bkgrms=psf_bkgrms, save=False, verbose=False, **psf_cfg,
                **nb_kwargs)
            row['step'] = 'psf_zp'
            # ZP_PSF: the reference_stars rules (photcal section, no region cut, defaults.MAG_RANGE) on MAG_PSF
            zp_kwargs = {k: v for k, v in cfg['photcal'].items() if k not in PSF_ZP_FIXED}
            zp_kwargs.update(ra_deg=None, dec_deg=None, inner_fraction=1.0)
            zp_kwargs['mag_lower'], zp_kwargs['mag_upper'] = D.MAG_RANGE.get(target_img.filter, D.MAG_RANGE_DEFAULT)
            psf_catalog, zp_psf, zperr_psf, psf_row['n_ref_psf'] = psfphot.calibrate_zp(psf_img, psf_catalog,
                                                                                        save=False, **zp_kwargs)
            for v in values:
                v.update(psf_row)
            if not np.isfinite(zp_psf) or psf_row['n_ref_psf'] == 0:   # no .psf.cat is written then
                raise RuntimeError(f"no finite ZP_PSF: {psf_row['n_ref_psf']} reference stars")
            if is_sub:
                # The targets on sub_ with that ePSF and ZP_PSF at their positions there (sub_ WCS) plus the recenter
                # offset measured on psf_img (DX_PSF/DY_PSF: added to the input position, NaN = not moved), fixed;
                # background and annulus radii of the aperture photometry of sub_, noise ratio of empty positions,
                # host Poisson of conv_ (CONVOL00 TEMPLATE, host_poisson).
                row['step'] = 'psf_sub'
                psf_catalog_img, sci = psf_catalog, psf_catalog.data[:len(ra)]
                shift = [np.asarray(sci[k], dtype=float) if k in sci.colnames else np.full(len(ra), np.nan)
                         for k in ('DX_PSF', 'DY_PSF')]
                sub_img = target_img
                if seeing is not None:   # IMAGE: sub_ has the PSF of conv_; a separate object, target_img unchanged
                    sub_img = ScienceImage(path=target_img.path, telinfo=target_img.telinfo, load=True)
                    sub_img.header['SEEING'] = seeing
                    sub_img.savedir = target_img.savedir
                sub_kwargs = dict(psf_cfg, recenter=False)
                if sub_kwargs.get('annulus_radii_arcsec') is None:
                    sub_kwargs['annulus_radii_arcsec'] = _sub_annulus_arcsec(target_img, phot)
                if nb_cfg.get('neighbors'):   # sub_'s own catalog, also for the empty positions of the noise ratio
                    nb_kwargs, nb_source = _neighbors(sub_img, nb_cfg['neighbor_extended_seeing'])
                    sub_kwargs.update(nb_kwargs)
                    for v in values:
                        v['psf_neighbors'] = nb_source
                sub_bkgrms, _, _ = psfphot.errormap.calculate_errormap_from_image(
                    target_img=sub_img, errormap_type='bkgrms', save=False, verbose=False, visualize=False,
                    save_fig=False)
                n_empty = cfg['forced_photometry_sub'].get('n_empty_apertures')
                ratio = np.nan
                if n_empty:
                    ratio = _psf_noise_ratio(psfphot, sub_img, psf, psf_info, sub_bkgrms, host_img, sub_kwargs,
                                             int(n_empty))
                    if not np.isfinite(ratio):
                        print(f"[WARNING] too few empty PSF positions in {target_img.path.name}: "
                              "FLUXERR_PSF/UL5SKY_PSF not scaled (noise_ratio_psf empty)")
                x_t, y_t = skycoord_to_pixel(SkyCoord(ra=ra, dec=dec, unit='deg'), target_img.wcs, origin=0)
                x_t, y_t = np.atleast_1d(np.asarray(x_t, dtype=float)), np.atleast_1d(np.asarray(y_t, dtype=float))
                psf_catalog = psfphot.psf_photometry(
                    sub_img, x_t + np.nan_to_num(shift[0]), y_t + np.nan_to_num(shift[1]), psf=psf,
                    psf_info=psf_info, unit='pixel', target_bkgrms=sub_bkgrms, save=False, verbose=False,
                    **dict(sub_kwargs, zp_psf=zp_psf, zperr_psf=zperr_psf, host_img=host_img,
                           noise_ratio=ratio if np.isfinite(ratio) else None))
                data = psf_catalog.data   # as other .psf.cat: the requested positions, the applied offset, its bit 16
                data['X_IMAGE'], data['Y_IMAGE'], data['X_WORLD'], data['Y_WORLD'] = x_t, y_t, ra, dec
                flags = [np.ma.filled(np.ma.asarray(t['FLAG_PSF']), 0).astype(int) for t in (data, sci)]
                data['FLAG_PSF'] = flags[0] | (flags[1] & FLAG_RECENTER)
                if 'DX_PSF' in sci.colnames:
                    i = data.colnames.index('FLAG_PSF') + 1
                    data.add_columns(shift, names=['DX_PSF', 'DY_PSF'], indexes=[i, i])
                data.meta['comments'].append(
                    f'PSF_IMAGE = {psf_img.path.name}: ePSF, ZP_PSF and recenter offset (DX_PSF/DY_PSF, FLAG_PSF '
                    f'{FLAG_RECENTER}) from it, fixed positions here; NOISE_RATIO_PSF = {ratio:.4f} (empty positions)')
                for v in values:
                    v['noise_ratio_psf'] = ratio
            psf_catalog.write(verbose=False)
            data = psf_catalog.data
            for i, v in enumerate(values):
                v['psf_catalog'] = str(psf_catalog.path)   # only once written: empty in failed rows
                for k in PSF_KEYS + PSF_NB_KEYS[1:]:
                    if k in data.colnames:
                        v[k] = int(data[k][i]) if k in ('FLAG_PSF', 'NNEIGHBOR_PSF') else float(data[k][i])
        row['success'] = True
        row['step'] = 'done'
    except Exception as e:
        row['error'] = f"{type(e).__name__}: {e}"
    finally:
        if clear:
            for obj in (target_img, bkg, bkgrms, refcat, catalog, used, host_img, psf_catalog, psf_img, psf_bkg,
                        psf_bkgrms, psf_refcat, psf_catalog_img, sub_img, sub_bkgrms):
                if obj is not None:
                    try:
                        obj.clear(verbose=False)
                    except Exception:
                        pass
    return _target_rows(row, ra, dec, values)


def _empty_values(ra, psf: bool = False) -> List[dict]:
    psf_keys = PSF_ROW_KEYS + PSF_KEYS + PSF_SUB_KEYS + PSF_NB_KEYS if psf else []
    return [dict.fromkeys(FORCED_KEYS + NOISE_KEYS + psf_keys) for _ in ra]


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
    for k in ('psf_nstars', 'n_ref_psf', 'FLAG_PSF', 'NNEIGHBOR_PSF'):   # forced_photometry_psf enabled
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
                    help='also forced PSF photometry (config forced_photometry_psf enabled; difference images sub_: '
                         'ePSF and ZP_PSF of sci_/conv_ next to them, needs sci_<name>.fits.refcat from dia)')
    a = ap.parse_args(argv)
    config = {'forced_photometry_psf': dict(enabled=True)} if a.psf else None
    summary = forced_photometry(a.images, a.ra, a.dec, calibrate=a.calibrate, config=config, n_proc=a.n_proc,
                                out=a.out)
    return 0 if summary['success'].all() else 1


if __name__ == '__main__':
    raise SystemExit(main())
