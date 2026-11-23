"""
Default step parameters for the recipes when ezphot is used on its own. The values started from the 7DT
production settings of 2026-10 and are not kept in sync with any other pipeline's configuration; a caller
with its own settings passes them per call via the `config` argument of each recipe.
"""

CIRCULARMASK = dict(radius_arcsec=150)

INVALIDMASK = dict(threshold_invalid_connection=5.0, verbose=False, visualize=False, save=True, save_fig=False)

SOURCEMASK = dict(sigma=5.0, mask_radius_factor=3, saturation_level=50000,
                  verbose=False, visualize=False, save=True, save_fig=True)

BKGMAP = dict(is_2D_bkg=True, box_size=128, filter_size=3, correct_global_offset=True,
              verbose=False, visualize=False, save=True, save_fig=True)

BKGRMS = dict(box_size=128, filter_size=3, verbose=False, visualize=False, save=True, save_fig=True)

PHOTOMETRY = dict(detection_sigma=1.5, aperture_diameter_arcsec=[3, 5, 7, 10, 15],
                  aperture_diameter_seeing=[2.5, 3.5],
                  annulus_width_arcsec=None,   # SExtractor LOCAL background disabled (mode bias); bridge also null since 2026-10-24
                  saturation_level=60000, kron_factor=2.5,
                  verbose=False, visualize=False, save=True, save_fig=True)

# Reference-star magnitude range per filter (mag_lower, mag_upper); default for others.
MAG_RANGE_DEFAULT = (11.5, 15)
MAG_RANGE = {
    'g': (13, 16.5), 'r': (12.5, 16), 'i': (11, 14.5), 'z': (10, 13.5),
    'm375w': (10, 13.5), 'm425w': (12, 15.5), 'm466w': (12, 15.5), 'm692w': (12, 15.5),
    'm710w': (11, 14.5), 'm769w': (11, 14.5), 'm832w': (10, 13.5),
    'm386': (10, 13.5), 'm400': (11, 14.5),
    'm425': (11.5, 15), 'm438': (11.5, 15), 'm450': (11.5, 15), 'm475': (11.5, 15),
    'm483': (11.5, 15), 'm500': (11.5, 15), 'm512': (11.5, 15), 'm525': (11.5, 15),
    'm534': (11, 14.5), 'm550': (11, 14.5), 'm561': (11, 14.5), 'm575': (11, 14.5),
    'm586': (11, 14.5), 'm600': (11, 14.5), 'm615': (11, 14.5), 'm625': (11, 14.5),
    'm640': (11, 14.5), 'm650': (11, 14.5), 'm661': (11, 14.5), 'm675': (11, 14.5),
    'm700': (11, 14.5), 'm725': (10.5, 14), 'm750': (10.5, 14), 'm775': (10.5, 14),
    'm800': (10, 13.5), 'm825': (10, 13.5), 'm850': (10, 13.5), 'm875': (10, 13.5),
}

PHOTCAL = dict(catalog_type='GAIAXP_CORR_LAMOST', catalog_version='v1',
               max_distance_second=5, min_number_of_sources=10,
               calculate_color_terms=True, calculate_mag_terms=True,
               dynamic_mag_range=False, classstar_lower=0.5,
               elongation_upper=1.7, elongation_sigma=5,
               fwhm_lower_arcsec=1, fwhm_upper_arcsec=15, fwhm_sigma=5,
               flag_upper=1, maskflag_upper=1, radius_arcsec=1500,
               inner_fraction=0.8, isolation_radius_arcsec=10.0,
               magnitude_key='MAG_AUTO', magnitudeerr_key='MAGERR_AUTO',
               fwhm_key='FWHM_WORLD', ra_key='X_WORLD', dec_key='Y_WORLD',
               classstar_key='CLASS_STAR', elongation_key='ELONGATION',
               flag_key='FLAGS', maskflag_key='IMAFLAGS_ISO',
               verbose=False, visualize=False, save=True, save_fig=True, save_refcat=True)

# Stacking (bridge AlertProcessor.stacking + stack_prepare/stack). Images are grouped per night.
STACK_GROUP = dict(by_filter=True, by_exptime=False, by_objname=False, by_telname=False,
                   by_observatory=True, by_obsdate=True, obsdate_delta=0.5, obsdate_key='obsdate')

STACK_PREPARE = dict(scale=True, zp_key='ZP_APER_3', n_proc=1,
                     reproject=True, reproject_type='LANCZOS3', center_ra=None, center_dec=None,
                     pixel_scale=None, x_size=None, y_size=None,
                     convolve=False, seeing_key='SEEING', verbose=False, save=True, clear=True)

STACK = dict(n_proc=1, combine_type='median', clip_type='extrema', sigma=3.0, nlow=1, nhigh=1,
             verbose=False, save=True, remove_intermediate=True)

# Difference imaging (bridge DIA) and forced photometry at the target on the difference image.
DIA = dict(reject_variable_sources=True, negative_detection=True, reverse_subtraction=False,
           convim='auto', normim='i', tu=None, tl=None, iu=50000, il=-10000,
           nrx=1, nry=1, nsx=10, nsy=10, ko=2, bgo=1, r=10,
           verbose=False, visualize=False, save=True,
           save_transient_figure=True, save_candidate_figure=True,
           show_transient_numbers=100, show_candidate_numbers=100,
           # Kept out of the hotpants stamps: (ra, dec) deg positions besides the dia --ra/--dec target, radius arcsec
           exclude_coords=None, exclude_radius_arcsec=10.0)

FORCED_PHOTOMETRY = dict(aperture_diameter_arcsec=[3, 5, 7, 10, 15], aperture_diameter_seeing=[2.5, 3.5],
                         annulus_width_arcsec=None, verbose=False, visualize=False, save=True, save_fig=True)

# Forced photometry on difference images (sub_*; dia and forced_photometry), on top of FORCED_PHOTOMETRY:
# - local background: 5-sigma-clipped annulus mean, inner diameter = largest aperture + 1.5 seeing, outer = inner + 60"
#   (radii ~9-12" to ~39-42" at seeing 2.5-4.9"), used for the flux only (annulus_errors=False);
# - SKYSIG/UL3/UL5 and the background part of FLUXERR x the measured/predicted noise of 500 random empty apertures;
# - host_poisson: Poisson noise of the convolved reference (conv_, CONVOL00 TEMPLATE) added to FLUXERR/MAGERR only.
FORCED_PHOTOMETRY_SUB = dict(annulus_width_arcsec=60, annulus_errors=False, n_empty_apertures=500, host_poisson=True)

# Forced PSF photometry next to the aperture one (forced_photometry; off by default; difference images sub_: ePSF, ZP_PSF
# and recenter offset from sci_ (CONVOL00 TEMPLATE) or conv_ (IMAGE), then the targets on sub_ at fixed positions with
# the FORCED_PHOTOMETRY_SUB annulus radii, PSF noise ratio of n_empty_apertures empty positions and host Poisson):
# ePSF of the image (PSFPhotometry.build_epsf: grid N or (nx, ny) tiles, model 'grid' per tile or 'poly' polynomial of
# order <= poly_order in the position; max_stars_total, snr_lower, snr_lower_fallback, recenter_iter may be added),
# PSF amplitude fit (psf_photometry: background, fit radius fit_radius_fwhm x FWHM, weighting, recenter) and ZP_PSF from
# the image's reference stars (calibrate_zp). background 'annulus' (default) | 'fit' | 'plane' | 'none'; with 'fit' or
# 'plane' bkg_radius_fwhm may be added (fit-pixel radius in FWHM, also beyond the PSF array); targets and reference
# stars always share the mode.
# Default: 'poly' of order 3, evaluated on the automatic ~512 px tiling (grid 1); order ladder of build_epsf: order 3
# (>= 50 stars) needs a star in each of the 3 x 3 image zones, orders 2 (>= 30) and 1 (>= 15) in each of 2 x 2 zones,
# else one median ePSF (grid 1; Moffat below 20 stars); fewer than 50 stars at peak S/N 50: selection repeated at 30.
# recenter='offset': one smooth offset of the PSF centres from the measured positions of the reference stars
# (.refcat: positions measured on the image; PSF position fits of the sources at S/N >= 50) - a difference of centre
# definition or coordinate round trip - added to every target and reference-star position alike (no S/N step between
# sources). A WCS error of catalog-coordinate targets is in the reference-star positions too and is not corrected.
# neighbors (off by default): the sources of the measured image's SExtractor catalog <img>.fits.cat (sub_: its own) are
# fitted together with each target and reference star (psf_photometry neighbor_x/y, fixed 0-based X_IMAGE/Y_IMAGE,
# FLUX_AUTO > 0, brightest max_neighbors (default 6) within the fit radius + 2 FWHM but the nearest within 1 FWHM, the
# target's own detection, and those within 0.5 FWHM, not separable); extended ones (FWHM_IMAGE >
# neighbor_extended_seeing x the header seeing in pixels) are not fitted, only flagged (FLAG_PSF 32).
FORCED_PHOTOMETRY_PSF = dict(enabled=False, grid=1, model='poly', poly_order=3, fit_radius_fwhm=2.5,
                             background='annulus', weighting='uniform', recenter='offset',
                             neighbors=False, neighbor_extended_seeing=1.5)

# dia normalisation QA: norm_flag when |median 10" difference/science flux ratio of the field stars| exceeds this.
DIA_NORM_RESID_MAX = 0.035

# 10" aperture is MAG_APER_3 in ezphot catalogs (aperture_diameter_arcsec index 3).
# Pipeline (tract7dt) uses APER_5 / ZP_5 for 10": do not mix the two conventions.
APER10_KEY = 'APER_3'

# Header keys summarized per image after process_images (QA inputs).
# K_COLOR_*_g-r: photcal's slope of the 10" ZP residual vs g-r colour (qa color_max).
SUMMARY_HEADER_KEYS = ['SEEING', 'ELLIP', 'SKYVAL', 'SKYSIG',
                       f'ZP_{APER10_KEY}', f'ZPERR_{APER10_KEY}', f'UL5SKY_{APER10_KEY}',
                       f'K_COLOR_{APER10_KEY}_g-r']
