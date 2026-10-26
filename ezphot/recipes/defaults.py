"""
Default step parameters for the recipes. Mirrors the 7DT production settings in
BRIDGE/bridge/configuration/alertprocessor.config (2026-10) so that recipes are bridge-free
but give the same products. Override per call via the `config` argument of each recipe.
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
           show_transient_numbers=100, show_candidate_numbers=100)

FORCED_PHOTOMETRY = dict(aperture_diameter_arcsec=[3, 5, 7, 10, 15], aperture_diameter_seeing=[2.5, 3.5],
                         annulus_width_arcsec=None, verbose=False, visualize=False, save=True, save_fig=True)

# 10" aperture is MAG_APER_3 in ezphot catalogs (aperture_diameter_arcsec index 3).
# Pipeline (tract7dt) uses APER_5 / ZP_5 for 10": do not mix the two conventions.
APER10_KEY = 'APER_3'

# Header keys summarized per image after process_images (QA inputs).
SUMMARY_HEADER_KEYS = ['SEEING', 'ELLIP', 'SKYVAL', 'SKYSIG',
                       f'ZP_{APER10_KEY}', f'ZPERR_{APER10_KEY}', f'UL5SKY_{APER10_KEY}']
