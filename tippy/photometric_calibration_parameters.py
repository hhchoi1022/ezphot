#!/usr/bin/env python3
"""
All parameters for TIPPhotometricCalibration.photometric_calibration function
This file defines all parameters with their default values and descriptions.
"""

# Core Parameters
target_img = None  # Union[ScienceImage, ReferenceImage, CalibrationImage] - Target image for calibration
target_catalog = None  # TIPCatalog - Target catalog with detected sources
catalog_type = 'GAIAXP'  # str - Reference catalog type ('GAIAXP', 'GAIA', 'PS1')
max_distance_second = 1.0  # float - Maximum matching distance in arcseconds
calculate_color_terms = True  # bool - Whether to calculate color terms
calculate_mag_terms = True  # bool - Whether to calculate magnitude terms

# Star Selection Parameters
mag_lower = None  # float - Lower magnitude limit
mag_upper = None  # float - Upper magnitude limit
snr_lower = 20  # float - Lower SNR limit
snr_upper = 300  # float - Upper SNR limit
classstar_lower = 0.8  # float - Minimum CLASS_STAR value
elongation_upper = 1.7  # float - Maximum elongation
elongation_sigma = 5  # float - Sigma clipping for elongation
fwhm_lower = 1  # float - Lower FWHM limit in pixels
fwhm_upper = 15  # float - Upper FWHM limit in pixels
fwhm_sigma = 5  # float - Sigma clipping for FWHM
flag_upper = 1  # int - Maximum FLAGS value
maskflag_upper = 1  # int - Maximum IMAFLAGS_ISO value
inner_fraction = 0.7  # float - Fraction of image to use (inner region)
isolation_radius = 10.0  # float - Isolation radius in pixels

# Column Name Parameters
magnitude_key = 'MAG_AUTO'  # str - Magnitude column name
flux_key = 'FLUX_AUTO'  # str - Flux column name
fluxerr_key = 'FLUXERR_AUTO'  # str - Flux error column name
fwhm_key = 'FWHM_IMAGE'  # str - FWHM column name
x_key = 'X_IMAGE'  # str - X coordinate column name
y_key = 'Y_IMAGE'  # str - Y coordinate column name
classstar_key = 'CLASS_STAR'  # str - CLASS_STAR column name
elongation_key = 'ELONGATION'  # str - Elongation column name
flag_key = 'FLAGS'  # str - FLAGS column name
maskflag_key = 'IMAFLAGS_ISO'  # str - Mask flags column name

# Output Control Parameters
save = True  # bool - Whether to save results
verbose = True  # bool - Whether to print verbose output
visualize = True  # bool - Whether to show plots
save_fig = False  # bool - Whether to save plots
save_refcat = True  # bool - Whether to save reference catalog

# Example usage with all parameters
def get_all_parameters():
    """Return a dictionary with all parameters and their default values."""
    return {
        # Core Parameters
        'target_img': target_img,
        'target_catalog': target_catalog,
        'catalog_type': catalog_type,
        'max_distance_second': max_distance_second,
        'calculate_color_terms': calculate_color_terms,
        'calculate_mag_terms': calculate_mag_terms,
        
        # Star Selection Parameters
        'mag_lower': mag_lower,
        'mag_upper': mag_upper,
        'snr_lower': snr_lower,
        'snr_upper': snr_upper,
        'classstar_lower': classstar_lower,
        'elongation_upper': elongation_upper,
        'elongation_sigma': elongation_sigma,
        'fwhm_lower': fwhm_lower,
        'fwhm_upper': fwhm_upper,
        'fwhm_sigma': fwhm_sigma,
        'flag_upper': flag_upper,
        'maskflag_upper': maskflag_upper,
        'inner_fraction': inner_fraction,
        'isolation_radius': isolation_radius,
        
        # Column Name Parameters
        'magnitude_key': magnitude_key,
        'flux_key': flux_key,
        'fluxerr_key': fluxerr_key,
        'fwhm_key': fwhm_key,
        'x_key': x_key,
        'y_key': y_key,
        'classstar_key': classstar_key,
        'elongation_key': elongation_key,
        'flag_key': flag_key,
        'maskflag_key': maskflag_key,
        
        # Output Control Parameters
        'save': save,
        'verbose': verbose,
        'visualize': visualize,
        'save_fig': save_fig,
        'save_refcat': save_refcat
    }

# Example function call with all parameters
def example_photometric_calibration_call():
    """Example of how to call photometric_calibration with all parameters."""
    from tippy.methods import TIPPhotometricCalibration
    
    # Initialize the class
    photometric_calibration = TIPPhotometricCalibration()
    
    # Get all parameters
    params = get_all_parameters()
    
    # Example call (you need to provide actual target_img and target_catalog)
    # result_img, result_catalog, filtered_catalog = photometric_calibration.photometric_calibration(
    #     target_img=params['target_img'],
    #     target_catalog=params['target_catalog'],
    #     catalog_type=params['catalog_type'],
    #     max_distance_second=params['max_distance_second'],
    #     calculate_color_terms=params['calculate_color_terms'],
    #     calculate_mag_terms=params['calculate_mag_terms'],
    #     
    #     # Star Selection Parameters
    #     mag_lower=params['mag_lower'],
    #     mag_upper=params['mag_upper'],
    #     snr_lower=params['snr_lower'],
    #     snr_upper=params['snr_upper'],
    #     classstar_lower=params['classstar_lower'],
    #     elongation_upper=params['elongation_upper'],
    #     elongation_sigma=params['elongation_sigma'],
    #     fwhm_lower=params['fwhm_lower'],
    #     fwhm_upper=params['fwhm_upper'],
    #     fwhm_sigma=params['fwhm_sigma'],
    #     flag_upper=params['flag_upper'],
    #     maskflag_upper=params['maskflag_upper'],
    #     inner_fraction=params['inner_fraction'],
    #     isolation_radius=params['isolation_radius'],
    #     
    #     # Column Name Parameters
    #     magnitude_key=params['magnitude_key'],
    #     flux_key=params['flux_key'],
    #     fluxerr_key=params['fluxerr_key'],
    #     fwhm_key=params['fwhm_key'],
    #     x_key=params['x_key'],
    #     y_key=params['y_key'],
    #     classstar_key=params['classstar_key'],
    #     elongation_key=params['elongation_key'],
    #     flag_key=params['flag_key'],
    #     maskflag_key=params['maskflag_key'],
    #     
    #     # Output Control Parameters
    #     save=params['save'],
    #     verbose=params['verbose'],
    #     visualize=params['visualize'],
    #     save_fig=params['save_fig'],
    #     save_refcat=params['save_refcat']
    # )
    
    print("All parameters defined. Use get_all_parameters() to get the parameter dictionary.")
    return params

if __name__ == "__main__":
    # Print all parameters
    params = get_all_parameters()
    print("All parameters for photometric_calibration function:")
    print("=" * 60)
    for key, value in params.items():
        print(f"{key}: {value}")
    print("=" * 60)
    print(f"Total parameters: {len(params)}") 