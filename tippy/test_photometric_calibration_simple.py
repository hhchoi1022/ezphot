#!/usr/bin/env python3
"""
Simple test script for TIPPhotometricCalibration.photometric_calibration function
This script tests individual parameters and provides examples for different use cases.
"""

import numpy as np
from tippy.methods import TIPPhotometricCalibration
from tippy.helper import TIPDataBrowser

def test_photometric_calibration_parameters():
    """
    Test the photometric_calibration function with different parameter combinations.
    """
    
    # Initialize the photometric calibration class
    photometric_calibration = TIPPhotometricCalibration()
    
    # Initialize data browser
    databrowser = TIPDataBrowser('scidata')
    databrowser.observatory = '7DT'
    databrowser.objname = 'T22956'
    
    # Search for test data
    target_imglist = databrowser.search(
        pattern='calib*20250424_032003*100.fits', 
        return_type='science'
    )
    target_cataloglist = databrowser.search(
        pattern='calib*20250424_032003*100.fits.cat', 
        return_type='catalog'
    )
    
    if len(target_imglist) == 0 or len(target_cataloglist) == 0:
        print("No test data found. Please check the data path.")
        return
    
    target_img = target_imglist[0]
    target_catalog = target_cataloglist[0]
    
    print("=" * 60)
    print("PHOTOMETRIC CALIBRATION PARAMETER TESTS")
    print("=" * 60)
    
    # Test 1: Basic calibration with default parameters
    print("\n1. Testing basic calibration with default parameters...")
    try:
        result_img, result_catalog, filtered_catalog = photometric_calibration.photometric_calibration(
            target_img=target_img,
            target_catalog=target_catalog,
            catalog_type='GAIAXP',
            max_distance_second=1.0,
            calculate_color_terms=True,
            calculate_mag_terms=True,
            save=True,
            verbose=True,
            visualize=False,
            save_fig=False,
            save_refcat=True
        )
        print("✓ Basic calibration successful")
    except Exception as e:
        print(f"✗ Basic calibration failed: {e}")
    
    # Test 2: Different catalog types
    print("\n2. Testing different catalog types...")
    catalog_types = ['GAIAXP', 'GAIA', 'PS1']
    for catalog_type in catalog_types:
        try:
            result_img, result_catalog, filtered_catalog = photometric_calibration.photometric_calibration(
                target_img=target_img,
                target_catalog=target_catalog,
                catalog_type=catalog_type,
                max_distance_second=1.0,
                calculate_color_terms=True,
                calculate_mag_terms=True,
                save=True,
                verbose=False,
                visualize=False,
                save_fig=False,
                save_refcat=True
            )
            print(f"✓ Catalog type '{catalog_type}' successful")
        except Exception as e:
            print(f"✗ Catalog type '{catalog_type}' failed: {e}")
    
    # Test 3: Different matching distances
    print("\n3. Testing different matching distances...")
    distances = [0.5, 1.0, 2.0, 5.0]
    for distance in distances:
        try:
            result_img, result_catalog, filtered_catalog = photometric_calibration.photometric_calibration(
                target_img=target_img,
                target_catalog=target_catalog,
                catalog_type='GAIAXP',
                max_distance_second=distance,
                calculate_color_terms=True,
                calculate_mag_terms=True,
                save=True,
                verbose=False,
                visualize=False,
                save_fig=False,
                save_refcat=True
            )
            print(f"✓ Distance {distance} arcsec successful")
        except Exception as e:
            print(f"✗ Distance {distance} arcsec failed: {e}")
    
    # Test 4: Different magnitude ranges
    print("\n4. Testing different magnitude ranges...")
    mag_ranges = [
        {'mag_lower': 12.0, 'mag_upper': 18.0},
        {'mag_lower': 14.0, 'mag_upper': 16.0},
        {'mag_lower': None, 'mag_upper': None}
    ]
    for i, mag_range in enumerate(mag_ranges):
        try:
            result_img, result_catalog, filtered_catalog = photometric_calibration.photometric_calibration(
                target_img=target_img,
                target_catalog=target_catalog,
                catalog_type='GAIAXP',
                max_distance_second=1.0,
                calculate_color_terms=True,
                calculate_mag_terms=True,
                **mag_range,
                save=True,
                verbose=False,
                visualize=False,
                save_fig=False,
                save_refcat=True
            )
            print(f"✓ Magnitude range {i+1}: {mag_range} successful")
        except Exception as e:
            print(f"✗ Magnitude range {i+1}: {mag_range} failed: {e}")
    
    # Test 5: Different SNR ranges
    print("\n5. Testing different SNR ranges...")
    snr_ranges = [
        {'snr_lower': 20, 'snr_upper': 300},
        {'snr_lower': 50, 'snr_upper': 200},
        {'snr_lower': 10, 'snr_upper': 500}
    ]
    for i, snr_range in enumerate(snr_ranges):
        try:
            result_img, result_catalog, filtered_catalog = photometric_calibration.photometric_calibration(
                target_img=target_img,
                target_catalog=target_catalog,
                catalog_type='GAIAXP',
                max_distance_second=1.0,
                calculate_color_terms=True,
                calculate_mag_terms=True,
                **snr_range,
                save=True,
                verbose=False,
                visualize=False,
                save_fig=False,
                save_refcat=True
            )
            print(f"✓ SNR range {i+1}: {snr_range} successful")
        except Exception as e:
            print(f"✗ SNR range {i+1}: {snr_range} failed: {e}")
    
    # Test 6: Different FWHM ranges
    print("\n6. Testing different FWHM ranges...")
    fwhm_ranges = [
        {'fwhm_lower': 1, 'fwhm_upper': 15},
        {'fwhm_lower': 2, 'fwhm_upper': 10},
        {'fwhm_lower': 3, 'fwhm_upper': 8}
    ]
    for i, fwhm_range in enumerate(fwhm_ranges):
        try:
            result_img, result_catalog, filtered_catalog = photometric_calibration.photometric_calibration(
                target_img=target_img,
                target_catalog=target_catalog,
                catalog_type='GAIAXP',
                max_distance_second=1.0,
                calculate_color_terms=True,
                calculate_mag_terms=True,
                **fwhm_range,
                save=True,
                verbose=False,
                visualize=False,
                save_fig=False,
                save_refcat=True
            )
            print(f"✓ FWHM range {i+1}: {fwhm_range} successful")
        except Exception as e:
            print(f"✗ FWHM range {i+1}: {fwhm_range} failed: {e}")
    
    # Test 7: Different calculation options
    print("\n7. Testing different calculation options...")
    calculation_options = [
        {'calculate_color_terms': True, 'calculate_mag_terms': True},
        {'calculate_color_terms': True, 'calculate_mag_terms': False},
        {'calculate_color_terms': False, 'calculate_mag_terms': True},
        {'calculate_color_terms': False, 'calculate_mag_terms': False}
    ]
    for i, options in enumerate(calculation_options):
        try:
            result_img, result_catalog, filtered_catalog = photometric_calibration.photometric_calibration(
                target_img=target_img,
                target_catalog=target_catalog,
                catalog_type='GAIAXP',
                max_distance_second=1.0,
                **options,
                save=True,
                verbose=False,
                visualize=False,
                save_fig=False,
                save_refcat=True
            )
            print(f"✓ Calculation options {i+1}: {options} successful")
        except Exception as e:
            print(f"✗ Calculation options {i+1}: {options} failed: {e}")
    
    # Test 8: Different star selection parameters
    print("\n8. Testing different star selection parameters...")
    selection_params = [
        {
            'classstar_lower': 0.8,
            'elongation_upper': 1.7,
            'elongation_sigma': 5,
            'fwhm_sigma': 5,
            'flag_upper': 1,
            'maskflag_upper': 1,
            'inner_fraction': 0.7,
            'isolation_radius': 10.0
        },
        {
            'classstar_lower': 0.9,
            'elongation_upper': 1.5,
            'elongation_sigma': 3,
            'fwhm_sigma': 3,
            'flag_upper': 0,
            'maskflag_upper': 0,
            'inner_fraction': 0.5,
            'isolation_radius': 15.0
        }
    ]
    for i, params in enumerate(selection_params):
        try:
            result_img, result_catalog, filtered_catalog = photometric_calibration.photometric_calibration(
                target_img=target_img,
                target_catalog=target_catalog,
                catalog_type='GAIAXP',
                max_distance_second=1.0,
                calculate_color_terms=True,
                calculate_mag_terms=True,
                **params,
                save=True,
                verbose=False,
                visualize=False,
                save_fig=False,
                save_refcat=True
            )
            print(f"✓ Selection parameters {i+1} successful")
        except Exception as e:
            print(f"✗ Selection parameters {i+1} failed: {e}")
    
    # Test 9: Different column names
    print("\n9. Testing different column names...")
    column_configs = [
        {
            'magnitude_key': 'MAG_AUTO',
            'flux_key': 'FLUX_AUTO',
            'fluxerr_key': 'FLUXERR_AUTO',
            'fwhm_key': 'FWHM_IMAGE',
            'x_key': 'X_IMAGE',
            'y_key': 'Y_IMAGE',
            'classstar_key': 'CLASS_STAR',
            'elongation_key': 'ELONGATION',
            'flag_key': 'FLAGS',
            'maskflag_key': 'IMAFLAGS_ISO'
        },
        {
            'magnitude_key': 'MAG_APER',
            'flux_key': 'FLUX_APER',
            'fluxerr_key': 'FLUXERR_APER',
            'fwhm_key': 'FWHM_IMAGE',
            'x_key': 'X_IMAGE',
            'y_key': 'Y_IMAGE',
            'classstar_key': 'CLASS_STAR',
            'elongation_key': 'ELONGATION',
            'flag_key': 'FLAGS',
            'maskflag_key': 'IMAFLAGS_ISO'
        }
    ]
    for i, config in enumerate(column_configs):
        try:
            result_img, result_catalog, filtered_catalog = photometric_calibration.photometric_calibration(
                target_img=target_img,
                target_catalog=target_catalog,
                catalog_type='GAIAXP',
                max_distance_second=1.0,
                calculate_color_terms=True,
                calculate_mag_terms=True,
                **config,
                save=True,
                verbose=False,
                visualize=False,
                save_fig=False,
                save_refcat=True
            )
            print(f"✓ Column configuration {i+1} successful")
        except Exception as e:
            print(f"✗ Column configuration {i+1} failed: {e}")
    
    # Test 10: Different output options
    print("\n10. Testing different output options...")
    output_configs = [
        {'save': True, 'verbose': True, 'visualize': False, 'save_fig': False, 'save_refcat': True},
        {'save': True, 'verbose': False, 'visualize': False, 'save_fig': False, 'save_refcat': False},
        {'save': False, 'verbose': True, 'visualize': False, 'save_fig': False, 'save_refcat': False},
        {'save': True, 'verbose': False, 'visualize': False, 'save_fig': True, 'save_refcat': True}
    ]
    for i, config in enumerate(output_configs):
        try:
            result_img, result_catalog, filtered_catalog = photometric_calibration.photometric_calibration(
                target_img=target_img,
                target_catalog=target_catalog,
                catalog_type='GAIAXP',
                max_distance_second=1.0,
                calculate_color_terms=True,
                calculate_mag_terms=True,
                **config
            )
            print(f"✓ Output configuration {i+1} successful")
        except Exception as e:
            print(f"✗ Output configuration {i+1} failed: {e}")
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)

def main():
    """Main function to run the parameter tests."""
    test_photometric_calibration_parameters()

if __name__ == "__main__":
    main() 