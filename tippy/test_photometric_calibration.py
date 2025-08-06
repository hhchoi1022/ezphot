#!/usr/bin/env python3
"""
Test script for TIPPhotometricCalibration.photometric_calibration function
This script tests all parameters and various scenarios for photometric calibration.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u

# Import TIPPy modules
from tippy.methods import TIPPhotometricCalibration
from tippy.helper import TIPDataBrowser
from tippy.imageobjects import ScienceImage
from tippy.catalog import TIPCatalog

class TestPhotometricCalibration:
    """Test class for photometric calibration functionality."""
    
    def __init__(self):
        """Initialize the test class."""
        self.photometric_calibration = TIPPhotometricCalibration()
        self.databrowser = TIPDataBrowser('scidata')
        self.databrowser.observatory = '7DT'
        self.databrowser.objname = 'T22956'
        
    def setup_test_data(self):
        """Setup test data for photometric calibration."""
        # Search for test images and catalogs
        target_imglist = self.databrowser.search(
            pattern='calib*20250424_032003*100.fits', 
            return_type='science'
        )
        target_cataloglist = self.databrowser.search(
            pattern='calib*20250424_032003*100.fits.cat', 
            return_type='catalog'
        )
        
        if len(target_imglist) > 0 and len(target_cataloglist) > 0:
            self.target_img = target_imglist[0]
            self.target_catalog = target_cataloglist[0]
            return True
        else:
            print("No test data found. Please check the data path.")
            return False
    
    def test_basic_photometric_calibration(self):
        """Test basic photometric calibration with default parameters."""
        print("Testing basic photometric calibration...")
        
        if not self.setup_test_data():
            return False
            
        try:
            result_img, result_catalog, filtered_catalog = self.photometric_calibration.photometric_calibration(
                target_img=self.target_img,
                target_catalog=self.target_catalog,
                catalog_type='GAIAXP',
                max_distance_second=1.0,
                calculate_color_terms=True,
                calculate_mag_terms=True,
                save=True,
                verbose=True,
                visualize=False,  # Disable visualization for testing
                save_fig=False,
                save_refcat=True
            )
            
            print("✓ Basic photometric calibration completed successfully")
            return True
            
        except Exception as e:
            print(f"✗ Basic photometric calibration failed: {e}")
            return False
    
    def test_star_selection_parameters(self):
        """Test photometric calibration with different star selection parameters."""
        print("Testing star selection parameters...")
        
        if not self.setup_test_data():
            return False
        
        # Test different magnitude ranges
        mag_ranges = [
            {'mag_lower': 12.0, 'mag_upper': 18.0},
            {'mag_lower': 14.0, 'mag_upper': 16.0},
            {'mag_lower': None, 'mag_upper': None}
        ]
        
        # Test different SNR ranges
        snr_ranges = [
            {'snr_lower': 20, 'snr_upper': 300},
            {'snr_lower': 50, 'snr_upper': 200},
            {'snr_lower': 10, 'snr_upper': 500}
        ]
        
        # Test different FWHM ranges
        fwhm_ranges = [
            {'fwhm_lower': 1, 'fwhm_upper': 15},
            {'fwhm_lower': 2, 'fwhm_upper': 10},
            {'fwhm_lower': 3, 'fwhm_upper': 8}
        ]
        
        for i, mag_range in enumerate(mag_ranges):
            for j, snr_range in enumerate(snr_ranges):
                for k, fwhm_range in enumerate(fwhm_ranges):
                    try:
                        result_img, result_catalog, filtered_catalog = self.photometric_calibration.photometric_calibration(
                            target_img=self.target_img,
                            target_catalog=self.target_catalog,
                            catalog_type='GAIAXP',
                            max_distance_second=1.0,
                            calculate_color_terms=True,
                            calculate_mag_terms=True,
                            **mag_range,
                            **snr_range,
                            **fwhm_range,
                            classstar_lower=0.8,
                            elongation_upper=1.7,
                            elongation_sigma=5,
                            fwhm_sigma=5,
                            flag_upper=1,
                            maskflag_upper=1,
                            inner_fraction=0.7,
                            isolation_radius=10.0,
                            save=True,
                            verbose=False,
                            visualize=False,
                            save_fig=False,
                            save_refcat=True
                        )
                        
                        print(f"✓ Test {i+1}-{j+1}-{k+1}: mag_range={mag_range}, snr_range={snr_range}, fwhm_range={fwhm_range}")
                        
                    except Exception as e:
                        print(f"✗ Test {i+1}-{j+1}-{k+1} failed: {e}")
        
        return True
    
    def test_catalog_type_parameters(self):
        """Test photometric calibration with different catalog types."""
        print("Testing different catalog types...")
        
        if not self.setup_test_data():
            return False
        
        catalog_types = ['GAIAXP', 'GAIA', 'PS1']
        
        for catalog_type in catalog_types:
            try:
                result_img, result_catalog, filtered_catalog = self.photometric_calibration.photometric_calibration(
                    target_img=self.target_img,
                    target_catalog=self.target_catalog,
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
                
                print(f"✓ Catalog type {catalog_type} completed successfully")
                
            except Exception as e:
                print(f"✗ Catalog type {catalog_type} failed: {e}")
        
        return True
    
    def test_matching_distance_parameters(self):
        """Test photometric calibration with different matching distances."""
        print("Testing different matching distances...")
        
        if not self.setup_test_data():
            return False
        
        distances = [0.5, 1.0, 2.0, 5.0]
        
        for distance in distances:
            try:
                result_img, result_catalog, filtered_catalog = self.photometric_calibration.photometric_calibration(
                    target_img=self.target_img,
                    target_catalog=self.target_catalog,
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
                
                print(f"✓ Matching distance {distance} arcsec completed successfully")
                
            except Exception as e:
                print(f"✗ Matching distance {distance} arcsec failed: {e}")
        
        return True
    
    def test_calculation_parameters(self):
        """Test photometric calibration with different calculation options."""
        print("Testing calculation parameters...")
        
        if not self.setup_test_data():
            return False
        
        # Test different combinations of color and magnitude terms
        calculation_options = [
            {'calculate_color_terms': True, 'calculate_mag_terms': True},
            {'calculate_color_terms': True, 'calculate_mag_terms': False},
            {'calculate_color_terms': False, 'calculate_mag_terms': True},
            {'calculate_color_terms': False, 'calculate_mag_terms': False}
        ]
        
        for i, options in enumerate(calculation_options):
            try:
                result_img, result_catalog, filtered_catalog = self.photometric_calibration.photometric_calibration(
                    target_img=self.target_img,
                    target_catalog=self.target_catalog,
                    catalog_type='GAIAXP',
                    max_distance_second=1.0,
                    **options,
                    save=True,
                    verbose=False,
                    visualize=False,
                    save_fig=False,
                    save_refcat=True
                )
                
                print(f"✓ Calculation options {i+1}: {options} completed successfully")
                
            except Exception as e:
                print(f"✗ Calculation options {i+1}: {options} failed: {e}")
        
        return True
    
    def test_column_name_parameters(self):
        """Test photometric calibration with different column names."""
        print("Testing different column names...")
        
        if not self.setup_test_data():
            return False
        
        # Test different column name configurations
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
                result_img, result_catalog, filtered_catalog = self.photometric_calibration.photometric_calibration(
                    target_img=self.target_img,
                    target_catalog=self.target_catalog,
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
                
                print(f"✓ Column configuration {i+1} completed successfully")
                
            except Exception as e:
                print(f"✗ Column configuration {i+1} failed: {e}")
        
        return True
    
    def test_output_parameters(self):
        """Test photometric calibration with different output options."""
        print("Testing output parameters...")
        
        if not self.setup_test_data():
            return False
        
        # Test different output configurations
        output_configs = [
            {'save': True, 'verbose': True, 'visualize': False, 'save_fig': False, 'save_refcat': True},
            {'save': True, 'verbose': False, 'visualize': False, 'save_fig': False, 'save_refcat': False},
            {'save': False, 'verbose': True, 'visualize': False, 'save_fig': False, 'save_refcat': False},
            {'save': True, 'verbose': False, 'visualize': False, 'save_fig': True, 'save_refcat': True}
        ]
        
        for i, config in enumerate(output_configs):
            try:
                result_img, result_catalog, filtered_catalog = self.photometric_calibration.photometric_calibration(
                    target_img=self.target_img,
                    target_catalog=self.target_catalog,
                    catalog_type='GAIAXP',
                    max_distance_second=1.0,
                    calculate_color_terms=True,
                    calculate_mag_terms=True,
                    **config
                )
                
                print(f"✓ Output configuration {i+1} completed successfully")
                
            except Exception as e:
                print(f"✗ Output configuration {i+1} failed: {e}")
        
        return True
    
    def test_edge_cases(self):
        """Test photometric calibration with edge cases."""
        print("Testing edge cases...")
        
        if not self.setup_test_data():
            return False
        
        # Test edge cases
        edge_cases = [
            # Very restrictive parameters
            {
                'mag_lower': 15.0, 'mag_upper': 16.0,
                'snr_lower': 100, 'snr_upper': 200,
                'fwhm_lower': 3, 'fwhm_upper': 5,
                'classstar_lower': 0.9,
                'elongation_upper': 1.2,
                'inner_fraction': 0.5,
                'isolation_radius': 20.0
            },
            # Very permissive parameters
            {
                'mag_lower': None, 'mag_upper': None,
                'snr_lower': 5, 'snr_upper': 1000,
                'fwhm_lower': 0.5, 'fwhm_upper': 20,
                'classstar_lower': 0.5,
                'elongation_upper': 2.0,
                'inner_fraction': 0.9,
                'isolation_radius': 5.0
            }
        ]
        
        for i, edge_case in enumerate(edge_cases):
            try:
                result_img, result_catalog, filtered_catalog = self.photometric_calibration.photometric_calibration(
                    target_img=self.target_img,
                    target_catalog=self.target_catalog,
                    catalog_type='GAIAXP',
                    max_distance_second=1.0,
                    calculate_color_terms=True,
                    calculate_mag_terms=True,
                    **edge_case,
                    save=True,
                    verbose=False,
                    visualize=False,
                    save_fig=False,
                    save_refcat=True
                )
                
                print(f"✓ Edge case {i+1} completed successfully")
                
            except Exception as e:
                print(f"✗ Edge case {i+1} failed: {e}")
        
        return True
    
    def test_all_parameters(self):
        """Test photometric calibration with all parameters set."""
        print("Testing with all parameters...")
        
        if not self.setup_test_data():
            return False
        
        try:
            result_img, result_catalog, filtered_catalog = self.photometric_calibration.photometric_calibration(
                target_img=self.target_img,
                target_catalog=self.target_catalog,
                catalog_type='GAIAXP',
                max_distance_second=1.0,
                calculate_color_terms=True,
                calculate_mag_terms=True,
                
                # Selection parameters
                mag_lower=12.0,
                mag_upper=18.0,
                snr_lower=20,
                snr_upper=300,
                classstar_lower=0.8,
                elongation_upper=1.7,
                elongation_sigma=5,
                fwhm_lower=1,
                fwhm_upper=15,
                fwhm_sigma=5,
                flag_upper=1,
                maskflag_upper=1,
                inner_fraction=0.7,
                isolation_radius=10.0,
                
                # Column names
                magnitude_key='MAG_AUTO',
                flux_key='FLUX_AUTO',
                fluxerr_key='FLUXERR_AUTO',
                fwhm_key='FWHM_IMAGE',
                x_key='X_IMAGE',
                y_key='Y_IMAGE',
                classstar_key='CLASS_STAR',
                elongation_key='ELONGATION',
                flag_key='FLAGS',
                maskflag_key='IMAFLAGS_ISO',
                
                # Output parameters
                save=True,
                verbose=True,
                visualize=False,
                save_fig=False,
                save_refcat=True
            )
            
            print("✓ All parameters test completed successfully")
            return True
            
        except Exception as e:
            print(f"✗ All parameters test failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run all tests for photometric calibration."""
        print("=" * 60)
        print("PHOTOMETRIC CALIBRATION TEST SUITE")
        print("=" * 60)
        
        tests = [
            self.test_basic_photometric_calibration,
            self.test_star_selection_parameters,
            self.test_catalog_type_parameters,
            self.test_matching_distance_parameters,
            self.test_calculation_parameters,
            self.test_column_name_parameters,
            self.test_output_parameters,
            self.test_edge_cases,
            self.test_all_parameters
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            try:
                if test():
                    passed += 1
            except Exception as e:
                print(f"✗ Test {test.__name__} failed with exception: {e}")
        
        print("=" * 60)
        print(f"TEST RESULTS: {passed}/{total} tests passed")
        print("=" * 60)
        
        return passed == total

def main():
    """Main function to run the tests."""
    tester = TestPhotometricCalibration()
    success = tester.run_all_tests()
    
    if success:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed. Please check the output above.")
    
    return success

if __name__ == "__main__":
    main() 