# Photometric Calibration Test Suite

This directory contains comprehensive test files for the `TIPPhotometricCalibration.photometric_calibration` function.

## Files

1. **`test_photometric_calibration.py`** - Comprehensive test suite with class-based testing
2. **`test_photometric_calibration_simple.py`** - Simple parameter testing script
3. **`README_photometric_calibration_tests.md`** - This documentation file

## Usage

### Running the Simple Test Script

```bash
python test_photometric_calibration_simple.py
```

### Running the Comprehensive Test Suite

```bash
python test_photometric_calibration.py
```

## Test Parameters

The test files cover all parameters of the `photometric_calibration` function:

### Core Parameters
- `target_img`: Target image for calibration
- `target_catalog`: Target catalog with detected sources
- `catalog_type`: Reference catalog type ('GAIAXP', 'GAIA', 'PS1')
- `max_distance_second`: Maximum matching distance in arcseconds
- `calculate_color_terms`: Whether to calculate color terms
- `calculate_mag_terms`: Whether to calculate magnitude terms

### Star Selection Parameters
- `mag_lower`: Lower magnitude limit
- `mag_upper`: Upper magnitude limit
- `snr_lower`: Lower SNR limit
- `snr_upper`: Upper SNR limit
- `classstar_lower`: Minimum CLASS_STAR value
- `elongation_upper`: Maximum elongation
- `elongation_sigma`: Sigma clipping for elongation
- `fwhm_lower`: Lower FWHM limit in pixels
- `fwhm_upper`: Upper FWHM limit in pixels
- `fwhm_sigma`: Sigma clipping for FWHM
- `flag_upper`: Maximum FLAGS value
- `maskflag_upper`: Maximum IMAFLAGS_ISO value
- `inner_fraction`: Fraction of image to use (inner region)
- `isolation_radius`: Isolation radius in pixels

### Column Name Parameters
- `magnitude_key`: Magnitude column name
- `flux_key`: Flux column name
- `fluxerr_key`: Flux error column name
- `fwhm_key`: FWHM column name
- `x_key`: X coordinate column name
- `y_key`: Y coordinate column name
- `classstar_key`: CLASS_STAR column name
- `elongation_key`: Elongation column name
- `flag_key`: FLAGS column name
- `maskflag_key`: Mask flags column name

### Output Control Parameters
- `save`: Whether to save results
- `verbose`: Whether to print verbose output
- `visualize`: Whether to show plots
- `save_fig`: Whether to save plots
- `save_refcat`: Whether to save reference catalog

## Test Scenarios

### Simple Test Script (`test_photometric_calibration_simple.py`)

1. **Basic calibration** - Default parameters
2. **Catalog types** - GAIAXP, GAIA, PS1
3. **Matching distances** - 0.5, 1.0, 2.0, 5.0 arcsec
4. **Magnitude ranges** - Different magnitude limits
5. **SNR ranges** - Different signal-to-noise ratio limits
6. **FWHM ranges** - Different full-width half-maximum limits
7. **Calculation options** - Color terms and magnitude terms combinations
8. **Star selection parameters** - Different selection criteria
9. **Column names** - Different catalog column configurations
10. **Output options** - Different save and visualization options

### Comprehensive Test Suite (`test_photometric_calibration.py`)

The comprehensive test suite includes:

- **Basic photometric calibration** - Tests default functionality
- **Star selection parameters** - Tests various star selection criteria
- **Catalog type parameters** - Tests different reference catalogs
- **Matching distance parameters** - Tests different matching distances
- **Calculation parameters** - Tests color and magnitude term calculations
- **Column name parameters** - Tests different column name configurations
- **Output parameters** - Tests different output options
- **Edge cases** - Tests extreme parameter values
- **All parameters** - Tests with all parameters set

## Example Usage

### Basic Test
```python
from tippy.methods import TIPPhotometricCalibration

# Initialize
photometric_calibration = TIPPhotometricCalibration()

# Basic calibration
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
```

### Advanced Test with Custom Parameters
```python
# Advanced calibration with custom parameters
result_img, result_catalog, filtered_catalog = photometric_calibration.photometric_calibration(
    target_img=target_img,
    target_catalog=target_catalog,
    catalog_type='GAIAXP',
    max_distance_second=1.0,
    calculate_color_terms=True,
    calculate_mag_terms=True,
    
    # Star selection parameters
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
```

## Requirements

- TIPPy package installed
- Test data available in the specified path
- Required dependencies: numpy, matplotlib, astropy

## Notes

- The test scripts use the 7DT observatory and T22956 object by default
- Test data pattern: `calib*20250424_032003*100.fits`
- Visualization is disabled by default in tests to avoid blocking execution
- All tests include error handling and provide detailed feedback

## Troubleshooting

1. **No test data found**: Check the data path and ensure test files exist
2. **Import errors**: Ensure TIPPy is properly installed and accessible
3. **Memory issues**: Reduce the number of simultaneous tests or use smaller datasets
4. **Timeout errors**: Some tests may take longer with large datasets

## Expected Output

The tests will provide detailed feedback on each parameter combination:

```
============================================================
PHOTOMETRIC CALIBRATION PARAMETER TESTS
============================================================

1. Testing basic calibration with default parameters...
✓ Basic calibration successful

2. Testing different catalog types...
✓ Catalog type 'GAIAXP' successful
✓ Catalog type 'GAIA' successful
✗ Catalog type 'PS1' failed: No catalogs found

3. Testing different matching distances...
✓ Distance 0.5 arcsec successful
✓ Distance 1.0 arcsec successful
✓ Distance 2.0 arcsec successful
✓ Distance 5.0 arcsec successful

...

============================================================
ALL TESTS COMPLETED
============================================================
``` 