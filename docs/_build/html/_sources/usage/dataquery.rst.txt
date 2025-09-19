Data Query with DataBrowser
==========================

The DataBrowser is a powerful tool that provides a unified interface for searching and loading data from the telescope data directory. It's particularly useful when your data is organized in a structured save path format.

Overview
--------

The DataBrowser allows you to easily query different types of astronomical data objects from your organized data directory structure. It supports filtering by various attributes and can return data as different object types.

Supported Data Types
--------------------

With the DataBrowser, you can query the following data objects:

- **ScienceImage** - Science images from observations
- **ReferenceImage** - Reference images for photometry
- **CalibrationImage** - Calibration frames (bias, dark, flat)
- **Background** - Background images
- **Errormap** - Error maps associated with images
- **Mask** - Mask images
- **Catalog** - Source catalogs

Basic Usage
-----------

Import and Initialize
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from ezphot.utils import DataBrowser
    
    # Initialize DataBrowser for science data
    databrowser = DataBrowser('scidata')
    
    # For reference data
    ref_browser = DataBrowser('refdata')
    
    # For calibration data
    calib_browser = DataBrowser('calibdata')

Available Folder Types
~~~~~~~~~~~~~~~~~~~~~~

The DataBrowser supports different folder types corresponding to different data directories:

- **scidata** - Science data directory
- **refdata** - Reference data directory  
- **calibdata** - Calibration data directory
- **mcalibdata** - Master calibration data directory
- **obsdata** - Observation data directory

Filtering Data
--------------

Set Search Attributes
~~~~~~~~~~~~~~~~~~~~~

You can filter your search by setting various attributes:

.. code-block:: python

    databrowser = DataBrowser('scidata')
    
    # Filter by object name
    databrowser.objname = 'T00528'
    
    # Filter by filter band
    databrowser.filter = 'r'
    
    # Filter by telescope name
    databrowser.telname = 'LCOGT-1m-003'
    
    # Filter by observatory
    databrowser.observatory = 'LCO'
    
    # Filter by telescope key
    databrowser.telkey = '1m'
    
    # Filter by image type (for calibration data)
    databrowser.imgtype = 'flat'
    
    # Filter by observation date (for observation data)
    databrowser.obsdate = '2024-01-15'

View Current Filters
~~~~~~~~~~~~~~~~~~~~

To see your current search configuration:

.. code-block:: python

    print(databrowser)
    
    # Output shows current filters and search path
    # <DataBrowser(searchpath='/path/to/data', foldertype='scidata')>
    # Search Attributes:
    #   observatory : LCO
    #   telkey      : 1m
    #   imgtype     : *
    #   telname     : LCOGT-1m-003
    #   objname     : T00528
    #   filter      : r
    #   obsdate     : *

Searching and Loading Data
--------------------------

Basic Search
~~~~~~~~~~~~

Search for files and return file paths:

.. code-block:: python

    # Search for FITS files matching current filters
    file_paths = databrowser.search('*.fits', return_type='path')
    print(f"Found {len(file_paths)} files")

Load as Different Object Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The DataBrowser can return data as different object types:

Science Images
++++++++++++++

.. code-block:: python

    # Load science images
    science_images = databrowser.search('*.fits', return_type='science')
    print(f"Loaded {len(science_images)} science images")
    
    # Access individual images
    for img in science_images:
        print(f"Object: {img.objname}, Filter: {img.filter}")

Reference Images
++++++++++++++++

.. code-block:: python

    # Load reference images
    ref_browser = DataBrowser('refdata')
    ref_browser.objname = 'T00528'
    ref_browser.filter = 'r'
    
    reference_images = ref_browser.search('*.fits', return_type='reference')
    print(f"Loaded {len(reference_images)} reference images")

Calibration Images
++++++++++++++++++

.. code-block:: python

    # Load calibration images
    calib_browser = DataBrowser('calibdata')
    calib_browser.observatory = 'LCO'
    calib_browser.imgtype = 'flat'
    
    calibration_images = calib_browser.search('*.fits', return_type='calibration')
    print(f"Loaded {len(calibration_images)} calibration images")

Background Images
+++++++++++++++++

.. code-block:: python

    # Load background images
    background_images = databrowser.search('*background*.fits', return_type='background')
    print(f"Loaded {len(background_images)} background images")

Error Maps
++++++++++

.. code-block:: python

    # Load error maps
    error_maps = databrowser.search('*error*.fits', return_type='errormap')
    print(f"Loaded {len(error_maps)} error maps")

Masks
+++++

.. code-block:: python

    # Load mask images
    mask_images = databrowser.search('*mask*.fits', return_type='mask')
    print(f"Loaded {len(mask_images)} mask images")

Catalogs
++++++++

.. code-block:: python

    # Load catalogs
    catalogs = databrowser.search('*.cat', return_type='catalog')
    print(f"Loaded {len(catalogs)} catalogs")

Image Information
+++++++++++++++++

Get image metadata without loading full images:

.. code-block:: python

    # Get image information table
    imginfo = databrowser.search('*.fits', return_type='imginfo')
    print(imginfo)
    # Returns astropy Table with image metadata

Practical Examples
------------------

Example 1: Load Science Images for Photometry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from ezphot.utils import DataBrowser
    
    # Set up DataBrowser for science data
    databrowser = DataBrowser('scidata')
    databrowser.objname = 'T00528'
    databrowser.filter = 'r'
    
    # Load science images
    science_images = databrowser.search('*100.fits', return_type='science')
    print(f"Loaded {len(science_images)} science images for photometry")
    
    # Access the first image
    if science_images:
        first_image = science_images[0]
        print(f"First image: {first_image.objname}, {first_image.filter}")
        print(f"Exposure time: {first_image.exptime}")

Example 2: Load Reference and Calibration Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Load reference images
    ref_browser = DataBrowser('refdata')
    ref_browser.objname = 'T00528'
    ref_browser.filter = 'r'
    reference_images = ref_browser.search('*.fits', return_type='reference')
    
    # Load calibration images
    calib_browser = DataBrowser('calibdata')
    calib_browser.observatory = 'LCO'
    calib_browser.imgtype = 'flat'
    flat_images = calib_browser.search('*.fits', return_type='calibration')
    
    print(f"Reference images: {len(reference_images)}")
    print(f"Flat images: {len(flat_images)}")

Example 3: Search with Multiple Filters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Search across multiple filters
    databrowser = DataBrowser('scidata')
    databrowser.objname = 'T00528'
    # Don't set filter to search all filters
    
    # Get all available filters for this object
    available_filters = databrowser.keys['filter']
    print(f"Available filters: {available_filters}")
    
    # Load images for specific filter
    databrowser.filter = 'g'
    g_images = databrowser.search('*.fits', return_type='science')
    
    databrowser.filter = 'r'
    r_images = databrowser.search('*.fits', return_type='science')
    
    print(f"g-band images: {len(g_images)}")
    print(f"r-band images: {len(r_images)}")

Advanced Features
-----------------

Available Keys
~~~~~~~~~~~~~~

Check what values are available for each filter attribute:

.. code-block:: python

    databrowser = DataBrowser('scidata')
    available_keys = databrowser.keys
    
    print("Available observatories:", available_keys['observatory'])
    print("Available telescopes:", available_keys['telname'])
    print("Available objects:", available_keys['objname'])
    print("Available filters:", available_keys['filter'])

Help and Documentation
~~~~~~~~~~~~~~~~~~~~~~

Get help on available methods:

.. code-block:: python

    databrowser.help()
    
    # Or use Python's help system
    help(databrowser)

Search in Specific Folders
~~~~~~~~~~~~~~~~~~~~~~~~~~

Search in a specific folder instead of using the default search path:

.. code-block:: python

    # Search in a specific folder
    custom_folder = '/path/to/custom/folder'
    results = databrowser.search_folder('*.fits', custom_folder, return_type='science')

Notes
-----

- The DataBrowser works best with data organized in the default structured save path format
- All searches are performed using the current filter attributes
- Use `*` in patterns to match multiple files
- The DataBrowser uses multiprocessing for efficient loading of multiple files
- Failed file loads are handled gracefully with warning messages

Tips for Efficient Usage
~~~~~~~~~~~~~~~~~~~~~~~~

1. **Set specific filters first** - This narrows down the search space and improves performance
2. **Use appropriate file patterns** - Be specific with your file patterns (e.g., `*100.fits` instead of `*.fits`)
3. **Check available keys** - Use `databrowser.keys` to see what values are available before setting filters
4. **Use the right return type** - Choose the most appropriate return type for your use case
5. **Handle empty results** - Always check if results are empty before processing
