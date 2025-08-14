.. rubric:: Key Properties

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - **Property**
     - **Description**
   * - ``altitude``
     - Altitude of the telescope.
   * - ``azimuth``
     - Azimuth of the telescope.
   * - ``biaspath``
     - Path to the bias image.
   * - ``binning``
     - Binning of the image.
   * - ``bkgpath``
     - Path to the background image.
   * - ``bkgtype``
     - Type of the background image.
   * - ``ccdtemp``
     - CCD temperature.
   * - ``center``
     - Center pixel (0-based) and its world coordinates (RA, Dec).
   * - ``connected_files``
     - Return all associated files that would be deleted in `remove()` if remove_connected_files=True,
   * - ``darkpath``
     - Path to the dark image.
   * - ``data``
     - Image data array.
   * - ``dec``
     - Declination of the target.
   * - ``depth``
     - Depth of the image.
   * - ``egain``
     - Electron gain of the image.
   * - ``emappath``
     - Path to the emap image.
   * - ``emaptype``
     - Type of the emap image.
   * - ``exptime``
     - Exposure time of the image.
   * - ``filter``
     - Filter name.
   * - ``flatpath``
     - Path to the flat image.
   * - ``fovx``
     - Field of view along the first axis.
   * - ``fovy``
     - Field of view along the second axis.
   * - ``gain``
     - Gain of the image.
   * - ``header``
     - FITS header object.
   * - ``imgtype``
     - Type of the image. Among BIAS, DARK, FLAT, LIGHT, OBJECT, UNKNOWN.
   * - ``info``
     - Register necessary info fields
   * - ``is_data_loaded``
     - Whether the image data is loaded.
   * - ``is_exists``
     - Whether the image file exists.
   * - ``is_header_loaded``
     - Whether the image header is loaded.
   * - ``is_saved``
     - Check if the image has been saved
   * - ``jd``
     - Julian date of the observation.
   * - ``logger``
     - 
   * - ``maskpath``
     - Path to the mask image.
   * - ``masktype``
     - Type of the mask image.
   * - ``mjd``
     - Modified Julian date of the observation.
   * - ``naxis1``
     - Number of pixels along the first axis.
   * - ``naxis2``
     - Number of pixels along the second axis.
   * - ``ncombine``
     - Number of combined images.
   * - ``objname``
     - Object name.
   * - ``obsdate``
     - Observation date in UTC.
   * - ``observatory``
     - Name of the observatory.
   * - ``obsmode``
     - Observation mode.
   * - ``pixelscale``
     - Pixel scale of the image.
   * - ``ra``
     - Right ascension of the target.
   * - ``savedir``
     - Return the directory where this image and associated files will be saved.
   * - ``savepath``
     - Dynamically builds save paths based on current header info
   * - ``seeing``
     - Seeing of the image.
   * - ``specmode``
     - Spectroscopic mode.
   * - ``telname``
     - Name of the telescope.
   * - ``wcs``
     - WCS information of the image.
   * - ``zp``
     - Zero point of the image.
   * - ``zperr``
     - Zero point error of the image.
