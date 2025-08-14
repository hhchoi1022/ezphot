.. rubric:: Key Properties

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - **Property**
     - **Description**
   * - ``center``
     - Center coordinates of the image in RA and DEC.
   * - ``connected_files``
     - Return all associated files that would be deleted in `remove()` if remove_connected_files=True,
   * - ``data``
     - Image data array.
   * - ``egain``
     - Electron gain of the image.
   * - ``exptime``
     - Exposure time of the image.
   * - ``header``
     - FITS header object.
   * - ``info``
     - Register necessary info fields
   * - ``is_data_loaded``
     - Check if the image data is loaded.
   * - ``is_exists``
     - Check if the image file exists.
   * - ``is_header_loaded``
     - Check if the image header is loaded.
   * - ``is_saved``
     - Check if the image has been saved
   * - ``logger``
     - 
   * - ``naxis1``
     - Number of pixels along the first axis.
   * - ``naxis2``
     - Number of pixels along the second axis.
   * - ``savedir``
     - Return the directory where this image and associated files will be saved.
   * - ``savepath``
     - Dynamically builds save paths based on current header info
   * - ``target_img``
     - 
   * - ``wcs``
     - WCS information of the image.
