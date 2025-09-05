Aperture Photometry
===================

The :class:`~ezphot.methods.aperturephotometry.AperturePhotometry` class provides
a flexible interface for performing aperture photometry. It supports:

1. SExtractor-based photometry
2. Forced photometry: Circular aperture(s)
3. Forced photometry: Elliptical aperture(s)
   
Input Options
-------------

All photometry methods in :class:`~ezphot.methods.aperturephotometry.AperturePhotometry`
accept the following optional inputs:

- ``target_bkg`` (:class:`~ezphot.imageobjects.background.Background`, optional)  
  A background image to subtract before performing photometry.  
.. tip::
  If background is not provided, no global background subtraction is applied, even when detection.
  In this case, you need to input background-subtracted target_img

- ``target_bkgrms`` (:class:`~ezphot.imageobjects.errormap.Errormap`, optional)  
  A background RMS (noise) map used to compute flux uncertainties and detection thresholds.  
.. tip::
  If background RMS is not provided, the RMS will be estimated internally (e.g. with grid-based sigma-clipping).
  In this case, error can be slightly overestimated in the case of extended and saturated sources.
  
- ``target_mask`` (:class:`~ezphot.imageobjects.mask.Mask`, optional)  
  A mask image to exclude unwanted pixels (e.g., cosmic rays, bad pixels, bright stars).  
  If not provided, the full image is used.  

.. note::
   Supplying accurate background and RMS maps can significantly improve
   photometric error estimates and limiting magnitude calculations.

Usage Examples
--------------

[Detection + Photometry] Source-Extractor photometry
----------------------------------------------------

.. code-block:: python

    from ezphot.methods.aperturephotometry import AperturePhotometry
    from ezphot.imageobjects import ScienceImage

    sci = ScienceImage("example.fits")
    aper = AperturePhotometry()

    catalog = aper.sex_photometry(
        target_img = sci,
        target_bkg = sci.bkgmap,
        target_bkgrms = sci.bkgrms,
        detection_sigma = 5,
        aperture_diameter_arcsec = [5, 7, 10],
        aperture_diameter_seeing = [3.5, 4.5],
        visualize = True,
        save_fig = True
    )

[Forced photometry] Circular aperture
-------------------------------------

.. code-block:: python

    # Forced photometry on pixel coordinates
    x_positions = [100.2, 230.5, 512.3]
    y_positions = [200.4, 420.7, 600.9]

    catalog = aper.circular_photometry(
        target_img = sci,
        target_bkg = sci.bkgmap,
        target_bkgrms = sci.bkgrms,
        x_arr = x_positions,
        y_arr = y_positions,
        aperture_diameter_arcsec = [5, 10],
        annulus_width_arcsec = 3,
        unit = 'pixel',
        visualize = True
    )

    # You can also supply RA/Dec positions directly if the image has a valid WCS.
    ra_positions = [150.11475, 150.11700, 150.11925]
    dec_positions = [2.2050, 2.2100, 2.2150]

    catalog = aper.circular_photometry(
        target_img = sci,
        x_arr = ra_positions,
        y_arr = dec_positions,
        aperture_diameter_arcsec = [5, 10],
        aperture_diameter_seeing = [2.5, 3.5],
        annulus_width_arcsec = 3,
        unit = 'coord', # unit to "coord"
        visualize = True
    )

[Forced photometry] Elliptical Aperture
---------------------------------------


.. code-block:: python

    x_positions = [250.5]
    y_positions = [300.5]
    sma_pixel = [3.0]    # semi-major axis in arcsec
    smi_pixel = [2.0]    # semi-minor axis in arcsec
    theta_deg = [45.0]    # position angle in degrees

    catalog = aper.elliptical_photometry(
        target_img = sci,
        target_bkg = sci.bkgmap,
        target_bkgrms = sci.bkgrms,
        x_arr = x_positions,
        y_arr = y_positions,
        sma_arr = sma_pixel,
        smi_arr = smi_pixel,
        theta_arr = theta_deg,
        unit = 'pixel',
        annulus_ratio = None,
        visualize = True
    )
    
    # You can also provide RA/Dec positions if the image has a valid WCS.
    ra_positions = [150.115]
    dec_positions = [2.210]
    sma_arcsec = [3.0]
    smi_arcsec = [2.0]
    theta_deg = [30.0]

    catalog = aper.elliptical_photometry(
        target_img = sci,
        target_bkg = sci.bkgmap,
        target_bkgrms = sci.bkgrms,
        x_arr = ra_positions,
        y_arr = dec_positions,
        sma_arr = sma_arcsec,
        smi_arr = smi_arcsec,
        theta_arr = theta_deg,
        unit = 'coord',
        annulus_ratio = None,
        visualize = True
    )


Output
------

All photometry methods return a :class:`~ezphot.dataobjects.catalog.Catalog` instance.

Typical catalog columns include:

- **X_IMAGE, Y_IMAGE** : source positions in pixels
- **X_WORLD, Y_WORLD** : sky coordinates (RA, Dec)
- **FLUX_APER, FLUXERR_APER, MAG_APER, MAGERR_APER** : circular aperture measurements
- **FLUX_ELIP, FLUXERR_ELIP, MAG_ELIP, MAGERR_ELIP** : elliptical aperture measurements
- **FLUX_AUTO, MAG_AUTO** : Kron-like aperture measurements (SExtractor only)
- **NPIX_APER, NPIX_ELIP, NPIX_AUTO** : effective aperture areas
- **SKYSIG, DETECT_THRESH** : background noise and detection thresholds

For more detail, see :class:`~ezphot.dataobjects.catalog.Catalog`

You can inspect the catalog directly via:

.. code-block:: python

    tbl = catalog.data
    print(tbl[:5])



Notes
-----

* All methods return a :class:`~ezphot.dataobjects.catalog.Catalog` object.
* Aperture diameters can be specified in **arcseconds** or relative to
  image seeing (via ``aperture_diameter_seeing``).
* Local background subtraction can be enabled with annuli in both circular
  and elliptical methods.
* Visualization produces a zoomed-in figure with drawn apertures.


