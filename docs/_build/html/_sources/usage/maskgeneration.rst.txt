Masking
=======

The :class:`~ezphot.methods.maskgenerator.MaskGenerator` class provides tools to 
generate masks for astronomical images. Masks are essential for excluding unwanted 
pixels (invalid, saturated, or contaminated) and for preparing images for accurate 
background estimation, photometry, and subtraction.

Overview
--------

Masks can be generated for different purposes:

1. **Invalid Pixel Mask** : flags NaNs and large connected regions of zeros.  
2. **Source Mask** : detects and masks sources using SExtractor-like segmentation.  
3. **Circular Mask** : adds circular apertures at specified positions.  
4. **Cosmic Ray Mask** : detects and masks cosmic rays using `astroscrappy`.  

Creating a MaskGenerator
------------------------

.. code-block:: python

    from ezphot.methods import MaskGenerator

    maskgen = MaskGenerator()
    print(maskgen)

    >>> Method class: MaskGenerator
     For help, run maskgen.help().

Invalid Pixel Mask
------------------

Detect and mask invalid pixels such as NaNs and large connected regions of zeros.

.. code-block:: python

    from ezphot.imageobjects import ScienceImage
    from ezphot.methods import MaskGenerator

    sci = ScienceImage("/path/to/science.fits")
    maskgen = MaskGenerator()

    invalid_mask = maskgen.mask_invalidpixel(sci, threshold_invalid_connection=50000,
                                             save=True, visualize=True)

    print(invalid_mask)

Source Mask
-----------

Automatically detect and mask sources using `photutils.segmentation`.

.. code-block:: python

    src_mask = maskgen.mask_sources(sci, sigma=5.0, mask_radius_factor=3,
                                    saturation_level=50000, save=True, visualize=True)

Circular Mask
-------------

Add a circular mask centered at pixel or sky coordinates.

.. code-block:: python

    circ_mask = maskgen.mask_circle(sci, mask_type='source',
                                    x_position=1024, y_position=1024,
                                    radius_arcsec=10.0, unit='pixel',
                                    save=True, visualize=True)

Or, using RA/Dec coordinates:

.. code-block:: python

    circ_mask = maskgen.mask_circle(sci, mask_type='source',
                                    x_position=150.114, y_position=2.205,
                                    radius_arcsec=5.0, unit='deg')

Cosmic Ray Mask
---------------

Detect and mask cosmic rays using `astroscrappy`.

.. code-block:: python

    cr_mask = maskgen.mask_cosmicray(sci, gain=sci.egain,
                                     readnoise=sci.telinfo['readnoise'],
                                     sigclip=6.0, niter=4,
                                     save=True, visualize=True)
                                     
Output of the MaskGenerator
---------------------------

All :class:`~ezphot.methods.maskgenerator.MaskGenerator` methods return a 
:class:`~ezphot.imageobjects.mask.Mask` instance.  
A `Mask` is a binary FITS image with metadata and status tracking. It can be saved, 
combined with other masks, and linked back to the target image. See :class:`~ezphot.imageobjects.mask.Mask`

.. code-block:: python

    src_mask = maskgen.mask_sources(sci)
    print(src_mask)
    >>> Mask(
      is_exists   = True,
      is_saved    = False,
      data_load   = True,
      header_load = True,
      path        = /path/to/mask.fits,
      savedir     = /.../scidata/OBSERVATORY/TELKEY/OBJNAME/TELNAME/FILTER
    )

Integration with ScienceImage
-----------------------------

You can also generate masks directly from a :class:`~ezphot.imageobjects.scienceimage.ScienceImage`
instance using built-in methods:

.. code-block:: python

    # Invalid pixel mask
    invmask = sci.calculate_invalidmask(save=True, visualize=True)

    # Source mask
    srcmask = sci.calculate_sourcemask(save=True, visualize=True)


These wrappers internally use :class:`~ezphot.methods.maskgenerator.MaskGenerator` and other helper
classes, but are provided for convenience in day-to-day workflows.

:class:`~ezphot.methods.maskgenerator.MaskGenerator` is a central tool in the **ezphot** pipeline for ensuring that 
invalid pixels, bright sources, and cosmic rays are properly excluded before 
background subtraction and photometry.
