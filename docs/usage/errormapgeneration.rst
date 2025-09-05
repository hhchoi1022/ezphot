Error Map Generation
====================

The :class:`~ezphot.methods.errormapgenerator.ErrormapGenerator` class provides 
methods to calclate error maps from science or background images.  
Error maps are essential for estimating uncertainties in photometry and image subtraction.  

Overview
--------

:class:`~ezphot.methods.errormapgenerator.ErrormapGenerator` supports:

1. **Propagation-based error maps**  
   - Uses master bias, dark, and flat images (with optional flat error maps).  
   - Produces either source RMS maps or background RMS maps.  

2. **Background RMS error maps**  
   - Uses background maps estimated with SEP or photutils.  

3. **Source RMS error maps**  
   - Includes both background noise and source shot noise.  

Two types of error maps are supported:

- **bkgrms**: RMS of background only (bias, dark, flat, background).  
- **sourcerms**: Total RMS including source shot noise.  

Creating an ErrormapGenerator
-----------------------------

.. code-block:: python

    from ezphot.methods import ErrormapGenerator

    errgen = ErrormapGenerator()
    print(errgen)

    >>> Method class: ErrormapGenerator
     For help, run errgen.help()

Calculating Source/Background RMS from Image
--------------------------------------------

Alternatively, error maps can be derived directly from science images 
using background RMS from SEP:

.. code-block:: python

    errormap, bkg, sep_bkg = errgen.calculate_errormap_from_image(
        target_img=sci,
        target_srcmask=sci.sourcemask,
        target_ivpmask=sci.invalidmask,
        errormap_type='sourcerms',   # or 'bkgrms'
        box_size=64,
        filter_size=3,
        save=True,
        visualize=True
    )

- Returns an :class:`~ezphot.imageobjects.errormap.Errormap` instance of type ``errormap_type``, a :class:`~ezphot.imageobjects.background.Background`, and the underlying `sep.Background` instance.  

Calculating Source RMS (Propagation)
------------------------------------

Compute the source RMS error map by propagating errors from calibration frames:

.. code-block:: python

    from ezphot.imageobjects import ScienceImage, CalibrationImage

    sci = ScienceImage("/path/to/science.fits")
    mbias = CalibrationImage("/path/to/master_bias.fits")
    mdark = CalibrationImage("/path/to/master_dark.fits")
    mflat = CalibrationImage("/path/to/master_flat.fits")

    sourcerms = errgen.calculate_sourcerms_from_propagation(
        target_img=sci,
        mbias_img=mbias,
        mdark_img=mdark,
        mflat_img=mflat,
        save=True,
        visualize=True
    )

This returns an :class:`~ezphot.imageobjects.errormap.Errormap` instance of type ``sourcerms``.

Calculating Background RMS (Propagation)
----------------------------------------

You can also propagate errors to estimate the background RMS directly from a 
background image:

.. code-block:: python

    from ezphot.imageobjects import Background

    bkg = sci.calculate_bkg()
    bkgrms = errgen.calculate_bkgrms_from_propagation(
        target_bkg=bkg,
        mbias_img=mbias,
        mdark_img=mdark,
        mflat_img=mflat,
        save=True,
        visualize=True
    )

This produces an :class:`~ezphot.imageobjects.errormap.Errormap` of type ``bkgrms``.

The Errormap Instance
---------------------

All methods return an :class:`~ezphot.imageobjects.errormap.Errormap` instance.  
This object represents the error map FITS file, with full metadata and status tracking.  
``Please note that once the errormap is saved, you can access to errormap instance with sci.sourcerms or sci.bkgrms``

Example:

.. code-block:: python

    print(sourcerms)
    >>> Errormap(
      is_exists   = True,
      is_saved    = False,
      data_load   = True,
      header_load = True,
      path        = /path/to/background.fits,
      savedir     = /.../scidata/OBSERVATORY/TELKEY/OBJNAME/TELNAME/FILTER
    )
    
For more information, please refer :class:`~ezphot.imageobjects.errormap.Errormap`.

Integration with ScienceImage
-----------------------------

Convenience methods in :class:`~ezphot.imageobjects.ScienceImage` allow direct calculation:

.. code-block:: python

    # Calculate background RMS map from image itself (grid-based calculation)
    bkgrms = sci.calculate_bkgrms(
        target_srcmask = sci.sourcemask,
        target_ivpmask = sci.invalidmask,
        box_size =64,
        filter_size = 3,
        save = True,
        verbose = True,
        visualize = True,
        save_fig = True)

    # Calculate background RMS map from error propagation
    bkgrms  = sci.calculate_bkgrms_from_propagation(
        target_bkg = None,
        mbias = None,
        mdark = None,
        mflat = None
        mflaterr = None,
        ncombine = 1,
        readout_noise = None,
        save = True,
        verbose = True,
        visualize = True
        save_fig = True)

    # Calculate source RMS map from image itself (grid-based calculation)
    sourcerms = sci.calculate_sourcerms(
        target_srcmask = sci.sourcemask,
        target_ivpmask = sci.invalidmask,
        box_size = 64,
        filter_size = 3,
        save = False,
        verbose = True,
        visualize = True,
        save_fig = True)
    
    # Calculate source RMS map from error propagation
    sourcerms  = sci.calculate_sourcerms_from_propagation(
        mbias = None,
        mdark = None,
        mflat = None
        mflaterr = None,
        save = True,
        verbose = True,
        visualize = True
        save_fig = True)

These wrappers internally use :class:`~ezphot.methods.errormapgenerator.ErrormapGenerator`.

Summary
-------

- **sourcerms**: Total noise (source + background).  
- **bkgrms**: Background-only noise.  
- All error maps are returned as :class:`~ezphot.imageobjects.errormap.Errormap`.  
- Can be saved, visualized, and converted between RMS and weight maps.  

