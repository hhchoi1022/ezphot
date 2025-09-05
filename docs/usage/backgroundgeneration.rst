Background Map Generation
=========================

The :class:`~ezphot.methods.backgroundgenerator.BackgroundGenerator` class provides 
methods to calculate and subtract the background of astronomical images. 
Accurate background estimation is critical for photometry, subtraction, 
and transient detection.

Overview
--------

`BackgroundGenerator` supports:

1. **2D background map estimation** using SEP (Source Extractor-like).  
2. **Constant background estimation** (1D).  
3. **Background subtraction** from science or reference images.  

.. note::

   The `estimate_with_photutils` method is **obsolete**. 

Creating a BackgroundGenerator
------------------------------

.. code-block:: python

    from ezphot.methods import BackgroundGenerator

    bkggenerator = BackgroundGenerator()
    print(bkggenerator)

    >>> Method class: BackgroundGenerator
     For help, run bkggenerator.help()

Estimating Background with SEP
------------------------------

Estimate the background map using SEP:

.. code-block:: python

    from ezphot.imageobjects import ScienceImage

    sci = ScienceImage("/path/to/science.fits")
    bkg, sep_bkg = bkggenerator.estimate_with_sep(
        target_img=sci,
        box_size=64,
        filter_size=3,
        target_ivpmask=sci.invalidmask,   # Optional invalid pixel mask
        target_srcmask=sci.sourcemask,    # Optional source mask
        save=True,
        visualize=True
    )

This returns both an :class:`~ezphot.imageobjects.background.Background` instance and the underlying `sep.bkg` Background.  

- **Invalid mask** (`invalidmask`) ensures NaNs or large zero regions are ignored.  
- **Source mask** (`sourcemask`) excludes bright sources from the background model.  

Subtracting the Background
--------------------------

You can subtract a previously estimated background map from the image:

.. code-block:: python

    subtracted = bkggenerator.subtract_background(
        target_img=sci,
        target_bkg=bkg,
        save=True,
        visualize=True
    )

This produces a new :class:`~ezphot.imageobjects.scienceimage.ScienceImage` with the background removed.

The Background Instance
-----------------------

:class:`~ezphot.methods.backgroundgenerator.BackgroundGenerator` methods return a background as
:class:`~ezphot.imageobjects.background.Background` instance.  
This object represents the **background FITS image** with full metadata and 
status tracking. It can be saved, removed, or linked back to its target image.
``Please note that once the bkg is saved, you can access to background instance with sci.bkgmap``

Example:

.. code-block:: python

    from ezphot.imageobjects import ScienceImage

    sci = ScienceImage("/path/to/science.fits")
    bkg, sep_bkg = bkggenerator.estimate_with_sep(
        target_img=sci,
        box_size=64,
        filter_size=3,
        target_ivpmask=sci.invalidmask,   # Optional invalid pixel mask
        target_srcmask=sci.sourcemask,    # Optional source mask
        save=False,
        visualize=True
    )

    print(bkg)
    >>> Background(
      is_exists   = True,
      is_saved    = False,
      data_load   = True,
      header_load = True,
      path        = /path/to/background.fits,
      savedir     = /.../scidata/OBSERVATORY/TELKEY/OBJNAME/TELNAME/FILTER
    )
    
    print(sci.bkgmap) # Since bkg instance is not saved, there is no file with the savepath. sci.bkgmap is not connected to any file. 
    >>> None
    
    bkg.write() # Write the background instance to bkg.savepath.savepath
    
    print(sci.bkgmap) # Then, sci.bkgmap indicate bkg instance 
    >>> Background(
      is_exists   = True,
      is_saved    = False,
      data_load   = True,
      header_load = True,
      path        = /path/to/background.fits,
      savedir     = /.../scidata/OBSERVATORY/TELKEY/OBJNAME/TELNAME/FILTER
    )


Integration with ScienceImage
-----------------------------

For convenience, you can estimate backgrounds directly from a 
:class:`~ezphot.imageobjects.scienceimage.ScienceImage` instance:

.. code-block:: python

    # Background map
    bkg = sci.calculate_bkg(
        target_srcmask=sci.sourcemask,
        target_ivpmask=sci.invalidmask,
        save=True, visualize=True
    )


These wrappers internally call :class:`~ezphot.methods.backgroundgenerator.BackgroundGenerator` 
but are provided for easier workflow integration.

Summary
-------

- Use ``estimate_with_sep`` for robust background estimation.  
- Combine with ``invalidmask`` and ``sourcemask`` for cleaner models.  
- ``subtract_background`` creates background-subtracted images.  
- All results are returned as :class:`~ezphot.imageobjects.background.Background` instances 
  that can be saved, visualized, and reloaded.

