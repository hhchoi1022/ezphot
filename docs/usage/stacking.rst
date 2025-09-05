Stacking
========

The :class:`~ezphot.methods.stack.Stack` class provides methods for
stacking and co-adding astronomical images.  
It supports multiprocessing-based stacking, SWarp integration, 
zero-point scaling, PSF/seeing matching, and image quality selection.

Usage Examples
--------------

Basic Multiprocess Stacking
---------------------------

Use multiple CPU cores to combine a list of images with optional
background subtraction and error maps.

.. code-block:: python

    from ezphot.methods.stack import Stack
    from ezphot.imageobjects import ScienceImage

    sci1 = ScienceImage("image1.fits")
    sci2 = ScienceImage("image2.fits")
    sci3 = ScienceImage("image3.fits")

    stacker = Stack()

    stacked_img, stacked_bkgrms = stacker.stack_multiprocess(
        target_imglist = [sci1, sci2, sci3],
        target_bkglist = [sci1.bkgmap, sci2.bkgmap, sci3.bkgmap],
        target_bkgrmslist = [sci1.bkgrms, sci2.bkgrms, sci3.bkgrms],
        combine_type = "median",
        clip_type = "sigma",
        sigma = 3.0,
        n_proc = 8,
        resample = True,
        resample_type = "LANCZOS3",
        save = True
    )

    print("Stacked image saved to:", stacked_img.path)

Stacking with SWarp
-------------------

[OBSOLETE] You can also use **SWarp** for large-scale resampling and co-addition.

.. code-block:: python

    stacked_img, stacked_weight = stacker.stack_swarp(
        target_imglist = [sci1, sci2, sci3],
        target_bkglist = [sci1.bkgmap, sci2.bkgmap, sci3.bkgmap],
        target_errormaplist = [sci1.bkgrms, sci2.bkgrms, sci3.bkgrms],
        combine_type = "median",
        resample = True,
        resample_type = "LANCZOS3",
        scale = True,
        zp_key = "ZP_APER_1",
        convolve = True,
        seeing_key = "SEEING",
        save = True
    )

    print("SWarp stack:", stacked_img.path)
    print("Weight map:", stacked_weight.path)

Selecting High-Quality Images to stack
--------------------------------------

Filter images by seeing, depth, and ellipticity before stacking.

.. code-block:: python

    selected_imgs = stacker.select_quality_images(
        target_imglist = [sci1, sci2, sci3],
        seeing_limit = 5.0,
        depth_limit = 20.0,
        ellipticity_limit = 0.3,
        visualize = True
    )

    print("Selected", len(selected_imgs), "images for stacking.")
    
    selected_bkglist = [img.bkgmap for img in selected_imgs]
    selected_bkgrmslist = [img.bkgrms for img in selected_imgs]
    
    stacked_img, stacked_bkgrms = stacker.stack_multiprocess(
        target_imglist = selected_imgs,
        target_bkglist = selected_bkglist,
        target_bkgrmslist = selected_bkgrmslist,
        combine_type = "median",
        clip_type = "sigma",
        sigma = 3.0,
        n_proc = 8,
        resample = True,
        resample_type = "LANCZOS3",
        save = True
    )
    

Matching Zeropoints
-------------------

Normalize images to a common photometric zeropoint before stacking.

.. code-block:: python

    scaled_imgs, scaled_errormaps = stacker.match_zeropoints(
        target_imglist = [sci1, sci2, sci3],
        target_errormaplist = [sci1.bkgrms, sci2.bkgrms, sci3.bkgrms],
        method = "median",
        zp_key = "ZP_APER_1",
        save = True
    )

Matching Seeing
---------------

Convolve images so that all have consistent PSF/seeing. 

.. code-block:: python

    matched_imgs, matched_errormaps = stacker.match_seeing(
        target_imglist = [sci1, sci2, sci3],
        target_errormaplist = [sci1.bkgrms, sci2.bkgrms, sci3.bkgrms],
        target_bkglist = [sci1.bkgmap, sci2.bkgmap, sci3.bkgmap],
        kernel = "gaussian",
        visualize = True,
        save = True
    )

.. note::
   Background subtraction, zeropoint scaling, and seeing matching are optional,
   but highly recommended for producing high-quality stacked images.

