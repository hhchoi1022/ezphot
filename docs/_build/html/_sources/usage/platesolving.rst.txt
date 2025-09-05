Platesolving
============

The :class:`~ezphot.methods.platesolve.Platesolve` class provides methods
for solving astrometry using external tools such as `Astrometry.net` and `SCAMP`.

Usage Examples
--------------

Astrometry.net Solution
-----------------------

The simplest way to add WCS information to an image is by running
`Astrometry.net` directly.

.. code-block:: python

    from ezphot.methods.platesolve import Platesolve
    from ezphot.imageobjects import ScienceImage

    # Load a science image without WCS
    sci = ScienceImage("example.fits")

    # Initialize platesolve
    solver = Platesolve()

    # Solve with Astrometry.net
    solved_img = solver.solve_astrometry(
        target_img = sci,
        overwrite = True,
        verbose = True
    )

    print("Updated WCS:", solved_img.wcs)

SCAMP Solution
--------------

For multiple images or more advanced WCS refinement, you can use `SCAMP`.

.. code-block:: python

    from ezphot.methods.platesolve import Platesolve
    from ezphot.imageobjects import ScienceImage

    sci = ScienceImage("example.fits")

    solver = Platesolve()

    solved_imgs = solver.solve_scamp(
        target_img = sci,
        scamp_sexparams = None,
        scamp_params = None,
        overwrite = True,
        verbose = True
    )

    for img in solved_imgs:
        print("Solved WCS:", img.wcs)
       
    # You can also run SCAMP for multiple images    
    sci1 = ScienceImage("example1.fits")
    sci2 = ScienceImage("example2.fits")

    solver = Platesolve()

    solved_imgs = solver.solve_scamp(
        target_img = [sci1, sci2],
        scamp_sexparams = None,
        scamp_params = None,
        overwrite = True,
        verbose = True
    )

    for img in solved_imgs:
        print("Solved WCS:", img.wcs)


.. note::

   while Astrometry.net is usually sufficient for solving **single images**, SCAMP is recommended for more accurate astrometric solution 

