Loading Images
==============

The :class:`~ezphot.imageobjects.ScienceImage` class is the core object in **ezphot**
for representing a single science FITS exposure. It manages metadata, processing
status, associated products, and provides tools for background estimation, masking,
and photometry preparation.

Creating a ScienceImage instance
--------------------------------

You can create a :class:`~ezphot.imageobjects.ScienceImage` by passing the path to a
FITS file:

.. code-block:: python

    from ezphot.imageobjects import ScienceImage

    sci = ScienceImage("/path/to/science.fits")

    print(sci)

    >>> ScienceImage(
      is_exists   = True,
      is_saved    = False,
      data_load   = True,
      header_load = True,
      imgtype     = Light,
      exptime     = 300.0,
      filter      = m625,
      path        = /path/to/science.fits,
      savedir     = /.../scidata/OBSERVATORY/TELKEY/OBJNAME/TELNAME/FILTER
    )

Accessing Data and Header
-------------------------

The loaded FITS data and header can be accessed directly:

.. code-block:: python

    # Pixel data as a NumPy array
    data = sci.data

    # FITS header as an astropy.io.fits.Header
    header = sci.header

Inspecting Properties
---------------------

:class:`~ezphot.imageobjects.ScienceImage` exposes important metadata through convenient properties.

.. code-block:: python

    print(sci.obsdate)     # Observation date
    print(sci.filter)      # Filter name
    print(sci.exptime)     # Exposure time
    print(sci.seeing)      # Measured seeing
    print(sci.objname)     # Object name

This information is collected from both FITS headers and telescope configuration.

Working with Save Paths
-----------------------

`ScienceImage` automatically generates a structured save path for all
connected products (status files, masks, catalogs, background maps, etc.).

.. code-block:: python

    # Access save paths
    print(sci.savepath.savepath)   # Path to the saved FITS
    print(sci.savepath.statuspath) # Path to status JSON
    print(sci.savepath.infopath)   # Path to info JSON
    print(sci.savepath.catalogpath)# Path to source catalog

You can override the default location by setting a custom ``savedir``:

.. code-block:: python

    sci.savedir = "/custom/output/directory"

Saving and Removing Files
-------------------------

Save the current image to disk (including status and info):

.. code-block:: python

    sci.write()

This will save the current data and header of the instance to ``savepath.savepath``.

Remove the main FITS file and/or all associated products:

.. code-block:: python

    sci.remove(remove_main=True, remove_connected_files=True)

Visualization
-------------

:class:`~ezphot.imageobjects.ScienceImage` provides built-in methods to quickly inspect images.

Show the full image
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    sci.show()

This opens a matplotlib window with the science image displayed.

Show a zoomed region around a position
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Display a region around pixel coordinates (x=1024, y=1024)
    sci.show_position(x_position=1024, y_position=1024, zoom_radius=50)

    # Or around sky coordinates (RA, Dec in degrees)
    sci.show_position(x_position=150.114, y_position=2.205,
                      coord_type="coord", zoom_radius=30)

The ``show_position`` method is useful for checking sources, transients, or calibration
stars in either pixel or sky coordinates.

Open with DS9
~~~~~~~~~~~~~

.. code-block:: python

    sci.run_ds9()

This will launch SAOImage DS9 with the image loaded.

:class:`~ezphot.imageobjects.ScienceImage` forms the basis of the **ezphot** workflow and connects seamlessly
with other objects such as :class:`~ezphot.imageobjects.referenceimage.ReferenceImage`,
:class:`~ezphot.imageobjects.mask.Mask`, and :class:`~ezphot.imageobjects.background.Background`.

