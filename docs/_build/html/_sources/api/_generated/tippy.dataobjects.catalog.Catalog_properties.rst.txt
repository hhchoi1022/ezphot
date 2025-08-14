.. rubric:: Key Properties

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - **Property**
     - **Description**
   * - ``connected_files``
     - Return all associated files that would be deleted in `remove()` if remove_connected_files=True,
   * - ``data``
     - Lazy-load table data from path by trying multiple formats.
   * - ``is_data_loaded``
     - Check if the data is loaded.
   * - ``is_exists``
     - Check if the catalog file exists.
   * - ``is_saved``
     - Check if the catalog has been saved.
   * - ``nselected``
     - Number of selected sources in the target data.
   * - ``nsources``
     - Number of sources in the catalog.
   * - ``savedir``
     - Return the directory where this image and associated files will be saved.
   * - ``savepath``
     - Dynamically builds save paths based on the path
   * - ``target_data``
     - Return the selected sources by self.select_sources().
