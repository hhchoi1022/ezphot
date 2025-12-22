Getting Started
===============

After installing **ezphot**, you need to perform some one-time initialization steps
to set up your configuration directory, connect to data folders, and optionally link
external catalogs.

Initialization
--------------

For the first time after installation, run the following script:

.. code-block:: python

    from ezphot.configuration import Configuration

    # Initialize the configuration folder in your home directory.
    config = Configuration()
    config.initialize()

This will create the ``~/ezphot`` folder and copy the default
configuration file to ``~/ezphot/config``. 

There are two types of configuration folders:

1. Common configurations (in ``~/ezphot/config/common``)

2. Telescope configurations (in ``~/ezphot/config/specific``)

Loading Telescope Configurations
--------------------------------

You can list all available telescope keys defined in the configuration:

.. code-block:: python

    print(config.available_telescope_keys)
    
    >>>['7DT_C361K_LOW_2x2',
        'SAO_C361K_1x1',
        '7DT_C361K_HIGH_2x2',
        ...]

To load configuration:

.. code-block:: python

    # You can load ONLY common configuration if tel_key is not defined
    config_common = Configuration() 
    print(config_common.config)
 
    # If tel_key is defined, common configurations + specific configurations are included in the configuration.
    config_telescope = Configuration('7DT_C361K_HIGH_1x1') 
    print(config_telescope.config)
    
    >>> {'PSFEX_CONFIG': '/home/hhchoi1022/ezphot/config/common/psfex/default.psfexconfig',
         'PSFEX_DIR': '/home/hhchoi1022/ezphot/config/common/psfex',
         'PSFEX_LOGDIR': '/home/hhchoi1022/ezphot/log/psfex/log',
          ...}
     



This will load all configurations in common and telescope spefcific folders.
**You can modify the configratuon** by editing the contents in the such folder.
    

Adding Telescope Configurations (TBD)
-------------------------------------

Currently, please tell me if you wish to add more telescope configuration.


(Optional) Connecting Data Directories
--------------------------------------

If you want to use the :class:`~ezphot.utils.sdtdataquerier.SDTDataQuerier`, create a symlink in the configured catalog directory.
The :class:`~ezphot.utils.sdtdataquerier.SDTDataQuerier` requires symbolic links to your raw and processed data directories.
For changing default directory of SDTData, check sdtdata.config file in configuration folder.

.. code-block:: python

    import os
    from pathlib import Path

    config = Configuration()
    ezphot_sdtdata_obsfolder = config.config['SDTDATA_OBSSOURCEDIR']
    ezphot_sdtdata_scifolder = config.config['SDTDATA_SCISOURCEDIR']

    sdtdata_obsfolder = '/lyman/data1/obsdata' # REPLACE IT TO YOUR PATH FOR RAWDATA FOLDER
    sdtdata_scifolder = '/lyman/data1/processed_1x1_gain2750' # REPLACE IT TO YOUR PATH FOR PROCESSED FOLDER

    os.system(f'rm -rf {ezphot_sdtdata_obsfolder}')
    os.system(f'rm -rf {ezphot_sdtdata_scifolder}')

    Path(ezphot_sdtdata_obsfolder).symlink_to(Path(sdtdata_obsfolder))
    Path(ezphot_sdtdata_scifolder).symlink_to(Path(sdtdata_scifolder))

(Optional) Connecting Sky Catalogs
----------------------------------

If you want to use the ``GaiaXP Catalogs"`` for :class:`~ezphot.skycatalog.skycatalog.SkyCatalog` class, create a symlink in the configured catalog directory:

.. code-block:: python

    gaiaxp_folder = '/lyman/data1/factory/ref_cat' # REPLACE IT TO YOUR PATH FOR SKY REFERENCE FOLDER 
    ezphot_catalog_folder = Path(config.config['CATALOG_DIR']) / 'GAIAXP'
    ezphot_catalog_folder.symlink_to(Path(gaiaxp_folder))


