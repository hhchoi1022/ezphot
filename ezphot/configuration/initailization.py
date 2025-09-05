

#%%
from ezphot.configuration import Configuration
from pprint import pprint
#%% 
"""
For the first time after installation, run below command.
This command will initialize the configuration folder in your home directory.
This will copy the default configuration file to the configuration folder.
"""
config = Configuration()
config.initialize()
# %%
"""
Check the telescope keys available in the configuration file.
"""
pprint(config.available_telescope_keys)
# %%
"""
Check the configuration for a specific telescope key.
"""
# Telescope specific configuration
config_telescope = Configuration('7DT_C361K_HIGH_1x1')
pprint(config_telescope.config)
# %%
"""
For SDTDataQuerier.py class, you need to set the symlink to the data directory.
"""
config = Configuration()
pprint(config.config['SDTDATA_OBSSOURCEDIR'])
pprint(config.config['SDTDATA_SCISOURCEDIR'])
ezphot_sdtdata_obsfolder = config.config['SDTDATA_OBSSOURCEDIR']
ezphot_sdtdata_scifolder = config.config['SDTDATA_SCISOURCEDIR']

# You can do this in bash shell. 
import os
from pathlib import Path
sdtdata_obsfolder = '/lyman/data1/obsdata'
sdtdata_scifolder = '/lyman/data1/processed_1x1_gain2750'
os.system(f'rm -rf {ezphot_sdtdata_obsfolder}')
os.system(f'rm -rf {ezphot_sdtdata_scifolder}')
try:
    Path(ezphot_sdtdata_obsfolder).symlink_to(Path(sdtdata_obsfolder))
    print(f'SDTData connected: {ezphot_sdtdata_obsfolder}')
except FileExistsError:
    print(f'SDTData already connected: {ezphot_sdtdata_obsfolder}')
    pass
try:
    Path(ezphot_sdtdata_scifolder).symlink_to(Path(sdtdata_scifolder))
    print(f'SDTData connected: {ezphot_sdtdata_scifolder}')
except FileExistsError:
    print(f'SDTData already connected: {ezphot_sdtdata_scifolder}')
    pass

# %%
"""
For GAIAXP SkyCatalog, you need to set the symlink to the skycatalog directory.
"""
gaiaxp_folder = '/lyman/data1/factory/ref_cat'
ezphot_catalog_folder = Path(config.config['CATALOG_DIR']) / 'GAIAXP'
try:
    Path(ezphot_catalog_folder).symlink_to(Path(gaiaxp_folder))
    print(f'GAIAXP Catalog connected: {ezphot_catalog_folder}')
except FileExistsError:
    print(f'GAIAXP Catalog already connected: {ezphot_catalog_folder}')
    pass
# %%

