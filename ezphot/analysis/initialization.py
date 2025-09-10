#%%
from ezphot.configuration import Configuration

# Initialize the configuration folder in your home directory.
config = Configuration()
config.initialize()
#%%

# You can load ONLY common configuration if tel_key is not defined
config_common = Configuration()
print(config_common.config)

# If tel_key is defined, common configurations + specific configurations are included in the configuration.
config_telescope = Configuration('7DT_C361K_HIGH_1x1')
print(config_telescope.config)

#%%

import os
from pathlib import Path

config = Configuration()
ezphot_sdtdata_obsfolder = config.config['SDTDATA_OBSSOURCEDIR']
ezphot_sdtdata_scifolder = config.config['SDTDATA_SCISOURCEDIR']
#%%
sdtdata_obsfolder = '/data/data1/obsdata' # REPLACE IT TO YOUR PATH FOR RAWDATA FOLDER
sdtdata_scifolder = '/data/data1/processed_1x1_gain2750' # REPLACE IT TO YOUR PATH FOR PROCESSED FOLDER

os.system(f'rm -rf {ezphot_sdtdata_obsfolder}')
os.system(f'rm -rf {ezphot_sdtdata_scifolder}')

Path(ezphot_sdtdata_obsfolder).symlink_to(Path(sdtdata_obsfolder))
Path(ezphot_sdtdata_scifolder).symlink_to(Path(sdtdata_scifolder))
#%%

gaiaxp_folder = '/data/data1/factory/ref_cat' # REPLACE IT TO YOUR PATH FOR SKY REFERENCE FOLDER
ezphot_catalog_folder = Path(config.config['CATALOG_DIR']) / 'GAIAXP'
ezphot_catalog_folder.symlink_to(Path(gaiaxp_folder))
#%%