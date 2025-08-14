


#%%
from ezphot.catalog import Catalog
from ezphot.helper import DataBrowser
#%%
databrowser = DataBrowser('obsdata')
# %%
databrowser.observatory = '7DT'
databrowser.objname = 'T22956'
#%%
all_paths = databrowser.search(pattern = '*.cat')
# %%
all_paths