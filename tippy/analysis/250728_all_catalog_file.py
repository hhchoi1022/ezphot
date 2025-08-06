


#%%
from tippy.catalog import TIPCatalog
from tippy.helper import TIPDataBrowser
#%%
databrowser = TIPDataBrowser('obsdata')
# %%
databrowser.observatory = '7DT'
databrowser.objname = 'T22956'
#%%
all_paths = databrowser.search(pattern = '*.cat')
# %%
all_paths