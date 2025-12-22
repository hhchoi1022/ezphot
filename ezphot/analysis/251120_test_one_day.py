
#%%
from ezphot.utils import DataBrowser
from ezphot.utils import SDTDataQuerier
#%%
sdtquerier = SDTDataQuerier()
sdtquerier.show_obssourcedata(foldername = '2025-11-15_gain2750')
sdtquerier.sync_obsdata(foldername = '2025-11-15_gain2750')
#%%
db = DataBrowser('obsdata')
db.obsdate = '2025-11-15_gain2750'
target_imgset = db.search('*.fits', return_type = 'science')

#%%