#%%
from typing import List
import glob
from pathlib import Path
from tqdm import tqdm

from tippy.utils import SDTData
from tippy.catalog import SourceCatalog, SourceCatalogDataset
from tippy.helper import Helper
from tippy.image import ScienceImage
#%%
class SDTSpectrum(Helper):
    
    def __init__(self):
        super().__init__()
        pass
    
    
    def register_catalog(self, catalog_dataset: SourceCatalogDataset):
        """
        Register a list of source catalogs to the SDTSpectrum instance.
        
        Parameters:
        list_catalog (List[SourceCatalog]): A list of SourceCatalog instances.
        """
        self.catalogs = catalog_dataset
        
#%%
if __name__ == "__main__":
    self = SDTSpectrum()
    target_name = 'T01462'
    B = self.search_catalog(target_name)
# %%
