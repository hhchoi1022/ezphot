#%%
import os
from tippy.helper import Helper
from tippy.image import ReferenceImage

#%%
class SMSSReference(ReferenceImage):
    """ Handles FITS image processing and tracks its status """

    def __init__(self, path: str, telinfo : dict):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")
        
        super().__init__(path = path, telinfo = telinfo)
        self._format_header()
        
    def __repr__(self):
        return f"SDTReference(type = {self.imgtype}, exptime = {self.exptime}, filter = {self.filter}, binning = {self.binning}, gain = {self.gain}, path = {os.path.basename(self.path)})"
    
    def _format_header(self):
        """ There is noting to be formatted in 7DT data """
        #self.update_info()
        pass
    
    def _format_info(self):
        pass
# %%
if __name__ == '__main__':
    image = '/home/hhchoi1022/data/refdata/main/Skymapper/Skymapper_DR1/T01462/SMSS/r/SkyMapper_r_20180622113232-22_232.255-67.924_326x612.fits'
    
    
    C = SMSSReference(image, telinfo = Helper().get_telinfo('7DT', 'C361K', 'HIGH', 1))

# %%
