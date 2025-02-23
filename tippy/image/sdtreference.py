#%%
import os
from tippy.image import ReferenceImage
from tippy.helper import Helper
#%%
class SDTReference(ReferenceImage):
    """ Handles FITS image processing and tracks its status """

    def __init__(self, path: str, telinfo : dict):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")
        
        super().__init__(path = path, telinfo = telinfo)
        self._format_info()
        
    def __repr__(self):
        return f"SDTReference(objname = {self.objname}(RA = %.4f, Dec = %.4f), fov = %.2fx%.2fdeg filter = {self.filter}, pixelscale = {self.telinfo['pixelscale']}, path = {os.path.basename(self.path)})"%(self.ra, self.dec, self.naxis1 * self.telinfo['pixelscale'] / 3600, self.naxis2 * self.telinfo['pixelscale'] / 3600)
    
    def _format_info(self):
        """ There is noting to be formatted in 7DT data """
        #self.update_info()
        pass
    
# %%
if __name__ == '__main__':
    sci_image = '/data/hhchoi1022/refdata/7DT/7DT_C361K_HIGH_1x1/T11623/calib_7DT05_T11623_20240423_020143_r_360.com.fits'
    ref_image = '/data/hhchoi1022/refdata/7DT/7DT_C361K_HIGH_1x1/T11623/calib_7DT09_T11623_20250110_032334_r_360.com.fits'
    from tippy.image import ScienceImage
    C = SDTReference(ref_image, telinfo = Helper().get_telinfo('7DT', 'C361K', 'HIGH', 1))

# %%
