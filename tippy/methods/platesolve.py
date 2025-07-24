#%%
from typing import Union, Optional, List
import numpy as np

from tippy.error import PlateSolveError
from tippy.helper import Helper
from tippy.imageojbects import ScienceImage, ReferenceImage
#%%
class TIPPlateSolve(Helper): ############## CHECKED ##############
    
    def __init__(self):
        super().__init__()
        
    def solve_astrometry(self,
                         # Input parameters
                         target_img: Union[ScienceImage, ReferenceImage],
                        
                         # Other parameters
                         overwrite: bool = True,
                         verbose: bool = True,
                         **kwargs
                         ):
        if overwrite:
            target_outpath = target_img.savepath.savepath
        else:
            target_outpath = target_img.savepath.savepath.parent / f'astrometry_{target_img.savepath.savepath.name}'
            
        result, astrometry_output_images = self.run_astrometry(
            target_path = target_img.path,
            astrometry_sexconfigfile = target_img.config['ASTROMETRY_SEXCONFIG'],
            ra = target_img.ra,
            dec = target_img.dec,
            radius = max(target_img.fovx, target_img.fovy) / 2,
            pixelscale = target_img.telinfo['pixelscale'],
            target_outpath = target_outpath,
            verbose = verbose,
        )
        if not result:
            raise PlateSolveError("Astrometry failed", target_img.path)
        else:
            output_img = type(target_img)(path = astrometry_output_images, telinfo = target_img.telinfo, status = target_img.status, load = True)
            output_img.update_status('ASTROMETRY')
            return output_img
        
    def solve_scamp(self,
                    # Input parameters
                    target_img: Optional[Union[ScienceImage, ReferenceImage, List[ScienceImage], List[ReferenceImage]]],
                    scamp_sexparams: dict = None,
                    scamp_params: dict = None,
                    # Other parameters
                    overwrite: bool = True,
                    verbose: bool = True,
                    **kwargs
                    ):
        target_imglist = target_img if isinstance(target_img, list) else [target_img]
        target_imglist_path = [target_img.path for target_img in target_imglist]
        
        if overwrite:
            output_dir = target_imglist[0].savepath.savedir
        else:
            output_dir = target_imglist[0].path.parent
            
        scamp_results, scamp_output_images = self.run_scamp(
            target_path = target_imglist_path,
            scamp_sexconfigfile = target_imglist[0].config['SCAMP_SEXCONFIG'],
            scamp_configfile = target_imglist[0].config['SCAMP_CONFIG'],
            scamp_sexparams = scamp_sexparams,
            scamp_params = scamp_params,
            output_dir = output_dir,
            overwrite = overwrite,
            verbose = verbose,            
        )
        
        if not all(scamp_results):
            raise PlateSolveError(f"SCAMP failed for {target_imglist_path}")
        else:    
            output_imglist = []
            for target_img, output_path in zip(target_imglist, scamp_output_images):
                output_img = type(target_imglist[0])(path = output_path, telinfo = target_img.telinfo, status = target_img.status, load = True)
                output_img.update_status('SCAMP')
                output_imglist.append(output_img)
            return output_imglist
        
# %%
if __name__ == '__main__':
    # Example usage
    import glob
    self = TIPPlateSolve()
    scamp_sexparams: dict = None
    sex_params: dict = None
    scamp_params: dict = None
    # Other parameters
    overwrite: bool = True
    verbose: bool = True
    #filelist = glob.glob('/home/hhchoi1022/data/scidata/7DT/7DT_C361K_HIGH_1x1/T22956/7DT13/m875/*.fits')
    target_path = '/data/data1/factory_hhchoi/data/scidata/7DT/7DT_C361K_HIGH_1x1/T22956/7DT14/m600/align_calib_7DT14_T22956_20250424_031813_m600_100.fits'
    target_img = ScienceImage(path = target_path, telinfo = Helper().get_telinfo('7DT', 'C361K', 'HIGH', 1), load = True)
                  #ScienceImage(path = filelist[1], telinfo = Helper().get_telinfo('7DT', 'C361K', 'HIGH', 1), load = True)]
    #target_image_output = self.solve_astrometry(target_img = target_img, overwrite = False, verbose = True)
    # target_imglist = self.solve_scamp(
    #     target_img = target_img,
    #     scamp_sexparams = scamp_sexparams,
    #     scamp_params = scamp_params,
    #     # Other parameters
    #     overwrite = False,
    #     verbose = verbose
    # )
# %%
