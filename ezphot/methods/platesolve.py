#%%
import inspect
from typing import Union, Optional, List

from ezphot.error import PlatesolveError
from ezphot.helper import Helper
from ezphot.imageobjects import ScienceImage, ReferenceImage
#%%
class Platesolve:
    """
    Platesolve class for solving astrometry and SCAMP.
    
    This class provides methods 
    
    1. Solving astrometry using Astrometry.net.
    
    2. Solving astrometry using SCAMP.
    """
    
    def __init__(self):
        """
        Initialize the Platesolve class.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        self.helper = Helper()

    def __repr__(self):
        return f"Method class: {self.__class__.__name__}\n For help, use 'help(self)' or `self.help()`."

    def help(self):
        # Get all public methods from the class, excluding `help`
        methods = [
            (name, obj)
            for name, obj in inspect.getmembers(self.__class__, inspect.isfunction)
            if not name.startswith("_") and name != "help"
        ]

        # Build plain text list with parameters
        lines = []
        for name, func in methods:
            sig = inspect.signature(func)
            params = [str(p) for p in sig.parameters.values() if p.name != "self"]
            sig_str = f"({', '.join(params)})" if params else "()"
            lines.append(f"- {name}{sig_str}")

        # Final plain text output
        help_text = ""
        self.helper.print(f"Help for {self.__class__.__name__}\n{help_text}\nPublic methods:\n" + "\n".join(lines), True)

    def solve_astrometry(self,
                         # Input parameters
                         target_img: Union[ScienceImage, ReferenceImage],
                        
                         # Other parameters
                         overwrite: bool = True,
                         verbose: bool = True,
                         **kwargs
                         ):
        """
        Solve astrometry using Astrometry.net.
        
        Parameters
        ----------
        target_img : Union[ScienceImage, ReferenceImage]
            The target image to solve astrometry for.
        overwrite : bool, optional
            Whether to overwrite the existing astrometry solution.
        verbose : bool, optional
            Whether to print verbose output.
            
        Returns
        -------
        output_img : ScienceImage
            The output image with astrometry solution.
        """
        if overwrite:
            target_outpath = target_img.savepath.savepath
        else:
            target_outpath = target_img.savepath.savepath.parent / f'astrometry_{target_img.savepath.savepath.name}'
            
        result, astrometry_output_images = self.helper.run_astrometry(
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
            raise PlatesolveError("Astrometry failed", target_img.path)
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
                    ):
        """
        Solve astrometry using SCAMP.
        
        Parameters
        ----------
        target_img : Union[ScienceImage, ReferenceImage, List[ScienceImage], List[ReferenceImage]]
            The target image(s) to solve astrometry for.
        scamp_sexparams : dict, optional
            The SExtractor parameters for SCAMP.
        scamp_params : dict, optional
            The SCAMP parameters for SCAMP.
        overwrite : bool, optional
            Whether to overwrite the existing astrometry solution.
        verbose : bool, optional
            Whether to print verbose output.
            
        Returns 
        -------
        output_imglist : List[ScienceImage]
            The output image(s) with astrometry solution.
        """
        target_imglist = target_img if isinstance(target_img, list) else [target_img]
        target_imglist_path = [target_img.path for target_img in target_imglist]
        
        if overwrite:
            output_dir = target_imglist[0].savepath.savedir
        else:
            output_dir = target_imglist[0].path.parent
            
        scamp_results, scamp_output_images = self.helper.run_scamp(
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
            raise PlatesolveError(f"SCAMP failed for {target_imglist_path}")
        else:    
            output_imglist = []
            for target_img, output_path in zip(target_imglist, scamp_output_images):
                output_img = type(target_imglist[0])(path = output_path, telinfo = target_img.telinfo, status = target_img.status, load = True)
                output_img.update_status('SCAMP')
                output_imglist.append(output_img)
            return output_imglist
    