

#%%
import inspect
from typing import Union, Optional

import numpy as np
import numexpr as ne
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ezphot.helper import Helper
from ezphot.methods import BackgroundGenerator
from ezphot.imageobjects import ScienceImage, ReferenceImage, CalibrationImage, Mask, Errormap, Background

#%%
class ErrormapGenerator:
    """
    Method class to generate error maps from science images.
    
    This class provides methods 
    
    1. Calculation of error maps (Background RMS or Total RMS) from science images or background images using propagation of errors from bias, dark, and flat images. 
    
    2. Calculation of error maps from SEP or photutils-based background RMS estimation.
    """
    def __init__(self):
        self.helper = Helper()
        self.backgroundgenerator = BackgroundGenerator()

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
        print(f"Help for {self.__class__.__name__}\n{help_text}\nPublic methods:\n" + "\n".join(lines))

    def calculate_sourcerms_from_propagation(self,
                                             target_img: Union[ScienceImage, ReferenceImage],
                                             mbias_img: CalibrationImage,
                                             mdark_img: CalibrationImage,                             
                                             mflat_img: CalibrationImage,
                                             mflaterr_img: Optional[Errormap] = None,
                                                 
                                             # Others
                                             save: bool = True,
                                             verbose: bool = True,
                                             visualize: bool = True,
                                             save_fig: bool = False,
                                             **kwargs
                                             ):
        """
        Calculate error maps from science images using propagation of errors from bias, dark, and flat images.
        
        Parameters
        ----------
        target_img : ScienceImage or ReferenceImage
            The target image to calculate the error map from.
        mbias_img : CalibrationImage
            The master bias image.
        mdark_img : CalibrationImage
            The master dark image.
        mflat_img : CalibrationImage
            The master flat image.
        mflaterr_img : Errormap, optional
            The master flat error map.
        save : bool, optional
            Whether to save the error map.
        verbose : bool, optional
            Whether to print verbose output.
        visualize : bool, optional
            Whether to visualize the error map.
        save_fig : bool, optional
            Whether to save the error map as a figure.
            
        Returns
        -------
        target_sourcerms : Errormap
            The source RMS error map instance.
        """
        # --- Inputs ---
        data = target_img.data                   # assumed to be calibrated science image in ADU
        ncombine = target_img.ncombine or 1      # number of science images combined to make master science image
        mbias = mbias_img.data                   # bias image in ADU
        ncombine_bias = mbias_img.ncombine or 9  # number of bias images combined to make master bias
        mdark = mdark_img.data                   # dark image in ADU
        ncombine_dark = mdark_img.ncombine or 9  # number of dark images combined to make master dark
        mflat = mflat_img.data                   # normalized flat image (unitless, ~1.0)
        egain = target_img.egain                 # electrons/ADU
        if target_img.ncombine is None:
            self.helper.print('Warning: target_img.ncombine is None. Using 1 as default value.', verbose)
        if mbias_img.ncombine is None:
            self.helper.print('Warning: mbias_img.ncombine is None. Using 9 as default value.', verbose)
        if mdark_img.ncombine is None:
            self.helper.print('Warning: mdark_img.ncombine is None. Using 9 as default value.', verbose)
        
        # --- Readout noise from master bias ---
        ny, nx = mbias.shape
        y0 = ny // 3
        y1 = 2 * ny // 3
        x0 = nx // 3
        x1 = 2 * nx // 3
        central_bias = mbias[y0:y1, x0:x1] # Central region of the bias image
        mbias_var = np.var(central_bias)          # in ADU
        sbias_var = mbias_var * ncombine_bias  # in ADU^2
        readout_noise = np.sqrt(sbias_var)  # Readout noise in ADU
        
        # --- Readout noise from master dark ---
        mdark_var = sbias_var / ncombine_dark + mbias_var

        # 
        if mflaterr_img is not None:
            mflat_err = mflaterr_img.data
            mflat_var = mflat_err**2
            mflaterr_path = str(mflaterr_img.path)
        else:
            mflat_err = 0
            mflat_var = 0
            mflaterr_path = None

        signal = np.abs(data + mdark)
        error_map = ne.evaluate("sqrt((signal / egain / mflat + sbias_var / mflat**2 + signal**2 * mflat_var / mflat**2) + mbias_var + mdark_var)")
        # HERE, mflat**2? or mflat? with signal / egain /
        target_errormap = Errormap(target_img.savepath.srcrmspath, emaptype = 'sourcerms' ,load = False)
        target_errormap.data = error_map
        target_errormap.header = target_img.header

        # Update header
        update_header_kwargs = dict(
            TGTPATH = str(target_img.path),
            BIASPATH = str(mbias_img.path),
            DARKPATH = str(mdark_img.path),
            FLATPATH = str(mflat_img.path),
            EFLTPATH = mflaterr_path,
            )
        target_errormap.header.update(update_header_kwargs)
        
        # Update header of the target image
        update_header_kwargs_image = dict(
            EMAPPATH = str(target_errormap.path),
            )
        target_img.header.update(update_header_kwargs_image)
        
        ## Update status          
        event_details = dict(type = 'sourcerms', readnoise = float(readout_noise), mbias = str(mbias_img.path), mdark = str(mdark_img.path), mflat =str(mflat_img.path), mflaterr = mflaterr_path)
        target_errormap.add_status("error_propagation", **event_details)
        
        if save:
            target_errormap.write(verbose = verbose)
        
        if save_fig or visualize:
            save_path = None
            if save_fig:
                save_path = str(target_errormap.savepath.savepath) + '.png'
            self._visualize(
                target_img = target_img,
                target_errormap = target_errormap,
                target_bkg = None,
                save_path = save_path,
                show = visualize
            )
        
        return target_errormap
    
    def calculate_bkgrms_from_propagation(self,
                                          target_bkg: Background,
                                          mbias_img: CalibrationImage,
                                          mdark_img: CalibrationImage,                             
                                          mflat_img: CalibrationImage,
    
                                          mflaterr_img: Errormap = None,
                                          ncombine: Optional[int] = None,
                                          readout_noise : Optional[float] = None,  # Readout noise in ADU
                                        
                                          # Other parameters
                                          save: bool = False,
                                          verbose: bool = True,
                                          visualize: bool = True,
                                          save_fig: bool = False,
                                          **kwargs
                                          ):  
        """
        Calculate error maps from background images using propagation of errors from bias, dark, and flat images.
        
        Parameters
        ----------
        target_bkg : Background
            The background image to calculate the error map from.
        mbias_img : CalibrationImage
            The master bias image.
        mdark_img : CalibrationImage
            The master dark image.
        mflat_img : CalibrationImage
            The master flat image.
        mflaterr_img : Errormap, optional
            The master flat error map.
        ncombine : int, optional
            The number of science images combined to make master science image.
        readout_noise : float, optional
            The readout noise in ADU.
        save : bool, optional
            Whether to save the error map.
        verbose : bool, optional
            Whether to print verbose output.
        visualize : bool, optional
            Whether to visualize the error map.
        save_fig : bool, optional
            Whether to save the error map as a figure.
            
        Returns
        -------
        target_bkgrms : Errormap
            The background RMS error map instance.
        """
        # --- Inputs ---
        data = target_bkg.data                   # assumed to be calibrated science image in ADU
        if ncombine is None:      # number of science images combined to make master science image
            self.helper.print('Warning: ncombine is None. Using 1 as default value.', verbose)

        mbias = mbias_img.data                   # bias image in ADU
        ncombine_bias = mbias_img.ncombine or 9  # number of bias images combined to make master bias
        mdark = mdark_img.data                   # dark image in ADU
        ncombine_dark = mdark_img.ncombine or 9  # number of dark images combined to make master dark
        mflat = mflat_img.data                   # normalized flat image (unitless, ~1.0)
        egain = target_bkg.egain                 # electrons/ADU
        if mbias_img.ncombine is None:
            self.helper.print('Warning: mbias_img.ncombine is None. Using 9 as default value.', verbose)
        if mdark_img.ncombine is None:
            self.helper.print('Warning: mdark_img.ncombine is None. Using 9 as default value.', verbose)
        
        # --- Readout noise from master bias ---
        ny, nx = mbias.shape
        y0 = ny // 3
        y1 = 2 * ny // 3
        x0 = nx // 3
        x1 = 2 * nx // 3
        central_bias = mbias[y0:y1, x0:x1] # Central region of the bias image
        mbias_var = np.var(central_bias)          # in ADU
        
        if readout_noise is None:
            sbias_var = mbias_var * ncombine_bias  # in ADU^2
            readout_noise = np.sqrt(sbias_var)  # Readout noise in ADU
        else:
            sbias_var = readout_noise **2 # in ADU^2
            pass
        
        # --- Readout noise from master dark ---
        mdark_var = sbias_var / ncombine_dark + mbias_var

        # 
        if mflaterr_img is not None:
            mflat_err = mflaterr_img.data
            mflat_var = mflat_err**2
            mflaterr_path = str(mflaterr_img.path)
        else:
            mflat_err = 0
            mflat_var = 0
            mflaterr_path = None

        signal = np.abs(data + mdark)
        error_map = ne.evaluate("sqrt(signal / egain / mflat + sbias_var / mflat**2 + signal**2 * mflat_var / mflat**2 + mbias_var / mflat**2 + mdark_var / mflat**2)")

        target_errormap = Errormap(str(target_bkg.path).replace('bkgmap','bkgrms'), emaptype = 'bkgrms', load = False)
        target_errormap.data = error_map
        target_errormap.header = target_bkg.header

        # Update header
        update_header_kwargs = dict(
            BKGPATH = str(target_bkg.path),
            BIASPATH = str(mbias_img.path),
            DARKPATH = str(mdark_img.path),
            FLATPATH = str(mflat_img.path),
            EFLTPATH = mflaterr_path,
            )
        target_errormap.header.update(update_header_kwargs)
        
        ## Update status          
        event_details = dict(type = 'sourcerms', readnoise = float(readout_noise), mbias = str(mbias_img.path), mdark = str(mdark_img.path), mflat =str(mflat_img.path), mflaterr = mflaterr_path)
        target_errormap.add_status("error_propagation", **event_details)
        
        if save:
            target_errormap.write(verbose = verbose)

        if save_fig or visualize:
            save_path = None
            if save_fig:
                save_path = str(target_errormap.savepath.savepath) + '.png'
            self._visualize(
                target_img = None,
                target_errormap = target_errormap,
                target_bkg = target_bkg,
                save_path = save_path,
                show = visualize
            )
        
        return target_errormap

    def calculate_errormap_from_image(self,
                                      # Input parameters
                                      target_img: Union[ScienceImage, ReferenceImage],
                                      target_srcmask: Optional[Mask] = None,
                                      target_ivpmask: Optional[Mask] = None,
                                      box_size: int = 128,
                                      filter_size: int = 3,
                                      errormap_type: str = 'bkgrms', # bkgrms or sourcerms

                                      # Others
                                      save: bool = True,
                                      verbose: bool = True,
                                      visualize: bool = True,
                                      save_fig: bool = False,
                                      **kwargs
                                      ):
        """
        Calculate error maps from science images using SEP or photutils-based background RMS estimation.
        
        Parameters
        ----------
        target_img : ScienceImage or ReferenceImage
            The target image to calculate the error map from.
        target_srcmask : Mask, optional
            The source mask to use for the error map calculation.
        target_ivpmask : Mask, optional
            The invalid pixel mask to use for the error map calculation.
        box_size : int, optional
            The size of the box for the background RMS estimation.
        filter_size : int, optional
            The size of the filter for the background RMS estimation.
        errormap_type : str, optional
            The type of error map to calculate. ['bkgrms', 'sourcerms']
        mode : str, optional
            The mode of the error map calculation. ['sep', 'photutils']
        save : bool, optional
            Whether to save the error map.
        verbose : bool, optional
            Whether to print verbose output.
        visualize : bool, optional
            Whether to visualize the error map.
        save_fig : bool, optional
            Whether to save the error map as a figure.
            
        Returns
        -------
        target_errormap : ezphot.imageobjects.Errormap
            The error map instance from ezphot.
        target_bkg : ezphot.imageobjects.Background
            The background image instance from ezphot.
        bkg : sep.Background or photutils.background.Background2D
            The background image instance from SEP or photutils.
        """
        target_bkg, bkg = self.backgroundgenerator.estimate_with_sep(
            target_img = target_img,
            target_srcmask = target_srcmask,
            target_ivpmask = target_ivpmask,
            box_size = box_size,
            filter_size = filter_size,
            save = False,
            verbose = verbose,
            visualize = False,
            save_fig = False
        )

        import numpy as np
        # Calculate error map
        bkg_rms_map = bkg.rms()
        if target_ivpmask is not None:
            bkg_rms_map[target_ivpmask.data] = np.nan
            
        if errormap_type.lower() == 'sourcerms':
            egain = target_img.egain
            bkg_map = target_bkg.data
            source_var_map = np.abs(self.helper.operation.subtract(target_img.data.astype(np.float32), bkg_map)) / egain
            error_map = self.helper.operation.sqrt(self.helper.operation.power(bkg_rms_map,2) + source_var_map)
            target_errormap = Errormap(target_img.savepath.srcrmspath, emaptype = 'sourcerms', load = False)
        else:
            error_map = bkg_rms_map
            target_errormap = Errormap(target_img.savepath.bkgrmspath, emaptype = 'bkgrms', load = False)

        target_errormap.data = error_map
        target_errormap.header = target_img.header

        # Update header
        update_header_kwargs = dict(
            TGTPATH = str(target_img.path),
            BKGPATH = str(target_bkg.path),
            MASKPATH = str(target_bkg.info.MASKPATH),
            )
        target_errormap.header.update(update_header_kwargs)
        
        ## Update status          
        if errormap_type.lower() == 'sourcerms':
            event_details = dict(type = 'sourcerms', bkg_path = str(target_bkg.path), bkg_mask = str(target_bkg.info.MASKPATH), box_size = box_size, filter_size = filter_size)
        else:
            event_details = dict(type = 'bkgrms', bkg_path = str(target_bkg.path), bkg_mask = str(target_bkg.info.MASKPATH), box_size = box_size, filter_size = filter_size)

        target_errormap.add_status("sourcemask", **event_details)
        
        if save:
            target_errormap.write(verbose = verbose)
        
        if save_fig or visualize:
            save_path = None
            if save_fig:
                save_path = str(target_errormap.savepath.savepath) + '.png'
            self._visualize(
                target_img = target_img,
                target_errormap = target_errormap,
                target_bkg = target_bkg,
                save_path = save_path,
                show = visualize
            )
        return target_errormap, target_bkg, bkg
    
    def _visualize(self,
                   target_errormap: Union[Errormap],
                   target_img: Union[ScienceImage, ReferenceImage, CalibrationImage] = None,
                   target_bkg: Union[Background] = None,
                   save_path: str = None,
                   show: bool = False):
        from astropy.visualization import ZScaleInterval
        interval = ZScaleInterval()        

        """
        Visualize the image and mask.
        """
        panels = []
        titles = []
        
        def downsample(data, factor=4):
            return data[::factor, ::factor]
        
        if target_img is not None:
            image_data_small = downsample(target_img.data)
            vmin, vmax = interval.get_limits(image_data_small)
            panels.append((image_data_small, dict(cmap='Greys_r', vmin=vmin, vmax=vmax)))
            titles.append("Original Image")

        if target_bkg is not None:
            bkg_map_small = downsample(target_bkg.data)
            vmin, vmax = interval.get_limits(bkg_map_small)
            panels.append((bkg_map_small, dict(cmap='Greys_r', vmin=vmin, vmax=vmax)))
            titles.append("2D Background")

        error_map_small = downsample(target_errormap.data)
        vmin, vmax = interval.get_limits(error_map_small)
        panels.append((error_map_small, dict(cmap='Greys_r', vmin=vmin, vmax=vmax)))
        titles.append("Error map")
            
        n = len(panels)
        if n == 0:
            self.helper.print("Nothing to visualize.", True)
            return

        fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
        if n == 1:
            axes = [axes]  # make iterable

        for i, (data, imshow_kwargs) in enumerate(panels):
            ax = axes[i]
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            im = ax.imshow(data, origin='lower', **imshow_kwargs)
            ax.set_title(titles[i])
            fig.colorbar(im, cax=cax, orientation='vertical')

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        plt.close(fig)
