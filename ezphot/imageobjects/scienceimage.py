#%%
import inspect
import json
import os
from pathlib import Path
from typing import Union
from types import SimpleNamespace
from dataclasses import dataclass, asdict

from astropy.time import Time
from astropy.io import fits

from ezphot.imageobjects import Logger, BaseImage, Mask, Background, Errormap, CalibrationImage

#%%
# === Status Class ===


@dataclass
class StepStatus:
    status: bool = False
    update_time: str = None

    def update(self, status=True):
        self.status = status
        self.update_time = Time.now().isot

    def to_dict(self):
        return asdict(self)
    
class Status:
    """Manages image processing steps with dot-access and timestamp tracking."""

    PROCESS_STEPS = [
        "BIASCOR", "DARKCOR", "FLATCOR",
        "ASTROMETRY", "SCAMP", "ASTROALIGN", "REPROJECT", 
        "BKGSUB", "ZPCALC", "STACK", 'ZPSCALE',
        "SUBTRACT", "PHOTOMETRY", "MASTER"
    ]

    def __init__(self, **kwargs):
        # Initialize all process steps
        self._steps = {}
        for step in self.PROCESS_STEPS:
            value = kwargs.get(step, None)
            if isinstance(value, dict):
                self._steps[step] = {
                    "status": value.get("status", False),
                    "update_time": value.get("update_time", None)
                }
            else:
                self._steps[step] = {
                    "status": False,
                    "update_time": None
                }

    def __getattr__(self, name):
        if '_steps' in self.__dict__ and name in self.__dict__['_steps']:
            return self.__dict__['_steps'][name]
        raise AttributeError(f"'Status' object has no attribute '{name}'")
    
    def __setattr__(self, name, value):
        if name == "_steps":
            super().__setattr__(name, value)
        elif '_steps' in self.__dict__ and name in self.__dict__['_steps']:
            if isinstance(value, dict) and "status" in value:
                self.__dict__['_steps'][name] = value
            else:
                raise ValueError(f"Status for '{name}' must be a dict with 'status' and 'update_time'")
        else:
            super().__setattr__(name, value)

    def update(self, process_name, status: bool = True):
        if process_name in self._steps:
            self._steps[process_name]["status"] = status
            self._steps[process_name]["update_time"] = Time.now().isot
        else:
            raise ValueError(f"Invalid process name: {process_name}")

    def to_dict(self):
        return self._steps

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def __repr__(self):
        lines = [f"{k}: {v}" for k, v in self._steps.items()]
        return "Status ============================================\n  " + "\n  ".join(lines) + "\n==================================================="

class Info:
    """Stores metadata of a FITS image with dot-access."""
    
    INFO_FIELDS = [
        "SAVEPATH", "BIASPATH", "DARKPATH", "FLATPATH", "BKGPATH", "BKGTYPE", "BKRMSPTH", "EMAPPATH", "EMAPTYPE", "MASKPATH", "MASKTYPE",
        "OBSERVATORY", "CCD", "TELKEY", "TELNAME", "OBSDATE", "NAXIS1", "NAXIS2", "PIXELSCALE", 
        "ALTITUDE", "AZIMUTH", "RA", "DEC", "FOVX", "FOVY", "OBJNAME", "IMGTYPE", "FILTER", "BINNING",
        "EXPTIME", "GAIN", "EGAIN", "CRVAL1", "CRVAL2", "SEEING",
        "ELONGATION", "SKYSIG", "SKYVAL", "APER", "ZP", "DEPTH"
    ]

    def __init__(self, **kwargs):
        self._fields = {field: kwargs.get(field, None) for field in self.INFO_FIELDS}

    def __getattr__(self, name):
        if name in self._fields:
            return self._fields[name]
        raise AttributeError(f"'Info' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name == "_fields":
            super().__setattr__(name, value)
        elif name in self._fields:
            self._fields[name] = value
        else:
            raise AttributeError(f"'Info' object has no attribute '{name}'")

    def update(self, key, value):
        if key in self._fields:
            self._fields[key] = value
        else:
            print(f"WARNING: Invalid key: {key}")

    def to_dict(self):
        return self._fields

    @classmethod
    def from_dict(cls, data):
        return cls(**{k: data.get(k) for k in cls.INFO_FIELDS})

    def __repr__(self):
        lines = [f"{k}: {v}" for k, v in self._fields.items()]
        return "Info ============================================\n  " + "\n  ".join(lines) + "\n==================================================="

    
#%%
class ScienceImage(BaseImage):
    """Class representing a science FITS image.

    Inherits from `BaseImage` and includes extended support for tracking processing
    status, managing file paths, saving metadata, and handling associated files
    such as background maps, error maps, masks, and source catalogs.
    """
    
    def __init__(self, path: Union[Path, str], telinfo : dict = None, status: Status = None, load: bool = True):
        """
        Initialize a ScienceImage instance.
        
        Parameters
        ----------
        path : str or Path
            Path to the science FITS image.
        telinfo : dict, optional
            Telescope metadata dictionary.
        status : Status, optional
            Initial status object. If not provided, status is loaded from file or initialized.
        load : bool, optional
            Whether to load status and header upon initialization.

        """
        path = Path(path)  
        super().__init__(path = path, telinfo = telinfo)
        
        # Initialize Status and Info
        self.status = Status()
        self._logger = None
        self._bkgmap = None
        self._bkgrms = None
        self._sourcerms = None
        self._bkgweight = None
        self._srcweight = None
        self._srcmask = None
        self._cat = None
        self._refcat = None
                
        # Initialize or load status
        if load:
            # Load status and info if paths exist
            self.header
            if self.savepath.statuspath is not None:
                if self.savepath.statuspath.exists():
                    self.status = self.load_status()
            else:                
                raise ValueError("WARNING: Status path is not defined. Check the required header keys: OBSERVATORY, TELKEY, OBJNAME, TELNAME, FILTER")
            self._check_status()
        
        if status is not None:
            self.status = status

    def __repr__(self):
        return (
            f"ScienceImage(\n"
            f"  is_exists   = {self.is_exists},\n"
            f"  is_saved    = {self.is_saved},\n"
            f"  data_load   = {self.is_data_loaded},\n"
            f"  header_load = {self.is_header_loaded},\n"
            f"  imgtype     = {self.imgtype},\n"
            f"  exptime     = {self.exptime},\n"
            f"  filter      = {self.filter},\n"
            f"  path        = {self.path},\n"
            f"  savedir     = {self.savedir}\n"
            f")"
        )
        
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
        print(f"Help for {self.__class__.__name__}\n{help_text}\n\nPublic methods:\n" + "\n".join(lines))
        
    def copy(self):
        """
        Return an in-memory deep copy of this ScienceImage instance,
        
        Parameters
        ----------
        None
        
        Returns
        -------
        copied_image : ScienceImage
            A deep copy of the ScienceImage instance.

        """
        from copy import deepcopy

        new_instance = ScienceImage(
            path=self.path,
            telinfo=deepcopy(self.telinfo),
            status=Status.from_dict(self.status.to_dict()),
            load=False
        )

        # Manually copy loaded data and header
        new_instance.data = None if self.data is None else self.data.copy()
        new_instance.header = None if self.header is None else self.header.copy()
        
        # Preserve savedir if manually set
        if hasattr(self, '_savedir') and self._savedir is not None:
            new_instance._savedir = self._savedir

        return new_instance
    
    def write(self, verbose: bool = True):
        """Write ScienceImage data to savepath.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        if self.data is None:
            raise ValueError("Cannot save ScienceImage: data is not registered.")
        if self.savepath.savepath is None:
            raise ValueError("Cannot save ScienceImage: save path is not defined.")
        os.makedirs(self.savepath.savedir, exist_ok=True)
        fits.writeto(self.savepath.savepath, self.data, self.header, overwrite=True)
        self.helper.print(f'Saved: {self.savepath.savepath}', verbose)
        self.save_status()
        self.save_info()
        self.path = self.savepath.savepath  # Update path to saved file
        self.loaded = True
        
    
    def remove(self, 
               remove_main: bool = True, 
               remove_connected_files: bool = True,
               skip_exts: list = [],
               verbose: bool = True) -> dict:
        """
        Remove the main FITS file and/or associated connected files.

        Parameters
        ----------
        remove_main : bool
            If True, remove the main FITS file (self.path)
        remove_connected_files : bool
            If True, remove associated files (status, mask, coadd, etc.)
        skip_exts : list
            List of file extensions to skip (e.g. ['.png', '.cat'])
        verbose : bool
            If True, print removal results

        Returns
        -------
        dict
            {file_path (str): success (bool)} for each file attempted

        """
        removed = {}

        def try_remove(p: Union[str, Path]):
            p = Path(p)
            if p.exists() and p.is_file():
                try:
                    p.unlink()
                    if verbose:
                        print(f"[REMOVE] {p}")
                    return True
                except Exception as e:
                    if verbose:
                        print(f"[FAILED] {p} - {e}")
                    return False
            return False

        # Remove main FITS file
        if remove_main and self.path and self.path.is_file():
            removed[str(self.path)] = try_remove(self.path)

        # Remove connected files
        if remove_connected_files:
            for f in self.connected_files:
                if f.suffix in skip_exts:
                    if verbose:
                        print(f"[SKIP] {f} (skipped due to extension)")
                    continue
                removed[str(f)] = try_remove(f)

        return removed
        
    def calculate_invalidmask(self,
                              threshold_invalid_connection: int = 100000,
                              save: bool = False,
                              verbose: bool = True,
                              visualize: bool = True,
                              save_fig: bool = False
                              ):
        """ 
        Calculate the invalid mask for this ScienceImage.
        The invalid mask is a mask of pixels that are invalid (zero or nan value).
        If save is True, the invalid mask is saved. Then, you can load the invalid mask with `self.invalidmask`.
        
        Parameters
        ----------
        threshold_invalid_connection : int
            The threshold for invalid connection.
        save : bool
            If True, save the invalid mask.
        verbose : bool
            If True, print verbose output.
        visualize : bool
            If True, visualize the invalid mask.
        save_fig : bool
            If True, save the figure of the invalid mask.
            
        Returns
        -------
        target_ivpmask : Mask
            The invalid mask.
        """
        from ezphot.methods import MaskGenerator
        maskgenerator = MaskGenerator()
        target_ivpmask = maskgenerator.mask_invalidpixel(
            target_img = self,
            threshold_invalid_connection= threshold_invalid_connection,
            # Others
            save = save,
            verbose = verbose,
            visualize = visualize,
            save_fig = save_fig
            )
        return target_ivpmask
    
    def calculate_circularmask(self,
                               target_srcmask: Mask = None,
                               x_position: float = None,
                               y_position: float = None,
                               radius_arcsec: float = None,
                               unit = 'coord',
                               save: bool = False,
                               verbose: bool = True,
                               visualize: bool = True,
                               save_fig: bool = False
                               ):
        """
        Calculate the circular mask for this ScienceImage.
        The circular mask is a mask of pixels that are within a circular region.
        If save is True, the circular mask is saved. Then, you can load the circular mask with `self.circularmask`.
        
        Parameters
        ----------
        target_srcmask : Mask
            The source mask.
        x_position : float
            The x position of the center of the circular mask.
        y_position : float
            The y position of the center of the circular mask.
        radius_arcsec : float
            The radius of the circular mask in arcseconds.
        unit : str
            The unit of the x and y position. 'coord' for coordinate, 'pixel' for pixel.
        save : bool 
            If True, save the circular mask.
        verbose : bool
            If True, print verbose output.
        visualize : bool
            If True, visualize the circular mask.
        save_fig : bool
            If True, save the figure of the circular mask.
        
        Returns 
        -------
        target_circularmask : Mask
            The circular mask.
        """
        from ezphot.methods import MaskGenerator
        maskgenerator = MaskGenerator()
        target_sourcemask = maskgenerator.mask_circle(
            target_img = self,
            target_mask = target_srcmask,
            mask_type = 'source',
            x_position = x_position,
            y_position = y_position,
            radius_arcsec = radius_arcsec,
            unit = unit,
            
            # Others
            save = save,
            verbose = verbose,
            visualize = visualize,
            save_fig = save_fig
            )
        return target_sourcemask
        
    
    def calculate_sourcemask(self,
                             target_srcmask: Mask = None,
                             sigma: float = 5.0,
                             mask_radius_factor: float = 3,
                             saturation_level: float = 50000,
                             save: bool = False,
                             verbose: bool = True,
                             visualize: bool = True,
                             save_fig: bool = False
                             ):
        """ 
        Calculate the source mask for this ScienceImage.
        The source mask is a mask of pixels that are sources. Detection is made with global background and background RMS map.
        If save is True, the source mask is saved. Then, you can load the source mask with `self.sourcemask`.
        
        Parameters
        ----------  
        target_srcmask : Mask
            The source mask. 
        sigma : float
            The sigma for the source detection.
        mask_radius_factor : float
            The radius factor for the source detection.
        saturation_level : float
            The saturation level for the source detection.
        save : bool
            If True, save the source mask.
        verbose : bool
            If True, print verbose output.
        visualize : bool
            If True, visualize the source mask.
        save_fig : bool
            If True, save the figure of the source mask.
            
        Returns
        -------
        target_sourcemask : Mask
            The source mask.
        """
        from ezphot.methods import MaskGenerator
        maskgenerator = MaskGenerator()
        target_sourcemask = maskgenerator.mask_sources(
            target_img = self,
            target_mask = target_srcmask,
            sigma = sigma,
            mask_radius_factor = mask_radius_factor,
            saturation_level = saturation_level,

            # Others
            save = save,
            verbose = verbose,
            visualize = visualize,
            save_fig = save_fig
            )
        return target_sourcemask

    def calculate_bkg(self,
                      target_srcmask: Mask = None,
                      target_ivpmask: Mask = None,
                      is_2D_bkg: bool = True,
                      box_size: int = 64,
                      filter_size: int = 3,
                      save: bool = False,
                      verbose: bool = True,
                      visualize: bool = True,
                      save_fig: bool = False):
        """
        Calculate the background map for this ScienceImage.
        The background map is a map of the background level of the image.
        If save is True, the background map is saved. Then, you can load the background map with `self.bkgmap`.
        
        Parameters
        ----------
        target_srcmask : Mask
            The source mask.
        target_ivpmask : Mask
            The invalid mask.
        is_2D_bkg : bool
            If True, use 2D background estimation.
        box_size : int
            The box size for the background estimation.
        filter_size : int
            The filter size for the background estimation.
        save : bool
            If True, save the background map.
        verbose : bool
            If True, print verbose output.
        visualize : bool
            If True, visualize the background map.
        save_fig : bool
            If True, save the figure of the background map.
        
        Returns
        -------
        target_bkg : Background
            The calculated background map.
        """
        from ezphot.methods import BackgroundGenerator
        bkggenerator = BackgroundGenerator()
        target_bkg, _ = bkggenerator.estimate_with_sep(
            target_img = self,
            target_srcmask = target_srcmask,
            target_ivpmask = target_ivpmask,
            is_2D_bkg = is_2D_bkg,
            box_size = box_size,
            filter_size = filter_size,
            save = save,
            verbose = verbose,
            visualize = visualize,
            save_fig = save_fig)

        return target_bkg
    
    def calculate_bkgrms(self, 
                         target_srcmask: Mask = None,
                         target_ivpmask: Mask = None,
                         box_size: int = 64,
                         filter_size: int = 3,
                         save: bool = False,
                         verbose: bool = True,
                         visualize: bool = True,
                         save_fig: bool = False):
        """
        Calculate the background RMS map for this ScienceImage.
        The background RMS map is a map of the background RMS level of the image.
        If save is True, the background RMS map is saved. Then, you can load the background RMS map with `self.bkgrms`.
        
        Parameters
        ----------
        target_srcmask : Mask
            The source mask.
        target_ivpmask : Mask
            The invalid mask.
        box_size : int
            The box size for the background RMS estimation.
        filter_size : int
            The filter size for the background RMS estimation.
        save : bool
            If True, save the background RMS map.
        verbose : bool
            If True, print verbose output.
        visualize : bool
            If True, visualize the background RMS map.
        save_fig : bool
            If True, save the figure of the background RMS map.
        
        Returns
        -------
        target_bkgrms : Errormap
            The background RMS map.
        """
        from ezphot.methods import ErrormapGenerator
        errormapgenerator = ErrormapGenerator()
        target_bkgrms, _, _ = errormapgenerator.calculate_errormap_from_image(
            target_img = self,  
            target_srcmask = target_srcmask,
            target_ivpmask = target_ivpmask,
            box_size = box_size,
            filter_size = filter_size,
            erormap_type = 'bkgrms',
            save = save,
            verbose = verbose,
            visualize = visualize,
            save_fig = save_fig)
        return target_bkgrms
    
    def calculate_bkgrms_from_propagation(self,
                                          target_bkg: Background = None,
                                          mbias: CalibrationImage = None,
                                          mdark: CalibrationImage = None,
                                          mflat: CalibrationImage = None,
                                          mflaterr: Errormap = None,
                                          ncombine: int = 1,
                                          readout_noise: float = None,
                                          save: bool = False,
                                          verbose: bool = True,
                                          visualize: bool = True,
                                          save_fig: bool = False):
        """
        Calculate the background RMS map for this ScienceImage from the background map, bias frame, dark frame, and flat frame.
        The background RMS map is a map of the background RMS level of the image. 
        If save is True, the background RMS map is saved. Then, you can load the background RMS map with `self.bkgrms`.

        Parameters
        ----------
        target_srcmask : Mask
            The source mask.
        target_ivpmask : Mask
            The invalid mask.
        mbias : CalibrationImage
            The bias frame.
        mdark : CalibrationImage
            The dark frame.
        mflat : CalibrationImage
            The flat frame.
        mflaterr : Errormap
            The flat error map.
        ncombine : int
            The number of frames to combine.
        readout_noise : float
            The readout noise.
        save : bool
            If True, save the background RMS map.
        verbose : bool
            If True, print verbose output.
        visualize : bool
            If True, visualize the background RMS map.
        save_fig : bool
            If True, save the figure of the background RMS map.

        Returns
        -------
        target_bkgrms : Errormap
            The background RMS map.
        """
        
        from ezphot.methods import Preprocess
        preprocess = Preprocess()
        # prepare the data
        if target_bkg is None:
            if self.bkgmap is None:
                raise ValueError("Cannot calculate background RMS map: Input background map. OR Register background map with scienceimage.calculate_background(save = True) first.")
            else:
                target_bkg = self.bkgmap
        
        mbias_path, mdark_path, mflat_path = None, None, None
        if mbias is None:
            mbias = preprocess.get_masterframe_from_image(self, 'bias', 30)[0]
        if mdark is None:
            mdark = preprocess.get_masterframe_from_image(self, 'dark', 30)[0]
        if mflat is None:
            mflat = preprocess.get_masterframe_from_image(self, 'flat', 30)[0]
        if mbias is None or mdark is None or mflat is None:
            raise ValueError("Cannot calculate background RMS: required calibration frames are missing.")

        from ezphot.methods import ErrormapGenerator
        errormapgenerator = ErrormapGenerator()
        target_bkgrms = errormapgenerator.calculate_bkgrms_from_propagation(
            target_bkg = target_bkg,
            mbias_img = mbias,
            mdark_img = mdark,
            mflat_img = mflat,
            mflaterr_img = mflaterr,
            ncombine = ncombine,
            readout_noise = readout_noise,
            save = save,
            verbose = verbose,
            visualize = visualize,
            save_fig = save_fig)
                
        return target_bkgrms
        
    def calculate_errormap(self, 
                           target_srcmask: Mask = None,
                           target_ivpmask: Mask = None,
                           box_size: int = 64,
                           filter_size: int = 3,
                           save: bool = False,
                           verbose: bool = True,
                           visualize: bool = True,
                           save_fig: bool = False):
        """
        Calculate the source RMS map for this ScienceImage.
        The source RMS map is a map of the source RMS level of the image.
        If save is True, the source RMS map is saved. Then, you can load the source RMS map with `self.sourcerms`.

        Parameters
        ----------
        target_srcmask : Mask
            The source mask.
        target_ivpmask : Mask
            The invalid mask.
        box_size : int
            The box size for the source RMS estimation.
        filter_size : int
            The filter size for the source RMS estimation.
        save : bool
            If True, save the source RMS map.
        verbose : bool
            If True, print verbose output.
        visualize : bool
            If True, visualize the source RMS map.
        save_fig : bool
            If True, save the figure of the source RMS map.

        Returns
        -------
        target_sourcerms : Errormap
            The source RMS map.
        """
        from ezphot.methods import ErrormapGenerator
        errormapgenerator = ErrormapGenerator()
        target_sourcerms = errormapgenerator.calculate_errormap_from_image(
            target_img = self,  
            target_srcmask = target_srcmask,
            target_ivpmask = target_ivpmask,
            box_size = box_size,
            filter_size = filter_size,
            erormap_type = 'sourcerms',
            save = save,
            verbose = verbose,
            visualize = visualize,
            save_fig = save_fig)
        return target_sourcerms
    
    def calculate_errormap_from_propagation(self,
                                            mbias: CalibrationImage = None,
                                            mdark: CalibrationImage = None,
                                            mflat: CalibrationImage = None,
                                            mflaterr: Errormap = None,
                                            save: bool = False,
                                            verbose: bool = True,
                                            visualize: bool = True,
                                            save_fig: bool = False):
        """
        Calculate the source RMS map for this ScienceImage from the background map, bias frame, dark frame, and flat frame.
        The source RMS map is a map of the source RMS level of the image.
        If save is True, the source RMS map is saved. Then, you can load the source RMS map with `self.sourcerms`.

        Parameters
        ----------
        target_srcmask : Mask
            The source mask.
        target_ivpmask : Mask
            The invalid mask.
        mbias : CalibrationImage
            The bias frame.
        mdark : CalibrationImage
            The dark frame.
        mflat : CalibrationImage
            The flat frame.
        mflaterr : Errormap
            The flat error map.
        save : bool
            If True, save the source RMS map.
        verbose : bool
            If True, print verbose output.
        visualize : bool
            If True, visualize the source RMS map.
        save_fig : bool
            If True, save the figure of the source RMS map.

        Returns
        -------
        target_sourcerms : Errormap
            The source RMS map.
        """
        from ezphot.methods import Preprocess
        preprocess = Preprocess()
        # prepare the data

        mbias_path, mdark_path, mflat_path = None, None, None
        if mbias is None:
            mbias = preprocess.get_masterframe_from_image(self, 'bias', 30)[0]
        if mdark is None:
            mdark = preprocess.get_masterframe_from_image(self, 'dark', 30)[0]
        if mflat is None:
            mflat = preprocess.get_masterframe_from_image(self, 'flat', 30)[0]
        if mbias is None or mdark is None or mflat is None:
            raise ValueError("Cannot calculate source RMS: required calibration frames are missing.")

        from ezphot.methods import ErrormapGenerator
        errormapgenerator = ErrormapGenerator()
        target_sourcerms = errormapgenerator.calculate_sourcerms_from_propagation(
            target_img = self,
            mbias_img = mbias,
            mdark_img = mdark,
            mflat_img = mflat,
            mflaterr_img = mflaterr,
            save = save,
            verbose = verbose,
            visualize = visualize,
            save_fig = save_fig)
            
        return target_sourcerms
        
    def to_referenceimage(self):
        """ Convert this ScienceImage to a ReferenceImage
        
        Parameters
        ----------
        None
        
        Returns
        -------
        ReferenceImage
        """
        from ezphot.imageobjects import ReferenceImage
        referenceimage = ReferenceImage(self.path, telinfo=self.telinfo, load=False)
        referenceimage.data = self.data.copy() if self.data is not None else None
        referenceimage.header = self.header.copy() if self.header is not None else None
        return referenceimage

    def load_status(self):
        """ Load processing status from a JSON file
        
        Parameters
        ----------
        None
        
        Returns
        -------
        status : Status
            Status object loaded from the JSON file.
        
        """
        if self.savepath.statuspath is None:
            raise ValueError("Cannot load ScienceImage status: save path is not defined.")
        with open(self.savepath.statuspath, 'r') as f:
            status_data = json.load(f)
        return Status.from_dict(status_data)

    def save_status(self):
        """ Save processing status to a JSON file
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        if self.savepath.statuspath is None:    
            raise ValueError("Cannot save ScienceImage status: save path is not defined.")    
        with open(self.savepath.statuspath, 'w') as f:
            json.dump(self.status.to_dict(), f, indent=4)

    def update_status(self, process_name):
        """ Mark a process as completed and update time
        
        Parameters
        ----------
        process_name : str
            Name of the process to update.
        """
        self.status.update(process_name)
    
    def save_info(self):
        """ Save processing info to a JSON file
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        if self.savepath.infopath is None:
            raise ValueError("Cannot save ScienceImage info: save path is not defined.")
        with open(self.savepath.infopath, 'w') as f:
            json.dump(self.info.to_dict(), f, indent=4)
        
    @property
    def logger(self):
        if self._logger is None and self.savepath.loggerpath is not None:
            self._logger = Logger(logger_name=str(self.savepath.loggerpath)).log()
        return self._logger
                
    @property
    def info(self):
        """ Information instance of the image. Info is defined in `Info` class.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        info : Info
            Information instance of the image.
        """
        info = Info(
            SAVEPATH = str(self.savepath.savepath), BIASPATH = self.biaspath, DARKPATH = self.darkpath, FLATPATH = self.flatpath, 
            BKGPATH = self.bkgpath, BKGTYPE = self.bkgtype, EMAPPATH = self.emappath, EMAPTYPE = self.emaptype, MASKPATH = self.maskpath, MASKTYPE = self.masktype,
            OBSERVATORY =  self.telinfo['obs'], CCD = self.telinfo['ccd'],
            TELKEY = self.telkey, TELNAME = self.telname, OBSDATE = self.obsdate,
            NAXIS1 = self.naxis1, NAXIS2 = self.naxis2, PIXELSCALE = self.telinfo['pixelscale'],
            ALTITUDE = self.altitude, AZIMUTH = self.azimuth, RA = self.ra, DEC = self.dec, FOVX = self.fovx, FOVY = self.fovy,
            OBJNAME = self.objname, IMGTYPE = self.imgtype, FILTER = self.filter,
            BINNING = self.binning, EXPTIME = self.exptime, GAIN = self.gain)
        header = self.header
        if header is not None:
            for key in info.INFO_FIELDS:
                if key in self._key_variants:
                    key_variants = self._key_variants[key]
                    for variant in key_variants:
                        if variant in header:
                            info.update(key, header[variant])
                        else:
                            pass
        return info
    
    @property
    def savedir(self) -> Union[Path, None]:
        """
        Return the directory where this image and associated files will be saved.
        If a custom savedir was set, use it. Otherwise, build from config and metadata.
        Returns None if required fields are not available.
        """
        # Use manually set savedir if provided
        if hasattr(self, '_savedir') and self._savedir is not None:
            return self._savedir

        # Check required fields
        required_fields = [self.observatory, self.telkey, self.objname, self.telname, self.filter]
        if any(v is None for v in required_fields):
            return self.path.parent  # Return parent directory if any field is missing

        # Default construction from config
        base_dir = Path(self.config['SCIDATA_DIR'])
        return base_dir / self.observatory / self.telkey / self.objname / self.telname / self.filter

    @savedir.setter
    def savedir(self, value: Union[str, Path]):
        """Set a custom directory for saving the image and associated products."""
        if value is None:
            self._savedir = None
            return
        value = Path(value)
        if value.is_file():
            value = value.parent
        self._savedir = value

    @property
    def savepath(self):
        """Dynamically builds save paths based on current header info"""
        savedir = self.savedir
        filename = self.path.name
        return SimpleNamespace(
            savedir=savedir,
            savepath=savedir / filename,
            statuspath=savedir / (filename + '.status'),
            infopath=savedir / (filename + '.info'),
            loggerpath=savedir / (filename + '.log'),
            # Mask
            maskpath=savedir / (filename + '.mask'),
            invalidmaskpath=savedir / (filename + '.invalidmask'),
            srcmaskpath= savedir / (filename + '.srcmask'),
            crmaskpath= savedir / (filename + '.crmask'),
            bpmaskpath= savedir / (filename + '.bpmask'),
            submaskpath= savedir / (filename + '.submask'),
            # Modified images
            alignpath = savedir / ('align_' + filename),
            combinepath = savedir / ('com_' + filename),
            coaddpath = savedir / ('coadd_' + filename),
            scalepath = savedir / ('scale_' + filename),
            convolvepath = savedir / ('conv_' + filename),
            subtractpath = savedir / ('sub_' + filename),
            invertedpath = savedir / ('inv_' + filename),
            # Byproducts
            bkgpath= savedir / (filename + '.bkgmap'),
            bkgrmspath = savedir / (filename + '.bkgrms'),
            srcrmspath = savedir / (filename + '.srcrms'),
            bkgweightpath = savedir / (filename + '.bkgweight'),
            srcweightpath = savedir / (filename + '.srcweight'),
            catalogpath = savedir / (filename + '.cat'),
            psfcatalogpath = savedir / (filename + '.psfcat'),
            refcatalogpath = savedir / (filename + '.refcat'),
            stampcatalogpath = savedir / (filename + '.stampcat')
        )
    
    @property
    def is_saved(self):
        """ Check if the image has been saved """
        if self.savepath.savepath is None:
            return False
        return self.savepath.savepath.exists()
    
    @property
    def connected_files(self):
        """
        Return all associated files that would be deleted in `remove()` if remove_connected_files=True,
        excluding the main FITS file (`self.path`).

        Only includes existing files, not directories.

        Returns
        -------
        connected_files : set
            All connected auxiliary files.
        """
        connected = set()

        # Files in same directory that start with the same base name (excluding self.path)
        base_dir = self.path.parent
        base_name = self.path.name
        for f in base_dir.iterdir():
            if f.is_file() and f.name.startswith(base_name) and f != self.path:
                connected.add(f)
        return connected

    # === Lazy-loaded auxiliary objects ===
    @property
    def bkgmap(self):
        """Background map of the image. If not exists, return None."""
        if self._bkgmap is None and self.savepath.bkgpath.exists():
            from ezphot.imageobjects import Background
            self._bkgmap = Background(self.savepath.bkgpath, load=True)
        return self._bkgmap

    @property
    def bkgrms(self):
        """Background RMS map of the image. If not exists, return None."""
        if self._bkgrms is None and self.savepath.bkgrmspath.exists():
            from ezphot.imageobjects import Errormap
            self._bkgrms = Errormap(self.savepath.bkgrmspath, emaptype='bkgrms', load=True)
        return self._bkgrms

    @property
    def invalidmask(self):
        """Invalid mask of the image. If not exists, return None."""
        if self._invalidmask is None and self.savepath.invalidmaskpath.exists():
            from ezphot.imageobjects import Mask
            self._invalidmask = Mask(self.savepath.invalidmaskpath, masktype='invalid', load=True)
        return self._invalidmask
    
    @property
    def sourcemask(self):
        """Source mask of the image. If not exists, return None."""
        if self._srcmask is None and self.savepath.srcmaskpath.exists():
            from ezphot.imageobjects import Mask
            self._srcmask = Mask(self.savepath.srcmaskpath, masktype='source', load=True)
        return self._srcmask

    @property
    def catalog(self):
        """Source catalog of the image. If not exists, return None."""
        if self._cat is None and self.savepath.catalogpath.exists():
            from ezphot.dataobjects import Catalog
            self._cat = Catalog(self.savepath.catalogpath, catalog_type='all', load=True)
        return self._cat

    @property
    def refcatalog(self):
        """Reference catalog of the image. If not exists, return None."""
        if self._refcat is None and self.savepath.refcatalogpath.exists():
            from ezphot.dataobjects import Catalog
            self._refcat = Catalog(self.savepath.refcatalogpath, catalog_type='reference', load=True)
        return self._refcat

    def _check_status(self):
        """ Update status case as you want! """
        # FOR gppy results
        if str(self.path.name).startswith('calib'):
            self.status.update('BIASCOR')
            self.status.update('DARKCOR')
            self.status.update('FLATCOR')
            self.status.update('ASTROMETRY')
            self.status.update('SCAMP')
            #self.status.update('ZPCALC')        
        if '.com.' in str(self.path.name):
            self.status.update('REPROJECT')
            self.status.update('BKGSUB')
            self.status.update('STACK')
            self.status.update('PHOTOMETRY')
        
        header = self.header
        key_variants = self._key_variants
        for key in key_variants['CTYPE1']:
            if key in header:
                self.status.update('ASTROMETRY')
            
        for key in key_variants['SEEING']:
            if key in header:            
                self.status.update('BIASCOR')
                self.status.update('DARKCOR')
                self.status.update('FLATCOR')
                self.status.update('ASTROMETRY')
                self.status.update('SCAMP')
        
        for key in key_variants['DEPTH']:
            if key in header:
                self.status.update('BIASCOR')
                self.status.update('DARKCOR')
                self.status.update('FLATCOR')
                self.status.update('ASTROMETRY')
                self.status.update('SCAMP')
                #self.status.update('ZPCALC')
