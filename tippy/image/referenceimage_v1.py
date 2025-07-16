#%%
from astropy.time import Time
from astropy.io import fits
import json
import os
from pathlib import Path
from typing import Union
from types import SimpleNamespace
from dataclasses import dataclass, asdict
from typing import Dict


from tippy.configuration import TIPConfig
from tippy.helper import Helper
from tippy.image import Logger
from tippy.image import BaseImage


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
        "SUBTRACT", "PHOTOMETRY"
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
class ReferenceImage(BaseImage):
    """ Handles FITS image processing and tracks its status """

    def __init__(self, path: Union[Path, str], telinfo : dict, status: Status = None, load: bool = True):
        path = Path(path)
        super().__init__(path = path, telinfo = telinfo)

        # Initialize Status and Info
        self.status = Status()
        self._logger = None
        
        # Initialize or load status
        if load:
            # Load status and info if paths exist
            self.load_header_from_path()
            if self.savepath.statuspath is not None:
                if self.savepath.statuspath.exists():
                    self.status = self.load_status()
                    self.logger.info(f"Status loaded from {self.savepath.statuspath}")
            else:                
                raise ValueError("WARNING: Status path is not defined. Check the required header keys: OBSERVATORY, TELKEY, OBJNAME, TELNAME, FILTER")

        
        if status is not None:
            self.status = status
        self._check_status()
        
    def __repr__(self):
        return f"ReferenceImage[is_exists = {self.is_exists}, data = {self.is_data_loaded}, header = {self.is_header_exists}, imgtype = {self.imgtype}, exptime = {self.exptime}, filter = {self.filter}, path = {self.path})"
    

    def copy(self) -> "ReferenceImage":
        """
        Return an in-memory deep copy of this ReferenceImage instance,

        """
        from copy import deepcopy

        new_instance = ReferenceImage(
            path=self.path,
            telinfo=deepcopy(self.telinfo),
            status=Status.from_dict(self.status.to_dict()),
            load=False
        )

        # Manually copy loaded data and header
        new_instance.data = None if self.data is None else self.data.copy()
        new_instance.header = None if self.header is None else self.header.copy()

        return new_instance
    
    @property
    def logger(self):
        if self._logger is None and self.savepath.loggerpath is not None:
            self._logger = Logger(logger_name=str(self.savepath.loggerpath)).log()
        return self._logger
                
    @property
    def info(self):
        """ Register necessary info fields """
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
                if key in self.key_variants:
                    key_variants = self.key_variants[key]
                    for variant in key_variants:
                        if variant in header:
                            info.update(key, header[variant])
                        else:
                            pass
        return info

    @property
    def savepath(self):
        """Dynamically builds save paths based on current header info"""
        required_fields = [self.observatory, self.telkey, self.objname, self.telname, self.filter]
        if any(v is None for v in required_fields):
            return SimpleNamespace(
                savedir=None,
                savepath=None,
                statuspath=None,
                infopath=None,
                loggerpath=None,
                # Mask
                maskpath=None,
                invalidmaskpath=None,
                srcmaskpath=None,
                crmaskpath=None,
                bpmaskpath=None,
                submaskpath=None,
                # Modified images
                cutoutpath=None,
                alignpath=None,
                combinepath=None,
                coaddpath=None,
                scalepath=None,
                convolvepath=None,
                subtractpath=None,
                invertedpath=None,
                # Byproducts
                bkgpath=None,
                bkgrmspath=None,
                srcrmspath=None,
                bkgweightpath=None,
                srcweightpath=None,
                catalogpath=None,
                psfcatalogpath=None,
                refcatalogpath=None,
                starcatalogpath=None,
                stampcatalogpath=None
            )

        base_dir = Path(self.config['REFDATA_DIR'])
        savedir = base_dir / self.observatory / self.telkey / self.objname / self.telname / self.filter
        savedir.mkdir(parents=True, exist_ok=True)

        filename = self.path.name
        return SimpleNamespace(
            savedir=savedir,
            savepath=savedir / filename,
            statuspath=savedir / (filename + '.status'),
            infopath=savedir / (filename + '.info'),
            loggerpath=savedir / (filename + '.log'),
            # Mask
            maskpath=savedir / (filename + '.mask'),
            invalidmaskpath= savedir / (filename + '.invalidmask'),
            srcmaskpath= savedir / (filename + '.srcmask'),
            crmaskpath= savedir / (filename + '.crmask'),
            bpmaskpath= savedir / (filename + '.bpmask'),
            submaskpath= savedir / (filename + '.submask'),
            # Modified images
            cutoutpath = savedir / ('cut_' + filename),
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
            starcatalogpath = savedir / (filename + '.starcat'),
            stampcatalogpath = savedir / (filename + '.stampcat'),
        )
    
    @property
    def is_saved(self):
        """ Check if the image has been saved """
        if self.savepath.savepath is None:
            return False
        return self.savepath.savepath.exists()

    def write(self):
        """Write ScienceImage data to FITS file."""
        if self.data is None:
            raise ValueError("Cannot save ScienceImage: data is not registered.")
        if self.savepath.savepath is None:
            raise ValueError("Cannot save ScienceImage: save path is not defined.")
        os.makedirs(self.savepath.savedir, exist_ok=True)
        fits.writeto(self.savepath.savepath, self.data, self.header, overwrite=True)
        self.save_status()
        self.save_info()
        self.loaded = True
    
    def remove(self, 
               remove_all: bool = True, 
               skip_exts: list =  ['.png', '.cat'],
               verbose: bool = False) -> dict:
        """
        Remove files associated with this ScienceImage.

        Parameters:
        - remove_all (bool): If True, remove all related files (FITS, .status, .info, .mask, etc.)
                            If False, remove only the FITS file.
        - verbose (bool): Print/log removed files.

        Returns:
        - dict: {filename (str): success (bool)} for each attempted removal
        """
        removed = {}

        def try_remove(p: Union[str, Path]):
            p = Path(p)
            if p.exists():
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

        # Remove the main FITS file
        if self.path and self.path.is_file():
            removed[str(self.path)] = try_remove(self.path)

        # Remove other savepath-related files (excluding dirs)
        if remove_all:
            for attr in vars(self.savepath).values():
                if isinstance(attr, Path) and attr != self.path and attr.is_file():
                    if attr.suffix in skip_exts:
                        if verbose:
                            print(f"[SKIP] {attr} (skipped due to extension)")
                        continue
                    removed[str(attr)] = try_remove(attr)

        return removed

    def load_status(self):
        """ Load processing status from a JSON file """
        if self.savepath.statuspath is None:
            raise ValueError("Cannot load ScienceImage status: save path is not defined.")
        with open(self.savepath.statuspath, 'r') as f:
            status_data = json.load(f)
        return Status.from_dict(status_data)

    def save_status(self):
        """ Save processing status to a JSON file """
        if self.savepath.statuspath is None:    
            raise ValueError("Cannot save ScienceImage status: save path is not defined.")    
        with open(self.savepath.statuspath, 'w') as f:
            json.dump(self.status.to_dict(), f, indent=4)

    def update_status(self, process_name):
        """ Mark a process as completed and update time """
        self.status.update(process_name)
    
    def save_info(self):
        """ Save processing info to a JSON file """
        if self.savepath.infopath is None:
            raise ValueError("Cannot save ScienceImage info: save path is not defined.")
        with open(self.savepath.infopath, 'w') as f:
            json.dump(self.info.to_dict(), f, indent=4)
    
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
        key_variants = self.key_variants
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

        #self.save_status() 


# %%
if __name__ == '__main__':
    image = '/home/hhchoi1022/data/refdata/7DT/7DT_C361K_HIGH_1x1/T11623/calib_7DT05_T11623_20240423_020143_r_360.com.fits'
    C = ReferenceImage(image, telinfo = Helper().get_telinfo('7DT', 'C361K', 'HIGH', 1))

# %%
