#%%
import inspect
from pathlib import Path
import json
import os
from typing import Union, Dict
from types import SimpleNamespace
import fnmatch


from astropy.time import Time
from astropy.io import fits

from ezphot.imageobjects import BaseImage#, Logger
#%%


class Info:
    INFO_FIELDS = [
        "SAVEPATH", "BIASPATH", "DARKPATH", "OBSERVATORY", 
        "CCD", "TELKEY", "TELNAME", "OBSDATE", "NAXIS1", "NAXIS2", "PIXELSCALE", 
        "OBJNAME", "IMGTYPE", "FILTER", "BINNING", "EXPTIME", "GAIN", "EGAIN"
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
            
    def copy(self):
        return Info.from_dict(self.to_dict())

    def to_dict(self):
        return self._fields

    @classmethod
    def from_dict(cls, data):
        return cls(**{k: data.get(k) for k in cls.INFO_FIELDS})

    def __repr__(self):
        lines = [f"{k}: {v}" for k, v in self._fields.items()]
        return "Info ============================================\n  " + "\n  ".join(lines) + "\n==================================================="

#%%
class CalibrationImage(BaseImage):
    """
    Class representing a calibration FITS image (e.g., bias or dark frame).

    Inherits from `BaseImage` and provides methods for managing calibration image metadata,
    handling associated products like logs, masks.
    This class is used to manage bias or dark frames and is typically used in preprocessing steps
    like bias correction or dark subtraction.

    Parameters
    ----------
    path : str or Path
        Path to the calibration FITS image.
    telinfo : dict, optional
        Telescope metadata dictionary (e.g., observatory name, CCD ID, pixel scale).
    load : bool, optional
        Whether to load the header upon initialization.
    """

    def __init__(self, path: Union[Path, str], telinfo : dict = None, load: bool = True):
        
        path = Path(path)  
        super().__init__(path = path, telinfo = telinfo)
        
        # Initialize Info
        # self._logger = None
        
        if load:
            self.header

    def __repr__(self):
        return (
            f"CalibrationImage(\n"
            f"  is_exists   = {self.is_exists},\n"
            f"  is_saved    = {self.is_saved},\n"
            f"  data_load   = {self.is_data_loaded},\n"
            f"  header_load = {self.is_header_loaded},\n"
            f"  imgtype     = {self.imgtype},\n"
            f"  exptime     = {self.totexp},\n"
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
        Make a deep copy of the CalibrationImage instance.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        new_instance : CalibrationImage
            A deep copy of the CalibrationImage instance.
        """
        from copy import deepcopy

        new_instance = CalibrationImage(
            path=self.path,
            telinfo=deepcopy(self.telinfo),
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
        """Write CalibrationImage data to savepath.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        if self.data is None:
            raise ValueError("Cannot save CalibrationImage: data is not registered.")
        if self.savepath.savepath is None:
            raise ValueError("Cannot save CalibrationImage: save path is not defined.")
        os.makedirs(self.savepath.savedir, exist_ok=True)
        fits.writeto(self.savepath.savepath, self.data, self.header, overwrite=True)
        self.helper.print(f'Saved: {self.savepath.savepath}', verbose)
        self.save_info()
        self.path = self.savepath.savepath  # Update path to saved file
        self.loaded = True

    def remove(self, 
            remove_main: bool = True, 
            remove_connected_files: bool = True,
            skip_patterns: list = [],
            verbose: bool = True) -> dict:

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

        def should_skip(p: Path):
            for pattern in skip_patterns:
                if fnmatch.fnmatch(p.name, pattern):
                    return True
            return False

        # Remove main FITS file
        if remove_main and self.path and self.path.is_file():
            removed[str(self.path)] = try_remove(self.path)

        # Remove connected files
        if remove_connected_files:
            for f in self.connected_files:

                if should_skip(f):
                    if verbose:
                        print(f"[SKIP] {f} (matched skip pattern)")
                    continue

                removed[str(f)] = try_remove(f)

        return removed
    
    def save_info(self):
        """ Save processing info to a JSON file.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        if self.savepath.infopath is None:
            raise ValueError("Cannot save CalibrationImage info: save path is not defined.")
        with open(self.savepath.infopath, 'w') as f:
            json.dump(self.info.to_dict(), f, indent=4)
        
    # @property
    # def logger(self):
    #     if self._logger is None and self.savepath.loggerpath is not None:
    #         self._logger = Logger(logger_name=str(self.savepath.loggerpath)).log()
    #     return self._logger
    
    @property
    def info(self):
        """ Register necessary info fields """
        info = Info(
            SAVEPATH = str(self.savepath.savepath), OBSERVATORY =  self.observatory, CCD = self.ccd,
            TELKEY = self.telkey, TELNAME = self.telname, OBSDATE = self.obsdate,
            NAXIS1 = self.naxis1, NAXIS2 = self.naxis2, PIXELSCALE = self.telinfo['pixelscale'],
            OBJNAME = self.objname, IMGTYPE = self.imgtype, FILTER = self.filter,
            BINNING = self.binning, EXPTIME = self.exptime, GAIN = self.gain, EGAIN = self.egain)
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
        required_fields = [self.observatory, self.telkey, self.imgtype, self.telname]
        if any(v is None for v in required_fields):
            return self.path.parent  # Return parent directory if any field is missing

        # Default construction from config
        base_dir = Path(self.config['CALIBDATA_DIR'])
        return base_dir / self.observatory / self.telkey / self.imgtype / self.telname

    @savedir.setter
    def savedir(self, value: Union[str, Path]):
        """
        Set a custom directory for saving the image and associated products.
        """
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
            infopath=savedir / (filename + '.info'),
            bpmaskpath= savedir / (filename + '.bpmask'),
            # loggerpath=savedir / (filename + '.log')
        )
        
    @property
    def is_saved(self):
        """ Check if the image has been saved """
        if self.savepath.savepath is None:
            return False
        return self.savepath.savepath.exists()
    
    @property
    def connected_files(self) -> set:
        """
        Return all associated files that would be deleted in `remove()` if remove_connected_files=True,
        excluding the main FITS file (`self.path`).

        Only includes existing files, not directories.

        Returns
        -------
        set of Path: All connected auxiliary files.
        """
        connected = set()

        # Files in same directory that start with the same base name (excluding self.path)
        base_dir = self.path.parent
        base_name = self.path.name
        for f in base_dir.iterdir():
            if f.is_file() and f.name.startswith(base_name) and f != self.path:
                connected.add(f)

        # Files explicitly listed in savepath (excluding self.path)
        for p in vars(self.savepath).values():
            if isinstance(p, Path) and p.exists() and p.is_file() and p != self.path:
                connected.add(p)

        return connected

