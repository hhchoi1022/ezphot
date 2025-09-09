#%%
import inspect
from pathlib import Path
import json
import os
from typing import Union, Dict
from types import SimpleNamespace
from dataclasses import dataclass, asdict

from astropy.time import Time
from astropy.io import fits

from ezphot.imageobjects import Logger, BaseImage
#%%


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
        "BIASCOR", "DARKCOR"
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
    processing status tracking, and handling associated products like logs, masks, and status files.
    This class is used to manage bias or dark frames and is typically used in preprocessing steps
    like bias correction or dark subtraction.

    Parameters
    ----------
    path : str or Path
        Path to the calibration FITS image.
    telinfo : dict, optional
        Telescope metadata dictionary (e.g., observatory name, CCD ID, pixel scale).
    status : Status, optional
        Status object tracking processing completion for each calibration step.
        If not provided, status is initialized or loaded from disk.
    load : bool, optional
        Whether to automatically load header and status from the file at initialization.
    """

    def __init__(self, path: Union[Path, str], telinfo : dict = None, status: Status = None, load: bool = True):
        
        path = Path(path)  
        super().__init__(path = path, telinfo = telinfo)
        
        # Initialize Status and Info
        self.status = Status()
        self._logger = None
        
        # Initialize or load status
        if load:
            # Load status and info if paths exist
            self.header
            if self.savepath.statuspath is not None:
                if self.savepath.statuspath.exists():
                    self.status = self.load_status()
            else:
                raise ValueError("WARNING: Status path is not defined. Check the required header keys: OBSERVATORY, TELKEY, IMGTYPE, TELNAME")
            self._check_status()
        
        if status is not None:
            self.status = status

    def __repr__(self):
        return (
            f"CalibrationImage(\n"
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
        self.save_status()
        self.save_info()
        self.path = self.savepath.savepath  # Update path to saved file
        self.loaded = True
    
    def remove(self, 
               remove_main: bool = True, 
               remove_connected_files: bool = True,
               skip_exts: list = [],
               verbose: bool = False) -> dict:
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
    
    def load_status(self):
        """ Load processing status from a JSON file.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        status : Status
            Status object loaded from the JSON file.
        """
        if self.savepath.statuspath is None:
            raise ValueError("Cannot load CalibrationImage status: save path is not defined.")
        with open(self.savepath.statuspath, 'r') as f:
            status_data = json.load(f)
        return Status.from_dict(status_data)

    def save_status(self):
        """ Save processing status to a JSON file.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        if self.savepath.statuspath is None:    
            raise ValueError("Cannot save CalibrationImage status: save path is not defined.")    
        with open(self.savepath.statuspath, 'w') as f:
            json.dump(self.status.to_dict(), f, indent=4)

    def update_status(self, process_name):
        """ Mark a process as completed and update time.
        
        Parameters
        ----------
        process_name : str
            Name of the process to update.
        """
        self.status.update(process_name)
    
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
        
    @property
    def logger(self):
        if self._logger is None and self.savepath.loggerpath is not None:
            self._logger = Logger(logger_name=str(self.savepath.loggerpath)).log()
        return self._logger
    
    @property
    def info(self):
        """ Register necessary info fields """
        info = Info(
            SAVEPATH = str(self.savepath.savepath), OBSERVATORY =  self.telinfo['obs'], CCD = self.telinfo['ccd'],
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
            statuspath=savedir / (filename + '.status'),
            infopath=savedir / (filename + '.info'),
            bpmaskpath= savedir / (filename + '.bpmask'),
            loggerpath=savedir / (filename + '.log')
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

    def _check_status(self):
        pass
