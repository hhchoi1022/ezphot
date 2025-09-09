#%%
import inspect
import os
import json
from pathlib import Path
from typing import Union
from types import SimpleNamespace

import numpy as np
from numba import jit
from astropy.io import fits
from astropy.time import Time

from ezphot.imageobjects import DummyImage, Logger

#%%
class Status:
    """Tracks multiple mask processing steps, including multiple source masks."""

    def __init__(self, status=None):
        # Initialize status dictionary directly from input if provided, else empty
        self.status = status if status is not None else {}
        
    def add_event(self, event_name, **event_details):
        dict_event = dict(update_time=Time.now().isot, **event_details)

        original_event_name = event_name
        counter = 1
        while event_name in self.status:
            event_name = f"{original_event_name}_{counter}"
            counter += 1
        self.status[event_name] = dict_event
    
    def remove_event(self, event_name):
        if event_name in self.status:
            del self.status[event_name]
        else:
            print(f'WARNING: Event not found: {event_name}')

    def to_dict(self):
        return self.status

    @classmethod
    def from_dict(cls, data):
        return cls(status = data)
    
    def __repr__(self):
        """ Represent process status as a readable string """
        status_list = [f"{key} = {value['update_time']}" for key, value in self.status.items()]
        return "Status ============================================\n  " + "\n  ".join(status_list) + "\n==================================================="

class Info:
    """Stores metadata for the mask with dot-access support."""

    INFO_FIELDS = ["TGTPATH", "MASKTYPE", "MASKATMP"]
    DEFAULT_VALUES = ["", "", 0]

    def __init__(self, **kwargs):
        # Set defaults, then override with user-provided values
        self._fields = {
            field: kwargs.get(field, default)
            for field, default in zip(self.INFO_FIELDS, self.DEFAULT_VALUES)
        }
        
    def __getattr__(self, name):
        # Prevent infinite recursion when _fields is not yet initialized
        if '_fields' in self.__dict__ and name in self._fields:
            return self._fields[name]
        raise AttributeError(f"'Info' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name == "_fields":
            super().__setattr__(name, value)
        elif "_fields" in self.__dict__ and name in self._fields:
            self._fields[name] = value
        else:
            raise AttributeError(f"'Info' object has no attribute '{name}'")

    def update(self, key, value):
        if key in self._fields:
            self._fields[key] = value
        else:
            print(f"WARNING: Invalid key: {key}")

    def to_dict(self):
        return dict(self._fields)

    @classmethod
    def from_dict(cls, data):
        return cls(**{key: data.get(key) for key in cls.INFO_FIELDS})

    def __repr__(self):
        lines = [f"{key}: {value}" for key, value in self._fields.items()]
        return "Info ============================================\n  " + "\n  ".join(lines) + "\n==================================================="

import time
class Mask(DummyImage):
    """
    Class representing a mask image for astronomical data processing.
    
    Inherits from `DummyImage` and provides methods for creating, combining, and managing
    various types of masks including source masks, bad pixel masks, and cosmic ray masks.
    
    Parameters
    ----------
    path : str or Path
        Path to the mask FITS image.
    masktype : str, optional
        Type of mask ('none', 'invalid', 'source', 'badpixel', 'cosmicray', 'subtraction').
    status : Status, optional
        Initial status object. If not provided, status is loaded from file or initialized.
    load : bool, optional
        Whether to load status and header upon initialization.
    """

    def __init__(self, path: Union[str, Path], masktype: str = 'None', status: Status = None, load: bool = False):
        path = Path(path)
        super().__init__(path=path)

        self.masktype = masktype.lower()
        if self.masktype not in ['none', 'invalid', 'source', 'badpixel', 'cosmicray', 'subtraction']:
            raise ValueError(f"Invalid mask type: {self.masktype}. Must be 'source' or 'badpixel' or 'subtraction'.")
        
        self.header.update(dict(MASKTYPE=self.masktype))
        
        # Initialize Status and Info
        self.status = Status()
        self._logger = None
        self._target_img = None

        if load:
            # Load status and info if paths exist
            self.header
            if self.savepath.statuspath.exists():
                self.status = self.load_status()
        if status is not None:
            self.status = status
        
    def __repr__(self):
        return (
            f"Mask(\n"
            f"  is_exists   = {self.is_exists},\n"
            f"  is_saved    = {self.is_saved},\n"
            f"  data_load   = {self.is_data_loaded},\n"
            f"  header_load = {self.is_header_loaded},\n"
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
        
    def copy(self) -> "Mask":
        """
        Return an in-memory deep copy of this Mask instance,
        
        Parameters
        ----------
        None
        
        Returns
        -------
        copied_image : Mask
            A deep copy of the Mask instance.

        """
        from copy import deepcopy

        new_instance = Mask(
            path=self.path,
            masktype=self.masktype,
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
 
    def write(self, compress : bool = False, verbose: bool = True):
        """Write MaskImage data to savepath.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        if self.data is None:
            raise ValueError("Cannot save MaskImage: data is not registered.")
        os.makedirs(self.savepath.savedir, exist_ok=True)
        if compress:
            # Save as compressed image (NOT compatible with SExtractor)
            hdu = fits.CompImageHDU(data=self.data.astype(np.int16),
                                    header=self.header,
                                    compression_type='RICE_1')
        else:
            # ? Save as a single-extension standard FITS image
            hdu = fits.PrimaryHDU(data=self.data.astype(np.int16),
                                header=self.header)

        # Write to disk
        hdu.writeto(self.savepath.savepath, overwrite=True)
        self.helper.print(f'Saved: {self.savepath.savepath}', verbose)
        self.save_status()
        self.save_info()
        self.path = self.savepath.savepath
        self.loaded = True
        
    def remove(self, 
            remove_main: bool = True, 
            remove_connected_files: bool = True,
            skip_exts: list = ['.png', '.cat'],
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
            raise ValueError("Cannot load MaskImage status: save path is not defined.")
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
            raise ValueError("Cannot save MaskImage status: save path is not defined.")    
        with open(self.savepath.statuspath, 'w') as f:
            json.dump(self.status.to_dict(), f, indent=4)

    def add_status(self, event_name, **event_details):
        """ Add a status event to the mask.
        
        Parameters
        ----------
        event_name : str
            Name of the event to add.
        event_details : dict
            Details of the event to add.
        """
        self.status.add_event(event_name, **event_details)

    def save_info(self):
        """Save processing info to JSON file.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        with open(self.savepath.infopath, 'w') as f:
            json.dump(self.info.to_dict(), f, indent=4)
            
    def combine_mask(self, new_mask: Union[str, Path, np.ndarray], operation='or'):
        """ Combine the current mask with a new mask.
        
        Parameters
        ----------
        new_mask : str or Path or np.ndarray
            Path to the new mask FITS image or numpy array.
        operation : str, optional
            Operation to combine the current mask with the new mask.
        """
        # Avoid loading if already ndarray
        if isinstance(new_mask, (str, Path)):
            new_mask = self.load_fits(new_mask)[0]

        @jit(nopython=False)
        def combine_two_mask(mask1, mask2):
            if operation == 'or':
                return np.logical_or(mask1, mask2).astype(np.int8)
            elif operation == 'and':
                return np.logical_and(mask1, mask2).astype(np.int8)
            elif operation == 'add':
                return (mask1 + mask2).astype(np.int8)
            elif operation == 'subtract':
                return np.clip(mask1 - mask2, 0, None).astype(np.int8)
            else:
                raise ValueError(f"Invalid operation: {operation}. Use 'or', 'and', 'add', or 'subtract'.")

        if self.data is None:
            self.data = new_mask.copy().astype(np.int8)
        else:
            mask1 = self.data.astype(np.int8)
            mask2 = new_mask.astype(np.int8)
            self.data = combine_two_mask(mask1, mask2)
            
    @property
    def logger(self):
        if self._logger is None and self.savepath.loggerpath is not None:
            self._logger = Logger(logger_name=str(self.savepath.loggerpath)).log()
        return self._logger

    @property
    def info(self):
        """ Information instance of the mask. Info is defined in `Info` class. 
        
        Parameters
        ----------
        None
        
        Returns
        -------
        info : Info
            Information instance of the mask.
        """
        info = Info()
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

        # Default construction from config
        base_dir = self.path.parent
        return base_dir 
    
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
            targetpath=(savedir / filename).with_suffix(''),
            statuspath=savedir / (filename + '.status'),
            infopath=savedir / (filename + '.info'),
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
        excluding the main FITS file (`self.path`) and the `targetpath`.
        Only includes existing files, not directories.
        """
        connected = set()

        base_dir = self.path.parent
        base_name = self.path.name
        protected = {self.path, self.savepath.targetpath}  # protect targetpath

        # Files in same directory that start with the same base name (excluding self.path)
        for f in base_dir.iterdir():
            if f.is_file() and f.name.startswith(base_name) and f not in protected:
                connected.add(f)

        # Files explicitly listed in savepath (excluding self.path and targetpath)
        for p in vars(self.savepath).values():
            if isinstance(p, Path) and p.exists() and p.is_file() and p not in protected:
                connected.add(p)

        return connected
    
    @property
    def target_img(self):
        """Target image of the mask. If not exists, return None."""
        if self._target_img is None and self.savepath.targetpath.exists():
            from ezphot.imageobjects import ScienceImage
            self._target_img = ScienceImage(self.savepath.targetpath, load = True)
        return self._target_img
    

#%%