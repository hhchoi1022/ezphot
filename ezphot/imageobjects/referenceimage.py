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
from astropy.table import Table, vstack

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
    """
    Class representing a reference FITS image.
    
    Inherits from `BaseImage` and provides methods for managing reference image metadata,
    processing status tracking, and handling associated products like background maps, error maps, masks, and source catalogs.

    Parameters
    ----------
    path : str or Path
        Path to the reference FITS image.
    telinfo : dict, optional
        Telescope metadata dictionary.
    status : Status, optional
        Initial status object. If not provided, status is loaded from file or initialized.
    load : bool, optional
        Whether to load status and header upon initialization.
    """

    def __init__(self, path: Union[Path, str], telinfo : dict = None, status: Status = None, load: bool = True):
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
            f"ReferenceImage(\n"
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
        
        # Preserve savedir if manually set
        if hasattr(self, '_savedir') and self._savedir is not None:
            new_instance._savedir = self._savedir

        return new_instance

    def write(self, verbose: bool = True):
        """Write ReferenceImage data to savepath.
        
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
               skip_exts: list = ['.png', '.cat'],
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
    
    def to_scienceimage(self):
        """
        Convert the reference image to a ScienceImage.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        scienceimage : ScienceImage
            The converted ScienceImage instance.
        """
        
        from ezphot.imageobjects import ScienceImage
        scienceimage = ScienceImage(self.path, telinfo=self.telinfo, load=False)
        scienceimage.data = self.data.copy() if self.data is not None else None
        scienceimage.header = self.header.copy() if self.header is not None else None
        return scienceimage

    def register(self):
        """
        Register the reference image to the reference data directory.
        
        Parameters
        ----------
        None

        Returns
        -------
        None
        """
            
        # Save the reference image to the reference data directory
        if not self.is_saved:
            self.write()
        
        def update_reference_summary(
            target_img: Union[ReferenceImage],
            output_table = f'{self.config["REFDATA_DIR"]}/summary.ascii_fixed_width',
            format = 'ascii.fixed_width',
            ):
            if not target_img.is_saved:
                print(f"Target image {target_img.path.name} does not exist.")
                return
                    
            # Initialize new row dictionary
            rows = {
                'file': [],
                'observatory': [],
                'telkey': [],
                'objname': [],
                'ra': [],
                'dec': [],
                'fov_ra': [],
                'fov_dec': [],
                'telname': [],
                'exptime': [],
                'obsdate': [],
                'filtername': [],
                'depth': [],
                'seeing': [],
                'file_size_bytes': [],
                'modified_time': [],
            }   
            
            summary_path = Path(output_table)
            if summary_path.exists():
                try:
                    existing_table = Table.read(output_table, format=format)
                    existing_paths = set(str(f) for f in existing_table['file'])
                    print(f"Loaded {len(existing_table)} existing entries from summary.")
                except Exception as e:
                    print(f"Warning: Failed to read existing summary file: {e}")
            else:
                row_types = ['str', 'str', 'str', 'str', 
                                'float64', 'float64', 'float64', 'float64', 
                                'str', 'float64', 'str', 'str', 
                                'float64', 'float64', 'int64', 'str']
                existing_table = Table(names = rows.keys(), dtype = row_types)
                existing_paths = set()
            
            target_path_str = str(target_img.savepath.savepath.resolve())
            if target_path_str in existing_paths:
                print(f"Skipping {target_path_str}, already processed.")
                return
            else:
                print(f"Processing {target_path_str}...")
                try:
                    file_path_str = target_path_str
                    center_info = target_img.center
                    rows['file'].append(file_path_str)
                    rows['observatory'].append(target_img.observatory)
                    rows['telkey'].append(target_img.telkey)
                    rows['ra'].append(center_info['ra'])
                    rows['dec'].append(center_info['dec'])
                    rows['fov_ra'].append(target_img.fovx)
                    rows['fov_dec'].append(target_img.fovy)
                    rows['telname'].append(target_img.telname)
                    rows['exptime'].append(target_img.exptime)
                    rows['obsdate'].append(target_img.obsdate)
                    rows['filtername'].append(target_img.filter)
                    rows['depth'].append(target_img.depth)
                    rows['seeing'].append(target_img.seeing)
                    rows['objname'].append(target_img.objname)
                    file_stat = target_img.savepath.savepath.stat()
                    rows['file_size_bytes'].append(float(file_stat.st_size))
                    rows['modified_time'].append(Time.now().isot)
                except (IndexError, ValueError, AttributeError) as e:
                    print(f"Skipping {target_path_str}: {e}")
            
            # Build and write updated table
            if len(rows['file']) > 0:
                new_table = Table(rows)
                if existing_table is not None:
                    full_table = vstack([existing_table, new_table])
                else:
                    full_table = new_table
            else:
                print("No new rows to add.")
                return

            full_table.sort(['observatory', 'telkey', 'objname', 'filtername', 'depth'])
            full_table.write(output_table, format=format, overwrite=True)
            print(f"Saved {len(full_table)} total rows to {output_table} in format '{format}'")
    
        # Update the reference summary with the new reference image
        update_reference_summary(self)

    def load_status(self):
        """ Load processing status from a JSON file.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        if self.savepath.statuspath is None:
            raise ValueError("Cannot load ScienceImage status: save path is not defined.")
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
            raise ValueError("Cannot save ScienceImage status: save path is not defined.")    
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
        base_dir = Path(self.config['REFDATA_DIR'])
        return base_dir / self.observatory / self.telkey / self.objname / self.telname / self.filter

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
            loggerpath=savedir / (filename + '.log'),
            # Mask
            maskpath=savedir / (filename + '.mask'),
            invalidmaskpath= savedir / (filename + '.invalidmask'),
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
            stampcatalogpath = savedir / (filename + '.stampcat'),
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

        #self.save_status() 
