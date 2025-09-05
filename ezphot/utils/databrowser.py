

#%%
import inspect
from pathlib import Path
from typing import List, Union
from functools import partial
from multiprocessing import Pool

from tqdm import tqdm

from ezphot.helper import Helper

#%%

def _get_imginfo(filelist, pattern):
    helper = Helper()   
    imginfo = helper.get_imginfo(filelist, pattern=pattern)
    return imginfo

def _load_image(cls, path, telinfo=None):
    try:
        estimated_telinfo = None
        # Estimate telinfo if it's a callable
        if callable(telinfo):
            estimated_telinfo = telinfo(path)
        else:
            estimated_telinfo = telinfo

        # Construct with or without telinfo
        if estimated_telinfo is not None:
            return cls(path, telinfo=estimated_telinfo, load=True)
        else:
            return cls(path, load=True)

    except Exception as e:
        print(f"[WARNING] Failed to load {path}: {e}")
        return None

class DataBrowser:
    """
    DataBrowser is a class that provides a unified interface for searching and loading data 
    from the telescope data directory.  

    It provides:

    1. Search for files matching the current filters and return them as different types of objects.

       Types of objects:
       
       - ``ImageSet`` of ``ScienceImage``
       - ``ImageSet`` of ``ReferenceImage``
       - ``ImageSet`` of ``CalibrationImage``
       - ``ImageSet`` of ``Background``
       - ``ImageSet`` of ``Errormap``
       - ``ImageSet`` of ``Mask``
       - ``CatalogSet`` of ``Catalog``
    """
    
    def __init__(self, foldertype: str = None):
        self.helper = Helper()
        self.foldertype = foldertype
        self.basepath = self._get_default_path()
        # Search attributes
        self.observatory = None
        self.telkey = None
        self.imgtype = None
        self.telname = None
        self.objname = None
        self.filter = None
        self.obsdate = None
    
    def __repr__(self):
        txt = f"<DataBrowser(searchpath='{self.searchpath}', foldertype='{self.foldertype}')>\n"
        txt += "Search Attributes:\n"
        txt += f"  observatory : {self.observatory or '*'}\n"
        txt += f"  telkey      : {self.telkey or '*'}\n"
        txt += f"  imgtype     : {self.imgtype or '*'}\n"
        txt += f"  telname     : {self.telname or '*'}\n"
        txt += f"  objname     : {self.objname or '*'}\n"
        txt += f"  filter      : {self.filter or '*'}\n"
        txt += f"  obsdate     : {self.obsdate or '*'}\n"
        txt += f'\n For help, use \'help(self)\' or `self.help()`.'
        return txt

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

    def search(self, 
               pattern: str ='*.fits', 
               return_type: str = 'path') -> Union[dict, List[Path]]:

        """
        Search for FITS files matching the current attributes and return them grouped by telescope or as objects.
        
        This method will search for the files in the searchpath defined by the current filters 
        (observatory, telkey, objname, telname, filter, obsdate).        
        
        Parameters
        ----------
        pattern : str
            Filename pattern to match (e.g., ``'*.fits'``).
        return_type : str
            ``'path'``, ``'science'``, ``'reference'``, ``'calibration'``, 
            ``'background'``, ``'errormap'``, ``'mask'`` or ``'imginfo'`` 
            to convert paths to objects.
        
        Returns
        -------
        output : Dict, Table, ImageSet, CatalogSet
            - If ``return_type``=``'path'``, returns ``dict[telname] = list of file paths``.
            - If ``return_type``=``'imginfo'``, returns ``astropy.table.Table`` of imginfo.
            - If ``return_type``=``'science'``, ``'reference'``, ``'calibration'``, ``'mask'``, ``'background'``, ``'errormap'``,  returns ``'ImageSet'``.
            - If ``return_type``=``'catalog'``, returns CatalogSet.
        """
    
        return self.search_folder(pattern=pattern, folder=self.searchpath, return_type=return_type)
    
    def search_folder(self, 
                      pattern: str, 
                      folder: str,
                      return_type: str = 'path') -> Union[dict, List[Path]]:
        """
        Search for FITS files matching the current attributes and return them grouped by telescope or as objects.
        
        This method will search for the files in the searchpath defined by the current filters 
        (observatory, telkey, objname, telname, filter, obsdate).        

        Parameters
        ----------
        pattern : str
            Filename pattern to match (e.g., ``'*.fits'``).
        folder : str
            Folder to search in. 
        return_type : str
            ``'path'``, ``'science'``, ``'reference'``, ``'calibration'``, 
            ``'background'``, ``'errormap'``, ``'mask'`` or ``'imginfo'`` 
            to convert paths to objects.
        
        Returns
        -------
        output : Dict, Table, ImageSet, CatalogSet
            - If ``return_type``=``'path'``, returns ``dict[telname] = list of file paths``.
            - If ``return_type``=``'imginfo'``, returns ``astropy.table.Table`` of imginfo.
            - If ``return_type``=``'science'``, ``'reference'``, ``'calibration'``, ``'mask'``, ``'background'``, ``'errormap'``,  returns ``'ImageSet'``.
            - If ``return_type``=``'catalog'``, returns ``'CatalogSet'``.
        """
        
        import glob
        from collections import defaultdict

        glob_pattern = str(folder / pattern)
        matched_files = glob.glob(glob_pattern, recursive=True)
        print(f"[INFO] Found {len(matched_files)} files matching '{glob_pattern}'")

        if return_type == 'path':
            return matched_files
        
        elif return_type == 'imginfo':
            return self._to_imginfo(matched_files, pattern)

        elif return_type == 'science':
            from ezphot.imageobjects import ImageSet
            return ImageSet(self._to_science_images(matched_files))

        elif return_type == 'reference':
            from ezphot.imageobjects import ImageSet
            return ImageSet(self._to_reference_images(matched_files))

        elif return_type == 'calibration':
            from ezphot.imageobjects import ImageSet
            return ImageSet(self._to_calibration_images(matched_files))
        
        elif return_type == 'background':
            from ezphot.imageobjects import ImageSet
            return ImageSet(self._to_background(matched_files))
        
        elif return_type == 'errormap':
            from ezphot.imageobjects import ImageSet
            return ImageSet(self._to_errormap(matched_files))
        
        elif return_type == 'catalog':
            from ezphot.dataobjects import CatalogSet
            return CatalogSet(self._to_catalog(matched_files))

        else:
            raise ValueError(f"Invalid return_type: {return_type}. Choose from 'path', 'science', 'reference', 'calibration', 'background', 'errormap', 'catalog'.")

    def _get_default_path(self):
        default_path_dict = {
            "scidata": self.helper.config["SCIDATA_DIR"],
            "refdata": self.helper.config["REFDATA_DIR"],
            "calibdata": self.helper.config["CALIBDATA_DIR"],
            "mcalibdata": self.helper.config["CALIBDATA_MASTERDIR"],
            "obsdata": self.helper.config["OBSDATA_DIR"]
            }
        if self.foldertype not in default_path_dict:
            print(f"[WARNING] Unknown foldertype: {self.foldertype}. Available types: {list(default_path_dict.keys())}")
            return None
        else:
            return Path(default_path_dict[self.foldertype])
    
    def _to_imginfo(self, filepaths: List[Union[str, Path]], pattern: str = '*.fits'):
        from astropy.table import vstack
        if not filepaths:
            return None

        # Group files by their parent directory
        from collections import defaultdict
        dir_to_files = defaultdict(list)
        for path in filepaths:
            path = Path(path)
            dir_to_files[path.parent].append(path)

        # Convert grouped files to a list for multiprocessing
        file_groups = list(dir_to_files.values())

        # Run multiprocessing
        if len(file_groups) > 1:
            with Pool(16) as pool:
                results = list(
                    tqdm(
                        pool.starmap(_get_imginfo, [(group, pattern) for group in file_groups]),
                        total=len(file_groups),
                        desc="Collecting ImgInfo"
                    )
                )       
        else:
            results = [_get_imginfo(file_groups[0], pattern)]

        # Combine results
        tables = [tbl for tbl in results if tbl is not None and len(tbl) > 0]
        if not tables:
            return None
        return vstack(tables, metadata_conflicts='silent')

    def _to_science_images(self, filepaths: List[Union[str, Path]]):
        from ezphot.imageobjects import ScienceImage
        with Pool(16) as pool:
            func = partial(_load_image, ScienceImage, telinfo= self.helper.estimate_telinfo)
            images = list(tqdm(pool.imap(func, filepaths), total=len(filepaths), desc="Loading Science Images"))
        return [img for img in images if img is not None]

    def _to_reference_images(self, filepaths: List[Union[str, Path]]):
        from ezphot.imageobjects import ReferenceImage
        telinfo = self.helper.estimate_telinfo(filepaths[0])
        with Pool(16) as pool:
            func = partial(_load_image, ReferenceImage, telinfo=self.helper.estimate_telinfo)
            images = list(tqdm(pool.imap(func, filepaths), total=len(filepaths), desc="Loading Reference Images"))
        return [img for img in images if img is not None]

    def _to_calibration_images(self, filepaths: List[Union[str, Path]]):
        from ezphot.imageobjects import CalibrationImage
        with Pool(16) as pool:
            func = partial(_load_image, CalibrationImage)
            images = list(tqdm(pool.imap(func, filepaths), total=len(filepaths), desc="Loading Calibration Images"))
        return [img for img in images if img is not None]

    def _to_background(self, filepaths: List[Union[str, Path]]):
        from ezphot.imageobjects import Background
        with Pool(16) as pool:
            func = partial(_load_image, Background)
            backgrounds = list(tqdm(pool.imap(func, filepaths), total=len(filepaths), desc="Loading Backgrounds"))
        return [bg for bg in backgrounds if bg is not None]
    
    def _to_errormap(self, filepaths: List[Union[str, Path]]):
        from ezphot.imageobjects import Errormap
        with Pool(16) as pool:
            func = partial(_load_image, Errormap)
            errormaps = list(tqdm(pool.imap(func, filepaths), total=len(filepaths), desc="Loading Errormaps"))
        return [emap for emap in errormaps if emap is not None]

    def _to_mask(self, filepaths: List[Union[str, Path]]):
        from ezphot.imageobjects import Mask
        with Pool(16) as pool:
            func = partial(_load_image, Mask)
            masks = list(tqdm(pool.imap(func, filepaths), total=len(filepaths), desc="Loading Masks"))
        return [mask for mask in masks if mask is not None]
    
    def _to_catalog(self, filepaths: List[Union[str, Path]]):
        from ezphot.dataobjects import Catalog
        with Pool(16) as pool:
            func = partial(_load_image, Catalog)
            masks = list(tqdm(pool.imap(func, filepaths), total=len(filepaths), desc="Loading Catalogs"))
        return [mask for mask in masks if mask is not None]

    @property
    def searchpath(self):
        """
        Get the search path of the current filters.
        
        Returns
        -------
        pathlib.Path
            The search path.
        """
        if self.foldertype == 'scidata' or self.foldertype == 'refdata':
            path = self.basepath / (self.observatory or '*') / (self.telkey or '*') / (self.objname or '*') / (self.telname or '*') / (self.filter or '*')
        elif self.foldertype == 'calibdata':
            path = self.basepath / (self.observatory or '*') / (self.telkey or '*') / (self.imgtype or '*') / (self.telname or '*')
        elif self.foldertype == 'mcalibdata':
            path = self.basepath / '*' / (self.observatory or '*') / (self.telkey or '*') / (self.telname or '*') / (self.imgtype or '*')
        elif self.foldertype == 'obsdata':
            path = self.basepath / (self.observatory or '*') / (self.telname or '*') / (self.obsdate or '*')
        else:
            raise ValueError(f"Unknown foldertype: {self.foldertype}")
        return path

    @property
    def keys(self) -> dict:
        """
        Get the keys of the current filters.
        
        Returns
        -------
        dict
            Dictionary of keys.
        """
        base = self.basepath
        result = {
            'observatory': set(),
            'telkey': set(),
            'objname': set(),
            'telname': set(),
            'filter': set(),
            'imgtype': set(),
            'obsdate': set(),
        }

        if not base.exists():
            return result

        try:
            if self.foldertype in ['scidata', 'refdata']:
                for obs in base.iterdir():
                    if not obs.is_dir():
                        continue
                    if self.observatory and obs.name != self.observatory:
                        continue

                    observatory_has_match = False

                    for telkey in obs.iterdir():
                        if not telkey.is_dir():
                            continue
                        if self.telkey and telkey.name != self.telkey:
                            continue

                        telkey_has_match = False

                        for obj in telkey.iterdir():
                            if not obj.is_dir():
                                continue
                            if self.objname and obj.name != self.objname:
                                continue

                            obj_has_match = False

                            for tel in obj.iterdir():
                                if not tel.is_dir():
                                    continue
                                if self.telname and tel.name != self.telname:
                                    continue

                                tel_has_match = False

                                for filt in tel.iterdir():
                                    if not filt.is_dir():
                                        continue
                                    if self.filter and filt.name != self.filter:
                                        continue

                                    # Success!
                                    result['filter'].add(filt.name)
                                    tel_has_match = True

                                if tel_has_match:
                                    result['telname'].add(tel.name)
                                    obj_has_match = True

                            if obj_has_match:
                                result['objname'].add(obj.name)
                                telkey_has_match = True

                        if telkey_has_match:
                            result['telkey'].add(telkey.name)
                            observatory_has_match = True

                    if observatory_has_match:
                        result['observatory'].add(obs.name)
                                        
            elif self.foldertype == 'calibdata':
                for obs in base.iterdir():
                    if not obs.is_dir(): continue
                    if self.observatory and obs.name != self.observatory: continue
                    result['observatory'].add(obs.name)

                    for telkey in (obs / self.telkey if self.telkey else obs).iterdir():
                        if not telkey.is_dir(): continue
                        result['telkey'].add(telkey.name)

                        for imgtype_dir in telkey.iterdir():
                            if not imgtype_dir.is_dir(): continue
                            if self.imgtype and imgtype_dir.name != self.imgtype: continue
                            result['imgtype'].add(imgtype_dir.name)

                            for telname_dir in imgtype_dir.iterdir():
                                if telname_dir.is_dir():
                                    result['telname'].add(telname_dir.name)

            elif self.foldertype == 'mcalibdata':
                for user in base.iterdir():
                    if not user.is_dir(): continue
                    for obs in user.iterdir():
                        if not obs.is_dir(): continue
                        if self.observatory and obs.name != self.observatory: continue
                        result['observatory'].add(obs.name)

                        for telkey in obs.iterdir():
                            if not telkey.is_dir(): continue
                            result['telkey'].add(telkey.name)

                            for tel in telkey.iterdir():
                                if not tel.is_dir(): continue
                                result['telname'].add(tel.name)

                                for imgtype in tel.iterdir():
                                    if imgtype.is_dir():
                                        result['imgtype'].add(imgtype.name)

            elif self.foldertype == 'obsdata':
                for obs in base.iterdir():
                    if not obs.is_dir(): continue
                    if self.observatory and obs.name != self.observatory: continue
                    result['observatory'].add(obs.name)

                    for tel in obs.iterdir():
                        if not tel.is_dir(): continue
                        result['telname'].add(tel.name)

                        for obsdate in tel.iterdir():
                            if obsdate.is_dir():
                                result['obsdate'].add(obsdate.name)

        except Exception as e:
            print(f"[WARNING] list_available failed: {e}")

        return result
