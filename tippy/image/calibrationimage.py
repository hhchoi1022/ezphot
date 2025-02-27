#%%
from astropy.time import Time
from astropy.io import fits
import json
import os

from tippy.configuration import TIPConfig
from tippy.helper import Helper
from tippy.image import Logger
from tippy.image import BaseImage
#%%
# === Status Class ===
class Status:
    """ Manages the image processing steps status """
    
    PROCESS_STEPS = [
        "biascor", "darkcor", "combine", "master"
    ]
    
    def __init__(self, **kwargs):
        """ Initialize status dictionary with uniform dict structure. """
        # Initialize all status with dict(status=False, update_time=None)
        self.status = {step: dict(status=False, update_time=None) for step in self.PROCESS_STEPS}
        
        # Allow overriding default values
        for key, value in kwargs.items():
            if key in self.status:
                self.status[key] = value

    def update(self, process_name):
        """ Mark a process as completed and update timestamp. """
        if process_name in self.status:
            if self.status[process_name]['status'] == False:
                self.status[process_name] = dict(status=True, update_time=Time.now().iso)
        else:
            raise ValueError(f"Invalid process name: {process_name}")

    def to_dict(self):
        return self.status

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def __repr__(self):
        """ Represent process status as a readable string """
        status_list = [f"{key}: {value}" for key, value in self.status.items()]
        return "Status =====================================\n  " + "\n  ".join(status_list) + "\n==================================================="

# === Info Class ===
class Info:
    """Stores metadata of a FITS image in a uniform dict format."""
    
    INFO_FIELDS = [
        "PATH", "OBSERVATORY", "CCD", "TELKEY", "TELNAME",
        "OBSDATE", "NAXIS1", "NAXIS2", "PIXELSCALE", "OBJNAME", 
        "IMGTYPE", "FILTER", "BINNING", "EXPTIME", "GAIN"
    ]
    
    def __init__(self, **kwargs):
        """ Initialize info dictionary with uniform dict structure. """
        # Initialize all info fields with dict(value=None, update_time=None)
        self.info = {field: None for field in self.INFO_FIELDS}
        
        # Allow overriding default values
        for key, value in kwargs.items():
            if key in self.info:
                self.info[key] = value

    def update(self, key, value):
        """ Update an info field and set the update time. """
        if key in self.info:
            self.info[key] = value
        else:
            print(f'WARNING: Invalid key: {key}')
    
    def add(self, key, value):
        """ Add a new info field and set the update time. """
        if key not in self.info:
            self.info[key] = value
        else:
            print(f'WARNING: Invalid key: {key}')
    
    def remove(self, key):
        """ Remove an info field. """
        if key in self.info:
            del self.info[key]
        else:
            print(f'WARNING: Invalid key: {key}')

    def to_dict(self):
        return self.info

    @classmethod
    def from_dict(cls, data):
        """ Create an Info instance from a dictionary. """
        return cls(**{key: data.get(key) for key in cls.INFO_FIELDS})

    def __repr__(self):
        """ Represent info as a readable string. """
        info_list = [f"{key}: {value}" for key, value in self.info.items()]
        return "Info =====================================\n  " + "\n  ".join(info_list) + "\n==================================================="


#%%
class CalibrationImage(BaseImage):
    """ Handles FITS image processing and tracks its status """

    def __init__(self, path: str, telinfo : dict, status : Status = None, savedir : str = None):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")

        super().__init__(path = path, telinfo = telinfo)

        if self.imgtype not in ['BIAS', 'DARK', 'FLAT']:
            raise ValueError(f"Invalid image type: {self.imgtype}")
        
        filename = os.path.basename(path)
        if savedir:
            self.savedir = savedir
        else:
            self.savedir = os.path.join(self.config['CALIBDATA_DIR'], self.observatory, self.telkey, self.imgtype, self.telname)
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir, exist_ok=True)
            
        self.statuspath = os.path.join(self.savedir, filename.split('.fits')[0] + '.calibim_status')
        self.infopath = os.path.join(self.savedir, filename.split('.fits')[0] + '.calibim_info')
        self.loggerpath = os.path.join(self.savedir, filename.split('.fits')[0] + '.calibim_log')
        self.logger = Logger(logger_name = self.loggerpath).log()
                
        # Initialize or load status
        if status:
            # If status is provided, use it
            self.status = status
        # Otherwise, load from file or create a new instance
        elif os.path.exists(self.statuspath):
            self.status = self.load_status()
        else:
            self.status = Status()
            self._check_status()
        self.save_status()
        if os.path.exists(self.infopath):
            self.info = self.load_info()
        else:
            self.info = Info(PATH = path, OBSERVATORY = self.telinfo['obs'], CCD = self.telinfo['ccd'], 
                             TELKEY = self.telkey, TELNAME = self.telname, OBSDATE = self.obsdate, 
                             NAXIS1 = self.naxis1, NAXIS2 = self.naxis2, PIXELSCALE = self.telinfo['pixelscale'], 
                             OBJNAME = self.objname, IMGTYPE = self.imgtype, FILTER = self.filter, 
                             BINNING = self.binning, EXPTIME = self.exptime, GAIN = self.gain)
            self._check_info()
        self.save_info()
        
    def __repr__(self):
        return f"CalibrationImage(type = {self.imgtype}, binning = {self.binning}, gain = {self.gain}, path = {os.path.basename(self.path)})"
        
    def write(self, path : str):
        """ Write fits CalibrationImage into fits file """
        data = self.data
        header = self.header
        status = self.status
        os.makedirs(os.path.dirname(path), exist_ok = True)
        fits.writeto(path, data, header, overwrite=True)
        self.logger.info(f"CalibrationImage is written to {path}")
        updated_instance = CalibrationImage(path = path, telinfo = self.telinfo, status = status)
        updated_instance.save_status()
        return updated_instance
        
    def load_status(self):
        """ Load processing status from a JSON file """
        with open(self.statuspath, 'r') as f:
            status_data = json.load(f)
        return Status.from_dict(status_data)

    def save_status(self):
        """ Save processing status to a JSON file """
        with open(self.statuspath, 'w') as f:
            json.dump(self.status.to_dict(), f, indent=4)

    def update_status(self, process_name):
        """ Mark a process as completed and update time """
        self.status.update(process_name)
        self.save_status()
        
    def load_info(self):
        """ Load processing info from a JSON file """
        with open(self.infopath, 'r') as f:
            info_data = json.load(f)
        return Info.from_dict(info_data)
    
    def save_info(self):
        """ Save processing info to a JSON file """
        with open(self.infopath, 'w') as f:
            json.dump(self.info.to_dict(), f, indent=4)
    
    def update_info(self, key, value):
        """ Update processing info """
        self.info.update(key, value)
        self.save_info()
        
    def add_info(self, key, value):
        """ Add a new info field """
        self.info.add(key, value)
        self.save_info()
    
    def remove_info(self, key):
        """ Remove an info field """
        self.info.remove(key)
        self.save_info()
        
    def _check_info(self):
        """ Register necessary info fields """
        header = self.header
        for key in self.info.INFO_FIELDS:
            if key in self.key_variants:
                key_variants = self.key_variants[key]
                for variant in key_variants:
                    if variant in header:
                        self.info.update(key, header[variant])
                    else:
                        pass
        self.save_info()
    
    def _check_status(self):
        """ Dummy function to check if all status are completed """
        pass

# %%
if __name__ == '__main__':
    path = '/home/hhchoi1022/data/obsdata/7DT_C361K_gain2750_1x1/7DT02/2025-02-02_gain2750/7DT02_20250203_101702_DARK_m700_1x1_100.0s_0000.fits'
    original_path = '/lyman/data1/obsdata/7DT02/2025-02-17_gain2750/7DT02_20250218_105201_BIAS_m700_1x1_0.0s_0000.fits'
    C = CalibrationImage(path = original_path, telinfo =  Helper().get_telinfo('7DT', 'C361K', 'HIGH', 1))
    
    self = C
# %%
