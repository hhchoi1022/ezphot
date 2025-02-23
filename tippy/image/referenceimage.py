#%%
from astropy.time import Time
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
        "biascor", "darkcor", "flatcor",
        "astrometrycalc", "scampcalc", "zpcalc",
        "bkgsub", "combine", 
        "subtract", "photometry"
    ]
    
    def __init__(self, **kwargs):
        """ Initialize status dictionary with uniform dict structure. """
        # Initialize all processes with dict(status=False, update_time=None)
        self.processes = {step: dict(status=False, update_time=None) for step in self.PROCESS_STEPS}
        
        # Allow overriding default values
        for key, value in kwargs.items():
            if key in self.processes:
                self.processes[key] = value

    def update(self, process_name):
        """ Mark a process as completed and update timestamp. """
        if process_name in self.processes:
            if self.processes[process_name]['status'] == False:
                self.processes[process_name] = dict(status=True, update_time=Time.now().iso)
        else:
            raise ValueError(f"Invalid process name: {process_name}")

    def to_dict(self):
        return self.processes

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def __repr__(self):
        """ Represent process status as a readable string """
        status_list = [f"{key}: {value}" for key, value in self.processes.items()]
        return "Status =====================================\n  " + "\n  ".join(status_list) + "\n==================================================="

# === Info Class ===
class Info:
    """Stores metadata of a FITS image in a uniform dict format."""
    
    INFO_FIELDS = [
        "PATH", "OBSERVATORY", "CCD", "TELKEY", "TELNAME",
        "OBSDATE", "NAXIS1", "NAXIS2", "PIXELSCALE", 
        "ALT", "AZ", "RA", "DEC", "FOVX", "FOVY", "OBJNAME", "IMGTYPE", "FILTER", "BINNING",
        "EXPTIME", "GAIN", "COMBINE_IMLIST", "CRVAL1", "CRVAL2", "SEEING",
        'ELONGATION', "SKYSIG", "SKYVAL", "APER", "ZP", 'DEPTH'
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
            pass
    
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
class ReferenceImage(BaseImage):
    """ Handles FITS image processing and tracks its status """

    def __init__(self, path: str, telinfo : dict, status : Status = None):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")
        
        super().__init__(path = path, telinfo = telinfo)

        filename = os.path.basename(path)
        savedir = os.path.join(self.config['REFDATA_DIR'], self.observatory, self.telkey, self.objname)
        if not os.path.exists(savedir):
            os.makedirs(savedir, exist_ok=True)
        
        self.statuspath = os.path.join(savedir, filename.split('.fits')[0] + '.refim_status')
        self.infopath = os.path.join(savedir, filename.split('.fits')[0] + '.refim_info')
        self.loggerpath = os.path.join(savedir, filename.split('.fits')[0] + '.refim_log')
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
            self._check_info()
        else:
            self.info = Info(PATH = path, OBSERVATORY =  self.telinfo['obs'], CCD = self.telinfo['ccd'],
                             TELKEY = self.telkey, TELNAME = self.telname, OBSDATE = self.obsdate,
                             NAXIS1 = self.naxis1, NAXIS2 = self.naxis2, PIXELSCALE = self.telinfo['pixelscale'],
                             ALT = self.alt, AZ = self.az, RA = self.ra, DEC = self.dec, FOVX = self.fovx, FOVY = self.fovy,
                             OBJNAME = self.objname, IMGTYPE = self.imgtype, FILTER = self.filter,
                             BINNING = self.binning, EXPTIME = self.exptime, GAIN = self.gain, COMBINE_IMLIST = [])
            self._check_info()
        self.save_info()
        
    def __repr__(self):
        return f"ReferenceImage(object = {self.objname}, filter = {self.filter}, binning = {self.binning}, gain = {self.gain}, path = {os.path.basename(self.path)})"
    
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
        """ Update status case as you want! """
        # FOR gppy results
        self.status.update('biascor')
        self.status.update('darkcor')
        self.status.update('flatcor')
        self.status.update('astrometrycalc')
        self.status.update('scampcalc')
        self.status.update('zpcalc')
        if '.com.' in self.path:
            self.status.update('combine')
        self.save_status()


# %%
if __name__ == '__main__':
    image = '/home/hhchoi1022/data/refdata/7DT/7DT_C361K_HIGH_1x1/T11623/calib_7DT05_T11623_20240423_020143_r_360.com.fits'
    C = ReferenceImage(image, telinfo = Helper().get_telinfo('7DT', 'C361K', 'HIGH', 1))

# %%
