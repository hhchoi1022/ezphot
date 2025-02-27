#%%
import logging
import logging.handlers
import os
from astropy.io import fits
from astropy.time import Time


from tippy.configuration import TIPConfig
from tippy.catalog import Catalog
from tippy.helper import Helper
#%%

    
class Logger:
    """
    A class for creating and managing loggers.

    Parameters
    ----------
    unitnum : int
        The unit number.
    logger_name : str
        The name of the logger.
    **kwargs : dict, optional
        Additional keyword arguments.

    Methods
    -------
    log()
        Get the logger instance.
    createlogger(logger_name)
        Create a logger instance.
    """
    def __init__(self,
                 logger_name):
        self._log = self.createlogger(logger_name)
        self.path = logger_name
    
    def log(self):
        """
        Get the logger instance.

        Returns
        -------
        logging.Logger
            The logger instance.
        """
        return self._log
    
    def createlogger(self,
                     logger_name,
                     logger_level = 'INFO'):
        """
        Create a logger instance.

        Parameters
        ----------
        logger_name : str
            The name of the logger.

        Returns
        -------
        logging.Logger
            The created logger instance.
        """
        # Create Logger
        logger = logging.getLogger(logger_name)
        # Check handler exists
        if len(logger.handlers) > 0:
            return logger # Logger already exists
        logger.setLevel(logger_level)
        formatter = logging.Formatter(datefmt = '%Y-%m-%d %H:%M:%S',fmt = f'[%(levelname)s] %(asctime)-15s] | %(message)s')
        
        # Create Handlers
        streamHandler = logging.StreamHandler()
        streamHandler.setLevel(logger_level)
        streamHandler.setFormatter(formatter)
        logger.addHandler(streamHandler)
        fileHandler = logging.FileHandler(filename = logger_name)
        fileHandler.setLevel(logger_level)
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)
        return logger
    
#%%
class BaseImage(TIPConfig):
    """ Handles FITS image processing and tracks its status """

    def __init__(self, path: str, telinfo : dict = None):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")

        self.helper = Helper()
        self.path = path
        self.telinfo = telinfo
        self.telkey = self._get_telkey()
        self._data = None
        self._header = None
        filename = os.path.basename(path)
        
        # Initialize or load status
        super().__init__(telkey = self.telkey)

    
    @property
    def data(self):
        """ Lazy loading of FITS data """
        if self._data is None:
            self._data = fits.getdata(self.path)
        return self._data
    
    @property
    def header(self):
        """ Lazy loading of FITS header """
        if self._header is None:
            self._header = fits.getheader(self.path)
        return self._header
    
    @property
    def observatory(self):
        return self.telinfo['obs']
    
    @property
    def telname(self):
        header = self.header
        for key in self.key_variants['TELESCOP']:
            if key in header:
                return header[key]
    @property
    def imgtype(self):
        header = self.header
        for key in self.key_variants['IMGTYPE']:
            if key in header:
                imgtype = header[key]
                imgtype_variants = dict(BIAS= ['BIAS', 'Bias', 'bias', 'ZERO', 'Zero', 'zero'],
                                        DARK= ['DARK', 'Dark', 'dark'],
                                        FLAT= ['FLAT', 'Flat', 'flat'],
                                        LIGHT= ['LIGHT', 'Light', 'light', 'OBJECT', 'Object', 'object'])
                for key, variants in imgtype_variants.items():
                    if imgtype in variants:
                        return key
        print('WARNING: IMGTYPE not found in header')
        return 'UNKNOWN'
    
    @property
    def alt(self):
        header = self.header
        for key in self.key_variants['ALTITUDE']:
            if key in header:
                return header[key]
    
    @property
    def az(self):
        header = self.header
        for key in self.key_variants['AZIMUTH']:
            if key in header:
                return header[key]
    
    @property 
    def ra(self):
        header = self.header
        for key in self.key_variants['RA']:
            if key in header:
                return header[key]
    
    @property
    def dec(self):
        header = self.header
        for key in self.key_variants['DEC']:
            if key in header:
                return header[key]
    
    @property
    def objname(self):
        header = self.header
        for key in self.key_variants['OBJECT']:
            if key in header:
                return header[key]
    
    @property
    def filter(self):
        header = self.header
        for key in self.key_variants['FILTER']:
            if key in header:
                return header[key]
    
    @property
    def obsdate(self):
        header = self.header
        # First, search for utcdate
        for key in self.key_variants['DATE-OBS']:
            if key in header:
                obsdate = Time(header[key], format = 'isot', scale = 'utc')
                return obsdate.isot
        # If not found, search for jd
        for key in self.key_variants['JD']:
            if key in header:
                obsdate = Time(header[key], format = 'jd')
                return obsdate.isot
        # If not found, search for mjd
        for key in self.key_variants['MJD']:
            if key in header:
                obsdate = Time(header[key], format = 'mjd')
                return obsdate.isot

    @property
    def exptime(self):
        header = self.header
        for key in self.key_variants['EXPTIME']:
            if key in header:
                return header[key]
    
    @property
    def binning(self):
        header = self.header
        for key in self.key_variants['BINNING']:
            if key in header:
                return header[key]
    
    @property
    def gain(self):
        header = self.header
        for key in self.key_variants['GAIN']:
            if key in header:
                return header[key]
    
    @property
    def naxis1(self):
        header = self.header
        for key in self.key_variants['NAXIS1']:
            if key in header:
                return header[key]

    @property
    def naxis2(self):
        header = self.header
        for key in self.key_variants['NAXIS2']:
            if key in header:
                return header[key]
            
    @property
    def fovx(self):
        return '%.3f' %(self.telinfo['pixelscale'] * self.telinfo['x'] / 3600)
    
    @property
    def fovy(self):
        return '%.3f' %(self.telinfo['pixelscale'] * self.telinfo['y'] / 3600)
        
    @property
    def key_variants(self):
        # Define key variants, if a word is duplicated in the same variant, posit the word with the highest priority first
        key_variants_upper = {
            # Default key in rawimages 
            'ALTITUDE': ['ALT', 'ALTITUDE'],
            'AZIMUTH': ['AZ', 'AZIMUTH'],
            'GAIN': ['GAIN'],
            'CCD-TEMP': ['CCDTEMP', 'CCD-TEMP'],
            'FILTER': ['FILTER', 'FILTNAME', 'BAND'],
            'IMGTYPE': ['IMGTYPE', 'IMAGETYP', 'IMGTYP'],
            'EXPTIME': ['EXPTIME', 'EXPOSURE'],
            'DATE-OBS': ['DATE-OBS', 'OBSDATE', 'UTCDATE'],
            'DATE-LOC': ['DATE-LOC', 'DATE-LTC', 'LOCDATE', 'LTCDATE'],
            'JD' : ['JD', 'JD-HELIO', 'JD-UTC', 'JD-OBS'],
            'MJD' : ['MJD', 'MJD-HELIO', 'MJD-UTC', 'MJD-OBS'],
            'RA': ['RA', 'CRVAL1', 'OBSRA'],
            'DEC': ['DEC', 'DECL', 'DEC.', 'DECL.', 'CRVAL2', 'OBSDEC'],   
            'TELESCOP' : ['TELESCOP', 'TELNAME'],
            'BINNING': ['BINNING', 'XBINNING'],
            'OBJECT': ['OBJECT', 'OBJNAME', 'TARGET', 'TARNAME'],
            'OBJCTID': ['OBJCTID', 'OBJID', 'ID'],
            'OBSMODE': ['OBSMODE', 'MODE'],
            'SPECMODE': ['SPECMODE'],
            'NTELESCOP': ['NTELESCOP', 'NTEL'],
            'NOTE': ['NOTE'],
            'NAXIS1': ['NAXIS1'],
            'NAXIS2': ['NAXIS2'],
            # Additional key after processing
            'CTYPE1': ['CTYPE1'],
            'CTYPE2': ['CTYPE2'],
            'CRVAL1': ['CRVAL1'],
            'CRVAL2': ['CRVAL2'],
            'SEEING': ['SEEING'],
            'ELONGATION': ['ELONGATION', 'ELONG'],
            'SKYSIG': ['SKYSIG', 'SKY_SIG'],
            'SKYVAL': ['SKYVAL', 'SKY_VAL'],
            'APER': ['APER_2'],
            'ZP': ['ZP_2'],
            'ZPERR': ['EZP_2'],
            'DEPTH': ['UL5_2'],
        }

        # Sort each list in the dictionary by string length (descending order)
        sorted_key_variants_upper = {
            key: sorted(variants, key=len, reverse=True)
            for key, variants in key_variants_upper.items()
        }
        return sorted_key_variants_upper

    def _get_telkey(self):
        """ Get the telescope name from the FITS header """
        telinfo = self.telinfo
        if telinfo['mode']:
            telkey = f"{telinfo['obs']}_{telinfo['ccd']}_{telinfo['mode']}_{telinfo['binning']}x{telinfo['binning']}"
        else:
            telkey = f"{telinfo['obs']}_{telinfo['ccd']}_{telinfo['binning']}x{telinfo['binning']}"
        return telkey

# %%
if __name__ == '__main__':
    C = BaseImage(path = '/lyman/data1/processed_1x1_gain2750/T03898/7DT02/m450/calib_7DT02_T03898_20240901_074507_m450_100.fits' , telinfo =  Helper().get_telinfo('7DT', 'C361K', 'HIGH', 1))
    self = C
# %%
