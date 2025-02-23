
# %%
from astropy.io import fits
from astropy.io import ascii
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.table import Table

from tqdm import tqdm
import os
import numpy as np
from astropy.table import unique
import inspect
from astropy import constants as const

# %%

class SpectroscopyHelper():

    def __init__(self):
        self.c = const.c.value
        self.sigma = const.sigma_sb.cgs.value
        self.d10pc = 10 * const.pc.cgs.value
        self.c = const.c.cgs.value
        self.h = const.h.cgs.value
        self.k = const.k_B.cgs.value
        
    @property
    def specpath(self):
        # Get the file where the class is defined
        file_path = inspect.getfile(SpectroscopyHelper)

        # Convert the file path to an absolute path using os.path.abspath
        absolute_path = os.path.abspath(file_path)

        path_dir = os.path.join(os.path.dirname(absolute_path),'../spectroscopy')

        return path_dir

    def __repr__(self):
        methods = [f'SpectroscopyHelper.{name}()\n' for name, method in inspect.getmembers(
            SpectroscopyHelper, predicate=inspect.isfunction) if not name.startswith('_')]
        txt = '[Methods]\n'+''.join(methods)
        return txt

# %%
