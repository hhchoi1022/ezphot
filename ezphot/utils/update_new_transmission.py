
#%%
import numpy as np
import glob
from typing import Union
from ezphot.helper import Helper
from pathlib import Path
from astropy.table import Table
from astropy.io import ascii
import matplotlib.pyplot as plt
import pyphot
from pyphot.phot import Filter
from pyphot import unit

class FilterRegister:
    
    def __init__(self, filter_dir: Union[str, Path] = None):
        self.helper = Helper()
        if filter_dir is None:
            filter_dir = Path(self.helper.config['SYNPHOT_FILTERDIR'])
        self.filter_dir = Path(filter_dir)
        
    def filter_info(self, filter_name: str):
        if not filter_name in self.list_filter():
            try:
                lib = pyphot.get_library()
                filt = lib[filter_name]
                print(f"Filter {filter_name} found in pyphot library")
            except Exception as e:
                raise ValueError(f"Filter {filter_name} not found")    
        else:
            transmission_tbl = self.read_transmission_file(self.list_filter()[filter_name])
            filt = Filter(
                transmission_tbl['wavelength'] * unit['AA'],
                transmission_tbl['response'],
                name=filter_name
            )
            
        filter_info_dict = dict()
        filter_info_dict['name'] = filter_name
        filter_info_dict['wavelength_pivot'] = filt.lpivot
        filter_info_dict['wavelength_effective'] = filt.leff
        filter_info_dict['wavelength_min'] = filt.lmin
        filter_info_dict['wavelength_max'] = filt.lmax
        filter_info_dict['fwhm'] = filt.fwhm
        filter_info_dict['width'] = filt.width
        filter_info_dict['wavelength'] = np.array(filt.wavelength)
        filter_info_dict['transmission'] = np.array(filt.transmit)        
        return filter_info_dict
    
    def list_filter(self):
        transmission_files = {file.stem:file for file in list(self.filter_dir.glob('*'))}
        return transmission_files
    
    def format_transmission_file(self, file_name: str,
                                 wl_range_AA: tuple = (1000, 11000),
                                 step_size_AA: float = 10):
        transmission_tbl = self.read_transmission_file(file_name)
        
        wl_range = np.arange(wl_range_AA[0], wl_range_AA[1], step_size_AA)
        response = np.interp(wl_range, transmission_tbl['wavelength'], transmission_tbl['response'])
        formatted_tbl = Table()
        formatted_tbl['wavelength'] = wl_range
        formatted_tbl['response'] = response
        return formatted_tbl
    
    def read_transmission_file(self, file_name: str):
        # Try different file formats
        supported_formats = [None, 'ascii', 'csv', 'txt', 'dat', 'fits', 'ascii', 'json', 'yaml', 'yml', 'hdf5', 'h5', 'hdf', 'hdf4', 'hdf3', 'hdf2', 'hdf1', 'hdf0']
        for format in supported_formats:
            try:
                tbl = ascii.read(file_name, format=format)
                if len(tbl.colnames) == 2:
                    tbl.rename_columns(tbl.colnames, ['wavelength', 'response'])
                    tbl.sort('wavelength')
                    df = tbl.to_pandas()
                    # .5 for wavelength and response
                    df['wavelength'] = df['wavelength'].round(5)
                    df['response'] = df['response'].round(5)
                    tbl = Table.from_pandas(df)
                else:
                    raise ValueError()
                return tbl
            except Exception as e:
                continue
        raise ValueError(f"Unsupported file format: {file_name}")
    
    def plot_transmission_file(self, 
                               file_name: str, 
                               filter_name: str = None,
                               ax: plt.Axes = None, 
                               c = 'k', 
                               title: str = None):
        file_name = Path(file_name)
        transmission_tbl = self.read_transmission_file(file_name)
        
        if filter_name is None:
            filter_name = file_name.stem
        if ax is None:
            fig, ax = plt.subplots(figsize = (10, 6))
        else:
            fig = ax.figure
        ax.plot(transmission_tbl['wavelength'], transmission_tbl['response'], c = c, label = filter_name)
        if title is not None:
            ax.set_title(title)
        ax.set_xlabel('Wavelength (Å)')
        ax.set_ylabel('Response')
        ax.grid(True)
        fig.tight_layout()
        plt.show()
        return fig, ax

# %%
self = FilterRegister()
# %%
filt = self.filter_info('m425w')
# %%
filt

# %%
