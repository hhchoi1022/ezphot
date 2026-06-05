
#%%
from ezphot.helper import Helper
from ezphot.dataobjects import LightCurve, PhotometricSpectrum
from specutils import Spectrum1D, SpectralRegion
from specutils.manipulation import median_smooth, extract_region

from astropy.nddata import StdDevUncertainty
from astropy.io import ascii
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt

import pyphot
from pyphot import unit
from pyphot.phot import Filter

import os
import datetime
from astropy.time import Time
#%%
class SpectrumFile:
    """
    A class to represent and read spectroscopy files in either FITS or ASCII format.

    Attributes
    ----------
    filename : str
        Path to the spectroscopy file.
    format : str
        Format of the file ('fits' or 'ascii').
    header : dict
        Header information from the file.
    wavelength : numpy.ndarray
        Array of wavelength values.
    flux : numpy.ndarray
        Array of flux values.

    Methods
    -------
    __init__(filename)
        Initializes the SpectroscopyFile object and determines the file format.
    __repr__()
        Returns a string representation of the SpectroscopyFile object.
    read_fits()
        Reads the FITS file format and extracts wavelength and flux data.
    read_ascii()
        Reads the ASCII file format and extracts wavelength and flux data.
    """

    def __init__(self, filename: str, flux_unit: str = None, wavelength_unit: str = None, verbose: bool = True):
        """
        Initializes the SpectroscopyFile object and determines the file format.

        Parameters
        ----------
        filename : str
            Path to the spectroscopy file.
        
        Raises
        ------
        ValueError
            If the file format is not recognized.
        """
        self.filename = filename
        self.obsdate = None
        self.wavelength = None
        self.flux = None
        self.fluxerr = None
        self.flux_unit = flux_unit
        self.wavelength_unit = wavelength_unit
        self.header = None
        try:
            self._read_fits(verbose)
            self.format = 'fits'
        except:
            pass
        try:
            self._read_ascii(verbose)
            self.format = 'ascii'
        except:
            pass
        try:
            self._read_ascii_with_END(verbose)
            self.format = 'ascii'
        except:
            pass
        if not hasattr(self, 'format'):
            raise ValueError('File format not recognized')
        else:
            if self.flux_unit is None:
                self.flux_unit = self._estimate_flux_unit(self.flux, self.header, verbose)
            if self.wavelength_unit is None:
                self.wavelength_unit = self._estimate_wavelength_unit(self.wavelength, self.header, verbose)
            
    def __repr__(self):
        """
        Returns a string representation of the SpectroscopyFile object.

        Returns
        -------
        str
            A string representation of the SpectroscopyFile object.
        """
        return f"SpectroscopyFile(Filename = {os.path.basename(self.filename)}, Format = {self.format})"
    
    def _read_ascii_with_END(self, verbose: bool = True):
        with open(self.filename) as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if line.strip() == "END":
                start = i + 1
                break
        header_lines = lines[:start]
                # Parse the header (if needed, for now we are just reading it)
        header = {}
        for line in header_lines:
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip().lstrip('# ')
                value = value.split('/')[0].strip().replace("'", "").replace(" ", "")
                header[key] = value
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lstrip('# ')
                value = value.strip().replace("'", "").replace(" ", "")
                header[key] = value

        data = np.loadtxt(lines[start:])
        # If the data has 3 columns, the third column is the flux error
        if data.shape[1] == 3:
            wave = data[:, 0]
            flux = data[:, 1]
            fluxerr = data[:, 2]
        else:
            wave = data[:, 0]
            flux = data[:, 1]
            fluxerr = None

        self.header = header
        self.wavelength = wave
        self.flux = flux
        self.fluxerr = fluxerr
        self.obsdate = self._get_obsdate_from_header()
        if not self.obsdate:
            self.obsdate = self._get_obsdate_from_filename(self.filename)
        
        if verbose:
            print(f"ASCII file with END keyword read: {self.filename}")
        
    def _read_fits(self, verbose: bool = True):
        """
        Reads the FITS file format and extracts wavelength and flux data.

        This method sets the header, wavelength, and flux attributes for the FITS file.

        Raises
        ------
        Exception
            If there is an error in reading the FITS file.
        """
        from astropy.io import fits
        import numpy as np

        # Open the FITS file
        hdulist = fits.open(self.filename)
        
        # Access the flux data
        flux = hdulist[0].data

        # Extract the header
        header = hdulist[0].header

        # Extract WCS information from the header
        crval1 = header['CRVAL1']  # Reference value (starting wavelength)
        cdelt1 = header['CDELT1']  # Wavelength increment per pixel
        crpix1 = header['CRPIX1']  # Reference pixel (usually 1)

        # Calculate the wavelength array
        n_pixels = flux.size
        wavelength = crval1 + cdelt1 * (np.arange(n_pixels) + 1 - crpix1)

        # Close the FITS file
        hdulist.close()

        # Set the attributes
        self.header = header
        self.wavelength = wavelength
        self.flux = flux
        self.obsdate = self._get_obsdate_from_header()
        if not self.obsdate:
            self.obsdate = self._get_obsdate_from_filename(self.filename)
        if verbose:
            print(f"FITS file read: {self.filename}")

    def _read_ascii(self, verbose: bool = True):
        """
        Reads the ASCII file format and extracts wavelength and flux data.

        This method sets the header, wavelength, and flux attributes for the ASCII file.

        Raises
        ------
        Exception
            If there is an error in reading the ASCII file.
        """
        # # Read the data
        data = ascii.read(self.filename)
        wavelength = np.array(data[data.colnames[0]])
        flux = np.array(data[data.colnames[1]])
        if len(data.colnames) == 3:
            fluxerr = np.array(data[data.colnames[2]])
        else:
            fluxerr = None
        
        # Set the attributes
        self.wavelength = wavelength
        self.flux = flux
        self.fluxerr = fluxerr
        self.obsdate = self._get_obsdate_from_header()
        if not self.obsdate:
            self.obsdate = self._get_obsdate_from_filename(self.filename)
        if verbose:
            print(f"ASCII file read: {self.filename}")
            
    def _get_obsdate_from_header(self):
        obsdate = None
        try:
            obsdate = Time(self.header['DATE-OBS']).isot
        except:
            pass
        try:
            obsdate = Time(self.header['JD'], format = 'jd').isot
        except:
            pass
        return obsdate
        
    def _get_obsdate_from_filename(self,
                                   filename : str,
                                   date_patternlist : list[str] = ['(\d\d\d\d)-(\d\d)-(\d\d)_(\d\d)-(\d\d)', 
                                                                   '(245\d\d\d\d.\d\d\d)', '(20\d\d-\d\d-\d\d)']
                                   ):
        import re
        is_found = False
        for date_pattern in date_patternlist:
            try:
                year, month, day, hour, minute = re.search(date_pattern, filename).groups()
                is_found = True
                break
            except:
                pass
        if is_found:
            dt = datetime.datetime(year = int(year), month = int(month), day= int(day), hour = int(hour), minute = int(minute))
            obsdate = Time(dt).isot
            return obsdate
        else:
            return None
    
    @staticmethod
    def _estimate_flux_unit(flux : np.array, header: dict = None, verbose: bool = True):
        """
        Estimate the flux unit of the spectrum.

        Returns
        -------
        str or None
            One of ['FNU', 'FLAMB', 'JY', 'MJY'], or None if unknown.
        """
        flux_unit = None
        if header is None:
            header = {}
        
        # 1. Header-based detection
        for key in ['BUNIT', 'FLUXUNIT', 'TUNIT1', 'TUNIT2']:
            if key in header:
                val = str(header[key]).lower()
                if 'erg' in val and ('/a' in val or 'angstrom' in val):
                    flux_unit = 'FLAMB'
                if 'erg' in val and '/hz' in val:
                    flux_unit = 'FNU'
                if 'mjy' in val:
                    flux_unit = 'MJY'
                if 'jy' in val:
                    flux_unit = 'JY'

        # 2. Value-scale heuristic (last resort)
        flux = np.asarray(flux)
        flux = flux[np.isfinite(flux)]
        if flux.size == 0:
            flux_unit = None

        med = np.nanmedian(np.abs(flux))
        if 1e-20 < med < 1e-12:
            flux_unit = 'FLAMB'
        if 1e-33 < med < 1e-26:
            flux_unit = 'FNU'
        if 1e-6 < med < 1e0:
            flux_unit = 'JY'
        if 1e0 < med:
            flux_unit = 'MJY'
        
        if flux_unit is None:
            if verbose:
                print("Flux unit is not provided and cannot be estimated")
            return None
        if verbose:
            print(f"Flux unit estimated from data: {flux_unit}")
            
        return flux_unit

    @staticmethod
    def _estimate_wavelength_unit(wavelength: np.ndarray, header: dict = None, verbose: bool = True):
        """
        Estimate the wavelength unit of the spectrum.

        Parameters
        ----------
        wavelength : np.ndarray
            Wavelength array.
        header : dict, optional
            Header dictionary.

        Returns
        -------
        str
            One of ['AA', 'nm', 'um']
        """
        wavelength_unit = None
        if header is None:
            header = {}

        # --------------------------------------------------
        # 1. Header-based detection (most reliable)
        # --------------------------------------------------
        for key in ['CUNIT1', 'WAVEUNIT', 'TUNIT1']:
            if key in header:
                val = str(header[key]).lower()

                if 'angstrom' in val or val in ['a', 'aa', 'å']:
                    wavelength_unit = 'AA'
                if val in ['nm', 'nanometer', 'nanometers']:
                    wavelength_unit = 'nm'
                if val in ['um', 'micron', 'microns', 'µm']:
                    wavelength_unit = 'um'

        # --------------------------------------------------
        # 2. Value-scale heuristic (last resort)
        # --------------------------------------------------
        wave = np.asarray(wavelength)
        wave = wave[np.isfinite(wave)]

        if wave.size == 0:
            wavelength_unit = 'AA'

        med = np.nanmedian(wave)

        # Optical / NIR conventions
        if 1000 < med < 20000:
            wavelength_unit = 'AA'
        if 100 < med < 2000:
            wavelength_unit = 'nm'
        if 0.05 < med < 20:
            wavelength_unit = 'um'
        
        if wavelength_unit is None:
            if verbose:
                print("Wavelength unit is not provided and cannot be estimated")
            return None
        if verbose:
            print(f"Wavelength unit estimated from data: {wavelength_unit}")
            
        return wavelength_unit


#%%

class Spectrum:
    
    def __init__(self,
                 path: str = None,
                 wavelength: np.array = None,
                 flux: np.array = None,
                 fluxerr: np.array = None,
                 wavelength_unit: str = None,
                 flux_unit: str = None,
                 verbose: bool = True
                 ):
        
        self.helper = Helper()

        # 1. Load data
        (wavelength, 
         flux, 
         fluxerr,
         header, 
         obsdate, 
         wavelength_unit, 
         flux_unit) = self._load_input(
            path, wavelength, flux, fluxerr, wavelength_unit, flux_unit, verbose
        )

        # 2. Resolve missing units
        wavelength_unit, flux_unit = self._resolve_units(
            wavelength, flux, header, wavelength_unit, flux_unit, verbose
        )
        # 3. Set internal state
        self._wavelength_data = wavelength
        self._flux_data = flux
        self._fluxerr_data = fluxerr
        self._wavelength_unit = wavelength_unit
        self._flux_unit = flux_unit
        self.header = header
        self.obsdate = obsdate

        # 4. Build spectrum
        self._build_spectrum()
        
    def __repr__(self):
        return f"Spectrum(Range:[{self._wavelength_data.min():.2f}, {self._wavelength_data.max():.2f}] {self._wavelength_unit}, Flux error exists: {self.is_exist_fluxerr})"
        
    def show(self,
             show_flux_unit : str = 'fnu',
             ax = None,
             smooth_factor : int = 1,
             redshift : float = 0,
             normalize : bool = False,
             normalize_region : tuple[int, int] = (6700, 6720),
             linewidth : int = 1,
             linestyle : str = '-',
             label : str = '',
             color : str = 'k',
             log : bool = False,
             offset : float = 0,
             title : str = '',
             ):
        if not show_flux_unit.upper() in ['FNU', 'FLAMB', 'JY', 'AB', 'MJY']:
            raise ValueError('%s is not registered as spectral flux unit'%show_flux_unit)
            
        if show_flux_unit.upper() == 'FNU':
            spec = self.fnu
            flux_label = 'Flux Density (erg s-1 cm-2 Hz-1)'
        elif show_flux_unit.upper() == 'FLAMB':
            spec = self.flamb
            flux_label = 'Flux Density (erg s-1 cm-2 Å-1)'
        elif show_flux_unit.upper() == 'JY':
            spec = self.fjy
            flux_label = 'Flux Density (Jy)'
        elif show_flux_unit.upper() == 'MJY':
            spec = self.fmjy
            flux_label = 'Flux Density (mJy)'
        elif show_flux_unit.upper() == 'AB':
            spec = self.ab
            flux_label = 'Magnitude (AB)'
        wavelength_label = spec.wavelength.unit.to_string()
            
        if normalize:
            norm_region = SpectralRegion(normalize_region[0]*u.AA, normalize_region[1]*u.AA)
            norm_spec = extract_region(spec, norm_region)
            spec /= np.mean(norm_spec.flux)
            flux_label = 'Normalized Flux Density'
            if show_flux_unit.upper() == 'AB':
                flux_label = 'Normalized Magnitude (AB)'
            
        if smooth_factor > 1:
            spec = median_smooth(spec, smooth_factor)
            
        if redshift != 0:
            spec = Spectrum1D(spectral_axis = spec.wavelength * (1+redshift), flux = spec.flux, uncertainty = spec.uncertainty)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure
            
        if log and show_flux_unit.upper() != 'AB':
            flux_label += ' (log scale)'
            ax.scatter(np.log10(spec.wavelength.value), np.log10(spec.flux.value), c = color, s = 10)
            ax.step(np.log10(spec.wavelength.value), np.log10(spec.flux.value), label = label, c = color, linestyle = linestyle, linewidth = linewidth, where = 'mid')
            if self.is_exist_fluxerr:
                ax.fill_between( np.log10(spec.wavelength.value), np.log10(spec.flux.value) - spec.uncertainty.array, np.log10(spec.flux.value) + spec.uncertainty.array, alpha = 0.5)
        else:
            ax.scatter(spec.wavelength.value, spec.flux.value + offset, c = color, s = 10)
            ax.step(spec.wavelength.value, spec.flux.value + offset, label = label, c = color, linestyle = linestyle, linewidth = linewidth, where = 'mid')
            if self.is_exist_fluxerr:
                ax.fill_between(spec.wavelength.value, spec.flux.value + offset - spec.uncertainty.array, spec.flux.value + offset + spec.uncertainty.array, alpha = 0.5, color = 'red')
        ax.set_xlabel(wavelength_label)
        ax.set_ylabel(flux_label)
        ax.set_title(title)
        if show_flux_unit.upper() == 'AB':
            if not ax.yaxis_inverted():
                ax.invert_yaxis()
            
        return fig, ax
            
    def synphot(self,
                filterset : list[str] = ['U', 'B', 'V', 'R', 'I', 'u', 'g', 'r', 'i', 'z', 'medium'],
                visualize : bool = False,
                visualize_unit : str = 'AB', # mag or flux
                visualize_spectrum: bool = True,
                visualize_transmission : bool = True,
                visualize_photometry_label: bool = True,
                pyphot_filters : dict[Filter] = {},
                ax = None,
                ):
        
        def load_filter_from_config(transmission_path: str):

            """
            Load a filter transmission curve from config directory
            and return a pyphot Filter object.
            """
            filepath = transmission_path
            filter_name = filepath.stem
            if not filepath.exists():
                raise FileNotFoundError(f"Filter file not found: {filepath}")

            tbl = ascii.read(filepath)
            wave = np.asarray(tbl[tbl.colnames[0]])  # wavelength
            trans = np.asarray(tbl[tbl.colnames[1]])  # transmission

            # Normalize transmission if needed
            if np.nanmax(trans) > 1.0:
                trans = trans / np.nanmax(trans)

            filt = Filter(
                wave * unit['AA'],
                trans,
                name=filter_name
            )
            return filt
           
        filterset = np.atleast_1d(filterset)
        from pathlib import Path
        transmission_dir = Path(self.helper.config['SYNPHOT_FILTERDIR'])
        transmission_files = {file.stem:file for file in list(transmission_dir.glob('*'))}
        available_filters_in_config = sorted(list(transmission_files.keys()))
        if 'medium' in filterset:
            # Delete 'medium' from filterset array.
            filterset = filterset[filterset != 'medium']
            medium_filters = [filter_name for filter_name in available_filters_in_config if filter_name.startswith('m')]
            filterset = np.concatenate([filterset, medium_filters])
        
        # Check whether the filterset is in the available_filters_in_config
        photometry_filters = dict()
        pyphot_lib = None
        for filter_ in filterset:
            if filter_ in pyphot_filters.keys():
                photometry_filters[filter_] = pyphot_filters[filter_]
            else:
                if filter_ in available_filters_in_config:
                    photometry_filters[filter_] = load_filter_from_config(transmission_files[filter_])
                else:
                    if pyphot_lib is None:
                        pyphot_lib = pyphot.get_library()
                    try:
                        photometry_filters[filter_] = pyphot_lib[filter_]
                    except:
                        raise ValueError(f"Filter {filter_} is not found in the library")

        phot_tbl = dict()
        FILTER_BANDWIDTH = PhotometricSpectrum.FILTER_BANDWIDTH_NM 
        FILTER_CENTER_WAVELENGTH = PhotometricSpectrum.FILTER_PIVOT_WAVELENGTH_NM 
        spec_min = self.wavelength.min()
        spec_max = self.wavelength.max()
        for filter_, filt_pyphot in photometry_filters.items():
            # print(filter_, ': ', filt_pyphot.lpivot.value)
            phot_tbl[filter_] = dict()
            phot_tbl[filter_]['wl_pivot'] = FILTER_CENTER_WAVELENGTH[filter_] * u.nm
            phot_tbl[filter_]['bandwidth'] = FILTER_BANDWIDTH[filter_] * u.nm
            
            filt_min = FILTER_CENTER_WAVELENGTH[filter_] * u.nm - FILTER_BANDWIDTH[filter_] * u.nm / 2
            filt_max = FILTER_CENTER_WAVELENGTH[filter_] * u.nm + FILTER_BANDWIDTH[filter_] * u.nm / 2
            
            overlap_min = np.maximum(filt_min, spec_min)
            overlap_max = np.minimum(filt_max, spec_max)

            overlap_width = overlap_max - overlap_min
            filt_width = filt_max - filt_min

            overlap_frac = (overlap_width / filt_width).value
            
            if overlap_frac < 0.9:
                phot_tbl[filter_]['mag'] = np.nan
                phot_tbl[filter_]['flux'] = np.nan
                continue
            

            # 1. Flux calculation
            flux = filt_pyphot.get_flux(self.wavelength.value * unit['AA'], 
                                        self.flamb.flux.value * unit['ergs/s/cm**2/AA'])
            
            # 2. Flux error calculation
            mag_err = np.nan # 기본값
            if self.is_exist_fluxerr:
                wav = self.wavelength.value
                trans = filt_pyphot.transmit
                t_interp = np.interp(wav, filt_pyphot.wavelength.value, trans, left=0, right=0)
                
                integrand_weight = t_interp * wav
                
                d_lambda = np.gradient(wav)
                numerator = np.sqrt(np.sum((self.flamb.uncertainty.array * integrand_weight * d_lambda)**2))
                denominator = np.sum(integrand_weight * d_lambda)
                
                flux_err = numerator / denominator
                mag_err = 1.0857 * (flux_err / flux)

            # 3. Magnitude calculation
            if filter_ in 'UBVRI':
                mag = -2.5 * np.log10(flux) - filt_pyphot.Vega_zero_mag
            else:
                mag = -2.5 * np.log10(flux) - filt_pyphot.AB_zero_mag
            
            phot_tbl[filter_]['mag'] = np.round(mag, 4)
            phot_tbl[filter_]['mag_err'] = np.round(mag_err, 4) if not np.isnan(mag_err) else np.nan
            phot_tbl[filter_]['flux'] = flux
            phot_tbl[filter_]['flux_err'] = flux_err if self.is_exist_fluxerr else np.nan
        
        if visualize:
            FILTER_COLOR = LightCurve.FILTER_COLOR
            # Create a figure and axis for the spectrum
            if ax is None:
                fig, ax = plt.subplots(figsize = (10,6))
            else:
                fig = ax.figure
                ax = ax
                
            title = 'Spectrum with Filter Transmission Curves' if visualize_transmission else 'Spectrum with Filter Magnitudes'
            ax.set_title(title)
            ax.set_xlabel('Wavelength (Å)')
            if visualize_unit.upper() == 'AB':
                ax.set_ylabel('Magnitude (AB)', color='k')
                if visualize_spectrum:
                    self.show(show_flux_unit = 'AB', smooth_factor = 1, log = False, color = 'k', ax = ax)
                if not ax.yaxis_inverted():
                    ax.invert_yaxis()
                    
            else:
                ax.set_ylabel('Flux Density (erg s-1 cm-2 Å-1)', color='k')   
                if visualize_spectrum:
                    self.show(show_flux_unit = 'flamb', smooth_factor = 1, log = False, color = 'k', ax = ax)
                    
            ax.tick_params(axis='y', labelcolor='k')
            
            ax.set_xlim(np.min(self.wavelength.value) -1000, np.max(self.wavelength.value) + 1000)

            # Create a secondary axis for filter transmission
            ax_transmission = None 
            if visualize_transmission:
                ax_transmission = ax.twinx()
                ax_transmission.set_ylabel('Filter Transmission', color='k')
                ax_transmission.tick_params(axis='y', labelcolor='k')
            
            for filter_, filt_pyphot in photometry_filters.items():
                # Check if the transmission is in percentage or fraction and normalize
                transmission = filt_pyphot.transmit
                magnitude = phot_tbl[filter_]['mag']
                flux = phot_tbl[filter_]['flux']
                pivot_wl = filt_pyphot.lpivot.value
                shape = 'D' if filter_.startswith('m') else 's'
                color = FILTER_COLOR[filter_]
                
                if np.max(transmission) > 5:  # Assuming if max value is greater than 1, it is in percentage
                    transmission_normalized = transmission / 100.0
                else:  # It is already in fraction form
                    transmission_normalized = transmission
                    
                # Plot the photometric point on the spectrum axis
                label_photometry = f'{filter_}: {magnitude:.2f}' if visualize_photometry_label else None
                if visualize_unit.upper() == 'AB':
                    ax.scatter(pivot_wl, magnitude, label=label_photometry, color=color, marker=shape)
                    if self.is_exist_fluxerr:
                        ax.errorbar(
                            pivot_wl,
                            phot_tbl[filter_]['mag'],
                            yerr=phot_tbl[filter_].get('mag_err', None),
                            fmt=shape,
                            color=color,
                            capsize=2
                        )
                else:
                    ax.scatter(pivot_wl, flux, label=label_photometry, color=color, marker=shape)
                    if self.is_exist_fluxerr:
                        ax.errorbar(
                        pivot_wl,
                        flux,
                        yerr=phot_tbl[filter_].get('flux_err', None),
                        fmt=shape,
                        color=color,
                        capsize=2
                        )
                # Plot the normalized filter transmission curve on the secondary axis
                if visualize_transmission:
                    ax_transmission.plot(filt_pyphot.wavelength.value, transmission_normalized, alpha=0.5, color=color)

            ax.legend(
                loc='center left',
                bbox_to_anchor=(1.1, 0.5),
                ncol=2,
                fontsize=8
            )
            return phot_tbl, photometry_filters, fig, ax, ax_transmission
        
        else:
            return phot_tbl, photometry_filters, None, None, None
        
    def _load_input(self, path, wavelength, flux, fluxerr, wavelength_unit, flux_unit, verbose: bool = True):
        if path is None and (wavelength is None or flux is None):
            raise ValueError("Either path or (wavelength, flux) must be provided")

        if path is not None:
            specfile = SpectrumFile(path, flux_unit, wavelength_unit, verbose)
            return (
                specfile.wavelength,
                specfile.flux,
                specfile.fluxerr,
                specfile.header,
                specfile.obsdate,
                wavelength_unit or getattr(specfile, 'wavelength_unit', None),
                flux_unit or getattr(specfile, 'flux_unit', None),
            )

        # array-based input
        return wavelength, flux, fluxerr, None, None, wavelength_unit, flux_unit
    
    def _resolve_units(self, wavelength, flux, header, wavelength_unit, flux_unit, verbose: bool = True):
        import warnings

        if flux_unit is None:
            flux_unit = SpectrumFile._estimate_flux_unit(flux, header, verbose)
            if flux_unit is None:
                raise ValueError("Flux unit is not provided and cannot be estimated")
            warnings.warn(f"Flux unit estimated from data: {flux_unit}")

        if wavelength_unit is None:
            wavelength_unit = SpectrumFile._estimate_wavelength_unit(wavelength, header, verbose)
            if wavelength_unit is None:
                raise ValueError("Wavelength unit is not provided and cannot be estimated")
            warnings.warn(f"Wavelength unit estimated from data: {wavelength_unit}")

        return wavelength_unit, flux_unit

    def _build_spectrum(self):
        # Set the wavelength unit
        if self._wavelength_unit.upper() not in ['AA', 'NM', 'UM']:
            raise ValueError(f"{self._wavelength_unit} is not a supported wavelength unit")
        if self._wavelength_unit.upper() == 'AA':
            wavelength_unit = u.AA
        elif self._wavelength_unit.upper() == 'NM':
            wavelength_unit = u.nm
        elif self._wavelength_unit.upper() == 'UM':
            wavelength_unit = u.um
        wavelength = self._wavelength_data * wavelength_unit
        # Set the flux unit
        if self._flux_unit.upper() not in ['FNU', 'FLAMB', 'JY', 'MJY']:
            raise ValueError(f"{self._flux_unit} is not a supported flux unit")
        if self._flux_unit.upper() == 'FNU':
            f_unit = 'erg cm-2 s-1 /Hz'
        elif self._flux_unit.upper() == 'FLAMB':
            f_unit = 'erg cm-2 s-1 /AA'
        elif self._flux_unit.upper() == 'JY':
            f_unit = 'Jy'
        elif self._flux_unit.upper() == 'MJY':
            f_unit = 'mJy'
        flux = self._flux_data * u.Unit(f_unit)
        if self.is_exist_fluxerr:
            fluxerr = self._fluxerr_data * u.Unit(f_unit)
            self.fluxerr = fluxerr
            fluxerr = StdDevUncertainty(fluxerr)
        else:
            fluxerr = None    
            self.fluxerr = fluxerr
        spec = Spectrum1D(spectral_axis = wavelength, flux = flux, uncertainty = fluxerr)
        self.spectrum = spec
        self.wavelength = self.spectrum.wavelength
        self.flux = self.spectrum.flux
    
    @property
    def fnu(self):
        f_unit = 'erg cm-2 s-1 /Hz'
        return self.spectrum.with_flux_unit(u.Unit(f_unit))
    
    @property
    def flamb(self):
        f_unit = 'erg cm-2 s-1 /AA'
        return self.spectrum.with_flux_unit(u.Unit(f_unit))
    
    @property
    def fjy(self):
        f_unit = 'Jy'
        return self.spectrum.with_flux_unit(u.Unit(f_unit))
    
    @property
    def fmjy(self):
        f_unit = 'mJy'
        return self.spectrum.with_flux_unit(u.Unit(f_unit))
    
    @property
    def ab(self):
        fnu_spec = self.fnu
        mag = (-2.5*np.log10(fnu_spec.flux.value) - 48.6) * u.AB
        if self.is_exist_fluxerr:
            magerr = (2.5/np.log(10) * (fnu_spec.uncertainty.array / fnu_spec.flux.value)) * u.AB
            spec_ab = Spectrum1D(spectral_axis = fnu_spec.wavelength, flux = mag, uncertainty = StdDevUncertainty(magerr))
        else:
            spec_ab = Spectrum1D(spectral_axis = fnu_spec.wavelength, flux = mag)
        return spec_ab
    
    @property
    def is_exist_fluxerr(self):
        return self._fluxerr_data is not None

# %%
