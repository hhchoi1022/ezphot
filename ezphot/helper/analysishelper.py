
# %%
from astropy.io import fits
from astropy.io import ascii
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.table import Table

import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from tqdm import tqdm
import os
import numpy as np
from astropy.table import unique
import inspect
from astropy import constants as const

# %%

class AnalysisHelper():

    def __init__(self):
        self.c = const.c.value
        self.sigma = const.sigma_sb.cgs.value
        self.d10pc = 10 * const.pc.cgs.value
        self.c = const.c.cgs.value
        self.h = const.h.cgs.value
        self.k = const.k_B.cgs.value

    @property
    def analpath(self):
        # Get the file where the class is defined
        file_path = inspect.getfile(AnalysisHelper)

        # Convert the file path to an absolute path using os.path.abspath
        absolute_path = os.path.abspath(file_path)

        path_dir = os.path.join(os.path.dirname(absolute_path),'../analyze')

        return path_dir

    def __repr__(self):
        methods = [f'AnalysisHelper.{name}()\n' for name, method in inspect.getmembers(
            AnalysisHelper, predicate=inspect.isfunction) if not name.startswith('_')]
        txt = '[Methods]\n'+''.join(methods)
        return txt

    def mosfit_format(self,
                      mjd,
                      mag,
                      e_mag,
                      filter_,
                      observatory):
        
        formatted_tbl = Table()
        formatted_tbl['mjd'] = mjd
        formatted_tbl['mag'] = mag
        formatted_tbl['e_mag'] = e_mag
        formatted_tbl['filter'] = filter_
        formatted_tbl['telescope'] = observatory
        return formatted_tbl

    def SNcosmo_format(self,
                       mjd,
                       mag,
                       e_mag,
                       filter_,
                       magsys,
                       zp):
        tbl = Table()
        tbl['mjd'] = mjd
        tbl['band'] = filter_
        tbl['flux'] = 10**((mag -25)/-2.5)
        tbl['fluxerr'] = e_mag*tbl['flux']*2.303/2.5
        tbl['zp'] = zp
        tbl['magsys'] = magsys
        tbl.sort('band')
        tbl.sort('mjd')
        return tbl

    def load_marker_keys(self, length = None):
        '''
        parameters
        ----------
        {lenth}
        
        returns 
        -------
        1. marker_key with the length 
        
        notes 
        -----
        Order = ['o', 's', '^', 'p', 'D', 'v', 'h', 'H', '*', '<', 'x', '>', 'd', '|', '_', ',', '.', '8', 'P', 'X', '1', '2', '3', '4']

        -----
        '''
        marker = ['o', 's', '^', 'p', 'D', 'v', 'h', 'H', '*', '<', 'x', '>', 'd', '|', '_', ',', '.', '8', 'P', 'X', '1', '2', '3', '4']

        if length == None:
            return marker
        else:
            result_marker = marker[:length]
            return result_marker

    def load_filt_keys(self, filter_key = None):
        '''
        parameters
        ----------
        the {list of filter}
        
        returns 
        -------
        1. color_key(matplotlib)
        2. offset_key(offset bw/ filters)
        3. filter_key(sncosmo)
        4. filter_key(pyphot)
        5. name_key(color_key+ offset_key)
        
        notes 
        -----
        5. name_key is for visualizing labels when plotting 
        -----
        '''
        color_key = dict(
                        U = 'cyan', 
                        B = 'b',
                        V = 'g',
                        R = 'r',
                        I = 'k',
                        u = 'magenta',
                        g = 'lightseagreen',
                        r = 'orange',
                        i = 'lightcoral',
                        z = 'gray',
                        ATLAS_c = 'darkcyan',
                        ATLAS_o = 'darkorange',
                        )
        offset_key = dict(
                        U = -4, 
                        B = -2,
                        V = 0,
                        R = 1,
                        I = 3,
                        u = -5,
                        g = -1,
                        r = 1,
                        i = 2,
                        z = 5,
                        ATLAS_c = -1.5,
                        ATLAS_o = 1.5,
                        )
        
        salt2_key = dict(
                        U = 'standard::u',
                        B = 'standard::b',
                        V = 'standard::v',
                        R = 'standard::r',
                        I = 'standard::i',
                        u = 'sdssu',
                        g = 'sdss::g',
                        r = 'sdss::r',
                        i = 'sdss::i',
                        z = 'sdss::z',
                        ATLAS_c = 'atlasc',
                        ATLAS_o = 'atlaso'
                        )
        
        pyphot_key = dict(
                        U = 'GROUND_JOHNSON_U',
                        B = 'GROUND_JOHNSON_B',
                        V = 'GROUND_JOHNSON_V',
                        R = 'GROUND_COUSINS_R',
                        I = 'GROUND_COUSINS_I',
                        u = 'SDSS_u',
                        g = 'SDSS_g',
                        r = 'SDSS_r',
                        i = 'SDSS_i',
                        z = 'SDSS_z',
                        ATLAS_c = 'sdssg',
                        ATLAS_o = 'sdssr'
                        )

        if filter_key ==None:    
            filter_key = [
                        'U',
                        'B',
                        'V',
                        'R',
                        'I',
                        'u',
                        'g',
                        'r',
                        'i',
                        'z',
                        'ATLAS_c',
                        'ATLAS_o'
                        ]
        result_color = {k: color_key[k] for k in filter_key if k in color_key}
        result_offset = {k: offset_key[k] for k in filter_key if k in offset_key}
        result_saltkey = {k: salt2_key[k] for k in filter_key if k in salt2_key}
        result_pyphotkey = {k: pyphot_key[k] for k in filter_key if k in pyphot_key}
        label_row = {k: k+'{:+}'.format(offset_key[k]) for k in filter_key if k in color_key}
        return result_color, result_offset, result_saltkey, result_pyphotkey, label_row

    def read_HESMA(self, filename):
        '''
        parameters
        ----------
        HESMA {filename}
        
        returns 
        -------
        there are various type of files in HESMA database.
        
        [[[[Spectrum file]]]]
        1. wavelength
        2. phase[days]
        3. data table in flux_lambda [erg/s/cm^2/AA]
        4. data table in flux_nu [erg/s/cm^2/Hz]
        5. data table in AB magnitude
        6. synthetic photometry data table in AB magnitude 
        
        [[[[Bolometric light curve file]]]]
        1. phase[days]
        2. data table in flux [erg/s]

        [[[[early light curve file]]]]
        1. phase[days]
        2. data table in AB magnitude
        notes 
        -----
        
        -----
        '''      
           
        def read_spectbl(filename,
                        filter_key ='UBVRIugriz'):
            tbl = ascii.read(filename)
            days = list(tbl[0])
            for i, colname in enumerate(tbl.copy().columns):
                tbl.rename_column(colname, days[i])
            tbl = tbl[1:]
            wl = list(tbl['0.0'])
            tbl.remove_column('0.0')
            f_lamb_tbl = tbl
            days = days[1:]
            f_nu_tbl = Table()
            mag_tbl = Table()
            for day in days:
                f_lamb = np.array(list(f_lamb_tbl[str(day)]))
                f_nu = lflux_to_nuflux(f_lamb, wl)
                mag = fnu_to_mag(f_nu)
                f_nu_tbl[f'{day}'] = f_nu
                mag_tbl[f'{day}'] = mag
            lib = pyphot.get_library()
            _, _, _, pyphot_key, _ = load_filt_keys(filter_key)
            synth_phot = Table()
            synth_phot['filter'] = list(filter_key)
            synth_phot.add_index('filter')
            for day in days:
                magset = []
                for filt_ in filter_key:
                    filt_pyphot = lib[pyphot_key[filt_]]
                    flux = filt_pyphot.get_flux(wl*unit['AA'],f_lamb_tbl[str(day)]*unit['ergs/s/cm**2/AA'], axis = 1)
                    mag = -2.5*np.log10(flux.value) - filt_pyphot.AB_zero_mag
                    magset.append(mag)
                synth_phot[f'{day}'] = magset
            return wl, days, f_lamb_tbl, f_nu_tbl, mag_tbl, synth_phot
        
        def read_lctbl(filename):
            result_tbl = ascii.read(filename)
            result_tbl.rename_column('col1','days')
            result_tbl.rename_column('col2','luminosity')
            days = result_tbl['days']
            return days, result_tbl
        
        def read_earlylctbl(filename):
            result_tbl = ascii.read(filename)
            return result_tbl
        
        if 'spectra' in filename:
            return read_spectbl(filename)
        elif 'lightcurve' in filename:
            if 'early' in filename:
                return read_earlylctbl(filename)
            else:
                return read_lctbl(filename)
        else:
            raise ValueError(f'{os.path.basename(filename)} cannot be interpreted')

    def read_Polin2019(self, filename):
        '''
        parameters
        ----------
        {filename} of Polin 2019 data to be read
        
        returns 
        -------
        1. data table

        notes 
        -----
        This is specialized to read the table from Polin 2019.
        -----
        '''
        header = ['phase', 'u', 'g', 'r', 'i', 'z', 'U', 'B', 'V', 'R', 'I']
        tbl = ascii.read(filename, format = 'fixed_width')
        tbl.rename_columns(tbl.colnames, header)
        return tbl

    def read_Polin2019_spec(self, filename):
        '''
        parameters
        ----------
        {filename} of Polin 2019 spectrum data to read
        
        returns 
        -------
        1. data table
        2. phase

        notes 
        -----
        This is specialized to read the table from Polin 2019.
        -----
        '''

        import h5py
        from astropy.table import hstack
        f = h5py.File(filename, 'r')
        Lnu = f['Lnu'][:]    
        mu = f['mu'][0]
        nu = f['nu'][:]
        lamb = self.nu_to_lamb(nu)
        Llamb = self.nuflux_to_lflux(Lnu, lamb)
        time = f['time'][:]/86400
        Lnu = np.flip(Lnu, axis = 1)
        Llamb = np.flip(Llamb, axis = 1)
        nu = np.flip(nu)
        lamb = np.flip(lamb)
        Llamb /= 4*np.pi* (10*3.086e18)**2
        Lnu /= 4*np.pi* (10*3.086e18)**2
        return Lnu, Llamb, lamb, time, mu

    def nu_to_lamb(self, nu):
        '''
        parameters
        ----------
        1. nu : frequency (Hz)
        
        returns 
        -------
        1. wl_AA : wavelength (AA)
        
        notes 
        -----
        lamb = c(speed of light) / nu
        -----
        '''
        wl_AA = self.c * 1e8 / nu
        return wl_AA

    def lamb_to_nu(self, wl_AA):
        '''
        parameters
        ----------
        1. wl_AA : wavelength (AA)
        
        returns 
        -------
        1. nu : frequency (Hz)
        
        notes 
        -----
        nu = c(speed of light) / wl_AA
        -----
        '''
        nu = self.c * 1e8 / wl_AA
        return nu

    def lflux_to_nuflux(self, f_lamb, wl_AA):
        '''
        parameters
        ----------
        1. f_lamb : flux [erg/s/cm^2/AA]
        2. Wavelength : [AA]
        
        returns 
        -------
        1. f_nu : flux [erg/s/cm^2/Hz]
        
        notes 
        -----
        -----
        '''
        f_nu = f_lamb  * wl_AA * wl_AA*1e-8 / c
        return f_nu

    def nuflux_to_lflux(self, f_nu, wl_AA):
        '''
        parameters
        ----------
        1. nuflux [erg/s/cm^2/Hz]
        2. wl_AA : wavelength [AA]
        
        returns 
        -------
        1. f_lamb : flux [erg/s/cm^2/AA]
        
        notes 
        -----
        -----
        '''
        f_lamb = f_nu * c / wl_AA / wl_AA / 1e-8
        return f_lamb

    def fnu_to_ABmag(self, f_nu):
        '''
        parameters
        ----------
        {f_nu[erg/s/cm^2/Hz]}
        
        returns 
        -------
        1. AB magnitude
        
        notes 
        -----
        -----
        '''
        mag = -2.5*np.log10(f_nu)-48.6
        return mag

    def flamb_to_ABmag(self, f_lamb, wl_AA):

        '''
        parameters
        ----------
        {f_lamb[erg/s/cm^2/AA]}
        
        returns 
        -------
        1. AB magnitude
        
        notes 
        -----
        -----
        '''
        f_nu = self.nuflux_to_lflux(f_lamb, wl_AA)
        mag = -2.5*np.log10(f_nu)-48.6
        return mag
    
    def flambSI_to_ABmag(self, f_lamb_SI, wl_AA):
        """
        Convert flux density in W / (nm * m²) at a given wavelength [Å] 
        to AB magnitude.

        Parameters
        ----------
        f_lamb_SI : float or array-like
            Spectral flux density in W / (nm * m²)
        wl_AA : float or array-like
            Wavelength in Angstrom (Å)

        Returns
        -------
        ABmag : float or array-like
            AB magnitude
        """
        # Convert flux from W / (nm·m²) to erg / (s·cm²·Å)
        # 1 W = 1e7 erg/s, 1 m² = 1e4 cm², 1 nm = 10 Å
        f_lambda_cgs = f_lamb_SI * 1e7 / 1e4 / 10  # ? erg / s / cm² / Å

        # Convert to f_nu
        f_nu = f_lambda_cgs * (wl_AA**2) / const.c.cgs.value / 1e8  # Hz^-1

        # Convert to AB magnitude
        ABmag = -2.5 * np.log10(f_nu) - 48.6

        return ABmag
    
    def ABmag_to_fnu(self, ABmag):
        '''
        parameters
        ----------
        1. AB magnitude
        
        returns 
        -------
        1. fnu [erg/s/cm^2/Hz]
        
        notes 
        -----
        -----
        '''
        fnu = 10**((ABmag + 48.6)/(-2.5)) 
        return fnu

    def ABmag_to_flamb(self, ABmag, wl_AA):

        '''
        parameters
        ----------
        1. AB magnitude
        2. wavelength [AA]
        
        returns 
        -------
        1. flamb [erg/s/cm^2/Hz]
        
        notes 
        -----
        -----
        '''
        fnu = self.ABmag_to_fnu(ABmag)
        flamb = self.nuflux_to_lflux(fnu, wl_AA)
        return flamb
    
    def ABmag_to_flambSI(self, ABmag, wl_AA):
        """
        Convert AB magnitude to flux density in W / (nm * m²)
        at a given wavelength in Ångström.

        Parameters
        ----------
        ABmag : float or array-like
            AB magnitude
        wl_AA : float or array-like
            Wavelength in Angstrom (Å)

        Returns
        -------
        f_lambda_SI : float or array-like
            Spectral flux density in W / (nm * m²)
        """
        # Convert AB mag to f_nu in cgs units (erg / s / cm² / Hz)
        f_nu = 10**((-ABmag - 48.6) / 2.5)

        # Convert f_nu to f_lambda in erg / s / cm² / Å
        f_lambda_cgs = f_nu * const.c.cgs.value / (wl_AA**2) * 1e8

        # Convert f_lambda to W / (nm * m²)
        # 1 erg = 1e-7 J, 1 cm² = 1e-4 m², 1 Å = 0.1 nm
        f_lambda_SI = f_lambda_cgs * 1e-7 * 1e4 * 10  # W / nm / m²

        return f_lambda_SI

    def mag_to_flux(self, mag, zp = 25):
        '''
        parameters
        ----------
        {magnitude}
        
        returns 
        -------
        1. arbitrary flux with the zeropoint
        
        notes 
        -----
        -----

        '''
        flux = 10**((mag -zp)/-2.5)
        return flux

    def magerr_to_fluxerr(self, flux, magerr):
        """
        Convert magnitude uncertainty to flux uncertainty.

        Parameters
        ----------
        flux : float or array-like
            Flux value (e.g., in erg/s/cm^2/Hz or arbitrary units)
        magerr : float or array-like
            Magnitude uncertainty

        Returns
        -------
        fluxerr : float or array-like
            Flux uncertainty
        """
        flux = np.asarray(flux)
        magerr = np.asarray(magerr)

        fluxerr = flux * np.log(10) / 2.5 * magerr
        return fluxerr        

    def flux_to_mag(self, flux, zp = 25):
        '''
        parameters
        ----------
        {flux} with the zeropoint
        
        returns 
        -------
        1. magnitude
        
        notes 
        -----
        -----
        '''
        mag = -2.5*np.log10(flux) + zp
        return mag
    
    def fluxerr_to_magerr(self, flux, fluxerr):
        """
        Convert flux uncertainty to magnitude uncertainty.

        Parameters
        ----------
        flux : float or array-like
            Flux value
        fluxerr : float or array-like
            Flux uncertainty

        Returns
        -------
        magerr : float or array-like
            Magnitude uncertainty
        """
        flux = np.asarray(flux)
        fluxerr = np.asarray(fluxerr)
        
        # Avoid division by zero or negative flux
        with np.errstate(divide='ignore', invalid='ignore'):
            magerr = 2.5/np.log(10) * (fluxerr / flux)
            magerr = np.where(flux <= 0, np.nan, magerr)  # nan if flux is zero or negative

        return magerr
    


    def interpolate_spline(self, x, y, weight = None, k = 3, smooth = 0.05, show = False):
        '''
        parameters
        ----------
        Calculate interpolation with UnivariateSpline of the data{x}{y}
        weight : weight for each data point
        smooth : sum((w[i] * (y[i]-spl(x[i])))**2, axis=0) <= smooth
        show : show interpolated data 
        returns 
        -------
        1. {Spline function} that demonstrate data points 

        notes 
        -----
        -----
        '''

        s = UnivariateSpline(x,y, w = weight, s= smooth, k = k)
        fig = None
        if show:
            xgrid = np.arange(np.min(x),np.max(x), 0.005)
            fig = plt.figure(dpi =300)
            plt.gca().invert_yaxis()
            plt.scatter(x,y,marker= '+', s = 5, c = 'r', label  ='Raw data')
            plt.plot(xgrid, s(xgrid), c = 'k', linewidth = 1, label = 'Interpolation')
            plt.legend(loc = 0)
        return s, fig

    def interpolate_linear(self, xdata, ydata, xrange_min, xrange_max, nx = 1000):
        '''
        parameters
        ----------
        1. xdata : np.array or list
                x data array for interpolation
        2. ydata : np.array or list
                y data array for interpolation
        3. xrange_min : float
                min limit of the interpolated output
        4. xrange_max : float
                max limit of the interpolated output
        5. nx : int
                the number of interpo;lated output        
                
        returns 
        -------
        1. output : list 
                the list of two arrays(interpolated_x, interpolated_y)

        notes 
        -----
        -----
        '''
        xgrid = np.linspace(xrange_min, xrange_max, nx)
        ygrid = np.interp(xgrid, xdata, ydata)
        return [xgrid, ygrid]
    
    def planck(self, temperature : float, wl_AA = None, nu = None):
        """
        =======
        Parameters
        =======
        1. temperature : float = Effective temperature in Kelvin unit
        2. wl_AA : float = wavelength in Angstron unit
        3. nu : float = Frequency in Hz unit
        
        =======
        output
        =======
        result : dict = {'wl': w, 'nu': nu, 'fnu': fnu, 'flamb': flamb} 
        """
        
        if (wl_AA is None) & (nu is None):
            wl_AA = np.arange(100, 11000, 10)
        if wl_AA is None:
            wl_AA = self.c *1e8 / nu
        elif nu is None:
            nu = (self.c / wl_AA) * 1e8
        else:
            raise ValueError('Either wavelength or frequency should be given')
        # constants appropriate to cgs units.
        
        fnu_term1 = 2 * np.pi * self.h * nu**3 / self.c**2
        fnu_term2 = np.exp((self.h*nu)/(self.k*temperature))
        fnu = (fnu_term1 * (1/(fnu_term2 - 1)))
        
        flamb = (fnu * 3e18 / wl_AA**2)
        result = dict()
        result['wl'] = wl_AA
        result["nu"] = nu
        result['fnu'] = fnu
        result['flamb'] = flamb
        return result


# %%
AnalysisHelper().planck(10000)
# %%
