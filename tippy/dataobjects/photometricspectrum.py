#%%
from typing import List, Union
import glob
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from astropy.table import Table
from astropy.time import Time
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from matplotlib import cycler
from itertools import cycle
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from tippy.utils import SDTData
from tippy.catalog import TIPCatalog, TIPCatalogDataset
from tippy.helper import Helper
from tippy.image import ScienceImage
from tippy.utils import ExternalData

#%%
class PhotometricSpectrum(Helper):
    
    def __init__(self, source_catalogs: TIPCatalogDataset = None):
        super().__init__()
        if not isinstance(source_catalogs, TIPCatalogDataset):
            raise TypeError("source_catalogs must be an instance of TIPCatalogDataset")
        self.source_catalogs = source_catalogs
        self.data = None
        self.plt_params = self._plt_params()
        self.externaldata = ExternalData(catalog_key = None)
        
    EFFECTIVE_WAVELENGTHS_NM = {
        'm400': 400.0,
        'm412': 412.5,
        'm425': 425.0,
        'm437': 437.5,
        'm450': 450.0,
        'm462': 462.5,
        'm475': 475.0,
        'm487': 487.5,
        'm500': 500.0,
        'm512': 512.5,
        'm525': 525.0,
        'm537': 537.5,
        'm550': 550.0,
        'm562': 562.5,
        'm575': 575.0,
        'm587': 587.5,
        'm600': 600.0,
        'm612': 612.5,
        'm625': 625.0,
        'm637': 637.5,
        'm650': 650.0,
        'm662': 662.5,
        'm675': 675.0,
        'm687': 687.5,
        'm700': 700.0,
        'm712': 712.5,
        'm725': 725.0,
        'm737': 737.5,
        'm750': 750.0,
        'm762': 762.5,
        'm775': 775.0,
        'm787': 787.5,
        'm800': 800.0,
        'm812': 812.5,
        'm825': 825.0,
        'm837': 837.5,
        'm850': 850.0,
        'm862': 862.5,
        'm875': 875.0,
        'm887': 887.5,
        # SDSS ugriz (https://mfouesneau.github.io/pyphot/libcontent.html)
        'u': 355.7,
        'g': 470.2,
        'r': 617.6,
        'i': 749.0,
        'z': 894.7,
        # PS1 ugizy (https://mfouesneau.github.io/pyphot/libcontent.html)
        'g_ps1': 484.9,
        'r_ps1': 620.2,
        'i_ps1': 753.5,
        'z_ps1': 867.4,
        'y_ps1': 962.8,
        # Johnson-Cousins UBVRI (Ground based, https://mfouesneau.github.io/pyphot/libcontent.html)
        'U': 363.5,
        'B': 429.7,
        'V': 547.0,
        'R': 647.1,
        'I': 787.2,
        # 2MASS JHK (Ground based, https://mfouesneau.github.io/pyphot/libcontent.html)
        'J': 1230.3,
        'H': 1640.3,
        'K': 2202.7,
        # WISE W1-W4 (https://mfouesneau.github.io/pyphot/libcontent.html)
        'W1': 3368.0,
        'W2': 4618.0,
        'W3': 12073.0,
        'W4': 22194.0,
    }
    
    def __repr__(self):
        txt = f'PHOTOMETRIC SPECTRUM OBJECT (n_catalogs = {len(self.source_catalogs.catalogs)})\n'
        txt += str(self.plt_params)
        return txt
    
    def _plt_params(self):
        class PlotParams: 
            def __init__(self):
                self._rcparams = {
                    'figure.figsize': (20, 12),
                    'figure.dpi': 100,
                    'savefig.dpi': 300,
                    'font.family': 'serif',
                    'mathtext.fontset': 'cm',
                    'axes.titlesize': 16,
                    'axes.labelsize': 14,
                    'axes.xmargin': 0.1,
                    'axes.ymargin': 0.2,
                    'axes.prop_cycle': cycler(color=[
                        'black', 'blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray',
                        'olive', 'cyan', 'navy', 'gold', 'teal', 'coral', 'darkgreen', 'magenta'
                    ]),
                    'xtick.labelsize': 12,
                    'ytick.labelsize': 12,
                    'legend.fontsize': 9,
                    'lines.linewidth': 1.5,
                    'lines.markersize': 6,
                    'errorbar.capsize': 3,
                    'axes.grid': True,
                    'grid.alpha': 0.3,
                    'xtick.direction': 'in',
                    'ytick.direction': 'in',
                    'xtick.top': True,
                    'ytick.right': True,

                    }
                # Custom axis control
                self.xlim = [350, 925]
                self.ylim = None
                self.xticks = np.arange(400, 901, 50)
                self.yticks = None
                
                # Error bar parameters
                self.errorbar_enabled = True  # Optional switch
                self.errorbar_markersize = 7
                self.errorbar_hollow_marker = True  # True = hollow, False = filled
                self.errorbar_capsize = 3.5
                self.errorbar_elinewidth = 1.2
                

            def __getattr__(self, name):
                rc_name = name.replace('_', '.')
                if rc_name in self._rcparams:
                    return self._rcparams[rc_name]
                raise AttributeError(f"'PlotParams' object has no attribute '{name}'")

            def __setattr__(self, name, value):
                if name.startswith('_') or name in ('xlim', 'ylim', 'xticks', 'yticks', 'errorbar_capsize', 'errorbar_elinewidth', 'errorbar_markersize', 'errorbar_enabled', 'errorbar_hollow_marker'):
                    super().__setattr__(name, value)
                else:
                    rc_name = name.replace('_', '.')
                    if rc_name in self._rcparams:
                        self._rcparams[rc_name] = value
                    else:
                        raise AttributeError(f"'PlotParams' has no rcParam '{rc_name}'")
            
            def get_errorbar_kwargs(self, color, shape: str = None):     
                errorbar_kwargs = dict(
                    capsize=self.errorbar_capsize,
                    elinewidth=self.errorbar_elinewidth,
                    markersize=self.errorbar_markersize,
                )
                errorbar_kwargs['mec'] = color
                errorbar_kwargs['color'] = color
                errorbar_kwargs['fmt'] = shape
                
                if self.errorbar_hollow_marker is True:
                    errorbar_kwargs['mfc'] = 'none'
                else:
                    errorbar_kwargs['mfc'] = color

                if self.errorbar_enabled is False:
                    errorbar_kwargs['elinewidth'] = 0
                    errorbar_kwargs['capsize'] = 0
                    
                return errorbar_kwargs
            
            def update(self, **kwargs):
                self._rcparams.update(kwargs)

            def apply(self):
                import matplotlib.pyplot as plt
                return plt.rc_context(self._rcparams)

            def __repr__(self):
                txt = 'PLOT CONFIGURATION ============\n'
                for k, v in self._rcparams.items():
                    txt += f"{k.replace('.', '_')} = {v}\n"
                txt += 'Axis Limits and Ticks -----------\n'
                txt += f"xlim   = {self.xlim}\n"
                txt += f"ylim   = {self.ylim}\n"
                txt += f"xticks = {self.xticks}\n"
                txt += f"yticks = {self.yticks}\n"
                txt += 'Error Bar Configuration ---------\n'
                txt += f"errorbar_enabled = {self.errorbar_enabled}\n"
                txt += f"errorbar_markersize = {self.errorbar_markersize}\n"
                
                txt += f"errorbar_capsize = {self.errorbar_capsize}\n"
                txt += f"errorbar_elinewidth = {self.errorbar_elinewidth}\n"
                
                return txt
        return PlotParams()


    def add_catalogs(self,
                     catalogs: Union[List[TIPCatalog], TIPCatalog]):
        """Add catalogs to the dataset."""
        if isinstance(catalogs, TIPCatalog):
            catalogs = [catalogs]
            
        succeeded_catalogs = []
        failed_catalogs = []
        for catalog in catalogs:
            if not catalog.is_loaded:
                load_result = catalog.load_target_img(target_img = None)
                if load_result is False:
                    original_img = catalog.find_corresponding_fits()
                    if original_img is not None:
                        target_img = ScienceImage(original_img, telinfo = self.helper.estimate_telinfo(original_img), load=True)
                        load_result = catalog.load_target_img(target_img=target_img)
            if catalog.is_loaded:
                print(f"Loaded catalog: {catalog_file}")
                succeeded_catalogs.append(catalog)
            else:
                print(f"[ERROR] Failed to load catalog: {catalog_file}")
                failed_catalogs.append(catalog_file)
            
        previous_source_catalogs = self.source_catalogs
        if previous_source_catalogs is None:
            self.source_catalogs = TIPCatalogDataset(catalogs=succeeded_catalogs)
        else:
            previous_catalogs = previous_source_catalogs.catalogs
            updated_catalogs = previous_catalogs + succeeded_catalogs
            self.source_catalogs = TIPCatalogDataset(catalogs=updated_catalogs)
        print(f"Total {len(succeeded_catalogs)} catalogs loaded successfully.")
    
    def update_data(self,
                    ra_key: str = 'X_WORLD',
                    dec_key: str = 'Y_WORLD',
                    max_distance_arcsec: float = 2,
                    join_type: str = 'outer',
                    data_keys: list = ['MAGSKY_AUTO', 'MAGERR_AUTO', 'MAGSKY_APER', 'MASERR_APER', 'MAGSKY_APER_1', 'MAGERR_APER_1', 'MAGSKY_APER_2', 'MAGERR_APER_2', 'MAGSKY_APER_3', 'MAGERR_APER_3', 'MAGSKY_CIRC', 'MAGERR_CIRC']):
        self.data, self.metadata = self.source_catalogs.merge_catalogs(
            max_distance_arcsec=max_distance_arcsec,
            ra_key=ra_key,
            dec_key=dec_key,
            join_type=join_type,
            data_keys=data_keys
        )
        
    def search_sources(self, 
                       ra: Union[float, list, np.ndarray],
                       dec: Union[float, list, np.ndarray],
                       matching_radius_arcsec: float = 5.0):
        """
        Match input positions to a catalog using cKDTree (fast) for both pixel and sky coordinates.
        Returns matched catalog sorted by separation, along with separations (arcsec) and indices.
        """
        if self.data is None:
            self.update_data(max_distance_arcsec=matching_radius_arcsec)

        ra = np.atleast_1d(ra)
        dec = np.atleast_1d(dec)
        input_coords = SkyCoord(ra=ra, dec=dec, unit='deg')
        
        target_catalog = self.data
        catalog_coords = target_catalog['coord']
        
        matched_catalog, matched_input, unmatched_catalog = self.cross_match(catalog_coords, input_coords, matching_radius_arcsec)
        print(f"Matched {len(matched_catalog)} sources out of {len(input_coords)} input positions.")
        return target_catalog[matched_catalog]

    def plot(self, 
             ra: float,
             dec: float,
             matching_radius_arcsec: float = 5.0,
             flux_key: str = 'MAGSKY_APER_1',
             fluxerr_key: str = 'MAGERR_APER_1',
             color_key: str = 'OBSERVATORY',
             overplot_gaiaxp: bool = False,
             overplot_sdss: bool = False,
             overplot_ps1: bool = False
             ):
        """
        Plot photometric spectrum for the given match_id.
        """
        matched_sources = self.search_sources(ra, dec, matching_radius_arcsec=matching_radius_arcsec)
        if flux_key + f'_idx0' not in matched_sources.colnames:
            self.update_data(data_keys=[flux_key, fluxerr_key])
            matched_sources = self.search_sources(ra, dec, matching_radius_arcsec=matching_radius_arcsec)
            
        if len(matched_sources) == 0:
            print(f"[WARNING] No sources found within {matching_radius_arcsec}\" of RA={ra}, Dec={dec}")
            return [], []
            
        figs, axes = [], []
        for idx, source in enumerate(matched_sources):
            with self.plt_params.apply():
                color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

                fig, ax = plt.subplots(figsize=(8,6))
                coord = source['coord']
                ra_str = coord.ra.to_string(unit=u.hourangle, sep='', pad=True, precision=2)   # HHMMSS.ss
                dec_str = coord.dec.to_string(sep='', alwayssign=True, pad=True, precision=1)  # ±DDMMSS.s
                jname = f'J{ra_str}{dec_str}'
                wavelengths = []
                fluxes = []
                errors = []
                labels = []
                groups = []
                
                for key, meta in self.metadata.items():
                    target_img = self.source_catalogs.target_catalogs[key].target_img
                    filter_name = meta['filter']
                    observatory_name = meta.get('observatory', 'Unknown')
                    obsdate_name = meta.get('obsdate', 'Unknown')
                    obsdate_name = Time(obsdate_name).to_value('iso', subfmt = 'date')
                    eff_wl = self.EFFECTIVE_WAVELENGTHS_NM.get(filter_name, None)

                    flux_val = source.get(flux_key + f'_idx{key}')
                    flux_err = source.get(fluxerr_key + f'_idx{key}')
                    if fluxerr_key.replace('MAGERR', 'ZPERR') in target_img.header:
                        zp_err = target_img.header[fluxerr_key.replace('MAGERR', 'ZPERR')]
                    else:
                        zp_err = None
                    wavelengths.append(eff_wl)
                    if flux_val is None:
                        flux_val = np.nan
                    else:
                        pass     
                    fluxes.append(flux_val)
                    if flux_err is None or zp_err is None:
                        flux_err = np.nan
                    else:
                        flux_err = np.sqrt(flux_err**2 + zp_err**2)
                    errors.append(flux_err)
                    labels.append(obsdate_name)
                    # Determine group label (color by)
                    if color_key.lower() == 'obsdate':
                        group_val = obsdate_name
                    else:
                        group_val = observatory_name
                        color_key = 'OBSERVATORY'
                    groups.append(group_val)
                    
                # Sorting by wavelength
                wavelengths, fluxes, errors, labels, groups = zip(*sorted(zip(wavelengths, fluxes, errors, labels, groups)))

                # Unique groups >>> colors
                unique_groups = sorted(set(groups))
                obs_mjds = np.array([Time(g).mjd for g in unique_groups])
                norm = Normalize(vmin=obs_mjds.min(), vmax=obs_mjds.max())
                cmap = plt.cm.plasma
                scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
                group_colors = cmap(norm(obs_mjds))
                
                offset = 0
                for grp_idx, grp in enumerate(unique_groups):
                    idx_grp = [i for i, g in enumerate(groups) if g == grp]
                    x = np.array([wavelengths[i] for i in idx_grp])
                    y = np.array([fluxes[i] for i in idx_grp])
                    y += offset
                    yerr = np.array([errors[i] for i in idx_grp])
                    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr)
                    x, y, yerr = x[valid], y[valid], yerr[valid]
                    print(f"[{grp}] valid points: {np.sum(valid)}")


                    color = group_colors[grp_idx]
                    ax.plot(x,y, c = 'k', alpha = 0.3)
                    ax.errorbar(x, y, yerr=yerr,
                                label=grp + f"(+{offset})" if offset !=0 else grp,
                                **self.plt_params.get_errorbar_kwargs(color=color, shape='s'))
                    offset += 2
                    
                ax.set_xlabel("Effective Wavelength [nm]")
                ax.set_ylabel("Magnitude (+ offset)" if "MAG" in flux_key.upper() else "Flux")
                ax.invert_yaxis() if "MAG" in flux_key.upper() else None
                ax.grid(True, which='major', alpha=0.3)
                
                ax.set_title(f"{jname}")
                if self.plt_params.xlim:
                    ax.set_xlim(*self.plt_params.xlim)
                if self.plt_params.ylim:
                    ax.set_ylim(*self.plt_params.ylim)
                if self.plt_params.xticks is not None:
                    ax.set_xticks(self.plt_params.xticks)
                if self.plt_params.yticks is not None:
                    ax.set_yticks(self.plt_params.yticks)
                ax.minorticks_on()
                
                if overplot_gaiaxp:
                    if 'FLUX' in flux_key.upper():
                        print('Only AB magnitudes can be plotted with GAIAXP spectra')
                    else:
                        self.externaldata.change_catalog('GAIAXP')
                        gaiaxp_data = self.externaldata.query_catalog(coord = coord, radius_arcsec=matching_radius_arcsec)
                        if len(gaiaxp_data) > 0:
                            gaiaxp_spec_tbl = gaiaxp_data[0]
                            closest_source_id = gaiaxp_spec_tbl['Source'][0]
                            gaiaxp_spec_tbl = gaiaxp_spec_tbl[gaiaxp_spec_tbl['Source'] == closest_source_id]
                            wl_nm = gaiaxp_spec_tbl['lambda']
                            flux_SI = gaiaxp_spec_tbl['Flux']
                            fluxerr_SI = gaiaxp_spec_tbl['e_Flux']
                            wl_AA = wl_nm * 10  # Convert nm to Angstrom
                            mag = self.flambSI_to_ABmag(flux_SI, wl_AA)
                            magerr = self.fluxerr_to_magerr(flux = flux_SI, fluxerr = fluxerr_SI)
                            valid_mask = np.array(magerr) >= 0
                            wl_nm = np.array(wl_nm)[valid_mask]
                            mag = np.array(mag)[valid_mask]
                            magerr = np.array(magerr)[valid_mask]
                            ax.errorbar(wl_nm, mag, yerr=magerr, fmt='None', color='magenta', label='GaiaXP', alpha = 0.3)
                            ax.legend(loc = 'best')
                        else:
                            ax.plot([], [], ' ', label='GaiaXP (No data)')  # Invisible point with label
                            print(f"[WARNING] No GAIAXP data found for {jname} within {matching_radius_arcsec}\"")
                            overplot_gaiaxp = False
                            
                if overplot_sdss:
                    if 'FLUX' in flux_key.upper():
                        print('Only AB magnitudes can be plotted with SDSS spectra')
                    else:
                        self.externaldata.change_catalog('SDSS')
                        sdss_data = self.externaldata.query_catalog(coord = coord, radius_arcsec=matching_radius_arcsec)
                        if len(sdss_data) > 0:
                            sdss_source = sdss_data[0][0]
                            sdss_bands = ['u', 'g', 'r', 'i', 'z']
                            mags = []
                            magerrs = []
                            wls = []
                            for band in sdss_bands:
                                mag = sdss_source.get(f'{band}mag')
                                magerr = sdss_source.get(f'e_{band}mag')
                                if mag is None or magerr is None:
                                    continue
                                wl = self.EFFECTIVE_WAVELENGTHS_NM.get(band)
                                if wl is None:
                                    continue
                                mags.append(mag)
                                magerrs.append(magerr)
                                wls.append(wl)
                            valid_mask = np.array(magerrs) >= 0
                            wls = np.array(wls)[valid_mask]
                            mags = np.array(mags)[valid_mask]
                            magerrs = np.array(magerrs)[valid_mask]
                            ax.errorbar(wls, mags, yerr=magerrs,
                                       label='SDSS', **self.plt_params.get_errorbar_kwargs('green', '^'))

                        else:
                            ax.plot([], [], ' ', label='SDSS (No data)')  # Invisible point with label
                            print(f"[WARNING] No SDSS data found for {jname} within {matching_radius_arcsec}\"")
                            overplot_sdss = False

                if overplot_ps1:
                    if 'FLUX' in flux_key.upper():
                        print('Only AB magnitudes can be plotted with SDSS spectra')
                    else:
                        self.externaldata.change_catalog('PS1')
                        ps1_data = self.externaldata.query_catalog(coord = coord, radius_arcsec=matching_radius_arcsec)
                        if len(ps1_data) > 0:
                            ps1_source = ps1_data[0][0]
                            sdss_bands = ['u', 'g', 'r', 'i', 'z', 'y']
                            mags = []
                            magerrs = []
                            wls = []
                            for band in sdss_bands:
                                mag = ps1_source.get(f'{band}mag')
                                magerr = ps1_source.get(f'e_{band}mag')
                                if mag is None or magerr is None:
                                    continue
                                wl = self.EFFECTIVE_WAVELENGTHS_NM.get(band)
                                if wl is None:
                                    continue
                                mags.append(mag)
                                magerrs.append(magerr)
                                wls.append(wl)
                            valid_mask = np.array(magerrs) >= 0
                            wls = np.array(wls)[valid_mask]
                            mags = np.array(mags)[valid_mask]
                            magerrs = np.array(magerrs)[valid_mask]

                            ax.errorbar(wls, mags, yerr=magerrs,
                                       label='PS1',  **self.plt_params.get_errorbar_kwargs('blue', 'o'))

                        else:
                            ax.plot([], [], ' ', label='PS1 (No data)')  # Invisible point with label
                            print(f"[WARNING] No SDSS data found for {jname} within {matching_radius_arcsec}\"")
                            overplot_ps1 = False

                if (len(unique_groups) > 1) or (overplot_gaiaxp or overplot_sdss, overplot_ps1):
                    ax.legend(loc='best', ncol = 2)
                figs.append(fig)
                axes.append(ax)
                plt.show()
        return figs, axes, matched_sources

        
    
#%%
if __name__ == "__main__":
    source_catalogs = TIPCatalogDataset()
    source_catalogs.search_catalogs(
        target_name = 'T01462',
        search_key = 'calib*100.com.fits.circ.cat'
     )    
# %%
if __name__ == "__main__":
    ra = 233.857430764 
    dec = 12.0577222937
    ra = 233.7658333
    dec = 11.9574303
    ra = 233.322342 # S250206dm for T01462
    dec = -68.007909
    source_catalogs.select_catalogs(obs_start = '2025-02-12', obs_end = '2025-02-13')
    #source_catalogs.select_sources(ra, dec, radius = 10)

#%%
if __name__ == "__main__":
    self = PhotometricSpectrum(source_catalogs)
#%%
    self.update_data()
    self.plt_params.ylim = [25, 17]
# %%
if __name__ == "__main__":

    flux_key = 'MAGSKY_APER_1'
    fluxerr_key = 'MAGERR_APER_1'
    matching_radius_arcsec = 10
    color_key: str = 'OBSDATE'
    overplot_gaiaxp = False
    overplot_sdss = False
    overplot_ps1 = False
    figs, axs, matched_sources = self.plot(ra, 
                          dec, 
                          flux_key=flux_key, 
                          fluxerr_key =fluxerr_key,
                          color_key = color_key, 
                          matching_radius_arcsec=matching_radius_arcsec,
                          overplot_gaiaxp=overplot_gaiaxp,
                          overplot_sdss = overplot_sdss,
                          overplot_ps1 = overplot_ps1)
# %%
