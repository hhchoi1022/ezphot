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

from tippy.utils import SDTData
from tippy.catalog import TIPCatalog, TIPCatalogDataset
from tippy.helper import Helper
from tippy.image import ScienceImage
from tippy.utils import ExternalData

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from astropy.time import Time


#%%
class LightCurve(Helper):
    
    def __init__(self, source_catalogs: TIPCatalogDataset = None):
        super().__init__()
        if not isinstance(source_catalogs, TIPCatalogDataset):
            raise TypeError("source_catalogs must be an instance of TIPCatalogDataset")
        self.source_catalogs = source_catalogs
        self.data = None
        self.plt_params = self._plt_params()
        self.externaldata = ExternalData(catalog_key = None)
        
    FILTER_OFFSET = {
        'm400': -5.0,
        'm412': -4.75,
        'm425': -4.5,
        'm437': -4.25,
        'm450': -4.0,
        'm462': -3.75,
        'm475': -3.5,
        'm487': -3.25,
        'm500': -3.0,
        'm512': -2.75,
        'm525': -2.5,
        'm537': -2.25,
        'm550': -2.0,
        'm562': -1.75,
        'm575': -1.5,
        'm587': -1.25,
        'm600': -1.0,
        'm612': -0.75,
        'm625': -0.5,
        'm637': -0.25,
        'm650': 0.0,
        'm662': 0.25,
        'm675': 0.5,
        'm687': 0.75,
        'm700': 1.0,
        'm712': 1.25,
        'm725': 1.5,
        'm737': 1.75,
        'm750': 2.0,
        'm762': 2.25,
        'm775': 2.5,
        'm787': 2.75,
        'm800': 3.5,
        'm812': 4.0,
        'm825': 4.5,
        'm837': 5.0,
        'm850': 6.0,
        'm862': 6.5,
        'm875': 8.5,
        'm887': 9.0,
        # SDSS ugriz 
        'u': -2.0,
        'g': 0,
        'r': 0,
        'i': 2.0,
        'z': 3.0,
    }
    
    # Global: Filter effective wavelengths (nm)
    FILTER_WAVELENGTHS_NM = {
        'm400': 400, 'm412': 412, 'm425': 425, 'm437': 437, 'm450': 450,
        'm462': 462, 'm475': 475, 'm487': 487, 'm500': 500, 'm512': 512,
        'm525': 525, 'm537': 537, 'm550': 550, 'm562': 562, 'm575': 575,
        'm587': 587, 'm600': 600, 'm612': 612, 'm625': 625, 'm637': 637,
        'm650': 650, 'm662': 662, 'm675': 675, 'm687': 687, 'm700': 700,
        'm712': 712, 'm725': 725, 'm737': 737, 'm750': 750, 'm762': 762,
        'm775': 775, 'm787': 787, 'm800': 800, 'm812': 812, 'm825': 825,
        'm837': 837, 'm850': 850, 'm862': 862, 'm875': 875, 'm887': 887,
    }

    # Compute normalized color map
    _wls = np.array(list(FILTER_WAVELENGTHS_NM.values()))
    _normed_wls = (_wls - _wls.min()) / (_wls.max() - _wls.min())
    _cmap = plt.cm.plasma
    _rgba_colors = _cmap(_normed_wls)
    _hex_colors = [mcolors.to_hex(c) for c in _rgba_colors]

    # ? Global dictionary
    FILTER_COLOR = dict(zip(FILTER_WAVELENGTHS_NM.keys(), _hex_colors))
    
    # Step 2: Override for broadbands (fixed colors)
    FILTER_COLOR.update({
        'u': 'cyan',
        'g': 'green',
        'r': 'red',
        'i': 'black',         
        'z': 'brown',         
        'y': 'darkorange',    
        'B': 'royalblue',
        'V': 'limegreen',
        'R': 'firebrick',
        'I': 'maroon',
    })   
    
    def __repr__(self):
        txt = f'LIGHTCURVE OBJECT (n_catalogs = {len(self.source_catalogs.catalogs)})\n'
        txt += str(self.plt_params)
        return txt
    
    def _plt_params(self):
        class PlotParams: 
            def __init__(self):
                self._rcparams = {
                    'figure.figsize': (20, 12),
                    'figure.dpi': 300,
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
                self.xlim = None
                self.ylim = [21, 8]
                self.xticks = None
                self.yticks = None
                
                # Error bar parameters
                self.errorbar_enabled = True  # Optional switch
                self.errorbar_markersize = 7
                self.errorbar_hollow_marker = False  # True = hollow, False = filled
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
                errorbar_kwargs['mec'] = 'black'
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
            data_keys=data_keys)
        
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
             color_key: str = 'FILTER',
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
            #return [], []
            
        figs, axes = [], []
        for idx, source in enumerate(matched_sources):
            with self.plt_params.apply():
                color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

                fig, ax = plt.subplots()
                coord = source['coord']
                ra_str = coord.ra.to_string(unit=u.hourangle, sep='', pad=True, precision=2)   # HHMMSS.ss
                dec_str = coord.dec.to_string(sep='', alwayssign=True, pad=True, precision=1)  # ±DDMMSS.s
                jname = f'J{ra_str}{dec_str}'
                obsdates = []
                fluxes = []
                depths = []
                errors = []
                labels = []
                groups = []
                
                for key, meta in self.metadata.items():
                    target_img = self.source_catalogs.target_catalogs[key].target_img
                    filter_name = meta['filter']
                    observatory_name = meta.get('observatory', 'Unknown')
                    telname = meta.get('telname', 'Unknown')
                    obsdate_name = meta.get('obsdate', 'Unknown')
                    obsdate_name = Time(obsdate_name).mjd

                    flux_val = source.get(flux_key + f'_idx{key}')
                    flux_err = source.get(fluxerr_key + f'_idx{key}')
                    if fluxerr_key.replace('MAGERR', 'ZPERR') in target_img.header:
                        zp_err = target_img.header[fluxerr_key.replace('MAGERR', 'ZPERR')]
                    else:
                        zp_err = None
                    
                    obsdates.append(obsdate_name)
                    if flux_val is None:
                        flux_val = np.nan
                    else:
                        flux_val += self.FILTER_OFFSET[filter_name]
                    fluxes.append(flux_val)
                    if flux_err is None or zp_err is None:
                        errors.append(np.nan)
                    else:
                        errors.append(np.sqrt(flux_err**2 + zp_err**2))
                    depths.append(meta.get('depth', np.nan))
                    
                    filter_name += "+%.1f" %self.FILTER_OFFSET[filter_name]
                    labels.append(filter_name)
                    # Determine group label (color by)
                    if color_key.lower() == 'filter':
                        group_val = filter_name
                    else:
                        group_val = telname
                        color_key = 'OBSERVATORY'
                    groups.append(group_val)

                # Sorting by wavelength
                # Get sorted indices of obsdates
                sorted_indices = np.argsort(obsdates)

                # Apply the same ordering to all arrays
                obsdates = np.array(obsdates)[sorted_indices]
                fluxes = np.array(fluxes)[sorted_indices]
                depths = np.array(depths)[sorted_indices]
                errors = np.array(errors)[sorted_indices]
                labels = np.array(labels)[sorted_indices]
                groups = np.array(groups)[sorted_indices]

                # Unique groups >>> colors
                unique_groups = sorted(set(groups))

                # Plot fluxes with error bars
                for grp in unique_groups:
                    idx_grp = [i for i, g in enumerate(groups) if g == grp]
                    x = np.array([obsdates[i] for i in idx_grp])
                    y = np.array([fluxes[i] for i in idx_grp])
                    yerr = np.array([errors[i] for i in idx_grp])
                    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr)
                    x,y,yerr = np.array(x)[valid], np.array(y)[valid], np.array(yerr)[valid]

                    # Draw hollow square markers using scatter
                    base_filter = grp.split('+')[0]
                    color = self.FILTER_COLOR.get(base_filter, next(color_cycle))
                    alpha = 0.5# if base_filter.startswith('m') else 1.0

                    ax.errorbar(x, y, yerr=yerr,
                                label=grp,
                                alpha=alpha,
                                **self.plt_params.get_errorbar_kwargs(color, 's'))

                # Plot depth only for non-detections (NaN flux)
                for grp in unique_groups:
                    idx_grp = [i for i, g in enumerate(groups) if g == grp]
                    x_all = np.array([obsdates[i] for i in idx_grp])
                    y_all = np.array([fluxes[i] for i in idx_grp])
                    d_all = np.array([depths[i] for i in idx_grp])
                    
                    # Select non-detections
                    nondet_mask = np.isfinite(x_all) & np.isnan(y_all) & np.isfinite(d_all)
                    x_nondet = x_all[nondet_mask]
                    d_nondet = d_all[nondet_mask]
                    
                    if len(x_nondet) == 0:
                        continue
                    
                    base_filter = grp.split('+')[0]
                    color = self.FILTER_COLOR.get(base_filter, next(color_cycle))

                    # Plot inverted triangles for depth
                    ax.scatter(x_nondet, d_nondet, color=color, marker='v',
                               label=rf'5$\sigma$ limit', alpha=0.7)

                ax.set_xlabel("Obsdate [MJD]")
                ax.set_ylabel("Magnitude" if "MAG" in flux_key.upper() else "Flux")
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
                            band_colors = [self.FILTER_COLOR.get(b, 'gray') for b in sdss_bands]
                            for wl, mag, magerr, band, color in zip(wls, mags, magerrs, sdss_bands, band_colors):
                                ax.errorbar([wl], [mag], yerr=[magerr],
                                            label=f'SDSS {band}', 
                                            **self.plt_params.get_errorbar_kwargs(color, '^'))
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

                            band_colors = [self.FILTER_COLOR.get(b, 'gray') for b in sdss_bands]
                            for wl, mag, magerr, band, color in zip(wls, mags, magerrs, sdss_bands, band_colors):
                                ax.errorbar([wl], [mag], yerr=[magerr],
                                            label=f'SDSS {band}', 
                                            **self.plt_params.get_errorbar_kwargs(color, '^'))
                        else:
                            ax.plot([], [], ' ', label='PS1 (No data)')  # Invisible point with label
                            print(f"[WARNING] No SDSS data found for {jname} within {matching_radius_arcsec}\"")
                            overplot_ps1 = False

                # Sort legend by broadband vs medium-band
                broadbands = ['u', 'g', 'r', 'i', 'z', 'y', 'B', 'V', 'R', 'I']

                # Get current legend labels and handles
                handles, labels = ax.get_legend_handles_labels()

                # Helper: extract base filter name (before '+')
                def get_base_filter(label):
                    return label.split('+')[0].strip()

                # Split into broadband vs others
                broadband_pairs = []
                mediumband_pairs = []

                for handle, label in zip(handles, labels):
                    base = get_base_filter(label)
                    if base in broadbands:
                        broadband_pairs.append((base, label, handle))
                    else:
                        mediumband_pairs.append((base, label, handle))

                # Sort each group
                broadband_pairs.sort(key=lambda x: broadbands.index(x[0]))
                mediumband_pairs.sort(key=lambda x: x[0])

                # Combine and unpack
                sorted_labels = [p[1] for p in broadband_pairs + mediumband_pairs]
                sorted_handles = [p[2] for p in broadband_pairs + mediumband_pairs]

                # Show sorted legend
                ax.legend(sorted_handles, sorted_labels, loc='best', ncol=2)

                # Convert MJD xticks to UTC dates
                xticks = ax.get_xticks()
                xticks_time = Time(xticks, format='mjd')
                xtick_labels = xticks_time.to_value('iso', subfmt='date')  # 'YYYY-MM-DD'
                ax.set_xticks(xticks)
                ax.set_xticklabels(xtick_labels, rotation=45)
                figs.append(fig)
                axes.append(ax)
                plt.show()
        return figs, axes, matched_sources

        
    
#%%
if __name__ == "__main__":
    source_catalogs = TIPCatalogDataset()
    source_catalogs.search_catalogs('NGC1566', 'Calib*60.com.fits.cat', folder = '/home/hhchoi1022/data/scidata/RASA36/RASA36_KL4040_HIGH_1x1')
    source_catalogs.select_sources(ra = 64.9725, dec= -54.948081, radius = 15)
        
# %%
if __name__ == "__main__":
    ra = 233.857430764 # SN2025fvw
    dec = 12.0577222937
    # ra = 233.7658333 #EB
    # dec = 11.9574303
    # ra = 234.3112500 #AGN
    # dec = 12.1974444 
    # ra = 234.2416667 #SB
    # dec = 12.0027778
    # ra = 233.9041667 #QSO
    # dec = 11.9508333
    # ra = 233.6121667  #UGC9901
    # dec = 12.2710611
    # ra = 233.8625000  # EB
    # dec = 12.103333
    ra = 233.322342 # S250206dm for T01462
    dec = -68.007909
    # ra = 259.757396
    # dec = -67.360176
    #ra = 241.62392408
    #dec = -70.327141108
    # ra = 262.154312
    # dec = -68.789571
    ra = 263.916460
    dec = -70.346012
    ra = 234.685513
    dec = -68.794466
    ra = 232.069681
    dec = -67.896978
    source_catalogs.select_catalogs(filter = ['g', 'r', 'i'], obs_start = '2025-01-01', obs_end = '2025-03-01')
    source_catalogs.select_sources(ra, dec, radius =  60)
#%%
if __name__ == "__main__":
    self = LightCurve(source_catalogs)
    self.update_data(max_distance_arcsec = 5)
# %%
if __name__ == "__main__":
    source =self.data[0]

    flux_key = 'MAGSKY_APER'
    fluxerr_key = 'MAGERR_APER'
    matching_radius_arcsec = 5
    color_key: str = 'filter'#'OBSDATE'
    overplot_gaiaxp = False
    overplot_sdss = False
    overplot_ps1 = False
    self.plt_params.figure_figsize = (10,6)
    self.plt_params.ylim = [22, 17]
#%%
if __name__ == "__main__":
    figs, axs, matched_sources = self.plot(ra, 
                          dec, 
                          flux_key=flux_key, 
                          color_key = color_key, 
                          matching_radius_arcsec=matching_radius_arcsec,
                          overplot_gaiaxp=overplot_gaiaxp,
                          overplot_sdss = overplot_sdss,
                          overplot_ps1 = overplot_ps1)
    axs[0].scatter(Time('2025-02-12T08:04:30').mjd, 19.21, c = 'red', marker='*', s=100, label='KMTNet R band')
    axs[0].legend(loc='upper right')
    


# %%
