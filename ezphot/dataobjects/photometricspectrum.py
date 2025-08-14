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
import matplotlib.cm as cm



from ezphot.dataobjects import Catalog, CatalogSet
from ezphot.helper import Helper
from ezphot.imageobjects import ScienceImage
from ezphot.utils import SDTDataQuerier
from ezphot.utils import CatalogQuerier

#%%
class PhotometricSpectrum:
    """
    Photometric spectrum object.
    
    This object is used to plot the photometric spectrum of a source.
    It is initialized with a source_catalogs object, which is a CatalogSet object.
    The source_catalogs object is used to extract the source information from the catalogs.
    
    To change the plot parameters, modify the plt_params attribute.
    """
    
    def __init__(self, source_catalogs: CatalogSet = None):
        """
        Initialize the PhotometricSpectrum object.
        
        Parameters
        ----------
        source_catalogs : CatalogSet
            The source catalogs to use for the photometric spectrum.
        """
        if not isinstance(source_catalogs, CatalogSet):
            raise TypeError("source_catalogs must be an instance of CatalogSet")
        self.helper = Helper()
        self.source_catalogs = source_catalogs
        self.catalogs_tbl = None
        self.plt_params = self._plt_params()
        self.CatalogQuerier = CatalogQuerier(catalog_key = None)
        self.data = None
        
    OFFSET = 2
        
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
    
    FILTER_SHAPE = {
        'm400': 'o',
        'm412': 'o',
        'm425': 'o',
        'm437': 'o',
        'm450': 'o',
        'm462': 'o',
        'm475': 'o',
        'm487': 'o',
        'm500': 'o',
        'm512': 'o',
        'm525': 'o',
        'm537': 'o',
        'm550': 'o',
        'm562': 'o',
        'm575': 'o',
        'm587': 'o',
        'm600': 'o',
        'm612': 'o',
        'm625': 'o',
        'm637': 'o',
        'm650': 'o',
        'm662': 'o',
        'm675': 'o',
        'm687': 'o',
        'm700': 'o',
        'm712': 'o',
        'm725': 'o',
        'm737': 'o',
        'm750': 'o',
        'm762': 'o',
        'm775': 'o',
        'm787': 'o',
        'm800': 'o',
        'm812': 'o',
        'm825': 'o',
        'm837': 'o',
        'm850': 'o',
        'm862': 'o',
        'm875': 'o',
        'm887': 'o',
        # SDSS ugriz
        'u': 's',
        'g': 's',
        'r': 's',
        'i': 's',
        'z': 's',
        # PS1 ugizy
        'g_ps1': 's',
        'r_ps1': 's',
        'i_ps1': 's',
        'z_ps1': 's',
        'y_ps1': 's',
        # Johnson-Cousins UBVRI
        'U': 's',
        'B': 's',
        'V': 's',
        'R': 's',
        'I': 's',
        # 2MASS JHK
        'J': 's',
        'H': 's',
        'K': 's',
        # WISE W1-W4
        'W1': 's',
        'W2': 's',
        'W3': 's',
        'W4': 's',
    }
    
    FILTER_COLOR = {
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
    } 
    
    def __repr__(self):
        txt = f'PHOTOMETRIC SPECTRUM OBJECT (n_catalogs = {len(self.source_catalogs.catalogs)})\n'
        txt += str(self.plt_params)
        return txt
    
    def plot(self, 
             ra: float,
             dec: float,
             objname: str = None,
             matching_radius_arcsec: float = 5.0,
             flux_key: str = 'MAGSKY_APER_1',
             fluxerr_key: str = 'MAGERR_APER_1',
             color_key: str = 'OBSDATE',        # 'OBSDATE' or 'OBSERVATORY'
             overplot_gaiaxp: bool = False,
             overplot_sdss: bool = False,
             overplot_ps1: bool = False,
             verbose: bool = True,
            ):
        """
        Plot photometric spectrum (wavelength vs magnitude/flux) for the given source.
        
        The figure parameters are set in the plt_params attribute.
        
        Parameters
        ----------
        ra : float
            Right ascension of the source in degrees.
        dec : float
            Declination of the source in degrees.
        objname : str, optional
            Name of the source.
        matching_radius_arcsec : float, optional
            Matching radius in arcseconds.
        flux_key : str, optional
            Key for the flux column.
        fluxerr_key : str, optional
            Key for the flux error column.
        color_key : str, optional
            Key for the color column.
        overplot_gaiaxp : bool, optional
            Whether to overplot GaiaXP data.
        overplot_sdss : bool, optional
            Whether to overplot SDSS data.
        overplot_ps1 : bool, optional
            Whether to overplot PS1 data.
        verbose : bool, optional
            Whether to print verbose output.
            
        Returns
        -------
        figs : list
            List of figures.
        axs : list
            List of axes.
        tbl : astropy.table.Table
            Table of data.
        """
        # Ensure per-exposure table exists
        if self.data is None:
            self.extract_source_info(
                ra, dec,
                flux_keys=[flux_key],
                fluxerr_keys=[fluxerr_key],
                matching_radius_arcsec=matching_radius_arcsec
            )
        if self.data is None or len(self.data) == 0:
            self.helper.print(f"[WARNING] No sources found within {matching_radius_arcsec}\" of RA={ra}, Dec={dec}", verbose)
            return None, None, None
        
        tbl = self.data.copy()

        # Basic info & title
        coord = tbl['coord'][0] if 'coord' in tbl.colnames else SkyCoord(ra*u.deg, dec*u.deg)
        ra_str  = coord.ra.to_string(unit=u.hourangle, sep='', pad=True, precision=2)
        dec_str = coord.dec.to_string(sep='', alwayssign=True, pad=True, precision=1)
        jname   = f'J{ra_str}{dec_str}'
        title   = objname if objname is not None else jname

        # labels / grouping
        if color_key.lower() == 'obsdate':
            # 1) ensure a 'group' column exists (whatever your group_table does)
            tbl = self.helper.group_table(tbl, 'mjd')        # must add/keep tbl['group']
            gview = tbl.group_by('group')

            # 2) compute a mean-MJD label per group
            key_vals = np.array(gview.groups.keys['group'])  # one key per group
            labels_map = {}
            for i, g in enumerate(gview.groups):
                mjd_mean = float(np.nanmean(g['mjd']))
                date_str = Time(mjd_mean, format='mjd').to_value('iso', subfmt='date_hm')
                labels_map[key_vals[i]] = date_str

            # 3) propagate to all rows
            obsdate_group = np.empty(len(tbl), dtype=object)
            for k, lab in labels_map.items():
                obsdate_group[tbl['group'] == k] = lab
            tbl['obsdate_group'] = obsdate_group

            # 4) this is the per-row groups array you sort alongside wl/vals/errs
            groups = np.array(tbl['obsdate_group'], dtype=object)

        elif color_key.lower() == 'observatory':
            groups = np.array(tbl['observatory'], dtype=object)
            color_key = 'OBSERVATORY'
        
        elif color_key.lower() == 'telname':
            groups = np.array(tbl['telname'], dtype=object)
            color_key = 'TELNAME'
        
        else:
            self.helper.print(f"[WARNING] Invalid color_key: {color_key}", verbose)
            return None, None, None
            
        # Build arrays per row
        # wavelength
        wl = np.array([self._band_to_wavelength_nm(b) for b in tbl['filter']], dtype=float)
        vals  = np.array(tbl[flux_key], dtype=float)
        merrs = np.array(tbl[fluxerr_key], dtype=float)
        zperrs = np.array(tbl['zp_err'], dtype=float)
        errs  = np.array([self._combine_err(m, z) for m, z in zip(merrs, zperrs)], dtype=float)

        # Sort by wavelength for nicer lines
        order = np.argsort(wl)
        wl, vals, errs, groups = wl[order], vals[order], errs[order], np.array(groups)[order]

        # BEFORE sorting:
        wl   = np.array([self._band_to_wavelength_nm(b) for b in tbl['filter']], dtype=float)
        vals = np.array(tbl[flux_key], dtype=float)
        merrs = np.array(tbl[fluxerr_key], dtype=float)
        zperrs = np.array(tbl['zp_err'], dtype=float)
        errs = np.array([self._combine_err(m, z) for m, z in zip(merrs, zperrs)], dtype=float)
        filt = np.array(tbl['filter'], dtype=object)  # <-- keep filters as an array

        # Sort everything together:
        order = np.argsort(wl)
        wl, vals, errs, groups, filt = wl[order], vals[order], errs[order], np.array(groups)[order], filt[order]


        unique_groups = list(dict.fromkeys(groups))  # preserve order
        self.plt_params.set_palette(n=len(unique_groups))
        with self.plt_params.apply():
            # Group colors: use observation time as a continuous colorbar when grouping by date
            
            if color_key.lower() == 'obsdate':
                group_mjds = [Time(g).mjd for g in unique_groups]
                norm = Normalize(vmin=np.nanmin(group_mjds), vmax=np.nanmax(group_mjds))
                cmap_name = getattr(self.plt_params, 'cmap', None)
                if cmap_name is None:
                    cmap_name = 'jet'
                cmap = cm.get_cmap(cmap_name)
                colors = [cmap(norm(x)) for x in group_mjds]
            else:
                # categorical palette
                base_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
                colors = [next(base_cycle) for _ in unique_groups]
                
            # dynamic height based on #groups and your offset
            # dynamic height
            fig_height = 3 + len(unique_groups) * 0.35 * self.OFFSET

            # use your chosen width and dpi from plt_params
            width, _ = self.plt_params.figure_figsize
            self.plt_params._rcparams['figure.figsize'] = (width, fig_height)
            self.plt_params._rcparams['figure.dpi'] = self.plt_params.figure_dpi

            fig, ax = plt.subplots()  # no figsize here

            offset = 0.0
            # for cg, col in zip(unique_groups, colors):
            #     m = (groups == cg)
            #     x = wl[m]
            #     y = vals[m].copy()
            #     e = errs[m]

            #     y = y + offset
            #     valid = np.isfinite(x) & np.isfinite(y)
            #     if np.any(valid):
            #         ax.plot(x[valid], y[valid], c=col, alpha=0.5, linestyle=self.plt_params.line_style)
            #         ax.errorbar(x[valid], y[valid], yerr=e[valid],
            #                     label=f"{cg} (+{offset:.1f})" if offset != 0 else f"{cg}",
            #                     **self.plt_params.get_errorbar_kwargs(color=col, shape='o'))
            #     offset += self.OFFSET
            
            has_medium = False  # circle 'o'
            has_broad = False   # square 's'
            for cg, col in zip(unique_groups, colors):
                m = (groups == cg)
                x = wl[m]
                y = vals[m].copy()
                e = errs[m]
                f = filt[m]

                y = y + offset
                valid = np.isfinite(x) & np.isfinite(y)
                if np.any(valid):
                    ax.plot(x[valid], y[valid], c=col, alpha=0.5,
                            linestyle=self.plt_params.line_style)

                    first_point = True
                    for xi, yi, ei, ff in zip(x[valid], y[valid], e[valid], f[valid]):
                        shape = self.FILTER_SHAPE.get(ff, 'o')
                        color = self.FILTER_COLOR.get(ff, col)

                        # Track shape usage
                        if shape == 'o':
                            has_medium = True
                        elif shape == 's':
                            has_broad = True

                        if first_point:
                            ax.errorbar(
                                xi, yi, yerr=ei,
                                label=f"{cg} (+{offset:.1f})" if offset != 0 else f"{cg}",
                                **self.plt_params.get_errorbar_kwargs(color=color, shape=shape)
                            )
                            first_point = False
                        else:
                            ax.errorbar(
                                xi, yi, yerr=ei,
                                **self.plt_params.get_errorbar_kwargs(color=color, shape=shape)
                            )

                offset += self.OFFSET
            
            # --- Conditionally add black legend entries ---
            if has_medium:
                ax.errorbar([], [], **self.plt_params.get_errorbar_kwargs(color='k', shape='o'), label='Medium bands')
            if has_broad:
                ax.errorbar([], [], **self.plt_params.get_errorbar_kwargs(color='k', shape='s'), label='Broad bands')

            # Axes / style
            ax.set_xlabel("Effective Wavelength [nm]")
            ax.set_ylabel("Magnitude (+ offset)" if "MAG" in flux_key.upper() else "Flux (+ offset)")
            if "MAG" in flux_key.upper():
                ax.invert_yaxis()
            if self.plt_params.xlim:  ax.set_xlim(*self.plt_params.xlim)
            if self.plt_params.ylim:  ax.set_ylim(*self.plt_params.ylim)
            if self.plt_params.xticks is not None: ax.set_xticks(self.plt_params.xticks)
            if self.plt_params.yticks is not None: ax.set_yticks(self.plt_params.yticks)
            ax.grid(True, which='major', alpha=0.3)
            ax.minorticks_on()
            ax.set_title(f"Photometric spectrum for {title}")
            
            # ---------- External overplots ----------
            is_mag = ("MAG" in flux_key.upper())

            # GaiaXP: full low-res spectrum converted to AB mag vs nm
            if overplot_gaiaxp and is_mag:
                try:
                    self.CatalogQuerier.change_catalog('GAIAXP')
                    res = self.CatalogQuerier.query(coord=coord, radius_arcsec=matching_radius_arcsec)
                    if len(res) > 0:
                        gx = res[0]
                        # nearest source
                        closest = gx['Source'][0]
                        gx = gx[gx['Source'] == closest]
                        wl_nm = np.array(gx['lambda'])
                        f_si  = np.array(gx['Flux'])
                        fe_si = np.array(gx['e_Flux'])
                        wl_AA = wl_nm * 10.0
                        mag   = np.array(self.helper.flambSI_to_ABmag(f_si, wl_AA), dtype=float)
                        magerr = np.array(self.helper.fluxerr_to_magerr(flux=f_si, fluxerr=fe_si), dtype=float)
                        ok = np.isfinite(mag) & (magerr >= 0)
                        if np.any(ok):
                            ax.errorbar(wl_nm[ok], mag[ok], yerr=magerr[ok],
                                        fmt='None', color='magenta', alpha=0.3, label='GaiaXP')
                    else:
                        ax.plot([], [], ' ', label='GaiaXP (no data)')
                except Exception:
                    ax.plot([], [], ' ', label='GaiaXP (error)')

            # SDSS points (u,g,r,i,z)
            if overplot_sdss and is_mag:
                try:
                    self.CatalogQuerier.change_catalog('SDSS')
                    res = self.CatalogQuerier.query(coord=coord, radius_arcsec=matching_radius_arcsec)
                    if len(res) > 0 and len(res[0]) > 0:
                        sdss = res[0][0]
                        bands = ['u','g','r','i','z']
                        xs, ys, es = [], [], []
                        for b in bands:
                            m  = sdss.get(f'{b}mag')
                            me = sdss.get(f'e_{b}mag')
                            wl = self._band_to_wavelength_nm(b)
                            if m is None or me is None or not np.isfinite(wl):
                                continue
                            xs.append(wl); ys.append(m); es.append(me)
                        if xs:
                            ax.errorbar(xs, ys, yerr=es,
                                        label='SDSS', **self.plt_params.get_errorbar_kwargs('green','^'))
                    else:
                        ax.plot([], [], ' ', label='SDSS (no data)')
                except Exception:
                    ax.plot([], [], ' ', label='SDSS (error)')

            # PS1 points (g,r,i,z,y)
            if overplot_ps1 and is_mag:
                try:
                    self.CatalogQuerier.change_catalog('PS1')
                    res = self.CatalogQuerier.query(coord=coord, radius_arcsec=matching_radius_arcsec)
                    if len(res) > 0 and len(res[0]) > 0:
                        ps1 = res[0][0]
                        bands = ['g','r','i','z','y']
                        xs, ys, es = [], [], []
                        for b in bands:
                            m  = ps1.get(f'{b}mag')
                            me = ps1.get(f'e_{b}mag')
                            wl = self._band_to_wavelength_nm(f'{b}_ps1')  # use PS1-specific pivot
                            if (m is None) or (me is None) or (not np.isfinite(wl)):
                                continue
                            xs.append(wl); ys.append(m); es.append(me)
                        if xs:
                            ax.errorbar(xs, ys, yerr=es,
                                        label='PS1', **self.plt_params.get_errorbar_kwargs('blue','o'))
                    else:
                        ax.plot([], [], ' ', label='PS1 (no data)')
                except Exception:
                    ax.plot([], [], ' ', label='PS1 (error)')

            if (len(unique_groups) > 1) or any([overplot_gaiaxp, overplot_sdss, overplot_ps1]):
                ax.legend(loc=self.plt_params.label_position, ncol=self.plt_params.ncols,
                        fontsize=self.plt_params.legend_fontsize)

            plt.show()
            return [fig], [ax], tbl

    def search_source(self, 
                      ra: Union[float, list, np.ndarray],
                      dec: Union[float, list, np.ndarray],
                      ra_key: str = 'X_WORLD',
                      dec_key: str = 'Y_WORLD',
                      matching_radius_arcsec: float = 5.0):
        """
        Search for sources in the merged catalog.
        
        Parameters
        ----------
        ra, dec : float
            Sky position in degrees.
        ra_key : str
            Column name for right ascension. Default is 'X_WORLD'.
        dec_key : str
            Column name for declination. Default is 'Y_WORLD'.
        matching_radius_arcsec : float
            Search radius for the source match.

        """
        if self.merged_catalog is None:
            self._merge_catalogs(ra_key = ra_key, 
                                 dec_key = dec_key,
                                 max_distance_arcsec=matching_radius_arcsec,
                                 join_type = 'outer',
                                 )

        ra = np.atleast_1d(ra)
        dec = np.atleast_1d(dec)
        input_coords = SkyCoord(ra=ra, dec=dec, unit='deg')
        
        target_catalog = self.merged_catalog
        catalog_coords = target_catalog['coord']
        
        matched_catalog, matched_input, unmatched_catalog = self.helper.cross_match(catalog_coords, input_coords, matching_radius_arcsec)
        print(f"Matched {len(matched_catalog)} sources out of {len(input_coords)} input positions.")
        return target_catalog[matched_catalog]

    def extract_source_info(self,
                            ra: float,
                            dec: float,
                            ra_key: str = 'X_WORLD',
                            dec_key: str = 'Y_WORLD',
                            flux_keys=['MAGSKY_AUTO', 'MAGSKY_APER', 'MAGSKY_APER_1', 'MAGSKY_APER_2', 'MAGSKY_APER_3', 'MAGSKY_APER_4'], 
                            fluxerr_keys=['MAGERR_AUTO', 'MAGERR_APER', 'MAGERR_APER_1', 'MAGERR_APER_2', 'MAGERR_APER_3', 'MAGERR_APER_4'], 
                            matching_radius_arcsec=5.0, 
                            fit_filter_key=None):
        """
        Extract source information from the merged catalog.
        
        Each row of the returned table will be a per-exposure record with the metadata and photometry.
        
        Parameters
        ----------
        ra, dec : float
            Sky position in degrees.
        ra_key : str
            Column name for right ascension. Default is 'X_WORLD'.
        dec_key : str
            Column name for declination. Default is 'Y_WORLD'.
        flux_keys : str or sequence of str
            Photometry value columns you want to carry over (e.g., 'MAGSKY_APER_1').
            Accepts a single key or a list/tuple of keys.
        fluxerr_keys : str or sequence of str
            Corresponding error columns (e.g., 'MAGERR_APER_1'). Must be same length as `flux_keys`.
        matching_radius_arcsec : float
            Search radius for the source match.
        fit_filter_key : str or None
            Placeholder for future use (ignored).

        Returns
        -------
        astropy.table.Table or None
            One row per exposure/catalog with metadata + requested photometry.
            Returns None if no source is found.
        """
        # Normalize keys to lists
        if isinstance(flux_keys, str):
            flux_keys = [flux_keys]
        if isinstance(fluxerr_keys, str):
            fluxerr_keys = [fluxerr_keys]
        if len(flux_keys) != len(fluxerr_keys):
            raise ValueError("flux_keys and fluxerr_keys must have the same length.")

        # Ensure merged table exists (pulling at least the requested keys + their ZPERR counterparts)
        needed_data_keys = set()
        for fk, fek in zip(flux_keys, fluxerr_keys):
            needed_data_keys.add(fk)
            needed_data_keys.add(fek)
            needed_data_keys.add(fek.replace('MAGERR', 'ZPERR'))

        if getattr(self, "catalogs_tbl", None) is None:
            # Provide sane defaults for merge (user can override upstream if needed)
            self._merge_catalogs(
                ra_key = ra_key,
                dec_key = dec_key,
                max_distance_arcsec=matching_radius_arcsec,
                join_type = 'outer',
                data_keys=sorted(needed_data_keys),
            )
        else:
            # If any needed key is missing, re-merge including them
            need_merge = False
            for key in needed_data_keys:
                if f"{key}_idx0" not in self.merged_catalog.colnames:
                    need_merge = True
                    break
            if need_merge:
                self._merge_catalogs(
                    ra_key = ra_key,
                    dec_key = dec_key,
                    max_distance_arcsec=matching_radius_arcsec,
                    join_type = 'outer',
                    data_keys=sorted(needed_data_keys),
                )

        # Find closest source
        selected = self.search_source(ra, 
                                      dec, 
                                      ra_key = ra_key,
                                      dec_key = dec_key,
                                      matching_radius_arcsec=matching_radius_arcsec)
        if selected is None or len(selected) == 0:
            return None

        # Take the nearest match (row 0)
        row = selected[0]

        # Build per-exposure records from self.metadata indices
        # self.metadata[idx] should include per-catalog fields like filter, obsdate, depth, observatory, telname, exptime, etc.
        if not hasattr(self, "metadata") or self.metadata is None:
            raise RuntimeError("self.metadata is missing. Run merge_catalogs() first.")

        records = {
            idx: {k: v for k, v in meta.items() if k not in ('ra', 'dec')}
            for idx, meta in self.metadata.items()
        }

        # Copy all "*_idx{idx}" values from the matched row into each record
        # (This will include MAG*, MAGERR*, ZPERR*, and any other requested per-exposure columns.)
        for colname in row.colnames:
            if '_idx' not in colname:
                continue
            try:
                base, idx_str = colname.rsplit('_idx', 1)
                idx = int(idx_str)
            except Exception:
                continue
            if idx in records:
                records[idx][base] = row[colname]

        # For convenience, also add a unified 'zp_err' per exposure using the first available ZPERR among requested pairs
        for idx, rec in records.items():
            zperr_val = None
            for ferr in fluxerr_keys:
                zp_key = ferr.replace('MAGERR', 'ZPERR')
                if zp_key in rec:
                    zperr_val = rec[zp_key]
                    break
            rec['zp_err'] = zperr_val

        # Materialize table
        result_tbl = Table(rows=list(records.values()))

        # Attach coord/ra/dec convenience
        if 'coord' in row.colnames:
            result_tbl['coord'] = row['coord']
        else:
            sc = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
            result_tbl['coord'] = [sc] * len(result_tbl)

        if 'ra' not in result_tbl.colnames:
            result_tbl['ra'] = float(ra)
        if 'dec' not in result_tbl.colnames:
            result_tbl['dec'] = float(dec)

        # Time columns
        if 'obsdate' in result_tbl.colnames:
            t = Time(result_tbl['obsdate'])
            result_tbl['mjd'] = t.mjd
            result_tbl['jd'] = t.jd

        # Column ordering: metadata ? requested photometry ? remaining
        meta_order = ['ra', 'dec', 'coord', 'filter', 'exptime', 'obsdate', 'mjd', 'jd',
                    'seeing', 'depth', 'observatory', 'telname', 'zp_err']
        phot_cols = []
        for fk, fek in zip(flux_keys, fluxerr_keys):
            if fk in result_tbl.colnames:
                phot_cols.append(fk)
            if fek in result_tbl.colnames:
                phot_cols.append(fek)
            zpk = fek.replace('MAGERR', 'ZPERR')
            if zpk in result_tbl.colnames:
                phot_cols.append(zpk)

        ordered = [c for c in meta_order if c in result_tbl.colnames] + phot_cols
        remaining = [c for c in result_tbl.colnames if c not in ordered]
        result_tbl = result_tbl[ordered + remaining]

        # Cache for plotting
        self.data = result_tbl
        return result_tbl


    def _band_to_wavelength_nm(self, band: str) -> float:
        """Return effective wavelength (nm) for a band key."""
        # exact key
        if band in self.EFFECTIVE_WAVELENGTHS_NM:
            return self.EFFECTIVE_WAVELENGTHS_NM[band]
        # PS1 commonly comes as 'g','r','i','z','y' from services; map to *_ps1
        ps1_map = {'g':'g_ps1','r':'r_ps1','i':'i_ps1','z':'z_ps1','y':'y_ps1'}
        if band in ps1_map and ps1_map[band] in self.EFFECTIVE_WAVELENGTHS_NM:
            return self.EFFECTIVE_WAVELENGTHS_NM[ps1_map[band]]
        return np.nan

    def _combine_err(self, meas_err, zp_err):
        """Quadrature-combine measurement and zeropoint error when both finite."""
        m = meas_err if np.isfinite(meas_err) else np.nan
        z = zp_err   if np.isfinite(zp_err)   else np.nan
        if np.isfinite(m) and np.isfinite(z):
            return np.sqrt(m*m + z*z)
        return m

    def _merge_catalogs(self,
                        ra_key: str = 'X_WORLD',
                        dec_key: str = 'Y_WORLD',
                        max_distance_arcsec: float = 2,
                        join_type: str = 'outer',
                        data_keys: list = ['MAGSKY_AUTO', 'MAGERR_AUTO', 'MAGSKY_APER', 'MASERR_APER', 'MAGSKY_APER_1', 'MAGERR_APER_1', 'MAGSKY_APER_2', 'MAGERR_APER_2', 'MAGSKY_APER_3', 'MAGERR_APER_3', 'MAGSKY_CIRC', 'MAGERR_CIRC']):
        self.merged_catalog, self.metadata = self.source_catalogs.merge_catalogs(
            max_distance_arcsec=max_distance_arcsec,
            ra_key=ra_key,
            dec_key=dec_key,
            join_type=join_type,
            data_keys=data_keys)
    
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
                
                # Color parameters
                self.cmap = 'jet'
                
                # Label parameters
                self.label_position = 'best'  # 'best', 'upper right', 'lower left', etc.
                self.ncols = 2
                
                # Error bar parameters
                self.errorbar_enabled = True  # Optional switch
                self.errorbar_markersize = 7
                self.errorbar_hollow_marker = True  # True = hollow, False = filled
                self.errorbar_capsize = 3.5
                self.errorbar_elinewidth = 1.2
                
                # Plot parameters
                self.line_style = 'solid'
                

            def __getattr__(self, name):
                rc_name = name.replace('_', '.')
                if rc_name in self._rcparams:
                    return self._rcparams[rc_name]
                raise AttributeError(f"'PlotParams' object has no attribute '{name}'")

            def __setattr__(self, name, value):
                if name.startswith('_') or name in ('xlim', 'ylim', 'xticks', 'yticks', 'errorbar_capsize', 'errorbar_elinewidth', 'errorbar_markersize', 'errorbar_enabled', 'errorbar_hollow_marker', 'label_position', 'ncols', 'line_style', 'cmap'):
                    super().__setattr__(name, value)
                else:
                    rc_name = name.replace('_', '.')
                    if rc_name in self._rcparams:
                        self._rcparams[rc_name] = value
                    else:
                        raise AttributeError(f"'PlotParams' has no rcParam '{rc_name}'")
                    
            def set_palette(self, n: int = 40):
                base = cm.get_cmap(self.cmap, n)
                colors = [base(i) for i in range(base.N)]
                from matplotlib import cycler as _cycler
                self._rcparams['axes.prop_cycle'] = _cycler(color=colors)
            
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
                txt += 'Visualization Parameters -----------------\n'
                txt += f"line_style = {self.line_style}\n"
                txt += f"cmap = {self.cmap}\n"
                txt += 'Error Bar Configuration ---------\n'
                txt += f"errorbar_enabled = {self.errorbar_enabled}\n"
                txt += f"errorbar_markersize = {self.errorbar_markersize}\n"
                
                txt += f"errorbar_capsize = {self.errorbar_capsize}\n"
                txt += f"errorbar_elinewidth = {self.errorbar_elinewidth}\n"
                
                return txt
        return PlotParams()

        
        
    
#%%
if __name__ == "__main__":
    from ezphot.dataobjects import CatalogSet
    source_catalogs = CatalogSet()
    source_catalogs.search_catalogs(
        target_name = 'T01358',
        search_key = 'calib*com.fits.cat'
     )    
    ra  =239.03762083
    dec = -68.41128803
    source_catalogs.select_sources(ra, dec, radius = 10)
#%%
if __name__ == "__main__":
    source_catalogs.select_catalogs(obs_start = '2025-02-12', obs_end = '2025-02-13')
    self = PhotometricSpectrum(source_catalogs)
    self.OFFSET = 0
    self.extract_source_info(ra, dec)
#%%
# %%
if __name__ == "__main__":
    self.plt_params.figure_figsize = (10, 6)
    self.plt_params.figure_dpi = 500
    self.plt_params.cmap = 'jet'
    flux_key = 'MAGSKY_APER_2'
    fluxerr_key = 'MAGERR_APER_2'
    matching_radius_arcsec = 10
    color_key: str = 'telname'
    overplot_gaiaxp = True
    overplot_sdss = True
    overplot_ps1 = True
    self.plt_params.line_style = 'none'
    figs, axs, matched_sources = self.plot(ra, 
                          dec, 
                          flux_key=flux_key, 
                          fluxerr_key =fluxerr_key,
                          color_key = color_key, 
                          matching_radius_arcsec=matching_radius_arcsec,
                          overplot_gaiaxp=overplot_gaiaxp,
                          overplot_sdss = overplot_sdss,
                          overplot_ps1 = overplot_ps1,
                          objname = 'SN 2025fvw')
# %%
