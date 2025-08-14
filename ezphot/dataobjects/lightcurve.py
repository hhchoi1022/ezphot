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
from matplotlib import colors as mcolors

from ezphot.utils import SDTDataQuerier
from ezphot.dataobjects import Catalog, CatalogSet
from ezphot.helper import Helper
from ezphot.imageobjects import ScienceImage
from ezphot.utils import CatalogQuerier
#%%
class LightCurve:
    """
    Light curve object.
    
    This object is used to plot the light curve of a source.
    It is initialized with a source_catalogs object, which is a CatalogSet object.
    The source_catalogs object is used to extract the source information from the catalogs.    
    
    To change the plot parameters, modify the plt_params attribute.
    """
    
    def __init__(self, source_catalogs: CatalogSet = None):
        """
        Initialize the LightCurve object.
        
        Parameters
        ----------
        source_catalogs : CatalogSet
            The source catalogs to use for the light curve.
        """
        if not isinstance(source_catalogs, CatalogSet):
            raise TypeError("source_catalogs must be an instance of CatalogSet")
        self.helper = Helper()
        self.source_catalogs = source_catalogs
        self.merged_catalog = None
        self.plt_params = self._plt_params()
        self.CatalogQuerier = CatalogQuerier(catalog_key = None)
        self.data = None
        
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


    def plot(self,
             ra: float,
             dec: float,
             matching_radius_arcsec: float = 5.0,
             flux_key: str = 'MAGSKY_AUTO',
             fluxerr_key: str = 'MAGERR_AUTO',
             color_key: str = 'filter',     # or 'telname' or 'observatory'
             apply_filter_offsets: bool = True,
        ):
        """
        Plot light curve for the closest source to (ra, dec).
        
        The figure parameters are set in the plt_params attribute.

        Parameters
        ----------
        ra, dec : float
            Sky position in degrees.
        matching_radius_arcsec : float
            Search radius for the source match.
        flux_key : str
            Column name for flux.
        fluxerr_key : str
            Column name for flux error.
        color_key : str
            Column name for color.
        data : astropy.table.Table
            Table to plot.
        apply_filter_offsets : bool
            If True, apply filter offsets.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object.
        ax : matplotlib.axes.Axes
            Axes object.
        tbl : astropy.table.Table
            Table of data.
        """
        # Prepare data
        if self.data is None:
            self.extract_source_info(
                ra, dec,
                flux_keys=flux_key,
                fluxerr_keys=fluxerr_key,
                matching_radius_arcsec=matching_radius_arcsec,
            )
        if self.data is None or len(self.data) == 0:
            print(f"[WARNING] No sources found within {matching_radius_arcsec}\" of RA={ra}, Dec={dec}")
            return None, None, None

        tbl = self.data.copy()

        # Column names
        zp_key = fluxerr_key.replace('MAGERR', 'ZPERR')
        is_mag = "MAG" in flux_key.upper()

        # Apply medium-band offsets in magnitude space
        mags, errs, depths, labels = [], [], [], []
        for row in tbl:
            f = row['filter']
            off = self.FILTER_OFFSET.get(f, 0.0) if (apply_filter_offsets and is_mag) else 0.0

            # value
            v = row.get(flux_key)
            mags.append(v + off if np.isfinite(v) else np.nan)

            # error = combine measurement & ZP error if available
            me = row.get(fluxerr_key)
            ze = row.get(zp_key) if zp_key in tbl.colnames else row.get('zp_err')
            if np.isfinite(me) and np.isfinite(ze):
                errs.append(np.sqrt(me**2 + ze**2))
            else:
                errs.append(me if np.isfinite(me) else np.nan)

            # depth (for non-detections)
            d = row.get('depth')
            depths.append(d + off if (d is not None and np.isfinite(d)) else np.nan)

            labels.append(f"{f}+{off:.1f}" if (off and is_mag) else f"{f}")

        tbl['y']       = np.array(mags, dtype=float)
        tbl['yerr']    = np.array(errs, dtype=float)
        tbl['depth_y'] = np.array(depths, dtype=float)
        tbl['label']   = np.array(labels)

        # Choose grouping for color/legend
        if color_key.lower() not in ('filter', 'telname', 'observatory'):
            color_key = 'filter'
        groups = np.array(tbl[color_key])

        # Sort by time
        if 'mjd' not in tbl.colnames and 'obsdate' in tbl.colnames:
            t = Time(tbl['obsdate'])
            tbl['mjd'] = t.mjd
        order = np.argsort(tbl['mjd'])
        tbl = tbl[order]
        groups = groups[order]

        with self.plt_params.apply():
            from itertools import cycle
            color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
            fig, ax = plt.subplots()

            # Title as J-name
            coord = tbl['coord'][0] if 'coord' in tbl.colnames else SkyCoord(ra*u.deg, dec*u.deg)
            ra_str  = coord.ra.to_string(unit=u.hourangle, sep='', pad=True, precision=2)
            dec_str = coord.dec.to_string(sep='', alwayssign=True, pad=True, precision=1)
            jname = f'J{ra_str}{dec_str}'
            ax.set_title(jname)

            # Plot detections
            unique_groups = list(dict.fromkeys(groups))  # preserves order
            for g in unique_groups:
                m = (groups == g)
                x = np.array(tbl['mjd'][m], dtype=float)
                y = np.array(tbl['y'][m], dtype=float)
                yerr = np.array(tbl['yerr'][m], dtype=float)
                valid = np.isfinite(x) & np.isfinite(y)
                if not np.any(valid):
                    continue

                # color
                if color_key.lower() == 'filter':
                    base_filter = tbl['filter'][m][0]
                    c = self.FILTER_COLOR.get(base_filter, next(color_cycle))
                else:
                    base_filter = tbl['filter'][m][0]
                    c = self.FILTER_COLOR.get(base_filter, next(color_cycle))

                ax.errorbar(
                    x[valid], y[valid], yerr=yerr[valid],
                    label=str(g),
                    **self.plt_params.get_errorbar_kwargs(c, 's')
                )

            # Plot non-detections (depth only) as inverted triangles
            nd = ~np.isfinite(tbl['y']) & np.isfinite(tbl['depth_y'])
            if np.any(nd):
                for g in unique_groups:
                    m = nd & (groups == g)
                    if not np.any(m):
                        continue
                    x_nd = np.array(tbl['mjd'][m], dtype=float)
                    d_nd = np.array(tbl['depth_y'][m], dtype=float)
                    base_filter = tbl['filter'][m][0]
                    c = self.FILTER_COLOR.get(base_filter, next(color_cycle))
                    ax.scatter(x_nd, d_nd, marker='v')

            # Axes cosmetics
            ax.set_xlabel("Obsdate [MJD]")
            ax.set_ylabel("Magnitude" if is_mag else "Flux")
            if is_mag:
                ax.invert_yaxis()
            ax.grid(True, alpha=0.3)
            if self.plt_params.xlim: ax.set_xlim(*self.plt_params.xlim)
            if self.plt_params.ylim: ax.set_ylim(*self.plt_params.ylim)
            if self.plt_params.xticks is not None: ax.set_xticks(self.plt_params.xticks)
            if self.plt_params.yticks is not None: ax.set_yticks(self.plt_params.yticks)
            ax.minorticks_on()

            # Legend: broadband first, then medium
            handles, labels = ax.get_legend_handles_labels()
            broadbands = ['u','g','r','i','z','y','B','V','R','I']
            def base(lbl): return str(lbl).split('+')[0].strip()
            pairs = list(zip(handles, labels))
            bb  = [p for p in pairs if base(p[1]) in broadbands]
            mb  = [p for p in pairs if base(p[1]) not in broadbands]
            bb.sort(key=lambda p: broadbands.index(base(p[1])) if base(p[1]) in broadbands else 1e9)
            mb.sort(key=lambda p: base(p[1]))
            pairs = bb + mb
            if pairs:
                ax.legend([p[0] for p in pairs], [p[1] for p in pairs],
                        loc=self.plt_params.label_position, ncol=self.plt_params.ncols)

            # MJD ? ISO date labels
            xticks = ax.get_xticks()
            try:
                xt = Time(xticks, format='mjd')
                ax.set_xticks(xticks)
                ax.set_xticklabels(xt.to_value('iso', subfmt='date'), rotation=45)
            except Exception:
                pass

            plt.show()
            return fig, ax, tbl
        
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
                
                # label parameters
                self.label_position = 'best'
                self.ncols = 2
                
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
                if name.startswith('_') or name in ('xlim', 'ylim', 'xticks', 'yticks', 'errorbar_capsize', 'errorbar_elinewidth', 'errorbar_markersize', 'errorbar_enabled', 'errorbar_hollow_marker', 'label_position', 'ncols'):
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
    
        
    
#%%
if __name__ == "__main__":
    source_catalogs = CatalogSet()
    source_catalogs.search_catalogs('T22956', 'calib*100.com.fits.cat')
    source_catalogs.select_sources(ra = 233.857430764, dec= 12.0577222937, radius = 15)
        
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
    # ra = 233.322342 # S250206dm for T01462
    # dec = -68.007909
    # ra = 259.757396
    # dec = -67.360176
    #ra = 241.62392408
    #dec = -70.327141108
    # ra = 262.154312
    # dec = -68.789571
    # ra = 263.916460
    # dec = -70.346012
    # ra = 234.685513
    # dec = -68.794466
    # ra = 232.069681
    # dec = -67.896978
    # source_catalogs.select_catalogs(filter = ['g', 'r', 'i'], obs_start = '2025-01-01', obs_end = '2025-03-01')
    # source_catalogs.select_sources(ra, dec, radius =  60)
#%%
if __name__ == "__main__":
    self = LightCurve(source_catalogs)
    self.extract_source_info(ra, dec)
# %%
if __name__ == "__main__":

    flux_key = 'MAGSKY_AUTO'
    fluxerr_key = 'MAGERR_AUTO'
    matching_radius_arcsec = 1
    color_key: str = 'filter'#'OBSDATE'
    overplot_gaiaxp = False
    overplot_sdss = False
    overplot_ps1 = False
    self.plt_params.figure_figsize = (10,6)
    self.plt_params.ylim = [26, 8]
#%%
if __name__ == "__main__":
    figs, axs, matched_sources = self.plot(ra, 
                          dec, 
                          flux_key=flux_key, 
                          color_key = color_key, 
                          matching_radius_arcsec=matching_radius_arcsec,)
    # axs[0].scatter(Time('2025-02-12T08:04:30').mjd, 19.21, c = 'red', marker='*', s=100, label='KMTNet R band')
    # axs[0].legend(loc='upper right')
    


# %%
