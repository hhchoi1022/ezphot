#%%
from typing import Union
from tqdm import tqdm
import numpy as np
from numpy.ma import getmaskarray
from numpy.ma import is_masked
from itertools import cycle
import matplotlib.pyplot as plt
from matplotlib import cycler
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from astropy.table import Table
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.stats import sigma_clipped_stats
from astropy.table import MaskedColumn

from ezphot.dataobjects import CatalogSet
from ezphot.helper import Helper
from ezphot.utils import CatalogQuerier
from ezphot.error import *

#%%
class LightCurve:
    """
    Light curve object.
    
    This object is used to plot the light curve of a source.
    It is initialized with a source_catalogs object, which is a CatalogSet object.
    The source_catalogs object is used to extract the source information from the catalogs.    
    
    To change the plot parameters, modify the plt_params attribute.
    """
    
    def __init__(self, catalogset: CatalogSet = None):
        """
        Initialize the LightCurve object.
        
        Parameters
        ----------
        source_catalogs : CatalogSet
            The source catalogs to use for the light curve.
        """
        # if not isinstance(source_catalogs, CatalogSet):
        #     raise TypeError("source_catalogs must be an instance of CatalogSet")
        self.helper = Helper()
        self.catalogset = catalogset
        self.source_catalogs = None
        if self.catalogset is not None:
            self.source_catalogs = {i: catalog for i, catalog in enumerate(self.catalogset.catalogs)}
        self.plt_params = self._plt_params()
        self.CatalogQuerier = CatalogQuerier(catalog_key = None)
        self.data = None
        
    FILTER_OFFSET = {
        # 7DT Medium band filters
        'm375w': -8.0,  'm386': -7.5,  'm400': -7.0,  'm425w': -6.5,
        'm425':  -6.0,  'm438': -5.5,  'm450': -5.0,  'm466w': -4.5,

        'm475':  -4.0,  'm483': -3.5,  'm500': -3.0,  'm512': -2.5,
        'm525':  -2.0,  'm534': -1.5,  'm550': -1.0,  'm561': -0.5,

        'm575':   0.0,  'm586':  0.5,  'm600':  1.0,  'm615':  1.5,
        'm625':   2.0,  'm640':  2.5,  'm650':  3.0,  'm661':  3.5,

        'm675':   4.0,  'm692w': 4.5,  'm700':  5.0,  'm710w': 5.5,
        'm725':   6.0,  'm750':  6.5,  'm769w': 7.0,  'm775':  7.5,

        'm800':   8.0,  'm825':  8.5,  'm832w': 9.0,  'm850':  9.5,
        'm875':  10.0,
        # SDSS ugriz and Johnson-Cousins UBVRI
        'u': -9.0, 'g': -3.5, 'r': 1.5, 'i': 6.5, 'z': 11.5,
        'U': -9.5, 'B': -6.5, 'V': -0, 'R': 0.5, 'I': 1.5,
        'UVW2': -10.0, 'UVM2': -9.0, 'UVW1': -8.0,
        'G': 1.5
        }
    
    # Global: Filter effective wavelengths (nm)
    FILTER_PIVOT_WAVELENGTH_NM = {
        'm375w' : 382.4, 'm425w' : 427.2, 'm466w' : 467.7, 'm692w' : 690.6,
        'm710w' : 708.4, 'm769w' : 767.6, 'm832w' : 830.0, 
        'm386' : 387.5, 'm400' : 401.3, 'm425' : 425.5, 'm438' : 438.3,
        'm450' : 450.8, 'm475' : 475.3, 'm483' : 482.5, 'm500' : 500.3,
        'm512' : 511.9, 'm525' : 524.8, 'm534' : 533.8, 'm550' : 550.1,
        'm561' : 560.9, 'm575' : 574.9, 'm586' : 585.9, 'm600' : 600.1,
        'm615' : 614.9, 'm625' : 624.8, 'm640' : 639.5, 'm650' : 650.1,
        'm661' : 660.7, 'm675' : 674.5, 'm700' : 699.9, 'm725' : 724.6,
        'm750' : 748.9, 'm775' : 775.2, 'm800' : 799.2, 'm825' : 824.0, 
        'm850' : 848.3, 'm875' : 872.9,
        
        # GAIA filters (https://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?id=GAIA/GAIA0.Grp&&mode=browse&gname=GAIA&gname2=GAIA0#filter)
        # 'Gaia_G': 657.3, 'Gaia_bp': 525.5, 'Gaia_rp': 793.8,
        
        # Swift filter (https://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?mode=browse&gname=Swift&asttype=)
        # 'Swift_UVW2': 208.6, 'Swift_UVM2': 236.0, 'Swift_UVW1': 263.8, 'Swift_U': 347.1, 'Swift_B': 435.0, 'Swift_V': 541.9,
        
        # SDSS ugriz (DOI+2010)
        # 'u' : 371.4, 'g' : 479.4, 'r' : 616.2, 'i' : 758.7, 'z' : 876.6,
        # PS1 ugizy (TONRY+2012)
        # 'g_ps1': 481, 'r_ps1': 617, 'i_ps1': 752, 'z_ps1': 866, 'y_ps1': 962,
        # Johnson-Cousins UBVRI (Ground based, https://mfouesneau.github.io/pyphot/libcontent.html)
        # 'U' : 364.6, 'B' : 433.0, 'V' : 549.3, 'R' : 652.7, 'I' : 789.1,
        # 2MASS JHK (COHEN+2003)
        # 'J': 1235, 'H': 1662, 'K': 2159,
        # WISE W1-W4 (https://svo2.cab.inta-csic.es/svo/theory/fps/index.php?mode=browse&gname=WISE&asttype=)
        # 'W1': 3352.6, 'W2': 4620.8, 'W3': 11560.8, 'W4': 22088.3,
        }

    # Compute normalized color map
    _wls = np.array(list(FILTER_PIVOT_WAVELENGTH_NM.values()))
    _normed_wls = (_wls - _wls.min()) / (_wls.max() - _wls.min())
    _cmap = plt.cm.plasma
    _rgba_colors = _cmap(_normed_wls)
    _hex_colors = [mcolors.to_hex(c) for c in _rgba_colors]

    # ? Global dictionary
    FILTER_COLOR = dict(zip(FILTER_PIVOT_WAVELENGTH_NM.keys(), _hex_colors))
    
    # Step 2: Override for broadbands (fixed colors)
    FILTER_COLOR.update({
        'u': 'cyan', 'g': 'green', 'r': 'red', 'i': 'yellow', 'z': 'brown', 'y': 'darkorange',    
        'U': 'blue', 'B': 'royalblue', 'V': 'limegreen', 'R': 'firebrick', 'I': 'maroon',
        'Gaia_G': 'g', 'Gaia_bp': 'b', 'Gaia_rp': 'r',
        'Swift_UVW2': 'c', 'Swift_UVM2': 'm', 'Swift_UVW1': 'y', 'Swift_U': 'k', 'Swift_B': 'r', 'Swift_V': 'g'
    })   
    
    MARKER_CYCLE = ['P', '*', 'X', 'v']
    OBSERVATORY_MARKER = {
        'KCT': 'o',
        'RASA36': '^',
        'LSGT': 's',
        '7DT': 'D'
    }
    
    def __repr__(self):
        txt = f'LIGHTCURVE OBJECT (n_catalogs = {len(self.source_catalogs.values())})\n'
        txt += str(self.plt_params)
        return txt

    def plot(self,
             ra: float = None,
             dec: float = None,
             filter: str = None,
             matching_radius_arcsec: float = 5.0,
             ra_key: str = 'X_WORLD',
             dec_key: str = 'Y_WORLD',
             flux_key: str = 'MAGSKY_AUTO',
             fluxerr_key: str = 'MAGERR_AUTO',
             zperr_key: str = 'ZPERR_AUTO',
             depth_key: str = 'UL5SKY_AUTO',
             plot_all_in_one_figure: bool = True,
             apply_offset: bool = True,
             overplot_stamp: bool = False,
             
             verbose: bool = True,
             title: str = None
        ):
        """
        Plot light curve for the closest source to (ra, dec).
        
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
        shape_key : str
            Column name for marker shape. Options: 'filter', 'telname', 'observatory', or None.
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
        def _get_col(tbl, key):
            if (key in tbl.colnames) and not all(getmaskarray(tbl[key])):
                return np.array(tbl[key], dtype=float)
            else:
                return np.full(len(tbl), np.nan)
        
        # 1. Prepare data
        # 1.1. Data formatting
        if self.data is None:
            self.extract_source_info(
                ra, dec,
                ra_key = ra_key,
                dec_key = dec_key,
                flux_key=flux_key,
                fluxerr_key=fluxerr_key,
                zperr_key=zperr_key,
                depth_key=depth_key,
                matching_radius_arcsec=matching_radius_arcsec,
            )
        if self.data is None or len(self.data) == 0:
            self.helper.print(f"[WARNING] No sources found within {matching_radius_arcsec}\" of RA={ra}, Dec={dec}", verbose)
            return None, None, None
        tbl = self.data.copy()
        
        # 1.2. If filter is provided, filter the table to the closest group in time
        if filter is not None:
            tbl = tbl[tbl['filter'] == filter]
            
        if fluxerr_key is not None and all(getmaskarray(tbl[fluxerr_key])):
            print(f"[WARNING] {fluxerr_key} not found/or all masked in table. Ignoring flux error component.")
        if zperr_key is not None and all(getmaskarray(tbl[zperr_key])):
            print(f"[WARNING] {zperr_key} not found/or all masked in table. Ignoring zp error component.")
        if depth_key is not None and all(getmaskarray(tbl[depth_key])):
            print(f"[WARNING] {depth_key} not found/or all masked in table. Ignoring depth plot.")
        
        # 1.3. Genearte detection mask
        mag = tbl[flux_key]
        mask_mag = getmaskarray(mag)
        valid_detection = (~mask_mag)
        if fluxerr_key is not None and not all(getmaskarray(tbl[fluxerr_key])):
            err = tbl[fluxerr_key]
            mask_err = getmaskarray(err)
            valid_detection &= (~mask_err)
            
        # 1.4. Build arrays
        is_mag = "MAG" in flux_key.upper()
        mags = _get_col(tbl, flux_key)
        magerrs = _get_col(tbl, fluxerr_key)
        zperrs  = _get_col(tbl, zperr_key)
        depths  = _get_col(tbl, depth_key)
        if (magerrs is not None) and (zperrs is not None):
            errs = np.array([
                self._combine_err(m, z)
                for m, z in zip(magerrs, zperrs)
            ], dtype=float)
        else:
            errs = magerrs  # fallback
        
        # mags  = np.array(tbl[flux_key], dtype=float)
        # magerrs = np.array(tbl[fluxerr_key], dtype=float)
        # zperrs = np.array(tbl['zp_err'], dtype=float)
        # depths = np.array(tbl['depth'], dtype=float)
        # errs  = np.array([self._combine_err(m, z) for m, z in zip(magerrs, zperrs)], dtype=float)
        
        mjds = np.array(tbl['mjd'], dtype=float)
        telname = np.array(tbl['telname'], dtype=object)
        obs = np.array(tbl['observatory'], dtype=object)
        filt = np.array(tbl['filter'], dtype=object) # Filter names
        groups = np.array(tbl['filter_group'], dtype=object)
        order = np.argsort(mjds)
        mags = mags[order]
        magerrs = magerrs[order]
        zperrs = zperrs[order]
        depths = depths[order]
        errs = errs[order]
        mjds = mjds[order]
        telname = telname[order]
        obs = obs[order]
        filt = filt[order]
        groups = groups[order]
        valid_detection = valid_detection[order]
        
        filter_order = {f: i for i, f in enumerate(self.FILTER_OFFSET)}
        unique_groups = sorted(
            set(groups),
            key=lambda g: filter_order.get(str(g.split('|')[0]), np.inf)
        )

        if (plot_all_in_one_figure) and (len(unique_groups) > 1):
            plot_multiple_lightcurve = True
        else:
            plot_multiple_lightcurve = False
            
        if overplot_stamp:
            overplot_stamp_together = True
            if plot_multiple_lightcurve:
                self.helper.print(
                    "[WARNING] Detection panels are disabled when plotting multiple lightcurve "
                    "Detection panels are plotted separately for each filter.",
                    verbose
                )
                overplot_stamp_together = False
        else:
            overplot_stamp_together = False

        with self.plt_params.apply():
            color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
            
            # ==================================================
            # Case 1: plot WITH offset
            # ==================================================
            
            offsets = [0.0 for _ in unique_groups]
            
            if plot_multiple_lightcurve:
                colors = [self.FILTER_COLOR.get(str(f.split('|')[0]), next(color_cycle)) for f in unique_groups]
                markers = [self.OBSERVATORY_MARKER.get(str(f.split('|')[1]), None) for f in unique_groups]
                if apply_offset:
                    offsets = [self.FILTER_OFFSET.get(f.split('|')[0], 0.0) for f in unique_groups]
                width, height = self.plt_params.figure_figsize
                fig, ax = plt.subplots(figsize=(width, height),
                                       dpi=self.plt_params.figure_dpi)
                group_iter = zip(unique_groups, colors, markers, offsets)
            else:
                colors = [self.plt_params.scatter_color for _ in unique_groups]
                markers = [self.OBSERVATORY_MARKER.get(str(f.split('|')[1]), None) for f in unique_groups]
                group_iter = zip(unique_groups, colors, markers, offsets)
            
            # Plot detections
            legend_filter_all = []
            legend_observatory_all = []
            figures = dict()
            axes = dict()
            figures_detection = dict()
            for group, color, marker, offset in group_iter:
                m = (groups == group)
                if np.sum(m) == 0:
                    continue
                filter = group.split('|')[0]
                observatory = group.split('|')[1]
                detection_mask = m & valid_detection
                non_detection_mask = m & ~valid_detection
                non_detection_exist = np.sum(non_detection_mask) > 0
                
                x_detection = np.array(mjds[detection_mask], dtype=float)
                x_detection_datetime = Time(x_detection, format='mjd').datetime
                y_detection = np.array(mags[detection_mask], dtype=float)                
                yerr_detection = np.array(errs[detection_mask], dtype=float)
                depth_detection = np.array(depths[detection_mask], dtype=float)
                y_detection += offset
                depth_detection += offset
                y_mean, y_median, y_std = sigma_clipped_stats(y_detection, sigma=3)

                if non_detection_exist:
                    x_non_detection = np.array(mjds[non_detection_mask], dtype=float)
                    x_non_detection_datetime = Time(x_non_detection, format='mjd').datetime
                    y_non_detection = np.array(mags[non_detection_mask], dtype=float)
                    yerr_non_detection = np.array(errs[non_detection_mask], dtype=float)
                    depth_non_detection = np.array(depths[non_detection_mask], dtype=float)
                    y_non_detection += offset
                    depth_non_detection += offset                
                    
                if not plot_multiple_lightcurve:
                    width, height = self.plt_params.figure_figsize

                    if overplot_stamp_together:
                        n_stamp = np.sum(m)
                        lc_height = height
                        det_height = max(2.5, 0.4 * n_stamp)
                        fig = plt.figure(
                            figsize=(width, lc_height + det_height),
                            dpi=self.plt_params.figure_dpi
                        )
                        gs = GridSpec(
                            nrows=2,
                            ncols=1,
                            height_ratios=[lc_height, det_height],  # spectrum : detection
                            hspace= 0.25
                        )

                        ax = fig.add_subplot(gs[0])
                        ax_detection = fig.add_subplot(gs[1])
                    else:
                        fig, ax = plt.subplots(
                            figsize=(width, height),
                            dpi=self.plt_params.figure_dpi
                        )
                                
                # line: connect only time-ordered points
                if self.plt_params.line_style != 'none' and len(x_detection) > 1:
                    ax.plot(
                        x_detection_datetime, y_detection,
                        linestyle=self.plt_params.line_style,
                        color=color,
                        alpha=0.6,
                        zorder=1,
                    )

                # Detection scatters and errorbars
                ax.scatter(x_detection_datetime, y_detection, zorder = 5, 
                           **self.plt_params.get_scatter_kwargs(color, marker))
                ax.errorbar(x_detection_datetime, y_detection, yerr=yerr_detection, zorder = 4, 
                            **self.plt_params.get_errorbar_kwargs(color))
                
                # Non-detection scatters and errorbars
                if non_detection_exist:
                    if self.plt_params.non_detection_enabled:
                        ax.scatter(
                            x_non_detection_datetime, depth_non_detection,
                            s=self.plt_params.non_detection_markersize,
                            marker=self.plt_params.non_detection_marker,
                            facecolors=color,
                            edgecolors='k',
                            alpha=self.plt_params.non_detection_alpha,
                            zorder=6,
                        )
                        
                        for x, y in zip(x_non_detection_datetime,
                                            depth_non_detection):
                            ax.annotate(
                                '',
                                xy=(x, y + 0.1),     # arrow tip (downward in mag)
                                xytext=(x, y),
                                arrowprops=dict(
                                    arrowstyle=self.plt_params.non_detection_arrow_style,
                                    lw=self.plt_params.non_detection_arrow_width,
                                    color=self.plt_params.non_detection_arrow_color,
                                    alpha=self.plt_params.non_detection_alpha,
                                ),
                                zorder=5
                            )

                if overplot_stamp:
                    if overplot_stamp_together:
                        ax_detection.axis('off')
                        self.show_detection(
                            ra, dec,
                            filter=filter,
                            matching_radius_arcsec=matching_radius_arcsec,
                            ra_key = ra_key,
                            dec_key = dec_key,
                            flux_key = flux_key,
                            fluxerr_key = fluxerr_key,
                            downsample=self.plt_params.detection_downsample,
                            zoom_radius_pixel=self.plt_params.detection_zoom_radius_pixel,
                            cmap=self.plt_params.detection_cmap,
                            scale=self.plt_params.detection_scale,
                            aperture_radius_arcsec=self.plt_params.detection_radius_arcsec,
                            aperture_linewidth=self.plt_params.detection_aperture_linewidth,
                            ncols=self.plt_params.detection_ncols,
                            show_title=self.plt_params.detection_show_title,
                            ax_container=ax_detection,   
                        )
                    
                    else:
                        detection_figure = self.show_detection(
                            ra, dec,
                            filter=filter,
                            matching_radius_arcsec=matching_radius_arcsec,
                            ra_key = ra_key,
                            dec_key = dec_key,
                            flux_key = flux_key,
                            fluxerr_key = fluxerr_key,
                            downsample=self.plt_params.detection_downsample,
                            zoom_radius_pixel=self.plt_params.detection_zoom_radius_pixel,
                            cmap=self.plt_params.detection_cmap,
                            scale=self.plt_params.detection_scale,
                            aperture_radius_arcsec=self.plt_params.detection_radius_arcsec,
                            aperture_linewidth=self.plt_params.detection_aperture_linewidth,
                            ncols=self.plt_params.detection_ncols,
                            show_title=True,
                            ax_container=None,   
                        )
                        figures_detection[group] = detection_figure
                
                filter_label_str = (f'{group.split("|")[0]}{offset:+.1f}' if offset != 0 else group.split("|")[0])
                legend_filter = [Line2D([0], [0],
                                        marker = 'o',
                                        linestyle = 'None',
                                        markersize = 8,
                                        markerfacecolor = color,
                                        markeredgecolor = 'k',
                                        label = filter_label_str)]
                legend_filter_all.extend(legend_filter)
                
                legend_observatory = [Line2D([0], [0],
                                        marker = marker,
                                        linestyle = 'None',
                                        markersize = 8,
                                        markerfacecolor = 'none',
                                        markeredgecolor = 'k',
                                        label = observatory)]
                if self.plt_params.non_detection_enabled and non_detection_exist:
                    legend_observatory.append(Line2D([0], [0],
                                    marker = 'v',
                                    linestyle = 'None',
                                    markersize = 8,
                                    markerfacecolor = 'none',
                                    markeredgecolor = 'k',
                                    label = 'Non-detection'))
                legend_observatory_all.extend(legend_observatory)
                

                if not plot_multiple_lightcurve:
                    ax.set_xlabel("Obsdate [MJD]", fontsize=self.plt_params.xlabel_fontsize)
                    ax.set_ylabel("Magnitude" if is_mag else "Flux", fontsize=self.plt_params.ylabel_fontsize)

                    if self.plt_params.xlim:
                        ax.set_xlim(*self.plt_params.xlim)
                    if self.plt_params.ylim:
                        ax.set_ylim(*self.plt_params.ylim)
                    else:
                        try:
                            ax.set_ylim(y_median - 5 * y_std, y_median + 5 * y_std)
                        except:
                            pass
                    if self.plt_params.xticks is not None:
                        ax.set_xticks(self.plt_params.xticks)
                    if self.plt_params.yticks is not None:
                        ax.set_yticks(self.plt_params.yticks)
                    if "MAG" in flux_key.upper():
                        ax.invert_yaxis()
                    ax.tick_params(axis='x', labelrotation=45)
                    ax.grid(True, which='major', alpha=0.3)
                    ax.minorticks_on()
                    
                    if title is not None:
                        ax.set_title(f"{title}")
                        
                    leg1 = ax.legend(
                        handles=legend_filter,
                        ncol=self.plt_params.ncols,
                        loc=self.plt_params.scatter_legend_position,
                        fontsize=self.plt_params.scatter_legend_fontsize
                    )
                    ax.add_artist(leg1)
                    
                    leg2 = ax.legend(
                        handles=legend_observatory,
                        fontsize=self.plt_params.obsdate_legend_fontsize,
                        loc=self.plt_params.obsdate_legend_position,
                        )
                    ax.add_artist(leg2)    
                    figures[group] = fig
                    axes[group] = ax
            
            if plot_multiple_lightcurve:
                ax.set_xlabel("Obsdate [MJD]", fontsize=self.plt_params.xlabel_fontsize)
                ax.set_ylabel("Magnitude" if is_mag else "Flux", fontsize=self.plt_params.ylabel_fontsize)
                
                if self.plt_params.xlim:
                    ax.set_xlim(*self.plt_params.xlim)
                if self.plt_params.ylim:
                    ax.set_ylim(*self.plt_params.ylim)
                if self.plt_params.xticks is not None:
                    ax.set_xticks(self.plt_params.xticks)
                if self.plt_params.yticks is not None:
                    ax.set_yticks(self.plt_params.yticks)
                if "MAG" in flux_key.upper():
                    ax.invert_yaxis()
                ax.tick_params(axis='x', labelrotation=45)
                ax.grid(True, which='major', alpha=0.3)
                ax.minorticks_on()
                
                if title is not None:
                    ax.set_title(f"{title}")
                    
                leg1 = ax.legend(
                    handles=legend_filter_all,
                    ncol=self.plt_params.ncols,
                    loc=self.plt_params.scatter_legend_position,
                    fontsize=self.plt_params.scatter_legend_fontsize
                )
                ax.add_artist(leg1)
                
                leg2 = ax.legend(
                    handles=legend_observatory,
                    fontsize=self.plt_params.obsdate_legend_fontsize,
                    loc=self.plt_params.obsdate_legend_position,
                    )
                ax.add_artist(leg2)    
                figures[group] = fig
                axes[group] = ax

            return figures, figures_detection, axes, tbl
        
    def show_detection(self,
                       ra: float,
                       dec: float,
                       filter: str = None,
                       matching_radius_arcsec: float = 5.0,
                       ra_key: str = 'X_WORLD',
                       dec_key: str = 'Y_WORLD',
                       flux_key: str = 'MAGSKY_APER_2',
                       fluxerr_key: str = 'MAGERR_APER_2',
                       downsample: int = 1,
                       zoom_radius_pixel: int = 50,
                       cmap: str = 'grey_r',
                       scale: str = 'zscale',
                       aperture_radius_arcsec: float = 5.0,
                       aperture_linewidth: float = 1.5,
                       ncols: int = 5,
                       show_title: bool = True,
                       # Other parameters
                       ax_container = None,
                       ):
        """
        Show detections of a source in all filters.

        Returns
        -------
        fig or list[fig]
            One figure if obsdate is given, otherwise a list of figures.
        """
        def _relative_fontsize(ax, fig, scale=10):
            """Returns fontsize scaled to axis height."""
            bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            height = bbox.height  # axis height in inches
            return max(6, height * scale)  # floor to avoid too small fonts
        
        if self.data is None:
            self.extract_source_info(
                ra, dec,
                ra_key=ra_key,
                dec_key=dec_key,
                matching_radius_arcsec=matching_radius_arcsec
            )

        if self.data is None or len(self.data) == 0:
            return None

        tbl = self.data.copy()

        # --------------------------------------------------
        # Select obsdate group (if requested)
        # --------------------------------------------------
        if filter is not None:
            tbl = tbl[tbl['filter'] == filter]
        
        # --------------------------------------------------
        # Loop over obsdate groups
        # --------------------------------------------------
        figures = []

        for i, filter_group in enumerate(np.unique(tbl['filter_group'])):
            tbl_group = tbl[tbl['filter_group'] == filter_group]
            filter_group = tbl_group['filter'][0]
            telname = tbl_group['telname'][0]

            # Sort filters by predefined order
            obsdates = sorted(tbl_group['mjd'])
            n_obsdates = len(obsdates)
            if ax_container is None:
                if n_obsdates < ncols:
                    ncols = n_obsdates
            nrows = int(np.ceil(n_obsdates / ncols))

            if ax_container is None:
                # ---- original behavior ----
                fig, axes = plt.subplots(
                    nrows=nrows,
                    ncols=ncols,
                    figsize=(1.5*ncols, 1.5*nrows),
                    squeeze=False
                )
                axes_flat = axes.flatten()

            else:
                # ---- draw into existing axis ----
                fig = ax_container.figure
                ax_container.axis("off")

                gs = GridSpecFromSubplotSpec(
                    nrows, ncols,
                    subplot_spec=ax_container.get_subplotspec(),
                    wspace=0.02,
                    hspace=0.02
                )

                axes_flat = [
                    fig.add_subplot(gs[i])
                    for i in range(nrows * ncols)
                ]
            
            
            if show_title:
                fig.subplots_adjust(
                    left=0.01,
                    right=0.99,
                    bottom=0.01,
                    top=0.85,
                    wspace=0.0,
                    hspace=0.0
                )
            else:
                fig.subplots_adjust(
                    left=0.01,
                    right=0.99,
                    bottom=0.01,
                    top=0.99,
                    wspace=0.0,
                    hspace=0.0
                )
            
            for ax in axes_flat[len(obsdates):]:
                ax.set_visible(False)

            for i, obsdate in enumerate(obsdates):
                ax = axes_flat[i]

                row = tbl_group[tbl_group['mjd'] == obsdate][0]
                ra_detection = row[ra_key]
                dec_detection = row[dec_key]
                meta_id = row['meta_id']
                telname = row['telname']
                target_img = self.source_catalogs[meta_id].target_img
                
                # Check if the source is detected
                # Check if the source is detected
                c = 'g'
                label = 'Detected'
                if is_masked(row[flux_key]) or is_masked(row[fluxerr_key]):
                    c = 'r'
                    label = 'Not detected'
                    
                if np.isnan(row[flux_key]) or np.isnan(row[fluxerr_key]):
                    c = 'r'
                    label = 'Nan/Negative Flux'
                    
                try:
                    target_img.show_position(
                        ra_detection, dec_detection,
                        radius_arcsec=aperture_radius_arcsec,
                        coord_type='coord',
                        downsample=downsample,
                        zoom_radius_pixel=zoom_radius_pixel,
                        cmap=cmap,
                        scale=scale,
                        figsize = (3,3),
                        aperture_linewidth = aperture_linewidth,
                        aperture_color = c,
                        aperture_label = label,
                        ax=ax,
                        title=None,
                        title_fontsize = None
                    )
                    target_img.clear(verbose = False)
                except OutofCoverageError:
                    c = 'r'
                    label = 'Out of coverage'
                    ax.text(
                        0.5, 0.98, f'{Time(obsdate, format= "mjd").datetime.strftime("%m-%d %H:%M")} ({telname})',
                        transform=ax.transAxes,
                        ha='center',
                        va='top',
                    )
                    ax.set_frame_on(False)
                
                ax.text(
                    0.5, 0.98, f'{Time(obsdate, format= "mjd").datetime.strftime("%m-%d %H:%M")} ({telname})',
                    transform=ax.transAxes,
                    ha='center',
                    va='top',
                    fontsize=_relative_fontsize(ax, fig, 8),
                    color='white',
                    weight='bold',
                    bbox=dict(
                        facecolor='black',
                        alpha=0.4,
                        edgecolor='none',
                        pad=0
                    )
                )
                
                ax.set_axis_on()
                ax.set_frame_on(True)

                # Explicitly re-enable spines
                for side in ["left", "right", "top", "bottom"]:
                    spine = ax.spines[side]
                    spine.set_visible(True)
                    spine.set_color("black")
                    spine.set_linewidth(2.0)
                    
                ax.set_xticks([])
                ax.set_yticks([])

            if show_title:
                fig.text(
                    0.5, 0.995,
                    f"Detection for {filter_group}",
                    ha="center",
                    va="top",
                    fontsize=_relative_fontsize(ax, fig, 10),
                    weight="bold"
                )

            figures.append(fig)

        # --------------------------------------------------
        # Return logic
        # --------------------------------------------------
        if obsdate is not None:
            return figures[0]
        return figures

    def extract_source_info(self,
                            ra: float,
                            dec: float,
                            ra_key: str = 'X_WORLD',
                            dec_key: str = 'Y_WORLD',
                            flux_key: Union[str, list[str]] = None,
                            fluxerr_key: Union[str, list[str]] = None, 
                            zperr_key: Union[str, list[str]] = None,
                            depth_key: Union[str, list[str]] = None,
                            matching_radius_arcsec=5.0,
                            verbose: bool = True):
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
        flux_key : str or sequence of str
            Photometry value columns you want to carry over (e.g., 'MAGSKY_APER_1').
            Accepts a single key or a list/tuple of keys.
        fluxerr_key : str or sequence of str
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
        flux_keys = ['MAGSKY_AUTO', 'MAGSKY_APER', 'MAGSKY_APER_1', 'MAGSKY_APER_2', 'MAGSKY_APER_3', 'MAGSKY_APER_4']
        fluxerr_keys = ['MAGERR_AUTO', 'MAGERR_APER', 'MAGERR_APER_1', 'MAGERR_APER_2', 'MAGERR_APER_3', 'MAGERR_APER_4']
        zperr_keys = ['ZPERR_AUTO', 'ZPERR_APER', 'ZPERR_APER_1', 'ZPERR_APER_2', 'ZPERR_APER_3', 'ZPERR_APER_4']
        depth_keys = ['UL5SKY_AUTO', 'UL5SKY_APER', 'UL5SKY_APER_1', 'UL5SKY_APER_2', 'UL5SKY_APER_3', 'UL5SKY_APER_4']
        
        if flux_key is not None:
            flux_keys.extend(np.atleast_1d(flux_key))
        if fluxerr_key is not None:
            fluxerr_keys.extend(np.atleast_1d(fluxerr_key))
        if zperr_key is not None:
            zperr_keys.extend(np.atleast_1d(zperr_key))
        if depth_key is not None:
            depth_keys.extend(np.atleast_1d(depth_key))
        
        flux_keys = list(set(flux_keys))
        fluxerr_keys = list(set(fluxerr_keys))
        zperr_keys = list(set(zperr_keys))
        depth_keys = list(set(depth_keys))
        
        # Ensure merged table exists (pulling at least the requested keys + their ZPERR counterparts)
        needed_data_keys = set()
        for fk, fek, zek, dk in zip(flux_keys, fluxerr_keys, zperr_keys, depth_keys):
            needed_data_keys.add(fk)
            needed_data_keys.add(fek)
            needed_data_keys.add(zek)
            needed_data_keys.add(dk)

        total_number_sources = 0
        for catalog in tqdm(list(self.source_catalogs.values()), desc="Reading catalogs..."):
            catalog.data
            total_number_sources += catalog.nselected
            
        if total_number_sources > 50000:
            self.helper.print(f"Total number of sources is greater than 30000. Only target nearby the given coordinates will be used for merged_tbl", verbose)
        
        for catalog in tqdm(list(self.source_catalogs.values()), desc="Selecting sources..."):
            catalog.select_sources(ra, dec, matching_radius=matching_radius_arcsec)
        
        self.merged_tbl, self.metadata = self._merge_catalogs(ra_key = ra_key,
                                                              dec_key = dec_key,
                                                              max_distance_arcsec=matching_radius_arcsec,
                                                              join_type = 'outer',
                                                              data_keys=sorted(needed_data_keys),
                                                              )

        # Find closest source
        selected = self.select_source(ra, 
                                      dec, 
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
            idx: {'meta_id': idx, **{k: v for k, v in meta.items() if k not in ('ra', 'dec')}}
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
        ra_basis = row['ra_basis']
        dec_basis = row['dec_basis']

        if isinstance(result_tbl['X_WORLD'], MaskedColumn):
            result_tbl['X_WORLD'] = result_tbl['X_WORLD'].filled(ra_basis)
        if isinstance(result_tbl['Y_WORLD'], MaskedColumn):
            result_tbl['Y_WORLD'] = result_tbl['Y_WORLD'].filled(dec_basis)

        result_tbl['coord'] = SkyCoord(
            ra=result_tbl['X_WORLD'],
            dec=result_tbl['Y_WORLD'],
            unit='deg'
        )

        # Time columns
        if 'obsdate' in result_tbl.colnames:
            t = Time(result_tbl['obsdate'])
            result_tbl['mjd'] = t.mjd
            result_tbl['jd'] = t.jd
            
        groups = [f"{f}|{o}" for f, o in zip(result_tbl['filter'], result_tbl['observatory'])]
        result_tbl['filter_group'] = groups

        # Column ordering: metadata ? requested photometry ? remaining
        meta_order = ['filter', 'exptime', 'obsdate', 'mjd', 'jd',
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
    
    def select_source(self, 
                      ra: Union[float, list, np.ndarray],
                      dec: Union[float, list, np.ndarray],
                      matching_radius_arcsec: float = 5.0):
        """
        Search for sources in the merged catalog.
        
        Parameters
        ----------
        ra, dec : float
            Sky position in degrees.
        matching_radius_arcsec : float
            Search radius for the source match.

        """
        if self.merged_tbl is None:
            raise RuntimeError("self.merged_tbl is missing. Run merge_catalogs() first.")


        ra = np.atleast_1d(ra)
        dec = np.atleast_1d(dec)
        input_coords = SkyCoord(ra=ra, dec=dec, unit='deg')
        
        catalog_coords = self.merged_tbl['coord']
        
        matched_catalog, matched_input, unmatched_catalog = self.helper.cross_match(catalog_coords, input_coords, matching_radius_arcsec)
        print(f"Matched {len(matched_catalog)} sources out of {len(input_coords)} input positions.")
        return self.merged_tbl[matched_catalog]
    
    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, value):
        self._data = value
        try:
            if self.catalogset is None:
                self.helper.print('Loading catalogs from input table...', True)
                catalog_pathlist = value['path']
                catalog_list = [Catalog(path) for path in catalog_pathlist]
                self.catalogset = CatalogSet(catalog_list)
                self.source_catalogs = {i: catalog for i, catalog in enumerate(self.catalogset.catalogs)}
        except:
            pass
    
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
        merged_tbl, metadata = self.catalogset.merge_catalogs(
            max_distance_arcsec=max_distance_arcsec,
            ra_key=ra_key,
            dec_key=dec_key,
            join_type=join_type,
            data_keys=data_keys)
        return merged_tbl, metadata
    
    def _plt_params(self):
        class PlotParams: 
            def __init__(self):
                self._rcparams = {
                    'figure.figsize': (13,8),
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
                    'xtick.labelsize': 15,
                    'ytick.labelsize': 15,
                    'legend.fontsize': 12,
                    'axes.grid': True,
                    'grid.alpha': 0.3,
                    'xtick.direction': 'in',
                    'ytick.direction': 'in',
                    'xtick.top': True,
                    'ytick.right': True,
                    }
                # Custom axis control
                self.xlim = None
                self.ylim = None
                self.xticks = None
                self.yticks = None
                
                # Color parameters
                self.cmap = 'jet'
                
                # Label parameters
                self.xlabel_fontsize = 20
                self.ylabel_fontsize = 20
                
                # Legend parameters
                self.scatter_legend_position = 'lower right'  # 'best', 'upper right', 'lower left', etc.
                self.scatter_legend_fontsize = 15
                self.obsdate_legend_position = 'upper right'  # 'best', 'upper right', 'lower left', etc.
                self.obsdate_legend_fontsize = 20
                self.ncols = 5
                
                # Scatter parameters
                self.scatter_hollowmarker = False  # True = hollow, False = filled
                self.scatter_color = 'y'
                self.scatter_markersize = 150
                self.scatter_alpha = 1.0
                
                # Error bar parameters
                self.errorbar_enabled = True  # Optional switch
                self.errorbar_color = 'k'
                self.errorbar_capsize = 3.5
                self.errorbar_elinewidth = 0.8
                self.errorbar_alpha = 0.8
                
                # Non-detection parameters
                self.non_detection_enabled = True
                self.non_detection_color = 'y'
                self.non_detection_markersize = 150
                self.non_detection_alpha = 0.2
                self.non_detection_marker = 'v'
                self.non_detection_arrow_length = 0.15
                self.non_detection_arrow_width = 2.5
                self.non_detection_arrow_style = '-|>'
                self.non_detection_arrow_color = 'k'

                # Line parameters
                self.line_enabled = True
                self.line_style = 'solid'
                self.line_color = 'y'
                self.line_width = 1.0
                self.line_alpha = 0.2
                
                # Detection parameters
                self.detection_aperture_arcsec = 10.0
                self.detection_aperture_linewidth = 1.0
                self.detection_downsample = 1
                self.detection_zoom_radius_pixel = 50
                self.detection_cmap = 'grey_r'
                self.detection_scale = 'zscale'
                self.detection_radius_arcsec = 5.0
                self.detection_ncols = 5
                self.detection_show_title = False

            def __getattr__(self, name):
                rc_name = name.replace('_', '.')
                if rc_name in self._rcparams:
                    return self._rcparams[rc_name]
                raise AttributeError(f"'PlotParams' object has no attribute '{name}'")

            def __setattr__(self, name, value):
                if name.startswith('_') or name in ('xlim', 
                                                    'ylim', 
                                                    'xticks', 
                                                    'yticks',
                                                    'cmap', 
                                                    'xlabel_fontsize',
                                                    'ylabel_fontsize',
                                                    'scatter_legend_position',
                                                    'obsdate_legend_position',
                                                    'scatter_legend_fontsize',
                                                    'obsdate_legend_fontsize',
                                                    'ncols',
                                                    'scatter_hollowmarker',
                                                    'scatter_color',
                                                    'scatter_markersize',
                                                    'scatter_alpha',
                                                    'errorbar_enabled', 
                                                    'errorbar_color',
                                                    'errorbar_capsize',
                                                    'errorbar_elinewidth',
                                                    'errorbar_alpha',
                                                    'non_detection_enabled',
                                                    'non_detection_color',
                                                    'non_detection_markersize',
                                                    'non_detection_alpha',
                                                    'non_detection_marker',
                                                    'non_detection_arrow_length',
                                                    'non_detection_arrow_width',
                                                    'non_detection_arrow_style',
                                                    'non_detection_arrow_color',
                                                    'line_enabled',
                                                    'line_style',
                                                    'line_color',
                                                    'line_width',
                                                    'line_alpha',
                                                    'detection_aperture_arcsec',
                                                    'detection_aperture_linewidth',
                                                    'detection_downsample',
                                                    'detection_zoom_radius_pixel',
                                                    'detection_cmap',
                                                    'detection_scale',
                                                    'detection_radius_arcsec',
                                                    'detection_ncols',
                                                    'detection_show_title',
                                                    ):
                    super().__setattr__(name, value)
                else:
                    rc_name = name.replace('_', '.')
                    if rc_name in self._rcparams:
                        self._rcparams[rc_name] = value
                    else:
                        raise AttributeError(f"'PlotParams' has no rcParam '{rc_name}'")

            def get_scatter_kwargs(self, color: str = None, shape: str = None):
                if color is None:
                    color = self.scatter_color
                scatter_kwargs = dict(s=self.scatter_markersize,        
                                      alpha=self.scatter_alpha,
                                      )
                scatter_kwargs['marker'] = shape
                
                if self.scatter_hollowmarker is True:
                    scatter_kwargs['facecolors'] = 'none'
                    scatter_kwargs['edgecolors'] = color
                else:
                    scatter_kwargs['facecolors'] = color
                    scatter_kwargs['edgecolors'] = 'k'
                return scatter_kwargs
            
            def get_errorbar_kwargs(self, color: str = None):     
                if color is None:
                    color = self.errorbar_color
                errorbar_kwargs = dict(
                    capsize=self.errorbar_capsize,
                    elinewidth=self.errorbar_elinewidth,
                    alpha=self.errorbar_alpha,
                )
                
                errorbar_kwargs['color'] = color
                if self.scatter_hollowmarker is True:
                    errorbar_kwargs['mfc'] = 'none'
                    errorbar_kwargs['mec'] = color
                else:
                    errorbar_kwargs['mfc'] = color
                    errorbar_kwargs['mec'] = 'k'

                if self.errorbar_enabled is False:
                    errorbar_kwargs['elinewidth'] = 0
                    errorbar_kwargs['capsize'] = 0
                    
                return errorbar_kwargs
            
            def get_line_kwargs(self, color: str = None):
                if color is None:
                    color = self.line_color
                line_kwargs = dict(
                    color=color,
                    alpha=self.line_alpha,
                    linewidth=self.line_width,
                )
                return line_kwargs
            
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
                txt += f"cmap = {self.cmap}\n"
                txt += f"xlabel_fontsize = {self.xlabel_fontsize}\n"
                txt += f"ylabel_fontsize = {self.ylabel_fontsize}\n"
                txt += f"scatter_legend_position = {self.scatter_legend_position}\n"
                txt += f"obsdate_legend_position = {self.obsdate_legend_position}\n"
                txt += f"scatter_legend_fontsize = {self.scatter_legend_fontsize}\n"
                txt += f"obsdate_legend_fontsize = {self.obsdate_legend_fontsize}\n"
                txt += f"ncols = {self.ncols}\n"
                
                txt += 'Scatter Parameters -----------------\n'
                txt += f"scatter_hollowmarker = {self.scatter_hollowmarker}\n"
                txt += f"scatter_color = {self.scatter_color}\n"
                txt += f"scatter_markersize = {self.scatter_markersize}\n"
                txt += f"scatter_alpha = {self.scatter_alpha}\n"
                
                txt += 'Error Bar Configuration ---------\n'
                txt += f"errorbar_enabled = {self.errorbar_enabled}\n"                
                txt += f"errorbar_color = {self.errorbar_color}\n"
                txt += f"errorbar_capsize = {self.errorbar_capsize}\n"
                txt += f"errorbar_elinewidth = {self.errorbar_elinewidth}\n"
                txt += f"errorbar_alpha = {self.errorbar_alpha}\n"
                
                txt += 'Line Parameters -----------------\n'
                txt += f"line_enabled = {self.line_enabled}\n"
                txt += f"line_style = {self.line_style}\n"
                txt += f"line_color = {self.line_color}\n"
                txt += f"line_width = {self.line_width}\n"
                txt += f"line_alpha = {self.line_alpha}\n"
                
                txt += 'Detection Parameters -----------------\n'
                txt += f"detection_aperture_arcsec = {self.detection_aperture_arcsec}\n"
                txt += f"detection_downsample = {self.detection_downsample}\n"
                txt += f"detection_zoom_radius_pixel = {self.detection_zoom_radius_pixel}\n"
                txt += f"detection_cmap = {self.detection_cmap}\n"
                txt += f"detection_scale = {self.detection_scale}\n"
                txt += f"detection_radius_arcsec = {self.detection_radius_arcsec}\n"
                txt += f"detection_ncols = {self.detection_ncols}\n"
                return txt
        return PlotParams()
    
# %%

# %%

if __name__ == '__main__':
    from ezphot.utils import DataBrowser
    dbrowser = DataBrowser('scidata')
    dbrowser.objname = 'T22956'
    catalog_set = dbrowser.search(pattern='coadd_scaled*com.fits.cat', return_type='catalog')
    self = LightCurve(catalog_set)
    self.extract_source_info(ra = 233.8575139, dec = 12.0577883, flux_key = 'MAGSKY_APER_2', fluxerr_key = 'MAGERR_APER_2', zperr_key = 'ZPERR_APER_2', depth_key = 'UL5SKY_APER_2', matching_radius_arcsec = 5.0, verbose = True)
# %%
if __name__ == '__main__':
    self = LightCurve(catalog_set)
    self.extract_source_info(ra = 233.8575139, dec = 12.0577883, flux_key = 'MAGSKY_APER_2', fluxerr_key = 'MAGERR_APER_2', zperr_key = 'ZPERR_APER_2', depth_key = 'UL5SKY_APER_2', matching_radius_arcsec = 5.0, verbose = True)

    ra = 233.8575139
    dec = 12.0577883
    filter = None
    matching_radius_arcsec = 5.0
    ra_key = 'X_WORLD'
    dec_key = 'Y_WORLD'
    flux_key = 'MAGSKY_APER_2'
    fluxerr_key = 'MAGERR_APER_2'
    zperr_key = 'ZPERR_APER_2'
    depth_key = 'UL5SKY_APER_2'
    plot_all_in_one_figure = False
    apply_offset = True
    overplot_stamp = True
    
    verbose = True
    title = None
    result = self.plot(ra = ra, dec = dec, filter = filter, matching_radius_arcsec = matching_radius_arcsec, ra_key = ra_key, dec_key = dec_key, flux_key = flux_key, fluxerr_key = fluxerr_key, zperr_key = zperr_key, depth_key = depth_key, plot_all_in_one_figure = plot_all_in_one_figure, apply_offset = apply_offset, overplot_stamp = overplot_stamp, verbose = verbose, title = title)

#%%