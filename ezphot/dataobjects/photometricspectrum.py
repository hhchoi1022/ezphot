#%%
from typing import Union
from tqdm import tqdm
import numpy as np
from numpy.ma import getmaskarray
from numpy.ma import is_masked
import matplotlib.pyplot as plt
from matplotlib import cycler
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from matplotlib.gridspec import GridSpecFromSubplotSpec
from astropy.table import Table
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.stats import sigma_clipped_stats
from astropy.table import MaskedColumn

from ezphot.dataobjects import CatalogSet
from ezphot.helper import Helper
from ezphot.utils import CatalogQuerier

#%%%
class PhotometricSpectrum:
    """
    Photometric spectrum object.
    
    This object is used to plot the photometric spectrum of a source.
    It is initialized with a catalogset object, which is a CatalogSet object.
    The catalogset object is used to extract the source information from the catalogs.
    
    To change the plot parameters, modify the plt_params attribute.
    """
    
    def __init__(self, catalogset: CatalogSet = None):
        """
        Initialize the PhotometricSpectrum object.
        
        Parameters
        ----------
        catalogset : CatalogSet
            The source catalogs to use for the photometric spectrum.
        """
        # if not isinstance(catalogset, CatalogSet):
        #     raise TypeError("catalogset must be an instance of CatalogSet")
        self.helper = Helper()
        self.catalogset = catalogset
        self.source_catalogs = {i: catalog for i, catalog in enumerate(catalogset.catalogs)}
        self.merged_tbl = None
        self.metadata = None
        self.plt_params = self._plt_params()
        self.CatalogQuerier = CatalogQuerier(catalog_key = None)
        self.data = None
        
    OFFSET = 2
        
    EFFECTIVE_WAVELENGTHS_NM = {
        'm400': 400.0, 'm412': 412.5, 'm425': 425.0, 'm437': 437.5, 
        'm450': 450.0, 'm462': 462.5, 'm475': 475.0, 'm487': 487.5,
        'm500': 500.0, 'm512': 512.5, 'm525': 525.0, 'm537': 537.5,
        'm550': 550.0, 'm562': 562.5, 'm575': 575.0, 'm587': 587.5,
        'm600': 600.0, 'm612': 612.5, 'm625': 625.0, 'm637': 637.5,
        'm650': 650.0, 'm662': 662.5, 'm675': 675.0, 'm687': 687.5,
        'm700': 700.0, 'm712': 712.5, 'm725': 725.0, 'm737': 737.5,
        'm750': 750.0, 'm762': 762.5, 'm775': 775.0, 'm787': 787.5,
        'm800': 800.0, 'm812': 812.5, 'm825': 825.0, 'm837': 837.5,
        'm850': 850.0, 'm862': 862.5, 'm875': 875.0, 'm887': 887.5,
        # SDSS ugriz (DOI+2010)
        'u': 349.8, 'g': 462.7, 'r': 613.9, 'i': 746.7, 'z': 892.7,
        # PS1 ugizy (TONRY+2012)
        'g_ps1': 481, 'r_ps1': 617, 'i_ps1': 752, 'z_ps1': 866, 'y_ps1': 962,
        # Johnson-Cousins UBVRI (Ground based, https://mfouesneau.github.io/pyphot/libcontent.html)
        'U': 363.5, 'B': 429.7, 'V': 547.0, 'R': 647.1, 'I': 787.2,
        # 2MASS JHK (COHEN+2003)
        'J': 1235, 'H': 1662, 'K': 2159,
        # WISE W1-W4 (https://svo2.cab.inta-csic.es/svo/theory/fps/index.php?mode=browse&gname=WISE&asttype=)
        'W1': 3352.6, 'W2': 4620.8, 'W3': 11560.8, 'W4': 22088.3,
    }
    
    FILTER_SHAPE = {
        'm400': 'o', 'm412': 'o', 'm425': 'o', 'm437': 'o',
        'm450': 'o', 'm462': 'o', 'm475': 'o', 'm487': 'o',
        'm500': 'o', 'm512': 'o', 'm525': 'o', 'm537': 'o',
        'm550': 'o', 'm562': 'o', 'm575': 'o', 'm587': 'o',
        'm600': 'o', 'm612': 'o', 'm625': 'o', 'm637': 'o',
        'm650': 'o', 'm662': 'o', 'm675': 'o', 'm687': 'o',
        'm700': 'o', 'm712': 'o', 'm725': 'o', 'm737': 'o',
        'm750': 'o', 'm762': 'o', 'm775': 'o', 'm787': 'o',
        'm800': 'o', 'm812': 'o', 'm825': 'o', 'm837': 'o',
        'm850': 'o', 'm862': 'o', 'm875': 'o', 'm887': 'o',
        # SDSS ugriz
        'u': 's', 'g': 's', 'r': 's', 'i': 's', 'z': 's',
        # PS1 ugizy
        'g_ps1': 's', 'r_ps1': 's', 'i_ps1': 's', 'z_ps1': 's', 'y_ps1': 's',
        # Johnson-Cousins UBVRI
        'U': 's', 'B': 's', 'V': 's', 'R': 's', 'I': 's',
        # 2MASS JHK
        'J': 's', 'H': 's', 'K': 's',
        # WISE W1-W4
        'W1': 's', 'W2': 's', 'W3': 's', 'W4': 's',
    }
    
    FILTER_BANDWIDTH_NM = {
        # -------------------------------------------------
        # 7DT / Medium-band filters (uniform)
        # -------------------------------------------------
        'm400': 25, 'm412': 25, 'm425': 25, 'm437': 25,
        'm450': 25, 'm462': 25, 'm475': 25, 'm487': 25,
        'm500': 25, 'm512': 25, 'm525': 25, 'm537': 25,
        'm550': 25, 'm562': 25, 'm575': 25, 'm587': 25,
        'm600': 25, 'm612': 25, 'm625': 25, 'm637': 25,
        'm650': 25, 'm662': 25, 'm675': 25, 'm687': 25,
        'm700': 25, 'm712': 25, 'm725': 25, 'm737': 25,
        'm750': 25, 'm762': 25, 'm775': 25, 'm787': 25,
        'm800': 25, 'm812': 25, 'm825': 25, 'm837': 25,
        'm850': 25, 'm862': 25, 'm875': 25, 'm887': 25,

        # -------------------------------------------------
        # SDSS ugriz (Doi+2010)
        # -------------------------------------------------
        'u': 64.1, 'g': 130.8, 'r': 115.5, 'i': 126.8, 'z': 117.8,
        # -------------------------------------------------
        # Pan-STARRS1 (Tonry+2012)
        # -------------------------------------------------
        'g_ps1': 137, 'r_ps1': 139, 'i_ps1': 129, 'z_ps1': 104, 'y_ps1': 83,
        # -------------------------------------------------
        # Johnson–Cousins UBVRI (Bessell 1990)
        # -------------------------------------------------
        'U': 66, 'B': 94, 'V': 88, 'R': 138, 'I': 149,
        # -------------------------------------------------
        # 2MASS JHK (https://svo2.cab.inta-csic.es/svo/theory/fps/index.php?mode=browse&gname=WISE&asttype=)
        # -------------------------------------------------
        'J': 162, 'H': 251, 'K': 262,
        # -------------------------------------------------
        # WISE W1–W4 (https://svo2.cab.inta-csic.es/svo/theory/fps/index.php?mode=browse&gname=WISE&asttype=)
        # -------------------------------------------------
        'W1': 662.6, 'W2': 1042.2, 'W3': 5505.5, 'W4': 4101.7,
    }

    
    def __repr__(self):
        txt = f'PHOTOMETRIC SPECTRUM OBJECT (n_catalogs = {len(self.catalogset.catalogs)})\n'
        txt += str(self.plt_params)
        return txt
    
    def plot(self, 
             ra: float,
             dec: float,
             obsdate: str = None,
             matching_radius_arcsec: float = 5.0,
             ra_key: str = 'X_WORLD',
             dec_key: str = 'Y_WORLD',
             flux_key: str = 'MAGSKY_APER_1',
             fluxerr_key: str = 'MAGERR_APER_1',
             
             plot_all_in_one_figure: bool = True,
             overplot_gaiaxp: bool = False,
             overplot_sdss: bool = False,
             overplot_ps1: bool = False,
             overplot_stamp: bool = False,
             
             verbose: bool = True,
             title: str = None
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
        
        # 1. Prepare data
        # 1.1. Data formatting
        if self.data is None:
            self.extract_source_info(
                ra, dec,
                ra_key=ra_key,
                dec_key=dec_key,
                flux_key=flux_key,
                fluxerr_key=fluxerr_key,
                matching_radius_arcsec=matching_radius_arcsec
            )
        if self.data is None or len(self.data) == 0:
            self.helper.print(f"[WARNING] No sources found within {matching_radius_arcsec}\" of RA={ra}, Dec={dec}", verbose)
            return None, None, None
        tbl = self.data.copy()
            
        # 1.2. If obsdate is provided, filter the table to the closest group in time
        if obsdate is not None:
            obsdate_mjd = self.helper.flexible_time_parser(obsdate).mjd
            
            # Match against per-group mean MJD
            group_mjd = np.array(tbl['obsdate_mjd_group'], dtype=float)

            # Find the closest group in time
            idx = np.nanargmin(np.abs(group_mjd - obsdate_mjd))
            target_group = tbl['obsdate_group'][idx]

            # Filter table to that single group
            tbl = tbl[tbl['obsdate_group'] == target_group]

            if len(tbl) == 0:
                self.helper.print(
                    f"[WARNING] No data found for obsdate={obsdate}",
                    verbose
                )
                return None, None, None
        
        # 1.3. Remove non-detection when plotting 
        if flux_key in tbl.colnames and fluxerr_key in tbl.colnames:
            mag = tbl[flux_key]
            err = tbl[fluxerr_key]

            mask_mag = getmaskarray(mag)
            mask_err = getmaskarray(err)

            valid_detection = (
                (~mask_mag) & (~mask_err) #&
                # np.isfinite(mag) & np.isfinite(err)
            )

            if len(tbl) == 0:
                print("[WARNING] No valid detections — skipping plot.")
                return None, None, None
            
        # 1.4. Build arrays
        is_mag = "MAG" in flux_key.upper()
        wl = np.array([self._band_to_wavelength_nm(b) for b in tbl['filter']], dtype=float)
        mags  = np.array(tbl[flux_key], dtype=float)
        magerrs = np.array(tbl[fluxerr_key], dtype=float)
        zperrs = np.array(tbl['zp_err'], dtype=float)
        depths = np.array(tbl['depth'], dtype=float)
        errs  = np.array([self._combine_err(m, z) for m, z in zip(magerrs, zperrs)], dtype=float)
        mjds = np.array(tbl['mjd'], dtype=float)
        telname = np.array(tbl['telname'], dtype=object)
        obs = np.array(tbl['observatory'], dtype=object)
        filt = np.array(tbl['filter'], dtype=object) # Filter names
        groups = np.array(tbl['obsdate_group'], dtype=object)
        mjd_groups = np.array(tbl['obsdate_mjd_group'], dtype=float)
        order = np.argsort(wl)
        wl = wl[order]
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
        mjd_groups = mjd_groups[order]
        valid_detection = valid_detection[order]        
        
        unique_groups = sorted(list(set(groups)))
        group_mjds = [Time(g).mjd for g in unique_groups]
        norm = Normalize(vmin=np.nanmin(group_mjds), vmax=np.nanmax(group_mjds))
        self.plt_params.set_palette(n=len(unique_groups))
        
        if (plot_all_in_one_figure) and (len(unique_groups) > 1):
            plot_multiple_spectrum = True
        else:
            plot_multiple_spectrum = False
            
        if overplot_stamp:
            overplot_stamp_together = True
            if plot_multiple_spectrum:
                self.helper.print(
                    "[WARNING] Detection panels are disabled when plotting multiple spectra "
                    "Detection panels are plotted separately for each epoch.",
                    verbose
                )
                overplot_stamp_together = False
        else:
            overplot_stamp_together = False
        
        # 2. Plot the data
        with self.plt_params.apply():
            cmap_name = getattr(self.plt_params, 'cmap', None)
            if cmap_name is None:
                cmap_name = 'jet'
            cmap = cm.get_cmap(cmap_name)

            figures = []  # <-- NEW (used only when plot_with_offset=False)
            figures_detection = []
            
            # ==================================================
            # Case 1: plot WITH offset
            # ==================================================
            if plot_multiple_spectrum:
                colors = [cmap(norm(x)) for x in group_mjds]
                fig_height = np.max([2.5 + len(unique_groups) * 0.35 * self.OFFSET, self.plt_params.figure_figsize[1]])
                width = self.plt_params.figure_figsize[0]
                self.plt_params._rcparams['figure.figsize'] = (width, fig_height)
                self.plt_params._rcparams['figure.dpi'] = self.plt_params.figure_dpi

                fig, ax = plt.subplots(figsize=(width, fig_height),
                                    dpi=self.plt_params.figure_dpi)

                group_iter = zip(unique_groups, colors)
                offset_step = self.OFFSET
            # ==================================================
            # Case 2: plot WITHOUT offset → one figure per group
            # ==================================================
            else:
                colors = [self.plt_params.scatter_color for _ in unique_groups]
                group_iter = zip(unique_groups, colors)
                offset_step = 0.0

            # ==================================================
            # Shared plotting logic
            # ==================================================
            offset = 0.0
            
            legend_color_all = []
            figures = dict()
            figures_detection = dict()
            for cg, col in group_iter:
                m = (groups == cg)
                if np.sum(m) == 0:
                    continue
                detection_mask = m & valid_detection
                non_detection_mask = m & ~valid_detection
                non_detection_exist = np.sum(non_detection_mask) > 0
                
                x_detection = np.array(wl[detection_mask], dtype=float)
                y_detection = np.array(mags[detection_mask], dtype=float)                
                yerr_detection = np.array(errs[detection_mask], dtype=float)
                depth_detection = np.array(depths[detection_mask], dtype=float)
                filter_detection = np.array(filt[detection_mask], dtype=object)
                y_detection += offset
                depth_detection += offset
                y_mean, y_median, y_std = sigma_clipped_stats(y_detection, sigma=3)
                is_medium_band = np.array([f.startswith('m') for f in filter_detection])
                is_broad_band = ~is_medium_band

                if non_detection_exist:
                    x_non_detection = np.array(wl[non_detection_mask], dtype=float)
                    y_non_detection = np.array(mags[non_detection_mask], dtype=float)
                    yerr_non_detection = np.array(errs[non_detection_mask], dtype=float)
                    depth_non_detection = np.array(depths[non_detection_mask], dtype=float)
                    filter_non_detection = np.array(filt[non_detection_mask], dtype=object)
                    y_non_detection += offset
                    depth_non_detection += offset    
                    
                if not plot_multiple_spectrum:
                    width, height = self.plt_params.figure_figsize
                    # fig, ax = plt.subplots(figsize=(width|, height),
                    #                        dpi=self.plt_params.figure_dpi)
                    if overplot_stamp_together:
                        n_stamp = np.sum(m)
                        spec_height = height
                        det_height = max(2.5, 0.4 * n_stamp)
                        fig = plt.figure(
                            figsize=(width, spec_height + det_height),
                            dpi=self.plt_params.figure_dpi
                        )
                        gs = GridSpec(
                            nrows=2,
                            ncols=1,
                            height_ratios=[spec_height, det_height],  # spectrum : detection
                            hspace= 0.12
                        )

                        ax = fig.add_subplot(gs[0])
                        ax_detection = fig.add_subplot(gs[1])
                    else:
                        fig, ax = plt.subplots(
                            figsize=(width, height),
                            dpi=self.plt_params.figure_dpi
                        )
                
                # connect only medium bands
                if self.plt_params.line_style != 'none' and len(x_detection) > 1:
                    ax.plot(
                        x_detection[is_medium_band],
                        y_detection[is_medium_band],
                        zorder = 1,
                        **self.plt_params.get_line_kwargs(color=col))
                    
                markers = [self.FILTER_SHAPE.get(f, None) for f in filter_detection]
                marker_shapes = set(markers)
                for marker in marker_shapes:
                    mask = np.array(markers) == marker
                    x_detection_mask = x_detection[mask]
                    y_detection_mask = y_detection[mask]
                    yerr_detection_mask = yerr_detection[mask]
                    ax.scatter(x_detection_mask, y_detection_mask, zorder = 5, 
                               **self.plt_params.get_scatter_kwargs(col, marker))
                    ax.errorbar(x_detection_mask, y_detection_mask, yerr=yerr_detection_mask, zorder = 4, fmt = 'none', 
                                **self.plt_params.get_errorbar_kwargs(col))
                
                if non_detection_exist:
                    if self.plt_params.non_detection_enabled:
                        ax.scatter(
                            x_non_detection, depth_non_detection,
                            s=self.plt_params.non_detection_markersize,
                            marker=self.plt_params.non_detection_marker,
                            facecolors=col,
                            edgecolors='k',
                            alpha=self.plt_params.non_detection_alpha,
                            zorder=6,
                        )
                        
                        for x, y in zip(x_non_detection,
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
                                zorder=5)
                    
                if overplot_stamp:
                    if overplot_stamp_together:
                        # Remove ticks/frame from container axis
                        ax_detection.axis('off')

                        # Ask show_detection to draw INTO this figure
                        self.show_detection(
                            ra, dec,
                            obsdate=cg,
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
                            obsdate=cg,
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
                        figures_detection[cg] = detection_figure
                
                # --- Legend 1: obsdate_group (color only) ---
                legend_color = [Line2D([0], [0],
                    marker='o',
                    linestyle='None',
                    markersize=8,
                    markerfacecolor=col,
                    markeredgecolor='k',
                    label=f'{cg} ({offset:+.1f})' if offset != 0 else cg)]
                legend_color_all.extend(legend_color)
                offset += offset_step
                
                # --- Legend 2: filter type (scatter shape only) ---
                legend_shape = []
                has_medium = np.any(is_medium_band)
                has_broad = np.any(is_broad_band)
                if has_medium:
                    legend_shape_medium = Line2D([0], [0],
                        marker='o', linestyle='None',markersize = 12,
                        markerfacecolor='none', markeredgecolor='k',
                        label='Medium band')
                    legend_shape.append(legend_shape_medium)
                if has_broad:
                    legend_shape_broad = Line2D([0], [0],
                        marker='s', linestyle='None', markersize = 12,
                        markerfacecolor='none', markeredgecolor='k',
                        label='Broad band')
                    legend_shape.append(legend_shape_broad)
                if self.plt_params.non_detection_enabled and non_detection_exist:
                    legend_shape.append(Line2D([0], [0],
                                    marker = 'v',
                                    linestyle = 'None',
                                    markersize = 12,
                                    markerfacecolor = 'none',
                                    markeredgecolor = 'k',
                                    label = 'Non-detection'))
                    
                if not plot_multiple_spectrum:
                    ax.set_xlabel("Effective Wavelength [nm]", fontsize=self.plt_params.xlabel_fontsize)
                    xlabel = 'Magnitude' if "MAG" in flux_key.upper() else "Flux"
                    xlabel += ' (+ offset)' if self.OFFSET != 0 else ''
                    ax.set_ylabel(xlabel, fontsize=self.plt_params.ylabel_fontsize)
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
                    ax.grid(True, which='major', alpha=0.3)
                    ax.minorticks_on()
                    
                    if title is not None:
                        ax.set_title(f"{title}")

                    leg1 = ax.legend(
                        handles=legend_shape,
                        ncol=self.plt_params.shape_legend_ncols,
                        loc=self.plt_params.shape_legend_position,
                        fontsize=self.plt_params.shape_legend_fontsize
                    )
                    ax.add_artist(leg1)
                    
                    leg2 = ax.legend(
                        handles=legend_color,
                        ncol = self.plt_params.color_legend_ncols,
                        fontsize=self.plt_params.color_legend_fontsize,
                        loc=self.plt_params.color_legend_position,
                        )
                    ax.add_artist(leg2)              
                    figures[cg] = fig
                                    
            if plot_multiple_spectrum:
                ax.set_xlabel("Effective Wavelength [nm]", fontsize=self.plt_params.xlabel_fontsize)
                xlabel = 'Magnitude' if "MAG" in flux_key.upper() else "Flux"
                xlabel += ' (+ offset)' if self.OFFSET != 0 else ''
                ax.set_ylabel(xlabel, fontsize=self.plt_params.ylabel_fontsize)
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
                ax.grid(True, which='major', alpha=0.3)
                ax.minorticks_on()
                
                if title is not None:
                    ax.set_title(f"{title}")
                    
                leg1 = ax.legend(
                    handles=legend_shape,
                    ncol=self.plt_params.shape_legend_ncols,
                    loc=self.plt_params.shape_legend_position,
                    fontsize=self.plt_params.shape_legend_fontsize)
                ax.add_artist(leg1)
                    
                leg2 = ax.legend(
                    handles=legend_color_all,
                    ncol=self.plt_params.color_legend_ncols,
                    fontsize=self.plt_params.color_legend_fontsize,
                    loc=self.plt_params.color_legend_position,
                    )
                ax.add_artist(leg2)
                figures[cg] = fig
                
            # ---------- External overplots ----------
            # GaiaXP: full low-res spectrum converted to AB mag vs nm
            coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
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
            
            return figures, figures_detection, tbl
        

    def show_detection(self,
                       ra: float,
                       dec: float,
                       obsdate: str = None,
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
        if obsdate is not None:
            obsdate_mjd = self.helper.flexible_time_parser(obsdate).mjd
            group_mjd = np.array(tbl['obsdate_mjd_group'], dtype=float)
            idx = np.nanargmin(np.abs(group_mjd - obsdate_mjd))
            target_group = tbl['obsdate_group'][idx]
            tbl = tbl[tbl['obsdate_group'] == target_group]

            # if len(tbl) == 0:
            #     return None
        # --------------------------------------------------
        # Loop over obsdate groups
        # --------------------------------------------------
        figures = []

        for i, obs_group in enumerate(np.unique(tbl['obsdate_group'])):
            tbl_group = tbl[tbl['obsdate_group'] == obs_group]

            # Sort filters by predefined order
            order_map = {f: i for i, f in enumerate(self.EFFECTIVE_WAVELENGTHS_NM)}
            filters = sorted(tbl_group['filter'],
                            key=lambda f: order_map.get(f, np.inf))

            n_filters = len(filters)
            if ax_container is None:
                if n_filters < ncols:
                    ncols = n_filters
            nrows = int(np.ceil(n_filters / ncols))
            
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
                    top=0.94,
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
            
            for ax in axes_flat[len(filters):]:
                ax.set_visible(False)

            for i, filt in enumerate(filters):
                ax = axes_flat[i]

                row = tbl_group[tbl_group['filter'] == filt][0]
                ra_detection = row[ra_key]
                dec_detection = row[dec_key]
                meta_id = row['meta_id']
                telname = row['telname']
                target_catalog = self.source_catalogs[meta_id]
                target_img = target_catalog.target_img
                
                # Check if the source is detected
                c = 'g'
                label = 'Detected'
                if is_masked(row[flux_key]) or is_masked(row[fluxerr_key]):
                    c = 'r'
                    label = 'Not detected'
                
                if np.isinf(row[flux_key]) or np.isinf(row[fluxerr_key]):
                    c = 'r'
                    label = 'Out of coverage'
                    
                if np.isnan(row[flux_key]) or np.isnan(row[fluxerr_key]):
                    c = 'r'
                    label = 'Nan/Negative Flux'
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
                
                ax.text(
                    0.5, 0.98, f'{filt} ({telname})',
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
                    f"Detection @ {obs_group}",
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
            Photometry value column you want to carry over (e.g., 'MAGSKY_APER_1').
        fluxerr_key : str or sequence of str
            Corresponding error column (e.g., 'MAGERR_APER_1'). Must be same length as `flux_key`.
        matching_radius_arcsec : float
            Search radius for the source match.

        Returns
        -------
        astropy.table.Table or None
            One row per exposure/catalog with metadata + requested photometry.
            Returns None if no source is found.
        """
        
        # Normalize keys to lists
        flux_keys = ['MAGSKY_AUTO', 'MAGSKY_APER', 'MAGSKY_APER_1', 'MAGSKY_APER_2', 'MAGSKY_APER_3', 'MAGSKY_APER_4']
        fluxerr_keys = ['MAGERR_AUTO', 'MAGERR_APER', 'MAGERR_APER_1', 'MAGERR_APER_2', 'MAGERR_APER_3', 'MAGERR_APER_4']
        
        if flux_key is not None:
            flux_keys.extend(np.atleast_1d(flux_key))
        if fluxerr_key is not None:
            fluxerr_keys.extend(np.atleast_1d(fluxerr_key))
        
        flux_keys = list(set(flux_keys))
        fluxerr_keys = list(set(fluxerr_keys))
        
        if len(flux_keys) != len(fluxerr_keys):
            raise ValueError("flux_keys and fluxerr_keys must have the same length.")

        # Ensure merged table exists (pulling at least the requested keys + their ZPERR counterparts)
        needed_data_keys = set()
        for fk, fek in zip(flux_keys, fluxerr_keys):
            needed_data_keys.add(fk)
            needed_data_keys.add(fek)
            needed_data_keys.add(fek.replace('MAGERR', 'ZPERR'))

        total_number_sources = 0
        for catalog in tqdm(self.catalogset.catalogs, desc="Reading catalogs..."):
            catalog.data
            total_number_sources += catalog.nselected
        if total_number_sources > 50000:
            self.helper.print(f"Total number of sources is greater than 30000. Only target nearby the given coordinates will be used for merged_tbl", verbose)
        
        for catalog in tqdm(self.catalogset.catalogs, desc="Selecting sources..."):
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
        result_tbl['coord'] = SkyCoord(ra=result_tbl['X_WORLD'], dec=result_tbl['Y_WORLD'], unit='deg')

        # Time columns
        if 'obsdate' in result_tbl.colnames:
            t = Time(result_tbl['obsdate'])
            result_tbl['mjd'] = t.mjd
            result_tbl['jd'] = t.jd
            
        # 1) ensure a 'group' column exists (whatever your group_table does)
        result_tbl = self.helper.group_table(result_tbl, 'mjd')        # must add/keep tbl['group']
        gview = result_tbl.group_by('group')

        # 2) compute a mean-MJD label per group
        key_vals = np.array(gview.groups.keys['group'])  # one key per group
        labels_map = {}
        for i, g in enumerate(gview.groups):
            mjd_mean = float(np.nanmean(g['mjd']))
            date_str = Time(mjd_mean, format='mjd').to_value('iso', subfmt='date_hm')
            labels_map[key_vals[i]] = date_str

        # 3) propagate to all rows
        # mean MJD for each group
        group_keys = np.array(gview.groups.keys['group'])
        group_mjd_map = {}

        for i, g in enumerate(gview.groups):
            mjd_mean = float(np.nanmean(g['mjd']))
            label = Time(mjd_mean, format='mjd').to_value('iso', subfmt='date_hm')
            group_mjd_map[group_keys[i]] = (label, mjd_mean)

        # propagate to all rowsa
        obsdate_group = np.empty(len(result_tbl), dtype=object)
        obsdate_mjd_group = np.empty(len(result_tbl), dtype=float)

        for k in group_keys:
            lab, mjd_val = group_mjd_map[k]
            mask = (result_tbl['group'] == k)
            obsdate_group[mask] = lab
            obsdate_mjd_group[mask] = mjd_val

        result_tbl['obsdate_group'] = obsdate_group
        result_tbl['obsdate_mjd_group'] = obsdate_mjd_group

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
        print(f"Matched {len(self.merged_tbl[matched_catalog])} sources out of {len(input_coords)} input positions.")
        return self.merged_tbl[matched_catalog]
    
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
                    'axes.grid': True,
                    'grid.alpha': 0.3,
                    'xtick.direction': 'in',
                    'ytick.direction': 'in',
                    'xtick.top': True,
                    'ytick.right': True,
                    }
                # Custom axis control
                self.xlim = [300, 1000]
                self.ylim = None
                self.xticks = np.arange(300, 1001, 100)
                self.yticks = None
                
                # Color parameters
                self.cmap = 'jet'
                
                # Label parameters
                self.xlabel_fontsize = 20
                self.ylabel_fontsize = 20

                # Legend parameters
                self.shape_legend_position = 'lower right'  # 'best', 'upper right', 'lower left', etc.
                self.shape_legend_fontsize = 18
                self.shape_legend_ncols = 2
                self.color_legend_position = 'upper right'  # 'best', 'upper right', 'lower left', etc.
                self.color_legend_fontsize = 15
                self.color_legend_ncols = 2
                
                # Scatter parameters
                self.scatter_hollowmarker = False  # True = hollow, False = filled
                self.scatter_color = 'y'
                self.scatter_markersize = 120
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
                                                    'shape_legend_position',
                                                    'color_legend_position',
                                                    'shape_legend_fontsize',
                                                    'color_legend_fontsize',
                                                    'shape_legend_ncols',
                                                    'color_legend_ncols',
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
                    
            def set_palette(self, n: int = 40):
                base = cm.get_cmap(self.cmap, n)
                colors = [base(i) for i in range(base.N)]
                from matplotlib import cycler as _cycler
                self._rcparams['axes.prop_cycle'] = _cycler(color=colors)
                
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
                txt += f"shape_legend_position = {self.shape_legend_position}\n"
                txt += f"color_legend_position = {self.color_legend_position}\n"
                txt += f"shape_legend_fontsize = {self.shape_legend_fontsize}\n"
                txt += f"color_legend_fontsize = {self.color_legend_fontsize}\n"
                txt += f"shape_legend_ncols = {self.shape_legend_ncols}\n"
                txt += f"color_legend_ncols = {self.color_legend_ncols}\n"
                
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

        
        
    
