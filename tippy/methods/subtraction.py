
#%%
from typing import Union, List
from pathlib import Path
from datetime import datetime
from astropy.table import Table, vstack
from astropy.time import Time
from astropy.io import fits, ascii
from tqdm import tqdm
import re
import numpy as np
from shapely.geometry import Polygon
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.units import Quantity
from astropy.wcs import WCS
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import MinMaxScaler
from astropy.visualization import ZScaleInterval, MinMaxInterval
import matplotlib.pyplot as plt
import uuid

from tippy.helper import Helper
from tippy.imageojbects import ScienceImage, ReferenceImage
from tippy.imageojbects import Background, Errormap, Mask
from tippy.methods import TIPStacking  # Adjust import path if needed
from tippy.methods import TIPPhotometricCalibration  # Adjust import path if needed
from tippy.methods import TIPAperturePhotometry  # Adjust import path if needed
from tippy.methods import TIPBackground
from tippy.methods import TIPMasking  # Adjust import path if needed
from tippy.catalog import TIPCatalog
from tippy.utils import ImageQuerier  # Adjust import path if needed

from tippy.methods import TIPProjection  # Adjust import path if needed
#%%
class TIPSubtraction(Helper):
    
    def __init__(self):
        super().__init__()
        self.background = TIPBackground()
        self.combiner = TIPStacking()
        self.aperphot = TIPAperturePhotometry()
        self.photcal = TIPPhotometricCalibration()
        self.masking = TIPMasking()
        self.projection = TIPProjection()
        self.imagequerier = ImageQuerier()
    
    def get_referenceframe_from_image(self, 
                                      target_img: Union[ScienceImage],
                                      max_obsdate: Union[str, float, Time] = None,
                                      sort_key: Union[str, List[str]] = ['fraction', 'depth'],
                                      return_groups: bool = True,
                                      group_overlap_threshold: float = 0.8
                                      ):
        if max_obsdate is None:
            max_obsdate = target_img.obsdate
            print(f"No max_obsdate provided, using target image's obsdate instead ({max_obsdate}).")

        # Define fallback attempts
        attempt_configs = [
            {'seeing': target_img.seeing, 'depth': target_img.depth},
            {'seeing': target_img.seeing + 2, 'depth': target_img.depth},
            {'seeing': target_img.seeing + 2, 'depth': target_img.depth - 2},
        ]

        reference_frames = None
        for cfg in attempt_configs:
            reference_frames = self.get_referenceframe(
                observatory = target_img.observatory,
                telkey = target_img.telkey,
                filter_ = target_img.filter,
                ra = target_img.ra,
                dec = target_img.dec,
                ra_fov = target_img.fovx,
                dec_fov = target_img.fovy,
                telname = None,
                max_obsdate = max_obsdate,
                seeing_limit = cfg['seeing'],
                depth_limit = cfg['depth'],
                return_groups = return_groups,
                group_overlap_threshold = group_overlap_threshold,
            )
            if reference_frames is not None and len(reference_frames) > 0:
                break  # success
            
        if reference_frames is None:
            print("No reference frames found for the target image.")
            return None

        # Normalize to list
        if isinstance(sort_key, str):
            sort_key = [sort_key]

        # Determine sort direction for each key
        # Use descending (reverse=True) for 'depth' and 'fraction' by default
        reverse_map = {
            'depth': True,
            'fraction': True,
            'seeing': False,
            'obsdate': False,
        }
        reverse_flags = [reverse_map.get(k, False) for k in sort_key]

        # Apply sort
        reference_frames.sort(sort_key, reverse=reverse_flags)

        reference_img = ReferenceImage(reference_frames[0]['file'], telinfo=target_img.telinfo, load=True)
        return reference_img, reference_frames
    
    def get_referenceframe(self,
                           observatory: str,
                           telkey: str,
                           filter_: str,
                           ra: float,
                           dec: float,
                           ra_fov: float = 1.35,
                           dec_fov: float = 0.9,
                           telname: str = None,
                           max_obsdate: str = None,
                           seeing_limit: float = None,
                           depth_limit: float = None,
                           return_groups: bool = True,
                           group_overlap_threshold: float = 0.8
                           ):
        
        """
        observatory: str = '7DT'
        telkey: str = '7DT_C361K_HIGH_1x1'
        filter_: str = 'm425'
        telname: str = None
        max_obsdate: str = None
        seeing_limit: float = None
        depth_limit: float = None
        """
        
        # Load summary tables
        all_referenceframe_info = {}
        referenceframe_summary_path = Path(self.config['REFDATA_DIR']) / 'summary.ascii_fixed_width'

        if referenceframe_summary_path.exists():
            tbl = ascii.read(referenceframe_summary_path, format='fixed_width')
            all_referenceframe_tbl = tbl
        else:
            all_referenceframe_tbl = Table()
                
        if len(all_referenceframe_tbl) == 0:
            raise FileNotFoundError("No calibration frame metadata found.")

        # Basic filtering
        mask = np.ones(len(all_referenceframe_tbl), dtype=bool)

        # Apply filters only if not None
        mask &= all_referenceframe_tbl['observatory'] == observatory
        mask &= all_referenceframe_tbl['telkey'] == telkey
        mask &= all_referenceframe_tbl['filtername'] == filter_
        if telname is not None:
            mask &= all_referenceframe_tbl['telname'] == telname                
        if max_obsdate is not None:
            obsdate_target = self.flexible_time_parser(max_obsdate)
            obs_times = Time(all_referenceframe_tbl['obsdate'], format='isot', scale='utc')
            mask &= obs_times < obsdate_target
        if seeing_limit is not None:
            mask &= all_referenceframe_tbl['seeing'] <= seeing_limit
        if depth_limit is not None:
            mask &= all_referenceframe_tbl['depth'] >= depth_limit
        
        filtered_tbl = all_referenceframe_tbl[mask]
        
        if len(filtered_tbl) == 0:
            try:
                print(f"No reference frames matched the filtering criteria. [Depth > %.1f, Seeing < %.1f, Obsdate <= %s]" %(depth_limit, seeing_limit, obsdate_target))
            except:
                print(f"No reference frames matched the filtering criteria.Obsdate <= %s" % max_obsdate)
            return None
        else:
            pass
        
        # Geometry filtering using RA, Dec, FOV
        target_poly = Polygon([
            (ra - ra_fov / 2, dec - dec_fov / 2),
            (ra + ra_fov / 2, dec - dec_fov / 2),
            (ra + ra_fov / 2, dec + dec_fov / 2),
            (ra - ra_fov / 2, dec + dec_fov / 2)
        ])
        target_area = target_poly.area

        # Geometry filtering: keep only intersecting reference frames
        matched_rows = []
        fractions = []

        for row in filtered_tbl:
            ref_poly = Polygon([
                (row['ra'] - row['fov_ra'] / 2, row['dec'] - row['fov_dec'] / 2),
                (row['ra'] + row['fov_ra'] / 2, row['dec'] - row['fov_dec'] / 2),
                (row['ra'] + row['fov_ra'] / 2, row['dec'] + row['fov_dec'] / 2),
                (row['ra'] - row['fov_ra'] / 2, row['dec'] + row['fov_dec'] / 2)
            ])

            if ref_poly.intersects(target_poly):
                matched_rows.append(row)
                inter_area = target_poly.intersection(ref_poly).area
                frac = inter_area / target_area if target_area > 0 else 0.0
                fractions.append(frac)

        if len(matched_rows) == 0:
            print(f"No reference frames found overlapping RA={ra}, Dec={dec} with FOV=({ra_fov}, {dec_fov})")
            return None

        # Build final table with overlap fractions
        ref_table = Table(rows=matched_rows, names=filtered_tbl.colnames)
        ref_table['fraction'] = fractions

        # Optional: assign overlap-based group IDs
        if return_groups:
            def assign_groups(ref_table: Table, overlap_threshold: float = 0.8) -> Table:
                n = len(ref_table)
                polygons = []
                for row in ref_table:
                    poly = Polygon([
                        (row['ra'] - row['fov_ra'] / 2, row['dec'] - row['fov_dec'] / 2),
                        (row['ra'] + row['fov_ra'] / 2, row['dec'] - row['fov_dec'] / 2),
                        (row['ra'] + row['fov_ra'] / 2, row['dec'] + row['fov_dec'] / 2),
                        (row['ra'] - row['fov_ra'] / 2, row['dec'] + row['fov_dec'] / 2),
                    ])
                    polygons.append(poly)

                adjacency = [set() for _ in range(n)]
                for i in range(n):
                    for j in range(i + 1, n):
                        inter_area = polygons[i].intersection(polygons[j]).area
                        min_area = min(polygons[i].area, polygons[j].area)
                        if min_area > 0 and (inter_area / min_area) >= overlap_threshold:
                            adjacency[i].add(j)
                            adjacency[j].add(i)

                visited = [False] * n
                group_ids = [-1] * n
                group = 0
                for i in range(n):
                    if not visited[i]:
                        queue = [i]
                        while queue:
                            current = queue.pop()
                            if not visited[current]:
                                visited[current] = True
                                group_ids[current] = group
                                queue.extend(adjacency[current])
                        group += 1

                ref_table['group'] = group_ids
                return ref_table

            ref_table = assign_groups(ref_table, overlap_threshold=group_overlap_threshold)

        return ref_table
    
    def query_referenceframe_from_image(self,
                                        target_img: Union[ScienceImage],
                                        catalog_key: str = 'SkyMapper/SMSS4/g',
                                        ):
        imagequerier = ImageQuerier(catalog_key = catalog_key)
        reference_path = target_img.savedir / f'{target_img.objname}_ref.fits'
        reference_img = imagequerier.query(
            width = int(target_img.naxis1 * 1.2),
            height = int(target_img.naxis2 * 1.2),
            ra = target_img.ra,
            dec = target_img.dec,
            pixelscale = np.mean(target_img.pixelscale),
            telinfo = target_img.telinfo,
            save_path = reference_path,
            objname = target_img.objname,
        )      
        return reference_img
         
    def select_reference_image(self, 
                               target_imglist: Union[List[ScienceImage], List[ReferenceImage]],
                               max_obsdate: Union[Time, str, float] = None,
                               seeing_key: str = 'SEEING',
                               depth_key: str = 'UL5_5',
                               ellipticity_key: str = 'ELLIP',
                               obsdate_key: str = 'DATE-OBS',
                               weight_ellipticity: float = 2.0,
                               weight_seeing: float = 1.0,
                               weight_depth: float = 1.5,
                               max_numbers: int = 1):
        seeinglist = []
        depthlist = []
        ellipticitylist = []
        obsdatelist = []
        for target_img in tqdm(target_imglist, desc = 'Querying reference images...'):
            seeinglist.append(target_img.header.get(seeing_key, None))
            depthlist.append(target_img.header.get(depth_key, None))
            ellipticitylist.append(target_img.header.get(ellipticity_key, None))
            obsdatelist.append(target_img.header.get(obsdate_key, None))
        
        try:
            obsdate_time = Time(obsdatelist)
            max_obs_time = self.flexible_time_parser(max_obsdate) if max_obsdate is not None else Time.now()
        except Exception as e:
            raise ValueError(f"Invalid date format: {e}")          
        # Mask for images before max_obsdate
        valid_obs_mask = obsdate_time < max_obs_time
        
        # Also apply validity mask for seeing, depth, ellipticity
        seeinglist = np.array(seeinglist, dtype=float)
        depthlist = np.array(depthlist, dtype=float)
        ellipticitylist = np.array(ellipticitylist, dtype=float)
        valid_value_mask = (~np.isnan(seeinglist)) & (~np.isnan(depthlist)) & (~np.isnan(ellipticitylist))
        combined_mask = valid_obs_mask & valid_value_mask
          
        # Apply final mask
        ell = np.array(ellipticitylist)[combined_mask]
        see = np.array(seeinglist)[combined_mask]
        dep = np.array(depthlist)[combined_mask]
        filtered_imgs = np.array(target_imglist)[combined_mask]
        filtered_obsd = np.array(obsdate_time)[combined_mask]

        # Normalize
        scaler = MinMaxScaler()
        ell_norm = scaler.fit_transform(ell.reshape(-1, 1)).flatten()
        see_norm = scaler.fit_transform(see.reshape(-1, 1)).flatten()
        dep_norm = scaler.fit_transform(dep.reshape(-1, 1)).flatten()

        # Compute combined score
        # You can adjust weights if needed
        score = (1 - ell_norm) * weight_ellipticity + (1 - see_norm) * weight_seeing + dep_norm * weight_depth

        # Rank and select best
        sorted_idx = np.argsort(score)[::-1]  # descending
        best_images = filtered_imgs[sorted_idx]

        # Top N or just best
        best_image = best_images[0]
        
        # Data for plotting
        x = np.array(seeinglist)
        y = np.array(depthlist)
        c = np.array(ellipticitylist)
        x_valid = x[combined_mask]
        y_valid = y[combined_mask]
        c_valid = c[combined_mask]
        best_idx = sorted_idx[0]
        best_x = see[best_idx]
        best_y = dep[best_idx]
        best_c = ell[best_idx]
        selected_idx = sorted_idx[:max_numbers]
        selected_x = see[selected_idx]
        selected_y = dep[selected_idx]
        selected_c = ell[selected_idx]
        marker_sizes = np.where(obsdate_time < max_obs_time, 50, 10)
        marker_alphas = np.where(obsdate_time < max_obs_time, 0.8, 0.2)

        # Calculate percentiles (90%, 75%, and 50%)
        p90_x, p75_x, p50_x, p25_x, p10_x = np.percentile(x_valid, [10, 25, 50, 75, 90])
        p90_y, p75_y, p50_y, p25_y, p10_y = np.percentile(y_valid, [90, 75, 50, 25, 10])

        # Calculate the number of images for each percentile
        num_images_p90 = np.sum((x_valid <= p90_x) & (y_valid >= p90_y))  # Number of images below or equal to the 10th percentile
        num_images_p75 = np.sum((x_valid <= p75_x) & (y_valid >= p75_y))  # Number of images below or equal to the 25th percentile
        num_images_p50 = np.sum((x_valid <= p50_x) & (y_valid >= p50_y))  # Number of images below or equal to the 50th percentile
        num_images_p25 = np.sum((x_valid <= p25_x) & (y_valid >= p25_y))  # Number of images below or equal to the 75th percentile

        # Create figure with GridSpec layout
        fig = plt.figure(figsize=(6, 6), dpi=300)
        gs = GridSpec(4, 4, fig)

        # Create scatter plot
        ax_main = fig.add_subplot(gs[1:, :-1])
        sc = ax_main.scatter(x[valid_value_mask], y[valid_value_mask],
                            c=c[valid_value_mask],
                            s=marker_sizes[valid_value_mask],
                            alpha=marker_alphas[valid_value_mask],
                            cmap='viridis', edgecolors='k', linewidths=0.5,
                            label = 'All images')        
        ax_main.scatter(0,0, s = 10, alpha = 0.2, label = 'Out of date range')
        cbar = fig.colorbar(sc, ax=ax_main, pad=0.01)
        cbar.set_label('Ellipticity')
        ax_main.axvline(p90_x, color='r', linestyle='--')
        ax_main.axvline(p75_x, color='b', linestyle='--')
        ax_main.axvline(p50_x, color='g', linestyle='--')
        ax_main.axvline(p25_x, color='k', linestyle='--')
        ax_main.axhline(p90_y, color='r', linestyle='--')
        ax_main.axhline(p75_y, color='b', linestyle='--')
        ax_main.axhline(p50_y, color='g', linestyle='--')
        ax_main.axhline(p25_y, color='k', linestyle='--')
        ax_main.set_xlim(p90_x - 0.5, p10_x + 0.5)
        ax_main.set_ylim(p10_y - 1, p90_y + 1)
        ax_main.set_xlabel('Seeing [arcsec]')
        ax_main.set_ylabel('Depth [AB]')
        ax_main.scatter(selected_x, selected_y, marker='*', s=200, c='red', edgecolors='black', label='Selected')
        ax_main.scatter(best_x, best_y, marker='*', s=200, c='red', edgecolors='black')
        ax_main.text(best_x, best_y + 0.3,
                    f"Best\nSeeing = {best_x:.2f} arcsec\nDepth = {best_y:.2f} AB\nEllipticity = {best_c:.2f}",
                    color='red', fontsize=8, ha='center', va='bottom',
                    bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.3'))
        ax_main.legend(loc='upper right', fontsize=8, frameon=True)


        # Create top histogram
        ax_histx = fig.add_subplot(gs[0, :-1], sharex=ax_main)
        ax_histx.hist(x_valid, bins=30, color='black', edgecolor='black', alpha=0.7)
        ax_histx.spines['top'].set_visible(False)  # Hide top spine
        ax_histx.spines['right'].set_visible(False)  # Hide right spine

        # Create right histogram
        ax_histy = fig.add_subplot(gs[1:, -1], sharey=ax_main)
        ax_histy.hist(y_valid, bins=30, color='black', edgecolor='black', alpha=0.7, orientation='horizontal')
        ax_histy.spines['top'].set_visible(False)  # Hide top spine
        ax_histy.spines['right'].set_visible(False)  # Hide right spine

        # Set limits for histograms to fit within the black box
        ax_histx.set_xlim(ax_main.get_xlim())
        ax_histy.set_ylim(ax_main.get_ylim())

        # Plot vertical regions for percentiles in histograms
        ax_histx.axvline(p90_x, color='r', linestyle='--', label='90%')
        ax_histx.axvline(p75_x, color='b', linestyle='--', label='75%')
        ax_histx.axvline(p50_x, color='g', linestyle='--', label='50%')
        ax_histx.axvline(p25_x, color='k', linestyle='--', label='25%')

        ax_histy.axhline(p90_y, color='r', linestyle='--', label='90%')
        ax_histy.axhline(p75_y, color='b', linestyle='--', label='75%')
        ax_histy.axhline(p50_y, color='g', linestyle='--', label='50%')
        ax_histy.axhline(p25_y, color='k', linestyle='--', label='25%')

        # Add text annotation in the upper right region of the scatter plot
        text = f'Percentile (# of images, Seeing, Depth):\n'
        text += f'90% ({num_images_p90}, {p90_x:.2f}, {p90_y:.2f})\n'
        text += f'75% ({num_images_p75}, {p75_x:.2f}, {p75_y:.2f})\n'
        text += f'50% ({num_images_p50}, {p50_x:.2f}, {p50_y:.2f})\n'
        text += f'25% ({num_images_p25}, {p25_x:.2f}, {p25_y:.2f})'
        ax_main.text(0.5, 0.15, text,
                    ha='center', va='center',
                    transform=ax_main.transAxes,
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
        
        from matplotlib.lines import Line2D

        dashed_lines = [
            Line2D([0], [0], color='red', linestyle='--', label='90%'),
            Line2D([0], [0], color='blue', linestyle='--', label='75%'),
            Line2D([0], [0], color='green', linestyle='--', label='50%'),
            Line2D([0], [0], color='black', linestyle='--', label='25%')
        ]

        fig.legend(handles=dashed_lines,
                loc='upper right',
                bbox_to_anchor=(0.95, 0.95),
                fontsize=10, frameon=True)
        plt.tight_layout()
        plt.show()
        
        selected_images = []
        for img in tqdm(best_images[:max_numbers], desc='Loading selected images...'):
            ref_img = ReferenceImage(img.path, img.telinfo, load=True)
            ref_img.load()
            ref_img.path = ref_img.savepath.savepath
            selected_images.append(ref_img)
        return selected_images

    def _reproject_to_target(self,
                             reference_img: ReferenceImage,
                             target_img: ScienceImage,
                             save: bool = True,
                             verbose: bool = True):
        """
        Reproject the reference image to the target image's central WCS
        """
        reference_img.savedir = target_img.savepath.savedir
        center_target = target_img.center
        reprojected_reference, _, reprojected_reference_ivpmask = self.projection.reproject(
            target_img = reference_img,
            swarp_params = None,
            resample_type = 'LANCZOS3',
            center_ra = center_target['ra'],
            center_dec = center_target['dec'],
            x_size = target_img.naxis1,
            y_size = target_img.naxis2,
            pixelscale = target_img.pixelscale.mean(),
            verbose = verbose,
            overwrite = False,
            save = save,
            return_ivpmask = True
        )
        
        if not self.is_wcs_equal(reprojected_reference.wcs, target_img.wcs):
            self.print(f"Warning: target_img is not reprojected (not aligned to the North). Run TIPProjection.reproject", verbose)
        return reprojected_reference, reprojected_reference_ivpmask

    def _prepare_subtract_region(self,
                                 target_img: ScienceImage,
                                 reference_img: ReferenceImage,
                                 target_ivpmask: Mask = None,
                                 reference_ivpmask: Mask = None,
                                 target_stamp: str = None,
                                 id_: int = 0,
                                 save: bool = True,
                                 verbose: bool = True,
                                 visualize: bool = True):
        """
        target_img: should be reprojected
        reference_img: should be reprojected
        So, both images should have the same WCS.
        target_ivpmask: Mask for the target image, if None, will be created
        reference_ivpmask: Mask for the reference image, if None, will be created
        This will create the subtracted region by trimming both images to the overlapping region.
        fullframe (of target_img) subtraction region also will be returned 
        """
        
        # If wcs is not equal, reproject target to reference
        if not self.is_wcs_equal(reference_img.wcs, target_img.wcs):
            raise RuntimeError("Target and reference images have different WCS. Please reproject the target image to match the reference image WCS.")
        else:
            if target_ivpmask is None:
                target_ivpmask = self.masking.mask_invalidpixel(
                    target_img = target_img,
                    target_mask = None,
                    save = False,
                    verbose = verbose,
                    visualize = False,
                    save_fig = False
                )
            if reference_ivpmask is None:
                reference_ivpmask = self.masking.mask_invalidpixel(
                    target_img = reference_img,
                    target_mask = None,
                    save = False,
                    verbose = verbose,
                    visualize = False,
                    save_fig = False
                )
            
        # All data and mask are ready
        # Create a new mask that combines the invalid pixel masks
        fullframe_subtract_mask = target_ivpmask.copy()
        fullframe_subtract_mask.masktype = 'subtraction'
        fullframe_subtract_mask.path = target_img.savepath.submaskpath
        fullframe_subtract_mask.combine_mask(reference_ivpmask.data, operation='or')
        fullframe_subtract_mask.data = 1 - fullframe_subtract_mask.data
        
        valid_mask = fullframe_subtract_mask.data == 1
        y_valid, x_valid = np.where(valid_mask)
        y_min, y_max = np.min(y_valid), np.max(y_valid) + 1
        x_min, x_max = np.min(x_valid), np.max(x_valid) + 1
        shape = (y_max - y_min, x_max - x_min)
        position = (x_min + shape[1] // 2, y_min + shape[0] // 2)  # (x_center, y_center)
        
        cutout_target = Cutout2D(data=target_img.data, position=position, size=shape, wcs=target_img.wcs)
        cutout_reference = Cutout2D(data=reference_img.data, position=position, size=shape, wcs=reference_img.wcs)

        # Update image and WCS
        subframe_target_img = target_img.copy()
        subframe_target_img.path = target_img.savepath.savedir / (target_img.savepath.savepath.stem + f'_subframe_{id_}' + target_img.savepath.savepath.suffix)
        subframe_target_img.data = cutout_target.data
        subframe_target_img.header.update(cutout_target.wcs.to_header())

        subframe_reference_img = reference_img.copy()
        subframe_reference_img.path = subframe_reference_img.savepath.savedir / (subframe_reference_img.savepath.savepath.stem + f'_subframe_{id_}' + subframe_reference_img.savepath.savepath.suffix)
        subframe_reference_img.savedir = subframe_target_img.savedir # Change the savedir to the same as target_img
        subframe_reference_img.data = cutout_reference.data
        subframe_reference_img.header.update(cutout_reference.wcs.to_header())
        
        subframe_target_ivpmask = target_ivpmask.copy()
        subframe_target_ivpmask.path = subframe_target_img.savepath.invalidmaskpath#target_ivpmask.savepath.savedir / (target_ivpmask.savepath.savepath.stem + f'_subframe_{id_}' + target_ivpmask.savepath.savepath.suffix)
        subframe_target_ivpmask.data = target_ivpmask.data[y_min:y_max, x_min:x_max]
        subframe_target_ivpmask.header.update(cutout_target.wcs.to_header())
        
        subframe_reference_ivpmask = reference_ivpmask.copy()
        subframe_reference_ivpmask.path = subframe_reference_img.savepath.invalidmaskpath#reference_ivpmask.savepath.savedir / (reference_ivpmask.savepath.savepath.stem + f'_subframe_{id_}' + reference_ivpmask.savepath.savepath.suffix)
        subframe_reference_ivpmask.data = reference_ivpmask.data[y_min:y_max, x_min:x_max]
        subframe_reference_ivpmask.header.update(cutout_reference.wcs.to_header())
        
        subframe_subtract_mask = subframe_target_ivpmask.copy()
        subframe_subtract_mask.masktype = 'subtraction'
        subframe_subtract_mask.path = 'None'
        subframe_subtract_mask.combine_mask(subframe_reference_ivpmask.data, operation='or')
        
        # If target_stamp is provided, trim the subframe_target_img to the target_stamp
        subframe_target_stamp_path = None
        if target_stamp is not None:
            target_stamp = Path(target_stamp)
            if not target_stamp.exists():
                self.print(f"Target stamp file {target_stamp} does not exist.", verbose)
            else:
                stamp_tbl = Table.read(target_stamp, format='ascii')
                x_key = stamp_tbl.colnames[0]
                y_key = stamp_tbl.colnames[1]
                x_full = np.array(stamp_tbl[x_key])
                y_full = np.array(stamp_tbl[y_key])

                # Filter for sources within the cutout region
                in_cutout = (
                    (x_full >= x_min) & (x_full < x_max) &
                    (y_full >= y_min) & (y_full < y_max)
                )

                # Shift coordinates to subframe system
                x_sub = x_full[in_cutout] - x_min
                y_sub = y_full[in_cutout] - y_min
                
                subframe_target_stamp = Table()
                subframe_target_stamp[x_key] = x_sub
                subframe_target_stamp[y_key] = y_sub
                subframe_target_stamp_path = target_stamp.parent / (target_stamp.stem + f'_subframe_{id_}' + target_stamp.suffix)
                subframe_target_stamp.write(subframe_target_stamp_path, format='ascii', overwrite=True)
                # self.to_regions(reg_x = x_sub,
                #                 reg_y = y_sub,
                #                 reg_size = 10,
                #                 unit = 'pixel',
                #                 output_file_path = str(subframe_target_stamp_path) +'.reg')
                
        if visualize:
            subframe_target_img.show()
            subframe_reference_img.show()
        
        if save:
            subframe_target_img.write()
            subframe_reference_img.write()
            subframe_target_ivpmask.write() 
            subframe_reference_ivpmask.write()
        
        return subframe_target_img, subframe_reference_img, subframe_target_ivpmask, subframe_reference_ivpmask, fullframe_subtract_mask, subframe_subtract_mask, subframe_target_stamp_path

    def subtract(self,
                 target_img: ScienceImage,
                 reference_img: ReferenceImage,
                 target_ivpmask: Mask = None,
                 reference_ivpmask: Mask = None,
                 target_stamp: str = None,
                 id_ : int = 0,
                 save: bool = True,
                 verbose: bool = True,
                 visualize: bool = True,
                 convim: str = None,
                 normim: str = 'i',
                 nrx: int = 3,
                 nry: int = 2,
                 iu: float = 60000,
                 il: float = -10000,
                 tu: float = 60000,
                 tl: float = -10000,
                 **hotpants_params):
        """
        This will subtract the target_img from reference-img. 
        It will prepare the subtracted region by trimming both images to the overlapping region.
        Then, run HOTPANTS subtraction.
        """
        # Set the target_stamp 
        if target_stamp is not None:
            self.print(f"Using target stamp from {target_stamp}", verbose)
                        
        target_seeing = target_img.seeing
        reference_seeing = reference_img.seeing
        if convim is None:
            if target_seeing is None or reference_seeing is None:
                self.print("Warning: One of the images does not have a valid seeing value. Using 't' for convolution.", verbose)
                convim = 't'
            elif target_seeing > reference_seeing:
                convim = 't'
            else:
                convim = 'i'
        
        # Trim images
        subframe_target_img, subframe_reference_img, subframe_target_ivpmask, subframe_reference_ivpmask, fullframe_subtract_mask, subframe_subtract_mask, subframe_target_stamp = self._prepare_subtract_region(
            target_img = target_img,
            reference_img = reference_img,
            target_ivpmask = target_ivpmask,
            reference_ivpmask = reference_ivpmask,
            target_stamp = target_stamp,
            id_ = id_,
            save = True,
            verbose = verbose,
            visualize = visualize
        )
            
        # Run HOTPANTS subtraction
        result = self.run_hotpants(
            target_path = subframe_target_img.savepath.savepath,
            reference_path = subframe_reference_img.savepath.savepath,
            convolve_path = subframe_target_img.savepath.convolvepath,
            target_mask = subframe_target_ivpmask.savepath.savepath,
            reference_mask = subframe_reference_ivpmask.savepath.savepath,
            stamp = subframe_target_stamp,
            target_outpath = subframe_target_img.savepath.subtractpath,
            convim = convim,
            normim = normim,
            nrx = nrx,
            nry = nry,
            iu = iu,
            il = il,
            tu = tu,
            tl = tl,
            **hotpants_params
        )            
        
        subframe_convolve_img = type(subframe_target_img)(subframe_target_img.savepath.convolvepath, telinfo = subframe_target_img.telinfo, load = True)
        #subframe_convolve_img.remove()
        subframe_subtract_img = type(target_img)(result, telinfo = target_img.telinfo, load = True)
        subframe_subtract_mask.path = subframe_subtract_img.savepath.invalidmaskpath

        if save:
            subframe_subtract_img.write()
            fullframe_subtract_mask.write()
            subframe_subtract_mask.write()
        else:
            subframe_subtract_img.data
            subframe_subtract_img.remove()
            
        if visualize:
            subframe_subtract_img.show()
        return subframe_target_img, subframe_reference_img, subframe_subtract_img, fullframe_subtract_mask, subframe_subtract_mask
    
    def find_transients(self,
                        target_img: ScienceImage,
                        reference_imglist: List[ReferenceImage],
                        target_bkg: Background = None,
                        detection_sigma: float = 5,
                        
                        target_transient_number: int = 5,
                        reject_variable_sources: bool = False,
                        negative_detection: bool = True,
                        reverse_subtraction: bool = False,
                        
                        save: bool = True,
                        verbose: bool = True,
                        visualize: bool = False,
                        show_transient_numbers: int = 10):
        """
        Find transients in the subtracted image.
        This function uses the subtracted image from the subtract method.
        CALCULATE THE REFERENCE IMAGE INVALIDMASK & TARGE INVALIDMASK
        CALCULATE THE DUPLICATED REGION & TRIM THE TARGE IMAGE TO THE DUPLICATED REGION
        HOTPANTS SUBTRACTION 
        AFTER SUBTRACTION, PERFORM APERTURE PHOTOMETRY ON THE SUBTRACTED IMAGE
        SELECTED SOURCES WILL BE DETECTED SOURCES FROM MANY SUBTRACTED IMAGES
        AFTER SUBTRCATION, COMBINE IMAGES?
        reference_imglist = [reference_img]
        target_bkg: Background = None
        detection_sigma: float = 5
        
        target_transient_number: int = 5
        reject_variable_sources: bool = True
        negative_detection: bool = True
        reverse_subtraction: bool = False
        
        save: bool = True
        verbose: bool = True
        visualize: bool = False
        show_transient_numbers: int = 10
        
        """

        # Prepare reprojected target_img
        if target_bkg is not None:
            target_img_sub = self.background.subtract_background(
                target_img = target_img,
                target_bkg = target_bkg,
                save = save,
                overwrite = False,
                visualize = visualize,
                save_fig = False)
        else:
            target_img_sub = target_img.copy()
        
        # Free memory
        target_img.clear()
        if target_bkg is not None:
            target_bkg.clear()
                
        self.print("================== Target Image Preparation ==================", verbose)
        center_target = target_img.center
        reprojected_target_img, _, reprojected_target_ivpmask = self.projection.reproject(
            target_img = target_img_sub,
            swarp_params = None,
            resample_type = 'LANCZOS3',
            center_ra = center_target['ra'],
            center_dec = center_target['dec'],
            x_size = target_img_sub.naxis1,
            y_size = target_img_sub.naxis2,
            pixelscale = target_img_sub.pixelscale.mean(),
            verbose = verbose,
            overwrite = False,
            save = save,
            return_ivpmask = True
        )
        # Free memory
        target_img_sub.clear()
        
        # Calculate typical detection criteria for transients
        reprojected_target_catalog = self.aperphot.sex_photometry(
            target_img = reprojected_target_img,
            target_bkg = None,  # No background for subtraction
            target_bkgrms = None,  # No background RMS for subtraction
            target_mask = None,
            detection_sigma = detection_sigma,
            aperture_diameter_arcsec = [5,7,10],
            saturation_level = 60000,
            kron_factor = 2.5,
            save = save,
            verbose = verbose,
            visualize = visualize,
            save_fig = False
        )
        
        reprojected_target_img, reprojected_target_catalog, reprojected_target_ref_catalog = self.photcal.photometric_calibration(
            target_img = reprojected_target_img,
            target_catalog = reprojected_target_catalog,
            catalog_type = 'GAIAXP',
            max_distance_second = 1.0,
            calculate_color_terms = False,
            calculate_mag_terms = False,
            snr_lower = 15,
            snr_upper = 500,
            save = True,
            verbose = verbose,
            visualize = visualize,
            save_fig = False,
            save_starcat = False,
            save_refcat = True
            )
        
        reprojected_target_img.remove(remove_main = False, remove_connected_files = True, skip_exts = ['.refcat', '.invalidmask'])
        
        transient_criteria = dict()
        reprojected_target_ref_catalog_data = reprojected_target_ref_catalog.data
        saturation_level = np.percentile(reprojected_target_ref_catalog_data['FLUX_MAX'], 99.9)
        transient_criteria['flux_upper_target'] = 60000#min(60000, saturation_level)
        transient_criteria['flux_lower_target'] = reprojected_target_img.info.SKYVAL - 15 * reprojected_target_img.info.SKYSIG if reprojected_target_img.info.SKYVAL is not None and reprojected_target_img.info.SKYSIG is not None else -10000
        transient_criteria['classstar_lower'] = min(0.5, 0.9 * np.median(reprojected_target_ref_catalog_data['CLASS_STAR']))
        transient_criteria['elongation_upper'] = max(1.5, 1.2 * np.median(reprojected_target_ref_catalog_data['ELONGATION']))
        is_criteria_determined = True
        reprojected_target_stamp = reprojected_target_ref_catalog.to_stamp(reprojected_target_img)

        self.print("=============== Target Image Preparation END ===============", verbose)
        
        self.print(f"=============== Subtraction with {len(reference_imglist)} images ===============", verbose)
        final_subtraction_mask = Mask(path = target_img.savepath.submaskpath, masktype = 'subtraction', load = False)
        final_subtraction_mask.data = np.zeros_like(reprojected_target_img.data, dtype=bool)
        
        all_catalogs = []
        transient_catalogs = []
        candidate_catalogs = []
        for i, reference_img in tqdm(enumerate(reference_imglist), total=len(reference_imglist), desc="Subtraction Progress"):
            reference_img_temp = reference_img.copy()
            reference_img_temp.path = reference_img.savedir / (uuid.uuid4().hex + '.fits')
            reference_img_temp.write()
            
            # Step 1: If reference_img.seeing is None, update seeing
            reference_catalog = None
            reference_ref_catalog = None
            if reference_img.seeing is None or reject_variable_sources:
                # If reference catalog is not available, run SExtractor
                if reference_img.savepath.catalogpath.exists():
                    reference_catalog = TIPCatalog(reference_img.savepath.catalogpath, catalog_type = 'all', load= True)
                    reference_catalog.load_target_img(reference_img)
                else:
                    reference_catalog = self.aperphot.sex_photometry(
                        target_img = reference_img_temp,
                        target_bkg = None,  # No background for subtraction
                        target_bkgrms = None,  # No background RMS for subtraction
                        target_mask = None,
                        detection_sigma = detection_sigma,
                        aperture_diameter_arcsec = [5,7,10],
                        saturation_level = 60000,
                        kron_factor = 2.5,
                        save = save,
                        verbose = verbose,
                        visualize = visualize,
                        save_fig = False)
                
                try:
                    reference_img_temp, reference_catalog, reference_ref_catalog = self.photcal.photometric_calibration(
                        target_img = reference_img_temp,
                        target_catalog = reference_catalog,
                        catalog_type = 'GAIAXP',
                        max_distance_second = 1.0,
                        calculate_color_terms = False,
                        calculate_mag_terms = False,
                        snr_lower = 0.0,
                        snr_upper = 500,
                        save = True,
                        verbose = verbose,
                        visualize = visualize,
                        save_fig = True,
                        save_starcat = False,
                        save_refcat = True
                        )         
                except:
                    reference_ref_catalog, status, peeing = self.photcal.select_stars(
                     target_catalog = reference_catalog   
                    )
                    reference_img_temp.header['SEEING'] = peeing * np.mean(reference_img_temp.pixelscale)

            # Step 2: Reproject reference images to reprojected_target_img
            reprojected_reference_img, reprojected_reference_ivpmask = self._reproject_to_target(
                reference_img = reference_img_temp,
                target_img = reprojected_target_img,
                save = save,
                verbose = verbose
            )
            reference_img_temp.clear()
            reference_img_temp.remove(remove_main = True, remove_connected_files = True, skip_exts = [''])
            reference_img.clear()
            reprojected_reference_img.remove(remove_main = False, remove_connected_files = True, skip_exts = ['.invalidmask'])
            
            if np.sum(reprojected_reference_ivpmask.data == 0) < 20000:
                self.print('Reference image and target image are not overlapping enough for subtraction.', verbose)
                failed_reference_images.append(i)
                continue
            
            # Step 3. Update transient criteria based on the seeing of reprojected images
            target_seeing = reprojected_target_img.seeing
            reference_seeing = reprojected_reference_img.seeing
            # If target_seeing larger, convolve to target_img, thus the subtracted image's seeing should be target_seeing 
            if target_seeing >= reference_seeing:
                subtract_seeing = target_seeing
                sigma_match = np.sqrt(target_seeing**2 - reference_seeing**2)
            else:
                subtract_seeing = reference_seeing
                sigma_match = np.sqrt(reference_seeing**2 - target_seeing**2)
            transient_criteria['seeing_upper'] = 1.3 * subtract_seeing
            transient_criteria['seeing_lower'] = max(1.0, 0.7 * subtract_seeing)    
            # Set saturation level
            if reference_ref_catalog is not None:
                if len(reference_ref_catalog.data) > 0:
                    saturation_level = np.percentile(reference_ref_catalog.data['FLUX_MAX'], 99.9)
                    transient_criteria['flux_upper_reference'] = 60000#min(saturation_level, 60000)
                else:
                    transient_criteria['flux_upper_reference'] = 60000
            else:
                transient_criteria['flux_upper_reference'] = 60000
            transient_criteria['flux_lower_reference'] = reprojected_reference_img.info.SKYVAL - 15 * reprojected_reference_img.info.SKYSIG if reprojected_reference_img.info.SKYVAL is not None and reprojected_reference_img.info.SKYSIG is not None else -10000
            
            #ng = f"3 6 %.2f 4 %.2f 2 %.2f" %(0.5 * sigma_match, sigma_match, 2.0 * sigma_match)
            
            # Step 4: Subtration
            subframe_target_imglist = []
            subframe_reference_imglist = []
            subframe_subtract_imglist = []
            #try:
            subframe_target_img, subframe_reference_img, subframe_subtract_img, fullframe_subtract_mask, subframe_subtract_ivpmask = self.subtract(
                target_img = reprojected_target_img,
                reference_img = reprojected_reference_img,
                target_ivpmask = reprojected_target_ivpmask,
                reference_ivpmask = reprojected_reference_ivpmask,
                target_stamp = reprojected_target_stamp,
                id_ = i,
                save = save,
                verbose = verbose,
                visualize = visualize,
                # HOTPANTS Parameters
                convim = None,
                normim = 'i',
                nrx = 1,
                nry = 1,
                iu = transient_criteria['flux_upper_target'],
                il = transient_criteria['flux_lower_target'],
                tu = transient_criteria['flux_upper_reference'],
                tl = transient_criteria['flux_lower_reference'],
                # Other hotpants parameters
                ko = 3,
                bgo = 1,
                nsx = 15,
                nsy = 15,
                r = 10,
                #ng = ng,
            )
            reprojected_reference_img.remove(remove_main = True, remove_connected_files = True)
            subframe_target_img.remove(remove_main = False, remove_connected_files = True)
            subframe_reference_img.remove(remove_main = False, remove_connected_files = True)
            subframe_subtract_img.remove(remove_main = False, remove_connected_files = True)
            final_subtraction_mask.combine_mask(fullframe_subtract_mask.data, operation='add')

            # Step 5: Extract sources
            tbl_first = self.aperphot.sex_photometry(
                target_img = subframe_subtract_img,
                target_bkg = None,  # No background for subtraction
                target_bkgrms = None,  # No background RMS for subtraction
                target_mask = subframe_subtract_ivpmask,
                sex_params = dict(SEEING_FWHM = subtract_seeing),
                detection_sigma = detection_sigma,
                aperture_diameter_arcsec = [5,7,10],
                saturation_level = 60000,
                save = save,
                verbose = verbose,
                visualize = visualize,
                save_fig = False
            )
            self.photcal.apply_zp(
                target_img = subframe_subtract_img,
                target_catalog = tbl_first,
                save = True
            )
        
            # Step 6: Filter the table for significant sources
            selected_tbl_first = self.select_transients(
                target_catalog = tbl_first,
                snr_lower = 5.0,
                fwhm_lower = transient_criteria['seeing_lower'],
                fwhm_upper = transient_criteria['seeing_upper'],
                flag_upper = 1,
                maskflag_upper = 1,
                class_star_lower = transient_criteria['classstar_lower'],
                elongation_upper = transient_criteria['elongation_upper'],
                flux_key = 'FLUX_AUTO',
                fluxerr_key = 'FLUXERR_AUTO',
                fwhm_key = 'FWHM_WORLD',
                flag_key = 'FLAGS',
                maskflag_key = 'IMAFLAGS_ISO',
                classstar_key = 'CLASS_STAR',
                elongation_key = 'ELONGATION',
                verbose = verbose,
                save = save,
                return_only_transient = True,
            )
            
            all_tbl = tbl_first.copy()  
            candidate_tbl = selected_tbl_first.copy()

            subframe_subtract_img.remove(remove_main = False, remove_connected_files = True, skip_exts = ['.invalidmask', '.transient', '.candidate', '.cat'])

            # Step 7: negative image & Photometry
            if negative_detection:
                negative_subframe_subtract_img = subframe_subtract_img.copy()
                negative_subframe_subtract_img.path = subframe_subtract_img.savepath.savedir / ('inv_' + subframe_subtract_img.savepath.savepath.name)
                negative_subframe_subtract_img.data = -subframe_subtract_img.data        

                tbl_second = self.aperphot.sex_photometry(
                    target_img = negative_subframe_subtract_img,
                    target_bkg = None,  # No background for subtraction
                    target_bkgrms = None,  # No background RMS for subtraction
                    target_mask = subframe_subtract_ivpmask,
                    sex_params = dict(SEEING_FWHM = subtract_seeing),
                    detection_sigma = detection_sigma,
                    aperture_diameter_arcsec = [5,7,10],
                    saturation_level = 60000,
                    kron_factor = 2.5,
                    save = False,
                    verbose = True,
                    visualize = visualize,
                    save_fig = False
                )

                # Remove
                negative_subframe_subtract_img.remove()
                
                # Only keep the unmatched sources from the first photometry step
                if len(tbl_second.data) > 0:
                    coord_first = SkyCoord(ra=candidate_tbl.data['X_WORLD'],
                                        dec=candidate_tbl.data['Y_WORLD'],
                                        unit = 'deg')
                    coord_second = SkyCoord(ra=tbl_second.data['X_WORLD'],
                                            dec=tbl_second.data['Y_WORLD'],
                                            unit = 'deg')
                    matched_first, matched_second, unmatched_first = self.cross_match(coord_first, coord_second, subtract_seeing)
                    candidate_tbl.data = candidate_tbl.data[unmatched_first]
                    self.print(f"Found {len(candidate_tbl.data)} transients after negative detection.", verbose)
                else:
                    self.print("No significant sources found in the second photometry step.", verbose)
                    pass
                
            # Match the first photometry results with the reference catalog to reject variable sources
            transient_tbl = None
            if reject_variable_sources:
                transient_tbl = candidate_tbl.copy()
                coord_first = SkyCoord(ra=transient_tbl.data['X_WORLD'],
                                    dec=transient_tbl.data['Y_WORLD'],
                                    unit = 'deg')
                coord_second = SkyCoord(ra=reference_catalog.data['X_WORLD'],
                                        dec=reference_catalog.data['Y_WORLD'],
                                        unit = 'deg')
                matched_first, matched_second, unmatched_first = self.cross_match(coord_first, coord_second, subtract_seeing)
                transient_tbl.data = transient_tbl.data[unmatched_first]
                
                

            # # Step 8: reverse subtraction (reference_img - target_img)
            # if reverse_subtraction:
            #     _, _, reverse_subframe_subtract_img, _, reverse_subframe_subtract_ivpmask = self.subtract(
            #         target_img = reprojected_reference_img,
            #         reference_img = reprojected_target_img,
            #         target_ivpmask = reprojected_reference_ivpmask,
            #         reference_ivpmask = reprojected_target_ivpmask,
            #         target_stamp = reprojected_target_stamp,
            #         id_ = i,
            #         save = save,
            #         verbose = verbose,
            #         visualize = visualize,
            #         nrx = 2,
            #         nry = 2,
            #         ko = 3,
            #         bgo = 3,
            #         nsx = 10,
            #         nsy = 10,
            #         r = 10,
            #         #ng = ng,
            #     )
            #     subtract_seeing = max(target_seeing, reference_seeing)
            #     tbl_third = self.aperphot.sex_photometry(
            #         target_img = reverse_subframe_subtract_img,
            #         target_bkg = None,  # No background for subtraction
            #         target_bkgrms = None,  # No background RMS for subtraction
            #         target_mask = reverse_subframe_subtract_ivpmask,
            #         sex_params = dict(SEEING_FWHM = subtract_seeing, BACK_TYPE = 'MANUAL'),
            #         detection_sigma = detection_sigma,
            #         aperture_diameter_arcsec = [6,9,12],
            #         saturation_level = 60000,
            #         save = save,
            #         verbose = verbose,
            #         visualize = visualize,
            #         save_fig = False
            #     )
            
            #     # Step 6: Filter the table for significant sources
            #     selected_tbl_third = self.select_transients(
            #         target_catalog = tbl_third,
            #         snr_lower = 3.0,
            #         fwhm_lower = transient_criteria['seeing_lower'],
            #         fwhm_upper = transient_criteria['seeing_upper'],
            #         flag_upper = 1,
            #         maskflag_upper = 1,
            #         class_star_lower = transient_criteria['classstar_lower'],
            #         elongation_upper = transient_criteria['elongation_upper'],
            #         flux_key = 'FLUX_AUTO',
            #         fluxerr_key = 'FLUXERR_AUTO',
            #         fwhm_key = 'FWHM_WORLD',
            #         flag_key = 'FLAGS',
            #         maskflag_key = 'IMAFLAGS_ISO',
            #         classstar_key = 'CLASS_STAR',
            #         elongation_key = 'ELONGATION',
            #         verbose = verbose,
            #         save = save,
            #         return_only_transient = True
            #     )

            #     # Remove 
            #     reverse_subframe_subtract_img.remove()
            #     reverse_subframe_subtract_ivpmask.remove()
                
            #     # Match the first and third photometry results
            #     if len(selected_tbl_third.data) > 0:
            #         coord_first = SkyCoord(ra=selected_tbl_first.data['X_WORLD'],
            #                             dec=selected_tbl_first.data['Y_WORLD'],
            #                             unit = 'deg')
            #         coord_third = SkyCoord(ra=selected_tbl_third.data['X_WORLD'],
            #                                 dec=selected_tbl_third.data['Y_WORLD'],
            #                                 unit = 'deg')
            #         matched_first, matched_third, unmatched_first = self.cross_match(coord_first, coord_third, seeing_upper_criteria * 2)
            #         transient_tbl.data = transient_tbl.data[unmatched_first]
            #     else:
            #         self.print("No significant sources found in the third photometry step.", verbose)
            #         pass
            
            #     if len(transient_tbl.data) <= target_transient_number:
            #         transient_tbl = self.photcal.apply_zp(
            #             target_img = subframe_subtract_img,
            #             target_catalog = transient_tbl,
            #             save = True
            #         )
                
            candidate_tbl = self.photcal.apply_zp(
                target_img = subframe_subtract_img,
                target_catalog = candidate_tbl,
                save = True
            )   
            transient_tbl = self.photcal.apply_zp(
                target_img = subframe_subtract_img,
                target_catalog = transient_tbl,
                save = True
            )
            
            if visualize:
                if transient_tbl is not None:
                    if len(transient_tbl.data) > 0:
                        for transient in transient_tbl.data[:show_transient_numbers]:
                            ra = transient['X_WORLD']
                            dec = transient['Y_WORLD']
                            idx = transient['NUMBER']
                            self.show_transient_positions(
                                science_img = subframe_target_img,
                                reference_img = subframe_reference_img,
                                subtracted_img = subframe_subtract_img,
                                x = ra,
                                y = dec,
                                coord_type = 'coord',
                                zoom_radius_pixel = 100,
                                downsample = 1,
                                cmap = 'gray',
                                scale = 'zscale',
                                figsize = (15, 5),
                                title = f'Transient at RA: {ra}, Dec: {dec}, Idx: {idx}',
                                subtitles = [f'Science', f'Reference', f'Subtracted']
                            )
                else:
                    if len(candidate_tbl.data) > 0:
                        for transient in candidate_tbl.data[:show_transient_numbers]:
                            ra = transient['X_WORLD']
                            dec = transient['Y_WORLD']
                            idx = transient['NUMBER']
                            self.show_transient_positions(
                                science_img = subframe_target_img,
                                reference_img = subframe_reference_img,
                                subtracted_img = subframe_subtract_img,
                                x = ra,
                                y = dec,
                                coord_type = 'coord',
                                zoom_radius_pixel = 100,
                                downsample = 1,
                                cmap = 'gray',
                                scale = 'zscale',
                                figsize = (15, 5),
                                title = f'Transient at RA: {ra}, Dec: {dec}, Idx: {idx}',
                                subtitles = [f'Science', f'Reference', f'Subtracted']
                            )
                            
            final_subtraction_mask.write()
            all_catalogs.append(all_tbl)
            candidate_catalogs.append(candidate_tbl)
            transient_catalogs.append(transient_tbl)
            subframe_target_img.clear()
            subframe_reference_img.clear()
            subframe_subtract_img.clear()
            subframe_target_imglist.append(subframe_target_img)
            subframe_reference_imglist.append(subframe_reference_img)
            subframe_subtract_imglist.append(subframe_subtract_img)
            #except Exception as e:
            #    self.print(f"Subtraction failed for reference image {i}: {e}", verbose)
            #    continue
        return all_catalogs, candidate_catalogs, transient_catalogs, subframe_target_imglist, subframe_reference_imglist, subframe_subtract_imglist
        
    def select_transients(self,
                            target_catalog: TIPCatalog,
                            snr_lower: float = 5.0,
                            fwhm_lower: float = 1.5,
                            fwhm_upper: float = 5.0,
                            flag_upper: int = 1,
                            maskflag_upper: int = 1,
                            class_star_lower: float = 0.9,
                            elongation_upper: float = 1.3,
                            flux_key: str = 'FLUX_AUTO',
                            fluxerr_key: str = 'FLUXERR_AUTO',
                            fwhm_key: str = 'FWHM_WORLD',
                            flag_key: str = 'FLAGS',
                            maskflag_key: str = 'NIMAFLAGS_ISO',
                            classstar_key: str = 'CLASS_STAR',
                            elongation_key: str = 'ELONGATION',
                            verbose: bool = True,
                            save: bool = True,
                            return_only_transient: bool = True):
        """
        Select valid sources from the catalog based on SNR, flags, class star, and elongation.
        """
        target_catalog_data = target_catalog.data
        target_catalog_data['SNR'] = target_catalog_data[flux_key] / target_catalog_data[fluxerr_key]
        if fwhm_key.upper() == 'FWHM_WORLD':
            target_catalog_data['SEEING'] = target_catalog_data['FWHM_WORLD'] * 3600
            fwhm_key = 'SEEING'
        
        snr_lower_idx = (target_catalog_data['SNR'] > snr_lower)
        fwhm_lower_idx = (target_catalog_data[fwhm_key] > fwhm_lower)
        fwhm_upper_idx = (target_catalog_data[fwhm_key] < fwhm_upper)
        flag_upper_idx = (target_catalog_data[flag_key] < flag_upper)
        maskflag_upper_idx = (target_catalog_data[maskflag_key] < maskflag_upper)
        classstar_lower_idx = (target_catalog_data[classstar_key] > class_star_lower)
        elongation_upper_idx = (target_catalog_data[elongation_key] < elongation_upper)
        all_idx = snr_lower_idx & fwhm_lower_idx & fwhm_upper_idx & flag_upper_idx & maskflag_upper_idx & classstar_lower_idx & elongation_upper_idx
        
        target_catalog_data['FLAG_SNR'] = snr_lower_idx
        target_catalog_data['FLAG_FWHM'] = fwhm_lower_idx & fwhm_upper_idx
        target_catalog_data['FLAG_MASK'] = maskflag_upper_idx
        target_catalog_data['FLAG_CLASS_STAR'] = classstar_lower_idx
        target_catalog_data['FLAG_ELONGATION'] = elongation_upper_idx
        target_catalog_data['FLAG_Transient'] = all_idx
        # Update the flags
        
        transient_catalog = TIPCatalog(path = target_catalog.savepath.transientcatalogpath, catalog_type = 'transient', load = False)
        transient_catalog.data = target_catalog_data[all_idx]
        transient_catalog.load_target_img()
        candidate_catalog = TIPCatalog(path = target_catalog.savepath.candidatecatalogpath, catalog_type = 'transient', load = False)
        candidate_catalog.data = target_catalog_data
        candidate_catalog.load_target_img()
        
        if verbose:
            print(f"Filtering sources based on criteria:")
            print(f"SNR > {snr_lower}: {np.sum(snr_lower_idx)}")
            print(f"{fwhm_key} > {fwhm_lower} and FWHM < {fwhm_upper}: {np.sum(fwhm_lower_idx & fwhm_upper_idx)}")
            print(f"{flag_key} < {flag_upper}: {np.sum(flag_upper_idx)}")
            print(f"{maskflag_key} < {maskflag_upper}: {np.sum(maskflag_upper_idx)}")
            print(f"{classstar_key} > {class_star_lower}: {np.sum(classstar_lower_idx)}")
            print(f"{elongation_key} < {elongation_upper}: {np.sum(elongation_upper_idx)}")
            print(f'Sources with all criteria met: {np.sum(all_idx)}')

        if save:
            transient_catalog.write()
            candidate_catalog.write()
        
        if return_only_transient:
            return transient_catalog
        else:
            return candidate_catalog
        
    def show_transient_positions(self, 
                                 science_img: Union[ScienceImage, ReferenceImage], 
                                 reference_img: Union[ScienceImage, ReferenceImage], 
                                 subtracted_img: Union[ScienceImage, ReferenceImage],
                                 x: float, y: float, coord_type='coord',
                                 zoom_radius_pixel=100, downsample=1,
                                 cmap='gray', scale='zscale',
                                 figsize=(15, 5), title: str=None, subtitles: list=None):

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        img_list = [science_img, reference_img, subtracted_img]
        subtitles = subtitles or ['Science', 'Reference', 'Subtracted']

        for ax, img, subtitle in zip(axes, img_list, subtitles):
            img.show_position(
                x=x, y=y,
                coord_type=coord_type,
                zoom_radius_pixel=zoom_radius_pixel,
                downsample=downsample,
                cmap=cmap,
                scale=scale,
                ax=ax
            )
            ax.set_title(subtitle)

        if title:
            fig.suptitle(title, fontsize=16)
            fig.subplots_adjust(top=0.88)

        plt.tight_layout()
        plt.show()
        return fig, axes

#%%
if __name__ == "__main__":
    import glob
    self = TIPSubtraction()
    from tippy.imageojbects import *
    from tippy.methods import TIPSubtraction, TIPStacking
    from tippy.utils import *
    from tippy.helper import Helper, TIPDataBrowser
    import numpy as np
    helper = Helper()
    sdt_data_querier = SDTData()
    databrowser = TIPDataBrowser('scidata')
    databrowser.objname = 'NGC1566'
    databrowser.observatory = 'KCT'
    stacked_imglist = databrowser.search(pattern='Calib*120.com.fits', return_type='science')
    
    target_img = stacked_imglist[0]
    reference_img = self.get_referenceframe_from_image(target_img)[0]
    telinfo = target_img.telinfo

    # stacking = TIPStacking()
    # target_imglist = [ScienceImage(img, telinfo = telinfo, load = True) for img in imginfo]
    # target_img = stacking.select_quality_images(target_imglist, max_numbers = 3)[0]
    # reference_img, _ = self.get_referenceframe_from_image(target_img)
#%%
if __name__ == "__main__":
    combine_type = 'weight'
    n_proc = 4
    clip_type = None
    sigma = 3.0
    nlow = 1
    nhigh = 1
    resample = True
    resample_type = 'LANCZOS3'
    center_ra = None
    center_dec = None
    pixel_scale = None
    x_size = None
    y_size = None
    scale = True
    scale_type = 'min'
    zp_key = 'ZP_APER_1'
    convolve = True
    seeing_key = 'SEEING'
    kernel = 'gaussian'
    detection_sigma = 5.0
    aperture_diameter_arcsec = [5,7,10]
    saturation_level = 40000
    kron_factor = 2.5
    catalog_type = 'GAIAXP'
    max_distance_second = 1.0
    calculate_mag_terms = True
    calculate_color_terms = True
    visualize_mag_key = 'MAG_AUTO'
    save = True
    verbose = True
    visualize = True
    save_fig = False
    import numpy as np
    # target_pathlist= glob.glob('/data/data1/factory_hhchoi/data/scidata/7DT/7DT_C361K_HIGH_1x1/T01357/7DT15/r/calib*.fits')
    # target_imglist = [ScienceImage(path, telinfo = self.get_telinfo('7DT', 'C361K', 'HIGH', 1), load = False) for path in target_pathlist]
    # reference_pathlist = glob.glob('/data/data1/factory_hhchoi/data/refdata/main/SkyMapper/SkyMapper_SG_32_Det_1x1/T01357/SkyMapper/g/DONE*fits')
    # reference_imglist = [ReferenceImage(path, telinfo = self.get_telinfo('SkyMapper', 'SG_32_Det'), load = False) for path in reference_pathlist]
    # target_img = target_imglist[0]
    # reference_img = reference_imglist[0]
#%%
if __name__ == "__main__":
    result = self.find_transients(
        target_img = target_img,
        reference_imglist = [reference_img],
        target_bkg = None,
        detection_sigma = 5,
        reject_variable_sources = True,
        negative_detection = True,
        reverse_subtraction = False,
        save = True,
        verbose = True,
        visualize = True,
        show_transient_numbers = 10
        
    )
    #TODO CHECK WHETHER THE MAGSKY IS CORRECT (ZP is not changed during subtraction)
    # %%
#%%
if __name__ == "__main__":

    from tippy.catalog import TIPCatalog
    T = result[2][0]
    ra = 233.857430764
    dec = 12.0577222937
    T.show_source(ra, dec)
    T.search_sources(ra, dec, 'coord')
# %%
