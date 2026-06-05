#%%

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
from astropy.table import Table
from tqdm import tqdm
#%%

class Formatter:

    def __init__(self,
                 image_paths: List[Path],
                 filter_list: List[str],
                 catalog_paths: Optional[List[Path]] = None):

        self.image_paths = [Path(p) for p in image_paths]
        self.filter_list = filter_list

        if len(self.filter_list) != len(self.image_paths):
            raise ValueError("filter_list and image_paths must have same length")

        self.catalog_paths = None
        if catalog_paths is not None:
            if len(catalog_paths) != len(filter_list):
                raise ValueError("catalog_paths and filter_list must have same length")
            self.catalog_paths = [Path(p) for p in catalog_paths]

        self.prepare()

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    def prepare(self):

        # images
        image_paths = []
        filter_list = []
        for filt, img_path in zip(self.filter_list, self.image_paths):

            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_path}")

            # if filt in self.image_dict:
            #     raise ValueError(f"Duplicate filter: {filt}")
            image_paths.append(str(img_path))
            filter_list.append(filt)

        # catalogs (optional)
        catalog_paths = []
        if self.catalog_paths is not None:

            for filt, cat_path in zip(self.filter_list, self.catalog_paths):

                if not cat_path.exists():
                    raise FileNotFoundError(f"Catalog not found: {cat_path}")

                catalog_paths.append(str(cat_path))
        
        self.image_paths = image_paths
        self.filter_list = filter_list
        self.catalog_paths = catalog_paths

    # --------------------------------------------------------
    # 1️⃣ Return images
    # --------------------------------------------------------

    def to_input_images(self, output_path: Path):
        """
        Save image paths to text file in filter_list order.

        Output format:
            one image path per line
        """

        if not self.image_paths:
            raise RuntimeError("image_dict is empty. Populate it before saving.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for image_path in self.image_paths:
                f.write(str(image_path) + "\n")
        
        return str((output_path).resolve())
        
    # --------------------------------------------------------
    # 2️⃣ Match catalogs to targets
    # --------------------------------------------------------
    def to_input_catalogs(self,
                          output_path: Path,
                          list_ra: List[float],
                          list_dec: List[float],
                          list_flux: List[float] = None,
                          list_type: List[str] = None,
                          list_ellip: List[float] = None,
                          list_Re: List[float] = None,
                          list_theta: List[float] = None,
                          
                          update_type_from_catalog: bool = True,
                          update_flux_from_catalog: bool = True,
                          update_ellip_from_catalog: bool = True,
                          update_Re_from_catalog: bool = True,
                          update_theta_from_catalog: bool = True,
                          
                          ra_key: str = 'X_WORLD',
                          dec_key: str = 'Y_WORLD',
                          type_key: str = 'CLASS_STAR',
                          flux_key: str = 'FLUX_AUTO',
                          ellip_key: str = 'ELLIPTICITY',
                          Re_key: str = 'FLUX_RADIUS',
                          theta_key: str = 'THETA_IMAGE',
                          matching_radius_arcsec: float = 5):

        """
        Match input targets to catalog entries.

        Parameters
        ----------
        list_ra : list of target RA (deg)
        list_dec : list of target DEC (deg)
        radius_arcsec : matching radius

        Returns
        -------
        dict:
            {
                filter_name: matched_flux_array
            }

        If no catalog provided:
            return None
        """
        def read_catalog(cat_path):
            """Lazy-load table data from path by trying multiple formats."""
            try_formats = [
                'fits',
                'ascii.sextractor',
                'ascii',
                'csv',
                'ascii.basic',
                'ascii.commented_header',
                'ascii.tab',
                'ascii.fast_no_header',
            ]
            for fmt in try_formats:
                try:
                    data = Table.read(cat_path, format=fmt)
                    return data  # Success
                except Exception:
                    continue  # Try next format
            return None  
        
        if len(list_ra) != len(list_dec):
            raise ValueError("list_ra and list_dec must have same length")
        if list_type is not None and len(list_type) != len(list_ra):
            raise ValueError("list_type and list_ra must have same length")
        
        n = len(list_ra)
        list_ra = np.atleast_1d(list_ra)
        list_dec = np.atleast_1d(list_dec)
        
        # Convert optional lists to numpy
        list_ellip = np.asarray(list_ellip) if list_ellip is not None else None
        list_Re = np.asarray(list_Re) if list_Re is not None else None
        list_theta = np.asarray(list_theta) if list_theta is not None else None
        list_flux = np.asarray(list_flux) if list_flux is not None else None
        list_type = np.asarray(list_type) if list_type is not None else None
        # Check lengths of lists
        if list_ellip is not None and len(list_ellip) != n:
            raise ValueError("list_ellip must have same length as list_ra and list_dec")
        if list_Re is not None and len(list_Re) != n:
            raise ValueError("list_Re must have same length as list_ra and list_dec")
        if list_theta is not None and len(list_theta) != n:
            raise ValueError("list_theta must have same length as list_ra and list_dec")
        if list_flux is not None and len(list_flux) != n:
            raise ValueError("list_flux must have same length as list_ra and list_dec")
        if list_type is not None and len(list_type) != n:
            raise ValueError("list_type must have same length as list_ra and list_dec")

        target_coord = SkyCoord(list_ra, list_dec, unit="deg")
        radius = matching_radius_arcsec * u.arcsec

        result = {}
        for filt, cat_path in tqdm(zip(self.filter_list, self.catalog_paths), desc = 'Constructing input catalogs...'):

            catalog_tbl = read_catalog(cat_path)
            if catalog_tbl is None:
                print(f"[CRITICAL] {filt}: Failed to read catalog")
                continue

            if ra_key not in catalog_tbl.colnames or dec_key not in catalog_tbl.colnames:
                print(f"[CRITICAL] {filt}: Missing RA/DEC")
                continue

            cat_coord = SkyCoord(catalog_tbl[ra_key],
                                catalog_tbl[dec_key],
                                unit="deg")

            idx, sep2d, _ = target_coord.match_to_catalog_sky(cat_coord)
            matched = sep2d < radius

            # Initialize arrays depending on update flags
            flux_array = np.full(n, np.nan)
            ellip_array = np.full(n, np.nan)
            Re_array = np.full(n, np.nan)
            theta_array = np.full(n, np.nan)
            type_array = np.full(n, 'EXP', dtype=object)

            # ---- If update is False → use provided lists ----
            if not update_ellip_from_catalog and list_ellip is not None:
                ellip_array = list_ellip.copy()

            if not update_Re_from_catalog and list_Re is not None:
                Re_array = list_Re.copy()

            if not update_theta_from_catalog and list_theta is not None:
                theta_array = list_theta.copy()
                
            if not update_flux_from_catalog and list_flux is not None:
                flux_array = list_flux.copy()

            if not update_type_from_catalog and list_type is not None:
                type_array = list_type.copy()

            # ---- Update from catalog if requested ----
            for i, (is_match, cat_index) in enumerate(zip(matched, idx)):

                if not is_match:
                    continue

                row = catalog_tbl[cat_index]

                if update_flux_from_catalog and flux_key in catalog_tbl.colnames:
                    flux_array[i] = row[flux_key]

                if update_ellip_from_catalog and ellip_key in catalog_tbl.colnames:
                    ellip_array[i] = row[ellip_key]

                if update_Re_from_catalog and Re_key in catalog_tbl.colnames:
                    Re_array[i] = row[Re_key]

                if update_theta_from_catalog and theta_key in catalog_tbl.colnames:
                    theta_array[i] = row[theta_key]

                if update_type_from_catalog and type_key in catalog_tbl.colnames:
                    type_value = row[type_key]
                    type_array[i] = 'STAR' if type_value > 0.5 else 'EXP'

            result[filt] = {
                "flux": flux_array,
                "ellip": ellip_array,
                "Re": Re_array,
                "theta": theta_array,
                "type": type_array
            }
            self.build_input_catalog(output_path = output_path,
                                      list_ra = list_ra,
                                      list_dec = list_dec,
                                      result_dict = result,
                                      list_id = None)

        return result

    def build_input_catalog(self,
                            output_path: Path,
                            list_ra: List[float],
                            list_dec: List[float],
                            result_dict: Dict,
                            list_id: Optional[List[str]] = None):

        n = len(list_ra)

        if list_id is None:
            list_id = [f"{i:05d}" for i in range(n)]

        df = pd.DataFrame({
            "ID": list_id,
            "RA": list_ra,
            "DEC": list_dec
        })

        # ==========================
        # TYPE → most probable (mode)
        # ==========================

        type_array = []

        for i in range(n):
            types_i = []

            for filt in self.filter_list:
                if filt in result_dict:
                    t = result_dict[filt]["type"][i]
                    if t is not None:
                        types_i.append(t)

            if len(types_i) == 0:
                type_array.append("EXP")
            else:
                # mode
                vals, counts = np.unique(types_i, return_counts=True)
                type_array.append(vals[np.argmax(counts)])

        df["TYPE"] = type_array

        # ==========================
        # Flux columns (unchanged)
        # ==========================

        for filt in self.filter_list:
            colname = f"FLUX_{filt}"
            if filt in result_dict:
                df[colname] = result_dict[filt]["flux"]
            else:
                df[colname] = np.nan

        # ==========================
        # Shape parameters
        # ==========================

        ell_array = []
        Re_array = []
        theta_array = []

        for i in range(n):

            ell_vals = []
            Re_vals = []
            theta_vals = []

            for filt in self.filter_list:
                if filt not in result_dict:
                    continue

                ell = result_dict[filt]["ellip"][i]
                Re_val = result_dict[filt]["Re"][i]
                theta = result_dict[filt]["theta"][i]

                if not np.isnan(ell):
                    ell_vals.append(ell)

                if not np.isnan(Re_val):
                    Re_vals.append(Re_val)

                if not np.isnan(theta):
                    theta_vals.append(theta)

            # ---- ELL median ----
            ell_array.append(np.nanmedian(ell_vals) if len(ell_vals) > 0 else np.nan)

            # ---- Re median ----
            Re_array.append(np.nanmedian(Re_vals) if len(Re_vals) > 0 else np.nan)

            # ---- THETA circular mean ----
            if len(theta_vals) > 0:
                theta_rad = np.deg2rad(theta_vals)
                mean_angle = np.arctan2(
                    np.nanmean(np.sin(theta_rad)),
                    np.nanmean(np.cos(theta_rad))
                )
                theta_array.append(np.rad2deg(mean_angle))
            else:
                theta_array.append(np.nan)

        df["ELL"] = ell_array
        df["Re"] = Re_array
        df["THETA"] = theta_array

        # ---- Save ----
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)

        return str(output_path.resolve())

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    def summary(self):
        print("Tract7DT Formatter")
        print("-------------------")
        print("Filters:", self.filter_list)
        print("N bands:", len(self.filter_list))
        print("Catalog loaded:", self.catalog_paths is not None)

# %%